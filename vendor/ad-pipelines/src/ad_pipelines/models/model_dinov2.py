from typing import Dict, Any, Optional, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import warnings

from transformers import (
    Dinov2Backbone,
    AutoImageProcessor
)
from transformers.modeling_outputs import BackboneOutput, BaseModelOutput
from PIL import Image
from os import PathLike

from .model_base import BaseModel, VisionEncoderOutput
from .config_base import ViTConfig

class CustomDinoV2Backbone(Dinov2Backbone):

    def forward(
        self, 
        pixel_values: torch.Tensor, 
        output_feature_maps_indices: Optional[Tuple[int, ...]] = None,
        **kwargs
    ) -> VisionEncoderOutput:
        """Forward pass through DINOv2 encoder.
        
        Args:
            pixel_values: Input tensor of shape (batch_size, channels, height, width).
            output_feature_maps_indices: Layer indices to extract feature maps from.
                Supports negative indexing.
            **kwargs: Additional arguments including output_attentions (bool).
        
        Returns:
            VisionEncoderOutput containing:
            - pooler_output: CLS token from last layer (batch_size, hidden_size).
            - feature_maps: Tuple of feature maps from specified layers.
            - attentions: Attention weights if requested, otherwise None.
        """
        embedding_output = self.embeddings(pixel_values)
        output: BaseModelOutput = self.encoder(embedding_output, output_hidden_states=True)
        hidden_states = output.hidden_states

        last_layer_idx = len(hidden_states) - 1
        normalized_indices = set(
            idx if idx >= 0 else last_layer_idx + 1 + idx 
            for idx in output_feature_maps_indices
        )

        feature_maps = []
        for stage, (stage_name, hidden_state) in enumerate(zip(self.stage_names, hidden_states)):
            if stage in normalized_indices:
                if self.config.apply_layernorm:
                    hidden_state = self.layernorm(hidden_state)
                if self.config.reshape_hidden_states:
                    hidden_state = hidden_state[:, 1:]
                    # this was actually a bug in the original implementation that we copied here,
                    # cause normally the order is height, width
                    batch_size, _, height, width = pixel_values.shape
                    patch_size = self.config.patch_size
                    hidden_state = hidden_state.reshape(batch_size, height // patch_size, width // patch_size, -1)
                    hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()
                feature_maps.append(hidden_state)

        cls_token = hidden_states[-1][:, 0]
        register_tokens = hidden_states[-1][:, 1 : 1 + self.num_register_tokens]
        if self.config.apply_layernorm:
            cls_token = self.layernorm(cls_token)

        attentions = output.attentions if kwargs.get("output_attentions", False) else None

        return VisionEncoderOutput(
            pooler_output=cls_token,
            feature_maps=tuple(feature_maps),
            attentions=attentions,
            registers=register_tokens
        )

class DinoV2Model(BaseModel):
    def __init__(
        self,
        model_path: Union[str, PathLike],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ):
        super().__init__(model_path, device, dtype, resolution, **kwargs)

        self.model_type = "dinov2"
        self.config = None
        self.model = None
        self.processor = None
        self.text_model = None
        
        self.load_model(self.model_path, **kwargs)

    def load_model(self, model_path: Union[str, PathLike], **kwargs) -> None:
        """
        Load a DINOv2 model and processor from a pretrained model path.
        This method initializes the image processor and DINOv2 backbone model from the
        specified pretrained model path, moves the model to the appropriate device and
        dtype, and extracts relevant configuration parameters.
        Args:
            model_path (Union[str, PathLike]): Path to the pretrained model directory
                or model identifier from Hugging Face Hub.
            **kwargs: Additional keyword arguments to pass to the model's from_pretrained
                method (e.g., torch_dtype, device_map, etc.).
        Returns:
            None
        Side Effects:
            - Sets self.processor: AutoImageProcessor instance for preprocessing images
            - Sets self.model: Dinov2Backbone model moved to specified device and dtype
            - Sets self.patch_size: Patch size from model configuration
            - Sets self.feature_dim: Hidden size (feature dimension) from model configuration
        """
        
        self.processor = AutoImageProcessor.from_pretrained(model_path, use_fast=True)
        self.model = CustomDinoV2Backbone.from_pretrained(
            model_path,
            **kwargs
        ).to(device=self.device, dtype=self.dtype)
        self.initialize_config(**kwargs)


    def initialize_config(self, **kwargs) -> None:
        """
        Initialize model configuration
        """
        self.config = ViTConfig(
            resolution=(self.model.config.image_size, self.model.config.image_size),
            hidden_size=self.model.config.hidden_size,
            head_dim=self.model.config.hidden_size // self.model.config.num_attention_heads,
            patch_size=self.model.config.patch_size,
            num_layers=self.model.config.num_hidden_layers,
            num_heads=self.model.config.num_attention_heads,
            is_cls_token=True,
            num_register_tokens=self.model.config.num_register_tokens,
            **kwargs
        )


    def preprocess(
        self, 
        images: Union[Image.Image, torch.Tensor, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
    ) -> torch.Tensor:
        extra_kwargs = {}

        # The pipeline always controls the target resolution explicitly, so the
        # processor's default center crop (e.g. DINOv2 BitImageProcessor crops to
        # 224x224) must never be applied. Leaving it enabled silently crops the
        # central region of reference/test images — catastrophic for object-centric
        # classes — whenever this is called without a truthy resolution (e.g. the
        # augmentation branch of get_prompt_features). Disable it unconditionally.
        extra_kwargs["do_center_crop"] = False

        if resolution is None:
            resolution = self.resolution
            
        if resolution:
            if isinstance(resolution, int):
                extra_kwargs["size"] = {"height": resolution, "width": resolution}
            else:
                extra_kwargs["size"] = {"width": resolution[0], "height": resolution[1]}
            extra_kwargs["do_resize"] = True
        else:
            extra_kwargs["do_resize"] = False

        # if do_resize is set, override the resize setting in processor
        if do_resize is not None:
            extra_kwargs["do_resize"] = do_resize

        if do_normalize is not None:
            extra_kwargs["do_normalize"] = do_normalize

        if do_rescale is not None:
            extra_kwargs["do_rescale"] = do_rescale

        model_input = self.processor(
            images, return_tensors="pt", **extra_kwargs
        )
        return model_input["pixel_values"]

    def get_features(
        self, 
        image_tensors: torch.Tensor,
        return_attentions: bool = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
    ) -> VisionEncoderOutput:
        """Get features from the model.
        
        Args:
            image_tensors: Input tensor of shape [B, C, H, W].
            return_attentions: Whether to return attention weights.
            output_feature_maps_indices: Indices of layers to return feature maps from.
                If None, only returns the last layer's feature map (default behavior).
                Use tuple like (-1,) for last layer, or (2, 5, 8, 11) for specific layers.
                
        Returns:
            VisionEncoderOutput containing pooler_output, feature_maps, and optionally attentions.
        """
        # Decide which layers' feature maps to output
        if output_feature_maps_indices is None:
            # Default to last layer only
            output_feature_maps_indices = (-1,)
        else:
            output_feature_maps_indices = tuple(output_feature_maps_indices)
            
        # Check if intermediate layers are needed but model is not configured for them
        num_layers = len(self.model.stage_names)
        needs_intermediate = any(
            0 <= idx < num_layers - 1 or (-num_layers <= idx < -1)
            for idx in output_feature_maps_indices
        )
        
        if needs_intermediate and len(self.model.out_features) <= 0:
            warnings.warn(
                "Requesting intermediate layer features, but model was not initialized "
                "with out_features. Consider re-initializing the model."
            )

        outputs = self.model(
            image_tensors,
            output_feature_maps_indices=output_feature_maps_indices,
            output_attentions=return_attentions,
            return_dict=True,
        )

        return VisionEncoderOutput(
            pooler_output=outputs.pooler_output,
            feature_maps=outputs.feature_maps,
            attentions=outputs.attentions if return_attentions else None      
        )


    def forward(
        self, 
        pixel_values: torch.Tensor,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        output_attentions: bool = False,
    ) -> VisionEncoderOutput:
        """Forward pass through the model.
        
        Args:
            pixel_values: Input tensor of shape [B, C, H, W].
            output_feature_maps_indices: Indices of layers to return. Defaults to (-1,).
            output_attentions: Whether to return attention weights.
        """
        if output_feature_maps_indices is None:
            output_feature_maps_indices = (-1,)
            
        return self.model(
            pixel_values,
            output_feature_maps_indices=tuple(output_feature_maps_indices),
            output_attentions=output_attentions,
        )

    def __call__(
        self, 
        images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        return_attentions: bool = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        **kwargs
    ) -> VisionEncoderOutput:
        """Process images and extract features.
        
        Args:
            images: Input images in various formats.
            resolution: Target resolution for preprocessing.
            do_normalize: Whether to normalize the images.
            return_attentions: Whether to return attention weights.
            output_feature_maps_indices: Indices of layers to return feature maps from.
                Defaults to (-1,) for last layer only.
        """
        model_inputs = self.preprocess(
            images, 
            resolution=resolution, 
            do_normalize=do_normalize, 
            **kwargs
        ).to(device=self.device, dtype=self.dtype)
        
        return self.get_features(
            model_inputs,
            return_attentions=return_attentions,
            output_feature_maps_indices=output_feature_maps_indices,
        )
