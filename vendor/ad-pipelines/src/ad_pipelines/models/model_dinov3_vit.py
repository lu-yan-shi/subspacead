from typing import Dict, Any, Optional, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DINOv3ViTModel,
    DINOv3ViTConfig,
    AutoImageProcessor
)
from PIL import Image
from os import PathLike

from .model_base import BaseModel, VisionEncoderOutput
from .model_dinov2_with_register import DinoV2WithRegisterModel
from .hf_model_dinov3_vit import DINOv3ViTModel

class DINOv3ViTBackbone(DINOv3ViTModel):
    def __init__(self, config: DINOv3ViTConfig):
        super().__init__(config)

    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        output_feature_maps_indices: tuple = (-1,),
    ) -> VisionEncoderOutput:
        all_self_attentions = () if output_attentions else None

        if not output_feature_maps_indices or len(output_feature_maps_indices) == 0:
            raise ValueError("At least one feature map index must be specified.")
        
        last_layer_idx = len(self.layer) - 1 # last layer index is in range [0, num_layers-1]
        # output_feature_maps_indices is in [1, num_layers], so we need to adjust for negative indexing
        normalized_indices = set(
            idx if idx >= 0 else (last_layer_idx+1) + 1 + idx 
            for idx in output_feature_maps_indices
        ) # in [0, num_layers]

        pixel_values = pixel_values.to(self.embeddings.patch_embeddings.weight.dtype)

        batch_size, channels, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = self.embeddings(pixel_values, bool_masked_pos=bool_masked_pos)
        feature_map = self.norm(hidden_states)
        position_embeddings = self.rope_embeddings(pixel_values)

        feature_maps = ()
        if 0 in normalized_indices:
            feature_maps += (feature_map[:, 1 + self.config.num_register_tokens:, :].transpose(1, 2).reshape(
                batch_size, -1, height, width
            ),)
        cls_token = None
        register_tokens = None
        for i, layer_module in enumerate(self.layer):                     
            layer_head_mask = head_mask[i] if head_mask is not None else None
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=layer_head_mask,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]

            # Currently, we only support extracting attentions from the last layer
            if output_attentions and i == last_layer_idx:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

            feature_map = self.norm(hidden_states)

            if i == last_layer_idx:
                cls_token = feature_map[:, 0]
                register_tokens = feature_map[:, 1 : 1 + self.config.num_register_tokens]
            
            feature_map = feature_map[:, 1 + self.config.num_register_tokens:, :].transpose(1, 2).reshape(
                batch_size, -1, height, width
            )

            if i+1 in normalized_indices:
                feature_maps += (feature_map,)

        return VisionEncoderOutput(
            pooler_output=cls_token,
            feature_maps=feature_maps,
            attentions=all_self_attentions,
            registers=register_tokens
        )


class DinoV3ViTModel(DinoV2WithRegisterModel):
    def __init__(
        self,
        model_path: Union[str, PathLike],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ):
        BaseModel.__init__(self, model_path, device, dtype, resolution, **kwargs)

        self.model_type = "dinov3_vit"
        self.config = None
        self.model = None
        self.processor = None
        self.text_model = None
        
        self.load_model(self.model_path, **kwargs)
    

    def load_model(self, model_path: Union[str, PathLike], **kwargs) -> None:
        """
        Load a DINOv3 model and processor from a pretrained model path.
        This method initializes the image processor and DINOv3 backbone model from the
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
        
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = DINOv3ViTBackbone.from_pretrained(
            model_path,
            **kwargs
        ).to(device=self.device, dtype=self.dtype)
        self.initialize_config(**kwargs)


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
        if output_feature_maps_indices is None:
            output_feature_maps_indices = (-1,)
        else:
            output_feature_maps_indices = tuple(output_feature_maps_indices)

        outputs = self.model(
            image_tensors,
            output_attentions=return_attentions,
            output_feature_maps_indices=output_feature_maps_indices,
        )

        return outputs

    def forward(
        self, 
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
    ) -> VisionEncoderOutput:
        """Forward pass through the model.
        
        Args:
            pixel_values: Input tensor of shape [B, C, H, W].
            bool_masked_pos: Optional mask for masked image modeling.
            head_mask: Optional mask for attention heads.
            output_attentions: Whether to return attention weights.
            output_feature_maps_indices: Indices of layers to return. Defaults to (-1,).
        """
        if output_feature_maps_indices is None:
            output_feature_maps_indices = (-1,)
            
        return self.model(
            pixel_values=pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_feature_maps_indices=tuple(output_feature_maps_indices),
        )

    def __call__(
        self, 
        images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        return_attentions: bool = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        **kwargs
    ) -> VisionEncoderOutput:
        """Process images and extract features.
        
        Args:
            images: Input images in various formats.
            resolution: Target resolution for preprocessing.
            do_normalize: Whether to normalize the images.
            do_resize: Whether to resize images.
            return_attentions: Whether to return attention weights.
            output_feature_maps_indices: Indices of layers to return feature maps from.
                Defaults to (-1,) for last layer only.
        """
        model_inputs = self.preprocess(
            images, 
            resolution=resolution, 
            do_normalize=do_normalize, 
            do_resize=do_resize, 
            **kwargs
        ).to(device=self.device, dtype=self.dtype)
        
        return self.get_features(
            model_inputs,
            return_attentions=return_attentions,
            output_feature_maps_indices=output_feature_maps_indices,
        )