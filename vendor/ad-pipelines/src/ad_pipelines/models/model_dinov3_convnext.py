from typing import Dict, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BackboneOutput, BaseModelOutputWithPoolingAndNoAttention
from transformers import (
    DINOv3ConvNextModel,
    DINOv3ConvNextConfig,
    AutoImageProcessor
)
from PIL import Image
from os import PathLike
import numpy as np

from .config_base import ConvNextConfig
from .model_base import BaseModel, VisionEncoderOutput
from .model_dinov2_with_register import DinoV2WithRegisterModel

class DinoV3ConvNextModel(DinoV2WithRegisterModel):
    def __init__(
        self,
        model_path: Union[str, PathLike],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ):
        BaseModel.__init__(self, model_path, device, dtype, resolution, **kwargs)

        self.model_type = "dinov3_convnext"
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
        self.model = DINOv3ConvNextModel.from_pretrained(
            model_path,
            **kwargs
        ).to(device=self.device, dtype=self.dtype)
        self.initialize_config(**kwargs)


    def initialize_config(self, **kwargs) -> None:
        """
        Initialize model configuration
        """
        self.config = ConvNextConfig(
            resolution=(self.model.config.image_size, self.model.config.image_size),
            hidden_size=self.model.config.hidden_sizes[-1],
            patch_size=32,
            is_cls_token=True,
            **kwargs
        )


    def get_features(
        self, 
        image_tensors: torch.Tensor,
        return_attentions: Optional[bool] = False,
        return_layer_features: Optional[bool] = False,
        output_feature_maps_indices: Optional[Tuple[int, ...]] = None,
        **kwargs
    ) -> VisionEncoderOutput:
        """Get features from the model
        
        Args:
            image_tensors: Input tensor of shape [B, C, H, W].
            return_attentions: Whether to return attention weights.
            return_layer_features: Whether to return features from intermediate layers.
            output_feature_maps_indices: Indices of layers to return feature maps from.
                If provided and contains indices other than -1, will raise NotImplementedError.
        """
        if return_attentions:
            raise NotImplementedError("Attention extraction is not supported for DINOv3 ConvNext model.")
        if output_feature_maps_indices is not None and output_feature_maps_indices != (-1,):
            raise NotImplementedError("output_feature_maps_indices with multiple layers is not implemented for DINOv3 ConvNext model.")

        batch_size, channels, height, width = image_tensors.shape
        height = height // self.config.patch_size
        width = width // self.config.patch_size

        outputs = self.model(
            image_tensors,
            output_hidden_states=return_layer_features,
            **kwargs
        )

        if return_layer_features:
            feature_maps = outputs.hidden_states
        else:
            start_idx = int(self.config.is_cls_token) + self.config.num_register_tokens
            feature_maps = outputs.last_hidden_state[:, start_idx:, :] # Bs, patches, hidden
            feature_maps = (feature_maps.transpose(1, 2).reshape(batch_size, -1, height, width),)

        return VisionEncoderOutput(
            pooler_output=outputs.pooler_output,
            feature_maps=feature_maps,
            attentions=None,
        )


    def forward(
        self, 
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = None,
    ) -> BaseModelOutputWithPoolingAndNoAttention:
        return self.model(
            pixel_values,
            output_hidden_states=output_hidden_states,
        )

    def __call__(
        self, 
        images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        return_attentions: Optional[bool] = False,
        return_layer_features: Optional[bool] = False,
        **kwargs
    ):
        if return_attentions:
            raise NotImplementedError("Attention extraction is not supported for DINOv3 ConvNext model.")

        model_inputs = self.preprocess(
            images, 
            resolution=resolution, 
            do_normalize=do_normalize, 
            **kwargs
        )
        return self.get_features(
            model_inputs,
            return_layer_features=return_layer_features,
            **kwargs
        )