from typing import Dict, Any, Optional, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn as nn

from transformers import (
    AutoImageProcessor
)
from transformers.modeling_outputs import BackboneOutput, BaseModelOutput
from transformers.utils import auto_docstring
from transformers.utils.generic import check_model_inputs
from PIL import Image
from os import PathLike

from .model_base import BaseModel, VisionEncoderOutput
from .model_dinov2 import DinoV2Model
from .config_base import ViTConfig
from .hf_model_dinov2_register import Dinov2WithRegistersBackbone # Custom backbone with attention weight support

class CustomDinov2WithRegistersBackbone(Dinov2WithRegistersBackbone):

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_feature_maps_indices: Optional[Tuple[int, ...]] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> VisionEncoderOutput:
        """Forward pass through DINOv2 encoder with register tokens.
        
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
        output: BaseModelOutput = self.encoder(
            embedding_output, 
            output_hidden_states=True,
            output_attentions=output_attentions,
        )
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
                    hidden_state = hidden_state[:, 1 + self.num_register_tokens :]
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

        attentions = output.attentions if output_attentions else None

        return VisionEncoderOutput(
            pooler_output=cls_token,
            feature_maps=tuple(feature_maps),
            attentions=attentions,
            registers=register_tokens
        )


class DinoV2WithRegisterModel(DinoV2Model):
    def __init__(
        self,
        model_path: Union[str, PathLike],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ):
        BaseModel.__init__(self, model_path, device, dtype, resolution, **kwargs)

        self.model_type = "dinov2_with_register"
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
        self.model = CustomDinov2WithRegistersBackbone.from_pretrained(
            model_path,
            **kwargs
        ).to(device=self.device, dtype=self.dtype)
        self.initialize_config(**kwargs)
