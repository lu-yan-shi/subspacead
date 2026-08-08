from typing import Dict, Any, Optional, Tuple, Union, Sequence

import numpy as np
import torch
import torch.nn as nn

from transformers import (
    AutoImageProcessor,
    MetaClip2VisionModel
)
from transformers.modeling_outputs import BackboneOutput, BaseModelOutput, BaseModelOutputWithPooling
from transformers.utils import auto_docstring
from transformers.utils.generic import check_model_inputs
from transformers.models.metaclip_2.configuration_metaclip_2 import MetaClip2VisionConfig
from PIL import Image
from os import PathLike

from .model_base import BaseModel, VisionEncoderOutput
from .model_dinov2 import DinoV2Model
from .config_base import ViTConfig
from .hf_model_meta_clip_2 import MetaClip2VisionModel as HFMetaClip2VisionModel


class MetaCLIP2VisionBackbone(HFMetaClip2VisionModel):
    """Custom MetaCLIP2 Vision Backbone with multi-layer feature extraction and attention weight support."""
    
    def __init__(self, config: MetaClip2VisionConfig):
        super().__init__(config)
        
    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        interpolate_pos_encoding: bool | None = False,
        output_attentions: bool = False,
        output_feature_maps_indices: tuple = (-1,),
        use_attention_before_softmax: bool = False,
        **kwargs
    ) -> VisionEncoderOutput:
        """
        Forward pass with multi-layer feature extraction and attention weight support.
        
        Args:
            pixel_values: Input images tensor [B, C, H, W]
            interpolate_pos_encoding: Whether to interpolate position embeddings
            output_attentions: Whether to output attention weights
            output_feature_maps_indices: Which layers to extract feature maps from.
                Indices in [1, num_layers], negative indexing supported. E.g., (-1,) for last layer.
            use_attention_before_softmax: If True, returns attention weights before softmax.
                Note: Currently always returns pre-softmax weights from eager_attention_forward.
        
        Returns:
            VisionEncoderOutput with pooler_output, feature_maps, and optionally attentions
        """
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")
        
        all_self_attentions = () if output_attentions else None
        
        if not output_feature_maps_indices or len(output_feature_maps_indices) == 0:
            raise ValueError("At least one feature map index must be specified.")
        
        batch_size, channels, height, width = pixel_values.shape
        patch_size = self.config.patch_size
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        
        # Normalize indices to [0, num_layers]
        num_layers = len(self.vision_model.encoder.layers)
        normalized_indices = set(
            idx if idx >= 0 else (num_layers + 1) + idx 
            for idx in output_feature_maps_indices
        )
        
        # Embeddings and pre-layernorm
        hidden_states = self.vision_model.embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)
        
        # Extract initial feature map if requested (before any transformer layer)
        feature_maps = ()
        if 0 in normalized_indices:
            # Skip CLS token [B, 1+N, D] -> [B, N, D] -> [B, D, H, W]
            feature_map = hidden_states[:, 1:, :].transpose(1, 2).reshape(
                batch_size, -1, num_patches_h, num_patches_w
            )
            feature_maps += (feature_map,)
        
        cls_token = None
        last_layer_idx = num_layers - 1
        
        # Process through encoder layers
        for i, encoder_layer in enumerate(self.vision_model.encoder.layers):
            hidden_states, attention_weights = encoder_layer(
                hidden_states,
                attention_mask=None,
                output_attentions=output_attentions and i == last_layer_idx,
                **kwargs
            )
            
            # Collect attention weights only from last layer
            if output_attentions and i == last_layer_idx:
                all_self_attentions = all_self_attentions + (attention_weights,)
            
            # Extract feature map from this layer if requested
            if (i + 1) in normalized_indices:
                feature_map = hidden_states[:, 1:, :].transpose(1, 2).reshape(
                    batch_size, -1, num_patches_h, num_patches_w
                )
                feature_maps += (feature_map,)
        
        # Extract CLS token and apply post-layernorm
        cls_token = hidden_states[:, 0, :]
        cls_token = self.vision_model.post_layernorm(cls_token)
        
        return VisionEncoderOutput(
            pooler_output=cls_token,
            feature_maps=feature_maps,
            attentions=all_self_attentions,
            registers=None  # MetaCLIP2 doesn't have register tokens
        )

class MetaCLIP2Model(DinoV2Model):
    def __init__(
        self,
        model_path: Union[str, PathLike],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ):
        # Don't call super().__init__ to avoid loading model twice
        BaseModel.__init__(self, model_path, device, dtype, resolution, **kwargs)
        
        self.model_type = "meta_clip2"
        self.config = None
        self.model = None
        self.processor = None
        self.text_model = None
        
        self.load_model(self.model_path, **kwargs)

    def load_model(self, model_path: Union[str, PathLike], **kwargs) -> None:
        """
        Load a MetaCLIP2 model and processor from a pretrained model path.
        """
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        
        # Load as custom backbone
        self.model = MetaCLIP2VisionBackbone.from_pretrained(
            model_path,
            **kwargs
        ).to(device=self.device, dtype=self.dtype)
        self.initialize_config(**kwargs)

    def initialize_config(self, **kwargs) -> None:
        """Initialize model configuration."""
        self.config = ViTConfig(
            resolution=(self.model.config.image_size, self.model.config.image_size),
            hidden_size=self.model.config.hidden_size,
            head_dim=self.model.config.hidden_size // self.model.config.num_attention_heads,
            patch_size=self.model.config.patch_size,
            num_layers=self.model.config.num_hidden_layers,
            num_heads=self.model.config.num_attention_heads,
            is_cls_token=True,
            num_register_tokens=0,  # MetaCLIP2 doesn't have register tokens
            **kwargs
        )
    
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
            pixel_values=image_tensors,
            output_attentions=return_attentions,
            output_feature_maps_indices=output_feature_maps_indices,
        )

        return outputs