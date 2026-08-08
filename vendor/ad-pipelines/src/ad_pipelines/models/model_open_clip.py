from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import open_clip
import torch
import torch.nn as nn
import warnings

from pathlib import Path
from PIL import Image
from os import PathLike

from .model_base import BaseModel, VisionEncoderOutput
from .config_base import ViTConfig
    
PATH_VARIENT_MAP: Dict[str, str] = {
    "vit_base_patch16_plus_clip_240": "ViT-B-16-plus-240",
}

class OpenCLIPModel(BaseModel):
    def __init__(
        self,
        model_path: Union[str, PathLike],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ):
        super().__init__(model_path, device, dtype, resolution, **kwargs)

        self.model_type = "openclip"
        self.model_variant = None
        self.config = None
        self.model = None
        self.processor = None
        self.text_model = None
        
        self.load_model(self.model_path, **kwargs)

    def load_model(self, model_path: Union[str, PathLike], **kwargs) -> None:
        """
        Load a OpenCLIP model and processor from a pretrained model path.
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
        model_path = Path(model_path)
        self.model_variant = PATH_VARIENT_MAP.get(model_path.name.split(".")[0], None)
        if self.model_variant is None:
            raise ValueError(f"Model variant for path {model_path} not recognized.")
        
        model_bin_path = model_path / "open_clip_model.safetensors"
        if not model_bin_path.exists():
            model_bin_path = model_path / "open_clip_pytorch_model.bin"
        
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_variant, pretrained=model_bin_path.as_posix()
        )
        self.processor = preprocess
        self.model = model.to(device=self.device, dtype=self.dtype)

        self.model.visual.output_tokens = True # Enable output of token features
        self.model.visual.proj = None  # Remove projection layer if exists

        self.initialize_config(**kwargs)

        if self.resolution is None:
            self.resolution = self.config.resolution


    def initialize_config(self, **kwargs) -> None:
        """
        Initialize model configuration
        """
        self.config = ViTConfig(
            resolution=self.model.visual.image_size,
            hidden_size=self.model.visual.ln_pre.normalized_shape[0],
            head_dim=self.model.visual.transformer.resblocks[0].attn.head_dim,
            patch_size=self.model.visual.patch_size[0],
            num_layers=self.model.visual.transformer.layers,
            num_heads=self.model.visual.transformer.resblocks[0].attn.num_heads,
            is_cls_token=True,
            num_register_tokens=0,
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

        if resolution is None:
            resolution = self.resolution

        # if do_resize is set, override the resize setting in processor
        if do_resize is not None:
            warnings.warn("do_resize override not implemented for OpenCLIP processor.")

        if do_normalize is not None:
            warnings.warn("do_normalize override not implemented for OpenCLIP processor.")
        
        if not do_rescale:
            warnings.warn("do_rescale override not implemented for OpenCLIP processor.")

        # Convert single images to list format
        if isinstance(images, (np.ndarray, Image.Image)):
            images = [images]
        elif isinstance(images, torch.Tensor) and images.ndim == 3:
            images = [images]
        
        # Process images through the processor
        if isinstance(images, list):
            processed_images = torch.stack([self.processor(img) for img in images])
        else:
            # Already a batched tensor
            processed_images = images
        
        return processed_images

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
        if return_layer_features:
            raise NotImplementedError("return_layer_features is not implemented for OpenCLIPModel.")
        if return_attentions:
            raise NotImplementedError("return_attentions is not implemented for OpenCLIPModel.")
        if output_feature_maps_indices is not None and output_feature_maps_indices != (-1,):
            raise NotImplementedError("output_feature_maps_indices with multiple layers is not implemented for OpenCLIPModel.")
        
        _, channel, height, width = image_tensors.shape
        
        cls_token, token_features = self.model.encode_image(
            image_tensors,
            normalize=False,
            **kwargs
        )

        bs, seq_len, hidden_dim = token_features.shape
        token_features = token_features.reshape(
            bs, height // self.config.patch_size, width // self.config.patch_size, hidden_dim
        ).permute(0, 3, 1, 2)  # Bs, hidden_dim, H', W'

        if return_layer_features and len(self.model.out_features) <= 0:
            warnings.warn("Model was not initialized to return intermediate layer features. "
                          "Please re-initialize the model with out_features set.")

        return VisionEncoderOutput(
            pooler_output=cls_token,
            feature_maps=(token_features, ),
            attentions=None
        )


    def forward(
        self, 
        image: Optional[torch.Tensor] = None,
        text: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(
            image,
            text=text,
        )

    def __call__(
        self, 
        images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        **kwargs
    ):
        model_inputs = self.preprocess(
            images, 
            **kwargs
        ).to(device=self.device, dtype=self.dtype)

        return self.get_features(
            model_inputs,
            **kwargs
        )