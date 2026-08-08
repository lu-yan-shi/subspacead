from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Sequence, Tuple, Union
from typing_extensions import Self

import torch
import torch.nn as nn
from transformers.utils import ModelOutput
from PIL import Image
from os import PathLike
from torch.nn import functional as F
import numpy as np

from .config_base import BaseConfig

@dataclass
class VisionEncoderOutput(ModelOutput):
    """
    Base class for outputs of vision backbones.

    Args:
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`, *optional*):
            Last layer hidden-state of the first token of the sequence (classification token) further processed by
            a pooling operation.
        feature_maps (`tuple(torch.FloatTensor)` of shape `(batch_size, num_channels, height, width)`):
            Feature maps of the stages.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Only applicable if the backbone uses attention. 
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        registers (`torch.FloatTensor` of shape `(batch_size, num_registers, register_dim)`, *optional*):
            Register features from the model, if applicable.
    """

    pooler_output: Optional[torch.FloatTensor] = None
    feature_maps: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    registers: Optional[torch.FloatTensor] = None

class BaseModel(ABC):
    def __init__(
        self,
        model_path: Union[str, PathLike],
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ):
        super().__init__()

        self.model_path = model_path
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype or torch.float32
        
        self.config = None # BaseConfig
        self.model_type = None
        self.patch_size = None
        self.feature_dim = None
        self.resolution = resolution
        if self.resolution and isinstance(self.resolution, int):
            self.resolution = (self.resolution, self.resolution)

        self.model = None
        self.preprocessor = None
        self.text_model = None
    
        
    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """Load model and processor"""
        pass


    @abstractmethod
    def initialize_config(self, **kwargs) -> None:
        """Initialize model configuration"""
        pass
    
        
    @abstractmethod
    def preprocess(
        self, 
        images: Union[Image.Image, torch.Tensor, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        do_resize: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        **kwargs
    ) -> torch.Tensor:
        """Preprocess images for the model"""
        pass
    
        
    @abstractmethod
    def get_features(
        self, 
        images: torch.Tensor,
        return_attentions: Optional[bool] = False,
        return_layer_features: Optional[bool] = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        **kwargs
    ) -> VisionEncoderOutput:
        """
        Get features from the model

        Args:
            images (torch.Tensor): Input images tensor with shape (batch_size, channels, height, width).
            return_attentions (Optional[bool], optional): Whether to return attention weights. Defaults to False.
            return_layer_features (Optional[bool], optional): Whether to return features from intermediate layers. 
                Defaults to False. When True, the returned VisionEncoderOutput will contain a tuple of feature maps in
                `feature_maps`, one for each layer. Otherwise, only the final layer's feature map is returned.
            output_feature_maps_indices (Optional[Sequence[int]], optional): Indices of layers to return feature maps from.
                If None, only returns the last layer's feature map (default behavior).
                Use tuple like (-1,) for last layer, or (2, 5, 8, 11) for specific layers.
            **kwargs: Additional keyword arguments for model-specific feature extraction.
        Returns:
            VisionEncoderOutput: Object containing extracted features and optionally attention weights 
                and/or intermediate layer features.
        Raises:
            NotImplementedError: This is an abstract method that must be implemented by subclasses.
        """
        pass


    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


    @abstractmethod
    def __call__(
        self, 
        images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        return_attentions: Optional[bool] = False,
        return_layer_features: Optional[bool] = False,
        *args, 
        **kwargs
    ):
        pass

    # Copied from HuggingFace Diffusers
    def to(self, *args, **kwargs) -> Self:
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)

        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype_arg = args[0]
            else:
                device_arg = torch.device(args[0]) if args[0] is not None else None
        elif len(args) == 2:
            if isinstance(args[0], torch.dtype):
                raise ValueError(
                    "When passing two arguments, make sure the first corresponds to `device` and the second to `dtype`."
                )
            device_arg = torch.device(args[0]) if args[0] is not None else None
            dtype_arg = args[1]
        elif len(args) > 2:
            raise ValueError("Please make sure to pass at most two arguments (`device` and `dtype`) `.to(...)`")

        if dtype is not None and dtype_arg is not None:
            raise ValueError(
                "You have passed `dtype` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        dtype = dtype or dtype_arg

        if device is not None and device_arg is not None:
            raise ValueError(
                "You have passed `device` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        device = device or device_arg
        self.model.to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

        return self
    
    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()