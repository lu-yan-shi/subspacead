from typing import Dict, Any, Optional, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass

@dataclass
class BaseConfig:
    resolution: Tuple[int, int]
    hidden_size: int

@dataclass
class ViTConfig(BaseConfig):
    num_layers: int
    num_heads: int
    head_dim: int
    patch_size: int
    is_cls_token: bool
    num_register_tokens: int
    
@dataclass
class ConvNextConfig(BaseConfig):
    patch_size: int
    is_cls_token: bool
    num_register_tokens: int = 0