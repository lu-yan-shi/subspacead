from . import config_base as config
from .model_base import BaseModel, VisionEncoderOutput
from .model_dinov2 import DinoV2Model
from .model_dinov2_with_register import DinoV2WithRegisterModel
try:
    from .model_visreg import VisRegModel
except ImportError:
    VisRegModel = None  # timm optional dependency not installed
try:
    from .model_open_clip import OpenCLIPModel
except ImportError:
    OpenCLIPModel = None  # open_clip optional dependency not installed
from .model_meta_clip2 import MetaCLIP2Model
try:
    from .model_lingbot_vision import LingBotVisionModel
except ImportError:
    LingBotVisionModel = None  # lingbot_vision optional dependency not installed
try:
    from .model_eupe_vit import EUPEViTModel
except ImportError:
    EUPEViTModel = None  # EUPE repo not available
