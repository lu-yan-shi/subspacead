# Models Module

## Overview

The `models` module provides a **unified interface** for multiple vision foundation model backbones used in the anomaly detection pipelines. All models expose the same API regardless of the underlying architecture, making it straightforward to swap backbones or add new ones.

## File Structure

```
models/
├── model_base.py                    # Abstract base class (BaseModel) and VisionEncoderOutput
├── config_base.py                   # Model configuration dataclasses (ViTConfig, ConvNextConfig)
├── model_dinov2.py                  # DINOv2 backbone
├── model_dinov2_with_register.py    # DINOv2 with register tokens
├── model_dinov3_vit.py              # DINOv3 ViT (RoPE) backbone
├── model_dinov3_convnext.py         # DINOv3 ConvNeXt backbone
├── model_visreg.py                   # VISReg ViT backbone
├── model_open_clip.py               # OpenCLIP backbone
├── model_meta_clip2.py              # MetaCLIP2 backbone
├── model_lingbot_vision.py          # LingBot-Vision backbone
├── hf_model_dinov2_register.py      # HuggingFace backbone definition for DINOv2 + registers
├── hf_model_dinov3_vit.py           # HuggingFace backbone definition for DINOv3 ViT
├── hf_model_meta_clip_2.py          # HuggingFace backbone definition for MetaCLIP2
└── experiment.py                    # Experimental model utilities
```

---

## Supported Models

| Class | `--model` key | Architecture | Attention | Multi-layer |
|---|---|---|---|---|
| `DinoV2Model` | `dinov2` | DINOv2 ViT (HuggingFace) | Yes | Yes |
| `DinoV2WithRegisterModel` | `dinov2_with_register` | DINOv2 + Register Tokens | Yes | Yes |
| `DinoV3ViTModel` | `dinov3_vit` | DINOv3 ViT with RoPE | Yes | Yes |
| `DinoV3ConvNextModel` | `dinov3_convnext` | DINOv3 ConvNeXt | No | No |
| `VisRegModel` | `visreg` | VISReg ViT (ImageNet-1K) | Yes | Yes |
| `OpenCLIPModel` | `open_clip` | OpenCLIP ViT | No | No |
| `MetaCLIP2Model` | `meta_clip2` | MetaCLIP2 ViT | Yes | Yes |
| `LingBotVisionModel` | `lingbot_vision` | LingBot-Vision ViT | Yes | Yes |

---

## Core Classes

### `VisionEncoderOutput`

The standardized output dataclass returned by every model's `get_features()` and `__call__()`.

```python
@dataclass
class VisionEncoderOutput(ModelOutput):
    pooler_output: Optional[torch.FloatTensor]        # (B, D)       — CLS / global token
    feature_maps:  Optional[tuple[torch.FloatTensor]] # (B, C, H, W) — per-layer spatial grids
    attentions:    Optional[tuple[torch.FloatTensor]] # (B, Heads, H, W) — attention maps
    registers:     Optional[torch.FloatTensor]        # (B, N, C)    — register tokens (DINO only)
```

- `feature_maps` is a tuple with one entry per requested layer. Each entry is shaped `(B, C, H, W)` where `H` and `W` are the spatial patch grid dimensions.
- Requesting `output_feature_maps_indices=(-1,)` always returns the last layer.

### `BaseModel`

Abstract base class that all backbone implementations inherit from.

**Constructor arguments:**

| Argument | Type | Description |
|---|---|---|
| `model_path` | `str / PathLike` | Path to local model weights directory |
| `device` | `str / torch.device` | Target device (defaults to CUDA if available) |
| `dtype` | `str / torch.dtype` | Computation dtype (`float32`, `float16`, `bfloat16`) |
| `resolution` | `int / Tuple[int, int]` | Override model input resolution (optional) |

**Key methods:**

| Method | Description |
|---|---|
| `load_model(...)` | Load weights, processor, and configuration |
| `preprocess(images)` | Convert PIL images to normalized tensors |
| `get_features(pixel_values, output_feature_maps_indices, return_attentions)` | Extract `VisionEncoderOutput` |
| `forward(...)` | Raw forward pass through the underlying model |
| `__call__(images, ...)` | End-to-end: preprocess → get_features |
| `to(device, dtype)` | Move model to device/dtype |
| `eval()` / `train()` | Set evaluation/training mode |

### Model Configurations

Dataclasses in `config_base.py` standardize model metadata:

```python
@dataclass
class ViTConfig(BaseConfig):
    resolution: Tuple[int, int]
    hidden_size: int
    num_layers: int
    num_heads: int
    head_dim: int
    patch_size: int
    is_cls_token: bool
    num_register_tokens: int

@dataclass
class ConvNextConfig(BaseConfig):
    resolution: Tuple[int, int]
    hidden_size: int
    patch_size: int
    is_cls_token: bool
```

---

## Resolution Handling

- For flexible ViT models (DINOv2, DINOv3 ViT, VISReg, MetaCLIP2), `resolution` can be `None` to allow dynamic multi-resolution inference.
- For fixed-input models (OpenCLIP, ConvNeXt), `resolution` must match the model's required input size.
- Resolution can be overridden in `eval.py` using `--model_resolution` (model native input) or `--image_resolution` (pipeline-level override).

## VISReg Checkpoints

`VisRegModel` requires `timm>=1.0.0` and local VISReg ImageNet-1K weights. Pass either a checkpoint file named `visreg-vit-b-inet1k.pth` or `visreg-vit-l-inet1k.pth`, or a directory containing exactly one matching checkpoint. The filename selects the matching ViT-B/16 or ViT-L/14 architecture; classifier-head weights are intentionally discarded because AD uses the feature extractor only.

---

## Usage

### Loading a Model

```python
from ad_pipelines.models import DinoV3ViTModel

model = DinoV3ViTModel(
    model_path="/path/to/dinov3-vit-base",
    device="cuda",
    dtype=torch.float32,
)
```

### Extracting Features

```python
from PIL import Image

images = [Image.open("image1.png"), Image.open("image2.png")]

# Preprocess
pixel_values = model.preprocess(images)  # (B, C, H, W)

# Extract features from the last two layers, with attention maps
output = model.get_features(
    pixel_values,
    output_feature_maps_indices=(-2, -1),
    return_attentions=True,
)

print(output.pooler_output.shape)     # (B, D)
print(output.feature_maps[-1].shape)  # (B, C, H, W)
print(output.attentions[-1].shape)    # (B, num_heads, H, W)
```

### End-to-End Inference

```python
# __call__ runs preprocess + get_features in one step
output = model(images, output_feature_maps_indices=(-1,), return_attentions=True)
```

---

## Adding a Custom Model

1. Inherit from `BaseModel` in `model_base.py`.
2. Implement `load_model()`, `preprocess()`, `get_features()`, and `forward()`.
3. Populate `self.config` with the appropriate `ViTConfig` or `ConvNextConfig`.
4. Register the model key in `eval.py`'s `MODEL_MAP`.

```python
from ad_pipelines.models.model_base import BaseModel, VisionEncoderOutput

class MyCustomModel(BaseModel):
    def load_model(self, model_path, **kwargs):
        # Load your model here
        ...

    def preprocess(self, images, resolution=None):
        # Convert PIL images → normalized tensor (B, C, H, W)
        ...

    def get_features(self, pixel_values, output_feature_maps_indices=(-1,), return_attentions=False):
        # Return VisionEncoderOutput
        ...

    def forward(self, pixel_values, **kwargs):
        # Raw model forward pass
        ...
```
