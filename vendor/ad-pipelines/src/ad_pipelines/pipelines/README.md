# Pipelines Module

## Overview

The `pipelines` module contains implementations of the anomaly detection (AD) algorithms. Each pipeline orchestrates feature extraction from a backbone model, constructs a memory bank from normal reference images, and computes patch-level anomaly scores for test images.

All pipelines share a **strict inheritance hierarchy** and a consistent evaluation API.

## File Structure

```
pipelines/
├── pipeline_base.py                  # Abstract base class and evaluation loop
├── pipeline_patchead.py              # PatchEAD — core patch similarity algorithm
├── pipeline_patchiad.py              # PatchIAD — CLS saliency + multi-layer fusion
├── pipeline_attentionad.py           # DuoAD — public attention-logit aggregation
└── README.md                         # This documentation
```

---

## Inheritance Hierarchy

```
AnomalyDetectionPipelineBase (ABC)
  └── PatchEADPipeline
        └── PatchIADPipeline
              └── DuoADPipeline
```

---

## Pipeline Descriptions

### `PatchEADPipeline` — Patch-Exclusive Anomaly Detection

The foundational algorithm. All other pipelines build on top of this.

**IMPORTANT!** This release implementation is not a paper-exact reproduction of PatchEAD; it reflects the maintained release code path.

**Core algorithm:**

1. **Prompt phase**: Extract patch features from N normal reference images → flatten into a `[N×H×W, C]` memory bank.
2. **Test phase**: Extract patch features from the test image → `[B, H×W, C]`.
3. **Similarity**: Compute cosine similarity between each test patch and all memory bank patches → `[B, H×W, N×H×W]`.
4. **Anomaly map**: Take the maximum similarity across prompt patches per test patch → reshape to `[B, H, W]`. Low similarity = high anomaly score.
5. **Saliency**: Weight the anomaly map by a CLS-attention-based saliency mask to suppress background noise.
6. **Image score**: Aggregate pixel scores to a single image-level score (`top1` or `max`).

**Multi-view support:**

`PatchEADPipeline` implements `multi_view_few_shot` with true per-view memory banks and two evaluation modes (see [Multi-View Evaluation](#multi-view-evaluation) below).

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `attention_roi_threshold` | `0.05` | Threshold for CLS attention ROI mask |
| `attention_unroi_weight` | `0.5` | Weight applied to out-of-ROI regions |
| `cosine_sim_clamp` | `(-1, 1)` | Clamp range for cosine similarity |
| `resolution` | `None` | Override image resolution |
| `eval_resolution` | `None` | Resolution for pixel-metric upsampling (defaults to `resolution`); lower values reduce memory on high-res datasets (e.g. Real-IAD) |

---

### `PatchIADPipeline` — Patch Inclusive Anomaly Detection

Extends `PatchEADPipeline` with:

- **CLS-Patch saliency map**: Replaces raw attention saliency with cosine similarity between the CLS token and each patch token, giving a better foreground/object mask.
- **Multi-layer feature fusion**: Supports combining features from multiple ViT layers.
- **Morphological post-processing**: Applies closing to fill holes in the anomaly map.

**Multi-layer fusion methods** (`--fusion`):

| Method | Description |
|---|---|
| `score_avg` | Average anomaly scores independently computed per layer |
| `score_max` | Take the max anomaly score across layers |
| `feature_avg` | Average feature maps across layers before scoring |
| `feature_concat` | Concatenate feature maps across layers (channel dim) before scoring |

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `similarity_aggregation` | `max` | `max` or `top1_mean` |
| `closing_iterations` | `3` | Morphological closing iterations |
| `closing_kernel_size` | `3` | Morphological closing kernel size |
| `layer_fusion_method` | `score_avg` | Multi-layer fusion strategy |

---

### `DuoADPipeline` — DuoAD Release Variant

Extends `PatchIADPipeline` with the attention-based saliency and some experimental features.

**Similarity aggregation methods** (`--duoad_sim_method`):

| Method | Description |
|---|---|
| `max` | Hard maximum — nearest-neighbor matching |
| `top1_mean` | Mean of top-1% similarities |

**Key parameters:**

| Parameter | Default | Description |
|---|---|---|
| `similarity_aggregation` | `max` | Aggregation method (see table above) |


---

## Pipeline Output

All pipelines return a `PatchEADOutput` dataclass from `__call__`:

```python
@dataclass
class PatchEADOutput:
    anomaly_map:               torch.Tensor   # (B, H, W) — raw pixel-level scores
    anomaly_score:             torch.Tensor   # (B,)      — image-level score
    attentioned_anomaly_map:   torch.Tensor   # (B, H, W) — saliency-weighted anomaly map
    attentioned_anomaly_score: torch.Tensor   # (B,)      — saliency-weighted image score
    attention_map:             torch.Tensor   # (B, H, W) — CLS attention / saliency map
```

---

## Evaluation API

### `pipeline.evaluation_multi_run(...)`

The primary evaluation interface. Runs the pipeline over multiple random seeds and aggregates results.

```python
pipeline.evaluation_multi_run(
    output_path    = "./results",
    setting        = AnomalyDetectionSetting.FEW_SHOT,
    dataset        = test_dataset,
    prompt_dataset = prompt_dataset,
    shots          = 4,
    seeds          = [2356],
    batch_size     = 16,
    save_result_images = False,
    # additional kwargs forwarded to __call__:
    augmentation_mode       = "none",
    output_feature_maps_indices = (-1,),
)
```

**`AnomalyDetectionSetting` values:**

| Value | Description |
|---|---|
| `ZERO_SHOT` | No normal reference images (Not implemented) |
| `FEW_SHOT` | Standard few-shot with a shared memory bank |
| `MULTI_VIEW_FEW_SHOT` | Multi-view few-shot (Real-IAD); builds per-view memory banks |

### `pipeline.__call__(prompt_images, test_images, ...)`

Direct inference on a batch of images.

```python
output = pipeline(
    prompt_images = normal_images,   # List[PIL.Image] or Tensor
    test_images   = query_images,    # List[PIL.Image] or Tensor
)
# output.attentioned_anomaly_score → (B,) image-level anomaly scores
# output.attentioned_anomaly_map   → (B, H, W) pixel-level anomaly map
```

---

## Multi-View Evaluation

For Real-IAD (and other multi-view datasets), use `AnomalyDetectionSetting.MULTI_VIEW_FEW_SHOT`.  The evaluation loop dispatches to `pipeline.multi_view_few_shot(...)`, which `PatchEADPipeline` implements with true per-view memory banks.

### Evaluation modes

| `evaluation_mode` | Description |
|---|---|
| `"view_as_image"` | Each camera view is scored independently; metrics accumulate over individual views |
| `"views_as_sample"` | All views of the same physical object are scored together; anomaly maps/scores are aggregated before metrics update |

### View aggregation (for `"views_as_sample"`)

| `view_aggregation` | Description |
|---|---|
| `"max"` | Take the maximum score/map across views (default) |
| `"mean"` | Average scores/maps across views |

### Memory bank construction

`_build_per_view_memory_bank` samples `shots` normal images **per camera view**, so each view gets its own independent reference set.  Per-view augmentation configs can be supplied via `view_aug_configs`.

### Example

```python
from ad_pipelines.pipelines.pipeline_base import AnomalyDetectionSetting
from ad_pipelines.datas import RealIADDataset, SplitType

pipeline.evaluation_multi_run(
    output_path    = "./results",
    setting        = AnomalyDetectionSetting.MULTI_VIEW_FEW_SHOT,
    dataset        = test_dataset,
    prompt_dataset = prompt_dataset,
    shots          = 4,
    seeds          = [2356],
    batch_size     = 16,
    # RealIAD-specific kwargs:
    evaluation_mode  = "views_as_sample",
    view_aggregation = "max",
    eval_resolution  = 224,   # cap pixel-metric memory for 1024×1024 images
)
```

---

## Usage Example

```python
from ad_pipelines.models import DinoV3ViTModel
from ad_pipelines.pipelines import PatchIADPipeline
from ad_pipelines.pipelines.pipeline_base import AnomalyDetectionSetting
from ad_pipelines.datas import MVTecDataset, SplitType

# 1. Load model
model = DinoV3ViTModel("/path/to/dinov3-vit-base", device="cuda")

# 2. Initialize pipeline
pipeline = PatchIADPipeline(model, device="cuda", layer_fusion_method="score_avg")

# 3. Load datasets
test_dataset   = MVTecDataset("/path/to/mvtec", split=SplitType.TEST)
prompt_dataset = MVTecDataset("/path/to/mvtec", split=SplitType.TRAIN)

# 4. Run multi-seed evaluation for 1, 2, 4 shots
for shots in [1, 2, 4]:
    pipeline.evaluation_multi_run(
        output_path    = "./results",
        setting        = AnomalyDetectionSetting.FEW_SHOT,
        dataset        = test_dataset,
        prompt_dataset = prompt_dataset,
        shots          = shots,
        seeds          = [2356],
        batch_size     = 16,
        augmentation_mode = "none",
        output_feature_maps_indices = (-1,),
    )
```

---

## Adding a Custom Pipeline

Inherit from any existing pipeline and override `__call__` (and optionally `get_prompt_features`):

```python
from ad_pipelines.pipelines import PatchIADPipeline

class MyPipeline(PatchIADPipeline):
    def __call__(self, prompt_images, test_images, **kwargs):
        # Extract prompt features (from parent)
        prompt_features = self.get_prompt_features(prompt_images, **kwargs)

        # Custom anomaly scoring logic here
        ...

        return PatchEADOutput(
            anomaly_map=anomaly_map,
            anomaly_score=anomaly_score,
            attentioned_anomaly_map=attentioned_map,
            attentioned_anomaly_score=attentioned_score,
        )
```
