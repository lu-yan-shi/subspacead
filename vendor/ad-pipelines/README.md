# DuoAD: Leveraging [CLS] Dual Characteristics for Training-Free Few-Shot Anomaly Detection

<p align="center">
  <a href="https://arxiv.org/abs/2607.23924"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2607.23924-b31b1b.svg"></a>
  <a href="https://www.alphaxiv.org/abs/2607.23924"><img alt="alphaXiv" src="https://img.shields.io/badge/alphaXiv-2607.23924-6A5ACD.svg"></a>
</p>

**AD-Pipelines** is the official code release for the paper above — a Python research framework for **training-free, few-shot Anomaly Detection (AD)** on industrial image data.

The framework evaluates state-of-the-art vision foundation models against standard industrial AD benchmarks. It is designed for high-resolution industrial inspection with an efficient, training-free approach that requires only a handful of normal reference images.

## Features

- **Training-free**: Leverages few-shot capabilities of pre-trained foundation models — no fine-tuning required.
- **Multi-model**: Unified interface supporting DINOv2, DINOv3, VISReg, OpenCLIP, MetaCLIP2, and LingBot-Vision.
- **Multi-dataset**: Built-in support for MVTec AD, VisA, and Real-IAD benchmarks.
- **Multi-view**: Native support for multi-view datasets (Real-IAD) with per-view memory banks and two evaluation modes.
- **Reproducible evaluation**: Multi-seed evaluation with per-class metrics and summary reports.

---

## Installation

**Prerequisites:** Python >= 3.8, PyTorch >= 2.0

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Project Structure

```
AD-Pipelines/
├── eval.py                        # Main evaluation entry point (CLI)
├── pyproject.toml                 # Package metadata
├── requirements.txt               # Dependencies
    └── src/ad_pipelines/
        ├── models/                    # Vision backbone wrappers (unified interface)
        ├── pipelines/                 # AD algorithm implementations
        ├── datas/                     # Dataset loaders (MVTec, VisA, Real-IAD)
        └── utils/                     # Shared GPU-accelerated utilities
```

Each sub-module has its own `README.md` with detailed documentation:
- [`src/ad_pipelines/models/README.md`](src/ad_pipelines/models/README.md)
- [`src/ad_pipelines/pipelines/README.md`](src/ad_pipelines/pipelines/README.md)
- [`src/ad_pipelines/datas/README.md`](src/ad_pipelines/datas/README.md)
- [`src/ad_pipelines/utils/README.md`](src/ad_pipelines/utils/README.md)

---

## Usage

The main entry point is `eval.py`.

### Basic Example

DINOv3
```bash
python eval.py \
  --model dinov3_vit \
  --model_path /path/to/dinov3-vitb16-pretrain-lvd1689m \
  --data_path /path/to/MVTec \
  --pipeline duoad \
  --image_resolution 512 \
  --layers 8 10 12 \
  --augmentation_mode auto \
  --shots 1 2 4 \
  --output_path ./outputs/mvtec_dinov3
```

DINOv2
```bash
python eval.py \
  --model dinov2_with_register \
  --model_path /path/to/dinov2-with-registers-base \
  --data_path /path/to/MVTec \
  --pipeline duoad \
  --image_resolution 448 \
  --layers 8 10 12 \
  --augmentation_mode auto \
  --shots 1 2 4 \
  --output_path ./outputs/mvtec_dinov2
```

### Real-IAD Multi-View Example

```bash
python eval.py \
  --model dinov2_with_register \
  --model_path /path/to/dinov2-with-registers-base \
  --data_path /path/to/realiad_root \
  --dataset realiad \
  --pipeline duoad \
  --shots 1 2 4 \
  --output_path ./outputs/realiad_dinov2 \
  --camera_views C1 C2 C3 C4 C5 \
  --evaluation_mode view_as_image \
  --view_aggregation max
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `dinov3_vit` | Model key (see [Models](#models)) |
| `--model_path` | *(required)* | Path to pre-downloaded model weights |
| `--data_path` | *(required)* | Root path to the dataset |
| `--dataset` | *(auto-detect)* | Dataset type: `mvtec`, `visa`, `realiad`. Inferred from `--data_path` if omitted |
| `--pipeline` | `patchead` | AD algorithm (see [Pipelines](#pipelines)) |
| `--shots` | `1 2 4` | Number of normal reference images (supports multiple) |
| `--seeds` | `2356` | Random seeds for multi-run evaluation (supports multiple) |
| `--output_path` | *(required)* | Directory to save results |
| `--device` | `cuda` | Compute device (`cuda`, `cpu`) |
| `--dtype` | `fp32` | Precision: `fp32`, `fp16`, `bf16` |
| `--model_resolution` | `None` | Override the model's native input resolution |
| `--image_resolution` | `None` | Override the pipeline input/image resolution |
| `--eval_resolution` | `None` | Resolution for pixel-metric upsampling (defaults to `--image_resolution`; for Real-IAD defaults to 224 if unset) |
| `--augmentation_mode` | `none` | Prompt augmentation mode: `none`, `auto`, `force`, `rotate`, `rotate90`, `flip` |
| `--augmentation_analysis_split` | `test` | Split used by `auto` augmentation analysis: `test` or `train` |
| `--layers` | `8 10 12` | ViT layer indices to extract features from |
| `--fusion` | `None` | Multi-layer fusion method (`score_avg`, `score_max`, `feature_avg`, `feature_concat`) |
| `--duoad_sim_method` | `None` | Similarity aggregation for DuoAD (`max`, `top1_mean`) |
| `--no_saliency` | `False` | Disable saliency weighting |
| `--save_results` | `False` | Save anomaly heatmap images to disk |
| `--batch_budget` | `16` | Batch size for inference |
| `--warmup_count` | `30` | Number of warmup samples for augmentation analysis |

#### Real-IAD-specific arguments

| Argument | Default | Description |
|---|---|---|
| `--eval_image_dir` | `realiad_1024` | Image subdirectory inside the dataset root |
| `--eval_json_dir` | `realiad_jsons` | JSON subdirectory inside the dataset root |
| `--camera_views` | *(all views)* | Camera views to use (e.g. `--camera_views C1 C2 C3`) |
| `--evaluation_mode` | `view_as_image` | `view_as_image` or `views_as_sample` (see [Pipelines README](src/ad_pipelines/pipelines/README.md)) |
| `--view_aggregation` | `max` | Score aggregation across views in `views_as_sample` mode: `max` or `mean` |

### Debug environment variables

| Variable | Effect |
|---|---|
| `AD_DEBUG_AUGMENTATION` | Print per-class augmentation-analysis decisions (rotation/flip tolerances and selected augmentations). |
| `AD_DEBUG_SCORES` | Print per-batch mean anomaly scores for normal and anomalous samples. |

### Dataset Auto-detection

The dataset type is automatically detected from `--data_path` when `--dataset` is not specified:

| Path contains | Dataset used |
|---|---|
| `mvtec` | MVTec AD |
| `visa` | VisA |
| `realiad` / `real_iad` / `real-iad` | Real-IAD |

Prefer passing `--dataset` explicitly to avoid ambiguity.

---

## Models

| `--model` key | Architecture | Attention | Multi-layer |
|---|---|---|---|
| `dinov2` | DINOv2 (HuggingFace) | Yes | Yes |
| `dinov2_with_register` | DINOv2 + Register Tokens | Yes | Yes |
| `dinov3_vit` | DINOv3 ViT (RoPE) | Yes | Yes |
| `dinov3_convnext` | DINOv3 ConvNeXt | No | No |
| `visreg` | VISReg ViT (ImageNet-1K) | Yes | Yes |
| `open_clip` | OpenCLIP | No | No |
| `meta_clip2` | MetaCLIP2 | Yes | Yes |
| `lingbot_vision` | LingBot-Vision ViT | Yes | Yes |

All models share a unified `BaseModel` interface. See [`models/README.md`](src/ad_pipelines/models/README.md) for details on the output format and how to add custom models.

`lingbot_vision` uses the optional upstream `[lingbot-vision](https://github.com/Robbyant/lingbot-vision)` package and loads `robbyant/lingbot-vision-vit-base` style checkpoints via its official loader. Pass `--model_path robbyant/lingbot-vision-vit-base` or a local checkpoint directory; `--model_resolution 512` is recommended and non-multiple resolutions are snapped down to the ViT/16 patch grid.

`visreg` loads a local VISReg ImageNet-1K checkpoint. Pass `--model_path` either to `visreg-vit-b-inet1k.pth` or `visreg-vit-l-inet1k.pth`, or to a directory containing exactly one such file. VISReg supports dynamic input resolution; use `--model_resolution` or `--image_resolution` to override its default 224 px input size.

**Resolution notes for patch-based models:**
- For ViT/16 backbones (`dinov2`, `dinov2_with_register`, `dinov3_vit`, `meta_clip2`), the effective patch grid is derived as `resolution // patch_size`. Resolutions that are not exact multiples of the patch size silently truncate the remainder (e.g. `--image_resolution 518` with patch 16 yields a 32×32 grid and drops the last 14 pixels). Use a multiple of the patch size or the model's native resolution for best results.
- `dinov3_convnext` uses a 32×32 patch grid; non-multiple resolutions are also truncated.
- `dinov3_convnext` also does not support multi-layer feature extraction; `--layers` must be `-1` (the default).

---

## Pipelines

| `--pipeline` key | Class | Description |
|---|---|---|
| `patchead` | `PatchEADPipeline` | Patch-based cosine similarity against a normal memory bank |
| `patchiad` | `PatchIADPipeline` | Extends PatchEAD with CLS-token saliency and multi-layer fusion |
| `duoad` | `DuoADPipeline` | PatchIAD variant with public `max` / `top1_mean` similarity aggregation options |

All pipelines follow a strict inheritance hierarchy. See [`pipelines/README.md`](src/ad_pipelines/pipelines/README.md) for algorithm details.

### Release Notes

N/A

---

## Evaluation Output

Each run produces the following files under `--output_path`:

| File | Content |
|---|---|
| `config_{N}_shots.json` | Full pipeline configuration and git commit hash |
| `evaluation_results_{N}_shots.json` | Per-class metrics (AUROC, AUPR, F1Max at image and pixel level) |
| `evaluation_results_{N}_shots.csv` | Same as above in CSV format |
| `statistics_{N}_shots.json` | Mean ± std across multiple seeds |
| `statistics_{N}_shots.csv` | Same in CSV format |
| `summary_{N}_shots.txt` | Human-readable formatted report |

### Metrics

- **Image-level**: AUROC, AUPR, AP, F1Max
- **Pixel-level**: AUROC, AUPRO, F1Max

---

## Extending the Framework

### Adding a New Model

Inherit from `BaseModel` in `src/ad_pipelines/models/model_base.py` and implement the required interface. See [`models/README.md`](src/ad_pipelines/models/README.md).

### Adding a New Pipeline

Inherit from `AnomalyDetectionPipelineBase` (or any existing pipeline) in `src/ad_pipelines/pipelines/pipeline_base.py`. See [`pipelines/README.md`](src/ad_pipelines/pipelines/README.md).

### Adding a New Dataset

Inherit from `BaseAnomalyDataset` and `BaseAnomalyClassDataset` in `src/ad_pipelines/datas/dataset.py`. See [`datas/README.md`](src/ad_pipelines/datas/README.md).

---

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Code formatting
black src/

# Type checking
mypy src/
```

---

## Citation

Please cite DuoAD as follows:

```bibtex
@misc{tang2026duoad,
  title         = {{DuoAD}: Leveraging {[CLS]} Dual Characteristics for Training-Free Few-Shot Anomaly Detection},
  author        = {Tang, Jyun-Ze and Huang, Po-Han and Chang, Ming-Ching and Hsu, Chih-Fan and Li, Jeng-Lin},
  year          = {2026},
  eprint        = {2607.23924},
  archiveprefix = {arXiv},
  primaryclass  = {cs.CV},
  url           = {https://arxiv.org/abs/2607.23924}
}
```

Please also cite our previous work, PatchEAD. Note that the PatchEAD implementation in this repository differs from the method described in the paper.

```bibtex
@inproceedings{huang2026patchead,
  title        = {{PatchEAD}: Unifying Industrial Visual Prompting Frameworks for Patch-Exclusive Anomaly Detection},
  author       = {Huang, Po-Han and Li, Jeng-Lin and Huang, Po-Hsuan and Chang, Ming-Ching and Chen, Wei-Chao},
  booktitle    = {2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages        = {5531--5540},
  year         = {2026},
  organization = {IEEE}
}
```

---

## Authors

Developed by **Inventec AI Center**.
