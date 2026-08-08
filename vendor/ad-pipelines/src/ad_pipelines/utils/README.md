# Utils Module

## Overview

The `utils` module provides GPU-accelerated shared utilities used across the anomaly detection pipelines. All operations are implemented in pure PyTorch, with no external C++ extensions required.

## File Structure

```
utils/
├── morphology.py          # Morphological operations (erosion, dilation, closing, opening)
├── otsu_binarization.py   # Batch GPU-accelerated Otsu thresholding
├── clustering.py          # Foreground-aware clustering (KMeans, HDBSCAN, Spectral)
├── image.py               # PIL / tensor image conversion utilities
├── __init__.py            # Module exports
└── README.md              # This documentation
```

---

## Utilities Reference

### `morphology.py` — Morphological Operations

GPU-accelerated morphological operations implemented via `torch.nn.functional.unfold`. All functions operate on `(B, H, W)` tensors.

| Function | Description |
|---|---|
| `erode(tensor, kernel_size, iterations)` | Binary erosion — shrinks foreground regions |
| `dilate(tensor, kernel_size, iterations)` | Binary dilation — expands foreground regions |
| `opening(tensor, kernel_size, iterations)` | Erosion followed by dilation (removes small noise) |
| `closing(tensor, kernel_size, iterations)` | Dilation followed by erosion (fills small holes) |
| `safe_closing(tensor, kernel_size, iterations)` | Closing with replicate border padding to avoid edge artifacts |
| `safe_closing_with_dilate(...)` | Safe closing followed by an additional dilation pass |
| `morphological_gradient(tensor, kernel_size)` | Dilation minus erosion (edge detection) |

**Usage:**

```python
from ad_pipelines.utils import closing, safe_closing

# anomaly_map: (B, H, W) float tensor
smoothed = safe_closing(anomaly_map, kernel_size=3, iterations=3)
```

**Note:** These are used internally by all pipelines to post-process anomaly maps, filling holes and smoothing boundaries before final scoring.

---

### `otsu_binarization.py` — Batch Otsu Thresholding

Fully GPU-accelerated batch implementation of Otsu's method for automatic threshold selection.

| Function | Description |
|---|---|
| `batch_otsu_binarization(tensor, num_bins, return_threshold)` | Compute optimal threshold independently per batch item, return binary mask |
| `adaptive_otsu_binarization(tensor, num_blocks, num_bins)` | Block-wise local Otsu thresholding (last dimension divided into blocks) |

**Algorithm:** Maximizes between-class variance via parallel histogram computation using a one-hot encoding trick. Runs entirely on GPU, suitable for real-time use.

**Usage:**

```python
from ad_pipelines.utils import batch_otsu_binarization

# anomaly_map: (B, H, W), values in [-1, 1]
binary_mask = batch_otsu_binarization(anomaly_map, num_bins=256)
# → (B, H, W) with values in {0, 1}

binary_mask, thresholds = batch_otsu_binarization(anomaly_map, return_threshold=True)
# thresholds: (B,) — per-image optimal threshold
```

---

### `clustering.py` — Foreground-Aware Clustering

Clustering utilities for grouping prompt patch features. All algorithms share a foreground-aware logic:

1. Compute CLS-Patch cosine similarity to identify foreground vs. background patches.
2. Apply Otsu thresholding (on similarity values) to separate foreground patches.
3. Run clustering **only on foreground patches**.

| Class | Algorithm | K Selection |
|---|---|---|
| `ForegroundAwareKMeans` | KMeans | Fixed K or silhouette-based auto-K |
| `ForegroundAwareSpectralClustering` | Spectral Clustering | Eigengap heuristic |
| `ForegroundAwareHDBSCAN` | HDBSCAN (density-based) | Automatic (no K needed) |
| `AutoSphericalKMeans` | L2-normalized KMeans | Silhouette score auto-K |

**Usage:**

```python
from ad_pipelines.utils.clustering import ForegroundAwareHDBSCAN

clusterer = ForegroundAwareHDBSCAN(min_cluster_size=5)
labels = clusterer.fit_predict(
    patch_features,  # (N, C) — all patch features
    cls_token,       # (C,)   — CLS token for foreground detection
)
```

---

### `image.py` — Image Conversion Utilities

Helper functions for converting between PIL images, NumPy arrays, and PyTorch tensors, including normalization and heatmap generation utilities used for visualization.

| Function | Description |
|---|---|
| `ensure_pil_image(image)` | Convert tensor or ndarray to PIL Image |
| `ensure_numpy_image(image)` | Convert PIL Image or tensor to NumPy array |
| `get_heatmap(anomaly_map, colormap)` | Render a float anomaly map as a coloured PIL heatmap |
| `concat_images(images, direction)` | Concatenate a list of PIL images horizontally or vertically |

---

## Exported Symbols

The following are exported from `utils/__init__.py` and can be imported directly from `ad_pipelines.utils`:

```python
from ad_pipelines.utils import (
    # Morphology
    erode, dilate, opening, closing,
    safe_closing, safe_closing_with_dilate,
    morphological_gradient,

    # Otsu
    batch_otsu_binarization,
    adaptive_otsu_binarization,

    # Clustering
    ForegroundAwareSpectralClustering,

    # Image utilities
    get_heatmap,
    concat_images,
)
```
