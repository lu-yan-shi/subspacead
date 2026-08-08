# Datas Module

## Overview

The `datas` module provides a standardized framework for loading and accessing anomaly detection datasets. It supports multiple benchmark dataset formats through a common interface, enabling the pipelines to work seamlessly across different datasets.

## File Structure

```
datas/
├── dataset.py         # Abstract base classes, enums, and shared logic
├── mvtec.py           # MVTec AD dataset implementation
├── visa.py            # VisA dataset implementation
├── realiad.py         # Real-IAD multi-view dataset implementation
├── __init__.py        # Module exports
└── README.md          # This documentation
```

---

## Supported Datasets

| Class | Dataset | Categories | Notes |
|---|---|---|---|
| `MVTecDataset` / `MVTecClassDataset` | MVTec AD | 15 | Textures and objects, `.png` images |
| `VisADataset` / `VisAClassDataset` | VisA | 12 | `.JPG` images, no `_mask` suffix |
| `RealIADDataset` / `RealIADClassDataset` | Real-IAD | 30 | Multi-view (up to 5 cameras), JSON-based splits, `.jpg` images |

---

## Class Hierarchy

```
BaseAnomalyDataset (ABC)          — manages all categories of a dataset
  ├── MVTecDataset
  ├── VisADataset
  └── RealIADDataset

BaseAnomalyClassDataset (ABC)     — manages images for a single category
  ├── MVTecClassDataset
  ├── VisAClassDataset
  └── RealIADClassDataset
```

---

## Core Classes

### `BaseAnomalyDataset`

Manages multiple object categories. Acts as a dict-like container mapping category names to their `BaseAnomalyClassDataset` instances.

**Constructor arguments:**

| Argument | Type | Default | Description |
|---|---|---|---|
| `root_path` | `str / Path` | *(required)* | Root directory of the dataset |
| `split` | `SplitType` | *(required)* | `TRAIN` or `TEST` |
| `transform` | callable | `None` | Transform applied to images |
| `mask_transform` | callable | `None` | Transform applied to masks |
| `image_size` | `Tuple[int, int]` | `None` | Resize all images to this size |
| `random_seed` | `int` | `42` | Seed for reproducible sampling |

### `BaseAnomalyClassDataset`

Manages normal and anomalous images for a **single category**. Inherits from `torch.utils.data.Dataset`.

**Key methods:**

| Method | Description |
|---|---|
| `get_normal_samples(sample_count, seed)` | Sample N normal images from `TRAIN` split |
| `get_anomaly_samples(sample_count, seed)` | Sample N anomalous images |
| `get_dataset_statistics()` | Returns count, ratio, defect type distribution |
| `get_collate_fn()` | Returns a collate function suitable for `DataLoader` |

---

## Enumerations

```python
class SplitType(Enum):
    TRAIN = "train"    # Normal training images (used as prompt/reference)
    TEST  = "test"     # Mixed normal + anomalous images with ground truth

class AnomalyLabel(Enum):
    NORMAL  = 0
    ANOMALY = 1

class DatasetType(Enum):
    MVTEC      = "mvtec"
    VISA       = "visa"
    REALIAD    = "realiad"
    CUSTOM     = "custom"
```

---

## Dataset Item Format

Each `__getitem__` call returns a dict:

```python
{
    "image":       PIL.Image,          # The input image
    "label":       AnomalyLabel,       # NORMAL (0) or ANOMALY (1)
    "mask":        PIL.Image | None,   # Ground truth anomaly mask (None for normal)
    "image_path":  Path,               # Absolute path to the image file
    "defect_type": str,                # Defect category string (e.g. "crack", "good")
    "category":    str,                # Object category (e.g. "bottle")
    "index":       int,                # Dataset index
    # RealIAD only:
    "view_id":     str,                # Camera view identifier (e.g. "C0", "C1")
    "sample_id":   str,                # Physical object ID shared across views
}
```

Masks are binary PNG images where `0` = normal pixel, `255` = anomalous pixel.

---

## Expected Directory Structures

### MVTec AD

```
mvtec/
├── bottle/
│   ├── train/
│   │   └── good/              # Normal training images (*.png)
│   ├── test/
│   │   ├── good/              # Normal test images
│   │   ├── broken_large/      # Anomaly type 1
│   │   └── broken_small/      # Anomaly type 2
│   └── ground_truth/
│       ├── broken_large/      # Masks: {stem}_mask.png
│       └── broken_small/
└── cable/ ...
```

### VisA

```
visa/
├── cashew/
│   ├── train/
│   │   └── good/              # Normal training images (*.JPG)
│   ├── test/
│   │   ├── good/
│   │   └── Defect/
│   └── ground_truth/
│       └── Defect/            # Masks: {stem}.png (no _mask suffix)
└── ...
```

### Real-IAD

```
realiad_root/
├── realiad_1024/           # (or realiad_raw/ for full-resolution)
│   ├── audiojack/
│   │   ├── OK/
│   │   │   └── S0001/
│   │   │       ├── audiojack_0001_C0.jpg
│   │   │       ├── audiojack_0001_C1.jpg
│   │   │       └── ...
│   │   └── AK/             # defect subdirectory
│   │       └── S0100/
│   │           ├── audiojack_0100_AK_C0.jpg
│   │           ├── audiojack_0100_AK_C0.png  # mask (if exists)
│   │           └── ...
│   └── ...
└── realiad_jsons/          # (or realiad_jsons_sv/ for single-view splits)
    ├── audiojack.json
    └── ...
```

**Image naming convention:** `{category}_{index}_{defect_type}_C{camera_id}.jpg`
(normal images omit the defect type: `{category}_{index}_C{camera_id}.jpg`)

**Known defect type codes:** `AK` (pit), `BX` (deformation), `CH` (abrasion), `HS` (scratch), `PS` (damage), `QS` (missing parts), `YW` (foreign objects), `ZW` (contamination)

---

## Usage

### Loading a Dataset

```python
from ad_pipelines.datas import MVTecDataset, SplitType

test_dataset   = MVTecDataset("/path/to/mvtec", split=SplitType.TEST)
prompt_dataset = MVTecDataset("/path/to/mvtec", split=SplitType.TRAIN)

# Access a specific category
bottle = test_dataset["bottle"]
print(f"{len(bottle)} test samples in bottle")
```

### Few-Shot Sampling

```python
# Get 4 normal images to use as prompt (reference) set
normal_samples = prompt_dataset["bottle"].get_normal_samples(sample_count=4, seed=42)
# normal_samples["image"]       → list of PIL.Image
# normal_samples["image_path"]  → list of Path
```

### Using with a DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    test_dataset["bottle"],
    batch_size=8,
    shuffle=False,
    collate_fn=test_dataset["bottle"].get_collate_fn(),
)

for batch in dataloader:
    images      = batch["image"]       # list of PIL.Image
    labels      = batch["label"]       # list of AnomalyLabel
    masks       = batch["mask"]        # list of PIL.Image | None
    defect_types = batch["defect_type"]
```

### Loading Real-IAD (Multi-View)

```python
from ad_pipelines.datas import RealIADDataset, SplitType

# Load all camera views (default: C1 only)
test_dataset   = RealIADDataset(
    "/path/to/realiad_root",
    split=SplitType.TEST,
    json_dir="realiad_jsons",
    image_dir="realiad_1024",
    camera_views=None,          # None = all views; or e.g. ["C0", "C1"]
)
prompt_dataset = RealIADDataset(
    "/path/to/realiad_root",
    split=SplitType.TRAIN,
    json_dir="realiad_jsons",
    image_dir="realiad_1024",
    camera_views=None,
)

# Each item carries extra multi-view fields:
item = test_dataset["audiojack"][0]
print(item["view_id"])    # e.g. "C1"
print(item["sample_id"])  # e.g. "audiojack_0001_AK"
```

---

## Adding a Custom Dataset

### 1. Implement `BaseAnomalyClassDataset`

```python
from ad_pipelines.datas.dataset import BaseAnomalyClassDataset, AnomalyLabel, DatasetType

class MyClassDataset(BaseAnomalyClassDataset):
    def _load_dataset(self):
        for img_path in (self.root_path / "normal").glob("*.png"):
            self.datas.append({
                "image_path":  img_path,
                "mask_path":   None,
                "label":       AnomalyLabel.NORMAL,
                "defect_class": "good",
            })
        for img_path in (self.root_path / "anomaly").glob("*.png"):
            mask = self.root_path / "masks" / f"{img_path.stem}_mask.png"
            self.datas.append({
                "image_path":  img_path,
                "mask_path":   mask if mask.exists() else None,
                "label":       AnomalyLabel.ANOMALY,
                "defect_class": "defect",
            })

    def get_dataset_type(self):
        return DatasetType.CUSTOM
```

### 2. Implement `BaseAnomalyDataset`

```python
from ad_pipelines.datas.dataset import BaseAnomalyDataset

class MyDataset(BaseAnomalyDataset):
    def _load_dataset(self):
        for category_dir in self.root_path.iterdir():
            if category_dir.is_dir():
                self.classes[category_dir.name] = MyClassDataset(
                    root_path=category_dir,
                    split=self.split,
                    category=category_dir.name,
                    image_size=self.image_size,
                    random_seed=self.random_seed,
                )
```
