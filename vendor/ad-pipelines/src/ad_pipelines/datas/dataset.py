from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

import json
import numpy as np
import random
import torch
import warnings

from enum import Enum
from torch.utils.data import Dataset
from os import PathLike
from pathlib import Path
from PIL import Image


class DatasetType(Enum):
    """Enumeration of supported dataset types."""
    MVTEC = "mvtec"
    VISA = "visa"
    REALIAD = "realiad"
    CUSTOM = "custom"
    DEFAULT = "default"


class SplitType(Enum):
    """Enumeration of dataset split types."""
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "val"


class AnomalyLabel(Enum):
    """Enumeration of anomaly detection labels."""
    NORMAL = 0
    ANOMALY = 1


def load_config_if_exists(root_path: Path, file_name: str) -> Dict[str, Any]:
    config_path = root_path / file_name
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


class BaseAnomalyDataset(ABC):
    def __init__(
        self,
        root_path: PathLike,
        split: SplitType,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        mask_transform: Optional[Any] = None,
        image_size: Optional[Tuple[int, int]] = None,
        random_seed: int = 42,
        extra_config: Optional[Dict[str, Any]] = None
    ):
        self.root_path = Path(root_path)
        self.split = split
        
        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        self.image_size = image_size
        self.random_seed = random_seed

        self.dataset_type = DatasetType.DEFAULT

        self.extra_config = extra_config or load_config_if_exists(self.root_path, "config.json")

        # Data storage containers
        self.classes: Dict[str, BaseAnomalyClassDataset] = {}

        self._load_dataset()

        for category, class_dataset in self.classes.items():
            self.extra_config[category] = class_dataset.extra_config

    @abstractmethod
    def _load_dataset(self) -> None:
        # Load dataset implementation
        pass
    
    def items(self):
        return self.classes.items()
    
    def __len__(self) -> int:
        """Return the number of classes in the dataset."""
        return len(self.classes)
    
    def __getitem__(self, class_name: str) -> 'BaseAnomalyClassDataset':
        """Allow subscript access to class datasets."""
        return self.classes[class_name]
    
    def __str__(self):
        return f"Dataset{self.__class__.__name__}(root_path={self.root_path}, split={self.split}, dataset_type={self.dataset_type})"

class BaseAnomalyClassDataset(Dataset, ABC):
    """
    Base class for anomaly detection datasets.

    Provides common interfaces and functionality for anomaly detection datasets,
    supporting few-shot and zero-shot experiments with episode generation for
    support and query sets.

    Category mapping:
      - category_map can be:
        * int: 0 or 1 -> label for that category
        * dict: { 'label': int, 'defect_type': str } (defect_type optional)
        * callable: func(image_path: Optional[Path], category: str) -> (label:int, defect_type:str)
      - If no explicit mapping provided, default rule is:
        * category in ("good","normal","none") -> NORMAL (0)
        * otherwise -> ANOMALY (1) with defect_type = category
    """

    def __init__(
        self,
        root_path: PathLike,
        split: SplitType,
        category: str,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        mask_transform: Optional[Any] = None,
        image_size: Optional[Tuple[int, int]] = None,
        random_seed: int = 42,
        extra_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.root_path = Path(root_path)
        self.category = category
        self.split = split
        
        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        self.image_size = image_size
        self.random_seed = random_seed

        self.extra_config = extra_config or load_config_if_exists(self.root_path, "config.json")

        # Data storage containers
        self.datas = []
        """
        List of dictionaries containing:
        {
            "image_path": Path,
            "mask_path": Optional[Path],
            "label": AnomalyLabel,
            "defect_class": str
        }
        """
        self.defect_types: List[str] = []

        # Index caches (built after loading data)
        self._normal_indices: List[int] = []
        self._anomaly_indices: List[int] = []
        self._indices_by_defect: Dict[str, List[int]] = {}

        # Dataset metadata
        self.dataset_info: Dict[str, Any] = {}

        # Load dataset and build indices
        # self._load_dataset()
        # self._build_indices()

    @abstractmethod
    def _load_dataset(self) -> None:
        """
        Load dataset implementation.

        Subclasses should populate:
          - self.image_paths (List[Path])
          - self.labels (List[int])  <-- can use self.resolve_category_mapping(...)
          - self.mask_paths (List[Optional[Path]])
          - self.defect_types (List[str])
        """
        pass

    def _build_indices(self) -> None:
        self._normal_indices = [
            i for i, data in enumerate(self.datas)
            if data["label"] == AnomalyLabel.NORMAL
        ]
        self._anomaly_indices = [
            i for i, data in enumerate(self.datas)
            if data["label"] == AnomalyLabel.ANOMALY
        ]

    @abstractmethod
    def get_dataset_type(self) -> DatasetType:
        pass

    def __len__(self) -> int:
        return len(self.datas)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = self.datas[idx]["image_path"]
        image = self._load_image(image_path)
        label = self.datas[idx]["label"]
        defect_type = self.datas[idx]["defect_class"]

        mask = None
        if self.datas[idx]["mask_path"] is not None:
            mask = self._load_mask(self.datas[idx]["mask_path"])
        if self.split == SplitType.TEST and mask is None and label == AnomalyLabel.NORMAL:
            mask = Image.new('L', image.size, 0)  # Create a blank mask for normal images

        if self.transform is not None:
            image = self.transform(image)
        if mask is not None and self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return {
            'image': image,
            'label': label,
            'mask': mask,
            'image_path': image_path,
            'defect_type': defect_type,
            'category': self.category,
            'index': idx
        }

    def _load_image(self, image_path: Path) -> Image.Image:
        image = Image.open(image_path).convert('RGB')
        if self.image_size:
            image = image.resize(self.image_size, Image.LANCZOS)
        return image

    def _load_mask(self, mask_path: Union[Path, Tuple[Path, ...], List[Path]]) -> Optional[Image.Image]:
        """
        Load mask(s) and combine if multiple paths are provided.
        
        Supports:
        - Single Path: Load one mask
        - Tuple/List of Paths: Load and combine multiple masks using logical OR (max)
        
        This is useful when one image may have multiple mask files.
        """
        # Handle both single path and multiple paths (tuple/list)
        mask_paths = mask_path if isinstance(mask_path, (tuple, list)) else [mask_path]
        
        # Load and combine all masks
        masks = []
        for single_mask_path in mask_paths:
            if single_mask_path.exists():
                loaded_mask = Image.open(single_mask_path).convert('L')
                if self.image_size:
                    loaded_mask = loaded_mask.resize(self.image_size, Image.NEAREST)
                masks.append(np.array(loaded_mask))
        
        if not masks:
            return None
        
        # Combine masks using logical OR (max): any pixel marked as anomaly in any mask
        combined_mask_array = np.maximum.reduce(masks) if len(masks) > 1 else masks[0]
        return Image.fromarray(combined_mask_array)
    
    def get_samples_by_interest_indices(
        self, 
        indices: List[int], 
        sample_count: int, 
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        rng = self._rng(seed)
        selected_indices = rng.sample(indices, sample_count)
        items = {}
        for i in selected_indices:
            item = self[i]  # Use __getitem__ to get the processed item
            for key, value in item.items():
                if key not in items:
                    items[key] = []
                items[key].append(value)
        return items

    def get_normal_samples(self, sample_count: int, seed: Optional[int] = None) -> Dict[str, Any]:
        return self.get_samples_by_interest_indices(
            self._normal_indices, sample_count, seed
        )

    def get_anomaly_samples(self, sample_count: int, seed: Optional[int] = None) -> Dict[str, Any]:
        return self.get_samples_by_interest_indices(
            self._anomaly_indices, sample_count, seed
        )

    def get_defect_types(self) -> List[str]:
        raise NotImplementedError("Subclasses must implement this method")

    def get_class_types(self) -> List[str]:
        """
        Get the list of class types in the dataset.
        """
        return list(set(self.class_types))

    def get_dataset_statistics(self) -> Dict[str, Any]:
        normal_count = len(self.get_normal_samples())
        anomaly_count = len(self.get_anomaly_samples())
        defect_types = self.get_defect_types()

        stats = {
            'total_samples': len(self),
            'normal_samples': normal_count,
            'anomaly_samples': anomaly_count,
            'anomaly_ratio': anomaly_count / len(self) if len(self) > 0 else 0,
            'defect_types': defect_types,
            'defect_type_count': len(defect_types),
            'category': self.category,
            'split': self.split.value,
            'dataset_type': self.get_dataset_type().value,
            'category_map_keys': list(self.category_map.keys())
        }

        defect_type_counts = {}
        for defect_type in defect_types:
            defect_type_counts[defect_type] = len(
                self.get_samples_by_defect_type(defect_type)
            )
        stats['defect_type_distribution'] = defect_type_counts

        return stats

    def _rng(self, seed: Optional[int] = None) -> random.Random:
        if seed is None:
            seed = self.random_seed
        return random.Random(seed)

    @staticmethod
    def get_collate_fn() -> Callable:
        def collate(batch):
            result = {
                'image': [item['image'] for item in batch],
                'label': [item['label'] for item in batch],
                'mask': [item['mask'] for item in batch],
                'image_path': [item['image_path'] for item in batch],
                'defect_type': [item['defect_type'] for item in batch],
                'category': [item['category'] for item in batch],
                'index': [item['index'] for item in batch],
            }
            # Pass through any extra per-item fields (e.g. view_id, sample_id)
            extra_keys = set(batch[0].keys()) - set(result.keys())
            for key in extra_keys:
                result[key] = [item[key] for item in batch]
            return result
        return collate
