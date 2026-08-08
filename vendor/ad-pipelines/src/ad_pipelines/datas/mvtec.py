from typing import Dict, List, Tuple, Optional, Union, Any, Callable

from os import PathLike
from pathlib import Path

from .dataset import BaseAnomalyDataset, BaseAnomalyClassDataset, SplitType, DatasetType, AnomalyLabel

class MVTecDataset(BaseAnomalyDataset):
    """
    MVTec dataset class for anomaly detection.
    """

    def __init__(
        self,
        root_path: PathLike,
        split: SplitType,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        mask_transform: Optional[Any] = None,
        image_size: Optional[Tuple[int, int]] = None,
        random_seed: int = 42,
        **kwargs
    ):
        self.root_path = Path(root_path)
        self.split = split
        
        self.transform = transform
        self.target_transform = target_transform
        self.mask_transform = mask_transform
        self.image_size = image_size
        self.random_seed = random_seed

        self._load_dataset()

    def _load_dataset(self, **kwargs) -> None:
        # Load dataset implementation
        self.classes = {}
        for category in self.root_path.iterdir():
            if category.is_dir() and not category.name.startswith('.'):
                class_dataset = MVTecClassDataset(
                    root_path=category,
                    split=self.split,
                    category=category.name,
                    transform=self.transform,
                    target_transform=self.target_transform,
                    mask_transform=self.mask_transform,
                    image_size=self.image_size,
                    random_seed=self.random_seed,
                    **kwargs
                )
                self.classes[category.name] = class_dataset

class MVTecClassDataset(BaseAnomalyClassDataset):
    """
    MVTec dataset class for anomaly detection.
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
        **kwargs
    ):
        super().__init__(
            root_path=root_path,
            split=split,
            category=category,
            transform=transform,
            target_transform=target_transform,
            mask_transform=mask_transform,
            image_size=image_size,
            random_seed=random_seed,
            **kwargs
        )
        
        self.dataset_type = DatasetType.MVTEC

        self._load_dataset()
        self._build_indices()

    def _load_dataset(self) -> None:
        """
        Load the MVTec dataset.
        """
        self.defect_types = self._get_defect_types()
        for defect_type in self.defect_types:
            defect_path = self.root_path / self.split.value / defect_type
            if defect_path.exists():
                for image_path in sorted(defect_path.glob("*.png")):
                    mask_path = None
                    if self.split == SplitType.TEST:
                        mask_path = self.root_path / "ground_truth" / defect_type / image_path.name.replace(".png", "_mask.png")
                        if not mask_path.exists():
                            mask_path = None

                    label = AnomalyLabel.NORMAL if defect_type == "good" else AnomalyLabel.ANOMALY
                    
                    self.datas.append({
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "label": label,
                        "defect_class": defect_type
                    })

    def _get_defect_types(self) -> List[str]:
        """
        Get the list of defect types in the dataset.
        """
        defects_path = self.root_path / self.split.value
        self.defect_types = [subdir.name for subdir in sorted(defects_path.iterdir()) if subdir.is_dir()]
        return self.defect_types

    def get_dataset_type(self) -> DatasetType:
        return DatasetType.MVTEC
