
from .dataset import BaseAnomalyDataset, BaseAnomalyClassDataset, SplitType, DatasetType, AnomalyLabel
from .mvtec import MVTecDataset, MVTecClassDataset

class VisADataset(MVTecDataset):
    def _load_dataset(self, **kwargs) -> None:
        # Load dataset implementation
        self.classes = {}
        for category in sorted(self.root_path.iterdir()):
            if category.is_dir() and not category.name.startswith('.'):
                class_dataset = VisAClassDataset(
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

class VisAClassDataset(MVTecClassDataset):
    def _load_dataset(self) -> None:
        """
        Load the MVTec dataset.
        """
        self.defect_types = self._get_defect_types()
        for defect_type in self.defect_types:
            defect_path = self.root_path / self.split.value / defect_type
            if defect_path.exists():
                for image_path in sorted(defect_path.glob("*.JPG")):
                    mask_path = None
                    if self.split == SplitType.TEST:
                        mask_path = self.root_path / "ground_truth" / defect_type / f"{image_path.stem}.png"
                        if not mask_path.exists():
                            mask_path = None

                    label = AnomalyLabel.NORMAL if defect_type == "good" else AnomalyLabel.ANOMALY
                    
                    self.datas.append({
                        "image_path": image_path,
                        "mask_path": mask_path,
                        "label": label,
                        "defect_class": defect_type
                    })