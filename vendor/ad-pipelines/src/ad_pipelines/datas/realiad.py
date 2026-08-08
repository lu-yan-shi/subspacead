from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import json
import re

from os import PathLike
from pathlib import Path

from .dataset import BaseAnomalyDataset, BaseAnomalyClassDataset, SplitType, DatasetType, AnomalyLabel


class RealIADDataset(BaseAnomalyDataset):
    """
    RealIAD dataset class for multi-view anomaly detection.
    
    RealIAD is organized differently from MVTec:
    - Uses JSON files for train/test splits
    - Multi-view images with camera indices (C0, C1, C2, ...)
    - Images are in .jpg format, masks in .png format
    - Flat directory structure: all images in one folder per category
    
    Directory structure:
        root_path/
        ├── realiad_1024/ (or realiad_raw/)
        │   ├── category_name/
        │   │   ├── image1.jpg
        │   │   ├── image1.png (mask, if exists)
        │   │   └── ...
        │   └── ...
        └── realiad_jsons/ (or realiad_jsons_sv/)
            ├── category_name.json
            └── ...
    """

    def __init__(
        self,
        root_path: PathLike,
        split: SplitType,
        json_dir: str = "realiad_jsons",
        image_dir: str = "realiad_1024",
        camera_views: Optional[List[str]] = ["C1"],
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        mask_transform: Optional[Any] = None,
        image_size: Optional[Tuple[int, int]] = None,
        random_seed: int = 42,
        **kwargs
    ):
        """
        Args:
            root_path: Path to the RealIAD dataset root
            split: Train or test split
            json_dir: Name of the JSON directory (e.g., 'realiad_jsons', 'realiad_jsons_sv')
            image_dir: Name of the image directory (e.g., 'realiad_1024', 'realiad_raw')
            camera_views: List of camera views to load (e.g., ['C0', 'C1']). If None, load all views.
            transform: Image transformation
            target_transform: Target transformation
            mask_transform: Mask transformation
            image_size: Target image size
            random_seed: Random seed
        """
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.camera_views = camera_views
        
        super().__init__(
            root_path=root_path,
            split=split,
            transform=transform,
            target_transform=target_transform,
            mask_transform=mask_transform,
            image_size=image_size,
            random_seed=random_seed,
            **kwargs
        )
        self.dataset_type = DatasetType.REALIAD

    def _load_dataset(self, **kwargs) -> None:
        """
        Load RealIAD dataset.
        Scan JSON directory for category definitions.
        """
        self.classes = {}
        json_path = self.root_path / self.json_dir
        
        if not json_path.exists():
            raise FileNotFoundError(f"JSON directory not found: {json_path}")
        
        # Iterate through JSON files to find categories
        for json_file in sorted(json_path.glob("*.json")):
            category_name = json_file.stem
            class_dataset = RealIADClassDataset(
                root_path=self.root_path,
                split=self.split,
                category=category_name,
                json_dir=self.json_dir,
                image_dir=self.image_dir,
                camera_views=self.camera_views,
                transform=self.transform,
                target_transform=self.target_transform,
                mask_transform=self.mask_transform,
                image_size=self.image_size,
                random_seed=self.random_seed,
                **kwargs
            )
            if len(class_dataset) > 0:  # Only add if has data
                self.classes[category_name] = class_dataset


class RealIADClassDataset(BaseAnomalyClassDataset):
    """
    RealIAD class dataset for multi-view anomaly detection.
    
    Image naming pattern: {category}_{index}_{defect_type}_C{camera_id}.jpg
    Example: audiojack_0001_AK_C0.jpg
    
    Defect type codes:
        - Normal: no defect code
        - AK: pit
        - BX: deformation
        - CH: abrasion
        - HS: scratch
        - PS: damage
        - QS: missing parts
        - YW: foreign objects
        - ZW: contamination
    """

    def __init__(
        self,
        root_path: PathLike,
        split: SplitType,
        category: str,
        json_dir: str = "realiad_jsons",
        image_dir: str = "realiad_1024",
        camera_views: Optional[List[str]] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Any] = None,
        mask_transform: Optional[Any] = None,
        image_size: Optional[Tuple[int, int]] = None,
        random_seed: int = 42,
        **kwargs
    ):
        """
        Args:
            root_path: Path to the RealIAD dataset root
            split: Train or test split
            category: Category name
            json_dir: Name of the JSON directory
            image_dir: Name of the image directory
            camera_views: List of camera views to load. If None, load all views.
            transform: Image transformation
            target_transform: Target transformation
            mask_transform: Mask transformation
            image_size: Target image size
            random_seed: Random seed
        """
        self.json_dir = json_dir
        self.image_dir = image_dir
        self.camera_views = camera_views
        
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
        
        self.dataset_type = DatasetType.REALIAD
        self._load_dataset()
        self._build_indices()

    def _load_dataset(self) -> None:
        """
        Load the RealIAD dataset from JSON file.
        """
        json_file = self.root_path / self.json_dir / f"{self.category}.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Get the appropriate split data
        split_key = "train" if self.split == SplitType.TRAIN else "test"
        if split_key not in data:
            return
        
        split_data = data[split_key]
        image_folder = self.root_path / self.image_dir / self.category
        
        if not image_folder.exists():
            raise FileNotFoundError(f"Image folder not found: {image_folder}")
        
        # Load images based on split
        for item in split_data:
            # item can be a string (image name) or dict with more info
            if isinstance(item, str):
                image_name = item
                defect_type = self._parse_defect_type(item)
            elif isinstance(item, dict):
                # RealIAD JSON format uses 'image_path' key with subdirectories
                image_name = item.get("image_path", item.get("filename", item.get("image", "")))
                # anomaly_class is the defect type label
                defect_type = item.get("anomaly_class", "unknown")
            else:
                continue
            
            # Skip empty or invalid image names
            if not image_name or not image_name.strip():
                continue
            
            # Extract filename from path (may contain subdirectories like "OK/S0004/filename.jpg")
            filename = Path(image_name).name
            
            # Extract view_id (camera ID) from filename
            view_id = self._parse_view_id(filename)
            
            # Extract sample_id (filename without camera suffix)
            sample_id = self._parse_sample_id(filename)
            
            # Filter by camera view if specified
            if self.camera_views is not None:
                if view_id not in self.camera_views:
                    continue
            
            # Construct full image path
            image_path = image_folder / image_name
            
            # Skip if path doesn't exist or is a directory
            if not image_path.exists() or image_path.is_dir():
                continue
            
            # Determine label: "OK" is normal, others are anomalies
            label = AnomalyLabel.NORMAL if defect_type == "OK" else AnomalyLabel.ANOMALY
            # Normalize defect type name
            normalized_defect = "good" if defect_type == "OK" else defect_type
            
            # Mask path (same name but .png extension)
            mask_path = image_path.with_suffix('.png')
            if not mask_path.exists():
                mask_path = None
            
            self.datas.append({
                "image_path": image_path,
                "mask_path": mask_path,
                "label": label,
                "defect_class": normalized_defect,
                "view_id": view_id,
                "sample_id": sample_id
            })
        
        # Collect defect types
        self.defect_types = sorted(list(set(item["defect_class"] for item in self.datas)))

    def _parse_view_id(self, filename: str) -> str:
        """
        Parse view ID (camera ID) from filename.
        
        Pattern: {category}_{index}_{defect_type}_C{camera_id}.jpg
        
        Args:
            filename: Image filename
            
        Returns:
            View ID string (e.g., "C0", "C1"), or "C0" if not found
        """
        match = re.search(r'_C(\d+)', filename)
        if match:
            return f"C{match.group(1)}"
        # Default to C0 if no camera ID found
        return "C0"
    
    def _parse_sample_id(self, filename: str) -> str:
        """
        Parse sample ID from filename (remove camera suffix).
        
        Pattern: {category}_{index}_{defect_type}_C{camera_id}.jpg
        Returns: {category}_{index}_{defect_type}
        
        Args:
            filename: Image filename
            
        Returns:
            Sample ID string without camera suffix
        """
        stem = Path(filename).stem
        # Remove _C{id} suffix
        sample_id = re.sub(r'_C\d+$', '', stem)
        return sample_id

    def _parse_defect_type(self, filename: str) -> str:
        """
        Parse defect type from filename.
        
        Pattern: {category}_{index}_{defect_type}_C{camera_id}.jpg
        or {category}_{index}_C{camera_id}.jpg for normal images
        
        Args:
            filename: Image filename
            
        Returns:
            Defect type string, or "good" for normal images
        """
        stem = Path(filename).stem
        parts = stem.split('_')
        
        # Find camera index
        camera_idx = None
        for i, part in enumerate(parts):
            if part.startswith('C') and part[1:].isdigit():
                camera_idx = i
                break
        
        if camera_idx is None:
            # No camera index found, assume normal
            return "good"
        
        # Defect type is between index and camera
        # Pattern: category_index_defect_camera or category_index_camera
        if camera_idx >= 3:
            # Has defect type
            defect_type = parts[camera_idx - 1]
            return defect_type
        else:
            # No defect type, it's normal
            return "good"

    def _get_defect_types(self) -> List[str]:
        """
        Get the list of defect types in the dataset.
        """
        return self.defect_types

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item with multi-view information.
        
        Returns:
            Dictionary with image, label, mask, and multi-view metadata (view_id, sample_id)
        """
        # Get base item from parent class
        item = super().__getitem__(idx)
        
        # Add multi-view specific fields
        item['view_id'] = self.datas[idx]['view_id']
        item['sample_id'] = self.datas[idx]['sample_id']
        
        return item

    def get_dataset_type(self) -> DatasetType:
        return DatasetType.REALIAD
