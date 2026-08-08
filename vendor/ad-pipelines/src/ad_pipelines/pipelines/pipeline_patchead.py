from typing import Dict, Any, Optional, Tuple, Union, List, Sequence
from typing_extensions import Self
from collections import defaultdict

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers.utils import ModelOutput
from torchvision.transforms import ToTensor, ToPILImage
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from os import PathLike
from pathlib import Path
from PIL import Image
import numpy as np

from ..models.model_base import BaseModel, VisionEncoderOutput
from .pipeline_base import AnomalyDetectionPipelineBase, ADPerClassEvaluationOutput
from ..datas import BaseAnomalyDataset, BaseAnomalyClassDataset, AnomalyLabel
from ..utils import get_heatmap, concat_images

to_tensor = ToTensor()
to_pil = ToPILImage()

ALL_ROTATION_ANGLES = (0, 45, 90, 135, 180, 225, 270, 315)
ROTATE90_ANGLES = (0, 90, 180, 270)
VALID_AUGMENTATION_MODES = {"force", "rotate", "rotate90", "flip", "auto", "none"}

@dataclass
class PatchEADOutput(ModelOutput):
    """
    Output class for PatchEAD pipeline.
    Contains the features extracted from the model.
    """
    anomaly_map: torch.Tensor
    anomaly_score: Optional[torch.Tensor] = None
    attentioned_anomaly_map: Optional[torch.Tensor] = None
    attentioned_anomaly_score: Optional[torch.Tensor] = None
    attention_map: Optional[torch.Tensor] = None


class PatchEADPipeline(AnomalyDetectionPipelineBase):
    """
    Base class for anomaly detection pipelines
    Provides unified interface for handling different visual model backbones
    """
    
    def __init__(
        self,
        model: BaseModel,
        attention_roi_threshold: float = 0.05,
        attention_unroi_weight: float = 0.5,
        cosine_sim_clamp: tuple[float, float] = (-1, 1),
        analyze_augment_sample_count: int = 30,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        eval_device: Optional[Union[str, torch.device]] = None,
        eval_dtype: Optional[Union[str, torch.dtype]] = None,
        **kwargs
    ):
        """PatchEAD Pipeline

        Args:
            model (BaseModel): _description_
            attention_roi_threshold (Optional[float], optional): A threshold to determine ROI using [CLS]
                attention map. Defaults to 0.05.
            device (Optional[Union[str, torch.device]], optional): Pipeline device. Defaults to None.
            dtype (Optional[Union[str, torch.dtype]], optional): Pipeline dtype. Defaults to None.
            eval_device (Optional[Union[str, torch.device]], optional): Device for evaluation. Defaults to None (use self.device).
        """        
        super().__init__(model, device, dtype, eval_device=eval_device, eval_dtype=eval_dtype, **kwargs)
        self.attention_roi_threshold = attention_roi_threshold
        self.attention_unroi_weight = attention_unroi_weight
        self.cosine_sim_clamp = cosine_sim_clamp
        self.analyze_augment_sample_count = analyze_augment_sample_count
        self._augmentation_analysis_by_class: Dict[str, Any] = {}

        self.resolution = self.ensure_tuple_resolution(resolution)

        self.num_prefix_tokens = self.model.config.num_register_tokens + int(self.model.config.is_cls_token)

    def _reset_evaluation_metadata(self) -> None:
        self._augmentation_analysis_by_class = {}

    def _get_evaluation_config_metadata(self) -> Dict[str, Any]:
        if not self._augmentation_analysis_by_class:
            return {}
        return {"augmentation_analysis": self._augmentation_analysis_by_class}

    @staticmethod
    def _build_augmentation_analysis_record(
        sample_count: int,
        can_rotate: bool,
        can_rotate90s: bool,
        can_flip: bool,
        similarity_scores: Tuple[float, float, float, float],
        rotation_angles: Sequence[int],
        rot_tolerance: float,
        flip_tolerance: float,
        selected_prompt_augmentation: Dict[str, Any],
        analysis_split: str = "test",
    ) -> Dict[str, Any]:
        test_mean_sim, rot_mean_sim, rot_90s_mean_sim, flip_mean_sim = similarity_scores
        serialized_prompt_augmentation = {}
        for key, value in selected_prompt_augmentation.items():
            if key == "rotation_angles":
                serialized_prompt_augmentation[key] = [int(angle) for angle in value]
            else:
                serialized_prompt_augmentation[key] = value

        return {
            "sample_count": int(sample_count),
            "analysis_split": analysis_split,
            "rotation_angles": [int(angle) for angle in rotation_angles],
            "rot_tolerance": float(rot_tolerance),
            "flip_tolerance": float(flip_tolerance),
            "can_rotate": bool(can_rotate),
            "can_rotate90s": bool(can_rotate90s),
            "can_flip": bool(can_flip),
            "similarity_scores": {
                "test_mean": float(test_mean_sim),
                "rotate_mean": float(rot_mean_sim),
                "rotate_90s_mean": float(rot_90s_mean_sim),
                "flip_mean": float(flip_mean_sim),
            },
            "selected_prompt_augmentation": serialized_prompt_augmentation,
        }

    @staticmethod
    def _resolve_augmentation_mode(
        augmentation_mode: Optional[str] = None,
        augmentation: bool = False,
    ) -> str:
        if augmentation_mode is None:
            augmentation_mode = "auto" if augmentation else "none"

        augmentation_mode = augmentation_mode.lower()
        if augmentation_mode not in VALID_AUGMENTATION_MODES:
            raise ValueError(
                "augmentation_mode must be one of "
                f"{sorted(VALID_AUGMENTATION_MODES)}, got {augmentation_mode!r}"
            )

        return augmentation_mode

    @staticmethod
    def _rotation_angles_for_mode(
        augmentation_mode: str,
        requested_angles: Optional[Sequence[int]] = None,
    ) -> Tuple[int, ...]:
        if requested_angles is None:
            if augmentation_mode == "rotate90":
                return ROTATE90_ANGLES
            return ALL_ROTATION_ANGLES

        normalized_angles = tuple(int(angle) for angle in requested_angles)
        if augmentation_mode == "rotate90":
            normalized_angles = tuple(angle for angle in normalized_angles if angle % 90 == 0)
            if not normalized_angles:
                return ROTATE90_ANGLES

        return normalized_angles

    def _build_prompt_augmentation_kwargs(
        self,
        augmentation_mode: str,
        requested_angles: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        if augmentation_mode == "none":
            return {}

        aug_kwargs: Dict[str, Any] = {"augmentation": True}
        if augmentation_mode in {"force", "rotate", "rotate90"}:
            aug_kwargs["rotation_angles"] = self._rotation_angles_for_mode(
                augmentation_mode,
                requested_angles=requested_angles,
            )
        if augmentation_mode in {"force", "flip"}:
            aug_kwargs["is_flip"] = True

        return aug_kwargs

    def _build_auto_prompt_augmentation_kwargs(
        self,
        can_rotate: bool,
        can_rotate90s: bool,
        can_flip: bool,
        requested_angles: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        aug_kwargs: Dict[str, Any] = {}

        if can_rotate:
            aug_kwargs.update(
                self._build_prompt_augmentation_kwargs(
                    "rotate",
                    requested_angles=requested_angles,
                )
            )
        elif can_rotate90s:
            aug_kwargs.update(
                self._build_prompt_augmentation_kwargs(
                    "rotate90",
                    requested_angles=requested_angles,
                )
            )

        if can_flip:
            aug_kwargs["augmentation"] = True
            aug_kwargs["is_flip"] = True

        return aug_kwargs

    @staticmethod
    def _serialize_selected_prompt_augmentation(
        augmentation_mode: str,
        aug_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        selected_prompt_augmentation: Dict[str, Any] = {"augmentation_mode": augmentation_mode}
        if "rotation_angles" in aug_kwargs:
            selected_prompt_augmentation["rotation_angles"] = tuple(aug_kwargs["rotation_angles"])
        if aug_kwargs.get("is_flip"):
            selected_prompt_augmentation["is_flip"] = True

        return selected_prompt_augmentation

    @staticmethod
    def _append_aug_progress(
        progress_desc: Optional[str],
        aug_kwargs: Dict[str, Any],
    ) -> Optional[str]:
        if not progress_desc:
            return progress_desc

        suffixes: List[str] = []
        rotation_angles = aug_kwargs.get("rotation_angles")
        if rotation_angles:
            if any(angle % 90 != 0 for angle in rotation_angles):
                suffixes.append("Rot✅")
            else:
                suffixes.append("Rot90✅")
        if aug_kwargs.get("is_flip"):
            suffixes.append("Flip✅")

        if not suffixes:
            return progress_desc

        return progress_desc[:-1] + "|" + "|".join(suffixes) + "]"

    @staticmethod
    def ensure_tuple_resolution(resolution: Union[int, Tuple[int, int], None]) -> Tuple[int, int]:
        """
        Convert resolution parameter to tuple format.
        
        Args:
            resolution: Resolution specification as an integer (for square resolution),
                       a tuple of (width, height), or None.
        
        Returns:
            Tuple[int, int] or None: A tuple of (width, height) if resolution is provided,
                                    or None if resolution is None.
        
        Examples:
            >>> ensure_tuple_resolution(224)
            (224, 224)
            >>> ensure_tuple_resolution((224, 448))
            (224, 448)
            >>> ensure_tuple_resolution(None)
            None
        """
        """Get the tuple representation of resolution if not None."""
        if resolution is not None:
            if isinstance(resolution, int):
                return (resolution, resolution)
        return resolution
    
    @staticmethod
    def get_image_resolution(image: Union[torch.Tensor, Image.Image, np.ndarray, list]) -> Tuple[int, int]:
        """
        Get the resolution (width, height) of an image.
        This static method extracts the dimensions of an image regardless of its format,
        supporting PyTorch tensors, PIL Images, NumPy arrays, and lists of images.
        Args:
            image (Union[torch.Tensor, Image.Image, np.ndarray, list]): 
                The input image in one of the supported formats:
                - torch.Tensor: Expected shape (..., H, W) or (..., C, H, W)
                - Image.Image: PIL Image object
                - np.ndarray: NumPy array with shape (..., H, W) or (..., C, H, W)
                - list: List of images in any of the above formats (uses first element)
        Returns:
            Tuple[int, int]: A tuple containing (width, height) of the image.
        Raises:
            ValueError: If the image type is not one of the supported formats.
        Note:
            For torch.Tensor and np.ndarray, the method assumes the last two dimensions
            represent height and width respectively, returning them in (width, height) order.
        """
        if isinstance(image, list):
            image = image[0]
        if isinstance(image, torch.Tensor):
            return image.shape[-1], image.shape[-2]
        elif isinstance(image, Image.Image):
            return image.size
        elif isinstance(image, np.ndarray):
            return image.shape[-1], image.shape[-2]
        
        raise ValueError("Unsupported image type")

    def preprocess(
        self, 
        images: Union[Image.Image, torch.Tensor, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Preprocess images for the model."""
        return self.model.preprocess(
            images, 
            resolution=resolution,
            do_normalize=kwargs.get("do_normalize", True),
            do_resize=kwargs.get("do_resize", None),
            do_rescale=kwargs.get("do_rescale", True),
        ).to(device=self.device, dtype=self.dtype)


    def _get_salient_map(
        self,
        shape: Tuple[int, int, int],
        cls_token: Optional[torch.Tensor]=None,
        patch_features: Optional[torch.Tensor]=None,
        attentions: Optional[torch.Tensor]=None,
        eps: float = 1e-6,
        **kwargs
    ) -> torch.Tensor:
        attention_roi_threshold = kwargs.get("attention_roi_threshold", self.attention_roi_threshold)
        attention_unroi_weight = kwargs.get("attention_unroi_weight", self.attention_unroi_weight)
        interest_token_idx = kwargs.get("interest_token_idx", 0)

        if attentions is None:
            raise ValueError("Attentions must be provided to compute salient map in PatchEAD pipeline.")
        
        attention_map = attentions[-1].clone()  # Use the last layer's attention map
        del attentions # Free memory

        batch_size, height, width = shape

        # Get attention map for the [CLS] token
        if isinstance(interest_token_idx, int):
            interest_token_idx = (interest_token_idx, interest_token_idx+1)
        attention_map = attention_map[:, :, interest_token_idx[0]:interest_token_idx[1], self.num_prefix_tokens:]

        # Get the average attention across all heads
        attention_map = attention_map.mean(dim=1)  # Shape: [BS, N_interest_tokens, N_patches]
        # Normalize to 0~1 range
        attention_map = (attention_map - attention_map.min(dim=2, keepdim=True)[0]) / \
            (attention_map.max(dim=2, keepdim=True)[0] - attention_map.min(dim=2, keepdim=True)[0] + 1e-12)
        attention_map = attention_map.mean(dim=1)  # Average across interest tokens
        attention_map = torch.where(
            attention_map > attention_roi_threshold, 1.0, attention_unroi_weight
        )
        attention_map = attention_map.reshape(batch_size, height, width)  # Reshape to [BS, H, W]

        return attention_map


    def get_features(
        self, 
        image_tensors: torch.Tensor,
        return_attentions: bool = False,
        output_feature_maps_indices: Optional[Sequence[int]] = None,
        **kwargs
    ) -> VisionEncoderOutput:
        """Extract features from the model."""
        return self.model.get_features(
            image_tensors, 
            return_attentions=return_attentions,
            output_feature_maps_indices=output_feature_maps_indices,
            **kwargs
        )

    def _concatenate_nested_tensors(self, tensor_list: list) -> tuple:
        """Concatenate nested tensor structures (like attentions/hidden_states)"""
        if not tensor_list or tensor_list[0] is None:
            return None
        
        if isinstance(tensor_list[0], (list, tuple)):
            # Handle nested structure - concatenate each layer separately
            num_layers = len(tensor_list[0])
            return tuple(
                torch.cat([batch_tensors[layer_idx] for batch_tensors in tensor_list], dim=0)
                for layer_idx in range(num_layers)
            )
        else:
            # Handle single tensor
            return torch.cat(tensor_list, dim=0)
        
    
    def augment_images(
        self,
        image_tensors: torch.Tensor,
        rotation_angles: Tuple[int, ...] = (0, ),
        is_flip: bool = False,
    ):
        """Augment images with rotations and flips
        
        Args:
            image_tensors: Input image tensors of shape [BS, C, H, W]
            rotation_angles: Tuple of rotation angles (in degrees) to apply.
                Supports any angle (e.g., 0, 45, 90, 135, 180, 225, 270, 315)
            is_flip: Whether to include horizontal flip augmentation
            
        Returns:
            Augmented image tensors of shape [BS * N_augments, C, H, W]
        """            
        augmented_images = []
        for angle in rotation_angles:
            # Use torchvision functional rotate for arbitrary angles
            rotated = TF.rotate(
                image_tensors, angle=angle, interpolation=TF.InterpolationMode.BILINEAR, 
            )
            augmented_images.append(rotated)
            if is_flip:
                flipped = torch.flip(rotated, dims=[3])  # Horizontal flip
                augmented_images.append(flipped)
        
        augmented_images = torch.cat(augmented_images, dim=0)
        
        return augmented_images


    def get_prompt_features(
        self,
        prompt_images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        augmentation: bool = False,
        return_origin_dict: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, VisionEncoderOutput]:
        resolution = self.ensure_tuple_resolution(resolution or self.resolution)

        if augmentation:
            augment_kwargs = {}
            if "rotation_angles" in kwargs:
                augment_kwargs["rotation_angles"] = kwargs["rotation_angles"]
            if "is_flip" in kwargs:
                augment_kwargs["is_flip"] = kwargs["is_flip"]

            prompt_inputs = self.preprocess(
                prompt_images, 
                do_resize=False,
                do_normalize=False,  # Don't normalize yet - will normalize after augmentation
            )

            prompt_inputs = self.augment_images(
                prompt_inputs,
                **augment_kwargs
            )

            prompt_inputs = self.preprocess(
                prompt_inputs,
                resolution=resolution,
                do_rescale=False,
            )

        else:
            prompt_inputs = self.preprocess(
                prompt_images, 
                resolution=resolution,
            )

        extra_process_kwargs = {}
        if "output_feature_maps_indices" in kwargs:
            extra_process_kwargs["output_feature_maps_indices"] = kwargs["output_feature_maps_indices"]
            if len(kwargs["output_feature_maps_indices"]) > 1:
                return_origin_dict = True

        prompt_features = self.get_features(
            prompt_inputs,
            return_attentions=False,
            **extra_process_kwargs
        )
        # Flatten the features at batch and W,H dimensions

        if return_origin_dict:
            return prompt_features
        else:
            return prompt_features.feature_maps[-1].permute(0, 2, 3, 1).flatten(0, 2)  # Shape: [N_patches, C]


    def _collect_augmentation_samples(
        self,
        test_dataloader: Optional[DataLoader] = None,
        prompt_dataset: Optional[BaseAnomalyClassDataset] = None,
        seed: int = 42,
        source: str = "test",
        sample_count: Optional[int] = None,
        shots: Optional[int] = None,
        yield_rate: Optional[float] = None,
    ) -> Tuple[List[Any], int]:
        del yield_rate

        requested_sample_count = sample_count or shots or self.analyze_augment_sample_count

        if source == "test":
            if test_dataloader is None:
                raise ValueError("test_dataloader must be provided when source='test'.")

            sample_images = []
            sample_paths = []
            observed_batch_size = 0
            for batch in test_dataloader:
                if observed_batch_size == 0:
                    observed_batch_size = len(batch["image"])
                sample_images.extend(batch["image"])
                sample_paths.extend(batch["image_path"])
                if len(sample_images) >= requested_sample_count:
                    sample_images = sample_images[:requested_sample_count]
                    break

            if getattr(test_dataloader, "generator", None) is not None:
                test_dataloader.generator.manual_seed(seed)
            return sample_images, observed_batch_size

        if source == "train":
            if prompt_dataset is None:
                raise ValueError("prompt_dataset must be provided when source='train'.")

            available_normal_count = len(getattr(prompt_dataset, "_normal_indices", ()))
            if available_normal_count == 0:
                return [], 0

            requested_sample_count = min(requested_sample_count, available_normal_count)
            loader_batch_size = requested_sample_count
            if test_dataloader is not None and test_dataloader.batch_size is not None:
                loader_batch_size = min(test_dataloader.batch_size, requested_sample_count)

            train_generator = torch.Generator()
            train_generator.manual_seed(seed)
            train_dataloader = DataLoader(
                prompt_dataset,
                batch_size=max(1, loader_batch_size),
                shuffle=True,
                collate_fn=prompt_dataset.get_collate_fn(),
                generator=train_generator,
            )

            sample_images = []
            observed_batch_size = 0
            for batch in train_dataloader:
                normal_images = [
                    image
                    for image, label in zip(batch["image"], batch["label"])
                    if label == AnomalyLabel.NORMAL
                ]
                if observed_batch_size == 0 and normal_images:
                    observed_batch_size = len(normal_images)
                sample_images.extend(normal_images)
                if len(sample_images) >= requested_sample_count:
                    sample_images = sample_images[:requested_sample_count]
                    break

            if observed_batch_size == 0:
                observed_batch_size = min(loader_batch_size, len(sample_images))

            return sample_images, observed_batch_size

        raise ValueError(f"augmentation analysis source must be 'test' or 'train', got {source!r}")


    def analyze_augmentation(
        self,
        batch_images: Union[torch.Tensor, List[Union[torch.Tensor, Image.Image, np.ndarray]]],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        rotation_angles: Tuple[int, ...] = (0, 45, 90, 135, 180, 225, 270, 315),
        rot_tolerance: float = 0.017,
        flip_tolerance: float = 0.003,
    ):
        """
        Analyze whether rotation and flip augmentations preserve feature similarity.
        
        Compares CLS token similarities between: different images, same image with different
        rotations, and same image with flips. Determines if augmentations maintain similar
        features compared to different test images (default using 1.5% tolerance).
        
        Args:
            batch_images: Input images to analyze
            resolution: Target preprocessing resolution. Defaults to None
            rotation_angles: Rotation angles to test in degrees. 
                Defaults to (0, 45, 90, 135, 180, 225, 270, 315)
            rot_tolerance: Tolerance for rotation similarity. Defaults to 0.015
            flip_tolerance: Tolerance for flip similarity. Defaults to 0.003
        
        Returns:
            Tuple of (can_rotate_all, can_rotate_90s, can_flip, similarity_scores) where:
            - can_rotate_all: Whether all rotations preserve similarity
            - can_rotate_90s: Whether 90° rotations preserve similarity  
            - can_flip: Whether flips preserve similarity
            - similarity_scores: (test_mean, rot_mean, rot_90s_mean, flip_mean)
        """
        cpu_device = torch.device("cpu")
        batch_images = self.preprocess(
            batch_images, 
            resolution=None,
            do_resize=False,
            do_normalize=False,
        )
        
        batch_size = batch_images.shape[0]

        preprocessed_batch_images = self.preprocess(
            batch_images,
            resolution=resolution,
            do_rescale=False,
        )
        
        batch_features_dict = self.get_features(
            preprocessed_batch_images,
            return_attentions=False,
        )

        # Rotation augmentations
        rot_aug_inputs = self.augment_images(
            batch_images,
            rotation_angles=rotation_angles,
            is_flip=False
        )
        rot_aug_inputs = self.preprocess(
            rot_aug_inputs,
            resolution=resolution,
            do_rescale=False,
        )
        num_rot_augs = 8  # 0, 45, 90, 135, 180, 225, 270, 315
 
        rot_aug_features_dict = self.get_features(
            rot_aug_inputs,
            return_attentions=False,
        )

        # Flip augmentations won't change image size, so we can process directly
        flip_aug_inputs = self.augment_images(
            preprocessed_batch_images,
            rotation_angles=(0,),
            is_flip=True
        )
        num_flip_augs = 2  # original + flipped

        flip_aug_features_dict = self.get_features(
            flip_aug_inputs,
            return_attentions=False,
        )

        # CLS cosine similarity for test images (different images)
        test_cls_sim = self._calculate_cosine_similarity(
            batch_features_dict.pooler_output,
            batch_features_dict.pooler_output
        ).to(cpu_device)
        # Ignore self-similarity
        test_mask = torch.eye(batch_size, dtype=torch.bool, device=test_cls_sim.device)
        test_sim_values = test_cls_sim[~test_mask].cpu().numpy()
        test_mean_sim = test_sim_values.mean()
        
        # Rotation augmentation similarity (same image, different rotations)
        rot_cls_sim = self._calculate_cosine_similarity(
            rot_aug_features_dict.pooler_output,
            rot_aug_features_dict.pooler_output
        ).to(cpu_device)
        # Create mask to get same-image augmentation pairs (excluding self)
        rot_same_img_mask = self._create_augmentation_group_mask(batch_size, num_rot_augs, rot_cls_sim.device)
        rot_sim_values = rot_cls_sim[~rot_same_img_mask].cpu().numpy()
        
        # Exclude image with non-90-degree augmentations
        for i, rotation_angle in enumerate(rotation_angles):
            if rotation_angle % 90 != 0:
                # Exclude entire row and column for non-90-degree angles
                rot_same_img_mask[batch_size*i: batch_size*(i+1), :] = True
                rot_same_img_mask[:, batch_size*i: batch_size*(i+1)] = True

        rot_sim_90s_values = rot_cls_sim[~rot_same_img_mask].cpu().numpy()
        rot_mean_sim = rot_sim_values.mean()
        rot_90s_mean_sim = rot_sim_90s_values.mean()
        can_rotate_all = rot_mean_sim * (1 + rot_tolerance) >= test_mean_sim
        can_rotate_90s = rot_90s_mean_sim * (1 + rot_tolerance) >= test_mean_sim

        # Flip augmentation similarity (same image, original vs flipped)
        flip_cls_sim = self._calculate_cosine_similarity(
            flip_aug_features_dict.pooler_output,
            flip_aug_features_dict.pooler_output
        ).to(cpu_device)
        # Create mask to get same-image augmentation pairs (excluding self)
        flip_same_img_mask = self._create_augmentation_group_mask(batch_size, num_flip_augs, flip_cls_sim.device)
        flip_sim_values = flip_cls_sim[~flip_same_img_mask].cpu().numpy()
        flip_mean_sim = flip_sim_values.mean()
        can_flip = flip_mean_sim * (1 + flip_tolerance) >= test_mean_sim

        if os.environ.get("AD_DEBUG_AUGMENTATION"):
            required_rot_threshold = test_mean_sim / rot_mean_sim if rot_mean_sim > 0 else float('inf')
            required_rot90s_threshold = test_mean_sim / rot_90s_mean_sim if rot_90s_mean_sim > 0 else float('inf')
            required_flip_threshold = test_mean_sim / flip_mean_sim if flip_mean_sim > 0 else float('inf')
            print(f"\n=== Augmentation Debug ===")
            print(f"rot_tolerance={rot_tolerance}  flip_tolerance={flip_tolerance}")
            print(f"test_mean_sim:    {test_mean_sim:.6f}")
            print(f"rot_mean_sim:     {rot_mean_sim:.6f}   -> scaled {rot_mean_sim*(1+rot_tolerance):.6f}  can_rotate={can_rotate_all}  (required_tol={required_rot_threshold-1:.4f})")
            print(f"rot_90s_mean_sim: {rot_90s_mean_sim:.6f}   -> scaled {rot_90s_mean_sim*(1+rot_tolerance):.6f}  can_rotate90s={can_rotate_90s}  (required_tol={required_rot90s_threshold-1:.4f})")
            print(f"flip_mean_sim:    {flip_mean_sim:.6f}   -> scaled {flip_mean_sim*(1+flip_tolerance):.6f}  can_flip={can_flip}  (required_tol={required_flip_threshold-1:.4f})")
            print(f"==========================\n")

        return can_rotate_all, can_rotate_90s, can_flip, (
            test_mean_sim, rot_mean_sim, rot_90s_mean_sim, flip_mean_sim
        )

    def _create_augmentation_group_mask(
        self,
        batch_size: int,
        num_augs: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create a boolean mask to exclude same-image augmentation pairs.
        
        The augmented images are arranged as:
        [img0, img1, ..., imgN, img0_aug1, img1_aug1, ..., imgN_aug1, img0_aug2, ...]
        
        For example with batch_size=3, num_augs=4:
        indices 0,3,6,9 are all from img0 (should be masked from each other)
        indices 1,4,7,10 are all from img1 (should be masked from each other)
        indices 2,5,8,11 are all from img2 (should be masked from each other)
        
        Args:
            batch_size: Number of original images
            num_augs: Number of augmentations per image
            device: Device for the mask tensor
            
        Returns:
            Boolean mask of shape [batch_size * num_augs, batch_size * num_augs]
            where True indicates pairs to exclude (same original image)
        """
        total_size = batch_size * num_augs
        
        # Create indices for each position: which original image it belongs to
        # Position j belongs to original image (j % batch_size)
        original_indices = torch.arange(total_size, device=device) % batch_size
        
        # Mask is True where two positions belong to the same original image
        mask = original_indices.unsqueeze(0) == original_indices.unsqueeze(1)
        
        return mask
    

    def _calculate_cosine_similarity(
        self,
        features1: torch.Tensor,
        features2: torch.Tensor,
        cosine_sim_clamp: Optional[tuple[float, float]] = None,
    ) -> torch.Tensor:
        """Calculate cosine similarity between two feature tensors.

        Args:
            features1 (torch.Tensor): The first feature tensor, expected channel in last dim.
            features2 (torch.Tensor): The second feature tensor, expected channel in last dim.
            cosine_sim_clamp (Optional[tuple[float, float]], optional): Clamping range for cosine similarity. Defaults to use self.cosine_sim_clamp.

        Returns:
            torch.Tensor: The cosine similarity between the two feature tensors in [0, 1].
        """        
        if cosine_sim_clamp is None:
            cosine_sim_clamp = self.cosine_sim_clamp

        features1 = F.normalize(features1, dim=-1)
        features2 = F.normalize(features2, dim=-1)
        cosine_similarity = torch.matmul(features1, features2.T).clamp(*cosine_sim_clamp)

        # Ensure cosine similarity is in [0, 1] range (from [-1, 1] to [0, 1])
        cosine_similarity = (cosine_similarity - cosine_sim_clamp[0]) / (cosine_sim_clamp[1] - cosine_sim_clamp[0])

        return cosine_similarity


    def _calculate_anomaly_score(
        self,
        anomaly_map: torch.Tensor,
        anomaly_score_method: str,
        return_attentioned_anomaly_map: bool,
        attentioned_anomaly_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attentioned_anomaly_score = None

        if anomaly_score_method == "max":
            anomaly_score = torch.amax(anomaly_map, dim=(1, 2))
            if return_attentioned_anomaly_map:
                attentioned_anomaly_score = torch.amax(attentioned_anomaly_map, dim=(1, 2))
        else:
            # Calculate top 1% values instead of maximum
            flat_anomaly_map = anomaly_map.reshape(anomaly_map.size(0), -1)  # [BS, H*W]
            top_1_percent_count = max(1, int(0.01 * flat_anomaly_map.size(1)))
            anomaly_score = torch.topk(flat_anomaly_map, top_1_percent_count, dim=1)[0].mean(dim=1)
        
            if return_attentioned_anomaly_map:
                flat_attentioned_anomaly_map = attentioned_anomaly_map.reshape(attentioned_anomaly_map.size(0), -1)
                attentioned_anomaly_score = torch.topk(flat_attentioned_anomaly_map, top_1_percent_count, dim=1)[0].mean(dim=1)

        return anomaly_score, attentioned_anomaly_score

    def __call__(
        self,
        prompt_images: Union[torch.Tensor, Image.Image, np.ndarray, list],
        test_images: Optional[Union[torch.Tensor, Image.Image, np.ndarray, list]],
        is_prompt_features: bool = False,
        attention_unroi_weight: Optional[float] = None,
        attention_roi_threshold: Optional[float] = None,
        attention_interest_token_idx: Optional[Union[int, Tuple[int, int]]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        cosine_sim_clamp: Optional[tuple[float, float]] = None,
        anomaly_score_method: str = "top1", # "max" or "top1"
        upsample_anomaly_map: bool = False,
        upsample_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        return_attentioned_anomaly_map: bool = True,
        **kwargs
    ) -> PatchEADOutput:
        """
        Perform anomaly detection by comparing test images against prompt (normal) images.
        
        Args:
            prompt_images: Normal/reference images or pre-extracted features [N_patches, C] if is_prompt_features=True
            test_images: Images to be tested for anomalies
            is_prompt_features (bool): Whether prompt_images are already extracted features. Default: False
            attention_unroi_weight (float): Weight for non-ROI regions in attention map. 
            Default: self.attention_unroi_weight
            attention_roi_threshold (float): Threshold for attention weight ROI determination. 
            Default: self.attention_roi_threshold
            attention_interest_token_idx (int|Tuple[int,int]): Token index(es) for attention weight computation. 
            Default: None, will be determined by the function.  
            resolution (int|Tuple[int,int]): Pipeline resolution; images will be resized to this resolution 
                if specified. Default: None
            cosine_sim_clamp (Tuple[float,float]): Clamp range for cosine similarity. Default: self.cosine_sim_clamp
            anomaly_score_method (str): Method for score computation ("max" or "top1"). Default: "top1"
            upsample_anomaly_map (bool): Whether to upsample anomaly map to image size. Default: False
            upsample_resolution (int|Tuple[int,int]]): Target resolution for upsampling anomaly map.
                If None, uses original image size. Default: None
            return_attentioned_anomaly_map (bool): Whether to compute attention-weighted maps. Default: True
            
        Returns:
            PatchEADOutput: Contains anomaly_map, anomaly_score, attentioned_anomaly_map, 
              attentioned_anomaly_score, and attention_map
        """
        # Inputs check
        resolution = self.ensure_tuple_resolution(resolution or self.resolution)
        upsample_resolution = self.ensure_tuple_resolution(
            upsample_resolution or self.get_image_resolution(test_images)
        )

        anomaly_score_method = anomaly_score_method.lower()
        if anomaly_score_method not in ["max", "top1"]:
            raise ValueError("anomaly_score_method must be 'max' or 'top1'")

        if cosine_sim_clamp is None:
            cosine_sim_clamp = self.cosine_sim_clamp

        extra_feature_kwargs = {}
        if "output_feature_maps_indices" in kwargs:
            extra_feature_kwargs["output_feature_maps_indices"] = kwargs["output_feature_maps_indices"]

        if not is_prompt_features:
            prompt_features = self.get_prompt_features(
                prompt_images,
                resolution=resolution,
                **extra_feature_kwargs
            )
        else:
            prompt_features = prompt_images  # Assume already in [N_patches, C] format

        test_inputs = self.preprocess(
            test_images,
            resolution=resolution,
        )
        
        test_features_dict = self.get_features(
            test_inputs,
            return_attentions=return_attentioned_anomaly_map,
            **extra_feature_kwargs
        )
        test_features = test_features_dict.feature_maps[-1]  # Shape: [BS, C, H, W]

        # Reshape test features to [BS, H*W, C] for similarity computation
        bs, c, h, w = test_features.shape
        test_features_reshaped = test_features.permute(0, 2, 3, 1).reshape(bs, h*w, c)  # [BS, H*W, C]

        cosine_similarity = self._calculate_cosine_similarity(
            test_features_reshaped, prompt_features, cosine_sim_clamp
        ) # [BS, H*W, N_patches]
        
        # Convert similarity to anomaly score
        anomaly = 1 - cosine_similarity  # [BS, H*W, N_patches]

        # Take the max similarity for each test patch against all prompt patches
        anomaly_map, _ = torch.min(anomaly, dim=-1)  # [BS, H*W]

        # Reshape back to spatial dimensions
        anomaly_map = anomaly_map.reshape(bs, h, w)  # [BS, H, W]

        # Compute attention score
        saliency = None
        attentioned_anomaly_map = None

        if return_attentioned_anomaly_map:
            extra_attn_map_kwargs = {}
            if attention_roi_threshold is not None:
                extra_attn_map_kwargs["attention_roi_threshold"] = attention_roi_threshold
            if attention_unroi_weight is not None:
                extra_attn_map_kwargs["attention_unroi_weight"] = attention_unroi_weight
            if attention_interest_token_idx is not None:
                extra_attn_map_kwargs["interest_token_idx"] = attention_interest_token_idx
            
            saliency = self._get_salient_map(
                (bs, h, w),
                attentions=test_features_dict.attentions, 
                **extra_attn_map_kwargs
            )
            attentioned_anomaly_map = anomaly_map * saliency

        # Calculate anomaly scores
        anomaly_score, attentioned_anomaly_score = self._calculate_anomaly_score(
            anomaly_map,
            anomaly_score_method,
            return_attentioned_anomaly_map,
            attentioned_anomaly_map
        )

        if upsample_anomaly_map:
            anomaly_map = self.upsample_anomaly_map(anomaly_map, upsample_resolution)
            if return_attentioned_anomaly_map:
                attentioned_anomaly_map = self.upsample_anomaly_map(attentioned_anomaly_map, upsample_resolution)

        return PatchEADOutput(
            anomaly_map=anomaly_map.detach().cpu(),
            anomaly_score=anomaly_score.detach().cpu(),
            attentioned_anomaly_map=attentioned_anomaly_map.detach().cpu() if return_attentioned_anomaly_map else None,
            attentioned_anomaly_score=attentioned_anomaly_score.detach().cpu() if return_attentioned_anomaly_map else None,
            attention_map=saliency.detach().cpu() if return_attentioned_anomaly_map else None
        )
    
    def zero_shot(
        self, 
        test_dataloader: DataLoader, 
        seed: int = 42,
        # Call function paramters
        attention_unroi_weight: Optional[float] = None,
        attention_roi_threshold: Optional[float] = None,
        attention_interest_token_idx: Optional[Union[int, Tuple[int, int]]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        cosine_sim_clamp: Optional[tuple[float, float]] = None,
        return_attentioned_anomaly_map: bool = True,
        progress_desc: Optional[str] = None,
        save_results_path: Optional[PathLike] = None,
        anomaly_score_method: str = "top1", # "max" or "top1"
        **kwargs
    ) -> ADPerClassEvaluationOutput:
        pass
    
    def few_shot(
        self, 
        shots: int, 
        test_dataloader: DataLoader, 
        prompt_dataset: BaseAnomalyClassDataset,
        seed: int = 42,
        # Call function parameters
        attention_unroi_weight: Optional[float] = None,
        attention_roi_threshold: Optional[float] = None,
        attention_interest_token_idx: Optional[Union[int, Tuple[int, int]]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        eval_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        cosine_sim_clamp: Optional[tuple[float, float]] = None,
        return_attentioned_anomaly_map: bool = True,
        progress_desc: Optional[str] = None,
        save_results_path: Optional[PathLike] = None,
        save_with_normed_heatmap: bool = False,
        anomaly_score_method: str = "top1", # "max" or "top1"
        **kwargs
    ) -> ADPerClassEvaluationOutput:
        """Few-shot anomaly detection.

        Args:
            eval_resolution: Resolution to which anomaly maps are upsampled before
                pixel-level metrics (AUROC, AUPRO, F1Max) are computed.  Defaults to
                ``resolution`` (i.e. the inference resolution).  Setting this lower than
                the native image size is the primary lever for reducing the memory cost
                of pixel metrics on high-resolution datasets such as RealIAD.
        """
        eval_device = self.eval_device
        eval_dtype = self.eval_dtype
        
        self.model.eval()
        with torch.no_grad():
            aug_kwargs = {}

            augmentation_mode = self._resolve_augmentation_mode(
                kwargs.get("augmentation_mode"),
                kwargs.get("augmentation", False),
            )

            if augmentation_mode == "auto":
                augmentation_analysis_split = kwargs.get("augmentation_analysis_split", "test")
                if augmentation_analysis_split not in {"test", "train"}:
                    raise ValueError(
                        "augmentation_analysis_split must be 'test' or 'train', "
                        f"got {augmentation_analysis_split!r}"
                    )

                sample_images, _ = self._collect_augmentation_samples(
                    test_dataloader=test_dataloader,
                    prompt_dataset=prompt_dataset,
                    seed=seed,
                    source=augmentation_analysis_split,
                )
                if not sample_images:
                    raise ValueError(
                        f"No samples available for augmentation analysis from {augmentation_analysis_split!r} split."
                    )
                
                extracted_kwargs = {k: kwargs[k] for k in ("flip_tolerance", "rot_tolerance") if k in kwargs}
                analysis_rotation_angles = ALL_ROTATION_ANGLES

                can_rotate, can_rotate90s, can_flip, similarity_scores = self.analyze_augmentation(
                    sample_images, 
                    resolution=resolution, 
                    rotation_angles=analysis_rotation_angles,
                    **extracted_kwargs
                )
                aug_kwargs = self._build_auto_prompt_augmentation_kwargs(
                    can_rotate=can_rotate,
                    can_rotate90s=can_rotate90s,
                    can_flip=can_flip,
                    requested_angles=kwargs.get("rotation_angles"),
                )
                progress_desc = self._append_aug_progress(progress_desc, aug_kwargs)

                class_name = getattr(prompt_dataset, "category", "unknown")
                self._augmentation_analysis_by_class[class_name] = {
                    "mode": "few_shot",
                    **self._build_augmentation_analysis_record(
                        sample_count=len(sample_images),
                        can_rotate=can_rotate,
                        can_rotate90s=can_rotate90s,
                        can_flip=can_flip,
                        similarity_scores=similarity_scores,
                        rotation_angles=analysis_rotation_angles,
                        rot_tolerance=extracted_kwargs.get("rot_tolerance", 0.015),
                        flip_tolerance=extracted_kwargs.get("flip_tolerance", 0.003),
                        selected_prompt_augmentation=self._serialize_selected_prompt_augmentation(
                            augmentation_mode,
                            aug_kwargs,
                        ),
                        analysis_split=augmentation_analysis_split,
                    ),
                }

                torch.cuda.empty_cache()
            elif augmentation_mode != "none":
                aug_kwargs = self._build_prompt_augmentation_kwargs(
                    augmentation_mode,
                    requested_angles=kwargs.get("rotation_angles"),
                )
                progress_desc = self._append_aug_progress(progress_desc, aug_kwargs)

            if kwargs.get("output_feature_maps_indices", None):
                aug_kwargs["output_feature_maps_indices"] = kwargs.get("output_feature_maps_indices", None)
                if -1 not in aug_kwargs["output_feature_maps_indices"] and \
                    self.model.config.num_layers not in aug_kwargs["output_feature_maps_indices"]:
                    aug_kwargs["output_feature_maps_indices"] = aug_kwargs["output_feature_maps_indices"] + (-1,)
            if kwargs.get("layer_fusion_method", None) is not None:
                aug_kwargs["layer_fusion_method"] = kwargs["layer_fusion_method"]
            
            prompt_features = self.get_prompt_features(
                prompt_images=prompt_dataset.get_normal_samples(sample_count=shots, seed=seed)["image"],
                resolution=resolution,
                **aug_kwargs
            )


            progress = tqdm(test_dataloader, desc=progress_desc)
            for batch in progress:
                anomaly_dict = self(
                    prompt_images=prompt_features,
                    test_images=batch["image"],
                    is_prompt_features=True,
                    attention_unroi_weight=attention_unroi_weight,
                    attention_roi_threshold=attention_roi_threshold,
                    attention_interest_token_idx=attention_interest_token_idx,
                    resolution=resolution,
                    cosine_sim_clamp=cosine_sim_clamp,
                    anomaly_score_method=anomaly_score_method,
                    upsample_anomaly_map=True,
                    upsample_resolution=eval_resolution or resolution,
                    return_attentioned_anomaly_map=return_attentioned_anomaly_map,
                    **kwargs
                )

                labels = torch.tensor([l.value for l in batch["label"]]).to(device=eval_device, dtype=torch.int8)
                if return_attentioned_anomaly_map:
                    scores = anomaly_dict["attentioned_anomaly_score"].to(device=eval_device, dtype=eval_dtype)
                    self._update_eval_metrics(preds=scores, labels=labels, domain="image")
                else:
                    scores = anomaly_dict["anomaly_score"].to(device=eval_device, dtype=eval_dtype)
                    self._update_eval_metrics(preds=scores, labels=labels, domain="image")

                if os.environ.get("AD_DEBUG_SCORES"):
                    normal_mask = labels == 0
                    anomaly_mask = labels == 1
                    if normal_mask.any() and anomaly_mask.any():
                        print(f"  scores: normal_mean={scores[normal_mask].float().mean():.4f}  "
                              f"anomaly_mean={scores[anomaly_mask].float().mean():.4f}  "
                              f"[higher=more anomalous, expect anomaly_mean > normal_mean]")

                height, width = anomaly_dict["anomaly_map"].shape[-2:]

                # Resize masks to match anomaly map size
                # Use NEAREST to avoid interpolation artifacts
                masks = torch.stack([
                    to_tensor(mask.resize((width, height), resample=Image.NEAREST)) for mask in batch["mask"]
                ]).to(device=eval_device, dtype=eval_dtype)
                if return_attentioned_anomaly_map:
                    anomaly_maps = anomaly_dict["attentioned_anomaly_map"].to(device=eval_device, dtype=eval_dtype)
                    attn_maps = anomaly_dict["attention_map"]
                else:
                    anomaly_maps = anomaly_dict["anomaly_map"].to(device=eval_device, dtype=eval_dtype)
                    attn_maps = [None] * anomaly_maps.shape[0]
                
                self._update_eval_metrics(preds=anomaly_maps, labels=masks, domain="pixel")


                if save_results_path is not None:
                    save_results_path = Path(save_results_path)

                    file_paths = batch["image_path"]
                    defect_types = batch["defect_type"]

                    for image, mask, anomaly_map, attn_map, ori_path, defect_type in zip(
                        batch["image"], masks, anomaly_maps, attn_maps, file_paths, defect_types):
                        save_dir = save_results_path / defect_type
                        save_dir.mkdir(parents=True, exist_ok=True)
                        height, width = anomaly_map.shape[-2:]
                        if image.size[1] != height or image.size[0] != width:
                            image = image.resize((width, height))
                        if mask.shape[-2] != height or mask.shape[-1] != width:
                            mask = mask.resize((width, height))
                        if save_with_normed_heatmap:
                            anomaly_map = anomaly_map / (torch.amax(anomaly_map, dim=(-1, -2), keepdim=True) + 1e-8)
                        if attn_map is None:
                            concat_images(
                                [image, to_pil(mask), get_heatmap(to_pil(anomaly_map))]
                            ).save(save_dir / f"{Path(ori_path).stem}.jpg")
                        else:
                            concat_images([
                                image, to_pil(mask), get_heatmap(to_pil(anomaly_map)), 
                                get_heatmap(to_pil(attn_map - attn_map.amin(dim=[-2, -1], keepdim=True)).resize((width, height)))
                            ]).save(save_dir / f"{Path(ori_path).stem}.jpg")

            results = self._compute_eval_metrics()

        return ADPerClassEvaluationOutput(
            AUROC=results["image"].get("AUROC"),
            AUPR=results["image"].get("AUPR"),
            AP=results["image"].get("AP"),
            F1Max=results["image"].get("F1Max"),
            pixel_AUROC=results["pixel"].get("AUROC"),
            pixel_AUPRO=results["pixel"].get("AUPRO"),
            pixel_F1Max=results["pixel"].get("F1Max"),
        )

    # ------------------------------------------------------------------
    # Multi-view few-shot evaluation
    # ------------------------------------------------------------------

    def _build_per_view_memory_bank(
        self,
        prompt_dataset: BaseAnomalyClassDataset,
        shots: int,
        seed: int = 42,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        view_aug_configs: Optional[Dict[str, Dict]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Build a separate memory bank (prompt features) for each camera view.

        Args:
            prompt_dataset: Normal-image dataset that exposes ``view_id`` per sample.
            shots: Number of normal samples to draw **per view**.
            seed: Random seed for reproducible sampling.
            resolution: Passed through to ``get_prompt_features``.
            view_aug_configs: Optional per-view augmentation configurations.  When
                provided, each view's entry (e.g. ``{"augmentation": True,
                "rotation_angles": (...), "is_flip": True}``) is merged into the
                kwargs before calling ``get_prompt_features`` for that view.  Views
                absent from the dict receive no augmentation overrides.
            **kwargs: Additional keyword arguments forwarded to ``get_prompt_features``
                (e.g. ``output_feature_maps_indices``).

        Returns:
            Dict mapping ``view_id`` (str) → prompt-feature tensor.

        Raises:
            ValueError: If the dataset samples do not carry a ``view_id`` field.
        """
        # Group normal indices by view_id
        view_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx in prompt_dataset._normal_indices:
            item = prompt_dataset[idx]
            if "view_id" not in item:
                raise ValueError(
                    "Dataset does not provide 'view_id' in its samples. "
                    "Multi-view evaluation requires a multi-view dataset such as RealIADDataset."
                )
            view_to_indices[item["view_id"]].append(idx)

        rng = prompt_dataset._rng(seed)
        memory_banks: Dict[str, torch.Tensor] = {}

        for view_id, indices in view_to_indices.items():
            if len(indices) < shots:
                print(f"Warning: view {view_id} has only {len(indices)} normal samples; requested {shots}.")
                selected = indices
            else:
                selected = rng.sample(indices, shots)

            view_images = [prompt_dataset[idx]["image"] for idx in selected]

            # Merge per-view augmentation overrides (if any) into the kwargs for this view
            view_kwargs = kwargs.copy()
            if view_aug_configs and view_id in view_aug_configs:
                view_kwargs.update(view_aug_configs[view_id])

            memory_banks[view_id] = self.get_prompt_features(
                prompt_images=view_images,
                resolution=resolution,
                **view_kwargs,
            )

        return memory_banks

    def multi_view_few_shot(
        self,
        shots: int,
        test_dataloader: DataLoader,
        prompt_dataset: BaseAnomalyClassDataset,
        seed: int = 42,
        evaluation_mode: str = "view_as_image",
        view_aggregation: str = "max",
        attention_unroi_weight: Optional[float] = None,
        attention_roi_threshold: Optional[float] = None,
        attention_interest_token_idx: Optional[Union[int, Tuple[int, int]]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        eval_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        cosine_sim_clamp: Optional[tuple] = None,
        return_attentioned_anomaly_map: bool = True,
        progress_desc: Optional[str] = None,
        save_results_path: Optional[PathLike] = None,
        save_with_normed_heatmap: bool = False,
        anomaly_score_method: str = "top1",
        **kwargs,
    ) -> ADPerClassEvaluationOutput:
        """Multi-view few-shot anomaly detection.

        Builds a per-view memory bank from ``prompt_dataset``, then evaluates
        the test set in one of two modes:

        * ``"view_as_image"`` – every view of a sample is scored independently;
          metrics are accumulated over individual views.
        * ``"views_as_sample"`` – all views of the same physical object are scored
          together and their anomaly maps / scores are aggregated (``max`` or
          ``mean``) before updating metrics, giving one result per object.

        The key efficiency improvement over a naive loop is that same-view images
        are batched together and fed to ``self()`` as a single batch, fully
        utilising GPU parallelism.

        Args:
            shots: Normal shots **per view** for memory bank construction.
            test_dataloader: DataLoader whose batches contain ``"view_id"`` and,
                optionally, ``"sample_id"`` fields.
            prompt_dataset: Dataset providing normal samples with ``"view_id"``.
            seed: Random seed.
            evaluation_mode: ``"view_as_image"`` or ``"views_as_sample"``.
            view_aggregation: ``"max"`` or ``"mean"`` (used in
                ``"views_as_sample"`` mode only).
            attention_unroi_weight: Passed to ``self()``.
            attention_roi_threshold: Passed to ``self()``.
            attention_interest_token_idx: Passed to ``self()``.
            resolution: Target spatial resolution for inference.
            eval_resolution: Resolution to which anomaly maps are upsampled before
                pixel-level metrics are computed.  Defaults to ``resolution``.
                Set this lower than the native image size to reduce the memory cost
                of AUPRO/AUROC accumulation on high-resolution datasets (e.g. RealIAD).
            cosine_sim_clamp: Passed to ``self()``.
            return_attentioned_anomaly_map: Use attention-weighted maps for metrics.
            progress_desc: tqdm bar description.
            save_results_path: Directory to save visualisation images.
            save_with_normed_heatmap: Normalise heatmaps before saving.
            anomaly_score_method: ``"max"`` or ``"top1"``.
            **kwargs: Forwarded to ``get_prompt_features`` and ``self()``.

        Returns:
            :class:`ADPerClassEvaluationOutput` with image- and pixel-level metrics.
        """
        if evaluation_mode not in ("view_as_image", "views_as_sample"):
            raise ValueError(
                f"evaluation_mode must be 'view_as_image' or 'views_as_sample', got {evaluation_mode!r}"
            )
        if view_aggregation not in ("max", "mean"):
            raise ValueError(
                f"view_aggregation must be 'max' or 'mean', got {view_aggregation!r}"
            )

        self.model.eval()
        with torch.no_grad():
            # Build kwargs that are forwarded to get_prompt_features (no aug flags yet)
            prompt_kwargs: Dict[str, Any] = {}

            if kwargs.get("output_feature_maps_indices", None):
                prompt_kwargs["output_feature_maps_indices"] = kwargs["output_feature_maps_indices"]
                if (
                    -1 not in prompt_kwargs["output_feature_maps_indices"]
                    and self.model.config.num_layers not in prompt_kwargs["output_feature_maps_indices"]
                ):
                    prompt_kwargs["output_feature_maps_indices"] = (
                        prompt_kwargs["output_feature_maps_indices"] + (-1,)
                    )

            # --- Per-view augmentation analysis ----------------------------------
            # Augmentation symmetry can differ per camera angle, so we analyse each
            # view's selected split independently and build a per-view aug config.
            view_aug_configs: Dict[str, Dict] = {}
            augmentation_mode = self._resolve_augmentation_mode(
                kwargs.get("augmentation_mode"),
                kwargs.get("augmentation", False),
            )
            if augmentation_mode == "auto":
                augmentation_analysis_split = kwargs.get("augmentation_analysis_split", "test")
                if augmentation_analysis_split not in {"test", "train"}:
                    raise ValueError(
                        "augmentation_analysis_split must be 'test' or 'train', "
                        f"got {augmentation_analysis_split!r}"
                    )

                # Collect sample images keyed by view_id
                view_sample_images: Dict[str, List] = defaultdict(list)
                view_analysis: Dict[str, Any] = {}
                if augmentation_analysis_split == "test":
                    for batch in test_dataloader:
                        for img, vid in zip(batch["image"], batch["view_id"]):
                            if len(view_sample_images[vid]) < self.analyze_augment_sample_count:
                                view_sample_images[vid].append(img)
                        if all(
                            len(v) >= self.analyze_augment_sample_count
                            for v in view_sample_images.values()
                        ):
                            break

                    if getattr(test_dataloader, "generator", None) is not None:
                        test_dataloader.generator.manual_seed(seed)
                else:
                    normal_sample_count = min(
                        self.analyze_augment_sample_count,
                        len(getattr(prompt_dataset, "_normal_indices", ())),
                    )
                    if normal_sample_count == 0:
                        raise ValueError(
                            "No normal training samples available for per-view augmentation analysis."
                        )

                    loader_batch_size = normal_sample_count
                    if test_dataloader.batch_size is not None:
                        loader_batch_size = min(test_dataloader.batch_size, normal_sample_count)

                    train_generator = torch.Generator()
                    train_generator.manual_seed(seed)
                    train_dataloader = DataLoader(
                        prompt_dataset,
                        batch_size=max(1, loader_batch_size),
                        shuffle=True,
                        collate_fn=prompt_dataset.get_collate_fn(),
                        generator=train_generator,
                    )
                    for batch in train_dataloader:
                        normal_pairs = [
                            (img, vid)
                            for img, vid, label in zip(batch["image"], batch["view_id"], batch["label"])
                            if label == AnomalyLabel.NORMAL
                        ]
                        for img, vid in normal_pairs:
                            if len(view_sample_images[vid]) < self.analyze_augment_sample_count:
                                view_sample_images[vid].append(img)
                        if all(
                            len(v) >= self.analyze_augment_sample_count
                            for v in view_sample_images.values()
                        ):
                            break
                print("Analyzing augmentation per view…")
                aug_progress_parts: List[str] = []
                analysis_rotation_angles = ALL_ROTATION_ANGLES
                for vid, sample_images in view_sample_images.items():
                    can_rotate, can_rotate90s, can_flip, similarity_scores = self.analyze_augmentation(
                        sample_images,
                        resolution=resolution,
                        rotation_angles=analysis_rotation_angles,
                    )
                    view_aug_config = self._build_auto_prompt_augmentation_kwargs(
                        can_rotate=can_rotate,
                        can_rotate90s=can_rotate90s,
                        can_flip=can_flip,
                        requested_angles=kwargs.get("rotation_angles"),
                    )
                    status_parts: List[str] = [f"View {vid}:"]
                    if "rotation_angles" in view_aug_config and any(
                        angle % 90 != 0 for angle in view_aug_config["rotation_angles"]
                    ):
                        status_parts.append("Rot✅")
                    elif "rotation_angles" in view_aug_config:
                        status_parts.append("Rot90✅")
                    if view_aug_config.get("is_flip"):
                        status_parts.append("Flip✅")
                    view_aug_configs[vid] = view_aug_config
                    view_analysis[vid] = self._build_augmentation_analysis_record(
                        sample_count=len(sample_images),
                        can_rotate=can_rotate,
                        can_rotate90s=can_rotate90s,
                        can_flip=can_flip,
                        similarity_scores=similarity_scores,
                        rotation_angles=analysis_rotation_angles,
                        rot_tolerance=0.015,
                        flip_tolerance=0.003,
                        selected_prompt_augmentation=self._serialize_selected_prompt_augmentation(
                            augmentation_mode,
                            view_aug_config,
                        ),
                        analysis_split=augmentation_analysis_split,
                    )
                    print(" ".join(status_parts))
                    if len(status_parts) > 1:
                        aug_progress_parts.append(vid)

                class_name = getattr(prompt_dataset, "category", "unknown")
                self._augmentation_analysis_by_class[class_name] = {
                    "mode": "multi_view_few_shot",
                    "views": view_analysis,
                }

                if progress_desc and aug_progress_parts:
                    progress_desc = progress_desc[:-1] + f"|Aug:{','.join(aug_progress_parts)}]"

                torch.cuda.empty_cache()
            elif augmentation_mode != "none":
                prompt_kwargs.update(
                    self._build_prompt_augmentation_kwargs(
                        augmentation_mode,
                        requested_angles=kwargs.get("rotation_angles"),
                    )
                )
                progress_desc = self._append_aug_progress(progress_desc, prompt_kwargs)

            print(f"Building per-view memory banks ({shots} shots/view)…")
            memory_banks = self._build_per_view_memory_bank(
                prompt_dataset=prompt_dataset,
                shots=shots,
                seed=seed,
                resolution=resolution,
                view_aug_configs=view_aug_configs if view_aug_configs else None,
                **prompt_kwargs,
            )
            print(f"Memory banks built for views: {sorted(memory_banks)}")

            # Shared call kwargs (no augmentation flags — those only apply to prompt)
            excluded = {"augmentation", "augmentation_mode", "rotation_angles", "is_flip"}
            call_kwargs = {k: v for k, v in kwargs.items() if k not in excluded}

            if evaluation_mode == "view_as_image":
                return self._mv_evaluate_view_as_image(
                    test_dataloader=test_dataloader,
                    memory_banks=memory_banks,
                    attention_unroi_weight=attention_unroi_weight,
                    attention_roi_threshold=attention_roi_threshold,
                    attention_interest_token_idx=attention_interest_token_idx,
                    resolution=resolution,
                    eval_resolution=eval_resolution,
                    cosine_sim_clamp=cosine_sim_clamp,
                    return_attentioned_anomaly_map=return_attentioned_anomaly_map,
                    anomaly_score_method=anomaly_score_method,
                    progress_desc=progress_desc,
                    save_results_path=save_results_path,
                    save_with_normed_heatmap=save_with_normed_heatmap,
                    **call_kwargs,
                )
            else:
                return self._mv_evaluate_views_as_sample(
                    test_dataloader=test_dataloader,
                    memory_banks=memory_banks,
                    view_aggregation=view_aggregation,
                    attention_unroi_weight=attention_unroi_weight,
                    attention_roi_threshold=attention_roi_threshold,
                    attention_interest_token_idx=attention_interest_token_idx,
                    resolution=resolution,
                    eval_resolution=eval_resolution,
                    cosine_sim_clamp=cosine_sim_clamp,
                    return_attentioned_anomaly_map=return_attentioned_anomaly_map,
                    anomaly_score_method=anomaly_score_method,
                    progress_desc=progress_desc,
                    save_results_path=save_results_path,
                    save_with_normed_heatmap=save_with_normed_heatmap,
                    **call_kwargs,
                )

    def _mv_evaluate_view_as_image(
        self,
        test_dataloader: DataLoader,
        memory_banks: Dict[str, torch.Tensor],
        attention_unroi_weight: Optional[float] = None,
        attention_roi_threshold: Optional[float] = None,
        attention_interest_token_idx: Optional[Union[int, Tuple[int, int]]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        eval_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        cosine_sim_clamp: Optional[tuple] = None,
        return_attentioned_anomaly_map: bool = True,
        anomaly_score_method: str = "top1",
        progress_desc: Optional[str] = None,
        save_results_path: Optional[PathLike] = None,
        save_with_normed_heatmap: bool = False,
        **kwargs,
    ) -> ADPerClassEvaluationOutput:
        """Evaluate in view-as-image mode.

        Each view is treated as an independent image matched against its
        view-specific memory bank. Images in the same batch that share a view
        are processed together to maximise GPU utilisation.
        """
        eval_device = self.eval_device
        eval_dtype = self.eval_dtype
        fallback_view = next(iter(memory_banks))

        self._reset_eval_metrics()

        pbar = tqdm(test_dataloader, desc=progress_desc or "Multi-view eval [view_as_image]")

        for batch in pbar:
            images = batch["image"]        # List[PIL.Image]
            labels = batch["label"]
            masks = batch["mask"]
            image_paths = batch["image_path"]
            defect_types = batch.get("defect_type", ["unknown"] * len(images))
            view_ids = batch["view_id"]    # List[str]

            # Group batch indices by view so we can issue one __call__ per view
            view_to_batch_indices: Dict[str, List[int]] = defaultdict(list)
            for i, vid in enumerate(view_ids):
                view_to_batch_indices[vid].append(i)

            # Accumulators indexed by original batch position
            anomaly_maps_out = [None] * len(images)
            anomaly_scores_out = [None] * len(images)
            attentioned_anomaly_maps_out = [None] * len(images) if return_attentioned_anomaly_map else None
            attentioned_anomaly_scores_out = [None] * len(images) if return_attentioned_anomaly_map else None
            attn_maps_out = [None] * len(images)

            for vid, idxs in view_to_batch_indices.items():
                prompt_features = memory_banks.get(vid, memory_banks[fallback_view])
                sub_images = [images[i] for i in idxs]

                output = self(
                    prompt_images=prompt_features,
                    test_images=sub_images,
                    is_prompt_features=True,
                    attention_unroi_weight=attention_unroi_weight,
                    attention_roi_threshold=attention_roi_threshold,
                    attention_interest_token_idx=attention_interest_token_idx,
                    resolution=resolution,
                    cosine_sim_clamp=cosine_sim_clamp,
                    anomaly_score_method=anomaly_score_method,
                    upsample_anomaly_map=True,
                    upsample_resolution=eval_resolution or resolution,
                    return_attentioned_anomaly_map=return_attentioned_anomaly_map,
                    **kwargs,
                )

                for local_i, batch_i in enumerate(idxs):
                    anomaly_maps_out[batch_i] = output.anomaly_map[local_i : local_i + 1]
                    anomaly_scores_out[batch_i] = output.anomaly_score[local_i : local_i + 1]
                    if return_attentioned_anomaly_map:
                        attentioned_anomaly_maps_out[batch_i] = output.attentioned_anomaly_map[local_i : local_i + 1]
                        attentioned_anomaly_scores_out[batch_i] = output.attentioned_anomaly_score[local_i : local_i + 1]
                    attn_maps_out[batch_i] = (
                        output.attention_map[local_i : local_i + 1]
                        if output.attention_map is not None
                        else None
                    )

            anomaly_maps = torch.cat(anomaly_maps_out, dim=0)
            anomaly_scores = torch.cat(anomaly_scores_out, dim=0)
            attentioned_anomaly_maps = (
                torch.cat(attentioned_anomaly_maps_out, dim=0) if return_attentioned_anomaly_map else None
            )
            attentioned_anomaly_scores = (
                torch.cat(attentioned_anomaly_scores_out, dim=0) if return_attentioned_anomaly_map else None
            )

            # Choose which outputs drive the metrics
            eval_scores = attentioned_anomaly_scores if return_attentioned_anomaly_map else anomaly_scores
            eval_maps = attentioned_anomaly_maps if return_attentioned_anomaly_map else anomaly_maps
            attn_maps = attn_maps_out  # keep as list for optional saving

            labels_tensor = torch.tensor(
                [l.value for l in labels], device=eval_device, dtype=torch.int8
            )
            self._update_eval_metrics(
                preds=eval_scores.to(device=eval_device, dtype=eval_dtype),
                labels=labels_tensor,
                domain="image",
            )

            height, width = anomaly_maps.shape[-2:]
            masks_tensor = torch.stack([
                to_tensor(m.resize((width, height), resample=Image.NEAREST)) for m in masks
            ]).to(device=eval_device, dtype=eval_dtype)
            self._update_eval_metrics(
                preds=eval_maps.to(device=eval_device, dtype=eval_dtype),
                labels=masks_tensor,
                domain="pixel",
            )

            if save_results_path is not None:
                save_results_path = Path(save_results_path)
                for image, mask, anomaly_map, attn_map, ori_path, defect_type in zip(
                    images, masks, eval_maps, attn_maps, image_paths, defect_types
                ):
                    save_dir = save_results_path / defect_type
                    save_dir.mkdir(parents=True, exist_ok=True)
                    h, w = anomaly_map.shape[-2:]
                    if image.size[1] != h or image.size[0] != w:
                        image = image.resize((w, h))
                    if mask.size[1] != h or mask.size[0] != w:
                        mask = mask.resize((w, h))
                    if save_with_normed_heatmap:
                        anomaly_map = anomaly_map / (torch.amax(anomaly_map, dim=(-1, -2), keepdim=True) + 1e-8)
                    if attn_map is None:
                        concat_images([image, to_pil(mask), get_heatmap(to_pil(anomaly_map))]).save(
                            save_dir / f"{Path(ori_path).stem}.jpg"
                        )
                    else:
                        concat_images([
                            image, to_pil(mask), get_heatmap(to_pil(anomaly_map)),
                            get_heatmap(to_pil(attn_map - attn_map.amin(dim=[-2, -1], keepdim=True)).resize((w, h))),
                        ]).save(save_dir / f"{Path(ori_path).stem}.jpg")

        results = self._compute_eval_metrics()
        return ADPerClassEvaluationOutput(
            AUROC=results["image"].get("AUROC"),
            AUPR=results["image"].get("AUPR"),
            AP=results["image"].get("AP"),
            F1Max=results["image"].get("F1Max"),
            pixel_AUROC=results["pixel"].get("AUROC"),
            pixel_AUPRO=results["pixel"].get("AUPRO"),
            pixel_F1Max=results["pixel"].get("F1Max"),
        )

    def _mv_group_by_sample(self, dataloader: DataLoader) -> Dict[str, List[Dict]]:
        """Consume a dataloader and group every item by its ``sample_id``.

        If the batch does not contain ``"sample_id"``, it is inferred by
        stripping the trailing ``_C{digits}`` camera suffix from the filename.

        Returns:
            Dict mapping ``sample_id`` → list of per-view dicts, each containing
            ``image``, ``label``, ``mask``, ``image_path``, ``view_id``.
        """
        sample_groups: Dict[str, List[Dict]] = defaultdict(list)

        for batch in dataloader:
            images = batch["image"]
            labels = batch["label"]
            masks = batch["mask"]
            image_paths = batch["image_path"]
            view_ids = batch["view_id"]

            if "sample_id" in batch:
                sample_ids = batch["sample_id"]
            else:
                sample_ids = [
                    re.sub(r"_C\d+$", "", Path(p).stem) for p in image_paths
                ]

            for img, lbl, mask, path, vid, sid in zip(
                images, labels, masks, image_paths, view_ids, sample_ids
            ):
                sample_groups[sid].append(
                    {"image": img, "label": lbl, "mask": mask, "image_path": path, "view_id": vid}
                )

        return sample_groups

    def _mv_evaluate_views_as_sample(
        self,
        test_dataloader: DataLoader,
        memory_banks: Dict[str, torch.Tensor],
        view_aggregation: str = "max",
        attention_unroi_weight: Optional[float] = None,
        attention_roi_threshold: Optional[float] = None,
        attention_interest_token_idx: Optional[Union[int, Tuple[int, int]]] = None,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        eval_resolution: Optional[Union[int, Tuple[int, int]]] = None,
        cosine_sim_clamp: Optional[tuple] = None,
        return_attentioned_anomaly_map: bool = True,
        anomaly_score_method: str = "top1",
        progress_desc: Optional[str] = None,
        save_results_path: Optional[PathLike] = None,
        save_with_normed_heatmap: bool = False,
        **kwargs,
    ) -> ADPerClassEvaluationOutput:
        """Evaluate in views-as-sample mode.

        All views of the same physical object are concatenated into a single
        batch for inference.  The resulting anomaly maps and scores are then
        aggregated across the view dimension using ``view_aggregation``
        (``"max"`` or ``"mean"``), producing one score and one map per object.
        """
        eval_device = self.eval_device
        eval_dtype = self.eval_dtype
        fallback_view = next(iter(memory_banks))

        self._reset_eval_metrics()

        # Load entire dataloader into memory grouped by sample
        sample_groups = self._mv_group_by_sample(test_dataloader)

        pbar = tqdm(
            sample_groups.items(),
            desc=progress_desc or "Multi-view eval [views_as_sample]",
            total=len(sample_groups),
        )

        for sample_id, view_data in pbar:
            # All views of this sample — group by view so each sub-batch is
            # homogeneous and can share a single memory bank call.
            view_to_items: Dict[str, List[Dict]] = defaultdict(list)
            for item in view_data:
                view_to_items[item["view_id"]].append(item)

            per_view_anomaly_maps: List[torch.Tensor] = []
            per_view_anomaly_scores: List[torch.Tensor] = []
            per_view_att_maps: Optional[List[torch.Tensor]] = [] if return_attentioned_anomaly_map else None
            per_view_att_scores: Optional[List[torch.Tensor]] = [] if return_attentioned_anomaly_map else None
            per_view_masks: List = []
            sample_label = view_data[0]["label"]

            for vid, items in view_to_items.items():
                prompt_features = memory_banks.get(vid, memory_banks[fallback_view])
                sub_images = [it["image"] for it in items]

                output = self(
                    prompt_images=prompt_features,
                    test_images=sub_images,
                    is_prompt_features=True,
                    attention_unroi_weight=attention_unroi_weight,
                    attention_roi_threshold=attention_roi_threshold,
                    attention_interest_token_idx=attention_interest_token_idx,
                    resolution=resolution,
                    cosine_sim_clamp=cosine_sim_clamp,
                    anomaly_score_method=anomaly_score_method,
                    upsample_anomaly_map=True,
                    upsample_resolution=eval_resolution or resolution,
                    return_attentioned_anomaly_map=return_attentioned_anomaly_map,
                    **kwargs,
                )

                per_view_anomaly_maps.append(output.anomaly_map)
                per_view_anomaly_scores.append(output.anomaly_score)
                if return_attentioned_anomaly_map:
                    per_view_att_maps.append(output.attentioned_anomaly_map)
                    per_view_att_scores.append(output.attentioned_anomaly_score)
                for it in items:
                    per_view_masks.append(it["mask"])

            # Aggregate across all views of this sample
            all_maps = torch.cat(per_view_anomaly_maps, dim=0)   # [V, 1, H, W]
            all_scores = torch.cat(per_view_anomaly_scores, dim=0)
            all_att_maps = torch.cat(per_view_att_maps, dim=0) if return_attentioned_anomaly_map else None
            all_att_scores = torch.cat(per_view_att_scores, dim=0) if return_attentioned_anomaly_map else None

            if view_aggregation == "max":
                agg_map = all_maps.max(dim=0, keepdim=True)[0]
                agg_score = all_scores.max(dim=0, keepdim=True)[0]
                agg_att_map = all_att_maps.max(dim=0, keepdim=True)[0] if return_attentioned_anomaly_map else None
                agg_att_score = all_att_scores.max(dim=0, keepdim=True)[0] if return_attentioned_anomaly_map else None
            else:  # mean
                agg_map = all_maps.mean(dim=0, keepdim=True)
                agg_score = all_scores.mean(dim=0, keepdim=True)
                agg_att_map = all_att_maps.mean(dim=0, keepdim=True) if return_attentioned_anomaly_map else None
                agg_att_score = all_att_scores.mean(dim=0, keepdim=True) if return_attentioned_anomaly_map else None

            eval_score = agg_att_score if return_attentioned_anomaly_map else agg_score
            eval_map = agg_att_map if return_attentioned_anomaly_map else agg_map

            label_val = sample_label.value if hasattr(sample_label, "value") else sample_label
            labels_tensor = torch.tensor([label_val], device=eval_device, dtype=torch.int8)
            self._update_eval_metrics(
                preds=eval_score.to(device=eval_device, dtype=eval_dtype),
                labels=labels_tensor,
                domain="image",
            )

            # Aggregate masks: union (max) across all views
            height, width = agg_map.shape[-2:]
            resized_masks = torch.stack([
                to_tensor(m.resize((width, height), resample=Image.NEAREST))
                for m in per_view_masks
            ])
            agg_mask = resized_masks.max(dim=0, keepdim=True)[0].to(device=eval_device, dtype=eval_dtype)
            self._update_eval_metrics(
                preds=eval_map.to(device=eval_device, dtype=eval_dtype),
                labels=agg_mask,
                domain="pixel",
            )

        results = self._compute_eval_metrics()
        return ADPerClassEvaluationOutput(
            AUROC=results["image"].get("AUROC"),
            AUPR=results["image"].get("AUPR"),
            AP=results["image"].get("AP"),
            F1Max=results["image"].get("F1Max"),
            pixel_AUROC=results["pixel"].get("AUROC"),
            pixel_AUPRO=results["pixel"].get("AUPRO"),
            pixel_F1Max=results["pixel"].get("F1Max"),
        )
