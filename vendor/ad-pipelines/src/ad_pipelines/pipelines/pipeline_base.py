from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Union
from typing_extensions import Self

import numpy as np
import subprocess
import torch
import torch.nn as nn

from anomalib.metrics import AUROC, AUPR, AUPRO, F1Max, F1Score
from anomalib.data import ImageBatch
from enum import Enum
from os import PathLike
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAveragePrecision
from transformers.utils import ModelOutput

from ..models.model_base import BaseModel, VisionEncoderOutput
from ..datas import BaseAnomalyDataset, BaseAnomalyClassDataset
import json
import pandas as pd
from pathlib import Path

@dataclass
class ADPerClassEvaluationOutput(ModelOutput):
    """
    Output class for AD evaluation pipeline per 1 class.
    """
    AUROC: Optional[float] = None
    AUPR: Optional[float] = None
    AP: Optional[float] = None
    F1Max: Optional[float] = None
    pixel_AUROC: Optional[float] = None
    pixel_AUPRO: Optional[float] = None
    pixel_F1Max: Optional[float] = None
    
    def __str__(self) -> str:
        """Custom string representation for the evaluation output."""
        lines = ["=" * 80]
        
        # Header row
        lines.append(f"{'Metric':<12} {'Image-level':<12} {'Pixel-level':<12}")
        lines.append("-" * 80)
        
        # AUROC row
        image_auroc = f"{self.AUROC:.4f}" if self.AUROC is not None else "N/A"
        pixel_auroc = f"{self.pixel_AUROC:.4f}" if self.pixel_AUROC is not None else "N/A"
        lines.append(f"{'AUROC':<12} {image_auroc:<12} {pixel_auroc:<12}")
        
        # AUPR/AUPRO row
        image_aupr = f"{self.AUPR:.4f}" if self.AUPR is not None else "N/A"
        pixel_aupro = f"{self.pixel_AUPRO:.4f}" if self.pixel_AUPRO is not None else "N/A"
        lines.append(f"{'AUPR/AUPRO':<12} {image_aupr:<12} {pixel_aupro:<12}")
        
        # AP row (only for image-level)
        image_ap = f"{self.AP:.4f}" if self.AP is not None else "N/A"
        lines.append(f"{'AP':<12} {image_ap:<12} {'-':<12}")
        
        # F1Max row
        image_f1max = f"{self.F1Max:.4f}" if self.F1Max is not None else "N/A"
        pixel_f1max = f"{self.pixel_F1Max:.4f}" if self.pixel_F1Max is not None else "N/A"
        lines.append(f"{'F1Max':<12} {image_f1max:<12} {pixel_f1max:<12}")
        
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Custom representation for debugging."""
        return (f"ADPerClassEvaluationOutput("
                f"AUROC={self.AUROC:.4f}, AUPR={self.AUPR:.4f}, AP={self.AP:.4f}, F1Max={self.F1Max:.4f}, "
                f"pixel_AUROC={self.pixel_AUROC:.4f}, pixel_AUPRO={self.pixel_AUPRO:.4f}, "
                f"pixel_F1Max={self.pixel_F1Max:.4f})")

class AnomalyDetectionSetting(Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    MULTI_VIEW_FEW_SHOT = "multi_view_few_shot"

class AnomalyDetectionPipelineBase(ABC):
    """
    Base class for anomaly detection pipelines
    Provides unified interface for handling different visual model backbones
    """
    
    def __init__(
        self,
        model: BaseModel,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        eval_device: Optional[Union[str, torch.device]] = None,
        eval_dtype: Optional[Union[str, torch.dtype]] = None,
        pixel_auroc_thresholds: Optional[int] = None,
        **kwargs
    ):
        self.model = model
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.dtype = dtype or torch.float32
        self.eval_device = eval_device or self.device
        self.eval_dtype = eval_dtype or self.dtype

        # Torchmetrics like metrics class mapping
        self.eval_metrics = {
            "image": {
                "AUROC": AUROC(fields=["pred_score", "gt_label"]),
                "AUPR": AUPR(fields=["pred_score", "gt_label"]),
                "AP": BinaryAveragePrecision(),
                "F1Max": F1Max(fields=["pred_score", "gt_label"]),
            },
            "pixel": {
                # pixel_auroc_thresholds caps the number of operating points in the ROC curve.
                # Default None = unbounded (exact), which is correct for MVTec/ViSA.
                # For RealIAD (thousands of 1024×1024 images) pass e.g. 1000 to cap RAM usage;
                # be aware that pre-binned thresholds can meaningfully underestimate pixel-AUROC
                # when the score distribution is concentrated in a narrow sub-range.
                "AUROC": AUROC(fields=["anomaly_map", "gt_mask"], thresholds=pixel_auroc_thresholds),
                "AUPRO": AUPRO(fields=["anomaly_map", "gt_mask"]),
                "F1Max": F1Max(fields=["anomaly_map", "gt_mask"]),
            }
        }

    def _reset_evaluation_metadata(self) -> None:
        """Hook for clearing per-run metadata before evaluation starts."""
        pass

    def _get_evaluation_config_metadata(self) -> Dict[str, Any]:
        """Hook for extra config fields saved alongside evaluation kwargs."""
        return {}

    def _prepare_eval_metrics(self):
        """
        Reset and relocate evaluation metrics in-place.
        Iterates over self.eval_metrics (a mapping of domains to metric objects), calling
        metric.reset() and metric.to(self.device, self.dtype) for each metric so they are
        ready for evaluation on the configured device and dtype. No return value.
        """        
        for domain in self.eval_metrics.values():
            for metric in domain.values():
                metric.reset()
                metric.to(self.eval_device, self.eval_dtype)

    def _reset_eval_metrics(self):
        """
        Reset evaluation metrics in-place.
        Iterates over self.eval_metrics (a mapping of domains to metric objects), calling
        metric.reset() for each metric to clear its state. No return value.
        """
        for domain in self.eval_metrics.values():
            for metric in domain.values():
                metric.reset()

    def _update_eval_metrics(self, preds: torch.Tensor, labels: torch.Tensor, domain: str):
        if domain not in self.eval_metrics:
            raise ValueError(f"Unknown domain: {domain}. Available domains: {list(self.eval_metrics.keys())}")
        for metric_name, metric in self.eval_metrics[domain].items():
            if metric_name == "AP":
                # BinaryAveragePrecision expects direct tensor inputs
                metric.update(preds, labels)
            else:
                # AUROC, AUPR, AUPRO, F1Max expect field-based inputs
                if hasattr(metric, 'fields') and len(metric.fields) >= 2:
                    pred_field = metric.fields[0]
                    target_field = metric.fields[1]
                else:
                    raise NotImplementedError("Metric fields are not defined or not supported for this metric.")
                batch = {
                    pred_field: preds,
                    target_field: labels,
                }
                dummy_image = torch.zeros(1, 3, 512, 512)  # Dummy image tensor
                batch = ImageBatch(dummy_image, **batch)
                metric.update(batch)

    def _compute_eval_metrics(self, domain: str = None) -> Dict[str, torch.Tensor]:
        if domain and domain not in self.eval_metrics:
            raise ValueError(f"Unknown domain: {domain}. Available domains: {list(self.eval_metrics.keys())}")
        
        results = {}
        
        if domain is None:
            # Compute all metrics for all domains
            for domain_name, domain_metrics in self.eval_metrics.items():
                domain_results = {}
                for metric_name, metric in domain_metrics.items():
                    score = metric.compute().detach().cpu().item()
                    domain_results[metric_name] = score
                results[domain_name] = domain_results
        else:
            # Compute metrics for specific domain
            results[domain] = {}
            for metric_name, metric in self.eval_metrics[domain].items():
                score = metric.compute().detach().cpu().item()
                results[domain][metric_name] = score

        return results

    @abstractmethod
    def preprocess(
        self, 
        images: Union[Image.Image, torch.Tensor, np.ndarray, list],
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        do_normalize: Optional[bool] = None,
        **kwargs
    ) -> torch.Tensor:
        """Preprocess images for the model"""
        pass

    @abstractmethod
    def get_features(
        self, 
        image_tensors: torch.Tensor,
        return_attentions: Optional[bool] = False,
        return_layer_features: Optional[bool] = False,
        **kwargs
    ) -> VisionEncoderOutput:
        """Extract features from the model"""
        pass
        
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement the __call__ method to handle model inference.")

    # Copied from HuggingFace Diffusers
    def to(self, *args, **kwargs) -> Self:
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)

        if len(args) == 1:
            if isinstance(args[0], torch.dtype):
                dtype_arg = args[0]
            else:
                device_arg = torch.device(args[0]) if args[0] is not None else None
        elif len(args) == 2:
            if isinstance(args[0], torch.dtype):
                raise ValueError(
                    "When passing two arguments, make sure the first corresponds to `device` and the second to `dtype`."
                )
            device_arg = torch.device(args[0]) if args[0] is not None else None
            dtype_arg = args[1]
        elif len(args) > 2:
            raise ValueError("Please make sure to pass at most two arguments (`device` and `dtype`) `.to(...)`")

        if dtype is not None and dtype_arg is not None:
            raise ValueError(
                "You have passed `dtype` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        dtype = dtype or dtype_arg

        if device is not None and device_arg is not None:
            raise ValueError(
                "You have passed `device` both as an argument and as a keyword argument. Please only pass one of the two."
            )

        device = device or device_arg
        self.model.to(device=device, dtype=dtype)

        self.device = device
        self.dtype = dtype

        return self

    def _get_git_commit_hash(self) -> Optional[str]:
        """
        Get the current git commit hash for reproducibility tracking.
        
        This method attempts to retrieve the current git commit hash from the repository.
        If git is not available or the directory is not a git repository, it returns None.
        
        Returns:
            Optional[str]: The current git commit hash (short form, 7 characters) if available,
                          None otherwise.
        
        Example:
            >>> pipeline = SomePipeline(model)
            >>> commit_hash = pipeline.get_git_commit_hash()
            >>> print(commit_hash)  # 'a1b2c3d' or None
        """
        try:
            # Get the directory of the current file
            current_dir = Path(__file__).parent.parent.parent.parent
            
            # Try to get the git commit hash
            result = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                cwd=current_dir,
                capture_output=True,
                text=True,
                timeout=5,
                check=False
            )
            
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
                return commit_hash if commit_hash else None
            else:
                return None
                
        except (subprocess.SubprocessError, FileNotFoundError, Exception):
            # Git not available or not a git repository
            return None

    def _set_seed(self, seed: int) -> None:
        """
        Set all random seeds for reproducibility.
        
        This method sets seeds for:
        - Python's random module
        - NumPy's random number generator
        - PyTorch's random number generator (CPU and CUDA)
        - CUDA deterministic behavior
        
        Args:
            seed (int): Random seed value
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Enable deterministic behavior for CUDA operations
            # Note: This may reduce performance
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Request deterministic implementations for all ops that have them.
        # warn_only=True: ops without a deterministic implementation emit a
        # warning instead of raising RuntimeError, so evaluation still runs.
        torch.use_deterministic_algorithms(True, warn_only=True)
    
    def upsample_anomaly_map(
        self, 
        anomaly_map: torch.Tensor, 
        target_size: Union[int, Tuple[int, int]],
        mode: str = "bilinear"
    ) -> torch.Tensor:
        """
        Upsample the anomaly map to a target spatial size.

        Args:
            anomaly_map (torch.Tensor): Input tensor containing anomaly scores/maps (e.g. shape (N, C, H, W) or (C, H, W)).
            target_size (int or tuple[int, int]): Desired output spatial size in (width, height). 
                If an int is provided, it is treated as (target_size, target_size).
            mode (str): The interpolation mode to use (e.g. "bilinear", "nearest").

        Returns:
            torch.Tensor: The anomaly map resized to the given spatial dimensions, on the same device and with the same dtype.
            
        Notes:
            Uses torch.nn.functional.interpolate; align_corners is only passed for modes that support it.
        Upsample the anomaly map to the original image size.
        """
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        
        is_channel = True
        # Ensure the input tensor has 4 dimensions (N, C, H, W)
        if anomaly_map.dim() == 3:
            anomaly_map = anomaly_map.unsqueeze(1)  # Add channel dimension
            is_channel = False
        
        # Some interpolation modes (e.g., "bilinear", "bicubic", "trilinear") accept the align_corners argument,
        # while others like "nearest" do not. Only pass align_corners when supported.
        modes_with_align = {"linear", "bilinear", "bicubic", "trilinear"}
        if mode in modes_with_align:
            result = F.interpolate(anomaly_map, size=(target_size[1], target_size[0]), mode=mode, align_corners=False)
        else:
            result = F.interpolate(anomaly_map, size=(target_size[1], target_size[0]), mode=mode)
        
        # Return tensor with original number of dimensions
        if not is_channel:
            result = result.squeeze(1)  # Remove channel dimension if it was added

        return result

    def evaluation(
        self, 
        output_path: PathLike,
        setting: AnomalyDetectionSetting, 
        dataset: Union[BaseAnomalyDataset, BaseAnomalyClassDataset],
        prompt_dataset: Optional[Union[BaseAnomalyDataset, BaseAnomalyClassDataset]] = None,
        batch_size: int = 16,
        seed: int = 42,
        save_result_images: bool = False,
        **kwargs
    ):
        # Check inputs
        if prompt_dataset is not None:
            if not isinstance(prompt_dataset, type(dataset)):
                raise ValueError(f"prompt_dataset must be of the same type as dataset. "
                                 f"Expected {type(dataset)}, got {type(prompt_dataset)}")
        self._prepare_eval_metrics()
        self._reset_evaluation_metadata()
        
        # Set all random seeds for reproducibility
        self._set_seed(seed)
        dataset.random_seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)

        evaluation_result = {}
        self.model.eval()
        shots = kwargs.pop("shots", 0)
        with torch.no_grad():
            dataset_classes = dataset if isinstance(dataset, BaseAnomalyDataset) else {f"{dataset.category}": dataset}
            if isinstance(prompt_dataset, BaseAnomalyClassDataset):
                prompt_dataset = {prompt_dataset.category: prompt_dataset}

            for idx, (class_name, dataset_class) in enumerate(dataset_classes.items()):
                # Re-seed for each class to ensure consistency
                self._set_seed(seed)
                dataset_class.random_seed = seed
                generator.manual_seed(seed)

                self._reset_eval_metrics()
                a_prompt_dataset = prompt_dataset[class_name]
                dataloader = DataLoader(
                    dataset_class, batch_size=batch_size, shuffle=True, collate_fn=dataset_class.get_collate_fn(),
                    generator=generator
                )

                if setting == AnomalyDetectionSetting.ZERO_SHOT:
                    res = self.zero_shot(test_dataloader=dataloader, seed=seed, **kwargs)

                elif setting == AnomalyDetectionSetting.FEW_SHOT:
                    if shots is None:
                        raise ValueError("shots must be specified for few-shot evaluation.")
                    
                    save_results_path = None
                    if save_result_images:
                        save_results_path = output_path / f"{shots}-shot" / class_name
                        save_results_path.mkdir(parents=True, exist_ok=True)

                    res = self.few_shot(
                        test_dataloader=dataloader, 
                        seed=seed, 
                        prompt_dataset=a_prompt_dataset, 
                        shots=shots,
                        progress_desc=f"[{idx+1}/{len(dataset_classes)}|{shots}-shot|{class_name}]",
                        save_results_path=save_results_path,
                        **kwargs
                    )

                elif setting == AnomalyDetectionSetting.MULTI_VIEW_FEW_SHOT:
                    if shots is None:
                        raise ValueError("shots must be specified for multi-view few-shot evaluation.")

                    save_results_path = None
                    if save_result_images:
                        save_results_path = output_path / f"{shots}-shot-multiview" / class_name
                        save_results_path.mkdir(parents=True, exist_ok=True)

                    res = self.multi_view_few_shot(
                        test_dataloader=dataloader,
                        seed=seed,
                        prompt_dataset=a_prompt_dataset,
                        shots=shots,
                        progress_desc=f"[{idx+1}/{len(dataset_classes)}|{shots}-shot-mv|{class_name}]",
                        save_results_path=save_results_path,
                        **kwargs
                    )

                else:
                    raise ValueError(f"Unknown setting: {setting}. Expected ZERO_SHOT, FEW_SHOT, or MULTI_VIEW_FEW_SHOT.")

                evaluation_result[class_name] = res

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(res)

        # Save evaluation results and kwargs
        
        # Create output directory if it doesn't exist
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save kwargs as JSON
        kwargs_to_save = kwargs.copy()
        kwargs_to_save.update({
            'setting': setting.value,
            'batch_size': batch_size,
            'seed': seed,
            'git_commit': self._get_git_commit_hash()
        })
        # Keep run metadata in config JSON so result files remain metric-only.
        kwargs_to_save.update(self._get_evaluation_config_metadata())
        
        kwargs_file = Path(output_path) / f'config_{shots}_shots.json'
        with open(kwargs_file, 'w') as f:
            json.dump(kwargs_to_save, f, indent=4, default=str)
        
        # Save evaluation results as JSON
        results_json_file = Path(output_path) / f'evaluation_results_{shots}_shots.json'
        with open(results_json_file, 'w') as f:
            json.dump(evaluation_result, f, indent=4, default=str)
        
        # Convert evaluation_result to a more readable format for CSV
        rows = []
        for class_name, metrics in evaluation_result.items():
            row = {'class_name': class_name}
            row.update(metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Calculate mean of all classes
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        mean_row = {'class_name': 'mean'}
        for col in numeric_columns:
            mean_row[col] = df[col].mean()
        
        # Append mean row to dataframe
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
        
        results_csv_file = Path(output_path) / f'evaluation_results_{shots}_shots.csv'
        df.to_csv(results_csv_file, index=False)
        
        print(f"Results saved to:")
        print(f"  JSON: {results_json_file}")
        print(f"  CSV: {results_csv_file}")
        print(f"  Configs: {kwargs_file}")

        return evaluation_result


    def evaluation_multi_run(
        self, 
        output_path: PathLike,
        setting: AnomalyDetectionSetting, 
        dataset: Union[BaseAnomalyDataset, BaseAnomalyClassDataset],
        prompt_dataset: Optional[Union[BaseAnomalyDataset, BaseAnomalyClassDataset]] = None,
        batch_size: int = 16,
        seeds: Tuple[int] = (21, 42, 63),
        save_result_images: bool = False,
        **kwargs
    ):
        """
        Run evaluation multiple times with different seeds and compute statistics.
        
        Args:
            output_path: Path to save results
            setting: Zero-shot or few-shot setting
            dataset: Test dataset
            prompt_dataset: Prompt dataset for few-shot
            batch_size: Batch size for evaluation
            seeds: Tuple of random seeds to use
            save_result_images: Whether to save result images
            **kwargs: Additional arguments
            
        Returns:
            List of evaluation results for each seed
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Run evaluation for each seed
        results = []
        print(f"\n{'='*80}")
        print(f"Running multi-seed evaluation with seeds: {seeds}")
        print(f"{'='*80}\n")
        
        for i, seed in enumerate(seeds, 1):
            print(f"\n[Run {i}/{len(seeds)}] Seed: {seed}")
            print("-" * 80)
            # Set global seed before each run
            self._set_seed(seed)
            results.append(self.evaluation(
                output_path=output_path / f"seed_{seed}",
                setting=setting,
                dataset=dataset,
                prompt_dataset=prompt_dataset,
                batch_size=batch_size,
                seed=seed,
                save_result_images=save_result_images,
                **kwargs
            ))
        
        # Aggregate statistics across seeds
        print(f"\n{'='*80}")
        print("Computing statistics across seeds...")
        print(f"{'='*80}\n")
        
        # Get shots value for file naming
        shots = kwargs.get("shots", None)
        if shots is None:
            raise ValueError("shots must be specified in kwargs for evaluation.")
        
        # Collect all class names
        all_class_names = set()
        for result in results:
            all_class_names.update(result.keys())
        all_class_names = sorted(all_class_names)
        
        # Metric names to track
        metric_names = [
            'AUROC', 'AUPR', 'AP', 'F1Max',
            'pixel_AUROC', 'pixel_AUPRO', 'pixel_F1Max'
        ]
        
        # Compute statistics for each class
        statistics = {}
        for class_name in all_class_names:
            statistics[class_name] = {}
            
            # Collect values across seeds for each metric
            for metric_name in metric_names:
                values = []
                for result in results:
                    if class_name in result and metric_name in result[class_name]:
                        value = result[class_name][metric_name]
                        if value is not None:
                            values.append(value)
                
                if values:
                    mean_val = np.mean(values)
                    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
                    statistics[class_name][f'{metric_name}_mean'] = mean_val
                    statistics[class_name][f'{metric_name}_std'] = std_val
                else:
                    statistics[class_name][f'{metric_name}_mean'] = None
                    statistics[class_name][f'{metric_name}_std'] = None
        
        # Save statistics as JSON
        stats_json_file = output_path / f'statistics_{shots}_shots.json'
        with open(stats_json_file, 'w') as f:
            json.dump(statistics, f, indent=4, default=str)
        
        # Create detailed CSV report
        rows = []
        for class_name in all_class_names:
            row = {'class_name': class_name}
            for metric_name in metric_names:
                mean_key = f'{metric_name}_mean'
                std_key = f'{metric_name}_std'
                if mean_key in statistics[class_name] and statistics[class_name][mean_key] is not None:
                    mean_val = statistics[class_name][mean_key]
                    std_val = statistics[class_name][std_key]
                    row[f'{metric_name}_mean'] = mean_val
                    row[f'{metric_name}_std'] = std_val
                    # Combined format: "mean ± std"
                    row[f'{metric_name}'] = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    row[f'{metric_name}_mean'] = None
                    row[f'{metric_name}_std'] = None
                    row[f'{metric_name}'] = 'N/A'
            rows.append(row)
        
        df_stats = pd.DataFrame(rows)
        
        # Calculate overall mean across classes for each run
        overall_means_per_run = {metric_name: [] for metric_name in metric_names}
        
        for result in results:
            for metric_name in metric_names:
                # Collect metric values for all classes in this run
                metric_values = []
                for class_name in all_class_names:
                    if class_name in result and metric_name in result[class_name]:
                        value = result[class_name][metric_name]
                        if value is not None:
                            metric_values.append(value)
                
                # Compute mean across classes for this run
                if metric_values:
                    overall_means_per_run[metric_name].append(np.mean(metric_values))
        
        # Compute statistics of the overall means across runs
        overall_mean_row = {'class_name': 'overall_mean'}
        
        for metric_name in metric_names:
            mean_col = f'{metric_name}_mean'
            std_col = f'{metric_name}_std'
            
            if overall_means_per_run[metric_name]:
                # Mean and std of the per-run overall means
                overall_mean = np.mean(overall_means_per_run[metric_name])
                overall_std = np.std(overall_means_per_run[metric_name], ddof=1) if len(overall_means_per_run[metric_name]) > 1 else 0.0
                
                overall_mean_row[mean_col] = overall_mean
                overall_mean_row[std_col] = overall_std
                overall_mean_row[metric_name] = f"{overall_mean:.4f} ± {overall_std:.4f}"
            else:
                overall_mean_row[mean_col] = None
                overall_mean_row[std_col] = None
                overall_mean_row[metric_name] = 'N/A'
        
        df_stats = pd.concat([df_stats, pd.DataFrame([overall_mean_row])], ignore_index=True)
        
        # Save statistics CSV
        stats_csv_file = output_path / f'statistics_{shots}_shots.csv'
        df_stats.to_csv(stats_csv_file, index=False)
        
        # Create a summary report with formatted output
        summary_file = output_path / f'summary_{shots}_shots.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Multi-Seed Evaluation Summary ({len(seeds)} runs)\n")
            f.write(f"Seeds: {seeds}\n")
            f.write(f"Setting: {setting.value}\n")
            f.write(f"Shots: {shots}\n")
            f.write("=" * 80 + "\n\n")
            
            # Image-level metrics
            f.write("IMAGE-LEVEL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class':<20} {'AUROC':<13} {'AUPR':<13} {'AP':<13} {'F1Max':<13}\n")
            f.write("-" * 80 + "\n")
            
            for class_name in all_class_names:
                f.write(f"{class_name:<20} ")
                for metric in ['AUROC', 'AUPR', 'AP', 'F1Max']:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    if mean_key in statistics[class_name] and statistics[class_name][mean_key] is not None:
                        mean_val = statistics[class_name][mean_key] * 100
                        std_val = statistics[class_name][std_key] * 100
                        f.write(f"{mean_val:6.2f}±{std_val:5.2f}% ")
                    else:
                        f.write(f"{'N/A':<18} ")
                f.write("\n")
            
            # Overall mean
            if 'overall_mean' in df_stats['class_name'].values:
                overall_idx = df_stats[df_stats['class_name'] == 'overall_mean'].index[0]
                f.write("-" * 80 + "\n")
                f.write(f"{'MEAN':<20} ")
                for metric in ['AUROC', 'AUPR', 'AP', 'F1Max']:
                    mean_col = f'{metric}_mean'
                    std_col = f'{metric}_std'
                    if mean_col in df_stats.columns:
                        mean_val = df_stats.loc[overall_idx, mean_col]
                        std_val = df_stats.loc[overall_idx, std_col]
                        if pd.notna(mean_val):
                            f.write(f"{mean_val*100:6.2f}±{std_val*100:5.2f}% ")
                        else:
                            f.write(f"{'N/A':<18} ")
                f.write("\n")
            
            f.write("\n")
            
            # Pixel-level metrics
            f.write("PIXEL-LEVEL METRICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Class':<20} {'AUROC':<13} {'AUPRO':<13} {'F1Max':<13}\n")
            f.write("-" * 80 + "\n")
            
            for class_name in all_class_names:
                f.write(f"{class_name:<20} ")
                for metric in ['pixel_AUROC', 'pixel_AUPRO', 'pixel_F1Max']:
                    mean_key = f'{metric}_mean'
                    std_key = f'{metric}_std'
                    if mean_key in statistics[class_name] and statistics[class_name][mean_key] is not None:
                        mean_val = statistics[class_name][mean_key] * 100
                        std_val = statistics[class_name][std_key] * 100
                        f.write(f"{mean_val:6.2f}±{std_val:5.2f}% ")
                    else:
                        f.write(f"{'N/A':<18} ")
                f.write("\n")
            
            # Overall mean
            if 'overall_mean' in df_stats['class_name'].values:
                f.write("-" * 80 + "\n")
                f.write(f"{'MEAN':<20} ")
                for metric in ['pixel_AUROC', 'pixel_AUPRO', 'pixel_F1Max']:
                    mean_col = f'{metric}_mean'
                    std_col = f'{metric}_std'
                    if mean_col in df_stats.columns:
                        mean_val = df_stats.loc[overall_idx, mean_col]
                        std_val = df_stats.loc[overall_idx, std_col]
                        if pd.notna(mean_val):
                            f.write(f"{mean_val*100:6.2f}±{std_val*100:5.2f}% ")
                        else:
                            f.write(f"{'N/A':<18} ")
                f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        # Print summary to console
        print(f"\n{'='*80}")
        print("MULTI-SEED EVALUATION SUMMARY")
        print(f"{'='*80}")
        with open(summary_file, 'r') as f:
            print(f.read())
        
        print(f"\nResults saved to:")
        print(f"  Summary: {summary_file}")
        print(f"  JSON: {stats_json_file}")
        print(f"  CSV: {stats_csv_file}")
        print(f"{'='*80}\n")

        return results


    @abstractmethod
    def zero_shot(
        self,
        test_dataloader: DataLoader, 
        seed: int = 42,
        save_results_path: Optional[PathLike] = None,
        **kwargs
    ) -> ADPerClassEvaluationOutput:
        raise NotImplementedError("Zero-shot evaluation is not implemented yet.")
                    
    @abstractmethod
    def few_shot(
        self, 
        shots: int, 
        test_dataloader: DataLoader, 
        prompt_dataset: BaseAnomalyClassDataset,
        seed: int = 42,
        save_results_path: Optional[PathLike] = None,
        **kwargs
    ) -> ADPerClassEvaluationOutput:
        raise NotImplementedError("Few-shot evaluation is not implemented yet.")

    def multi_view_few_shot(
        self,
        shots: int,
        test_dataloader: DataLoader,
        prompt_dataset: BaseAnomalyClassDataset,
        seed: int = 42,
        save_results_path: Optional[PathLike] = None,
        **kwargs
    ) -> ADPerClassEvaluationOutput:
        """Multi-view few-shot evaluation.

        Default implementation falls back to standard ``few_shot``, treating
        each view as an independent image and using a single shared memory bank.
        Override this method in pipelines that need true per-view memory banks
        and cross-view aggregation (e.g. ``PatchEADPipeline``).
        """
        return self.few_shot(
            shots=shots,
            test_dataloader=test_dataloader,
            prompt_dataset=prompt_dataset,
            seed=seed,
            save_results_path=save_results_path,
            **kwargs
        )
        
                
