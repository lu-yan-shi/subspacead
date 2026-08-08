from typing import Any, Dict, List, Optional, Tuple, Union

import argparse
import os
import warnings

import torch

from pathlib import Path
from torchvision import transforms
from tqdm.auto import tqdm

from ad_pipelines.models import (
    DinoV2Model, 
    DinoV2WithRegisterModel, 
    DinoV3ViTModel, 
    DinoV3ConvNextModel, 
    VisRegModel,
    OpenCLIPModel, 
    MetaCLIP2Model,
    LingBotVisionModel,
    EUPEViTModel,
)
from ad_pipelines.pipelines.pipeline_base import AnomalyDetectionSetting
from ad_pipelines.pipelines import (
    PatchEADPipeline, 
    PatchIADPipeline, 
    DuoADPipeline
)
from ad_pipelines.datas.dataset import SplitType
from ad_pipelines.datas import MVTecDataset, VisADataset, RealIADDataset
from ad_pipelines import utils

def parse_args():
    parser = argparse.ArgumentParser(description="AD inference")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (e.g., 'cuda:0' or 'cpu').")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["bf16", "fp16", "fp32"],
                        help="Data type to use for inference (e.g., 'bf16', 'fp16', 'fp32').")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model file.")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="List of image file paths to process. Can be a single "
                             "folder or a list of folders.")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["mvtec", "visa", "realiad"],
                        help="Dataset type. If omitted, inferred from --data_path (legacy behaviour).")
    parser.add_argument("--model_resolution", type=int, default=None, help="Resolution of the model input.")
    parser.add_argument("--image_resolution", type=int, default=None, help="Resolution of the input images.")
    parser.add_argument("--eval_resolution", type=int, default=None,
                        help="Resolution to which anomaly maps are upsampled before pixel metrics are computed. "
                             "Defaults to --image_resolution. For RealIAD, defaults to 224 if not set.")
    parser.add_argument("--batch_budget", type=int, default=16, help="Batch size budget for inference.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the inference results.")
    parser.add_argument("--save_results", action="store_true", help="Save the results of the inference.")
    parser.add_argument("--seeds", type=int, nargs='+', default=[2356], 
                        help="List of seeds for reproducible sampling (e.g., --seeds 42 2356)")
    parser.add_argument("--shots", type=int, nargs='+', default=[1, 2, 4], 
                        help="List of shots for few-shot learning (e.g., --shots 1 4 8)")
    parser.add_argument("--pipeline", type=str, default="patchead", choices=PIPELINE_MAP.keys())
    parser.add_argument("--model", type=str, default="dinov3_vit", choices=MODEL_MAP.keys())
    parser.add_argument(
        "--augmentation_mode",
        type=str,
        default="none",
        choices=["force", "rotate", "rotate90", "flip", "auto", "none"],
        help="Prompt augmentation mode. 'auto' matches the old augmentation analysis flow.",
    )
    parser.add_argument(
        "--augmentation_analysis_split",
        type=str,
        default="test",
        choices=["test", "train"],
        help="Dataset split used by analyze_augmentation before selecting prompt augmentations.",
    )
    parser.add_argument("--no_saliency", action="store_true", help="Disable saliency weighting.")
    parser.add_argument("--duoad_sim_method", type=str, default=None, choices=DuoADPipeline.all_similarity_aggregations,
                        help="Similarity aggregation method to use for DuoAD.")
    parser.add_argument("--fusion", type=str, default=None, choices=PatchIADPipeline.all_layer_fusion_methods, 
                        help="similarity aggregation method to use.")
    parser.add_argument("--layers", type=int, nargs='+', default=[8, 10, 12], 
                        help="List of layer indices for AD computation.")
    parser.add_argument("--warmup_count", type=int, default=30, help="Number of warmup sample count.")
    # RealIAD-specific args
    parser.add_argument("--eval_image_dir", type=str, default="realiad_1024",
                        help="Image subdirectory name for RealIAD (e.g., 'realiad_1024', 'realiad_raw').")
    parser.add_argument("--eval_json_dir", type=str, default="realiad_jsons",
                        help="JSON subdirectory name for RealIAD (e.g., 'realiad_jsons', 'realiad_jsons_sv').")
    parser.add_argument("--camera_views", type=str, nargs='+', default=None,
                        help="Camera views to use for RealIAD (e.g., --camera_views C1 C2 ...). "
                             "Defaults to all views.")
    parser.add_argument("--evaluation_mode", type=str, default="view_as_image",
                        choices=["view_as_image", "views_as_sample"],
                        help="Multi-view evaluation mode for RealIAD.")
    parser.add_argument("--view_aggregation", type=str, default="max",
                        choices=["max", "mean"],
                        help="How to aggregate scores across views in 'views_as_sample' mode.")
    return parser.parse_args()

DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

_MODEL_MAP_ALL = {
    "dinov2": DinoV2Model,
    "dinov2_with_register": DinoV2WithRegisterModel,
    "dinov3_vit": DinoV3ViTModel,
    "dinov3_convnext": DinoV3ConvNextModel,
    "visreg": VisRegModel,
    "open_clip": OpenCLIPModel,
    "meta_clip2": MetaCLIP2Model,
    "lingbot_vision": LingBotVisionModel,
    "eupe_vit": EUPEViTModel,
}
MODEL_MAP = {k: v for k, v in _MODEL_MAP_ALL.items() if v is not None}

PIPELINE_MAP = {
    "patchead": PatchEADPipeline,
    "patchiad": PatchIADPipeline,
    "duoad": DuoADPipeline,
}

def _resolve_dataset(args) -> str:
    """Return the canonical dataset name, resolving from --dataset or path-sniffing."""
    if args.dataset is not None:
        return args.dataset
    # Legacy path-sniffing fallback
    path_lower = args.data_path.lower()
    if "mvtec_loco" in path_lower:
        raise ValueError("MVTec-LOCO is not included in this release. Use a supported dataset instead.")
    if "mvtec" in path_lower:
        return "mvtec"
    if "visa" in path_lower:
        return "visa"
    if "realiad" in path_lower or "real_iad" in path_lower or "real-iad" in path_lower:
        return "realiad"
    raise ValueError(
        f"Cannot infer dataset type from path '{args.data_path}'. "
        "Use --dataset to specify it explicitly."
    )

def main(args):
    print(args)

    dataset_name = _resolve_dataset(args)

    # Smart eval_resolution default for RealIAD
    eval_resolution = args.eval_resolution
    if dataset_name == "realiad" and eval_resolution is None:
        warnings.warn(
            "RealIAD images are 1024×1024. --eval_resolution was not set; "
            "defaulting to 224 to avoid OOM during pixel-metric accumulation. "
            "Pass --eval_resolution explicitly to suppress this warning.",
            stacklevel=2,
        )
        eval_resolution = 224

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    model_kwargs = {}
    if args.model_resolution:
        model_kwargs["resolution"] = args.model_resolution

    model_class = MODEL_MAP[args.model]
    model = model_class(
        args.model_path, **model_kwargs, dtype=dtype, device=device
    )

    pipeline_class = PIPELINE_MAP[args.pipeline]
    patchead = pipeline_class(model, device=device, dtype=dtype, analyze_augment_sample_count=args.warmup_count)

    if dataset_name == "mvtec":
        test_dataset = MVTecDataset(args.data_path, split=SplitType.TEST, random_seed=args.seeds[0])
        prompt_dataset = MVTecDataset(args.data_path, split=SplitType.TRAIN, random_seed=args.seeds[0])
        setting = AnomalyDetectionSetting.FEW_SHOT
    elif dataset_name == "visa":
        test_dataset = VisADataset(args.data_path, split=SplitType.TEST, random_seed=args.seeds[0])
        prompt_dataset = VisADataset(args.data_path, split=SplitType.TRAIN, random_seed=args.seeds[0])
        setting = AnomalyDetectionSetting.FEW_SHOT
    elif dataset_name == "realiad":
        test_dataset = RealIADDataset(
            args.data_path,
            split=SplitType.TEST,
            json_dir=args.eval_json_dir,
            image_dir=args.eval_image_dir,
            camera_views=args.camera_views,
            random_seed=args.seeds[0],
        )
        prompt_dataset = RealIADDataset(
            args.data_path,
            split=SplitType.TRAIN,
            json_dir=args.eval_json_dir,
            image_dir=args.eval_image_dir,
            camera_views=args.camera_views,
            random_seed=args.seeds[0],
        )
        setting = AnomalyDetectionSetting.MULTI_VIEW_FEW_SHOT
    else:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    extra_kwargs = {}
    if args.image_resolution:
        extra_kwargs["resolution"] = args.image_resolution
    if eval_resolution is not None:
        extra_kwargs["eval_resolution"] = eval_resolution
    extra_kwargs["augmentation_mode"] = args.augmentation_mode
    extra_kwargs["augmentation_analysis_split"] = args.augmentation_analysis_split

    if args.pipeline == "duoad" and args.duoad_sim_method is not None:
        extra_kwargs["similarity_aggregation"] = args.duoad_sim_method
    if args.pipeline in ("patchead", "patchiad", "duoad"):
        extra_kwargs["output_feature_maps_indices"] = tuple(args.layers)
        if args.model == "dinov3_convnext" and args.layers != [-1]:
            raise ValueError(
                "dinov3_convnext only supports a single last-layer feature map "
                "(--layers -1). Multi-layer fusion is not implemented for ConvNeXt backbones."
            )
    if args.pipeline in ("patchiad", "duoad"):
        if args.fusion is not None:
            extra_kwargs["layer_fusion_method"] = args.fusion

    if args.no_saliency:
        extra_kwargs["return_attentioned_anomaly_map"] = False

    # RealIAD multi-view kwargs
    if dataset_name == "realiad":
        extra_kwargs["evaluation_mode"] = args.evaluation_mode
        extra_kwargs["view_aggregation"] = args.view_aggregation

    for shot in args.shots:
        patchead.evaluation_multi_run(
            output_path, setting, test_dataset, prompt_dataset, 
            seeds=args.seeds, shots=shot, batch_size=args.batch_budget, save_result_images=args.save_results,
            save_with_normed_heatmap=True,
            **extra_kwargs
        )

if __name__ == "__main__":
    args = parse_args()
    main(args)
