"""
深瞳 异常检测器
基于 DINOv2 + 记忆库的少样本异常检测（Training-Free）
"""

from __future__ import annotations

import logging
import math
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# SubspaceAD pipeline imports (requires: pip install ad-pipelines)
_MISSING_PIPELINE_MSG = ""
try:
    from ad_pipelines.models import DinoV2WithRegisterModel
    from ad_pipelines.pipelines import DuoADPipeline, PatchEADOutput
    _HAS_PIPELINE = True
except ImportError as e:
    DinoV2WithRegisterModel = None  # type: ignore
    DuoADPipeline = None   # type: ignore
    PatchEADOutput = None  # type: ignore
    _HAS_PIPELINE = False
    _MISSING_PIPELINE_MSG = str(e)

# Localization module (compatible with SubspaceAD attention maps)
try:
    from .subspacead.core.localization import (
        ObjectLocalizer,
        LocalizationResult,
        # 别名导入：detect() 的 crop_to_roi 参数会遮蔽同名函数，调用处必须用此别名
        crop_to_roi as crop_to_roi_fn,
        map_anomaly_to_original,
        expand_bbox_square,
    )
    from .subspacead.utils.common import min_max_norm
except ImportError as e:
    raise ImportError(f"无法导入 subspacead 包。请确保文件结构完整。错误详情：{e}")

# LayoutAD double-check (optional GNN-based structural verification)
_LAYOUTAD_AVAILABLE = False
try:
    from .layoutad.inference import LayoutADInference
    _LAYOUTAD_AVAILABLE = True
except ImportError:
    LayoutADInference = None  # type: ignore

# Training-free graph structure check (always available, zero dependencies)
try:
    from .layoutad.graph_check import GraphStructureChecker
except ImportError as e:
    GraphStructureChecker = None  # type: ignore
    logger = logging.getLogger(__name__)
    logger.warning("GraphStructureChecker import failed: %s. Graph double-check disabled.", e)

logger = logging.getLogger(__name__)

# Default model: DINOv2 with registers (public, no auth required)
DEFAULT_MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "facebook/dinov2-with-registers-base",
)

# ============================================================
# 图片尺寸适配：保比例缩放 + 灰边填充（letterbox）
# 适配任意尺寸/长宽比的输入图片，避免非方形图被直接拉伸变形
# ============================================================
LETTERBOX_FILL = 128  # 灰边填充色


def _letterbox(img: Image.Image, res: int, fill: int = LETTERBOX_FILL) -> Tuple[Image.Image, Dict]:
    """保比例缩放到 res×res 方形画布，短边居中后用中性灰补齐。

    Returns:
        (画布, geom)。geom 记录内容区几何信息，用于后续裁边/换算坐标：
        scale=缩放比例, ox/oy=内容区左上角, cw/ch=内容区宽高,
        orig_w/orig_h=原始尺寸, res=画布边长。
    """
    ow, oh = img.size
    scale = min(res / ow, res / oh)
    nw, nh = max(1, int(round(ow * scale))), max(1, int(round(oh * scale)))
    resized = img.resize((nw, nh), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (res, res), (fill, fill, fill))
    ox, oy = (res - nw) // 2, (res - nh) // 2
    canvas.paste(resized, (ox, oy))
    return canvas, {
        "scale": scale, "ox": ox, "oy": oy, "cw": nw, "ch": nh,
        "orig_w": ow, "orig_h": oh, "res": res,
    }


def _crop_content(map2d: np.ndarray, geom: Dict) -> np.ndarray:
    """裁掉画布尺寸 2D 图（异常图/注意力图）上的灰边，只保留内容区。

    map2d 不一定是 res×res（可能处于 patch 低分辨率），按画布坐标等比换算。
    """
    h, w = map2d.shape[:2]
    sx, sy = w / float(geom["res"]), h / float(geom["res"])
    x0, y0 = int(round(geom["ox"] * sx)), int(round(geom["oy"] * sy))
    cw, ch = max(1, int(round(geom["cw"] * sx))), max(1, int(round(geom["ch"] * sy)))
    return map2d[y0:y0 + ch, x0:x0 + cw]


def _content_to_original(map2d: np.ndarray, geom: Dict) -> np.ndarray:
    """把画布尺寸的 2D 图裁掉灰边，并缩放回原始图片尺寸。"""
    cropped = _crop_content(map2d, geom)
    return cv2.resize(
        cropped.astype(np.float32),
        (int(geom["orig_w"]), int(geom["orig_h"])),
        interpolation=cv2.INTER_LINEAR,
    )


class SubspaceAnomalyDetector:
    """
    基于 DINOv2 + 记忆库的少样本异常检测器。

    特性:
        - Training-Free：直接构建正常图像特征记忆库，无需训练
        - CLS-Patch 显著性：利用 CLS token 与 patch 特征相似度定位异常
        - 多层特征融合：支持 score_avg / score_max / feature_avg / feature_concat
        - 双聚合模式：max 或 top1_mean 相似度聚合
        - 内建目标定位：attention_map 可直接用于 ROI 定位

    示例:
        >>> detector = SubspaceAnomalyDetector()
        >>> detector.train(template_images=["normal_1.jpg", "normal_2.jpg"])
        >>> results = detector.detect(test_images=["test_1.jpg", "test_2.jpg"])
        >>> for result in results:
        ...     print(f"{result['image_name']}: {result['anomaly_score']:.4f}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        image_res: int = 448,
        device: Optional[str] = None,
        use_clahe: bool = False,
        # SubspaceAD-specific parameters
        similarity_aggregation: str = "max",
        layer_fusion: str = "score_avg",
        layers: Tuple[int, ...] = (8, 10, 12),
        # Performance
        use_fp16: bool = True,
        # Localization parameters (preserved)
        enable_localization: bool = True,
        localization_method: str = "auto",
        crop_to_roi: bool = True,
        roi_margin: float = 0.10,
        # LayoutAD double-check (optional)
        enable_layoutad: bool = True,
        layoutad_checkpoint: Optional[str] = None,
        layoutad_mask2former_config: Optional[str] = None,
        layoutad_mask2former_weights: Optional[str] = None,
        # PatchCore-style memory-bank coreset + weighted k-NN (default: off)
        coreset_ratio: float = 0.0,   # 0.0 = 不采样; 0.05 = 保留 5% 最远点 coreset
        coreset_seed: int = 42,
        knn_k: int = 9,
        knn_temperature: float = 1.0,
    ):
        """
        Args:
            model_path: DINOv2 模型路径或 HuggingFace model id
            image_res: 输入图像分辨率（正方形）
            device: 计算设备 ("cuda"/"cpu")，默认自动选择
            use_clahe: 是否使用 CLAHE 增强
            similarity_aggregation: 相似度聚合方法 ("max"/"top1_mean")
            layer_fusion: 多层融合方法 ("score_avg"/"score_max"/"feature_avg"/"feature_concat")
            layers: 使用的层索引，如 (8, 10, 12)
            use_fp16: GPU 上使用 FP16 半精度（速度翻倍，显存减半）
            enable_localization: 是否启用目标定位
            localization_method: 定位策略 ("auto"/"saliency"/"contour"/"none")
            crop_to_roi: 定位后是否裁切 ROI 检测
            roi_margin: ROI 扩展边距比例
            enable_layoutad: 是否启用 LayoutAD GNN 结构 double-check
            layoutad_checkpoint: LayoutAD 模型权重路径
            coreset_ratio: PatchCore 式记忆库 coreset 比例 (0.0=关闭)
            coreset_seed: coreset 采样种子
            knn_k: knn_weighted 聚合的近邻数
            knn_temperature: knn_weighted 逆距离加权温度
        """
        if not _HAS_PIPELINE:
            raise ImportError(
                "SubspaceAD 依赖库 (ad-pipelines) 未安装。\n"
                "本地: pip install ad-pipelines\n"
                "Docker: 自动安装（见 Dockerfile）\n"
                f"详细错误: {_MISSING_PIPELINE_MSG}"
            )

        # Hardware
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._device_torch = torch.device(self.device)
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self._dtype = torch.float16 if self.use_fp16 else torch.float32

        logger.info(
            "Device: %s | Precision: %s",
            self.device, "FP16" if self.use_fp16 else "FP32",
        )

        # Model & Pipeline
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.image_res = image_res
        self.use_clahe = use_clahe
        self.similarity_aggregation = similarity_aggregation
        self.layer_fusion = layer_fusion
        self.layers = tuple(layers)

        self.coreset_ratio = coreset_ratio
        self.coreset_seed = coreset_seed
        self.knn_k = max(1, knn_k)
        self.knn_temperature = knn_temperature

        self.model = None
        self.pipeline: Optional[DuoADPipeline] = None
        self.prompt_features: Optional[torch.Tensor] = None  # memory bank
        self.is_trained = False
        # 分数口径：内部始终用「原始异常图 top-k 均值」（自匹配≈0，缺陷≈0.1+），
        # 对外通过 normalize_score() 锚定映射到 [0,1]。
        # 默认阈值 0.5 是归一化分数口径；校准会用正常图更新 base/scale 使其有意义。
        self.threshold: float = 0.5
        # 阈值标定状态：/api/calibrate 成功后置 True；reset 重建检测器后回到 False
        self.is_calibrated: bool = False

        # 归一化锚点：score_base=典型正常原始分，score_scale=e-folding 标度。
        # 未校准兜底 base=0、scale=0.10：以示例缺陷图为准（test-1 raw≈0.14、test-2 raw≈0.19），
        # raw 0.14/0.19 → 0.75/0.85，与默认阈值 0.5 拉开明显距离；模板自匹配 raw≈0 → ≈0。
        # 代价：scale 越小越陡峭，正常新图若 raw≈0.05-0.08 会映射到 0.39-0.55，
        # 产线正常图建议跑 /api/calibrate 用真实基线覆盖这两者。
        self.score_base: float = 0.0
        self.score_scale: float = 0.10

        # Localization
        self.enable_localization = enable_localization
        self.localization_method = localization_method
        self.crop_to_roi = crop_to_roi
        self.roi_margin = roi_margin
        self._localizer = ObjectLocalizer(default_method=localization_method)

        # LayoutAD double-check (GNN-based structural verification)
        self.enable_layoutad = enable_layoutad and _LAYOUTAD_AVAILABLE
        self._layoutad_engine: Optional[LayoutADInference] = None
        if self.enable_layoutad:
            try:
                self._layoutad_engine = LayoutADInference(
                    checkpoint_path=layoutad_checkpoint,
                    device=self.device,
                    mask2former_config=layoutad_mask2former_config,
                    mask2former_weights=layoutad_mask2former_weights,
                )
                if not self._layoutad_engine.available:
                    logger.warning(
                        "LayoutAD 已启用但模型权重未找到，double-check 将自动跳过。"
                    )
            except Exception as e:
                logger.warning("LayoutAD 初始化失败: %s，将跳过 double-check。", e)
                self._layoutad_engine = None

        # Training-free graph structure checker (always available if import succeeded)
        if GraphStructureChecker is not None:
            self._graph_checker = GraphStructureChecker()
        else:
            self._graph_checker = None
        self._graph_ref_built = False

        # Lazy init
        layout_status = "ON" if (self._layoutad_engine and self._layoutad_engine.available) else "OFF"
        coreset_status = "OFF" if (self.coreset_ratio <= 0.0 or self.coreset_ratio >= 1.0) \
            else f"ON(β={self.coreset_ratio})"
        logger.info(
            "SubspaceAnomalyDetector ready (lazy load).\n"
            "  model=%s  resolution=%d  sim=%s  fusion=%s  layers=%s  layoutad=%s  graph_check=ON"
            "  coreset=%s  knn_weighted(k=%d,T=%s)",
            self.model_path, self.image_res,
            self.similarity_aggregation, self.layer_fusion, self.layers,
            layout_status, coreset_status, self.knn_k, self.knn_temperature,
        )

    # ============================================================
    # Model State
    # ============================================================

    @property
    def is_loaded(self) -> bool:
        """模型权重是否已实际加载（pipeline 就绪）。

        懒加载模式下首次 train() 后才为 True，反映真实加载状态。
        """
        return self.pipeline is not None

    def weights_ready(self) -> bool:
        """本地目录模式下，权重文件是否已就绪（启动时的防御性检查）。

        返回 False 说明入口脚本/模型 volume 未提供权重文件，服务无法正常工作，
        此时应在 lifespan 中禁用检测器，让 /mse/health 如实报告 down。
        """
        path = self.model_path
        if os.path.isdir(path):
            return any(
                os.path.isfile(os.path.join(path, name))
                for name in ("model.safetensors", "pytorch_model.bin")
            )
        # HF model id 模式下由加载器负责下载，视为就绪
        return True

    # ============================================================
    # Lazy Model Loading
    # ============================================================

    def _pipeline_cfg_matches(self) -> bool:
        """已存在的 pipeline 是否与当前训练参数一致（用于 F2：重训参数变化时重建）。"""
        if self.pipeline is None:
            return False
        return (
            self.pipeline.similarity_aggregation == self.similarity_aggregation
            and self.pipeline.layer_fusion_method == self.layer_fusion
            and self.pipeline.coreset_ratio == self.coreset_ratio
            and self.pipeline.coreset_seed == self.coreset_seed
            and self.pipeline.knn_k == self.knn_k
            and self.pipeline.knn_temperature == self.knn_temperature
        )

    def _ensure_model(self):
        """Lazy-load model + pipeline with full optimizations.

        训练参数（相似度聚合/融合/coreset/k-NN）变化时只重建 pipeline 包装，
        复用已加载的模型权重（不重载 346MB）。参数变化会使旧记忆库失效，
        需重新 train 构建。
        """
        if self.pipeline is not None and self._pipeline_cfg_matches():
            return

        model_was_loaded = self.model is not None
        if not model_was_loaded:
            # Common kwargs: FP16 half-precision for GPU speed
            model_kwargs = {
                "device": self._device_torch,
                "dtype": self._dtype,
                "resolution": self.image_res,
            }

            logger.info("Loading DINOv2 with registers: %s (FP16=%s)...", self.model_path, self.use_fp16)
            self.model = DinoV2WithRegisterModel(self.model_path, **model_kwargs)
        else:
            logger.info("训练参数变更 — 重建 DuoAD pipeline（复用已加载模型 %s）", self.model_path)

        logger.info(
            "Creating SubspaceAD pipeline (res=%d, sim=%s, fusion=%s, layers=%s, coreset=%s, knn_weighted(k=%d))...",
            self.image_res, self.similarity_aggregation,
            self.layer_fusion, self.layers, self.coreset_ratio, self.knn_k,
        )
        self.pipeline = DuoADPipeline(
            model=self.model,
            resolution=self.image_res,
            similarity_aggregation=self.similarity_aggregation,
            layer_fusion_method=self.layer_fusion,
            coreset_ratio=self.coreset_ratio,
            coreset_seed=self.coreset_seed,
            knn_k=self.knn_k,
            knn_temperature=self.knn_temperature,
            device=self._device_torch,
            dtype=self._dtype,
        )
        self.pipeline.model.eval()

        # 参数变化导致 pipeline 重建 → 旧记忆库已不匹配，作废并要求重新 train
        if self.prompt_features is not None:
            self.prompt_features = None
            self.is_trained = False
            logger.info("记忆库已作废（pipeline 参数变化）— 请重新 train 构建。")

        if model_was_loaded:
            return  # 模型未重新加载，无需再 warmup

        # Warmup: run a dummy forward pass so first real request is instant
        logger.info("Running model warmup...")
        t0 = __import__("time").time()
        dummy = Image.new("RGB", (self.image_res, self.image_res), color=(128, 128, 128))
        with torch.no_grad():
            _ = self.pipeline.model.get_features(
                self.pipeline.model.preprocess([dummy], resolution=self.image_res).to(
                    device=self._device_torch, dtype=self._dtype,
                ),
                return_attentions=False,
                output_feature_maps_indices=(-1,),
            )
        logger.info("Warmup complete (%.1fs) — model ready.", __import__("time").time() - t0)

    # ============================================================
    # Training (build memory bank)
    # ============================================================

    def train(
        self,
        template_images: Union[List[str], List[Image.Image]],
        verbose: bool = True,
    ) -> Dict:
        """
        使用正常图像构建 SubspaceAD 特征记忆库（Training-Free，仅存储特征）。

        Args:
            template_images: 正常图像路径列表或 PIL Image 对象列表
            verbose: 是否打印日志

        Returns:
            训练信息字典
        """
        self._ensure_model()

        if verbose:
            logger.info("=" * 60)
            logger.info("🔧 Building SubspaceAD Memory Bank")
            logger.info("=" * 60)

        # Prepare images
        if isinstance(template_images[0], str):
            imgs = [Image.open(p).convert("RGB") for p in template_images]
        else:
            imgs = template_images

        # 尺寸适配：保比例缩放 + 灰边填充为方形画布，适配任意长宽比图片
        imgs = [_letterbox(i, self.image_res)[0] for i in imgs]

        if verbose:
            logger.info("Loaded %d template images (letterboxed to %d)", len(imgs), self.image_res)

        # Build prompt features
        t0 = __import__("time").time()
        with torch.no_grad():
            self.prompt_features = self.pipeline.get_prompt_features(
                prompt_images=imgs,
                resolution=self.image_res,
                output_feature_maps_indices=self.layers,
                layer_fusion_method=self.layer_fusion,
            )
        elapsed = __import__("time").time() - t0

        self.is_trained = True
        self._graph_ref_built = False  # Reset — will build on first detect

        # 重训 = 新记忆库 → 旧标定锚点全部失效，复位到默认。
        # 否则旧 base/scale/threshold 会继续作用在新库的分数上（曾出现"重训后分数对不上"）。
        # 复位后必须重新跑 /api/calibrate 才有有意义的锚点。
        self.is_calibrated = False
        self.score_base = 0.0
        self.score_scale = 0.10
        self.threshold = 0.5

        # Build graph reference from template for structural double-check
        if self._graph_checker is not None:
            try:
                # Run a quick detection on templates to get anomaly maps for graph building
                template_anomaly_maps = []
                for img in imgs:
                    with torch.no_grad():
                        temp_output: PatchEADOutput = self.pipeline(
                            prompt_images=self.prompt_features,
                            test_images=[img],
                            is_prompt_features=True,
                            resolution=self.image_res,
                            return_attentioned_anomaly_map=False,
                            upsample_anomaly_map=True,
                            upsample_resolution=img.size,
                            output_feature_maps_indices=self.layers,
                            layer_fusion_method=self.layer_fusion,
                            anomaly_score_method="top1",
                        )
                    template_anomaly_maps.append(temp_output.anomaly_map.squeeze().numpy())
                self._graph_checker.build_reference(imgs, template_anomaly_maps)
                self._graph_ref_built = self._graph_checker.is_built
                if self._graph_ref_built:
                    logger.info("Graph structure reference built for double-check.")
            except Exception as e:
                logger.warning("Graph reference build skipped: %s", e)

        # prompt_features may be a single tensor or a tuple (score-based fusion)
        if isinstance(self.prompt_features, tuple):
            first = self.prompt_features[0]
        else:
            first = self.prompt_features

        n_patches, feat_dim = first.shape

        if verbose:
            n_layers = len(self.prompt_features) if isinstance(self.prompt_features, tuple) else 1
            logger.info("  Memory bank: %d patches × %d dims × %d layers", n_patches, feat_dim, n_layers)
            logger.info("  Build time: %.2f s", elapsed)
            logger.info("✅ Memory bank ready!")
            logger.info("=" * 60)

        return {
            "num_patches": n_patches,
            "feature_dim": feat_dim,
            "num_templates": len(imgs),
            "build_time_ms": round(elapsed * 1000, 1),
        }

    # ============================================================
    # Detection
    # ============================================================

    def detect(
        self,
        test_images: Union[List[str], List[Image.Image]],
        save_dir: Optional[str] = None,
        save_visualizations: bool = False,
        viz_mode: str = "overlay",
        bbox_threshold: float = 0.5,
        top_k_ratio: float = 0.01,
        verbose: bool = True,
        enable_localization: Optional[bool] = None,
        localization_method: Optional[str] = None,
        crop_to_roi: Optional[bool] = None,
        roi_margin: Optional[float] = None,
    ) -> List[Dict]:
        """
        对测试图像进行异常检测（基于 SubspaceAD 记忆库余弦相似度）。

        Args:
            test_images: 测试图像路径列表或 PIL Image 对象列表
            save_dir: 结果保存目录
            save_visualizations: 是否保存可视化结果
            viz_mode: "overlay" / "side_by_side" / "bbox"
            bbox_threshold: 缺陷检测阈值
            top_k_ratio: 图像级分数的 top-k 比例
            verbose: 是否打印日志
            enable_localization: 是否启用目标定位
            localization_method: 定位策略
            crop_to_roi: 是否裁切 ROI 检测
            roi_margin: ROI 扩展边距比例

        Returns:
            检测结果列表
        """
        if not self.is_trained or self.prompt_features is None:
            raise RuntimeError("请先调用 train() 方法构建记忆库")

        self._ensure_model()

        do_localize = enable_localization if enable_localization is not None else self.enable_localization
        loc_method = localization_method or self.localization_method
        do_crop = crop_to_roi if crop_to_roi is not None else self.crop_to_roi
        loc_margin = roi_margin if roi_margin is not None else self.roi_margin

        if verbose:
            logger.info("\n🔍 Detecting anomalies (SubspaceAD)...")

        if save_dir and save_visualizations:
            os.makedirs(save_dir, exist_ok=True)

        # Load images
        if isinstance(test_images[0], str):
            imgs = [Image.open(p).convert("RGB") for p in test_images]
            img_names = [Path(p).stem for p in test_images]
        else:
            imgs = test_images
            img_names = [f"image_{i}" for i in range(len(test_images))]

        results = []

        for i, test_img in enumerate(imgs):
            test_name = img_names[i]
            localization_info = None
            pipeline_steps = []
            img_w, img_h = test_img.size  # 原始尺寸（用于输出对齐）

            # ── Step 0: 图像输入（读图 + letterbox）──
            t0 = time.time()
            canvas, geom = _letterbox(test_img, self.image_res)
            canvas_w, canvas_h = canvas.size
            pipeline_steps.append({
                "key": "input", "label": "图像输入",
                "status": "done", "ms": round((time.time() - t0) * 1000, 1),
                "detail": f"{img_w}×{img_h}",
            })

            # ── Step 1: Run SubspaceAD inference (on letterboxed canvas) ──
            t0 = time.time()
            try:
                with torch.no_grad():
                    output: PatchEADOutput = self.pipeline(
                        prompt_images=self.prompt_features,
                        test_images=[canvas],
                        is_prompt_features=True,
                        resolution=self.image_res,
                        # 必须 True：产出 attention_map（CLS-patch 显著性）供目标定位/ROI 裁切使用。
                        # 开关不影响 anomaly_map/anomaly_score 语义（见 _calculate_anomaly_score），仅多算显著性。
                        return_attentioned_anomaly_map=True,
                        upsample_anomaly_map=True,
                        upsample_resolution=(canvas_w, canvas_h),
                        output_feature_maps_indices=self.layers,
                        layer_fusion_method=self.layer_fusion,
                        anomaly_score_method="top1",
                    )
            except Exception as e:
                logger.error("SubspaceAD pipeline inference failed:\n%s", traceback.format_exc())
                raise RuntimeError(
                    f"SubspaceAD 推理失败: {e}\n"
                    f"Params: resolution={self.image_res}, layers={self.layers}, "
                    f"fusion={self.layer_fusion}, sim_agg={self.similarity_aggregation}"
                ) from e

            # Extract outputs
            anomaly_map_full = output.anomaly_map.squeeze().numpy()  # [H, W]
            anomaly_score_raw = float(output.anomaly_score.item())
            step1_ms = round((time.time() - t0) * 1000, 1)  # 全图推理耗时

            # Attention map (CLS-patch saliency — already at image resolution)
            if output.attention_map is not None:
                attention_map = output.attention_map.squeeze().numpy()  # [H, W]
            else:
                attention_map = None

            # ── Step 2: Object localization using SubspaceAD attention map ──
            t_localize = time.time()
            crop_ms = 0.0  # 未定位时无 ROI 复检
            if do_localize and loc_method != "none" and attention_map is not None:
                loc_result = self._localizer.localize(
                    canvas,
                    saliency_map=attention_map,
                    method=loc_method,
                )
                # bbox 从画布坐标换算回原始图片坐标（灰边偏移 + 缩放）
                bx0 = (loc_result.bbox[0] - geom["ox"]) / geom["scale"]
                by0 = (loc_result.bbox[1] - geom["oy"]) / geom["scale"]
                bx1 = bx0 + loc_result.bbox[2] / geom["scale"]
                by1 = by0 + loc_result.bbox[3] / geom["scale"]
                # 裁到图片边界内：显著性可能漂到 letterbox 灰边，导致负数/越界坐标
                bx0 = max(0.0, min(bx0, float(img_w)))
                by0 = max(0.0, min(by0, float(img_h)))
                bx1 = max(bx0, min(bx1, float(img_w)))
                by1 = max(by0, min(by1, float(img_h)))
                localization_info = {
                    "bbox": [
                        round(bx0, 2),
                        round(by0, 2),
                        round(bx1 - bx0, 2),
                        round(by1 - by0, 2),
                    ],
                    "confidence": round(loc_result.confidence, 4),
                    "method": loc_result.method,
                }

                # ── Step 3: Crop ROI and re-run if requested ──
                crop_start = time.time()
                if do_crop and loc_result.is_valid and loc_result.method not in (
                    "none", "saliency_fallback", "contour_fallback",
                ):
                    roi_img, roi_bbox = crop_to_roi_fn(canvas, loc_result.bbox, margin=loc_margin)
                    square_bbox = expand_bbox_square(roi_bbox, canvas.size)
                    roi_square = canvas.crop((
                        square_bbox[0], square_bbox[1],
                        square_bbox[0] + square_bbox[2], square_bbox[1] + square_bbox[3],
                    ))

                    with torch.no_grad():
                        roi_output: PatchEADOutput = self.pipeline(
                            prompt_images=self.prompt_features,
                            test_images=[roi_square],
                            is_prompt_features=True,
                            resolution=self.image_res,
                            return_attentioned_anomaly_map=False,
                            upsample_anomaly_map=True,
                            upsample_resolution=(
                                square_bbox[2], square_bbox[3],
                            ),
                            output_feature_maps_indices=self.layers,
                            layer_fusion_method=self.layer_fusion,
                            anomaly_score_method="top1",
                        )

                    roi_anomaly_map = roi_output.anomaly_map.squeeze().numpy()
                    anomaly_map_full = map_anomaly_to_original(
                        roi_anomaly_map, square_bbox, (canvas_h, canvas_w),
                    )
                    anomaly_map_full = cv2.resize(
                        anomaly_map_full.astype(np.float32),
                        (canvas_w, canvas_h),
                        interpolation=cv2.INTER_LINEAR,
                    )

                crop_ms = round((time.time() - crop_start) * 1000, 1)

            # localize = 全图推理(产 saliency/attention) + 定位计算；crop 计时单独扣出
            localize_ms = round(step1_ms + max(0.0, (time.time() - t_localize) * 1000 - crop_ms), 1)

            # ── Step 4: 热力图叠加（后处理：裁边、缩放回原图、归一化）──
            t0 = time.time()
            anomaly_map_full = _content_to_original(anomaly_map_full, geom)
            if attention_map is not None:
                attention_map = _content_to_original(attention_map, geom)

            # 分数必须在「原始异常图」上计算，不能取 min-max 归一化后的图：
            # min_max 会强制全局最大值=1.0，把自匹配/正常图的分数也抬到 0.5+，
            # 使阈值（0.3 等）完全失真——所有图都被判为异常。
            # 归一化图仅用于热力图显示，评分保持绝对口径。
            anomaly_map_normalized = min_max_norm(anomaly_map_full)
            heatmap_ms = round((time.time() - t0) * 1000, 1)

            # ── Step 5: 评定分（目标区域 top-k）──
            # 分数限定在目标区域（bbox）内：背景不计入 top-k，避免背景噪点稀释/干扰缺陷分。
            # localization_info 与 anomaly_map_full 均已是原图像素坐标，空间一致。
            t0 = time.time()
            if localization_info and localization_info["bbox"]:
                bx, by, bw, bh = [int(round(v)) for v in localization_info["bbox"]]
                h, w = anomaly_map_full.shape
                bx = max(0, min(bx, w)); by = max(0, min(by, h))
                bw = max(1, min(bw, w - bx)); bh = max(1, min(bh, h - by))
                flat_scores = anomaly_map_full[by:by + bh, bx:bx + bw].flatten()
            else:
                flat_scores = anomaly_map_full.flatten()  # 无定位/定位失败 → 全图兜底
            k = max(1, int(len(flat_scores) * top_k_ratio))
            top_k_mean = np.mean(np.sort(flat_scores)[-k:])
            score_ms = round((time.time() - t0) * 1000, 1)

            # ── Step 6: 结果输出（组装 result + 处理链）──
            t0 = time.time()
            result = {
                "image_name": test_name,
                # 内部口径保留原始 top-k 均值（供校准计算锚点），对外分数归一化到 [0,1]
                "anomaly_score_raw": float(top_k_mean),
                "anomaly_score": self.normalize_score(float(top_k_mean)),
                "anomaly_map": anomaly_map_normalized,
                "attention_map": attention_map,
            }
            if localization_info:
                result["localization"] = localization_info

            # 处理链（与用户规范一一对应：图像→定位→切割→检测→热力图→评定分→输出）
            did_crop = crop_ms > 0
            pipeline_steps.extend([
                {
                    "key": "localize", "label": "目标定位",
                    "status": "done" if localization_info else "skipped",
                    "ms": localize_ms,
                    "detail": (
                        f"{localization_info['method']} {localization_info['confidence'] * 100:.0f}%"
                        if localization_info else "未启用"
                    ),
                },
                {
                    "key": "crop", "label": "区域切割",
                    "status": "done" if did_crop else "skipped",
                    "ms": crop_ms,
                    "detail": "bbox 复检" if did_crop else "未启用",
                },
                {
                    "key": "detect", "label": "缺陷检测",
                    "status": "done", "ms": crop_ms if did_crop else step1_ms,
                    "detail": "ROI 复检" if did_crop else "全图推理",
                },
                {
                    "key": "heatmap", "label": "热力图叠加",
                    "status": "done", "ms": heatmap_ms, "detail": "已叠加",
                },
                {
                    "key": "score", "label": "评定分",
                    "status": "done", "ms": score_ms,
                    "detail": f"{self.normalize_score(float(top_k_mean)):.4f}",
                },
                {
                    "key": "output", "label": "结果输出",
                    "status": "done", "ms": round((time.time() - t0) * 1000, 1),
                    "detail": (
                        "异常" if self.normalize_score(float(top_k_mean)) > self.threshold else "正常"
                    ),
                },
            ])
            result["pipeline"] = pipeline_steps
            result["pipeline_ms"] = round((time.time() - t0) * 1000, 1)

            # ── Graph structure double-check (training-free) ──
            if self._graph_ref_built and self._graph_checker is not None:
                try:
                    graph_result = self._graph_checker.check(
                        anomaly_map=anomaly_map_normalized,
                        img_size=(img_w, img_h),
                    )
                    result["graph_check"] = {
                        "available": graph_result["available"],
                        "structural_score": graph_result["structural_score"],
                        "node_count": graph_result["node_count"],
                    }

                    if graph_result["available"]:
                        fused = self._graph_checker.fuse_scores(
                            subspace_score=float(top_k_mean),
                            graph_result=graph_result,
                        )
                        result["fused_score"] = self.normalize_score(fused["final_score"])
                        result["fusion_method"] = "graph_structure"
                        result["structural_score"] = fused["structural_score"]
                        if verbose:
                            logger.info(
                                "  Graph check: structural=%.4f fused=%.4f nodes=%d",
                                fused["structural_score"], fused["final_score"],
                                graph_result["node_count"],
                            )
                except Exception as e:
                    logger.warning("Graph check skipped: %s", e)

            # ── LayoutAD double-check (optional, requires trained weights) ──
            if self._layoutad_engine is not None and self._layoutad_engine.available:
                try:
                    layout_result = self._layoutad_engine.check(
                        image=test_img,
                        dino_features=torch.zeros(0),
                        attention_map=attention_map,
                        anomaly_map=anomaly_map_normalized,
                    )
                    result["layoutad"] = {
                        "available": layout_result["available"],
                        "layout_score": layout_result["layout_score"],
                        "node_count": layout_result["node_count"],
                        "graph_built": layout_result["graph_built"],
                    }

                    if layout_result["available"] and layout_result["graph_built"]:
                        fused = self._layoutad_engine.fuse_scores(
                            subspace_score=float(top_k_mean),
                            layout_result=layout_result,
                        )
                        # LayoutAD override if available (prioritize over graph check)
                        result["fused_score"] = self.normalize_score(fused["final_score"])
                        result["fusion_method"] = "layoutad_gnn"
                        result["layout_score"] = fused["layout_score"]
                        if verbose:
                            logger.info(
                                "  LayoutAD double-check: layout=%.4f fused=%.4f nodes=%d",
                                fused["layout_score"], fused["final_score"],
                                layout_result["node_count"],
                            )
                except Exception as e:
                    logger.warning("LayoutAD double-check skipped: %s", e)

            # Save visualizations
            if save_dir and save_visualizations:
                try:
                    from .subspacead.utils.viz import save_custom_visualization
                    viz_path = save_custom_visualization(
                        path=test_images[i] if isinstance(test_images[i], str) else f"image_{i}",
                        img=test_img,
                        anom_map=anomaly_map_normalized,
                        outdir=save_dir,
                        viz_mode=viz_mode,
                        score=float(top_k_mean),
                        bbox_threshold=bbox_threshold,
                    )
                    result["viz_path"] = viz_path
                except Exception as e:
                    logger.warning("Failed to save visualization: %s", e)

            results.append(result)

            if verbose:
                status = "⚠️  ANOMALY" if result["anomaly_score"] > self.threshold else "✅ Normal"
                loc_str = (
                    f" | ROI: {localization_info['bbox']}"
                    if localization_info else ""
                )
                logger.info(
                    "[%d/%d] %s: Score=%.4f (%s)%s",
                    i + 1, len(imgs), test_name,
                    result["anomaly_score"], status, loc_str,
                )

        if verbose:
            logger.info("=" * 60)
            logger.info("📊 Detection Summary (SubspaceAD):")
            for r in results:
                s = r["anomaly_score"]
                st = "⚠️ ANOMALY" if s > self.threshold else "✅ Normal"
                logger.info("%-20s | Score: %.4f | %s", r["image_name"], s, st)
            logger.info("=" * 60)

        return results

    # ============================================================
    # Single Image Detection
    # ============================================================

    def detect_single(
        self,
        image: Union[str, Image.Image],
        return_heatmap: bool = False,
        verbose: bool = True,
        enable_localization: Optional[bool] = None,
        localization_method: Optional[str] = None,
        crop_to_roi: Optional[bool] = None,
    ) -> Union[float, Tuple[float, np.ndarray]]:
        """单张图像快速检测。"""
        if isinstance(image, str):
            imgs = [Image.open(image).convert("RGB")]
        else:
            imgs = [image]

        results = self.detect(
            imgs, verbose=verbose,
            enable_localization=enable_localization,
            localization_method=localization_method,
            crop_to_roi=crop_to_roi,
        )
        result = results[0]

        if return_heatmap:
            return result["anomaly_score"], result["anomaly_map"]
        return result["anomaly_score"]

    # ============================================================
    # Score Normalization
    # ============================================================
    def normalize_score(self, raw: float) -> float:
        """把原始 top-k 分数锚定映射到 [0,1]，供对外展示与判定。

        映射:  raw = score_base        → 0
               raw = score_base+scale  → 1 - 1/e ≈ 0.63
               raw = score_base+3scale → ≈ 0.95
        score_base/score_scale 由校准（/api/calibrate）用正常图的原始分设定，
        未校准时是保守兜底值。映射单调有界，不会把正常图顶到 1.0。
        """
        d = max(0.0, float(raw) - self.score_base)
        return float(np.clip(1.0 - math.exp(-d / max(self.score_scale, 1e-6)), 0.0, 1.0))

    def set_threshold(self, threshold: float = 0.5):
        self.threshold = threshold
        logger.info("Anomaly threshold set to: %s", threshold)

    def is_anomaly(self, score: float) -> bool:
        return score > self.threshold

    # ============================================================
    # Video Detection
    # ============================================================

    def detect_video(
        self,
        video_path: str,
        sample_every_n_frames: int = 10,
        max_frames: int = 200,
        verbose: bool = True,
    ) -> List[Dict]:
        """Analyze a video file frame-by-frame for anomalies.

        Extracts frames at regular intervals, runs SubspaceAD detection on each,
        and returns per-frame scores plus a summary.

        Args:
            video_path: Path to video file
            sample_every_n_frames: Process every Nth frame (default 10)
            max_frames: Maximum number of frames to process
            verbose: Print progress

        Returns:
            List of per-frame result dicts with: frame_idx, timestamp_sec,
            anomaly_score, is_anomaly. Also includes a "_summary" entry at the end.
        """
        if not self.is_trained:
            raise RuntimeError("请先调用 train() 方法构建记忆库")
        self._ensure_model()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if verbose:
            logger.info(
                "🎬 Video: %d frames @ %.1f fps, sampling every %d frames (max %d)",
                total_frames, fps, sample_every_n_frames, max_frames,
            )

        results = []
        frame_idx = 0
        processed = 0

        while processed < max_frames:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_idx += 1

            if frame_idx % sample_every_n_frames != 0:
                continue

            # Convert BGR → PIL RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            timestamp = frame_idx / fps

            try:
                det_results = self.detect(
                    [pil_img],
                    enable_localization=False,
                    crop_to_roi=False,
                    verbose=False,
                )
                r = det_results[0]
            except Exception as e:
                logger.warning("Frame %d failed: %s", frame_idx, e)
                continue

            results.append({
                "frame_idx": frame_idx,
                "timestamp_sec": round(timestamp, 2),
                "anomaly_score": r["anomaly_score"],
                "is_anomaly": r["anomaly_score"] > self.threshold,
            })
            processed += 1

            if verbose and processed % 20 == 0:
                logger.info("  Processed %d frames...", processed)

        cap.release()

        if not results:
            return results

        scores = [r["anomaly_score"] for r in results]
        max_score = max(scores)
        max_frame = results[scores.index(max_score)]
        anomaly_frames = [r for r in results if r["is_anomaly"]]

        summary = {
            "_summary": True,
            "total_frames_scanned": len(results),
            "max_anomaly_score": max_score,
            "max_anomaly_frame": max_frame["frame_idx"],
            "max_anomaly_timestamp": max_frame["timestamp_sec"],
            "anomaly_frame_count": len(anomaly_frames),
            "anomaly_ratio": round(len(anomaly_frames) / len(results), 4),
            "mean_score": round(sum(scores) / len(scores), 6),
            "fps": fps,
            "sample_interval": sample_every_n_frames,
        }

        results.append(summary)

        if verbose:
            logger.info(
                "📊 Video summary: %d frames | max score %.4f (frame %d @ %.1fs) | %d/%d anomalous",
                len(results) - 1, max_score, max_frame["frame_idx"],
                max_frame["timestamp_sec"], len(anomaly_frames), len(results) - 1,
            )

        return results


# ============================================================
# Convenience Function
# ============================================================

def create_detector(
    model_path: Optional[str] = None,
    resolution: int = 448,
    **kwargs,
) -> SubspaceAnomalyDetector:
    """快速创建检测器实例。"""
    return SubspaceAnomalyDetector(
        model_path=model_path,
        image_res=resolution,
        **kwargs,
    )
