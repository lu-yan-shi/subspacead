"""Business API routes — SubspaceAD anomaly detection endpoints."""

import asyncio
import base64
import io
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from ..config import (
    AVAILABLE_MODELS,
    BUSINESS_PREFIX,
    DEFAULT_CROP_TO_ROI,
    DEFAULT_IMAGE_RES,
    DEFAULT_LOCALIZATION_METHOD,
    DEFAULT_LOCALIZE,
    SUBSPACE_CORESET_RATIO,
    SUBSPACE_CORESET_SEED,
    SUBSPACE_KNN_K,
    SUBSPACE_KNN_TEMPERATURE,
    SUBSPACE_LAYER_FUSION,
    SUBSPACE_LAYERS,
    SUBSPACE_SIMILARITY_AGGREGATION,
    MAX_FILE_SIZE_MB,
    ROI_MARGIN_RATIO,
    SERVICE_NAME,
    SERVICE_VERSION,
)
try:
    from models.detector import SubspaceAnomalyDetector
except ImportError:
    SubspaceAnomalyDetector = None  # type: ignore
from models.subspacead.utils.viz import (
    create_heatmap,
    ensure_rgb,
    find_defect_bbox,
    render_visualization,
)

business_router = APIRouter(prefix=BUSINESS_PREFIX, tags=["Business"])
logger = logging.getLogger(__name__)

# 检测器访问串行化：单模型、单记忆库，train/detect/reset 本就该互斥执行。
# 推理是同步 CPU/GPU 密集操作，用 asyncio.to_thread 挪出事件循环，
# 保证推理期间 /mse/health、/api/status 仍可响应（否则整个服务卡死数秒）。
_detector_lock = asyncio.Lock()

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}


# ============================================================
# Image Helpers
# ============================================================
def _img_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _numpy_to_base64(img_np: np.ndarray) -> str:
    if img_np.shape[-1] == 3:
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_np
    return _img_to_base64(Image.fromarray(img_rgb))


# ============================================================
# Validation Helpers
# ============================================================
def _validate_image(filename: str, contents: bytes) -> str:
    ext = Path(filename).suffix.lower().lstrip(".") if filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式: {ext}。支持的格式: {', '.join(ALLOWED_EXTENSIONS)}")
    max_bytes = int(MAX_FILE_SIZE_MB * 1024 * 1024)
    if len(contents) > max_bytes:
        raise HTTPException(400, f"文件大小超过限制 ({MAX_FILE_SIZE_MB}MB)")
    return ext


def _bytes_to_pil(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"图片解码失败: {str(e)}")


# ============================================================
# Business Endpoints
# ============================================================

def _is_hf_id(path: str) -> bool:
    """判断路径是否为 Hugging Face model id（形如 org/repo）。

    本地目录相对路径（如 `dinov3-vitb16-pretrain-lvd1689m`）不含 `/` 或以 `/` 开头，
    是绝对/相对本地路径而非 HF id。
    """
    return "/" in path and not path.startswith("/")


def _get_detector(request: Request):
    """Safe detector access — returns 503 if not initialized."""
    if SubspaceAnomalyDetector is None:
        raise HTTPException(503, "检测器模块加载失败，请检查服务日志")
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(503, "检测器未初始化，请检查服务日志")
    return detector


@business_router.post("/train", summary="Build Memory Bank")
async def train(
    request: Request,
    files: List[UploadFile] = File(..., description="正常图像（1 张即可，支持多张）"),
    image_res: int = Form(DEFAULT_IMAGE_RES, description="输入分辨率"),
    similarity_aggregation: str = Form(SUBSPACE_SIMILARITY_AGGREGATION,
        description="相似度聚合: max / top1_mean / knn_weighted"),
    layer_fusion: str = Form(SUBSPACE_LAYER_FUSION,
        description="多层融合: score_avg / score_max / feature_avg / feature_concat"),
    layers: str = Form(",".join(str(x) for x in SUBSPACE_LAYERS),
        description="特征层索引，逗号分隔 (如 8,10,12)"),
    coreset_ratio: float = Form(SUBSPACE_CORESET_RATIO,
        description="PatchCore 式记忆库 coreset 比例: 0.0=关闭, 0.05=保留 5%"),
    coreset_seed: int = Form(SUBSPACE_CORESET_SEED, description="coreset 采样种子"),
    knn_k: int = Form(SUBSPACE_KNN_K,
        description="knn_weighted 聚合的近邻数 (仅该模式生效)"),
    knn_temperature: float = Form(SUBSPACE_KNN_TEMPERATURE,
        description="knn_weighted 逆距离加权温度 (仅该模式生效)"),
):
    """使用正常图像构建 SubspaceAD 特征记忆库（Training-Free）。"""
    if not files or len(files) == 0:
        raise HTTPException(400, "请至少上传一张正常图像")

    pil_images = []
    for f in files:
        data = await f.read()
        _validate_image(f.filename or "image.jpg", data)
        pil_images.append(_bytes_to_pil(data))

    detector: SubspaceAnomalyDetector = _get_detector(request)

    t0 = time.time()
    try:
        async with _detector_lock:
            detector.image_res = image_res
            detector.similarity_aggregation = similarity_aggregation
            detector.layer_fusion = layer_fusion
            detector.layers = tuple(int(x.strip()) for x in layers.split(",") if x.strip())
            detector.coreset_ratio = coreset_ratio
            detector.coreset_seed = coreset_seed
            detector.knn_k = knn_k
            detector.knn_temperature = knn_temperature
            train_info = await asyncio.to_thread(
                detector.train, template_images=pil_images, verbose=True,
            )
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise HTTPException(500, f"训练失败: {str(e)}")

    elapsed = time.time() - t0

    request.app.state.is_trained = True
    request.app.state.train_info = {
        "num_patches": train_info["num_patches"],
        "feature_dim": train_info["feature_dim"],
        "num_templates": train_info["num_templates"],
        "image_res": image_res,
        "similarity_aggregation": similarity_aggregation,
        "layer_fusion": layer_fusion,
        "layers": layers,
        "coreset_ratio": coreset_ratio,
        "coreset_seed": coreset_seed,
        "knn_k": knn_k,
        "knn_temperature": knn_temperature,
    }

    return {
        "success": True,
        "num_patches": train_info["num_patches"],
        "feature_dim": train_info["feature_dim"],
        "num_templates": train_info["num_templates"],
        "build_time_ms": round(elapsed * 1000, 1),
    }


@business_router.post("/detect", summary="Detect Anomalies")
async def detect(
    request: Request,
    file: UploadFile = File(..., description="待检测图像"),
    viz_mode: str = Form("overlay", description="可视化模式: overlay / side_by_side / bbox"),
    return_heatmap: bool = Form(True, description="是否返回热力图"),
    bbox_threshold: float = Form(0.5, description="缺陷检测阈值 (仅 bbox 模式)"),
    top_k_ratio: float = Form(0.01, description="图像级分数的 top-k 比例"),
    enable_localization: bool = Form(DEFAULT_LOCALIZE, description="是否启用目标定位"),
    localization_method: str = Form(DEFAULT_LOCALIZATION_METHOD,
        description="定位策略: auto / saliency / contour / manual / none"),
    crop_to_roi: bool = Form(DEFAULT_CROP_TO_ROI, description="定位后是否裁切 ROI 检测"),
    roi_margin: float = Form(ROI_MARGIN_RATIO, description="ROI 扩展边距比例"),
):
    """对上传的图像进行异常检测。需要先调用 /api/train 训练模型。"""
    if not request.app.state.is_trained:
        raise HTTPException(400, "请先调用 /api/train 训练模型")

    data = await file.read()
    _validate_image(file.filename or "image.jpg", data)
    test_img = _bytes_to_pil(data)

    detector: SubspaceAnomalyDetector = _get_detector(request)

    t0 = time.time()
    try:
        async with _detector_lock:
            results = await asyncio.to_thread(
                detector.detect,
                test_images=[test_img],
                save_dir=None,
                save_visualizations=False,
                viz_mode=viz_mode,
                bbox_threshold=bbox_threshold,
                top_k_ratio=top_k_ratio,
                verbose=False,
                enable_localization=enable_localization,
                localization_method=localization_method,
                crop_to_roi=crop_to_roi,
                roi_margin=roi_margin,
            )
    except Exception as e:
        logger.error("Detection failed: %s", e)
        raise HTTPException(500, f"检测失败: {str(e)}")

    elapsed = time.time() - t0
    result = results[0]

    anomaly_score = result["anomaly_score"]
    anomaly_map = result["anomaly_map"]

    threshold = getattr(detector, "threshold", 0.3)
    is_anomaly = anomaly_score > threshold

    resp = {
        "success": True,
        "anomaly_score": round(float(anomaly_score), 6),
        "is_anomaly": is_anomaly,
        "threshold": threshold,
        "inference_time_ms": round(elapsed * 1000, 1),
    }

    if "localization" in result:
        resp["localization"] = result["localization"]

    # Double-check results
    if "graph_check" in result:
        resp["graph_check"] = result["graph_check"]
    if "layoutad" in result:
        resp["layoutad"] = result["layoutad"]
    if "fused_score" in result:
        resp["fused_score"] = result["fused_score"]
        resp["fusion_method"] = result.get("fusion_method", "subspace_only")
    if "structural_score" in result:
        resp["structural_score"] = result["structural_score"]
    if "layout_score" in result:
        resp["layout_score"] = result["layout_score"]

    # Include SubspaceAD CLS-patch attention map (saliency)
    if result.get("attention_map") is not None:
        attn_map_norm = np.clip(result["attention_map"], 0, 1)
        attn_heatmap = create_heatmap(attn_map_norm)
        resp["attention_map"] = _numpy_to_base64(attn_heatmap)

    if return_heatmap:
        heatmap_bgr = create_heatmap(anomaly_map)
        resp["heatmap"] = _numpy_to_base64(heatmap_bgr)

    try:
        roi_bbox = None
        if "localization" in result and result["localization"]:
            roi_bbox = tuple(result["localization"]["bbox"])
        viz_bgr = render_visualization(
            test_img, anomaly_map,
            viz_mode=viz_mode, score=anomaly_score,
            bbox_threshold=bbox_threshold, roi_bbox=roi_bbox,
        )
        resp["visualization"] = _numpy_to_base64(viz_bgr)
        resp["viz_mode"] = viz_mode
    except Exception as e:
        logger.warning("Visualization generation failed: %s", e)

    return resp


@business_router.post("/reset", summary="Reset Detector")
async def reset(request: Request):
    """重置检测器状态。"""
    async with _detector_lock:
        detector: SubspaceAnomalyDetector = _get_detector(request)
        old_res = detector.image_res
        old_model_path = detector.model_path
        old_sim_agg = detector.similarity_aggregation
        old_fusion = detector.layer_fusion
        old_layers = detector.layers
        old_coreset_ratio = detector.coreset_ratio
        old_coreset_seed = detector.coreset_seed
        old_knn_k = detector.knn_k
        old_knn_temperature = detector.knn_temperature

        request.app.state.detector = SubspaceAnomalyDetector(
            model_path=old_model_path, image_res=old_res,
            similarity_aggregation=old_sim_agg,
            layer_fusion=old_fusion, layers=old_layers,
            coreset_ratio=old_coreset_ratio,
            coreset_seed=old_coreset_seed,
            knn_k=old_knn_k,
            knn_temperature=old_knn_temperature,
        )
        request.app.state.is_trained = False
        request.app.state.train_info = {}

    return {"success": True, "message": "检测器已重置，请重新构建记忆库"}


@business_router.get("/models", summary="List Available Models")
async def list_models(request: Request):
    """列出可用骨干模型（DINOv2/DINOv3）及当前激活模型。"""
    detector = request.app.state.detector
    current = detector.model_path if detector is not None else "N/A"
    models = []
    for m in AVAILABLE_MODELS:
        available = False
        if detector is not None:
            path = m["path"]
            if os.path.isdir(path):
                available = any(
                    os.path.isfile(os.path.join(path, name))
                    for name in ("model.safetensors", "pytorch_model.bin")
                )
            elif _is_hf_id(path):
                available = True  # HF model id 在线加载
            # else: 本地目录路径但目录不存在 → 权重缺失, available=False
        models.append({
            "id": m["id"],
            "name": m["name"],
            "path": m["path"],
            "default_layers": m["default_layers"],
            "default_res": m["default_res"],
            "available": available,
        })
    return {
        "current": current,
        "models": models,
    }


@business_router.post("/model/switch", summary="Switch Backbone Model")
async def switch_model(
    request: Request,
    model_id: str = Form(..., description="模型 id（见 /api/models）"),
):
    """运行时切换骨干模型（DINOv2 ↔ DINOv3）。

    切换后记忆库作废（特征不兼容），需重新调用 /api/train 构建。
    """
    entry = next((m for m in AVAILABLE_MODELS if m["id"] == model_id), None)
    if entry is None:
        raise HTTPException(400, f"未知模型 id: {model_id}，可用: {[m['id'] for m in AVAILABLE_MODELS]}")

    detector: SubspaceAnomalyDetector = _get_detector(request)

    # 本地目录模式需确认权重存在，避免切到一个空目录或缺失目录
    path = entry["path"]
    if os.path.isdir(path):
        if not any(
            os.path.isfile(os.path.join(path, name))
            for name in ("model.safetensors", "pytorch_model.bin")
        ):
            raise HTTPException(400, f"模型 {model_id} 权重缺失: {path}（请先放置 model.safetensors）")
    elif not _is_hf_id(path):
        # 本地目录路径但目录不存在（既非目录也非 HF id）→ 权重未就位
        raise HTTPException(400, f"模型 {model_id} 本地权重目录不存在: {path}")

    async with _detector_lock:
        detector.switch_model(model_path=path, layers=entry["default_layers"])
        request.app.state.is_trained = False
        request.app.state.train_info = {}

    return {
        "success": True,
        "model_id": model_id,
        "model_name": entry["name"],
        "model_path": detector.model_path,
        "model_kind": detector.model_kind,
        "message": f"已切换至 {entry['name']}，请重新构建记忆库",
    }


@business_router.get("/status", summary="Service Status")
async def status(request: Request):
    """查看当前检测器状态和记忆库信息。"""
    detector = request.app.state.detector  # Don't use _get_detector — status should work even if detector failed
    if detector is None:
        return {
            "is_trained": False,
            "model_path": "N/A",
            "model_kind": "N/A",
            "device": "N/A",
            "image_res": 0,
            "similarity_aggregation": "N/A",
            "layer_fusion": "N/A",
            "layers": [],
            "train_info": {"error": "detector not initialized"},
            "error": getattr(request.app.state.train_info, "error", "detector failed to load"),
        }
    return {
        "is_trained": request.app.state.is_trained,
        "model_loaded": bool(getattr(detector, "is_loaded", False)),
        "model_path": detector.model_path,
        "model_kind": detector.model_kind,
        "device": detector.device,
        "image_res": detector.image_res,
        "similarity_aggregation": detector.similarity_aggregation,
        "layer_fusion": detector.layer_fusion,
        "layers": list(detector.layers),
        "train_info": request.app.state.train_info,
    }


# ============================================================
# Video Detection
# ============================================================

@business_router.post("/detect-video", summary="Video Anomaly Detection")
async def detect_video(
    request: Request,
    file: UploadFile = File(..., description="视频文件"),
    sample_every: int = Form(10, description="每 N 帧采样一次"),
    max_frames: int = Form(200, description="最大处理帧数"),
):
    """对视频逐帧进行异常检测，返回每帧分数和汇总。"""
    if not request.app.state.is_trained:
        raise HTTPException(400, "请先调用 /api/train 构建记忆库")

    # Save uploaded video to temp file
    data = await file.read()
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    detector: SubspaceAnomalyDetector = _get_detector(request)

    t0 = time.time()
    try:
        async with _detector_lock:
            results = await asyncio.to_thread(
                detector.detect_video,
                video_path=tmp_path,
                sample_every_n_frames=sample_every,
                max_frames=max_frames,
                verbose=True,
            )
    except Exception as e:
        logger.error("Video detection failed: %s", e)
        os.unlink(tmp_path)
        raise HTTPException(500, f"视频检测失败: {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    elapsed = time.time() - t0
    summary = results.pop() if results and results[-1].get("_summary") else {}

    return {
        "success": True,
        "frames": results,
        "summary": {
            "total_frames_scanned": summary.get("total_frames_scanned", len(results)),
            "max_anomaly_score": summary.get("max_anomaly_score", 0),
            "max_anomaly_frame": summary.get("max_anomaly_frame", 0),
            "max_anomaly_timestamp": summary.get("max_anomaly_timestamp", 0),
            "anomaly_frame_count": summary.get("anomaly_frame_count", 0),
            "anomaly_ratio": summary.get("anomaly_ratio", 0),
            "mean_score": summary.get("mean_score", 0),
            "fps": summary.get("fps", 0),
        },
        "processing_time_ms": round(elapsed * 1000, 1),
    }
