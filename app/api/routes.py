"""Business API routes — SubspaceAD anomaly detection endpoints."""

import base64
import io
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from ..config import (
    BUSINESS_PREFIX,
    DEFAULT_IMAGE_RES,
    DEFAULT_PCA_EV,
    DEFAULT_SCORE_METHOD,
    MAX_FILE_SIZE_MB,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from models.detector import SubspaceAnomalyDetector

business_router = APIRouter(prefix=BUSINESS_PREFIX, tags=["Business"])
logger = logging.getLogger(__name__)

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


def _ensure_rgb(img_np: np.ndarray) -> np.ndarray:
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    return img_np


def _create_heatmap(anom_map_norm: np.ndarray) -> np.ndarray:
    anom_map_u8 = (np.clip(anom_map_norm, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(anom_map_u8, cv2.COLORMAP_JET)


def _find_defect_bbox(anom_map_norm: np.ndarray, threshold: float = 0.5) -> Optional[tuple]:
    anom_map_u8 = (anom_map_norm * 255).astype(np.uint8)
    _, binary = cv2.threshold(anom_map_u8, int(threshold * 255), 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 100:
        return None
    return cv2.boundingRect(largest)


def generate_visualization(
    img: Image.Image,
    anom_map: np.ndarray,
    viz_mode: str = "overlay",
    score: Optional[float] = None,
    bbox_threshold: float = 0.5,
) -> np.ndarray:
    h, w = anom_map.shape
    img_np = np.array(img.resize((w, h)))
    img_np_rgb = _ensure_rgb(img_np)
    heatmap = _create_heatmap(anom_map)

    if viz_mode == "overlay":
        result = cv2.addWeighted(img_np_rgb, 0.6, heatmap, 0.4, 0)
        if score is not None:
            cv2.putText(result, f"Score: {score:.4f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    elif viz_mode == "side_by_side":
        left = img_np_rgb.copy()
        cv2.putText(left, "Original", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        right = cv2.addWeighted(img_np_rgb, 0.6, heatmap, 0.4, 0)
        cv2.putText(right, "With Heatmap", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        result = np.hstack([left, right])
    elif viz_mode == "bbox":
        result = img_np_rgb.copy()
        bbox = _find_defect_bbox(anom_map, threshold=bbox_threshold)
        if bbox is not None:
            x, y, bw, bh = bbox
            cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
            cv2.putText(result, f"Defect: {bw}x{bh}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        if score is not None:
            cv2.putText(result, f"Score: {score:.4f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        raise ValueError(f"Unknown viz_mode: {viz_mode}")

    return result


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

@business_router.post("/train", summary="Train PCA Model")
async def train(
    request: Request,
    files: List[UploadFile] = File(..., description="正常图像（1 张即可，支持多张）"),
    image_res: int = Form(DEFAULT_IMAGE_RES, description="输入分辨率"),
    pca_ev: float = Form(DEFAULT_PCA_EV, description="PCA 保留方差比例 (0-1)"),
    score_method: str = Form(DEFAULT_SCORE_METHOD,
        description="评分方法: reconstruction / mahalanobis / euclidean / cosine"),
):
    """使用正常图像训练 PCA 子空间模型。"""
    if not files or len(files) == 0:
        raise HTTPException(400, "请至少上传一张正常图像")

    pil_images = []
    for f in files:
        data = await f.read()
        _validate_image(f.filename or "image.jpg", data)
        pil_images.append(_bytes_to_pil(data))

    detector: SubspaceAnomalyDetector = request.app.state.detector
    detector.image_res = image_res
    detector.pca_ev = pca_ev
    detector.score_method = score_method

    t0 = time.time()
    try:
        train_info = detector.train(template_images=pil_images, verbose=True)
    except Exception as e:
        logger.error("Training failed: %s", e)
        raise HTTPException(500, f"训练失败: {str(e)}")

    elapsed = time.time() - t0

    request.app.state.is_trained = True
    request.app.state.train_info = {
        "pca_components": train_info["pca_components"],
        "feature_dim": train_info["feature_dim"],
        "grid_size": train_info["grid_size"],
        "num_templates": train_info["num_templates"],
        "image_res": image_res,
        "score_method": score_method,
    }

    return {
        "success": True,
        "pca_components": train_info["pca_components"],
        "feature_dim": train_info["feature_dim"],
        "grid_size": list(train_info["grid_size"]),
        "num_templates": train_info["num_templates"],
        "training_time_ms": round(elapsed * 1000, 1),
    }


@business_router.post("/detect", summary="Detect Anomalies")
async def detect(
    request: Request,
    file: UploadFile = File(..., description="待检测图像"),
    viz_mode: str = Form("overlay", description="可视化模式: overlay / side_by_side / bbox"),
    return_heatmap: bool = Form(True, description="是否返回热力图"),
    bbox_threshold: float = Form(0.5, description="缺陷检测阈值 (仅 bbox 模式)"),
    top_k_ratio: float = Form(0.01, description="图像级分数的 top-k 比例"),
):
    """对上传的图像进行异常检测。需要先调用 /api/train 训练模型。"""
    if not request.app.state.is_trained:
        raise HTTPException(400, "请先调用 /api/train 训练模型")

    data = await file.read()
    _validate_image(file.filename or "image.jpg", data)
    test_img = _bytes_to_pil(data)

    detector: SubspaceAnomalyDetector = request.app.state.detector

    t0 = time.time()
    try:
        results = detector.detect(
            test_images=[test_img],
            save_dir=None,
            save_visualizations=False,
            viz_mode=viz_mode,
            bbox_threshold=bbox_threshold,
            top_k_ratio=top_k_ratio,
            verbose=False,
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

    if return_heatmap:
        heatmap_bgr = _create_heatmap(anomaly_map)
        resp["heatmap"] = _numpy_to_base64(heatmap_bgr)

    try:
        viz_bgr = generate_visualization(
            test_img, anomaly_map,
            viz_mode=viz_mode, score=anomaly_score, bbox_threshold=bbox_threshold,
        )
        resp["visualization"] = _numpy_to_base64(viz_bgr)
        resp["viz_mode"] = viz_mode
    except Exception as e:
        logger.warning("Visualization generation failed: %s", e)

    return resp


@business_router.post("/reset", summary="Reset Detector")
async def reset(request: Request):
    """重置检测器状态。"""
    detector: SubspaceAnomalyDetector = request.app.state.detector
    old_res = detector.image_res
    old_ev = detector.pca_ev
    old_method = detector.score_method
    old_ckpt = detector.model_ckpt

    request.app.state.detector = SubspaceAnomalyDetector(
        model_ckpt=old_ckpt, image_res=old_res,
        pca_ev=old_ev, score_method=old_method,
    )
    request.app.state.is_trained = False
    request.app.state.train_info = {}

    return {"success": True, "message": "检测器已重置，请重新训练"}


@business_router.get("/status", summary="Service Status")
async def status(request: Request):
    """查看当前检测器状态和训练信息。"""
    detector: SubspaceAnomalyDetector = request.app.state.detector
    return {
        "is_trained": request.app.state.is_trained,
        "model": detector.model_ckpt,
        "device": detector.device,
        "image_res": detector.image_res,
        "pca_ev": detector.pca_ev,
        "score_method": detector.score_method,
        "train_info": request.app.state.train_info,
    }
