"""
SubspaceAD Anomaly Detection Service
=====================================

基于 DINOv2 + PCA 子空间建模的少样本异常检测服务，集成 MeSquare 监控标准端点。

标准端点:
  GET  /health           - 健康检查
  GET  /api-info         - 服务元信息
  GET  /metrics          - 请求统计
  GET  /resources        - 资源利用率
  GET  /endpoint-metrics - 各端点统计
  GET  /logs             - 服务日志

业务端点:
  POST /train   - 上传正常图像训练 PCA 模型
  POST /detect  - 上传待测图进行异常检测
  POST /reset   - 重置检测器状态
  GET  /status  - 查看当前状态
"""

import os
import io
import time
import base64
import socket
import threading
import logging
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import psutil
from PIL import Image
from datetime import datetime, timezone

from fastapi import FastAPI, Request, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from subspace_anomaly_detector import SubspaceAnomalyDetector


# ============================================================
# Configuration
# ============================================================
SERVICE_NAME = "SubspaceAD"
SERVICE_VERSION = "1.0.0"
SERVICE_DESCRIPTION = "基于 DINOv2 + PCA 子空间建模的少样本异常检测服务"
SERVICE_PORT = int(os.environ.get("PORT", 8703))
CAPABILITIES = [
    "anomaly_detection",
    "industrial_quality_inspection",
    "few_shot_learning",
]
SUPPORTED_FORMATS = ["png", "jpg", "jpeg", "bmp", "tiff"]
MAX_FILE_SIZE_MB = 50.0

try:
    SERVER_HOST = socket.gethostname()
except Exception:
    SERVER_HOST = "localhost"


# ============================================================
# Memory Log Handler
# ============================================================
_log_buffer: deque = deque(maxlen=2000)


class MemoryLogHandler(logging.Handler):
    """Captures log records to an in-memory buffer for MeSquare log retrieval."""

    def emit(self, record):
        try:
            msg = self.format(record)
            _log_buffer.append({
                "level": record.levelname,
                "message": msg,
                "timestamp": record.created,
                "logger": record.name,
            })
        except Exception:
            self.handleError(record)


def init_log_capture():
    handler = MemoryLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)
    logging.getLogger("uvicorn").addHandler(handler)


def shutdown_log_capture():
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, MemoryLogHandler)]


# ============================================================
# Metrics Collector
# ============================================================
class MetricsCollector:
    """Lightweight request metrics collector for MeSquare monitoring."""

    def __init__(self):
        self._lock = threading.Lock()
        self._total_requests = 0
        self._active_connections = 0
        self._errors_24h = deque(maxlen=100000)
        self._requests_timestamps = deque(maxlen=100000)
        self._latencies = deque(maxlen=2000)

    def record_request(self, latency_s: float, status_code: int):
        now = time.time()
        is_error = status_code >= 400
        with self._lock:
            self._total_requests += 1
            self._requests_timestamps.append(now)
            if is_error:
                self._errors_24h.append(now)
            self._latencies.append(latency_s * 1000)

    def inc_active(self):
        with self._lock:
            self._active_connections += 1

    def dec_active(self):
        with self._lock:
            self._active_connections = max(0, self._active_connections - 1)

    def get_metrics(self) -> dict:
        now = time.time()
        cutoff_1m = now - 60
        cutoff_1h = now - 3600
        cutoff_24h = now - 86400

        with self._lock:
            total = self._total_requests
            active = self._active_connections
            latencies = list(self._latencies)

            last_min = sum(1 for t in self._requests_timestamps if t > cutoff_1m)
            last_hour = sum(1 for t in self._requests_timestamps if t > cutoff_1h)
            last_24h = sum(1 for t in self._requests_timestamps if t > cutoff_24h)
            errors_24h = sum(1 for t in self._errors_24h if t > cutoff_24h)

        error_rate = errors_24h / last_24h if last_24h > 0 else 0.0

        avg_lat = p95 = p99 = 0.0
        if latencies:
            sorted_lat = sorted(latencies)
            n = len(sorted_lat)
            avg_lat = sum(sorted_lat) / n
            p95 = sorted_lat[min(int(n * 0.95), n - 1)]
            p99 = sorted_lat[min(int(n * 0.99), n - 1)]

        return {
            "total_requests": total,
            "requests_last_minute": float(last_min),
            "requests_last_hour": last_hour,
            "requests_last_24h": last_24h,
            "avg_latency_ms": round(avg_lat, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "error_rate_24h": round(error_rate, 4),
            "active_connections": active,
        }


# ============================================================
# Endpoint Metrics Tracker
# ============================================================
class EndpointMetricsTracker:
    """Tracks per-endpoint request metrics for MeSquare."""

    def __init__(self):
        self._lock = threading.Lock()
        self._endpoints: dict[tuple[str, str], dict] = {}

    def record(self, path: str, method: str, latency_ms: float, is_error: bool):
        now = time.time()
        key = (path, method)
        with self._lock:
            if key not in self._endpoints:
                self._endpoints[key] = {
                    "count": 0, "errors": 0,
                    "latency_sum": 0.0, "timestamps": deque(),
                }
            ep = self._endpoints[key]
            ep["count"] += 1
            ep["latency_sum"] += latency_ms
            if is_error:
                ep["errors"] += 1
            ep["timestamps"].append(now)

    def get_metrics(self) -> list[dict]:
        now = time.time()
        cutoff = now - 60.0
        result = []
        with self._lock:
            for (path, method), ep in self._endpoints.items():
                ts = ep["timestamps"]
                while ts and ts[0] <= cutoff:
                    ts.popleft()
                total = ep["count"]
                rpm = len(ts)
                avg_lat = ep["latency_sum"] / total if total > 0 else 0.0
                error_rate = ep["errors"] / total if total > 0 else 0.0
                result.append({
                    "path": path,
                    "method": method,
                    "total_requests": total,
                    "requests_per_minute": round(float(rpm), 1),
                    "avg_latency_ms": round(avg_lat, 2),
                    "error_rate": round(error_rate, 4),
                })
        result.sort(key=lambda x: x["total_requests"], reverse=True)
        return result


# ============================================================
# GPU Info
# ============================================================
def get_gpu_info() -> Optional[dict]:
    """Query NVIDIA GPU utilization. Returns None if no GPU or pynvml not installed."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "gpu_utilization_percent": float(util.gpu),
            "gpu_memory_used_mb": round(info.used / 1024**2, 1),
            "gpu_memory_total_mb": round(info.total / 1024**2, 1),
        }
    except Exception:
        return None


# ============================================================
# Pydantic Models
# ============================================================
class EndpointMetric(BaseModel):
    path: str = Field(description="API route path")
    method: str = Field(description="HTTP method (uppercase)")
    total_requests: int = Field(description="Total requests since startup", ge=0)
    requests_per_minute: float = Field(description="Requests in last 60 seconds", ge=0)
    avg_latency_ms: float = Field(description="Average response time in ms", ge=0)
    error_rate: float = Field(description="Error rate (4xx+5xx / total)", ge=0, le=1)


class EndpointMetricsResponse(BaseModel):
    endpoints: List[EndpointMetric] = Field(description="Per-endpoint metrics list")


# ============================================================
# Image processing helpers
# ============================================================
def _img_to_base64(pil_img: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _numpy_to_base64(img_np: np.ndarray) -> str:
    """Convert numpy image (H,W,3 BGR or RGB) to base64 PNG string."""
    if img_np.shape[-1] == 3:
        # cv2 uses BGR, PIL uses RGB
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img_np
    return _img_to_base64(Image.fromarray(img_rgb))


def _ensure_rgb(img_np: np.ndarray) -> np.ndarray:
    """Ensure numpy image is 3-channel."""
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        return cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    return img_np


def _create_heatmap(anom_map_norm: np.ndarray) -> np.ndarray:
    """Convert normalized anomaly map (0-1 float) to JET colormap BGR image."""
    anom_map_u8 = (np.clip(anom_map_norm, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(anom_map_u8, cv2.COLORMAP_JET)


def _find_defect_bbox(anom_map_norm: np.ndarray, threshold: float = 0.5) -> Optional[tuple]:
    """Find defect bounding box from normalized anomaly map."""
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
    """Generate visualization image (BGR numpy array)."""
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
# FastAPI Application
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_log_capture()
    app.state.start_time = time.time()
    app.state.metrics_collector = MetricsCollector()
    app.state.endpoint_metrics = EndpointMetricsTracker()

    # Initialize detector (model loaded lazily on first /train)
    app.state.detector = SubspaceAnomalyDetector(
        model_ckpt=None,  # auto-use local models/dinov2-small
        image_res=512,
        pca_ev=0.99,
        score_method="reconstruction",
    )
    app.state.is_trained = False
    app.state.train_info = {}

    logging.info(f"{SERVICE_NAME} v{SERVICE_VERSION} started on port {SERVICE_PORT}")
    yield
    # Shutdown
    shutdown_log_capture()


app = FastAPI(
    title=SERVICE_NAME,
    description=SERVICE_DESCRIPTION,
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Monitoring Middleware
# ============================================================
MONITOR_PATHS = frozenset({
    "/health", "/api-info", "/metrics", "/resources",
    "/endpoint-metrics", "/logs", "/openapi.json", "/docs", "/redoc",
})


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    if request.url.path in MONITOR_PATHS:
        return await call_next(request)

    collector = request.app.state.metrics_collector
    endpoint_metrics = request.app.state.endpoint_metrics
    collector.inc_active()
    start = time.time()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        latency = time.time() - start
        collector.record_request(latency, status_code)
        endpoint_metrics.record(
            request.url.path, request.method, latency * 1000, status_code >= 400
        )
        collector.dec_active()


# ============================================================
# MeSquare Standard Endpoints
# ============================================================

@app.get("/health", summary="Health Check", tags=["Monitoring"])
async def health(request: Request):
    """Returns service health status with component breakdown."""
    uptime = time.time() - request.app.state.start_time

    components = {}

    # Detector / Model status
    detector = request.app.state.detector
    if detector.extractor is not None:
        components["model"] = "healthy"
    else:
        components["model"] = "degraded"

    # Training status
    if request.app.state.is_trained:
        components["pca"] = "healthy"
    else:
        components["pca"] = "degraded"

    # Storage
    try:
        storage_path = os.getcwd()
        if os.access(storage_path, os.W_OK):
            components["storage"] = "healthy"
        else:
            components["storage"] = "degraded"
    except Exception:
        components["storage"] = "down"

    # GPU
    gpu_info = get_gpu_info()
    if gpu_info is not None:
        components["gpu"] = "healthy"

    # Overall status
    statuses = list(components.values())
    if "down" in statuses:
        overall = "down"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "uptime_seconds": round(uptime, 1),
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "components": components,
    }


@app.get("/api-info", summary="Service Metadata", tags=["Monitoring"])
async def api_info():
    """Returns service metadata and capabilities."""
    return {
        "name": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "description": SERVICE_DESCRIPTION,
        "server_host": SERVER_HOST,
        "server_port": SERVICE_PORT,
        "capabilities": CAPABILITIES,
        "supported_formats": SUPPORTED_FORMATS,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
    }


@app.get("/metrics", summary="Request Metrics", tags=["Monitoring"])
async def metrics(request: Request):
    """Returns aggregated request statistics."""
    return request.app.state.metrics_collector.get_metrics()


@app.get("/resources", summary="Resource Utilization", tags=["Monitoring"])
async def resources():
    """Returns server-level resource utilization."""
    cpu_percent = psutil.cpu_percent(interval=0)
    cpu_count = psutil.cpu_count()
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(os.getcwd())

    gpu = get_gpu_info()

    return {
        "cpu_percent": round(cpu_percent, 1),
        "cpu_count": cpu_count,
        "memory_used_mb": round(mem.used / 1024**2, 1),
        "memory_total_mb": round(mem.total / 1024**2, 1),
        "memory_percent": round(mem.percent, 1),
        "disk_used_gb": round(disk.used / 1024**3, 1),
        "disk_total_gb": round(disk.total / 1024**3, 1),
        "disk_percent": round(disk.percent, 1),
        "gpu_utilization_percent": gpu["gpu_utilization_percent"] if gpu else None,
        "gpu_memory_used_mb": gpu["gpu_memory_used_mb"] if gpu else None,
        "gpu_memory_total_mb": gpu["gpu_memory_total_mb"] if gpu else None,
    }


@app.get("/endpoint-metrics", response_model=EndpointMetricsResponse,
         summary="Per-Endpoint Metrics", tags=["Monitoring"])
async def endpoint_metrics(request: Request):
    """Returns per-endpoint request metrics."""
    return {"endpoints": request.app.state.endpoint_metrics.get_metrics()}


@app.get("/logs", summary="Service Logs", tags=["Monitoring"])
async def get_logs(limit: int = 500):
    """Returns recent service logs from memory buffer."""
    logs = list(_log_buffer)[-limit:]
    return {"logs": logs}


# ============================================================
# Business Endpoints — SubspaceAD
# ============================================================

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff"}


def _validate_image(filename: str, contents: bytes) -> str:
    """Validate image file type and size, return extension."""
    ext = Path(filename).suffix.lower().lstrip(".") if filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"不支持的文件格式: {ext}。支持的格式: {', '.join(ALLOWED_EXTENSIONS)}")

    max_bytes = int(MAX_FILE_SIZE_MB * 1024 * 1024)
    if len(contents) > max_bytes:
        raise HTTPException(400, f"文件大小超过限制 ({MAX_FILE_SIZE_MB}MB)")
    return ext


def _bytes_to_pil(data: bytes) -> Image.Image:
    """Convert bytes to PIL RGB image."""
    try:
        img = Image.open(io.BytesIO(data))
        return img.convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"图片解码失败: {str(e)}")


@app.get("/", response_class=HTMLResponse, summary="Test Page", tags=["Business"])
async def root():
    """Serve the visual test page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return HTMLResponse("<h1>SubspaceAD Service</h1><p>Test page not found.</p>")


@app.post("/train", summary="Train PCA Model", tags=["Business"])
async def train(
    request: Request,
    files: List[UploadFile] = File(...,
        description="正常图像（1 张即可，支持多张）"),
    image_res: int = Form(512, description="输入分辨率"),
    pca_ev: float = Form(0.99, description="PCA 保留方差比例 (0-1)"),
    score_method: str = Form("reconstruction",
        description="评分方法: reconstruction / mahalanobis / euclidean / cosine"),
):
    """
    使用正常图像训练 PCA 子空间模型。

    上传 1 张或多张正常（无缺陷）图像，提取 DINOv2 特征后拟合 PCA 模型。
    训练完成后即可调用 /detect 进行异常检测。
    """
    if not files or len(files) == 0:
        raise HTTPException(400, "请至少上传一张正常图像")

    pil_images = []
    for f in files:
        data = await f.read()
        ext = _validate_image(f.filename or "image.jpg", data)
        pil_images.append(_bytes_to_pil(data))

    detector: SubspaceAnomalyDetector = request.app.state.detector
    detector.image_res = image_res
    detector.pca_ev = pca_ev
    detector.score_method = score_method

    t0 = time.time()
    try:
        train_info = detector.train(
            template_images=pil_images,
            verbose=True,
        )
    except Exception as e:
        logging.error(f"Training failed: {e}")
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


@app.post("/detect", summary="Detect Anomalies", tags=["Business"])
async def detect(
    request: Request,
    file: UploadFile = File(..., description="待检测图像"),
    viz_mode: str = Form("overlay",
        description="可视化模式: overlay / side_by_side / bbox"),
    return_heatmap: bool = Form(True, description="是否返回热力图"),
    bbox_threshold: float = Form(0.5, description="缺陷检测阈值 (仅 bbox 模式)"),
    top_k_ratio: float = Form(0.01, description="图像级分数的 top-k 比例"),
):
    """
    对上传的图像进行异常检测。

    需要先调用 /train 训练模型。返回异常分数、热力图和可视化结果。
    """
    if not request.app.state.is_trained:
        raise HTTPException(400, "请先调用 /train 训练模型")

    # Read and validate upload
    data = await file.read()
    ext = _validate_image(file.filename or "image.jpg", data)
    test_img = _bytes_to_pil(data)

    detector: SubspaceAnomalyDetector = request.app.state.detector

    # Run detection (single image, no disk saving)
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
        logging.error(f"Detection failed: {e}")
        raise HTTPException(500, f"检测失败: {str(e)}")

    elapsed = time.time() - t0
    result = results[0]

    anomaly_score = result["anomaly_score"]
    anomaly_map = result["anomaly_map"]  # normalized 0-1 at original image size

    # Determine if anomaly
    threshold = getattr(detector, "threshold", 0.3)
    is_anomaly = anomaly_score > threshold

    resp = {
        "success": True,
        "anomaly_score": round(float(anomaly_score), 6),
        "is_anomaly": is_anomaly,
        "threshold": threshold,
        "inference_time_ms": round(elapsed * 1000, 1),
    }

    # Return heatmap if requested
    if return_heatmap:
        heatmap_bgr = _create_heatmap(anomaly_map)
        resp["heatmap"] = _numpy_to_base64(heatmap_bgr)

    # Generate and return visualization
    try:
        viz_bgr = generate_visualization(
            test_img, anomaly_map,
            viz_mode=viz_mode,
            score=anomaly_score,
            bbox_threshold=bbox_threshold,
        )
        resp["visualization"] = _numpy_to_base64(viz_bgr)
        resp["viz_mode"] = viz_mode
    except Exception as e:
        logging.warning(f"Visualization generation failed: {e}")

    return resp


@app.post("/reset", summary="Reset Detector", tags=["Business"])
async def reset(request: Request):
    """
    重置检测器状态。清除已训练的 PCA 模型，重新创建检测器实例。
    需要重新调用 /train 才能进行检测。
    """
    detector: SubspaceAnomalyDetector = request.app.state.detector

    # Re-create detector to reset all state
    old_res = detector.image_res
    old_ev = detector.pca_ev
    old_method = detector.score_method
    old_ckpt = detector.model_ckpt

    request.app.state.detector = SubspaceAnomalyDetector(
        model_ckpt=old_ckpt,
        image_res=old_res,
        pca_ev=old_ev,
        score_method=old_method,
    )
    request.app.state.is_trained = False
    request.app.state.train_info = {}

    return {"success": True, "message": "检测器已重置，请重新训练"}


@app.get("/status", summary="Service Status", tags=["Business"])
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


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = SERVICE_PORT

    print(f"🚀 Starting {SERVICE_NAME} v{SERVICE_VERSION}")
    print(f"📡 API docs: http://localhost:{port}/docs")
    print(f"💡 Register this service in MeSquare with base URL: http://<your-ip>:{port}")

    uvicorn.run("main:app", host=host, port=port, reload=False, workers=1)
