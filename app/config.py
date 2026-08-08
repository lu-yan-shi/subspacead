"""Application configuration constants."""

from __future__ import annotations

import os
import socket
from typing import FrozenSet


def _get_local_ip() -> str:
    """Get LAN IP via UDP socket — avoids gethostname() DNS issues on Windows/Docker."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _csv_env(name: str, default: str = "") -> list[str]:
    raw = os.environ.get(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


SERVICE_NAME = os.environ.get("SERVICE_NAME", "深瞳")
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "1.0.0")
SERVICE_DESCRIPTION = os.environ.get(
    "SERVICE_DESCRIPTION", "深瞳 — 基于 DINOv2 + 记忆库的少样本工业异常检测系统"
)
SERVICE_PORT = int(os.environ.get("PORT", "8704"))

# MeSquare platform URL
MESQUARE_BASE_URL = os.environ.get("MESQUARE_URL", "http://localhost:8000").rstrip("/")

# Service address advertised to MeSquare webhooks and /mse/api-info
SERVER_HOST = os.environ.get("SERVER_HOST", _get_local_ip())
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", f"http://{SERVER_HOST}:{SERVICE_PORT}")

# Business endpoints are mounted under this prefix and discovered from /openapi.json
BUSINESS_PREFIX = os.environ.get("BUSINESS_PREFIX", "/api")

CAPABILITIES = _csv_env(
    "CAPABILITIES", "anomaly_detection,industrial_quality_inspection,few_shot_learning"
)
SUPPORTED_FORMATS = _csv_env("SUPPORTED_FORMATS", "png,jpg,jpeg,bmp,tiff")
MAX_FILE_SIZE_MB = float(os.environ.get("MAX_FILE_SIZE_MB", "50"))

# SubspaceAD model settings
# Default: DINOv2 with registers (HF public model)
# For local weights: set MODEL_PATH=/path/to/local/model/dir
# For HF mirror: set HF_ENDPOINT=https://hf-mirror.com
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "facebook/dinov2-with-registers-base",
)
MODEL_TYPE = os.environ.get("MODEL_TYPE", "dinov2_with_register")
DEFAULT_IMAGE_RES = int(os.environ.get("DEFAULT_IMAGE_RES", "448"))
SUBSPACE_SIMILARITY_AGGREGATION = os.environ.get("SUBSPACE_SIMILARITY_AGGREGATION", "max")
SUBSPACE_LAYER_FUSION = os.environ.get("SUBSPACE_LAYER_FUSION", "score_avg")

# PatchCore-style memory-bank coreset + weighted k-NN (默认关闭，保持现有行为)
# CORESET_RATIO: 0.0 = 不采样; 0.05 = 保留 5% 最远点 coreset
SUBSPACE_CORESET_RATIO = float(os.environ.get("CORESET_RATIO", "0.0"))
SUBSPACE_CORESET_SEED = int(os.environ.get("CORESET_SEED", "42"))
SUBSPACE_KNN_K = int(os.environ.get("KNN_K", "9"))
SUBSPACE_KNN_TEMPERATURE = float(os.environ.get("KNN_TEMPERATURE", "1.0"))
SUBSPACE_LAYERS = tuple(
    int(x.strip()) for x in os.environ.get("SUBSPACE_LAYERS", "8,10,12").split(",")
    if x.strip()
)

CPU_SPIKE_THRESHOLD = float(os.environ.get("CPU_SPIKE_THRESHOLD", "50"))

# Object localization
DEFAULT_LOCALIZE = os.environ.get("DEFAULT_LOCALIZE", "true").lower() in {"1", "true", "yes"}
DEFAULT_LOCALIZATION_METHOD = os.environ.get("DEFAULT_LOCALIZATION_METHOD", "auto")
DEFAULT_CROP_TO_ROI = os.environ.get("DEFAULT_CROP_TO_ROI", "true").lower() in {"1", "true", "yes"}
ROI_MARGIN_RATIO = float(os.environ.get("ROI_MARGIN_RATIO", "0.10"))

# LayoutAD double-check (GNN-based structural verification)
# 工作模式: "light" (默认, SubspaceAD 热力图提取区域) 或 "full" (Mask2Former + CLIP)
ENABLE_LAYOUTAD = os.environ.get("ENABLE_LAYOUTAD", "true").lower() in {"1", "true", "yes"}
LAYOUTAD_CHECKPOINT = os.environ.get("LAYOUTAD_CHECKPOINT", "") or None
LAYOUTAD_PIPELINE_MODE = os.environ.get("LAYOUTAD_PIPELINE_MODE", "light")
# Full pipeline (Mask2Former + CLIP) — 仅 pipeline_mode=full 时需要
LAYOUTAD_MASK2FORMER_CONFIG = os.environ.get("LAYOUTAD_MASK2FORMER_CONFIG", "") or None
LAYOUTAD_MASK2FORMER_WEIGHTS = os.environ.get("LAYOUTAD_MASK2FORMER_WEIGHTS", "") or None

# Optional NVIDIA GPU resource collection
ENABLE_GPU_METRICS = os.environ.get("ENABLE_GPU_METRICS", "false").lower() in {"1", "true", "yes"}

MONITOR_PATHS: FrozenSet[str] = frozenset({
    "/mse/health", "/mse/api-info", "/mse/metrics", "/mse/resources",
    "/mse/endpoint-metrics", "/mse/logs", "/openapi.json", "/docs", "/redoc",
    "/mse/notify-api-change",
})
