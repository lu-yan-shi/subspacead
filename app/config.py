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


SERVICE_NAME = os.environ.get("SERVICE_NAME", "SubspaceAD")
SERVICE_VERSION = os.environ.get("SERVICE_VERSION", "1.0.0")
SERVICE_DESCRIPTION = os.environ.get(
    "SERVICE_DESCRIPTION", "基于 DINOv2 + PCA 子空间建模的少样本异常检测服务"
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
DEFAULT_IMAGE_RES = int(os.environ.get("DEFAULT_IMAGE_RES", "512"))
DEFAULT_PCA_EV = float(os.environ.get("DEFAULT_PCA_EV", "0.99"))
DEFAULT_SCORE_METHOD = os.environ.get("DEFAULT_SCORE_METHOD", "reconstruction")

CPU_SPIKE_THRESHOLD = float(os.environ.get("CPU_SPIKE_THRESHOLD", "50"))

# Optional NVIDIA GPU resource collection
ENABLE_GPU_METRICS = os.environ.get("ENABLE_GPU_METRICS", "false").lower() in {"1", "true", "yes"}

MONITOR_PATHS: FrozenSet[str] = frozenset({
    "/mse/health", "/mse/api-info", "/mse/metrics", "/mse/resources",
    "/mse/endpoint-metrics", "/mse/logs", "/openapi.json", "/docs", "/redoc",
    "/mse/notify-api-change",
})
