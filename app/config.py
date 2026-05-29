"""Application configuration constants."""

import os
from typing import FrozenSet, List


SERVICE_NAME = "SubspaceAD"
SERVICE_VERSION = "1.0.0"
SERVICE_DESCRIPTION = "基于 DINOv2 + PCA 子空间建模的少样本异常检测服务"
SERVICE_PORT = int(os.environ.get("PORT", 8703))

BUSINESS_PREFIX = os.environ.get("BUSINESS_PREFIX", "/api")

CAPABILITIES: List[str] = [
    "anomaly_detection",
    "industrial_quality_inspection",
    "few_shot_learning",
]

SUPPORTED_FORMATS: List[str] = ["png", "jpg", "jpeg", "bmp", "tiff"]
MAX_FILE_SIZE_MB = 50.0

# SubspaceAD model settings
DEFAULT_IMAGE_RES = int(os.environ.get("DEFAULT_IMAGE_RES", "512"))
DEFAULT_PCA_EV = float(os.environ.get("DEFAULT_PCA_EV", "0.99"))
DEFAULT_SCORE_METHOD = os.environ.get("DEFAULT_SCORE_METHOD", "reconstruction")

MESQUARE_BASE_URL = os.environ.get("MESQUARE_URL", "http://localhost:8000").rstrip("/")

CPU_SPIKE_THRESHOLD = float(os.environ.get("CPU_SPIKE_THRESHOLD", "50"))

MONITOR_PATHS: FrozenSet[str] = frozenset({
    "/mse/health", "/mse/api-info", "/mse/metrics", "/mse/resources",
    "/mse/endpoint-metrics", "/mse/logs", "/openapi.json", "/docs", "/redoc",
    "/mse/notify-api-change",
})
