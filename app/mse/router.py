"""MeSquare monitoring router — /mse/* endpoints."""

import os
import time
from datetime import datetime, timezone

import psutil
from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from .. import config
from ..utils.webhook import notify_mesquare_api_change
from .logging import get_recent_logs


mse_router = APIRouter(prefix="/mse", tags=["MeSquare Monitoring"])


class EndpointMetric(BaseModel):
    path: str = Field(description="API route path")
    method: str = Field(description="HTTP method (uppercase)")
    total_requests: int = Field(description="Total requests since startup", ge=0)
    requests_per_minute: float = Field(description="Requests in last 60 seconds", ge=0)
    avg_latency_ms: float = Field(description="Average response time in ms", ge=0)
    error_rate: float = Field(description="Error rate (4xx+5xx / total)", ge=0, le=1)


class EndpointMetricsResponse(BaseModel):
    endpoints: list[EndpointMetric] = Field(description="Per-endpoint metrics list")


class NotifyRequest(BaseModel):
    event: str = Field(default="endpoints_changed", description="Event type to send to MeSquare")


def get_gpu_info() -> dict | None:
    """Query NVIDIA GPU utilization. Returns None if no GPU or pynvml not installed."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return {
            "gpu_utilization_percent": float(util.gpu),
            "gpu_memory_used_mb": round(info.used / 1024 ** 2, 1),
            "gpu_memory_total_mb": round(info.total / 1024 ** 2, 1),
        }
    except Exception:
        try:
            import torch
            if torch.cuda.is_available():
                return {"gpu_utilization_percent": 0, "gpu_memory_used_mb": 0, "gpu_memory_total_mb": 0}
        except Exception:
            pass
        return None


@mse_router.get("/health", summary="Health Check")
async def health(request: Request):
    uptime = time.time() - request.app.state.start_time

    components = {}

    # Model: detector object exists = DINOv2 weights loaded successfully at startup
    detector = getattr(request.app.state, "detector", None)
    if detector is not None:
        components["model"] = "healthy"
    else:
        components["model"] = "down"

    # PCA: training state — not required for service to be alive, but needed for /api/detect
    if getattr(request.app.state, "is_trained", False):
        components["pca"] = "healthy"
    else:
        components["pca"] = "degraded"

    # GPU: real dependency for inference performance
    gpu_info = get_gpu_info()
    if gpu_info is not None:
        components["gpu"] = "healthy"
    else:
        components["gpu"] = "degraded"

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


@mse_router.get("/api-info", summary="Service Metadata")
async def api_info():
    return {
        "name": getattr(config, "SERVICE_NAME", "unknown-service"),
        "version": getattr(config, "SERVICE_VERSION", "1.0.0"),
        "description": getattr(config, "SERVICE_DESCRIPTION", ""),
        "server_host": getattr(config, "SERVER_HOST", ""),
        "server_port": getattr(config, "SERVICE_PORT", None),
        "capabilities": getattr(config, "CAPABILITIES", []),
        "supported_formats": getattr(config, "SUPPORTED_FORMATS", []),
        "max_file_size_mb": getattr(config, "MAX_FILE_SIZE_MB", None),
    }


@mse_router.get("/metrics", summary="Request Metrics")
async def metrics(request: Request):
    return request.app.state.metrics_collector.get_metrics()


@mse_router.get("/resources", summary="Resource Utilization")
async def resources(request: Request):
    cpu_monitor = request.app.state.cpu_monitor
    process = psutil.Process(os.getpid())
    cpu_count = psutil.cpu_count() or 1

    raw_cpu = process.cpu_percent(interval=0.1)
    cpu_percent = raw_cpu / cpu_count

    mem_info = process.memory_info()
    total_mem = psutil.virtual_memory().total
    memory_used_mb = round(mem_info.rss / 1024 ** 2, 1)
    memory_total_mb = round(total_mem / 1024 ** 2, 1)
    memory_percent = round(mem_info.rss / total_mem * 100, 1)

    disk = psutil.disk_usage(os.getcwd())
    gpu = get_gpu_info()

    return {
        "cpu_percent": round(cpu_percent, 1),
        "cpu_count": cpu_count,
        "memory_used_mb": memory_used_mb,
        "memory_total_mb": memory_total_mb,
        "memory_percent": memory_percent,
        "disk_used_gb": round(disk.used / 1024 ** 3, 1),
        "disk_total_gb": round(disk.total / 1024 ** 3, 1),
        "disk_percent": round(disk.percent, 1),
        "gpu_name": None,
        "gpu_utilization_percent": gpu["gpu_utilization_percent"] if gpu else None,
        "gpu_memory_used_mb": gpu["gpu_memory_used_mb"] if gpu else None,
        "gpu_memory_total_mb": gpu["gpu_memory_total_mb"] if gpu else None,
        "cpu_spikes": cpu_monitor.get_spikes(),
    }


@mse_router.get("/endpoint-metrics", response_model=EndpointMetricsResponse,
                summary="Per-Endpoint Metrics")
async def endpoint_metrics(request: Request):
    return {"endpoints": request.app.state.endpoint_metrics.get_metrics()}


@mse_router.get("/logs", summary="Service Logs")
async def get_logs(limit: int = 500):
    return {"logs": get_recent_logs(limit)}


@mse_router.post("/notify-api-change", summary="Notify MeSquare of API Changes")
async def trigger_notify(req: NotifyRequest = NotifyRequest()):
    await notify_mesquare_api_change(event=req.event)
    return {
        "status": "sent",
        "event": req.event,
        "mesquare_url": config.MESQUARE_BASE_URL,
    }
