"""FastAPI application — lifespan, middleware, CORS, route registration."""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from .config import (
    DEFAULT_IMAGE_RES,
    DEFAULT_PCA_EV,
    DEFAULT_SCORE_METHOD,
    MONITOR_PATHS,
    SERVICE_DESCRIPTION,
    SERVICE_NAME,
    SERVICE_VERSION,
)
from .mse.logging import init_log_capture, shutdown_log_capture
from .mse.metrics import CpuSpikeMonitor, EndpointMetricsTracker, MetricsCollector
from .utils.webhook import notify_mesquare_api_change

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Service lifespan: load model on startup, notify MeSquare of lifecycle events."""
    init_log_capture()

    app.state.start_time = time.time()
    app.state.metrics_collector = MetricsCollector()
    app.state.endpoint_metrics = EndpointMetricsTracker()
    app.state.cpu_monitor = CpuSpikeMonitor()
    app.state.cpu_monitor.start()

    from models.detector import SubspaceAnomalyDetector

    app.state.detector = SubspaceAnomalyDetector(
        image_res=DEFAULT_IMAGE_RES,
        pca_ev=DEFAULT_PCA_EV,
        score_method=DEFAULT_SCORE_METHOD,
    )
    app.state.is_trained = False
    app.state.train_info = {}

    logger.info("SubspaceAnomalyDetector initialized (device=%s)", app.state.detector.device)

    await notify_mesquare_api_change("service_restarted")
    yield
    await notify_mesquare_api_change("service_shutdown")
    app.state.cpu_monitor.stop()
    shutdown_log_capture()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
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
                request.url.path, request.method,
                latency * 1000, status_code >= 400,
            )
            collector.dec_active()

    @app.get("/", summary="Root", tags=["General"])
    async def root():
        templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        return FileResponse(os.path.join(templates_dir, "index.html"))

    from .mse.router import mse_router
    app.include_router(mse_router)

    from .api.routes import business_router
    app.include_router(business_router)

    return app


app = create_app()
