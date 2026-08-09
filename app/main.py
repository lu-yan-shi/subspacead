"""FastAPI application — lifespan, middleware, CORS, route registration."""

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from .auth import init_auth
from .config import (
    DEFAULT_IMAGE_RES,
    ENABLE_LAYOUTAD,
    LAYOUTAD_CHECKPOINT,
    LAYOUTAD_MASK2FORMER_CONFIG,
    LAYOUTAD_MASK2FORMER_WEIGHTS,
    LAYOUTAD_PIPELINE_MODE,
    SUBSPACE_CORESET_RATIO,
    SUBSPACE_CORESET_SEED,
    SUBSPACE_LAYER_FUSION,
    SUBSPACE_LAYERS,
    SUBSPACE_KNN_K,
    SUBSPACE_KNN_TEMPERATURE,
    SUBSPACE_SIMILARITY_AGGREGATION,
    MODEL_PATH,
    MONITOR_PATHS,
    PUBLIC_BASE_URL,
    SERVICE_DESCRIPTION,
    SERVICE_NAME,
    SERVICE_PORT,
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
    init_auth()

    app.state.start_time = time.time()
    app.state.metrics_collector = MetricsCollector()
    app.state.endpoint_metrics = EndpointMetricsTracker()
    app.state.cpu_monitor = CpuSpikeMonitor()
    app.state.cpu_monitor.start()

    try:
        from models.detector import SubspaceAnomalyDetector

        app.state.detector = SubspaceAnomalyDetector(
            model_path=MODEL_PATH,
            image_res=DEFAULT_IMAGE_RES,
            similarity_aggregation=SUBSPACE_SIMILARITY_AGGREGATION,
            layer_fusion=SUBSPACE_LAYER_FUSION,
            layers=SUBSPACE_LAYERS,
            coreset_ratio=SUBSPACE_CORESET_RATIO,
            coreset_seed=SUBSPACE_CORESET_SEED,
            knn_k=SUBSPACE_KNN_K,
            knn_temperature=SUBSPACE_KNN_TEMPERATURE,
            enable_layoutad=ENABLE_LAYOUTAD,
            layoutad_checkpoint=LAYOUTAD_CHECKPOINT,
            layoutad_mask2former_config=LAYOUTAD_MASK2FORMER_CONFIG,
            layoutad_mask2former_weights=LAYOUTAD_MASK2FORMER_WEIGHTS,
        )
        if not app.state.detector.weights_ready():
            model_path = app.state.detector.model_path
            logger.error(
                "Model weights not found at %s — detector disabled. 请检查入口脚本/模型 volume。",
                model_path,
            )
            app.state.detector = None
            app.state.is_trained = False
            app.state.train_info = {"error": f"模型权重缺失: {model_path}"}
        else:
            app.state.is_trained = False
            app.state.train_info = {}
            logger.info("SubspaceAnomalyDetector initialized (device=%s)", app.state.detector.device)
    except Exception as e:
        logger.error("Failed to initialize detector: %s", e)
        app.state.detector = None
        app.state.is_trained = False
        app.state.train_info = {"error": str(e)}

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
        # 显式白名单：allow_origins="*" 与 allow_credentials=True 组合会被浏览器拒绝。
        # 服务自身 origin + 本机访问可覆盖管理页面与内网调用方。
        allow_origins=[
            PUBLIC_BASE_URL,
            f"http://localhost:{SERVICE_PORT}",
            f"http://127.0.0.1:{SERVICE_PORT}",
        ],
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
        frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
        return FileResponse(os.path.join(frontend_dir, "index.html"))

    @app.get("/login", summary="Login", tags=["General"])
    async def login_page():
        frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")
        return FileResponse(os.path.join(frontend_dir, "login.html"))

    from .mse.router import mse_router
    app.include_router(mse_router)

    from .api.auth_routes import auth_router
    app.include_router(auth_router)

    from .api.routes import business_router
    app.include_router(business_router)

    return app


app = create_app()
