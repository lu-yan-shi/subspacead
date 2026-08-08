# Changelog

## [1.3.0] — 2026-08-08

### Changed（架构迁移）
- **核心算法从 PCA 子空间迁移到 DINOv2 记忆库（memory-bank）**：底层改为 ad_pipelines 包中的 DuoAD 管线（DINOv2-with-registers + CLS-patch 显著性 + 多层融合），删除旧 PCA 实现（`models/subspacead/core/pca.py`、`extractor.py`、`patching.py`、`post_process/scoring.py`、`specular.py`）
- 评分机制：PCA 重建误差 → 记忆库逐 patch 余弦相似度聚合（`max` / `top1_mean` / `knn_weighted`）
- ad_pipelines 改为从仓库内 `vendor/ad-pipelines/` 构建，不再 git clone GitHub，镜像重建不会丢失本地算法改动
- 文档同步更新（README / CLAUDE.md 移除 PCA 描述）

### Added
- PatchCore 式 coreset 记忆库子采样：`CORESET_RATIO`（默认 0.0=关闭）
- `knn_weighted` 相似度聚合：`KNN_K`、`KNN_TEMPERATURE`
- 训练参数变化时自动重建管线并复用模型权重（不再因重训参数失效）

## [1.2.2] — 2026-06-01

### Changed
- 服务端口从 8703 改为 8704（统一各项目端口号段）

## [1.2.1] — 2026-06-01

### Fixed
- 修复 GPU 模式启动时前端仍显示 CPU 的问题：`/mse/health` 的 `get_gpu_info()` 增加 `torch.cuda.is_available()` fallback，解决 `pynvml` 未安装时 GPU 检测失效
- requirements.txt 新增 `pynvml>=11.5.0`
- 修复 Docker CPU 构建失败：`libgl1-mesa-glx` 在 Debian trixie 中已移除，替换为 `libgl1`（同步修复 GPU Dockerfile）

## [1.2.0] — 2026-06-01

### Added
- 前端状态栏已有 GPU/CPU 设备信息显示（来自 `/api/status`）

### Changed
- Docker 基础镜像统一升级至 CUDA 12.6（`pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime`）
- Docker 镜像拆分为 GPU/CPU 两个版本（`Dockerfile.gpu` + `Dockerfile.cpu`）
- 新增 `docker-compose.yml`，支持 `--profile gpu/cpu` 构建和启动
- requirements.txt 新增 `httpx`（webhook 依赖）
- README 统一为 docker-compose 构建/启动命令，补充环境和参数说明

## [1.1.0] — 2026-05-29

### Added
- MeSquare v2 监控平台接入：双前缀架构（`/mse/*` 监控 + `/api/*` 业务）
- `app/main.py`：FastAPI 应用工厂（lifespan、CORS、中间件、路由注册）
- `app/config.py`：集中配置管理（SERVICE_NAME、MESQUARE_URL、BUSINESS_PREFIX 等）
- `app/mse/`：标准监控模块
  - `router.py`：`/mse/health`、`/mse/api-info`、`/mse/metrics`、`/mse/resources`、`/mse/endpoint-metrics`、`/mse/logs`、`/mse/notify-api-change`
  - `metrics.py`：MetricsCollector + EndpointMetricsTracker + CpuSpikeMonitor
  - `logging.py`：MemoryLogHandler 内存日志采集
- `app/utils/webhook.py`：MeSquare webhook 通知器（启动/关闭/接口变更）
- `app/api/routes.py`：业务端点拆分（`/api/train`、`/api/detect`、`/api/reset`、`/api/status`）

### Changed
- **破坏性变更**：业务端点前缀从 `/` 改为 `/api`
  - `/train` → `/api/train`
  - `/detect` → `/api/detect`
  - `/reset` → `/api/reset`
  - `/status` → `/api/status`
- 前端 `app/templates/index.html` 迁移至 MeSquare 暗色主题框架，支持亮/暗主题切换
- `api.py` 入口更新：`uvicorn.run("app.main:app", ...)`
- Dockerfile CMD 更新：`uvicorn app.main:app`
- 移除旧 `app/` 下的手写路由代码

## [1.0.1] — 2026-05-29

### Changed
- 目录重组：遵循 AI-Project 标准模板
  - `main.py` → `api.py`，服务代码拆分至 `app/`
  - `subspace_anomaly_detector.py` → `models/detector.py`
  - `src/subspacead/` → `models/subspacead/`
  - `static/` → `app/templates/`
  - `datas/` → `examples/`
  - `models/dinov2-small/` → `weights/`
  - `Dockerfile` → `deploy/Dockerfile`
- 导入路径更新：移除 sys.path hack，使用相对导入
- Dockerfile 更新 COPY 路径

## [1.0.0] — 2026-05-22

### Added
- FastAPI 服务入口 `main.py`，集成 MeSquare 监控标准端点
- `POST /train` 接口：上传正常图像训练 PCA 子空间模型
- `POST /detect` 接口：上传待测图进行异常检测，返回异常分数和热力图
- `POST /reset` 接口：重置检测器状态
- `GET /status` 接口：查看当前训练状态和模型信息
- 可视化测试页面 `static/index.html`，白色极简主题
- 支持三种可视化模式：热力图叠加 (overlay)、左右对比 (side_by_side)、缺陷框标注 (bbox)
- 支持四种评分方法：重建误差 (reconstruction)、马氏距离 (mahalanobis)、欧氏距离 (euclidean)、余弦距离 (cosine)
- Dockerfile，基于 `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
- README.md 和 CHANGELOG.md

### Changed
- 更新 requirements.txt，添加 FastAPI 和监控依赖
- Dockerfile 重构：分层 COPY、添加构建注释、配置清华 pip 镜像源
- **移除 HuggingFace 自动下载回退**：本地模型不存在时改为抛出 `FileNotFoundError`，提示手动放置
- 移除 test.py（已由可视化测试页面替代）
