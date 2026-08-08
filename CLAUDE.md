# CLAUDE.md

SubspaceAD — 少样本异常检测服务，基于 DINOv2 + 记忆库（memory-bank）的免训练建模。
已接入 MeSquare 监控平台。

## 项目概述

基于 DINOv2 特征 + 正常图特征记忆库的少样本异常检测服务，专为工业质检场景设计。仅需 1-2 张正常图像即可构建记忆库（Training-Free），通过余弦相似度逐 patch 匹配检测异常，支持多种相似度聚合与可视化模式。底层算法复用 ad_pipelines 包的 DuoAD 管线（DINOv2-with-registers + CLS-patch 显著性 + 多层融合）。

## 开发命令

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
python api.py
# → http://localhost:8704

# CLI 测试
python -c "
from models.detector import SubspaceAnomalyDetector
d = SubspaceAnomalyDetector()
d.train(['examples/template.jpg'])
score = d.detect_single('examples/test-1.jpg')
print(f'Score: {score:.4f}')
"
```

## 架构要点

- `api.py` — 服务入口（薄封装，调用 uvicorn）
- `app/main.py` — FastAPI 应用工厂（lifespan、中间件、CORS、路由注册）
- `app/config.py` — 所有配置常量（SERVICE_NAME、MESQUARE_URL、BUSINESS_PREFIX 等）
- `app/mse/` — 平台监控模块（MeSquare 热更新区域，用户勿修改）
  - `router.py` — `/mse/*` 标准监控端点
  - `metrics.py` — MetricsCollector + EndpointMetricsTracker + CpuSpikeMonitor
  - `logging.py` — MemoryLogHandler + 日志采集
- `app/api/routes.py` — 业务端点（`/api/train`、`/api/detect`、`/api/reset`、`/api/status`）
- `app/utils/webhook.py` — MeSquare webhook 通知器
- `models/detector.py` — SubspaceAnomalyDetector 封装类（调用 DuoAD 管线 + 定位/Graph 双检）
- `models/subspacead/` — 定位（localization）与可视化工具；核心算法在 vendored 的 ad_pipelines 包内
- `vendor/ad-pipelines/` — vendored DuoAD (ad_pipelines) 源码，Docker 构建时 `pip install`（含 coreset/knn_weighted 改动）
- `weights/` — 可选本地权重目录（默认走 HF 在线 `facebook/dinov2-with-registers-base`）
- `examples/` — 示例图片
- `app/frontend/index.html` — 服务管理界面（根路径 `/` 返回）
- `deploy/` — Docker 部署配置

## 双前缀架构

| 前缀 | 用途 | 示例 |
|------|------|------|
| `/mse/*` | 平台监控端点（MeSquare 管理） | `/mse/health`、`/mse/metrics` |
| `/api/*` | 业务端点（用户私有） | `/api/train`、`/api/detect` |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `8704` | 服务端口 |
| `HOST` | `0.0.0.0` | 绑定地址 |
| `MESQUARE_URL` | `http://localhost:8000` | MeSquare 平台地址 |
| `BUSINESS_PREFIX` | `/api` | 业务端点前缀 |
| `MODEL_PATH` | `facebook/dinov2-with-registers-base` | DINOv2 模型（HF id 或本地目录） |
| `DINOV3_VARIANT` | `dinov2` | 模型变体开关（compose 层）：dinov2 / dinov3-b16 / dinov3-s16 / dinov3-l16 |
| `DEFAULT_IMAGE_RES` | `448` | 默认输入分辨率 |
| `SUBSPACE_SIMILARITY_AGGREGATION` | `max` | 相似度聚合: max / top1_mean / knn_weighted |
| `SUBSPACE_LAYER_FUSION` | `score_avg` | 多层融合: score_avg / score_max / feature_avg / feature_concat |
| `SUBSPACE_LAYERS` | `8,10,12` | 特征层索引 |
| `CORESET_RATIO` | `0.0` | 记忆库 coreset 比例（0=关闭，0.05=保留 5%） |
| `CORESET_SEED` | `42` | coreset 采样种子 |
| `KNN_K` | `9` | knn_weighted 近邻数 |
| `KNN_TEMPERATURE` | `1.0` | knn_weighted 逆距离加权温度 |

## Docker

```bash
# GPU
docker-compose -f docker-compose.yml --profile gpu build
docker-compose -f docker-compose.yml --profile gpu up -d

# CPU
docker-compose -f docker-compose.yml --profile cpu build
docker-compose -f docker-compose.yml --profile cpu up -d
```
