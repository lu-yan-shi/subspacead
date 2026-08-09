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
- `app/auth.py` — 登录认证核心（纯 stdlib：SQLite 用户表 + PBKDF2 密码哈希 + HMAC 无状态 token，**不含 fastapi** 便于本地单测）
- `app/api/security.py` — FastAPI 鉴权依赖 `require_auth(request: Request)`（**必须保留 Request 类型注解**，否则被当 query 参数）
- `app/api/auth_routes.py` — 认证端点（`/api/auth/login`、`/api/auth/me`、`/api/auth/change-password`），独立 router 免鉴权（否则被业务锁死）
- `app/api/routes.py` — 业务端点（`/api/train`、`/api/detect`、`/api/reset`、`/api/status`），router 级依赖 `require_auth` 全量鉴权
- `app/utils/webhook.py` — MeSquare webhook 通知器
- `models/detector.py` — SubspaceAnomalyDetector 封装类（调用 DuoAD 管线 + 定位/Graph 双检）
  - 尺寸适配：任意长宽比图片经 **letterbox**（保比例缩放 + 灰边填充）进入 `image_res` 方形画布，避免拉伸变形；异常图/注意力图裁边后缩回原图尺寸，分数只在内容区计算
- `models/subspacead/` — 定位（localization）与可视化工具；核心算法在 vendored 的 ad_pipelines 包内
- `vendor/ad-pipelines/` — vendored DuoAD (ad_pipelines) 源码，Docker 构建时 `pip install`（含 coreset/knn_weighted 改动）
- `weights/` — 可选本地权重目录（默认走 HF 在线 `facebook/dinov2-with-registers-base`）
- `examples/` — 示例图片
- `app/frontend/index.html` — 服务管理界面（根路径 `/` 返回；未登录经 JS 跳转 `/login`）
- `app/frontend/login.html` — 独立登录页（`/login` 返回；登录成功存 token 后跳转 `/`，已持有有效 token 自动回 `/`）
- `deploy/` — Docker 部署配置

## 双前缀架构

| 前缀 | 用途 | 示例 |
|------|------|------|
| `/mse/*` | 平台监控端点（MeSquare 管理，**不鉴权**） | `/mse/health`、`/mse/metrics` |
| `/api/*` | 业务端点（用户私有，**需登录**） | `/api/train`、`/api/detect` |
| `/api/auth/*` | 认证端点（登录/校验/改密码，免鉴权） | `/api/auth/login` |

> 登录：独立登录页 `/login`，未登录访问 `/` 或 `/api/*` 返回 401 后前端跳转 `/login`。默认账号 `admin` / `admin123`（首次启动写入 SQLite，`data/users.db`，Docker 内 `/app/data` 持久卷，重启不重置）。登录后请尽快在界面「修改密码」。token 有效期 24h，前端 localStorage 保存，`/api/*` 需 `Authorization: Bearer <token>`。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `8704` | 服务端口 |
| `HOST` | `0.0.0.0` | 绑定地址 |
| `MESQUARE_URL` | `http://localhost:8000` | MeSquare 平台地址 |
| `BUSINESS_PREFIX` | `/api` | 业务端点前缀 |
| `MODEL_PATH` | `facebook/dinov2-with-registers-base` | DINOv2 模型（HF id 或本地目录） |
| `DEFAULT_IMAGE_RES` | `448` | 默认输入分辨率 |
| `SUBSPACE_SIMILARITY_AGGREGATION` | `max` | 相似度聚合: max / top1_mean / knn_weighted |
| `SUBSPACE_LAYER_FUSION` | `score_avg` | 多层融合: score_avg / score_max / feature_avg / feature_concat |
| `SUBSPACE_LAYERS` | `8,10,12` | 特征层索引 |
| `CORESET_RATIO` | `0.0` | 记忆库 coreset 比例（0=关闭，0.05=保留 5%） |
| `CORESET_SEED` | `42` | coreset 采样种子 |
| `KNN_K` | `9` | knn_weighted 近邻数 |
| `KNN_TEMPERATURE` | `1.0` | knn_weighted 逆距离加权温度 |
| `AUTH_DB_PATH` | `data/users.db` | SQLite 账号库路径（Docker 内 `/app/data` 持久卷） |
| `AUTH_SECRET` | `subspacead-dev-secret-change-me` | token 签名密钥（生产请覆盖） |
| `AUTH_TOKEN_EXPIRE_HOURS` | `24` | token 有效期（小时） |
| `AUTH_ADMIN_USER` | `admin` | 初始管理员用户名 |
| `AUTH_ADMIN_PASS` | `admin123` | 初始管理员密码（首次启动写入 DB） |

## Docker

```bash
# GPU
docker-compose -f docker-compose.yml --profile gpu build
docker-compose -f docker-compose.yml --profile gpu up -d

# CPU
docker-compose -f docker-compose.yml --profile cpu build
docker-compose -f docker-compose.yml --profile cpu up -d
```
