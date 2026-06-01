# CLAUDE.md

SubspaceAD — 少样本异常检测服务，基于 DINOv2 + PCA 子空间建模。
已接入 MeSquare 监控平台。

## 项目概述

基于 DINOv2 + PCA 子空间建模的少样本异常检测服务，专为工业质检场景设计。仅需 1-2 张正常图像即可训练，支持多种评分方法和可视化模式。

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
- `models/detector.py` — SubspaceAnomalyDetector 封装类
- `models/subspacead/` — 核心算法（特征提取、PCA、评分）
- `weights/` — DINOv2-small 模型权重
- `examples/` — 示例图片
- `app/templates/index.html` — 服务管理界面（根路径 `/` 返回）
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
| `DEFAULT_IMAGE_RES` | `512` | 默认输入分辨率 |
| `DEFAULT_PCA_EV` | `0.99` | PCA 保留方差比例 |
| `DEFAULT_SCORE_METHOD` | `reconstruction` | 默认评分方法 |

## Docker

```bash
# GPU
docker-compose -f docker-compose.yml --profile gpu build
docker-compose -f docker-compose.yml --profile gpu up -d

# CPU
docker-compose -f docker-compose.yml --profile cpu build
docker-compose -f docker-compose.yml --profile cpu up -d
```
