# SubspaceAD

基于 DINOv2 特征和 PCA 子空间建模的少样本异常检测服务，专为工业质检场景设计。仅需 1-2 张正常图像即可训练，支持多种评分方法和可视化模式。

---

## 快速开始

### GPU 模式（需 NVIDIA GPU + nvidia-container-toolkit）

```bash
docker-compose -f docker-compose.yml --profile gpu build
docker-compose -f docker-compose.yml --profile gpu up -d
```

### CPU 模式

```bash
docker-compose -f docker-compose.yml --profile cpu build
docker-compose -f docker-compose.yml --profile cpu up -d
```

打开 http://localhost:8703 — 状态栏显示当前运行设备（GPU/CPU）。

### 本地运行

```bash
pip install torch torchvision
pip install -r requirements.txt
python api.py
# → http://localhost:8703
```

---

## API 接口

### 交互式工具（`GET /`）

浏览器打开后可进行可视化测试 — 上传正常图像训练 PCA 模型，再对待测图像进行异常检测，支持热力图叠加、左右对比、缺陷框标注三种可视化模式。

### 业务端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/train` | 使用正常图像训练 PCA 子空间模型 |
| POST | `/api/detect` | 对待测图像进行异常检测 |
| POST | `/api/reset` | 重置检测器状态 |
| GET | `/api/status` | 查看训练状态和模型信息 |

#### `POST /api/train`

**请求格式**（multipart/form-data）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | File[] | 是 | 正常图像（1-2 张即可） |
| `image_res` | int | 否 | 输入分辨率（默认 512） |
| `pca_ev` | float | 否 | PCA 方差保留比例 0-1（默认 0.99） |
| `score_method` | string | 否 | 评分方法（默认 reconstruction） |

#### `POST /api/detect`

**请求格式**（multipart/form-data）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 待检测图像 |
| `viz_mode` | string | 否 | overlay / side_by_side / bbox（默认 overlay） |
| `return_heatmap` | bool | 否 | 是否返回热力图（默认 true） |
| `bbox_threshold` | float | 否 | bbox 模式缺陷阈值（默认 0.5） |

### 监控端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/mse/health` | 健康检查 |
| GET | `/mse/api-info` | 服务元信息 |
| GET | `/mse/metrics` | 请求统计 |
| GET | `/mse/resources` | CPU/内存/磁盘/GPU 利用率 |
| GET | `/mse/endpoint-metrics` | 各端点独立指标 |
| GET | `/mse/logs` | 最近日志 |

---

## 评分方法

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `reconstruction`（默认） | PCA 重建误差 | 通用场景 |
| `mahalanobis` | 马氏距离 | 对异常更敏感 |
| `euclidean` | 欧氏距离 | 计算最简单 |
| `cosine` | 余弦距离 | 对尺度不敏感 |

## 可视化模式

| 模式 | 说明 |
|------|------|
| `overlay`（默认） | 原图叠加热力图 |
| `side_by_side` | 原图 + 热力图左右对比 |
| `bbox` | 原图 + 缺陷框标注 |

---

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PORT` | `8703` | 服务端口 |
| `MESQUARE_URL` | `http://localhost:8000` | MeSquare 平台地址 |
| `BUSINESS_PREFIX` | `/api` | 业务端点前缀 |
| `DEFAULT_IMAGE_RES` | `512` | 默认输入分辨率 |
| `DEFAULT_PCA_EV` | `0.99` | PCA 方差保留比例 |
| `DEFAULT_SCORE_METHOD` | `reconstruction` | 默认评分方法 |

---

## 项目结构

```
SubspaceAD/
├── api.py                       # 服务入口
├── app/
│   ├── main.py                  # FastAPI 应用工厂（lifespan、中间件、CORS）
│   ├── config.py                # 集中配置管理
│   ├── mse/
│   │   ├── router.py            # /mse/* 监控端点
│   │   ├── metrics.py           # MetricsCollector + EndpointMetricsTracker + CpuSpikeMonitor
│   │   └── logging.py           # MemoryLogHandler + 日志采集
│   ├── api/
│   │   └── routes.py            # 业务端点（/api/train、/api/detect、/api/reset、/api/status）
│   ├── utils/
│   │   └── webhook.py           # MeSquare webhook 通知器
│   └── templates/
│       └── index.html           # 可视化测试页面
├── models/
│   ├── detector.py              # SubspaceAnomalyDetector 封装类
│   └── subspacead/              # 核心算法模块
│       ├── core/
│       │   ├── extractor.py     # DINOv2 特征提取
│       │   ├── pca.py           # GPU 加速 PCA
│       │   └── patching.py      # 图像分块
│       ├── post_process/
│       │   ├── scoring.py       # 异常分数计算
│       │   └── specular.py      # 高光滤波
│       └── utils/
│           ├── common.py        # 通用工具
│           └── viz.py           # 可视化工具
├── weights/                     # DINOv2 模型权重
├── examples/                    # 测试数据
├── deploy/
│   ├── Dockerfile.gpu           # GPU 镜像（CUDA 12.6）
│   └── Dockerfile.cpu           # CPU 镜像（PyTorch CPU）
├── docker-compose.yml           # Compose profiles: gpu / cpu
├── requirements.txt
└── README.md
```

---

## 许可

MIT License
