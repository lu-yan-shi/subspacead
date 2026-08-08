# SubspaceAD

基于 DINOv2 特征 + 正常图特征记忆库（memory-bank）的少样本异常检测服务，专为工业质检场景设计。仅需 1-2 张正常图像即可构建记忆库（Training-Free），通过余弦相似度逐 patch 匹配检测异常，支持多种相似度聚合与可视化模式。底层复用 DuoAD 管线（DINOv2-with-registers + CLS-patch 显著性 + 多层特征融合）。

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

打开 http://localhost:8704 — 状态栏显示当前运行设备（GPU/CPU）。

### 本地运行

```bash
pip install torch torchvision
pip install -r requirements.txt
python api.py
# → http://localhost:8704
```

---

## API 接口

### 交互式工具（`GET /`）

浏览器打开后可进行可视化测试 — 上传正常图像构建特征记忆库，再对待测图像进行异常检测，支持热力图叠加、左右对比、缺陷框标注三种可视化模式。

### 业务端点

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/train` | 使用正常图像构建特征记忆库 |
| POST | `/api/detect` | 对待测图像进行异常检测 |
| POST | `/api/reset` | 重置检测器状态 |
| GET | `/api/status` | 查看训练状态和模型信息 |

#### `POST /api/train`

**请求格式**（multipart/form-data）：

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | File[] | 是 | 正常图像（1-2 张即可） |
| `image_res` | int | 否 | 输入分辨率（默认 448） |
| `similarity_aggregation` | string | 否 | 相似度聚合：max / top1_mean / knn_weighted（默认 max） |
| `layer_fusion` | string | 否 | 多层融合：score_avg / score_max / feature_avg / feature_concat（默认 score_avg） |
| `layers` | string | 否 | 特征层索引，逗号分隔（默认 8,10,12） |
| `coreset_ratio` | float | 否 | 记忆库 coreset 比例，0.0=关闭（默认 0.0） |
| `coreset_seed` | int | 否 | coreset 采样种子（默认 42） |
| `knn_k` | int | 否 | knn_weighted 近邻数（默认 9） |
| `knn_temperature` | float | 否 | knn_weighted 逆距离加权温度（默认 1.0） |

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

## 相似度聚合（`similarity_aggregation`）

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| `max`（默认） | 取记忆库最近邻（top-1）相似度 | 通用，最快的单点聚合 |
| `top1_mean` | top 1% 近邻相似度均值 | 比 max 更稳，对离群点不敏感 |
| `knn_weighted` | top-k 近邻逆距离加权（softmax） | 更鲁棒，配合 `knn_k` / `knn_temperature` |

> 记忆库匹配基于逐 patch 余弦相似度（L2 归一化后 rescale 到 [0,1]），异常分数 = 1 − 聚合相似度。`knn_k=1` 的 `knn_weighted` 数值上等价于 `max`。

### 多层融合（`layer_fusion`）

| 方法 | 说明 |
|------|------|
| `score_avg`（默认） | 逐层分数平均 |
| `score_max` | 逐层分数取最大 |
| `feature_avg` | 层间特征平均后打分 |
| `feature_concat` | 层间特征拼接后打分 |

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
| `PORT` | `8704` | 服务端口 |
| `MESQUARE_URL` | `http://localhost:8000` | MeSquare 平台地址 |
| `BUSINESS_PREFIX` | `/api` | 业务端点前缀 |
| `MODEL_PATH` | `facebook/dinov2-with-registers-base` | DINOv2 模型（HF id 或本地目录） |
| `DEFAULT_IMAGE_RES` | `448` | 默认输入分辨率 |
| `SUBSPACE_SIMILARITY_AGGREGATION` | `max` | 相似度聚合：max / top1_mean / knn_weighted |
| `SUBSPACE_LAYER_FUSION` | `score_avg` | 多层融合方法 |
| `SUBSPACE_LAYERS` | `8,10,12` | 特征层索引 |
| `CORESET_RATIO` | `0.0` | 记忆库 coreset 比例（0=关闭） |
| `CORESET_SEED` | `42` | coreset 采样种子 |
| `KNN_K` | `9` | knn_weighted 近邻数 |
| `KNN_TEMPERATURE` | `1.0` | knn_weighted 温度 |
| `DEFAULT_LOCALIZE` | `true` | 是否启用目标定位 |
| `DEFAULT_LOCALIZATION_METHOD` | `auto` | 定位策略：auto / saliency / contour / manual / none |
| `DEFAULT_CROP_TO_ROI` | `true` | 定位后是否裁切 ROI 检测 |
| `ROI_MARGIN_RATIO` | `0.10` | ROI 扩展边距比例 |
| `ENABLE_LAYOUTAD` | `true` | 是否启用 LayoutAD GNN 结构 double-check |

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
│   └── frontend/
│       └── index.html           # 可视化测试页面
├── models/
│   ├── detector.py              # SubspaceAnomalyDetector 封装类（调用 DuoAD 管线）
│   ├── subspacead/              # 定位与可视化工具
│   │   ├── core/
│   │   │   └── localization.py  # ObjectLocalizer + ROI 裁切
│   │   └── utils/
│   │       ├── common.py        # 通用工具
│   │       └── viz.py           # 可视化工具
│   └── layoutad/                # LayoutAD GNN 结构 double-check
│       ├── inference.py
│       └── graph_check.py       # 免训练图结构校验
├── vendor/
│   └── ad-pipelines/            # vendored DuoAD 算法包（Docker 构建时 pip install）
│       └── src/ad_pipelines/
│           ├── pipelines/       # DuoAD/PatchIAD/PatchEAD 管线
│           └── utils/coreset.py # PatchCore 式贪心 coreset 采样
├── weights/                     # 可选本地 DINOv2 权重（默认 HF 在线加载）
├── examples/                    # 测试数据
├── scripts/                     # 模型下载脚本 + docker 入口
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
