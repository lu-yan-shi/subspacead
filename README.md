# SubspaceAD

基于 DINOv2 特征和 PCA 子空间建模的少样本异常检测服务。提供 FastAPI 接口和可视化测试页面，专为工业质检场景设计。

## 特性

- **少样本学习** — 仅需 1-2 张正常图像即可训练
- **DINOv2 特征提取** — 基于 Vision Transformer 的自监督特征
- **PCA 子空间建模** — GPU 加速的两遍式流式 PCA
- **四种评分方法** — 重建误差、马氏距离、欧氏距离、余弦距离
- **三种可视化模式** — 热力图叠加、左右对比、缺陷框标注
- **参数可调** — 分辨率、PCA 方差比、评分方法均可调节
- **开箱即用** — 已配置本地模型 (DINOv2-small)

## 快速开始

### Docker（推荐）

```bash
# GPU 模式
docker build -t subspacead .
docker run -d --gpus all -p 8703:8703 subspacead

# CPU 模式
docker run -d -p 8703:8703 subspacead
```

打开 http://localhost:8703 进入测试页面。

### 本地运行

```bash
pip install -r requirements.txt
python api.py
```

依赖：Python 3.8+，PyTorch 1.8+，transformers，OpenCV

## API 文档

启动后访问 http://localhost:8703/docs 查看 Swagger UI。

### POST /train

训练 PCA 子空间模型。

**请求格式：** `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | File[] | 是 | 正常图像（无缺陷的模板图，1 张即可） |
| `image_res` | int | 否 | 输入分辨率（默认 512） |
| `pca_ev` | float | 否 | PCA 保留方差比例 0-1（默认 0.99） |
| `score_method` | string | 否 | 评分方法（默认 reconstruction） |

**响应：**

```json
{
  "success": true,
  "pca_components": 45,
  "feature_dim": 768,
  "grid_size": [32, 32],
  "num_templates": 2,
  "training_time_ms": 5123
}
```

### POST /detect

对上传图像进行异常检测。**需要先调用 /train。**

**请求格式：** `multipart/form-data`

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 待检测图像 |
| `viz_mode` | string | 否 | 可视化模式（默认 overlay） |
| `return_heatmap` | bool | 否 | 是否返回热力图（默认 true） |
| `bbox_threshold` | float | 否 | 缺陷检测阈值（仅 bbox 模式，默认 0.5） |

**响应：**

```json
{
  "success": true,
  "anomaly_score": 0.8732,
  "is_anomaly": true,
  "threshold": 0.3,
  "inference_time_ms": 312,
  "heatmap": "base64...",
  "visualization": "base64...",
  "viz_mode": "overlay"
}
```

### 监控端点

| 路径 | 说明 |
|------|------|
| `GET /health` | 健康检查 |
| `GET /api-info` | 服务元信息 |
| `GET /metrics` | 请求统计 |
| `GET /resources` | 资源利用率（CPU/内存/磁盘/GPU） |
| `GET /endpoint-metrics` | 各端点统计 |
| `GET /logs` | 服务日志 |
| `GET /status` | 检测器状态 |

## 可视化模式

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `overlay`（默认） | 原图叠加热力图 | 直观显示缺陷位置 |
| `side_by_side` | 左边原图，右边叠加 | 对比观察变化 |
| `bbox` | 原图 + 红色缺陷框 | 报告展示、快速定位 |

## 评分方法

| 方法 | 说明 | 特点 |
|------|------|------|
| `reconstruction`（默认） | PCA 重建误差 | 通用场景 |
| `mahalanobis` | 马氏距离 | 对异常更敏感 |
| `euclidean` | 欧氏距离 | 计算最简单 |
| `cosine` | 余弦距离 | 对尺度不敏感 |

## 参数调优

| 参数 | 作用 | 建议 |
|------|------|------|
| `image_res` | 输入分辨率 | 256-384 快速，512-768 精细 |
| `pca_ev` | PCA 方差保留比例 | 0.95-0.99，越高保留细节越多 |
| `score_method` | 评分方法 | reconstruction 通用，mahalanobis 更敏感 |
| 异常阈值 | 判断是否异常（默认 0.3） | 严格 0.2，宽松 0.5 |

## 项目结构

```
SubspaceAD/
├── api.py                         # FastAPI 服务入口
├── app/                           # 服务模块
│   ├── main.py                    # FastAPI app 创建、中间件
│   ├── config.py                  # 配置常量
│   ├── schemas.py                 # Pydantic 模型
│   ├── monitoring.py              # 监控/指标收集
│   ├── routes.py                  # 路由处理器
│   └── templates/
│       └── index.html             # 可视化测试页面
├── models/                        # 模型代码
│   ├── detector.py               # 核心检测器
│   └── subspacead/                # 核心模块
│       ├── core/
│       │   ├── extractor.py       # DINOv2 特征提取
│       │   ├── pca.py             # GPU 加速 PCA
│       │   └── patching.py        # 图像分块
│       ├── post_process/
│       │   ├── scoring.py         # 异常分数计算
│       │   └── specular.py        # 高光滤波
│       └── utils/
│           ├── common.py          # 通用工具
│           └── viz.py             # 可视化工具
├── weights/                       # DINOv2 模型权重
├── examples/                      # 测试数据
├── deploy/
│   └── Dockerfile
├── requirements.txt
├── CHANGELOG.md
└── README.md
```

## 技术说明

- 基于 DINOv2 (Vision Transformer) 特征提取
- GPU 加速的两遍式 PCA（均值 → 协方差 → 特征分解）
- GPU 自动检测：有 CUDA 则用 GPU，否则退回 CPU
- 已配置本地模型 `weights/`，无需联网下载
- 单文件上限：50 MB
- 支持格式：PNG、JPG、JPEG、BMP、TIFF

## 许可

MIT License
