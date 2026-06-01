# SubspaceAD

基于 DINOv2 特征和 PCA 子空间建模的少样本异常检测服务，专为工业质检场景设计。仅需 1-2 张正常图像即可训练，支持多种评分方法和可视化模式。

---

## Quick Start

### GPU (requires NVIDIA GPU + nvidia-container-toolkit)

```bash
docker-compose -f docker-compose.yml --profile gpu build
docker-compose -f docker-compose.yml --profile gpu up -d
```

### CPU

```bash
docker-compose -f docker-compose.yml --profile cpu build
docker-compose -f docker-compose.yml --profile cpu up -d
```

Open http://localhost:8703 — the status bar shows the current device (GPU/CPU).

### Without Docker

```bash
pip install torch torchvision
pip install -r requirements.txt
python api.py
# → http://localhost:8703
```

---

## API

### Interactive Tool (`GET /`)

Open in browser for a visual testing page — upload normal images to train the PCA model, then detect anomalies on test images with heatmap overlay / side-by-side comparison / bounding box annotation.

### Business Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/train` | Train PCA subspace model with normal images |
| POST | `/api/detect` | Detect anomalies on test image |
| POST | `/api/reset` | Reset detector state |
| GET | `/api/status` | View training status and model info |

#### `POST /api/train`

**Request** (multipart/form-data):

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `files` | File[] | Yes | Normal images (1-2 images sufficient) |
| `image_res` | int | No | Input resolution (default 512) |
| `pca_ev` | float | No | PCA variance ratio 0-1 (default 0.99) |
| `score_method` | string | No | Scoring method (default reconstruction) |

#### `POST /api/detect`

**Request** (multipart/form-data):

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | File | Yes | Test image |
| `viz_mode` | string | No | overlay / side_by_side / bbox (default overlay) |
| `return_heatmap` | bool | No | Return heatmap (default true) |
| `bbox_threshold` | float | No | Defect threshold for bbox mode (default 0.5) |

### Monitoring Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/mse/health` | Health check |
| GET | `/mse/api-info` | Service metadata |
| GET | `/mse/metrics` | Request statistics |
| GET | `/mse/resources` | CPU/memory/disk/GPU utilization |
| GET | `/mse/endpoint-metrics` | Per-endpoint metrics |
| GET | `/mse/logs` | Recent logs |

---

## Scoring Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `reconstruction` (default) | PCA reconstruction error | General use |
| `mahalanobis` | Mahalanobis distance | Higher sensitivity |
| `euclidean` | Euclidean distance | Simplicity |
| `cosine` | Cosine distance | Scale-invariant |

## Visualization Modes

| Mode | Description |
|------|-------------|
| `overlay` (default) | Heatmap overlay on original image |
| `side_by_side` | Original + heatmap side by side |
| `bbox` | Defect bounding box annotation |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8703` | Service port |
| `MESQUARE_URL` | `http://localhost:8000` | MeSquare platform URL |
| `BUSINESS_PREFIX` | `/api` | Business endpoint prefix |
| `DEFAULT_IMAGE_RES` | `512` | Default input resolution |
| `DEFAULT_PCA_EV` | `0.99` | PCA variance ratio |
| `DEFAULT_SCORE_METHOD` | `reconstruction` | Default scoring method |

---

## Project Structure

```
SubspaceAD/
├── api.py                       # Service entry point
├── app/
│   ├── main.py                  # FastAPI app factory (lifespan, middleware, CORS)
│   ├── config.py                # Centralized configuration
│   ├── mse/
│   │   ├── router.py            # /mse/* monitoring endpoints
│   │   ├── metrics.py           # MetricsCollector + EndpointMetricsTracker + CpuSpikeMonitor
│   │   └── logging.py           # MemoryLogHandler + log capture
│   ├── api/
│   │   └── routes.py            # Business endpoints (/api/train, /api/detect, /api/reset, /api/status)
│   ├── utils/
│   │   └── webhook.py           # MeSquare webhook notifier
│   └── templates/
│       └── index.html           # Visual testing page
├── models/
│   ├── detector.py              # SubspaceAnomalyDetector wrapper
│   └── subspacead/              # Core algorithm modules
│       ├── core/
│       │   ├── extractor.py     # DINOv2 feature extraction
│       │   ├── pca.py           # GPU-accelerated PCA
│       │   └── patching.py      # Image patching
│       ├── post_process/
│       │   ├── scoring.py       # Anomaly score calculation
│       │   └── specular.py      # Specular highlight filter
│       └── utils/
│           ├── common.py        # Common utilities
│           └── viz.py           # Visualization utilities
├── weights/                     # DINOv2 model weights
├── examples/                    # Test data
├── deploy/
│   ├── Dockerfile.gpu           # GPU image (CUDA 12.6)
│   └── Dockerfile.cpu           # CPU image (PyTorch CPU)
├── docker-compose.yml           # Compose profiles: gpu / cpu
├── requirements.txt
└── README.md
```

---

## License

MIT License
