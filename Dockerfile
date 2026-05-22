# ============================================================
# SubspaceAD Dockerfile
# ============================================================
# 构建（默认 CUDA，支持 GPU/CPU 自动切换）:
#   docker build -t subspacead .
#
# 运行:
#   GPU: docker run --gpus all -p 8703:8703 subspacead
#   CPU: docker run -p 8703:8703 subspacead
# ============================================================

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# ── 系统依赖 ──
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Python 依赖 ──
# torch/torchvision 已由 base image 提供
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.timeout 120 && \
    pip install --no-cache-dir -r requirements.txt

# ── 复制项目文件 ──
COPY main.py subspace_anomaly_detector.py ./
COPY src/ ./src/
COPY models/ ./models/
COPY static/ ./static/

# ── 端口 ──
# SubspaceAD 内部通过 torch.cuda.is_available() 自动选择设备
ENV PORT=8703
EXPOSE 8703

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8703"]
