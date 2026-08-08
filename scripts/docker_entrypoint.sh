#!/bin/bash
# Docker entrypoint: ensure DINOv2 weights are present, then start uvicorn.
#
# 之前的问题: 仅检查 config.json 存在就跳过下载。HF 镜像在下载大权重
# (model.safetensors ~346MB) 时可能中断, 只留下小的配置文件, 形成
# 「有配置无权重」的坏状态, 之后每次重启都跳过下载, 训练时报
# "no file named pytorch_model.bin, model.safetensors ... found"。
#
# 现在以权重文件本身为准: 缺失或体积过小(残留 .incomplete)则重新下载,
# 全部镜像失败则显式报错退出, 不再启动一个注定训练失败的实例。
set -u

MODEL_DIR="${MODEL_PATH:-/app/weights/dinov2}"
MODEL_ID="facebook/dinov2-with-registers-base"
# DINOv2-with-registers-base 权重约 346MB; 小于 100MiB 视为无效(只下了配置/半截文件)
MIN_WEIGHT_BYTES=$((100 * 1024 * 1024))

has_valid_weight() {
    for f in "$MODEL_DIR/model.safetensors" "$MODEL_DIR/pytorch_model.bin"; do
        if [ -f "$f" ] && [ "$(stat -c %s "$f" 2>/dev/null || echo 0)" -ge "$MIN_WEIGHT_BYTES" ]; then
            return 0
        fi
    done
    return 1
}

if has_valid_weight; then
    echo "Model ready at $MODEL_DIR"
    ls -la "$MODEL_DIR"
    exec uvicorn app.main:app --host 0.0.0.0 --port 8704
fi

echo "Model weights missing or incomplete at $MODEL_DIR, downloading..."
mkdir -p "$MODEL_DIR"

# 依次尝试镜像; snapshot_download 会跳过已存在文件并续传残缺文件,
# 因此重复启动是安全的。全部失败则退出, 交给 Docker restart 策略重试。
for MIRROR in "https://hf-mirror.com" "https://huggingface.co"; do
    echo "  Trying: $MIRROR"
    if HF_ENDPOINT="$MIRROR" python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${MODEL_ID}', local_dir='${MODEL_DIR}', local_dir_use_symlinks=False)
"; then
        if has_valid_weight; then
            echo "Downloaded successfully."
            break
        fi
        echo "  Download finished but weight file missing/incomplete — retrying next mirror..."
    else
        echo "  Failed, trying next mirror..."
    fi
done

if has_valid_weight; then
    echo "Model ready at $MODEL_DIR"
    ls -la "$MODEL_DIR"
else
    echo "ERROR: failed to download DINOv2 weights to $MODEL_DIR." >&2
    echo "  Check network / mirror availability, then restart the container." >&2
    echo "  Or mount the model files at $MODEL_DIR." >&2
    exit 1
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8704
