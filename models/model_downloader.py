"""SubspaceAD 模型自动下载器。

构建时自动从 ModelScope Hub 下载权重文件。
Token 从项目根目录 ModelScope_token 文件读取。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("subspacead.downloader")

MODELSCOPE_MODEL_ID = "Zhongz/subspacead-model"
MODELSCOPE_FILES = [
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "README.md",
]


def _read_token_file() -> Optional[str]:
    search_paths = [
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        os.getcwd(),
    ]
    for base in search_paths:
        token_path = os.path.join(base, "ModelScope_token")
        if os.path.isfile(token_path):
            with open(token_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        return line
    return None


def ensure_weights(weights_dir: str = "weights", token: Optional[str] = None, model_id: str = MODELSCOPE_MODEL_ID):
    """确保所有权重文件存在，不存在则自动从 ModelScope 下载。"""
    weights_dir = os.path.abspath(weights_dir)
    missing = [f for f in MODELSCOPE_FILES if not os.path.isfile(os.path.join(weights_dir, f))]

    if not missing:
        logger.info("所有权重文件已存在: %s", weights_dir)
        return

    token = token or os.environ.get("MODELSCOPE_TOKEN") or _read_token_file()
    if not token:
        raise FileNotFoundError(
            f"权重文件缺失: {missing}\n"
            f"请设置 MODELSCOPE_TOKEN 环境变量或创建 ModelScope_token 文件。"
        )

    os.makedirs(weights_dir, exist_ok=True)

    from modelscope.hub.api import HubApi
    from modelscope.hub.file_download import model_file_download

    api = HubApi()
    api.login(token)

    for filename in missing:
        logger.info("从 ModelScope 下载 %s ...", filename)
        local_path = model_file_download(
            model_id=model_id,
            file_path=filename,
            cache_dir=weights_dir,
            revision="master",
        )
        target_path = os.path.join(weights_dir, filename)
        if os.path.abspath(local_path) != os.path.abspath(target_path):
            if os.path.exists(target_path):
                os.remove(target_path)
            try:
                os.symlink(local_path, target_path)
            except OSError:
                import shutil
                shutil.copy2(local_path, target_path)

        file_size = os.path.getsize(target_path) / 1024 / 1024
        logger.info("下载完成: %s (%.1f MB)", target_path, file_size)
