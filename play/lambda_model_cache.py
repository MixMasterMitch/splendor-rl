"""Lambda model cache for checkpoint management.

Provides module-level caching for S3 checkpoint downloads and loaded PyTorch
models. In the Lambda execution environment, /tmp is the only writable
filesystem and warm invocations reuse the same process — so module-level dicts
persist across requests within the same execution environment.

Requirements: 5.2, 5.3, 5.4
"""

from __future__ import annotations

import os
from typing import Any

import boto3

# Module-level caches — persist across warm Lambda invocations.
_MODEL_CACHE: dict[str, Any] = {}  # ckpt_key -> loaded model
_DOWNLOAD_CACHE: dict[str, str] = {}  # s3_key -> local /tmp path


def get_or_download_checkpoint(s3_bucket: str, s3_key: str) -> str:
    """Download checkpoint from S3 to /tmp if not already cached.

    Args:
        s3_bucket: Name of the S3 bucket containing the checkpoint.
        s3_key: Object key within the bucket.

    Returns:
        Local filesystem path to the downloaded checkpoint file.
    """
    if s3_key in _DOWNLOAD_CACHE:
        return _DOWNLOAD_CACHE[s3_key]

    basename = os.path.basename(s3_key)
    local_path = f"/tmp/{basename}"

    s3_client = boto3.client("s3")
    s3_client.download_file(s3_bucket, s3_key, local_path)

    _DOWNLOAD_CACHE[s3_key] = local_path
    return local_path


def get_or_load_model(ckpt_key: str, local_path: str, device: str) -> Any:
    """Load a PyTorch model from a checkpoint file, cached across invocations.

    Imports torch lazily to avoid cold-start penalty when the model is not
    needed (e.g. health checks, built-in opponent games).

    Args:
        ckpt_key: Unique key identifying this checkpoint (used as cache key).
        local_path: Path to the checkpoint file on the local filesystem.
        device: PyTorch device string (e.g. 'cpu').

    Returns:
        The loaded model object (checkpoint dict or nn.Module depending on
        what was saved).
    """
    if ckpt_key in _MODEL_CACHE:
        return _MODEL_CACHE[ckpt_key]

    import torch  # Lazy import — only when actually loading a model

    model = torch.load(local_path, map_location=device)
    _MODEL_CACHE[ckpt_key] = model
    return model
