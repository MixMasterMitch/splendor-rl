"""Tests for device resolution, device_info, and configure_device."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from agent.train.device import configure_device, device_info, resolve_device


# ---------------------------------------------------------------------------
# resolve_device()
# ---------------------------------------------------------------------------


class TestResolveDeviceCPU:
    def test_cpu_returns_cpu(self) -> None:
        assert resolve_device("cpu") == "cpu"

    def test_empty_string_without_cuda_returns_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            assert resolve_device("") == "cpu"

    def test_auto_without_cuda_returns_cpu(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            assert resolve_device("auto") == "cpu"


class TestResolveDeviceCUDA:
    def test_cuda_with_gpu_returns_cuda0(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1):
            assert resolve_device("cuda") == "cuda:0"

    def test_cuda0_with_gpu_returns_cuda0(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1):
            assert resolve_device("cuda:0") == "cuda:0"

    def test_auto_with_gpu_returns_cuda0(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("auto") == "cuda:0"

    def test_empty_with_gpu_returns_cuda0(self) -> None:
        with patch("torch.cuda.is_available", return_value=True):
            assert resolve_device("") == "cuda:0"

    def test_cuda1_with_two_gpus(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=2):
            assert resolve_device("cuda:1") == "cuda:1"


class TestResolveDeviceErrors:
    def test_cuda_without_gpu_raises(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(ValueError, match="CUDA requested"):
                resolve_device("cuda")

    def test_cuda0_without_gpu_raises(self) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(ValueError, match="CUDA requested"):
                resolve_device("cuda:0")

    def test_cuda_index_out_of_range_raises(self) -> None:
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.device_count", return_value=1):
            with pytest.raises(ValueError, match="CUDA device 5 not found"):
                resolve_device("cuda:5")

    def test_invalid_string_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported device"):
            resolve_device("tpu")

    def test_garbage_string_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported device"):
            resolve_device("not-a-device")


# ---------------------------------------------------------------------------
# device_info()
# ---------------------------------------------------------------------------


class TestDeviceInfo:
    def test_cpu_returns_expected_keys(self) -> None:
        info = device_info("cpu")
        assert info["device"] == "cpu"
        assert info["torch"] == torch.__version__
        # CPU info should NOT contain GPU-specific keys
        assert "gpu_name" not in info
        assert "gpu_vram_total_gb" not in info


# ---------------------------------------------------------------------------
# configure_device()
# ---------------------------------------------------------------------------


class TestConfigureDevice:
    def test_cpu_calls_configure_cpu_threads(self) -> None:
        with patch("agent.train.device.configure_cpu_threads") as mock_threads:
            mock_threads.return_value = {
                "cpu_count": 8,
                "torch_num_threads": 8,
                "torch_num_interop_threads": 1,
            }
            info = configure_device("cpu")
            mock_threads.assert_called_once()
            assert info["device"] == "cpu"
            assert info["torch"] == torch.__version__
            assert "cpu_count" in info

    def test_cpu_returns_device_info_keys(self) -> None:
        info = configure_device("cpu")
        assert "device" in info
        assert "torch" in info
