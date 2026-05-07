#!/bin/bash
# Build a Lambda layer with PyTorch CPU-only and numpy for Python 3.11 Linux x86_64.
# The layer is built into infra/layer/python/ which CDK will zip and upload.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAYER_DIR="$SCRIPT_DIR/python"

echo "Cleaning previous layer build..."
rm -rf "$LAYER_DIR"
mkdir -p "$LAYER_DIR"

echo "Installing torch (CPU-only) into layer..."
pip install \
    --target "$LAYER_DIR" \
    --no-deps \
    "torch==2.5.1+cpu" --index-url https://download.pytorch.org/whl/cpu

echo "Installing numpy into layer..."
pip install \
    --target "$LAYER_DIR" \
    --no-deps \
    "numpy>=1.26,<2.0"

echo "Installing torch dependencies (non-CUDA) into layer..."
pip install \
    --target "$LAYER_DIR" \
    --no-deps \
    typing_extensions sympy filelock jinja2 networkx markupsafe mpmath

# Remove unnecessary files to reduce layer size
echo "Trimming layer..."
find "$LAYER_DIR" -name "*.dist-info" -type d -exec rm -rf {} + 2>/dev/null || true
find "$LAYER_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$LAYER_DIR" -name "*.pyc" -delete 2>/dev/null || true
find "$LAYER_DIR" -name "tests" -type d -exec rm -rf {} + 2>/dev/null || true
find "$LAYER_DIR" -name "test" -type d -exec rm -rf {} + 2>/dev/null || true
# Remove CUDA/cuDNN libs (CPU-only build shouldn't have them, but just in case)
find "$LAYER_DIR" -name "libcudart*" -delete 2>/dev/null || true
find "$LAYER_DIR" -name "libcudnn*" -delete 2>/dev/null || true
find "$LAYER_DIR" -name "libnvrtc*" -delete 2>/dev/null || true

LAYER_SIZE=$(du -sh "$LAYER_DIR" | cut -f1)
echo "Layer size: $LAYER_SIZE"
echo "Done. Layer built at: $LAYER_DIR"
