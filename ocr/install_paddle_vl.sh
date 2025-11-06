#!/bin/bash
# Installation script for PaddleOCR-VL with GPU support
# Must be run with venv_vl activated

set -e  # Exit on error

echo "=========================================="
echo "Installing PaddlePaddle GPU for OCR-VL"
echo "=========================================="

# Step 1: Install PaddlePaddle GPU (CUDA 12.6)
echo ""
echo "[1/3] Installing PaddlePaddle GPU 3.2.0..."
python -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Step 2: Install/upgrade PaddleOCR with doc-parser extras
echo ""
echo "[2/3] Installing PaddleOCR with doc-parser..."
python -m pip install -U "paddleocr[doc-parser]"

# Step 3: Install custom safetensors with Paddle framework support
echo ""
echo "[3/3] Installing custom safetensors..."
python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

echo ""
echo "=========================================="
echo "Installation complete!"
echo "=========================================="
echo ""
echo "Test GPU availability:"
echo "  python -c 'import paddle; print(\"GPU available:\", paddle.is_compiled_with_cuda())'"
echo ""
echo "Start OCR-VL service:"
echo "  python ocr_vl_service.py"
echo ""
