#!/bin/bash
# Upgrade NVIDIA driver to latest version for CUDA 12.6+ support

set -e

echo "=========================================="
echo "Upgrading NVIDIA Driver for CUDA 12.6+"
echo "=========================================="
echo ""
echo "Current driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'Not found')"
echo ""

# Add NVIDIA CUDA repository for latest drivers
echo "[1/4] Adding NVIDIA CUDA repository..."
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

echo ""
echo "[2/4] Updating package list..."
sudo apt-get update

echo ""
echo "[3/4] Upgrading NVIDIA driver..."
sudo apt-get install -y cuda-drivers

echo ""
echo "[4/4] Cleanup..."
sudo apt-get autoremove -y

echo ""
echo "=========================================="
echo "Driver upgrade complete!"
echo "=========================================="
echo ""
echo "IMPORTANT: You must REBOOT for the new driver to take effect:"
echo "  sudo reboot"
echo ""
echo "After reboot, check driver version:"
echo "  nvidia-smi"
echo ""
