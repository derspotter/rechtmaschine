#!/bin/bash

# Monitor for Ollama offloading to RAM
# Run this while Ollama is processing

echo "Monitoring for Ollama RAM offloading..."
echo "Press Ctrl+C to stop"
echo ""

OLLAMA_PID=$(ps aux | grep "ollama runner" | grep -v grep | awk '{print $2}')

if [ -z "$OLLAMA_PID" ]; then
    echo "Error: Ollama process not found"
    exit 1
fi

echo "Monitoring Ollama PID: $OLLAMA_PID"
echo ""

while true; do
    timestamp=$(date +"%H:%M:%S")

    # Get VRAM usage
    vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)

    # Get Ollama RAM usage
    ram=$(ps -p $OLLAMA_PID -o rss= 2>/dev/null)
    ram_mb=$((ram / 1024))

    # Get CPU usage
    cpu=$(ps -p $OLLAMA_PID -o %cpu= 2>/dev/null)

    # Color coding
    if [ $ram_mb -gt 8000 ]; then
        ram_color="\033[1;31m"  # Red - heavy offloading
        warning=" ⚠️ HEAVY RAM OFFLOADING!"
    elif [ $ram_mb -gt 5000 ]; then
        ram_color="\033[1;33m"  # Yellow - moderate offloading
        warning=" ⚠️ Moderate offloading"
    else
        ram_color="\033[0;32m"  # Green - normal
        warning=""
    fi
    reset="\033[0m"

    echo -e "$timestamp | VRAM: ${vram} MiB | GPU: ${gpu_util}% | ${ram_color}RAM: ${ram_mb} MB${reset} | CPU: ${cpu}%${warning}"

    sleep 1
done
