# PaddleOCR-VL with vLLM Acceleration

Optimized setup for RTX 3060 (12GB VRAM) using Docker and vLLM server.

## Architecture

```
┌─────────────────────┐
│  OCR Service        │
│  (port 9005)        │
│  Python/FastAPI     │
└──────────┬──────────┘
           │
           │ HTTP requests
           │
┌──────────▼──────────┐
│  vLLM Server        │
│  (port 8118)        │
│  Docker Container   │
│  GPU-accelerated    │
└─────────────────────┘
```

## Quick Start

### 1. Start vLLM Server (Docker)

```bash
cd /home/jayjag/rechtmaschine/ocr
docker-compose up -d
```

The vLLM server will:
- Download the 13GB Docker image (first time only)
- Load PaddleOCR-VL-0.9B model (~2-3 minutes)
- Listen on port 8118
- Use 80% GPU memory (9.6GB of 12GB on RTX 3060)
- Handle up to 128 concurrent sequences

Check status:
```bash
docker-compose logs -f
```

Wait for: `[INFO] Server started successfully`

### 2. Start OCR Service (Python)

In a new terminal:
```bash
cd /home/jayjag/rechtmaschine/ocr
source venv_vl/bin/activate
python ocr_vl_service_vllm.py
```

Service will connect to vLLM server and listen on port 9005.

### 3. Test

```bash
# Health check
curl http://localhost:9005/health | jq .

# Test with single pages
python test_ocr_vl_single_pages.py "/path/to/test.pdf" --max-pages 3
```

## Performance Tuning

### vLLM Configuration

Edit `vllm_config.yaml`:

```yaml
# GPU memory usage (0.3-0.8)
# RTX 3060 (12GB): recommended 0.8
gpu-memory-utilization: 0.8

# Max concurrent sequences (16-256)
# Higher = better throughput for batch processing
max-num-seqs: 128
```

After changes:
```bash
docker-compose restart
```

### Client Concurrency

Edit `ocr_vl_service_vllm.py`:

```python
VL_REC_MAX_CONCURRENCY = 8  # Adjust 4-16 based on performance
```

Or set environment variable:
```bash
export VL_REC_MAX_CONCURRENCY=16
python ocr_vl_service_vllm.py
```

## Stopping Services

```bash
# Stop vLLM server
docker-compose down

# Stop OCR service (Ctrl+C in terminal)
```

## Comparison: Standard vs vLLM

| Metric | Standard (port 9005) | vLLM (port 8118) |
|--------|---------------------|------------------|
| Setup | Python venv | Docker container |
| Startup | ~30 seconds | ~2-3 minutes |
| Single page | ~10-15 seconds | **~3-5 seconds** |
| Batch processing | Sequential | **Parallel batching** |
| Memory usage | ~4-5 GB | ~9-10 GB |
| Throughput | 1x | **3-5x faster** |

## Troubleshooting

### vLLM server won't start

Check GPU:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

Check logs:
```bash
docker-compose logs
```

### OCR service can't connect to vLLM

Check if server is running:
```bash
curl http://localhost:8118/health
```

Check port:
```bash
docker-compose ps
ss -tlnp | grep 8118
```

### Out of memory errors

Reduce GPU memory utilization in `vllm_config.yaml`:
```yaml
gpu-memory-utilization: 0.6  # Use 60% instead of 80%
```

## Integration with Rechtmaschine

Once tested, update environment variables:
```bash
# In /var/opt/docker/rechtmaschine/app/.env
OCR_VL_SERVICE_URL=http://localhost:9005
```

The OCR service will automatically use vLLM acceleration when the Docker server is running.
