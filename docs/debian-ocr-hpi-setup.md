# Debian OCR HPI Setup

This runbook sets up the current host-based PaddleOCR HPI service on the Debian RAG/OCR machine.

The goal is to run OCR on Debian at `127.0.0.1:9003`, while desktop stays reserved for Qwen3.6 workloads.

## Known-Good Reference

The working desktop HPI venv currently uses:

- Python `3.11.13`
- `paddlepaddle-gpu==3.2.0`
- `paddleocr==3.5.0`
- `paddlex==3.5.1`
- `fastapi==0.128.0`
- `uvicorn==0.40.0`
- `python-multipart==0.0.21`
- `httpx==0.28.1`
- `ultra-infer-gpu-python` CUDA 12 wheel

Expected OCR VRAM usage is about `2-3 GB` when the OCR engine is loaded.

## Prerequisites

On Debian, confirm NVIDIA and Python first:

```bash
nvidia-smi
python3.11 --version
```

If Python 3.11 is missing, install it before creating the OCR venv. Keep this OCR venv separate from the app venv.

## Create The HPI Venv

```bash
cd ~/rechtmaschine/ocr

python3.11 -m venv .venv_hpi
source .venv_hpi/bin/activate

python -m pip install -U pip setuptools wheel
```

Install Paddle GPU first. The order matters.

```bash
python -m pip install paddlepaddle-gpu==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Then install PaddleOCR and service dependencies:

```bash
python -m pip install -U "paddleocr[doc-parser]"
python -m pip install fastapi uvicorn python-multipart httpx Pillow
```

If `ultra-infer-gpu-python` was not installed as part of the PaddleOCR/PaddleX stack, install the CUDA 12 wheel used by HPI:

```bash
python -m pip install \
  "https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/deploy/hpi/ultra_infer/releases/new_hpi/v1.2.0/cuda12/ultra_infer_gpu_python-1.2.0-cp311-cp311-linux_x86_64.whl"
```

Verify imports:

```bash
python - <<'PY'
import paddle
import paddleocr

print("paddle", paddle.__version__)
print("cuda compiled", paddle.device.is_compiled_with_cuda())
print("paddleocr", getattr(paddleocr, "__version__", "unknown"))
PY
```

Expected:

- `paddle 3.2.0`
- `cuda compiled True`
- PaddleOCR imports without error

## Patch PaddleX Word-Box Crash

HPI word boxes can crash on empty or near-blank pages with `KeyError: text_word_region`.

Patch:

```bash
PIPELINE="$PWD/.venv_hpi/lib/python3.11/site-packages/paddlex/inference/pipelines/ocr/pipeline.py"
cp "$PIPELINE" "$PIPELINE.bak"
```

Open `$PIPELINE` and ensure the OCR result initialization has both keys when `return_word_box` is enabled:

```python
if return_word_box:
    for res in results:
        res["text_word"] = []
        res["text_word_region"] = []
```

Also ensure final conversion reads the key safely if your installed PaddleX version does not initialize it on blank pages:

```python
if return_word_box:
    res.setdefault("text_word_region", [])
    res["text_word_boxes"] = [
        convert_points_to_boxes(line)
        for line in res.get("text_word_region", [])
    ]
```

Reapply this patch after recreating the venv or upgrading PaddleOCR/PaddleX.

## Start The OCR Service

Use the repo runner. It sets `LD_LIBRARY_PATH` for NVIDIA, Paddle, and UltraInfer libraries inside the venv.

```bash
cd ~/rechtmaschine
OCR_SERVICE_FILE=ocr_service_hibernate.py bash ocr/run_hpi_service.sh
```

The service listens on `0.0.0.0:9003`.

Health checks:

```bash
curl -fsS http://127.0.0.1:9003/health | jq .
curl -fsS -X POST http://127.0.0.1:9003/load | jq .
curl -fsS -X POST http://127.0.0.1:9003/unload | jq .
```

The hibernate service lazy-loads OCR. `/load` moves OCR into VRAM. `/unload` releases OCR VRAM while keeping the service process alive.

## Smoke Test

Use any scanned PDF:

```bash
curl -fsS -X POST http://127.0.0.1:9003/ocr \
  -F "file=@/path/to/scanned.pdf" \
  | jq '.full_text | length'
```

Watch VRAM while testing:

```bash
watch -n 1 nvidia-smi
```

Expected:

- OCR loads in roughly `10-15s` on a cold start.
- VRAM rises by about `2-3 GB`.
- `/unload` drops OCR VRAM again.

## Optional systemd User Service

Create `~/.config/systemd/user/rechtmaschine-ocr.service`:

```ini
[Unit]
Description=Rechtmaschine OCR HPI Service
After=default.target

[Service]
Type=simple
WorkingDirectory=%h/rechtmaschine
Environment=OCR_SERVICE_FILE=ocr_service_hibernate.py
ExecStart=%h/rechtmaschine/ocr/run_hpi_service.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Enable it:

```bash
systemctl --user daemon-reload
systemctl --user enable --now rechtmaschine-ocr
systemctl --user status rechtmaschine-ocr --no-pager
journalctl --user -u rechtmaschine-ocr -f
```

## Debian RAG Integration

The Debian RAG services should call OCR locally:

```env
OCR_SERVICE_URL=http://127.0.0.1:9003
```

Desktop should not be used for OCR once Debian is healthy. Desktop remains the Qwen3.6 worker for anonymization, metadata extraction, and segmentation.
