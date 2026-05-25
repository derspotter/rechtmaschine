# Rechtmaschine OCR Service

Current production OCR is the Debian host HPI service documented in `../docs/debian-ocr-hpi-setup.md`.

## Current Runtime

- Venv: `ocr/.venv_hpi`
- Runner: `bash ocr/run_hpi_service.sh`
- OCR backend port: `9003`
- Service manager port: `8004`
- Other machines should call: `http://debian:8004`
- Paddle stack: PaddlePaddle GPU `3.2.0`, PaddleOCR `3.5.x`, PaddleX `3.5.x`

Start manually from repo root:

```bash
OCR_SERVICE_FILE=ocr_service_hibernate.py bash ocr/run_hpi_service.sh
```

The normal autostart path is the user service `rechtmaschine-ocr`, which runs `service_manager.py` and starts the HPI OCR backend on demand.

```bash
systemctl --user status rechtmaschine-ocr --no-pager
journalctl --user -u rechtmaschine-ocr -f
```

## API

### Health

```bash
curl -fsS http://127.0.0.1:8004/health | jq .
curl -fsS http://127.0.0.1:9003/health | jq .
```

### OCR

```bash
curl -fsS -X POST http://127.0.0.1:8004/ocr \
  -H "X-Request-ID: manual-test-$(date +%s)" \
  -F "file=@/path/to/document.pdf" \
  | jq .
```

The response includes:

- `request_id`
- `page_count`
- `avg_confidence`
- `full_text`
- `pages[]` with `page_index`, `lines`, `line_count`, `confidences`, boxes, and per-page metadata
- `metadata.vram` with optional sampled VRAM baseline/peak/final values

## PDF Processing

PDFs are rendered and processed one page at a time with `pypdfium2`. This keeps GPU memory bounded and preserves page boundaries for citation checking.

Important environment knobs:

```env
OCR_PDF_RENDER_DPI=200
OCR_ENABLE_DPI_FALLBACK=1
OCR_PDF_FALLBACK_DPI=150
OCR_ENABLE_VRAM_SAMPLING=1
OCR_VRAM_SAMPLE_INTERVAL_SECONDS=0.25
```

If a page fails with a probable CUDA/GPU allocation error at the default DPI, the service clears CUDA cache and retries that page at the fallback DPI before failing the document.

## Logs

Requests are correlated with `request_id` in both the manager and OCR backend logs.

```bash
journalctl --user -u rechtmaschine-ocr -f | grep request_id
journalctl --user -u rechtmaschine-ocr -f | grep TIMING
```

Example timing lines:

```text
[request_id=...] [TIMING] OCR page render page=1 dpi=200 size=1654x2339 seconds=0.146
[request_id=...] [TIMING] OCR page total page=1/3 dpi=200 predict_seconds=1.328 total_seconds=3.223
[request_id=...] [TIMING] OCR document total pages=3 lines=99 bytes=106313 total_seconds=6.262 seconds_per_page=2.087
[request_id=...] [TIMING] OCR request total filename=document.pdf pages=3 total_seconds=6.300 peak_vram_mb=1234 peak_vram_delta_mb=290
```

## Host Tools

Useful Debian packages:

```bash
sudo apt install qpdf ccache nvtop ripgrep jq
```

- `qpdf`: repair fallback for malformed PDFs
- `ccache`: avoids repeated native extension compile work
- `nvtop`: live GPU monitoring
- `ripgrep`: fast code/log inspection
- `jq`: readable API responses

`img2pdf` is not required for the current OCR flow.

## Legacy Notes

Old instructions that referenced `~/Nextcloud/Kanzlei/ocr/venv`, port `8003`, direct desktop/Tailscale OCR, or `ocr_service.py` are obsolete for the Debian HPI setup. Use `../docs/debian-ocr-hpi-setup.md` as the authoritative runbook.
