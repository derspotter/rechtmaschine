# Fast Model Loading Research

## Problem Statement

The PaddleOCR service takes **~13 seconds** to load 4 models into VRAM (12GB GPU), while Ollama loads 8B parameter models in **~1 second**.

**Constraint:** Cannot keep both OCR (~2.2GB VRAM) and LLM models loaded simultaneously due to 12GB VRAM limit.

**Goal:** Reduce OCR cold start time when switching between OCR and LLM services.

---

## Current Benchmark Results

| Scenario | Time | Notes |
|----------|------|-------|
| Cold start | 16.2s | First request after service manager restart |
| Reload (after pkill) | 13.0s | Models already in disk cache |
| Warm (3 pages) | 2.4s | ~0.8s per page |
| Warm (5 pages) | 4.0s | ~0.8s per page |

### Models Loaded by PaddleOCR

1. `PP-LCNet_x1_0_doc_ori` - Document orientation classification
2. `PP-LCNet_x1_0_textline_ori` - Textline orientation classification
3. `PP-OCRv5_server_det` - Text detection
4. `PP-OCRv5_server_rec` - Text recognition

---

## Why Ollama is Fast

Ollama achieves ~1 second load times through:

1. **mmap (memory-mapped files)** - GGUF format mmaps model weights directly from disk. The OS page cache keeps recently-used models in RAM, so "loading" is just mapping pointers, not copying data.

2. **Single model file** - One `.gguf` file vs PaddleOCR's 4 separate models, each needing its own CUDA context initialization.

3. **Daemon architecture** - Ollama runs continuously and can keep models warm. Even when "unloaded", the mmap'd file stays in OS page cache.

4. **Lazy weight loading** - Weights are loaded on-demand as layers are accessed, not all upfront.

### Why PaddleOCR is Slow

- 4 separate models, each with full CUDA initialization
- Paddle Inference backend does eager weight loading
- No mmap - copies weights from disk → CPU RAM → VRAM
- Each model builds its own CUDA execution graph

---

## Solutions from Gemini

### RAM Disk + Hibernate Pattern

**Key insight:** Don't kill the process, just unload VRAM.

#### Step 1: Move Models to RAM Disk

```bash
# Create folder in RAM
mkdir -p /dev/shm/paddle_models

# Copy models to RAM (usually in ~/.paddleocr/whl/ or ~/.paddlex/)
cp -r ~/.paddleocr/whl/* /dev/shm/paddle_models/
cp -r ~/.paddlex/official_models/* /dev/shm/paddle_models/

# Set environment variables
export DET_MODEL_DIR="/dev/shm/paddle_models/det/..."
export REC_MODEL_DIR="/dev/shm/paddle_models/rec/..."
export CLS_MODEL_DIR="/dev/shm/paddle_models/cls/..."
```

#### Step 2: Implement Hibernate Endpoints

```python
import gc
import paddle

# Global variable for the engine
OCR_ENGINE = None

def load_engine():
    global OCR_ENGINE
    if OCR_ENGINE is None:
        print("[INFO] Loading OCR Engine into VRAM...")
        OCR_ENGINE = PaddleOCR(
            use_gpu=True,
            enable_hpi=ENABLE_HPI,
            # ... other params
        )

@app.post("/load")
def api_load():
    load_engine()
    return {"status": "loaded"}

@app.post("/unload")
def api_unload():
    global OCR_ENGINE
    if OCR_ENGINE:
        del OCR_ENGINE
        OCR_ENGINE = None

    # Force VRAM cleanup
    gc.collect()
    paddle.device.cuda.empty_cache()
    print("[INFO] OCR Engine unloaded from VRAM")
    return {"status": "unloaded"}

@app.post("/ocr")
async def ocr_endpoint(file: UploadFile = File(...)):
    if OCR_ENGINE is None:
        load_engine()
    # ... rest of the function
```

#### Step 3: Update Service Manager

- Don't kill the process with `pkill`
- Send `POST /unload` request when switching to LLM service
- The `/ocr` endpoint auto-loads (taking ~1-2s from `/dev/shm` instead of 13s)

### Expected Savings

| Component | Time Saved |
|-----------|------------|
| Process boot (imports/interpreter) | ~3-5s |
| Disk I/O (with /dev/shm) | ~2-4s |
| CUDA context creation | Saved if paddle keeps context alive |

---

## Solutions from Codex (OpenAI)

### Option 1: Keep Model Files in OS Page Cache

```bash
# Pre-touch files at service start
cat ~/.paddlex/official_models/*/* > /dev/null

# Or use vmtouch to lock pages in memory
vmtouch -t ~/.paddlex/official_models/
vmtouch -l ~/.paddlex/official_models/  # requires privileges

# Or use tmpfs (RAM disk)
mount -t tmpfs -o size=2G tmpfs /mnt/paddle_models
cp -r ~/.paddlex/official_models/* /mnt/paddle_models/
```

### Option 2: Keep Model Bytes in RAM (Load from Buffer)

Read model/params into memory once and keep those bytes in a long-lived process. Many inference APIs allow "load from buffer" so you avoid filesystem reads entirely.

```python
# Pseudocode - actual implementation depends on Paddle API
class ModelCache:
    def __init__(self):
        self.model_bytes = {}

    def preload(self, model_path):
        with open(model_path, 'rb') as f:
            self.model_bytes[model_path] = f.read()

    def get_model(self, model_path):
        return self.model_bytes.get(model_path)
```

### Option 3: Fork-After-Load (Copy-On-Write)

Start a parent process, load OCR weights into CPU memory, then `fork` a GPU worker when needed. The child inherits the already-resident pages (no re-reading), so GPU init only copies to VRAM.

```python
import os
import multiprocessing

class OCRManager:
    def __init__(self):
        # Load weights into CPU memory in parent process
        self.cpu_weights = self._load_weights_to_cpu()

    def process_request(self, file_path):
        # Fork a worker - child inherits CPU weights via COW
        p = multiprocessing.Process(target=self._gpu_worker, args=(file_path,))
        p.start()
        p.join()

    def _gpu_worker(self, file_path):
        # Only need to copy CPU weights to GPU (fast)
        model = self._load_to_gpu(self.cpu_weights)
        result = model.predict(file_path)
        return result
```

### Option 4: Pinned Host Memory

If the framework lets you keep weights in pinned (page-locked) memory, host→device copies are faster.

```python
import paddle

# Allocate pinned memory
weights = paddle.empty([size], dtype='float32').pin_memory()

# Copy to GPU is faster from pinned memory
weights_gpu = weights.cuda()  # Uses DMA, faster than pageable memory
```

**Tradeoff:** Pinned memory reduces pageable RAM and can hurt system performance if overused.

### Option 5: Unified Memory / Oversubscription

CUDA Unified Memory can "page" weights on demand, but performance can degrade with page faults.

```python
# Enable unified memory in Paddle (if supported)
paddle.set_device('gpu')
paddle.device.cuda.set_device(0)
# Unified memory allows oversubscription but with potential performance penalty
```

### Option 6: Shrink One Side

Quantize LLM (4-bit/8-bit) or swap to a smaller OCR model so both fit at once. This avoids the load/unload cycle entirely.

```bash
# Use mobile models instead of server models
# PP-OCRv5_mobile_det instead of PP-OCRv5_server_det
# Smaller, faster to load, slightly lower accuracy
```

---

## Solutions from Web Search

### vLLM Sleep Mode (Best Reference Implementation)

Source: [vLLM Blog - Zero-Reload Model Switching](https://blog.vllm.ai/2025/10/26/sleep-mode.html)

> **Level 1 offloads weights to CPU RAM (fast wake time)** - 18-200x faster than full reload.
> **Level 2 discards weights entirely** (nearly as fast wake time, minimal RAM usage).

vLLM keeps model weights in CPU pinned memory and moves them to GPU on demand. This is exactly what we need but for PaddleOCR.

### NVIDIA Run:ai GPU Memory Swap

Source: [NVIDIA Blog - GPU Memory Swap](https://developer.nvidia.com/blog/cut-model-deployment-costs-while-keeping-performance-with-gpu-memory-swap/)

> Models not getting requests within a specific time frame are swapped to CPU memory when not in use. On receiving a request, the model is immediately swapped back into GPU memory with **minimal latency**.

### Hugging Face Accelerate

Source: [HuggingFace Blog - Large Model Inference](https://huggingface.co/blog/accelerate-large-models)

> Weights offloaded on the hard drive are loaded in RAM then put on a GPU **just before the forward pass** and cleaned up just after.

```python
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

# Load model with automatic device placement
model = load_checkpoint_and_dispatch(
    model,
    checkpoint_path,
    device_map="auto",  # or "balanced", "sequential"
    offload_folder="offload",
    offload_state_dict=True,
)
```

### NVIDIA CPU-GPU Memory Sharing for LLM Inference

Source: [NVIDIA Blog - KV Cache Offload](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/)

Describes techniques for sharing memory between CPU and GPU for efficient inference.

---

## PaddlePaddle Limitations

**PaddlePaddle does not have native support for CPU↔GPU model swapping.**

The web search found:
- No Paddle-specific CPU↔GPU model hot-swapping API
- Limited documentation on memory management
- Users have asked about similar features but no official solution exists

### What Paddle Does Support

```python
import paddle

# Tensor movement between devices
t = paddle.randn([1000, 1000])
t_gpu = t.cuda()  # Move to GPU
t_cpu = t_gpu.cpu()  # Move back to CPU

# Clear GPU cache
paddle.device.cuda.empty_cache()
```

However, moving entire PaddleOCR pipeline objects (which contain Paddle Inference engines, not raw tensors) is not straightforward.

---

## Recommended Approach

### Short-term (Minimal Changes)

1. **Hibernate pattern** - Keep OCR process alive, use `/unload` endpoint instead of pkill
2. **RAM disk** - Put models on `/dev/shm` to eliminate disk I/O
3. **Disable unused models** - Skip orientation models if not needed

Expected improvement: **13s → 5-7s**

### Medium-term (More Work)

1. **Custom weight caching** - Load Paddle model weights into CPU tensors, implement manual GPU transfer
2. **Fork-after-load pattern** - Parent holds weights in CPU, fork workers for GPU inference

Expected improvement: **13s → 2-3s**

### Long-term (Significant Effort)

1. **Port to PyTorch + Accelerate** - Use PyTorch OCR (docTR, EasyOCR) with HuggingFace Accelerate offloading
2. **ONNX Runtime** - Convert Paddle models to ONNX, use ONNX Runtime's memory management
3. **TensorRT** - Pre-compile models to TensorRT engines (faster loading after initial compilation)

Expected improvement: **13s → 1-2s** (matching Ollama)

---

## Open Questions

1. Can Paddle Inference engines be serialized/deserialized from memory buffers?
2. Does PaddleOCR support loading models from custom paths (for /dev/shm)?
3. Can we access the underlying weight tensors in PaddleOCR's pipeline?
4. Is there a way to keep CUDA context alive while freeing model weights?

---

## References

- [vLLM Sleep Mode](https://blog.vllm.ai/2025/10/26/sleep-mode.html)
- [NVIDIA GPU Memory Swap](https://developer.nvidia.com/blog/cut-model-deployment-costs-while-keeping-performance-with-gpu-memory-swap/)
- [HuggingFace Accelerate Large Models](https://huggingface.co/blog/accelerate-large-models)
- [NVIDIA CPU-GPU Memory Sharing](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/)
- [PaddleOCR GitHub Discussions](https://github.com/PaddlePaddle/PaddleOCR/discussions)
- [Paddle Inference Documentation](https://paddleclas.readthedocs.io/en/latest/extension/paddle_inference_en.html)

---

*Research conducted: 2025-02-03*
*Contributors: Claude, Gemini, Codex*
