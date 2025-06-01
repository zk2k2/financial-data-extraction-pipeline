import os
import json
import time
import threading
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Request, Response
from fastapi.concurrency import run_in_threadpool

# ───── prometheus_client imports ────────────────────────────────────────────────
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# ───── pynvml for GPU stats ─────────────────────────────────────────────────────
import pynvml

# ───── Your existing OCR & LLM helpers ───────────────────────────────────────────
from helpers.ocr_helper import extract_text_from_image
from Chain import extract_invoice_data

# ───── Create a custom registry (no default collectors) ─────────────────────────
my_registry = CollectorRegistry()

# ───── Define exactly the metrics we want ────────────────────────────────────────
REQUEST_COUNT = Counter(
    "invoice_extract_requests_total",
    "Total number of invoice extraction requests",
    registry=my_registry,
)

REQUEST_LATENCY = Histogram(
    "invoice_extract_request_duration_seconds",
    "Histogram of request processing latency (seconds)",
    registry=my_registry,
)

GPU_UTIL = Gauge(
    "gpu_utilization_percentage",
    "Current GPU utilization (percent)",
    registry=my_registry,
)

GPU_MEM_USED = Gauge(
    "gpu_memory_used_bytes",
    "Current GPU memory used in bytes",
    registry=my_registry,
)

# ───── Initialize NVML & start polling thread ───────────────────────────────────
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assumes GPU index 0

def _poll_gpu_metrics():
    """
    Poll NVIDIA GPU stats every 5 seconds and set the gauges.
    """
    while True:
        util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle).gpu
        mem = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle).used
        GPU_UTIL.set(util)
        GPU_MEM_USED.set(mem)
        time.sleep(5)

# ───── Build FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI()

# Start GPU polling on startup
@app.on_event("startup")
def startup_event():
    thread = threading.Thread(target=_poll_gpu_metrics, daemon=True)
    thread.start()

# ───── Middleware to measure latency + count requests ───────────────────────────
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_ts = time.time()
    response = await call_next(request)
    latency = time.time() - start_ts

    # Observe latency
    REQUEST_LATENCY.observe(latency)

    # Increment only for /extract
    if request.url.path == "/extract":
        REQUEST_COUNT.inc()

    return response

# ───── /metrics endpoint (returns ONLY our custom registry) ──────────────────────
@app.get("/metrics")
async def metrics():
    data = generate_latest(my_registry)
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

# ───── /extract endpoint (unchanged OCR + LLM flow) ─────────────────────────────
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/extract")
async def extract(invoice_image: UploadFile = File(...)):
    if invoice_image.filename is None:
        return {"error": "No filename provided"}

    suffix = Path(invoice_image.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await invoice_image.read()
        tmp.write(content)
        tmp_path = tmp.name

    ocr_text = await run_in_threadpool(extract_text_from_image, tmp_path)
    result_text = await run_in_threadpool(extract_invoice_data, ocr_text=ocr_text)

    if isinstance(result_text, (dict, list)):
        structured = result_text
    else:
        try:
            structured = json.loads(result_text)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON from LLM", "raw": result_text}

    # (Optional) write to a file
    output_file = Path(tempfile.gettempdir()) / "structured_output.json"
    with open(output_file, "w") as f:
        json.dump(structured, f, indent=4)

    return structured

# ───── Run with Uvicorn ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,  # disable reload=True if you don’t need hot-reload
    )
