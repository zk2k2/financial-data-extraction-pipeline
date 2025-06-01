import os
import json
import time
import threading
import tempfile
from pathlib import Path

from sqlalchemy.orm import Session
from database.connection import get_db, create_tables
from services.invoice_service import InvoiceService
from services.pipeline_service import PipelineService

from fastapi import FastAPI, UploadFile, File, Request, Response, Depends
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
    create_tables()
    thread = threading.Thread(target=_poll_gpu_metrics, daemon=True)
    thread.start()


# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the main page
@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Add this endpoint before the main section

@app.get("/monitoring")
async def monitoring_dashboard():
    return FileResponse('static/monitoring.html')

@app.get("/api/pipeline-stats")
async def get_pipeline_stats(db: Session = Depends(get_db)):
    """Get overall pipeline statistics"""
    invoice_service = InvoiceService(db)
    invoices = invoice_service.get_invoices(limit=1000)
    
    stats = {
        "total_invoices": len(invoices),
        "successful_extractions": len([i for i in invoices if i.total_amount]),
        "failed_extractions": len([i for i in invoices if not i.total_amount]),
        "average_amount": sum([float(i.total_amount or 0) for i in invoices]) / max(len(invoices), 1),
        "recent_invoices": [i.to_dict() for i in invoices[-5:]]  # Last 5
    }
    
    return stats

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
async def extract(invoice_image: UploadFile = File(...), db: Session = Depends(get_db)):
    if invoice_image.filename is None:
        return {"error": "No filename provided"}

    suffix = Path(invoice_image.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await invoice_image.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import time
        invoice_id = int(time.time() * 1000)  # Use current timestamp as unique ID
        
        # Use pipeline service for processing
        pipeline_service = PipelineService()
        result = await run_in_threadpool(
            pipeline_service.process_invoice_pipeline,
            invoice_id,
            content,
            invoice_image.filename
        )
    except Exception as e:
        print(f"Error processing invoice: {e}")
        return {"error": str(e)}

    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    # return {"error": "An unexpected error occurred during processing"}


@app.get("/pipeline/status/{invoice_id}")
async def get_pipeline_status(invoice_id: int):
    """Get processing status of an invoice"""
    pipeline_service = PipelineService()
    return pipeline_service.get_pipeline_status(invoice_id)

@app.get("/pipeline/cleaned/{invoice_id}")
async def get_cleaned_data(invoice_id: int):
    """Get cleaned invoice data from MinIO"""
    pipeline_service = PipelineService()
    data = pipeline_service.get_cleaned_invoice_data(invoice_id)
    
    if data:
        return data
    else:
        return {"error": "Cleaned data not found"}

@app.post("/pipeline/reprocess/{invoice_id}")
async def reprocess_invoice(invoice_id: int):
    """Reprocess an invoice through validation pipeline"""
    pipeline_service = PipelineService()
    result = await run_in_threadpool(
        pipeline_service.reprocess_invoice,
        invoice_id
    )
    return result


@app.get("/invoices")
async def get_invoices(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Get list of stored invoices"""
    invoice_service = InvoiceService(db)
    invoices = invoice_service.get_invoices(skip=skip, limit=limit)
    return [invoice.to_dict() for invoice in invoices]

@app.get("/invoices/{invoice_id}")
async def get_invoice(invoice_id: int, db: Session = Depends(get_db)):
    """Get specific invoice by ID"""
    invoice_service = InvoiceService(db)
    invoice = invoice_service.get_invoice(invoice_id)
    if not invoice:
        return {"error": "Invoice not found"}
    return invoice.to_dict()



# ───── Run with Uvicorn ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,  # disable reload=True if you don’t need hot-reload
    )
