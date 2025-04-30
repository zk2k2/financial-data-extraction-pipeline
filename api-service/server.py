import os
import json
import time
import tempfile
from pathlib import Path
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
import httpx
import pynvml
from helpers.gpu_status import get_gpu_status
from Chain import chain  # your LLMChain instance configured to use OLLAMA_URL

# configure directories
UPLOAD_DIR = "invoices"
os.makedirs(UPLOAD_DIR, exist_ok=True)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# OCR microservice URL (set via docker-compose env)
OCR_URL = os.getenv("OCR_URL", "http://ocr-service:8001/ocr")

# initialize GPU monitoring
pynvml.nvmlInit()

app = FastAPI()


async def call_ocr_service(file_path: str, filename: str) -> str:
    """
    Send image/pdf bytes to the OCR microservice and return extracted text.
    """
    # read bytes
    data = Path(file_path).read_bytes()
    files = {"file": (filename, data)}
    async with httpx.AsyncClient() as client:
        resp = await client.post(OCR_URL, files=files, timeout=120)
    if resp.status_code != 200:
        raise HTTPException(502, detail=f"OCR service error: {resp.status_code}")
    body = resp.json()
    return body.get("text", "")


@app.post("/extract")
async def extract(invoice_image: UploadFile = File(...)):
    # save upload to temp file
    if not invoice_image.filename:
        raise HTTPException(400, detail="No filename provided")
    suffix = Path(invoice_image.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await invoice_image.read()
        tmp.write(content)
        tmp_path = tmp.name

    # call OCR microservice
    ocr_text = await call_ocr_service(tmp_path, invoice_image.filename)

    # run LLM extraction
    result_text = await run_in_threadpool(chain.run, ocr_output=ocr_text)
    try:
        structured = json.loads(result_text)
    except json.JSONDecodeError:
        raise HTTPException(500, detail={"raw": result_text})

    # save output
    out_file = (
        Path(RESULTS_DIR) / f"{Path(invoice_image.filename).stem}_structured.json"
    )
    out_file.write_text(json.dumps(structured, indent=2))
    return structured


@app.post("/extract_multiple")
async def extract_multiple():
    invoices = [f for f in Path(UPLOAD_DIR).iterdir() if f.is_file()]
    if not invoices:
        raise HTTPException(400, detail="No invoices found in invoices/ directory")

    all_results = []
    gpu_stats = []
    start_time = time.time()

    for invoice in invoices:
        # OCR
        ocr_text = await call_ocr_service(str(invoice), invoice.name)
        gpu_stats.append(get_gpu_status())
        # LLM
        result_text = await run_in_threadpool(chain.run, ocr_output=ocr_text)
        gpu_stats.append(get_gpu_status())
        try:
            structured = json.loads(result_text)
        except json.JSONDecodeError:
            structured = {"error": "Invalid JSON from LLM", "raw": result_text}

        # record and save
        all_results.append({"filename": invoice.name, "data": structured})
        out_file = Path(RESULTS_DIR) / f"{invoice.stem}_structured.json"
        out_file.write_text(json.dumps(structured, indent=2))

    latency = time.time() - start_time
    return {"results": all_results, "gpu_stats": gpu_stats, "latency_s": latency}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True
    )
