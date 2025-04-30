import os
import json
import time
import tempfile
from pathlib import Path
import psutil
import pynvml
from helpers.gpu_status import get_gpu_status
from fastapi import FastAPI, UploadFile, File
from helpers.ocr_helper import extract_text_from_image
from fastapi.concurrency import run_in_threadpool
import shutil
from Chain import chain  # Your existing LLMChain instance


UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming you're using GPU 0

app = FastAPI()


@app.post("/extract")
async def extract(invoice_image: UploadFile = File(...)):
    """
    Upload an invoice image, run OCR, then LLM extraction, and return structured JSON.
    """
    # Save uploaded image to a temporary file
    if invoice_image.filename is None:
        return {"error": "No filename provided"}

    suffix = Path(invoice_image.filename).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await invoice_image.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Run OCR on the saved image in a threadpool
    ocr_text = await run_in_threadpool(extract_text_from_image, tmp_path)

    # Run LLM chain on the OCR text in a threadpool
    result_text = await run_in_threadpool(chain.run, ocr_output=ocr_text)

    # Parse the JSON string returned by the LLM
    try:
        structured = json.loads(result_text)
    except json.JSONDecodeError:
        # If JSON invalid, return raw text for debugging
        return {"error": "Invalid JSON from LLM", "raw": result_text}

    # Write the structured JSON output to a file
    output_file = Path(tempfile.gettempdir()) / "structured_output.json"
    with open(output_file, "w") as f:
        json.dump(structured, f, indent=4)
    return structured


@app.post("/extract_multiple")
async def extract_multiple():
    """
    Automatically process all invoices in the invoices/ directory:
    1. Run OCR on each invoice.
    2. Run LLM chain on each OCR result.
    3. Return structured JSON for each invoice.
    """
    # Get all files from the 'invoices/' directory
    invoice_dir = Path("invoices/")
    invoices = [file for file in invoice_dir.iterdir() if file.is_file()]

    all_results = []
    gpu_stats = []
    start_time = time.time()

    # Process each invoice
    for invoice in invoices:
        invoice_path = Path(UPLOAD_DIR) / invoice.name
        print("Processing invoice:", invoice.name)
        # Save the uploaded file to the defined directory
        if not invoice_path.exists():
            # Copy the file to the upload directory
            shutil.copy(invoice, UPLOAD_DIR)

        # Run OCR on the saved image in a threadpool
        ocr_text = await run_in_threadpool(extract_text_from_image, str(invoice))
        gpu_stats.append(get_gpu_status())

        # Run LLM chain on the OCR text in a threadpool
        result_text = await run_in_threadpool(chain.run, ocr_output=ocr_text)
        gpu_stats.append(get_gpu_status())

        # Parse the JSON string returned by the LLM
        try:
            structured = json.loads(result_text)
        except json.JSONDecodeError:
            # If JSON invalid, include raw text for debugging
            structured = {"error": "Invalid JSON from LLM", "raw": result_text}

        all_results.append({"filename": invoice.name, "data": structured})
        # Write the structured JSON output to a file
        output_file = Path(RESULTS_DIR) / f"{invoice.stem}_structured_output.json"
        with open(output_file, "w") as f:
            json.dump(structured, f, indent=4)

        total_latency = time.time() - start_time
        gpu_stats.append(get_gpu_status())
        gpu_results = {
            "total_latency_seconds": total_latency,
            "gpu_stats": gpu_stats,
            "results": all_results,
            "total_invoices_processed": len(all_results),
            "average_latency_per_invoice": total_latency / len(all_results)
            if all_results
            else 0,
            "successful_extractions": sum(
                1 for result in all_results if "error" not in result["data"]
            ),
            "failed_extractions": sum(
                1 for result in all_results if "error" in result["data"]
            ),
        }

        # Write the stress test results to a file
        stress_test_file = Path(RESULTS_DIR) / "stress_test_results.json"
        with open(stress_test_file, "w") as f:
            json.dump(gpu_results, f, indent=4)

    # Log the GPU status and processing latency
    return {"gpu_results": gpu_results}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True
    )
