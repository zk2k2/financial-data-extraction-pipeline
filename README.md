# 🧾 Financial Data Extraction Pipeline

This is a pipeline designed to automate the extraction of structured financial data from invoices. Users upload invoices (PNG, PDF, etc.), which are first processed by PaddleOCR for text extraction. A Large Language Model (local via Ollama or cloud via OpenRouter) then post-processes the OCR output to generate JSON with relevant fields. Results undergo semantic validation (e.g., numeric checks), and the entire workflow is orchestrated through a FastAPI service. Monitoring and observability are provided by Prometheus and Grafana.

## Demo 🎥

https://github.com/user-attachments/assets/53c07621-a7fd-4455-8b5f-be66c9105d4f

## ✨ Features

- **Bimodal Extraction**: Combines PaddleOCR for text extraction with an LLM for semantic parsing.
- **Flexible LLM Integration**: Supports local inference via Ollama or cloud APIs via OpenRouter.
- **JSON Transformation**: Converts raw OCR text into a structured JSON schema (invoice number, dates, amounts, vendor info).
- **Semantic Validation**: Applies post‐processing checks (field presence, numeric consistency) before final output.
- **FastAPI Orchestration**: Single REST endpoint (`/extract`) handling uploads, OCR, LLM inference, validation, and storage.
- **Monitoring & Observability**: Integrated Prometheus metrics and Grafana dashboards for latency, error rates, and throughput.
- **Object Storage**: Uses MinIO (S3-compatible) to store original uploads, OCR outputs, and final JSON artifacts.

## 🚀 Getting Started

### ✅ Prerequisites

Ensure you have the following installed on your system:

- **Python 3.10+** (for FastAPI service)
- **Git** (to clone the repository)
- **Docker** (to run MinIO)
- **MinIO Client (mc)** or access to the MinIO console at `:9001`
- **Ollama** (for local LLM inference) _or_ valid OpenRouter API credentials

### 🛠 Setup

1. **Clone the repository:**
    ```sh
    git clone https://github.com/zk2k2/financial-data-extraction-pipeline.git
    ```

2. **Create a Python virtual environment and install dependencies:**
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

3. **Run the FastAPI service:**
    ```sh
    python server.py
    ```

4. **Start MinIO (object storage):**
    ```sh
    MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin123 \
    ./minio server ./minio-data --console-address ":9001"
    ```

That’s it—FastAPI will be listening (by default) on `http://localhost:8000`, MinIO on `http://localhost:9000` (console at `:9001`). You can now POST invoices to `http://localhost:8000/extract`.

## 🤝 Contributing

We welcome contributions to improve InvoiceX! Please fork the repository, create a feature branch, and submit a pull request for review.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions or more information, please contact the project maintainer via email.
