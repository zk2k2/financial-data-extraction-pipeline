# ocr_service/Dockerfile
FROM python:3.10-slim

# system deps for pdf2image
RUN apt-get update && \
    apt-get install -y poppler-utils libgl1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ../helpers/ocr_helper.py ./ocr_helper.py
COPY ocr_service.py ./
COPY ../requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8001
CMD ["uvicorn", "ocr_service:app", "--host", "0.0.0.0", "--port", "8001"]
