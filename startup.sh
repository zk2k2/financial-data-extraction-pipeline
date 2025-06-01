apt update && apt upgrade -y && \
apt install -y \
    git \
    curl \
    wget \
    unzip \


apt install -y libgl1-mesa-glx
apt install glibc
apt install -y poppler-utils
pip3 install pdf2image
apt install -y libcudnn8 libcudnn8-dev
python3 -m pip install paddlepaddle-gpu
apt install -r requirements.txt

#!/bin/bash
cd /workspace/financial-data-extraction
source venv/bin/activate
export PYTHONPATH="/workspace/financial-data-extraction:$PYTHONPATH"
python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload