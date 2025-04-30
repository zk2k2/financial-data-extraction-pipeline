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
