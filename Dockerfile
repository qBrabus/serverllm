# Base image with CUDA 12.4, cuDNN 9 and the latest PyTorch/Torchaudio builds.
# This image is optimised for DGX-class systems (including the H200) and already
# ships with the NVIDIA Container Toolkit integration.
FROM nvcr.io/nvidia/pytorch:24.05-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# System dependencies required by ffmpeg, NeMo and various audio front-ends.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ffmpeg \
        sox \
        libsox-fmt-all \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/app

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r requirements.txt

COPY app ./app

EXPOSE 5000

ENV MODELS_ROOT=/raid/workspace/qladane/llm/models \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000 \
    FLASK_DEBUG=0

CMD ["python", "-m", "app.manager"]
