# serverllm

This repository packages the vLLM management dashboard inside a GPU-enabled
Docker image that is ready for DGX-class systems (including the DGX H200).

## Features

- Flask web UI for downloading Hugging Face models and launching vLLM servers.
- CUDA 12.4 / cuDNN 9 base image from NVIDIA's PyTorch container.
- Pre-installed PyTorch, TorchAudio, vLLM, NVIDIA NeMo ASR toolkit, ffmpeg and
  supporting audio codecs.
- Docker Compose recipe with GPU reservations and persistent volumes for models
  and Hugging Face cache data.

## Project layout

```
.
├── app/manager.py        # Flask application
├── requirements.txt      # Python dependencies (vLLM, NeMo, etc.)
├── Dockerfile            # GPU-ready container image
├── docker-compose.yml    # Example runtime configuration
├── .dockerignore
└── README.md
```

## Building the image

The container must be built with the NVIDIA Container Toolkit available on the
host (standard on DGX systems).

```bash
docker build -t serverllm-manager:latest .
```

If you prefer to use Docker Compose, the service definition in
`docker-compose.yml` will build the image automatically on first run.

## Running with Docker

```bash
docker run -it --rm \
  --gpus all \
  -p 5000:5000 \
  -e MODELS_ROOT=/models \
  -v $(pwd)/models:/models \
  -v $(pwd)/hf-cache:/root/.cache/huggingface \
  serverllm-manager:latest
```

The web UI becomes available at http://localhost:5000/. Models downloaded from
Hugging Face are stored in the mounted `./models` directory so they persist
across container restarts.

## Running with Docker Compose

```bash
docker compose up -d
```

The Compose stack exposes the same port and volumes as the `docker run`
example. Stop the stack with `docker compose down`.

## Configuration

| Variable     | Default                               | Description |
|--------------|----------------------------------------|-------------|
| `MODELS_ROOT`| `/raid/workspace/qladane/llm/models`   | Location where models are stored. Override when mounting external storage. |
| `FLASK_HOST` | `0.0.0.0`                              | Bind address for the Flask app. |
| `FLASK_PORT` | `5000`                                 | Port exposed by the Flask app. |
| `FLASK_DEBUG`| `0`                                    | Set to `1` to enable Flask debug mode. |

Mounting `/root/.cache/huggingface` as shown above allows the container to
reuse previously downloaded files when accessing gated/private repositories.

## Notes on GPU support

- The base image already includes CUDA, cuDNN, NCCL, PyTorch and TorchAudio
  compiled for CUDA 12.4.
- The `pip install` step pulls vLLM wheels that depend on the CUDA 12.1
  toolchain distributed by PyTorch.
- NVIDIA NeMo (with the ASR extras) is installed to support speech pipelines.
- ffmpeg and SoX utilities are available in the container for audio handling.

When running on DGX hardware make sure the host driver version is compatible
with CUDA 12.4 (e.g. R550+). The NVIDIA Container Toolkit must also be at least
v1.13 to expose H200 GPUs to containers.
