FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements ./requirements
COPY pyproject.toml README.md requirements.txt ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements/gpu-linux.txt

COPY src ./src
COPY sam2_configs ./sam2_configs
COPY birefnet_sam2_full.py ./

RUN python -m pip install -e . --no-deps

CMD ["/bin/sh", "-c", "if [ \"$EXECUTOR_ROLE\" = \"gpu\" ]; then uvicorn sticker_bot.executors.gpu_app:app --host 0.0.0.0 --port ${PORT:-8002}; else uvicorn sticker_bot.executors.cpu_app:app --host 0.0.0.0 --port ${PORT:-8001}; fi"]
