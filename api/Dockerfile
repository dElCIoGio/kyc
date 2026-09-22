# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Paddle's CPU runtime uses OpenMP, while OpenCV's binary extension needs the
# GL and GLib runtime libraries even though this service is headless. Model
# artifacts are mounted at runtime and deliberately excluded from the image.
RUN apt-get update \
  && apt-get install --yes --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY api/pyproject.toml api/README.md ./api/
COPY api/src ./api/src

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && python -m pip install ".[ocr]" \
    && python -m pip install ./api

EXPOSE 8000

CMD ["uvicorn", "kyc_api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
