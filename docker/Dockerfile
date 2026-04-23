FROM registry.rcp.epfl.ch/rcp-courses/2026-ee-559:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    libgl1-mesa-dev && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && apt install -y \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    wget \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel packaging ninja
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/huggingface/diffusers.git