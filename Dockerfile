# added this myself 
FROM python:3.12-slim

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install torch matplotlib qiskit pylatexenc open-clip-torch 

# RUN with: docker run --rm -it \ -v "$PWD:/workspace/genQC" \ -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \ genqc-local-cpu \ bash
# RUN with (should be the same): docker run --rm -it -v "$PWD:/workspace/genQC" -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -w /workspace/genQC genqc-local-cpu bash
# (then inside run): pip install -e .
