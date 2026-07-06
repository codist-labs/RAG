FROM apache/airflow:2.8.1-python3.11

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /requirements.txt

# Install CPU-only PyTorch FIRST, before the rest of requirements.txt.
#
# requirements.txt pulls in sentence-transformers==2.3.1, which depends on
# torch>=1.11.0 with no upper bound and no CPU/GPU distinction. Left
# unpinned, pip's resolver picks the newest PyPI release (e.g. 2.12.1),
# whose default build is the full CUDA-enabled GPU wheel — which drags in
# ~2.5GB of NVIDIA runtime libraries (nvidia-cudnn-cu13, nvidia-cufft,
# nvidia-nccl-cu13, triton, cuda-toolkit, etc.) that this container can
# NEVER use, since Airflow here has no GPU access at all. This is exactly
# what caused a previous build to time out mid-download of a 214MB
# nvidia_cufft wheel.
#
# Installing the CPU wheel explicitly here satisfies sentence-transformers'
# "torch>=1.11.0" requirement, so the main requirements.txt install below
# sees torch already installed and skips it entirely — never downloading
# any CUDA package.
RUN pip install --no-cache-dir --user \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --user -r /requirements.txt

WORKDIR /opt/airflow