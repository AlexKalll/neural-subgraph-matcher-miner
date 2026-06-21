# Use an official Python 3.10 base image (Debian-based)
FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Make pip installs on slower networks
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_NO_BUILD_ISOLATION=1

# Install system build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libigraph-dev python3-igraph \
    pkg-config \
    git \
    libfreetype6-dev \
    libpng-dev \
    libqhull-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip/setuptools/wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Core numeric stack
RUN pip install --no-cache-dir numpy

# Visualization and ML libs
RUN pip install --no-cache-dir \
    matplotlib \
    scikit-learn \
    seaborn

# Torch + PyG ecosystem (CPU-only, PyG 2.3+ needs no separate scatter/sparse)
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    --find-links https://data.pyg.org/whl/torch-2.0.1+cpu.html

RUN pip install --no-cache-dir torch-geometric

# Other utilities
RUN pip install --no-cache-dir \
    deepsnap \
    networkx \
    test-tube \
    tqdm \
    requests

# Install FastAPI and related packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart

# Copy the project
COPY . .

# Expose port
EXPOSE 5000

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]
