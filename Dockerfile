# Use an official Python 3.10 base image
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

RUN pip install --no-cache-dir igraph

# Core numeric stack
RUN pip install --no-cache-dir 'numpy<2'

# Visualization and ML libs
RUN pip install --no-cache-dir \
    matplotlib \
    scikit-learn \
    seaborn

# Torch + PyG ecosystem (CPU-only)
RUN pip install torch==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    torch-geometric \
    --find-links https://data.pyg.org/whl/torch-2.0.1+cpu.html

# Other utilities
RUN pip install --no-cache-dir \
    deepsnap==0.2.1 \
    networkx \
    test-tube==0.7.5 \
    tqdm==4.43.0 \
    requests \
    'sentence-transformers>=2.2,<4' \
    'transformers<4.36' \
    scipy

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

