# Development container for GenRF Circuit Diffuser
# Based on official Python image with ML/AI tools and SPICE simulation support

FROM mcr.microsoft.com/devcontainers/python:3.10-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    pkg-config \
    # System libraries
    libffi-dev \
    libssl-dev \
    # Scientific computing
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    # Graphics and visualization
    libfreetype6-dev \
    libpng-dev \
    libjpeg-dev \
    # SPICE simulation tools
    ngspice \
    ngspice-doc \
    # Additional tools
    git \
    curl \
    wget \
    vim \
    htop \
    tree \
    jq \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Conda for better Python environment management
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -afy

# Add conda to PATH
ENV PATH="/opt/conda/bin:$PATH"

# Create conda environment for development
RUN conda create -n genrf python=3.10 -y \
    && conda clean -afy

# Activate conda environment by default
ENV CONDA_DEFAULT_ENV=genrf
ENV CONDA_PREFIX=/opt/conda/envs/genrf
ENV PATH="$CONDA_PREFIX/bin:$PATH"

# Install PyTorch with CUDA support (if GPU available)
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y \
    && conda clean -afy

# Install scientific computing libraries
RUN conda install numpy scipy matplotlib pandas scikit-learn jupyter ipykernel -y \
    && conda clean -afy

# Install development tools
RUN pip install --no-cache-dir \
    black \
    isort \
    flake8 \
    mypy \
    bandit \
    pytest \
    pytest-cov \
    pytest-benchmark \
    pre-commit \
    safety \
    # Documentation tools
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    # SPICE integration
    PySpice \
    # Additional ML/AI tools
    plotly \
    dash \
    wandb \
    tensorboard

# Install additional system tools for development
RUN apt-get update && apt-get install -y \
    # Network tools for debugging
    netcat \
    telnet \
    dnsutils \
    # Process monitoring
    procps \
    # File utilities
    file \
    unzip \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create workspace directories
RUN mkdir -p /opt/genrf/{models,cache,logs,outputs} \
    && chown -R vscode:vscode /opt/genrf

# Set up Git configuration for the container
USER vscode
RUN git config --global init.defaultBranch main \
    && git config --global pull.rebase false \
    && git config --global core.autocrlf input

# Create Jupyter kernel for the conda environment
RUN python -m ipykernel install --user --name genrf --display-name "GenRF Circuit Diffuser"

# Set working directory
WORKDIR /workspaces/genrf-circuit-diffuser

# Copy and run setup script
COPY .devcontainer/setup.sh /tmp/setup.sh
RUN chmod +x /tmp/setup.sh

# Default command
CMD ["/bin/bash"]