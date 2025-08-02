# Multi-stage build for GenRF Circuit Diffuser
FROM python:3.13-slim AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    ngspice \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r genrf && useradd -r -g genrf genrf

# Set work directory
WORKDIR /app

# Development stage
FROM base AS development

# Install development dependencies
COPY requirements-dev.txt .
COPY requirements.txt .
RUN pip install -r requirements-dev.txt

# Copy source code
COPY --chown=genrf:genrf . .

# Install package in development mode
RUN pip install -e ".[dev,spice]"

USER genrf

CMD ["python", "-m", "genrf.cli", "--help"]

# Production stage
FROM base AS production

# Install only production dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code
COPY --chown=genrf:genrf genrf/ ./genrf/
COPY --chown=genrf:genrf pyproject.toml .
COPY --chown=genrf:genrf README.md .

# Install package
RUN pip install -e ".[spice]"

USER genrf

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import genrf; print('GenRF is ready')"

ENTRYPOINT ["python", "-m", "genrf.cli"]

# Jupyter notebook stage for experimentation
FROM development AS jupyter

# Install Jupyter
RUN pip install jupyterlab

# Expose Jupyter port
EXPOSE 8888

USER genrf

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]