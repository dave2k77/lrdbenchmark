# LRDBenchmark Docker Image
# Multi-stage build for minimal production image

# Stage 1: Build environment
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Stage 2: Production environment
FROM python:3.10-slim as production

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY lrdbenchmark/ ./lrdbenchmark/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY pyproject.toml README.md ./

# Create non-root user for security
RUN useradd --create-home appuser && \
    chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    JAX_PLATFORM_NAME=cpu

# Default command: run quick benchmark
CMD ["python", "scripts/benchmarks/run_classical_failure_benchmark.py", "--profile", "quick"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import lrdbenchmark; print('OK')" || exit 1

# Labels
LABEL org.opencontainers.image.title="LRDBenchmark" \
      org.opencontainers.image.description="Long-Range Dependence Estimation Benchmarking Library" \
      org.opencontainers.image.version="3.0.0" \
      org.opencontainers.image.source="https://github.com/dave2k77/lrdbenchmark"
