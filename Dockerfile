FROM python:3.12-slim

LABEL maintainer="Ben Humphry"
LABEL description="LLM Relay - Multi-provider proxy with Ollama and OpenAI API compatibility"

# Install gosu for stepping down from root, and build tools for native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gosu \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Reinstall PyTorch with CUDA 12.6 to ensure Pascal GPU support (sm_61)
# CUDA 12.8+ builds dropped Pascal/Maxwell support, cu126 is the last to include sm_61
# See: https://dev-discuss.pytorch.org/t/cuda-toolkit-version-and-architecture-support-update-maxwell-and-pascal-architecture-support-removed-in-cuda-12-8-and-12-9-builds/3128
RUN pip install --no-cache-dir --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Copy application
COPY VERSION .
COPY version.py .
COPY proxy.py .
COPY providers/ ./providers/
COPY db/ ./db/
COPY admin/ ./admin/
COPY tracking/ ./tracking/
COPY routing/ ./routing/
COPY rag/ ./rag/
COPY context/ ./context/
COPY augmentation/ ./augmentation/
COPY scripts/ ./scripts/
COPY config/ ./config/

# Copy entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create non-root user and data directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /data && \
    chown -R appuser:appuser /app /data

# Default ports
ENV PORT=11434
ENV ADMIN_PORT=8080
ENV HOST=0.0.0.0

# Expose both API and Admin ports
EXPOSE 11434
EXPOSE 8080

# Health check against API server
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/')" || exit 1

# Use entrypoint to fix permissions, then run as appuser
ENTRYPOINT ["/entrypoint.sh"]
CMD ["gosu", "appuser", "python", "proxy.py"]
