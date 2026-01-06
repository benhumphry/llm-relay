FROM python:3.12-slim

LABEL maintainer="Ben Sherlock"
LABEL description="LLM Relay - Multi-provider proxy with Ollama and OpenAI API compatibility"
LABEL version="1.0.2"

# Install gosu for stepping down from root
RUN apt-get update && apt-get install -y --no-install-recommends gosu && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY proxy.py .
COPY providers/ ./providers/
COPY db/ ./db/
COPY admin/ ./admin/
COPY tracking/ ./tracking/
COPY routing/ ./routing/
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
