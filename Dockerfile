FROM python:3.12-slim

LABEL maintainer="Ben Sherlock"
LABEL description="Multi-provider LLM proxy with Ollama and OpenAI API compatibility"
LABEL version="2.0.0"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY proxy.py .
COPY providers/ ./providers/
COPY config/ ./config/
COPY db/ ./db/
COPY admin/ ./admin/

# Create non-root user and data directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appuser /app
USER appuser

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

# Run with Python directly (handles both servers internally)
CMD ["python", "proxy.py"]
