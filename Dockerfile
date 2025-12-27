FROM python:3.12-slim

LABEL maintainer="Ben Sherlock"
LABEL description="Ollama API proxy for Anthropic Claude models"
LABEL version="1.0.0"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY proxy.py .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default port (Ollama default)
ENV PORT=11434
ENV HOST=0.0.0.0

EXPOSE 11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/')" || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:11434", "--workers", "4", "--threads", "2", "--timeout", "120", "proxy:app"]
