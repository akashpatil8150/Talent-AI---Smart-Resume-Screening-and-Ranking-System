# Hugging Face Spaces compatible Dockerfile
# HF Spaces runs as non-root user 1000, port 7860
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (includes PyTorch CPU + sentence-transformers)
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Copy application files
COPY . .

# Create cache directories and set permissions for HF Spaces (runs as user 1000)
RUN mkdir -p .bert_cache .cache && \
    chmod -R 777 .bert_cache .cache

# Set environment variables for production
ENV PORT=7860
ENV BERT_FORCE_CPU=true
ENV SKIP_BERT_PRECOMPUTE=true
ENV MATCHING_MODE=hybrid
ENV PYTHONUNBUFFERED=1
ENV FLASK_DEBUG=false
ENV TRANSFORMERS_CACHE=/app/.cache

# Expose HF Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Run with gunicorn - 1 worker to stay within memory
CMD gunicorn app:app --bind 0.0.0.0:7860 --workers 1 --timeout 120 --max-requests 1000 --access-logfile - --error-logfile -
