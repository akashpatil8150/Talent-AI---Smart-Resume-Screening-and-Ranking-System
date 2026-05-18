# Use Python 3.11 slim base image for smaller size
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

# Install Python dependencies
# Use CPU-only PyTorch for smaller image size
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn==21.2.0

# Download NLTK data during build
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Copy application files
COPY . .

# Create cache directory for embeddings
RUN mkdir -p .bert_cache .cache

# Set environment variables for production
ENV PORT=8080
ENV BERT_FORCE_CPU=true
ENV SKIP_BERT_PRECOMPUTE=true
ENV PYTHONUNBUFFERED=1
ENV FLASK_DEBUG=false

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run with gunicorn
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --max-requests 1000 --access-logfile - --error-logfile -
