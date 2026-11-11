# Multi-stage Dockerfile for RAG Document Q&A Pipeline
FROM python:3.10-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency specification
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and sample data
COPY config/ config/
COPY src/ src/
COPY evaluation/ evaluation/
COPY sample_data/ sample_data/
COPY ui/ ui/

# Create data directories
RUN mkdir -p data/faiss_index evaluation/results

# Default environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    EMBEDDING_PROVIDER=huggingface \
    EMBEDDING_DEVICE=cpu

EXPOSE 8000 8501

# Default command: launch FastAPI server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
