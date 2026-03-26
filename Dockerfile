FROM python:3.11-slim

WORKDIR /app

# Install system deps for PDF parsing
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Model cache directory — mount a Railway volume here to persist across deploys
# Models are downloaded at first startup, not baked into the image
ENV HF_HOME=/app/models/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/models/sentence_transformers
ENV FASTEMBED_CACHE_PATH=/app/models/fastembed

COPY . .

EXPOSE 8000

# Railway injects $PORT dynamically; fallback to 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
