FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/models

WORKDIR /app

RUN pip install --no-cache-dir uv

# Install CPU-only torch first to avoid pulling the 2GB+ CUDA bundle
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

COPY libs/core /app/libs/core
COPY libs/providers /app/libs/providers
COPY apps/worker /app/apps/worker

RUN uv pip install --system -e /app/libs/core -e /app/libs/providers -e /app/apps/worker

ARG PRELOAD_EMBEDDING=false
RUN if [ "${PRELOAD_EMBEDDING}" = "true" ]; then \
      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"; \
    fi

WORKDIR /app/apps/worker

CMD ["python", "-m", "app.main"]

