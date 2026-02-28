FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/tmp/models

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY libs/core /app/libs/core
COPY libs/providers /app/libs/providers
COPY apps/worker /app/apps/worker

RUN uv pip install --system -e /app/libs/core -e /app/libs/providers -e /app/apps/worker

WORKDIR /app/apps/worker

CMD ["python", "-m", "app.main"]

