FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY apps/scheduler /app/apps/scheduler

RUN uv pip install --system -e /app/apps/scheduler

WORKDIR /app/apps/scheduler

CMD ["python", "-m", "app.main"]

