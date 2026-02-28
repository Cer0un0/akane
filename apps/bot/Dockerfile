FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY apps/bot /app/apps/bot

RUN uv pip install --system -e /app/apps/bot

WORKDIR /app/apps/bot

CMD ["python", "-m", "app.main"]

