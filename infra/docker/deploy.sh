#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Default overlay
OVERLAY="${1:-docker-compose.vps.yml}"

echo "==> git pull"
git -C ../.. pull

echo "==> docker builder prune"
docker builder prune -f || true

echo "==> docker compose down"
docker compose down

echo "==> docker compose up --build (overlay: $OVERLAY)"
docker compose -f docker-compose.yml -f "$OVERLAY" up -d --build

echo "==> done"
docker compose ps
