# akane

Discord 上で動作するコンテナ型 AI エージェントの実装リポジトリです。

## Workspace

- `apps/api` FastAPI サービス
- `apps/bot` Discord Bot サービス
- `apps/worker` 非同期ワーカー
- `apps/scheduler` 定期ジョブスケジューラ
- `libs/core` コアドメイン
- `libs/providers` LLM プロバイダ抽象層
- `infra/docker` compose 定義と環境変数
- `infra/workspace` ツール実行用ワークディレクトリ
- `design` 設計書

## 起動（開発）

```bash
cd infra/docker
cp .env.example .env
# 必須値を編集（AKANE_BOT_TOKEN, POSTGRES_PASSWORD, AKANE_CODEX_BASE_URL など）
# Codex app-server を使う場合: AKANE_CODEX_BASE_URL=ws://host.docker.internal:4000
docker compose up -d
```

## 起動（VPS: すべて同一ホスト）

```bash
cd infra/docker
cp .env.example .env
# 必須値を編集
# AKANE_BOT_TOKEN=...
# POSTGRES_PASSWORD=...
# AKANE_CODEX_API_TOKEN=...   # 必須（APIキー方式）
# AKANE_CODEX_BASE_URL 未設定時は https://api.openai.com
docker compose -f docker-compose.yml -f docker-compose.vps.yml -f docker-compose.prod.yml up -d --build
```

## 起動（VPS: Codex app-server 版）

```bash
cd infra/docker
cp .env.example .env
# 必須値を編集
# AKANE_BOT_TOKEN=...
# POSTGRES_PASSWORD=...
# AKANE_CODEX_BASE_URL は未設定でOK（ws://codex-app-server:4000 が使われる）
# AKANE_CODEX_API_TOKEN は任意（空なら codex login 方式）
docker compose -f docker-compose.yml -f docker-compose.vps.codex.yml -f docker-compose.prod.yml up -d --build
```

補足:

- `docker-compose.vps.yml`: APIキー直結（`https://api.openai.com`）
- `docker-compose.vps.codex.yml`: `codex-app-server` 経由（WS）
- `codex-app-server` を使う場合、`AKANE_CODEX_API_TOKEN` が空なら `codex login --device-auth` を1回実行
- API 側は WS 接続時に `AKANE_CODEX_API_TOKEN` があれば `account/login/start(type=apiKey)` を実行
