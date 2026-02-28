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
# AKANE_CODEX_API_TOKEN=...   # APIキー方式を使う場合のみ
# AKANE_CODEX_BASE_URL 未設定時は vps overlay が ws://codex-app-server:4000 を使う
docker compose -f docker-compose.yml -f docker-compose.vps.yml -f docker-compose.prod.yml up -d --build
```

補足:

- `docker-compose.vps.yml` で `codex-app-server` コンテナを追加し、`akane-api` から内部DNSで接続する
- 認証方式は2つ:
  - APIキー方式: `AKANE_CODEX_API_TOKEN` を設定
  - Codexログイン方式: `AKANE_CODEX_API_TOKEN` は空にして、`codex login --device-auth` を1回実行
- API 側は WS 接続時に `AKANE_CODEX_API_TOKEN` があれば `account/login/start(type=apiKey)` を実行する
- 詳細手順: `docs/vps-deployment.md`
