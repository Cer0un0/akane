FROM node:22-slim

ENV CODEX_HOME=/var/lib/codex

RUN npm install -g @openai/codex \
    && mkdir -p /var/lib/codex \
    && chown -R node:node /var/lib/codex

USER node
WORKDIR /home/node

EXPOSE 4000

CMD ["codex", "app-server", "--listen", "ws://0.0.0.0:4000"]
