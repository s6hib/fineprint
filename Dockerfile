# Three stages: build the web page, install the Python packages, then put both in a
# small image with the code and the saved index.

FROM node:24-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.12-slim AS packages
COPY --from=ghcr.io/astral-sh/uv:0.11.24 /uv /bin/
# Use the image's Python, copy files out of the cache, and compile them once here so
# the app starts faster.
ENV UV_PYTHON_DOWNLOADS=0 UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

FROM python:3.12-slim
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
COPY --from=packages /app/.venv .venv
# uv.lock and the PDFs are here because the index's fingerprint is a hash of them.
COPY api.py drafts.py fineprint.py limits.py uv.lock index.json meta.json ./
COPY data/ data/
COPY --from=web /web/dist web/dist
# Stop the build if the saved index does not match these files. Without this check the
# app would rebuild the index when it starts, and that calls the paid embedding service.
RUN python -c "import json, fineprint; saved = json.load(open('index.json'))['fingerprint']; assert saved == fineprint.fingerprint(fineprint.DATA_DIR), 'index.json is stale: run load_index() before building'"
# Keyword search needs SQLite's FTS5. Fail here, not on the first question, if it is missing.
RUN python -c "import sqlite3; sqlite3.connect(':memory:').execute('create virtual table t using fts5(x)')"
EXPOSE 8080
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
