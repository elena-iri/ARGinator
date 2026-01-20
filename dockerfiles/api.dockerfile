FROM ghcr.io/astral-sh/uv:python3.11-alpine AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

RUN uv sync --frozen --no-install-project

COPY src src/

RUN uv sync --frozen

EXPOSE $PORT

ENTRYPOINT ["streamlit", "run", "frontend.py", "--server.port", "$PORT", "--server.address=0.0.0.0"]