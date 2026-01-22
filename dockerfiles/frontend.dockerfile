FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install deps (Cached)
COPY uv.lock pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy Source
COPY src/arginator_protein_classifier/frontend.py src/arginator_protein_classifier/frontend.py
COPY README.md README.md
# Streamlit typically doesn't need 'configs/' unless your code reads them explicitly

# Install Project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

ENV PORT=8080
EXPOSE 8080

# Run Streamlit
# Note: We assume your frontend code is saved at src/arginator_protein_classifier/frontend.py
CMD ["uv", "run", "streamlit", "run", "src/arginator_protein_classifier/frontend.py", "--server.port", "8080", "--server.address", "0.0.0.0"]