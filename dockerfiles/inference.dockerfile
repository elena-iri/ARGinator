FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

#Essentials of python installation and build tools 

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# RUN uv sync --frozen --no-install-project 
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-install-project 
#These commands use the cache to speed up installation of dependencies

COPY src/ src/
COPY README.md README.md
COPY data/processed_data.pt data/processed_data.pt
COPY models/model.pth models/model.pth

RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "src/arginator_protein_classifier/evaluate.py"]
