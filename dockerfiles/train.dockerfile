FROM ghcr.io/astral-sh/uv:python3.11-alpine AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

#Essentials of python installation and build tools 
RUN apt update && \ 
    apt install --no-install-recommends - y build-essential gcc && \ 
    apt clean && rm -rf /var/lib/apt/lists/*

RUN uv sync --frozen --no-install-project 

COPY src/ src/
COPY README.md README.md
COPY data/ data/

RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT = ["uv", "run", "src/arginator_protein_classifier/train.py"]
# The -u flag ensures that any output goes to the terminal and doesn't stay in the container (if so use "docker logs")
ENTRYPOINT ["uv", "run", "src/arginator_protein_classifier/train.py"]
