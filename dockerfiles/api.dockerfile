# Base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml

#Essentials of python installation and build tools 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN uv sync --frozen --no-install-project 

COPY ./src/ ./src/
COPY README.md README.md
#COPY data/ data/

RUN uv sync --locked --no-cache --no-install-project

ENTRYPOINT ["uv", "run", "uvicorn", "src.arginator_protein_classifier.api:app", "--host", "0.0.0.0", "--port", "4001"]
