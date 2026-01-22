FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS base

# 1. Set working directory
WORKDIR /app

# 2. Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# 3. Install system dependencies (GCC is often needed for pandas/numpy extensions)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Copy lockfiles first (for caching)
COPY uv.lock pyproject.toml ./

# 5. Install dependencies only (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# 6. Copy the rest of the application
COPY src/ src/
COPY configs/ configs/
COPY README.md README.md

# 7. Install the project itself (Makes 'src' importable)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# 8. Expose the port Cloud Run expects
ENV PORT=8080
EXPOSE 8080

# 9. Run FastAPI using uvicorn
# Note: We assume your api code is saved at src/arginator_protein_classifier/api.py
CMD ["uv", "run", "uvicorn", "src.arginator_protein_classifier.backend:app", "--host", "0.0.0.0", "--port", "8080"]