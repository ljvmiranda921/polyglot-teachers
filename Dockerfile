FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen
COPY scripts/ ./scripts/
COPY data/ ./data/

ENV PYTHONUNBUFFERED=1

CMD ["/bin/bash"]
