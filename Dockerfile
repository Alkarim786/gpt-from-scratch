# syntax=docker/dockerfile:1
# CPU image: prepare data, train, evaluate and sample inside a container.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY tinygpt ./tinygpt
COPY scripts ./scripts
RUN pip install .
RUN useradd --create-home app && mkdir -p /app/data /app/runs && chown -R app /app
USER app
# Default: full pipeline with the small CPU config (see README for knobs).
CMD ["sh", "-c", "python scripts/download_data.py && python scripts/prepare.py && python -m tinygpt.train --steps 1500 && python -m tinygpt.evaluate"]
