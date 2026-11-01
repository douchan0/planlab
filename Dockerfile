# PlanLab — reproducible environment.
#
#   docker build -t planlab .
#   docker run --rm -v "$PWD/results:/app/results" planlab python -m planlab.cli verify
#   docker run --rm planlab bash scripts/reproduce.sh
#
# The processed data are committed, so the image runs offline; `data/build` fetches
# the source table from Eurostat when network access is available.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/mpl \
    PYTHONPATH=/app/src

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY tests ./tests
COPY data ./data

RUN pip install --no-cache-dir -e ".[dev,notebooks]"

COPY . .

CMD ["python", "-m", "planlab.cli", "verify"]
