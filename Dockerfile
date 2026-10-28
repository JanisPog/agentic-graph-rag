# ── Stage 1: dependency builder ───────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile some wheels (sentence-transformers, spacy, neo4j)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY a2a/requirements.txt a2a_requirements.txt

RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt \
 && pip install --prefix=/install --no-cache-dir -r a2a_requirements.txt


# ── Stage 2: runtime image ────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
COPY agents/       agents/
COPY graph/        graph/
COPY etl/          etl/
COPY a2a/          a2a/
COPY config.py     config.py
COPY graph_builder.py graph_builder.py
COPY run.py        run.py

# A2A server port
EXPOSE 8015

# Run from /app so that `from graph_builder import graph` and sibling imports work
CMD ["python", "a2a/main.py"]
