# Container image for the support assistant service.
#
# Three properties this image is built for, none of which v1's single-stage
# Dockerfile had:
#
# Reproducible. The base is pinned by digest, not by a floating tag, and
# dependencies come from a fully-resolved lock rather than from ">=" floors. A
# rebuild months from now produces the same image instead of whatever the
# registry and the resolver happen to serve that day.
#
# Small and unprivileged. Build tooling lives in a stage that is discarded; the
# runtime stage carries an interpreter, a virtualenv, and the corpus, and runs
# as a non-root user that owns nothing it does not need to write.
#
# Self-contained at runtime. The vector index -- including the ONNX embedding
# model chromadb downloads on first use -- is built during the image build. A
# container therefore starts without reaching the network, and its first request
# is not the one that pays for indexing.

# ---------------------------------------------------------------------------
# Stage 1: build the virtualenv
# ---------------------------------------------------------------------------
FROM python:3.12-slim-trixie@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Dependencies first, from the lock alone. This layer is keyed only on the lock
# file, so editing source code does not reinstall ~100 packages.
COPY docker/requirements-assistant.lock ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements-assistant.lock

# Then the application itself, with --no-deps: the lock is the authority on what
# is installed, and resolving again here could quietly drift from it.
COPY pyproject.toml README.md ./
COPY src ./src
RUN /opt/venv/bin/pip install --no-cache-dir --no-deps .

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim-trixie@sha256:e5c9fa26ffb76e11e0f054f30dc2523a2f9693f0c36c0cf1e39b27e152d899fc AS runtime

LABEL org.opencontainers.image.title="zepto-support-assistant" \
      org.opencontainers.image.description="Retrieval-augmented support assistant over the Zepto policy corpus" \
      org.opencontainers.image.source="https://github.com/karanamsai1976-dotcom/zepto-platform" \
      org.opencontainers.image.licenses="MIT"

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ZEPTO_ASSISTANT_CORPUS_DIR=/app/data/corpus \
    ZEPTO_ASSISTANT_CHROMA_DIR=/app/data/chroma \
    ZEPTO_ASSISTANT_HOST=0.0.0.0 \
    ZEPTO_ASSISTANT_PORT=8000 \
    HOME=/app

# A dedicated unprivileged account. Running as root inside a container means a
# container escape starts from root, and any process compromise can rewrite the
# application's own code.
RUN useradd --create-home --home-dir /app --shell /usr/sbin/nologin --uid 10001 zepto

COPY --from=builder --chown=root:root /opt/venv /opt/venv
COPY --chown=zepto:zepto data/corpus /app/data/corpus

WORKDIR /app
USER zepto

# Build the index at image build time. This both bakes in the embedding model,
# removing the runtime network dependency, and moves the one-off indexing cost
# out of the first request. Building it as `zepto` matters: built as root it
# would be unreadable-for-write by the account that actually runs.
RUN python -c "\
from zepto.assistant.retrieval import VectorStore, load_corpus; \
from zepto.assistant.settings import get_assistant_settings; \
s = get_assistant_settings(); \
store = VectorStore(settings=s); \
print('indexed', store.ingest(load_corpus(s.corpus_dir)), 'documents')"

EXPOSE 8000

# /ready rather than /health: it reports whether this instance can actually
# answer, which is the question an orchestrator is asking. There is no curl in
# the slim base, and adding one to run a health check would be a strange trade.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "\
import sys, urllib.request; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status == 200 else 1)"

CMD ["zepto-serve"]
