# Zepto Platform

A production-oriented data and AI platform in three parts: a resilient ingestion
pipeline, a reproducible ML analytics package, and a retrieval-augmented support
assistant served over HTTP.

## Status

**Phase 0 — foundation.** Packaging, tooling, and CI are being established before any
module is ported. Nothing is production-ready yet; see [Roadmap](#roadmap).

## Background

This is a ground-up rebuild of an earlier capstone project
(`zepto-data-ai-platform-capstone`, frozen at tag `v1.0-submission`). That version met
its academic requirements, but was structured as three independent folders of flat
scripts with no shared configuration, logging, error handling, packaging, or CI, and
carried several latent defects — an embedding model reloaded on every request,
un-timeouted HTTP calls, unenforced foreign keys, and unpinned dependencies among
them. This rebuild keeps the domain logic and rebuilds everything around it.

## Structure

```
src/zepto/
├── core/         shared settings, structured logging, typed errors
├── ingestion/    web scraping -> validation -> relational storage
├── analytics/    feature engineering, training, evaluation (importable, not notebook-bound)
└── assistant/    retrieval-augmented Q&A API
tests/
├── unit/         fast, no external dependencies
└── integration/  network, disk, and service-backed tests
notebooks/        thin narrative wrappers over src/zepto/analytics
docker/           container definitions
```

## Setup

Requires **Python 3.12**.

```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -e ".[all,dev]"
```

Dependency groups are declared as extras in `pyproject.toml` — install only what you
need (`.[ingestion]`, `.[assistant]`, etc.) rather than the whole stack.

## Development

```bash
ruff check .          # lint
ruff format .         # format
mypy src              # type check
pytest                # tests + coverage
```

All four run automatically on every commit via pre-commit, and again in CI on every
push.

## Roadmap

| Phase | Focus |
| --- | --- |
| 0 | Packaging, linting, type checking, CI, pre-commit |
| 1 | Fix the ten known defects carried over from v1 |
| 2 | Shared core: settings, logging, errors, data contracts |
| 3 | Production API: health checks, auth, rate limiting, metrics, tracing |
| 4 | ML maturity: notebook logic extracted to modules, experiment tracking, model registry, model card |
| 5 | RAG quality and AI safety: retrieval evaluation, reranking, prompt-injection defenses |
| 6 | Deployment: hardened image, secret management, monitoring |

## License

MIT
