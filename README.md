# Zepto Platform

A production-oriented data and AI platform in three parts: a resilient ingestion
pipeline, a reproducible ML analytics package, and a retrieval-augmented support
assistant served over HTTP.

## Status

**Ingestion is complete and verified end to end.** `core` and `ingestion` are ported,
fully typed, and covered by 114 tests at 100% branch coverage, enforced in CI.
`analytics` and `assistant` are not yet ported; see [Roadmap](#roadmap).

Latest verified run against the live source:

```
474 books, 9 categories, 0 rejected, 20.9s
foreign keys enforced, 0 orphan rows, 0 constraint violations, 0 price mismatches
```

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

## Running the ingestion pipeline

```bash
zepto-ingest
```

Scrapes the configured categories, validates every record, and atomically replaces
the catalogue at `data/books.db`. Everything is configurable by environment variable
without touching code:

```bash
ZEPTO_INGESTION_TIMEOUT_SECONDS=30
ZEPTO_INGESTION_MAX_RETRIES=5
ZEPTO_INGESTION_MIN_BOOKS=100
ZEPTO_LOG_JSON=true            # machine-readable logs for production
```

## Development

```bash
ruff check .          # lint
ruff format .         # format
mypy src tests        # type check
pytest                # tests + coverage (gate: 90%)
```

All four run automatically on every commit via pre-commit, and again in CI on every
push.

## Design decisions

**Money is `Decimal` in the domain and integer minor units in storage, never `float`.**
Binary floating point cannot represent most decimal fractions exactly. In v1 this
surfaced as five rows where Python's `round()` (banker's rounding) and SQLite's
`ROUND()` (half away from zero) disagreed on exact half-paisa ties — a discrepancy
that had to be investigated and explained rather than prevented. Decimal arithmetic
with an explicit rounding mode, stored as exact integers, makes the class of bug
impossible.

**Validation happens before storage, not after.** A scrape degraded by a site change
should fail the run, not quietly replace a good catalogue with a thin one. Combined
with the repository's atomic file swap, a failure at any point leaves the previous
catalogue untouched.

**Foreign keys are enforced, not merely declared.** SQLite ignores foreign key
constraints unless `PRAGMA foreign_keys = ON` is issued on every connection. v1
declared the constraint, documented it as enforced, and manually audited for orphans —
but the database was never actually checking.

**Dependencies are injected, not constructed internally.** The scraper takes an HTTP
session, the pipeline takes a scraper and a repository, each declared as a `Protocol`
covering only what is actually used. Every test runs without network access.

**Configuration is environment-driven and validated at startup.** A typo like
`ZEPTO_ENVIRONMENT=prodction` fails immediately with a clear error rather than
silently taking a wrong branch later.

## Roadmap

| Phase | Focus | Status |
| --- | --- | --- |
| 0 | Packaging, linting, type checking, CI, pre-commit | ✅ Done |
| 1 | Port `core` and `ingestion`, fixing the v1 defects | ✅ Done |
| 2 | Port `analytics`: notebook logic extracted into importable modules | Next |
| 3 | Port `assistant`: model lifecycle, input limits, real confidence scoring | Planned |
| 4 | Production API: health checks, auth, rate limiting, metrics, tracing | Planned |
| 5 | ML maturity: experiment tracking, model registry, model card | Planned |
| 6 | RAG quality and AI safety: retrieval evaluation, reranking, injection defenses | Planned |
| 7 | Deployment: hardened image, dependency lockfile, secret management, monitoring | Planned |

### v1 defects addressed so far

| Defect | Status |
| --- | --- |
| HTTP calls with no timeout, status check, or retries | Fixed, with regression tests |
| Parser crashed the entire run on one malformed listing | Fixed — skipped and logged, run continues |
| Foreign keys declared but never enforced | Fixed — `PRAGMA foreign_keys = ON`, orphans rejected |
| Destructive load left an empty database on crash | Fixed — atomic file swap |
| Money handled as `float` | Fixed — `Decimal` + integer minor units |
| Unvalidated dicts passed between layers | Fixed — Pydantic contracts at each boundary |
| `print()` for all output | Fixed — structured logging |
| Embedding model reloaded on every query | Pending (`assistant`) |
| No input length limit on the API | Pending (`assistant`) |
| Dead, unreachable retry path in LLM code | Pending (`assistant`) |
| Hardcoded `confidence: 1.0` | Pending (`assistant`) |
| Unpinned dependencies, container runs as root | Pending (Phase 7) |

## License

MIT
