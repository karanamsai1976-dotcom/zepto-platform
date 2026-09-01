# Zepto Platform

A production-oriented data and AI platform in three parts: a resilient ingestion
pipeline, a reproducible ML analytics package, and a retrieval-augmented support
assistant served over HTTP.

## Status

**All four modules are ported and verified end to end** — `core`, `ingestion`,
`analytics`, and `assistant` — fully typed, covered by 418 tests at ~100% branch
coverage, enforced in CI on every push. The assistant also ships as a container
image that CI builds, runs, queries, and vulnerability-scans on every change.

Latest measured assistant quality, against a labelled 77-case evaluation set
(`zepto-eval`). The two splits are reported separately on purpose: `dev` is what
the relevance floor was tuned against, and `test` was held back from every tuning
decision, so only the second column estimates performance on questions nobody
fitted to.

| Metric | dev (34) | test (43, held out) |
| --- | --- | --- |
| Retrieval hit rate @1 | 86.2% | 88.9% |
| Retrieval hit rate @3 | 96.6% | 100% |
| Mean reciprocal rank | 0.914 | 0.940 |
| Scope decision accuracy | 100% | 97.7% |
| In-scope questions answered | 100% | 100% |
| Out-of-scope questions declined | 100% | **85.7%** |

The last cell is the one worth reading. Retrieval generalises; the scope decision
does not, quite. `"Convert 500 dollars to rupees"` clears the relevance floor
because currency wording sits close to the delivery-fee document. It is left
standing rather than fixed by moving the floor, since retuning on the split that
caught it is exactly what the split exists to prevent.

Latest verified ingestion run against the live source:

```
474 books, 9 categories, 0 rejected, 20.9s
foreign keys enforced, 0 orphan rows, 0 constraint violations, 0 price mismatches
```

Latest verified training run:

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic regression | 0.804 | 0.793 | 0.667 | 0.724 | 0.844 |
| Decision tree | 0.816 | 0.790 | 0.710 | **0.748** | 0.790 |
| Random forest | 0.816 | 0.800 | 0.696 | 0.744 | 0.829 |

Selected by F1: decision tree. Logistic regression and decision tree reproduce v1's
figures to three decimals, confirming the rewrite changed the structure without
changing the behaviour.

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

## Training models

```bash
zepto-train
```

Loads the versioned dataset, builds features (refusing any that leak the target),
trains every configured model on an identical stratified split, and persists each as
a timestamped version with its provenance. Artifacts land under `models/<name>/<version>/`
as a `pipeline.joblib` plus a `metadata.json` recording the scikit-learn version,
Python version, platform, feature names, row counts, and metrics.

```bash
ZEPTO_ANALYTICS_TEST_SIZE=0.25
ZEPTO_ANALYTICS_SELECTION_METRIC=roc_auc
ZEPTO_ANALYTICS_MAX_SINGLE_FEATURE_ACCURACY=0.95   # tighten the leakage guard
```

## Running the support assistant

```bash
zepto-serve
```

Starts the API on port 8000, indexing the policy corpus at startup if the vector
store is empty. No API key is required: the default mode answers deterministically
from retrieved policy text.

```bash
curl -X POST localhost:8000/ask -H 'Content-Type: application/json' \
     -d '{"query": "What is the delivery fee?"}'
curl localhost:8000/health
curl localhost:8000/ready
```

```bash
ZEPTO_ASSISTANT_TOP_K=5
ZEPTO_ASSISTANT_MIN_RELEVANCE=0.20   # decline more readily
ZEPTO_ASSISTANT_MAX_QUERY_LENGTH=300
ZEPTO_ASSISTANT_MOCK_LLM=false       # requires GROQ_API_KEY and `pip install groq`
```

### Hardening it for exposure

Authentication and rate limiting are off and permissive by default, so the demo
runs with no setup. Both fail closed when switched on — enabling authentication
without configuring keys stops startup rather than serving an endpoint that
believes it is protected.

```bash
ZEPTO_ASSISTANT_REQUIRE_API_KEY=true
ZEPTO_ASSISTANT_API_KEYS=key-one,key-two   # sent as the X-API-Key header
ZEPTO_ASSISTANT_RATE_LIMIT_REQUESTS=60     # per client, per window
ZEPTO_ASSISTANT_RATE_LIMIT_WINDOW_SECONDS=60
ZEPTO_ASSISTANT_HOST=0.0.0.0               # loopback by default
```

`/health`, `/ready`, and `/metrics` bypass both guards deliberately: probes hit
them constantly, and a monitoring endpoint that needs a credential ends up
unmonitored. `/metrics` serves Prometheus exposition labelled by matched route
template rather than raw path, so unmatched URLs cannot inflate label
cardinality.

## Running it in Docker

```bash
docker build -f docker/Dockerfile -t zepto-assistant:0.1.0 .
docker run -d -p 8000:8000 zepto-assistant:0.1.0
```

Base pinned by digest, dependencies from a lock generated inside the target
container, build tooling discarded in a separate stage, runs as uid 10001, and
the vector index plus embedding model baked at build time so a container starts
without reaching the network. [`docker/README.md`](docker/README.md) records the
measured behaviour and the known limitations, including 38 base-OS advisories
with no upstream fix.

## Generating a model card

```bash
zepto-card
```

Writes `MODEL_CARD.md` beside the stored artifact, so the description stays
attached to the thing it describes. The quantitative sections are computed from
the model rather than written, because sections written by hand are the ones that
end up flattering.

Two of those computed sections carry the substance. **Disaggregated performance**
slices at attribute *intersections*, not single attributes — on this model every
single-attribute slice looks unremarkable, while the intersection shows it
predicts death for every man in second and third class, with no discriminating
ability whatsoever. **Comparison against a trivial baseline** asks how much the
model adds over a lookup table of majority class per subgroup: on this data,
+2.8 accuracy points, agreeing with the lookup on 92.7% of rows.

## Evaluating retrieval quality

```bash
zepto-eval
```

Scores retrieval, and the scope decision, against the labelled cases in
`data/eval/retrieval_cases.jsonl`. Retrieval is reported as hit rate @1, hit rate
@k, and mean reciprocal rank; the scope decision as separate in-scope and
out-of-scope recalls, because the two mistakes cost different things. Failing
cases are listed individually — a score tells you something regressed, the cases
tell you what to do about it.

The questions are written the way a customer would phrase them and deliberately
avoid reusing the corpus wording, so this measures semantic retrieval rather than
keyword overlap.

## Notebooks

[`notebooks/01_titanic_narrative.ipynb`](notebooks/01_titanic_narrative.ipynb) walks
through the dataset and what the model actually learned, with outputs committed so it
reads on GitHub without running anything.

It contains **no analysis logic**. Every computation is imported from
`src/zepto/analytics`, so the notebook cannot quietly disagree with the code that
ships. Three tests keep that true: one executes every notebook against the current
source tree, one rejects any `def` or `class` defined inside a notebook, and one
requires that notebooks import from the package.

That first test matters more than it sounds. Notebooks are imported by nothing, so a
rename breaks them without breaking a build, and the failure surfaces whenever someone
next opens the file — often months later. Running them in CI makes it a failure on the
commit that caused it.

```bash
pip install -e ".[all,dev]"
jupyter lab notebooks/
```

## Development

```bash
ruff check .              # lint
ruff format .             # format
mypy src tests            # type check
pytest                    # tests + coverage (gate: 90%)
pytest -m "not integration"   # skip notebook execution for a faster loop
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

**Target leakage is refused, not documented.** Leakage is the most dangerous defect in
a modelling project because it is silent: a leaking feature produces excellent metrics
from a model that has learned nothing. v1's defence was a comment and a `drop()` call
in a notebook cell. Here two guards run automatically, both when features are built and
again before fitting. The first refuses columns known to encode the target. The second
refuses any single feature that predicts the target at or above 0.99 accuracy —
catching leakage nobody anticipated. On this dataset the known leaking column scores
1.000 while the strongest legitimate feature reaches 0.789, a wide and comfortable
margin.

**Preprocessing cannot see the test split.** Not by convention, but structurally: the
preprocessor is a step inside a `Pipeline`, so one `fit` call trains preprocessing on
exactly the rows the estimator sees. Tests assert the learned imputer median and scaler
mean come from training data only.

**Model artifacts are versioned and carry their provenance.** v1 wrote one
`best_pipeline.joblib` and overwrote it every run — no history, no rollback, and no
record of what produced it. Since a joblib file embeds pickled scikit-learn objects,
loading one under a different library version can succeed and behave subtly
differently. Every artifact here records its training environment, and loading refuses
a version mismatch by default.

**The embedding model is loaded once, not per request.** v1 constructed a fresh
`SentenceTransformer` inside its retrieval function, so every query paid the full
model-loading cost — roughly five seconds, which is why its nine-test suite took forty
seconds. Measured here at 148ms per query. The switch to ChromaDB's bundled ONNX build
of the same `all-MiniLM-L6-v2` model also removes PyTorch, cutting ~2GB from the
install for identical 384-dimensional vectors.

**Confidence is measured, and named honestly.** v1 returned `confidence: 1.0` for
every answer, including ones assembled from unrelated text — a field that looked
meaningful and never was. Here it is cosine similarity to the best-matching passage:
0.509 for a question the corpus answers, 0.035 for one it does not. It is documented
as a similarity score rather than a calibrated probability, because calibrating it
would need labelled question-answer pairs this project does not have. Inventing that
precision would be worse than not offering it.

**The assistant declines rather than guessing.** Retrieval always returns something —
asking a vector index about football still yields the nearest policy document. v1
answered from it confidently. Below a configurable relevance floor, this declines
instead.

**Routing is by retrieval relevance, not keywords — and that change came from
measurement, not taste.** v1 classified intent by testing the question against eight
substrings, and this rebuild carried that over unchanged until the evaluation set
existed. Measured against 29 real customer questions, keyword routing scored **3.4%
recall**: it refused 28 of them, because customers ask *"How much does shipping
cost?"* rather than *"What is the delivery fee?"*. The queries used for manual
checking had all happened to contain keywords, which is exactly how the defect
survived hand-verification.

Relevance separates the same questions cleanly — in-scope 0.186–0.567, out-of-scope
0.000–0.089, no overlap — so the keyword list was deleted and the floor moved to
mid-gap. Scope accuracy went from 17.6% to 100%. The trade is that an off-topic
question now embeds before being declined (~150ms rather than instant), which is
worth answering 29 questions instead of one.

The floor is calibrated on the same 34 cases it is scored against. That is a real
limitation, noted in the code: as the set grows, recalibration belongs on a held-out
split.

**Errors do not leak internals.** An unhandled exception in v1's API returned a stack
trace to the caller. Errors now map to status codes with a safe message, with detail
going to the logs, and a test asserts the leak does not happen.

**Models are reported against a trivial baseline, not just an accuracy number.** v1
reported 80% accuracy and a deployment recommendation. It is a more useful and more
honest summary to say that a lookup table on `sex` and `pclass` scores 77.65%, that
the model scores 80.45%, and that the two agree on 92.7% of rows — so the model
contributes about +2.8 points and most of the headline comes from two base rates.
That comparison is computed automatically into every model card.

**Fairness reporting slices at intersections.** Sliced one attribute at a time, this
model looks unremarkable everywhere. Crossed, it predicts death for *every* man in
second and third class — precision 0.000, recall 0.000, accuracy equal to the base
rate. The first version of the card sliced single attributes and reported no problem,
which is exactly how this class of harm normally goes unnoticed.

## Roadmap

| Phase | Focus | Status |
| --- | --- | --- |
| 0 | Packaging, linting, type checking, CI, pre-commit | ✅ Done |
| 1 | Port `core` and `ingestion`, fixing the v1 defects | ✅ Done |
| 2 | Port `analytics`: notebook logic extracted into importable modules, leakage guard, versioned model registry | ✅ Done |
| 3 | Port `assistant`: model lifecycle, input limits, real confidence, abstain path, HTTP API | ✅ Done |
| 4 | Retrieval evaluation: labelled set, hit rate, MRR, scope-decision scoring | ✅ Done |
| 5 | Model card: intersectional disaggregation, trivial-baseline comparison | ✅ Done |
| 6 | Narrative notebook over tested modules, with a CI rot guard | ✅ Done |
| 7 | API hardening: auth, rate limiting, metrics | ✅ Done |
| 8 | RAG quality: chunking and hybrid retrieval measured and rejected; held-out eval split; prompt-injection containment | ✅ Done |
| 9 | Deployment: hardened image, dependency lockfile, image scanning in CI | ✅ Done |

Phase 7 originally listed distributed tracing. What shipped is request-id
correlation on every log line and an echoed response header, which is what a
single service needs. Spans across service boundaries would be instrumenting a
boundary that does not exist here, so the item was dropped rather than
half-satisfied and ticked.

Phase 8 was planned as reranking and a chunking strategy. Both were built and
measured, both made retrieval worse than the whole-document baseline, and
neither shipped — see `experiments/retrieval_ablation.py` for the table and
`retrieval.py` for the reasoning. The phase is marked done because the question
was answered, not because code was written.

### Not done, and known

| Gap | Why it is still open |
| --- | --- |
| Secret management is environment variables | Adequate for a single container; a real deployment wants a secret store with rotation and audit. |
| The real-LLM path has never made a live call | Fully exercised against a stub client, including retry and reply validation, but no request has ever reached Groq. Tested, not proven. |
| 4 unfixed CVEs in chromadb 1.5.9, two of them CRITICAL | 1.5.9 is the newest release; there is nothing to upgrade to. Exposure is reduced by how it is used — embedded `PersistentClient`, no chroma server, one listening socket in the container (verified via `/proc/net/tcp`) — but reduced is not fixed. `docker/README.md` has the detail. |
| 16 unfixed Debian advisories in the image | No upstream fix exists. Down from 26 after moving the base to trixie. |
| Out-of-scope recall is 85.7% on held-out data | `"Convert 500 dollars to rupees"` clears the relevance floor. Deliberately not fixed by moving the floor, because retuning on the split that caught it is what the split exists to prevent. |
| No public deployment | Runs locally and in Docker; nothing is hosted. |

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
| Leakage prevented only by a comment and a `drop()` call | Fixed — two automatic guards, one behavioural |
| Train-only preprocessing enforced by convention | Fixed — structural, via `Pipeline` composition |
| Single model artifact, overwritten, with no provenance | Fixed — versioned registry with environment metadata |
| ML logic trapped in notebooks, untestable | Fixed — importable modules; notebooks now hold narrative only, enforced by tests |
| Embedding model reloaded on every query (~5s) | Fixed — loaded once, 148ms per query |
| No input length limit on an unauthenticated endpoint | Fixed — bounded at the schema, rejected with 422 |
| Retry path that could never succeed, and was never tested | Fixed — prompt and parser agree; retry loop covered by tests |
| Hardcoded `confidence: 1.0` | Fixed — cosine similarity, 0.509 vs 0.035 measured |
| Answered confidently from irrelevant matches | Fixed — abstains below a relevance floor |
| Stack traces returned to API callers | Fixed — safe messages, detail to logs |
| Corpus edits silently never re-indexed | Fixed — upsert by stable id |
| UTF-8 BOM leaking into served answers | Fixed — stripped at the source and on read |
| Keyword routing refused 28 of 29 real questions | Fixed — relevance routing, 3.4% → 100% recall |
| Unpinned dependencies, container runs as root | Pending (Phase 7) |

## License

MIT
