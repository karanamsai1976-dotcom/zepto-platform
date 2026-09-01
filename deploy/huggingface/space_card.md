---
title: Zepto Support Assistant
emoji: 🛒
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Retrieval-augmented answers over a Zepto policy corpus
---

# Zepto Support Assistant

A retrieval-augmented support assistant. Ask a question about Zepto's delivery,
returns, membership, tracking, cancellation, gift card, or support policies, and
it quotes the matching policy — or declines, when the corpus does not cover the
question.

Open the Space to use the page, or call the API directly:

```bash
curl -X POST https://<space-host>/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"How much is the delivery fee?"}'
```

Also exposes `/health`, `/ready`, `/metrics` (Prometheus), and `/docs`.

## What it is doing

Questions are embedded with `all-MiniLM-L6-v2` (ONNX, via ChromaDB's bundled
runtime — no PyTorch) and matched against eight policy documents by cosine
similarity. Whether a question is in scope is decided by that same similarity
rather than by keyword matching, which is what makes it answer questions phrased
the way people actually phrase them.

Answers are assembled deterministically from the retrieved policy text. There is
no language model in the default path, so the assistant cannot invent a policy
that does not exist.

`confidence` is retrieval similarity on a 0–1 scale. It says how closely the
question resembled the matched policy. It is **not** a calibrated probability
that the answer is correct, and it is labelled that way rather than dressed up.

## Measured quality

Against a labelled 77-case set, split into cases used for tuning (`dev`) and
cases held back from every tuning decision (`test`):

| Metric | dev (34) | test (43, held out) |
| --- | --- | --- |
| Retrieval hit rate @1 | 86.2% | 88.9% |
| Retrieval hit rate @3 | 96.6% | 100% |
| Mean reciprocal rank | 0.914 | 0.940 |
| Out-of-scope questions declined | 100% | 85.7% |

The held-out column is the honest one. Retrieval generalises; the scope decision
does not, quite — `"Convert 500 dollars to rupees"` slips past the relevance
floor because currency wording sits close to the delivery-fee document. It is
left standing rather than tuned away, since retuning against the split that
caught it is exactly what the split exists to prevent.

## Source

[github.com/karanamsai1976-dotcom/zepto-platform](https://github.com/karanamsai1976-dotcom/zepto-platform)

The container image is built from the same Dockerfile CI builds, runs, queries,
and vulnerability-scans on every change.
