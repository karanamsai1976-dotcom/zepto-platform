# Container image

Packages the support assistant as a service image. Everything below was run
against the built image; nothing here is asserted from the Dockerfile alone.

## Build and run

```bash
docker build -f docker/Dockerfile -t zepto-assistant:0.1.0 .
docker run -d --name zepto -p 8000:8000 zepto-assistant:0.1.0
```

Then:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"How much is the delivery fee?"}'
```

## What the image does differently from v1's

**The base is pinned by digest**, not by the `python:3.12-slim` tag. A tag is a
moving pointer: the same Dockerfile built twice can produce two different images
with different vulnerabilities. The digest is the image.

**Dependencies come from a resolved lock** (`requirements-assistant.lock`), not
from the `>=` floors in `pyproject.toml`. The floors say what the code needs;
the lock says what was actually built and tested. The lock is generated inside
the target container, because a Windows freeze of the same extra differs — it
carries `colorama` and omits `uvloop` — and shipping that would pin packages the
image cannot use while missing one it does.

**Build tooling is discarded.** `pip`, the wheel build, and the source tree live
in a stage that does not reach the runtime image.

**It runs as uid 10001, not root.** Verified:

```
$ docker exec zepto id
uid=10001(zepto) gid=10001(zepto) groups=10001(zepto)
```

**The index is built at image build time**, not on first request. This bakes in
the ~79MB ONNX embedding model that chromadb otherwise downloads on first use,
so a container starts without reaching the network and the first user request
does not pay for indexing. The build log shows `indexed 8 documents`, and a
fresh container reports `documents=8` at startup rather than bootstrapping.

**There is a HEALTHCHECK**, against `/ready` rather than `/health` — the
question an orchestrator is asking is not "is the process alive" but "can this
instance answer", and an instance with an empty index is alive and useless. It
is written in Python because the slim base has no `curl`, and adding a package
in order to run a health check is a poor trade.

## Verified behaviour

Against `zepto-assistant:0.1.0`, container healthy (`Up 12 seconds (healthy)`):

| Check | Result |
| --- | --- |
| `GET /health` | `{"status":"ok","mock_llm":true,"documents_indexed":8}` |
| `POST /ask` (in scope) | 200, `intent=policy_question`, top source `doc_01` at relevance 0.5047 |
| `POST /ask` (out of scope) | 200, `intent=out_of_scope`, confidence 0.0538, no sources |
| `X-Request-ID` on response | present |
| `GET /metrics` | `zepto_documents_indexed 8.0`, per-route request counters, answers split by intent |

With `ZEPTO_ASSISTANT_REQUIRE_API_KEY=true`,
`ZEPTO_ASSISTANT_API_KEYS=demo-key-one,demo-key-two`,
`ZEPTO_ASSISTANT_RATE_LIMIT_REQUESTS=2`:

| Check | Result |
| --- | --- |
| `POST /ask` with no key | 401, `WWW-Authenticate: X-API-Key` |
| `POST /ask` with a valid key | 200 (twice) |
| Third request in the window | 429, `Retry-After: 60` |
| `GET /health` | 200 — unguarded, as intended |
| `REQUIRE_API_KEY=true` with no keys | container exits: `ConfigurationError: authentication is required but no API keys are configured` |

That last row is the point of failing closed. The alternative is a service that
starts, reports healthy, and serves an endpoint it believes is protected.

## Vulnerability scanning

CI builds the image, runs it, checks it answers, then scans it with Trivy and
fails on **fixable** HIGH/CRITICAL findings.

Measured locally (Trivy, `--severity HIGH,CRITICAL`):

| Base | Total | Fixable |
| --- | --- | --- |
| Debian bookworm (previous) | 38 | 0 |
| Debian trixie (current) | **20** | 0 |

Moving to trixie removed 18 advisories and cost nothing measurable: the image is
the same 1.17GB, the container is healthy on the same `/ready` probe, and it
returns the same answers.

The dependency set is unchanged, and that was checked rather than assumed —
resolving `.[assistant]` inside both bases on the same day produces byte-identical
output, all 99 packages at the same versions. A fresh resolution did differ from
the *committed* lock in two patch versions (`filelock`, `protobuf`), but that is
time passing, not the base changing, which is why the committed lock was left
alone. A base swap and a dependency bump are two changes and belong in two
commits.

CI gates on **fixable** findings only. No change in this repository can clear an
advisory with no upstream fix, so gating on those produces a permanently red
build, and a permanently red build trains people to stop reading it. A second,
non-blocking step prints the unfixed ones on every run, because "we ignore
unfixed" becomes "we never look" without something that puts them in front of
you.

### The four that are not the base OS

Sixteen of the twenty are Debian packages — `perl-base`, `libsqlite3-0`,
`ncurses`, `gzip`, `libacl1`. The other four are **chromadb 1.5.9 itself**, and
they deserve to be read rather than skimmed:

| CVE | Severity | Title |
| --- | --- | --- |
| CVE-2026-45829 | CRITICAL | Arbitrary code execution via pre-authentication code injection |
| CVE-2026-45833 | CRITICAL | Arbitrary code execution via code injection |
| CVE-2026-45830 | HIGH | Unauthorized data manipulation due to improper authorization validation |
| CVE-2026-45831 | HIGH | Unauthorized cross-tenant actions due to improper authorization checks |

1.5.9 is the latest release on PyPI, so there is no version to upgrade to. What
reduces the exposure here is how chromadb is used, and that is verifiable rather
than asserted:

- It runs as an **embedded library** via `PersistentClient`, not as a server.
  Two of the four advisories describe authorization and cross-tenant checks,
  which are server-mode concepts this deployment does not use.
- The container has exactly **one listening socket**, confirmed by reading
  `/proc/net/tcp` inside a running container: port 8000, the FastAPI app. There
  is no chroma endpoint reachable by anyone.
- Nothing untrusted reaches chromadb's control surface. Queries arrive as text
  through a validated Pydantic model and are used as query text only.

That is a reduction in exposure, not a fix, and the pre-authentication code
injection advisory is not one to be relaxed about. The action is to watch for a
patched release and upgrade on the day it lands. If this were internet-facing
and holding real data, the honest answer would be to reconsider the dependency
rather than reason around it.

## Known limitations

**The image is 1.17GB.** Almost all of that is `onnxruntime` and chromadb's
dependency tree, plus the baked model and index. It is honest for what it does —
it ships a real embedding model — but it is not small. Reducing it would mean
dropping chromadb for a lighter vector store, which is a larger change than this
phase.

**The filesystem is not read-only.** chromadb opens its SQLite file read-write
even for pure queries, so `--read-only` does not work without a writable mount
for `/app/data/chroma`.

## Regenerating the lock

After changing the `assistant` extra in `pyproject.toml`:

```bash
docker run --rm -v "$PWD:/src:ro" python:3.12-slim-trixie sh -c \
  'set -e; mkdir -p /tmp/pkg && cp /src/pyproject.toml /src/README.md /tmp/pkg/ \
   && cp -r /src/src /tmp/pkg/src && python -m venv /v \
   && /v/bin/pip install -q --upgrade pip && /v/bin/pip install -q "/tmp/pkg[assistant]" \
   && /v/bin/pip freeze --exclude-editable | grep -v "^zepto-platform"' \
  > docker/requirements-assistant.lock
```

Then restore the header comment at the top of that file.
