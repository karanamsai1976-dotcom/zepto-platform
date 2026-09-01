# Deploying to Hugging Face Spaces

The Space runs the same container image CI builds, runs, queries, and scans on
every change — not a separate deployment build that could quietly diverge from
what the tests cover.

## Why a generated tree

Spaces requires `Dockerfile` and a front-matter `README.md` at the repository
root. This project keeps its Dockerfile in `docker/`, and its README is a project
README rather than a Space card.

The tempting fix is a second Dockerfile shaped the way Spaces wants. That is the
one thing worth avoiding: it would drift from the Dockerfile CI actually
exercises, and the drift would stay invisible until a deployment behaved
differently from every test. So the deployable tree is generated, and the
generated Dockerfile is the repository's own copied verbatim with one clearly
marked block appended. The difference between what is tested and what is deployed
is exactly that block.

## Build the tree

```bash
python deploy/huggingface/build_space.py
```

Writes `build/space/` — gitignored, since it is an artifact rather than source.

## What the appended block sets, and why

| Setting | Value | Reason |
| --- | --- | --- |
| `ZEPTO_ASSISTANT_TRUSTED_PROXY_COUNT` | `1` | Spaces terminates TLS and proxies to the container, so every request arrives from the proxy's address. Without this, all callers share one rate-limit bucket and one client can exhaust everyone's quota. |
| `ZEPTO_ASSISTANT_RATE_LIMIT_REQUESTS` | `20` | Tighter than the local default of 60. The endpoint is public and unauthenticated, so the quota is what stands between a bored visitor and the whole CPU budget. |
| `ZEPTO_LOG_JSON` | `true` | Structured logs are what the platform's log viewer can filter on. |

Authentication is left at its default of off. A public demo behind a credential
nobody has is not a demo, and the exposure is bounded: the service reads a fixed
corpus, writes nothing a caller can reach, and holds no user data.

## Verified locally before pushing

The generated tree was built and run as a container:

| Check | Result |
| --- | --- |
| Container health | `Up 15 seconds (healthy)` |
| `GET /` | 200, `text/html; charset=utf-8`, 5,940 bytes |
| `POST /ask` — "Do you have a helpline I can dial?" | Answers from `doc_08`, the support policy |
| `X-Forwarded-For` handling | Accepted, request served |
| Logs | JSON, one object per request with `request_id` |

## Push it

Create the Space first at <https://huggingface.co/new-space> — **SDK: Docker**,
blank template. Then:

```bash
cd build/space
git init -b main
git remote add space https://huggingface.co/spaces/<user>/<space-name>
git add -A
git commit -m "Deploy Zepto support assistant"
git push --force space main
```

Authenticate with a **write** token from
<https://huggingface.co/settings/tokens>, used as the password when git prompts.

`--force` is correct here and not a shortcut: the tree is regenerated from
scratch each time, so it has no shared history with what is already on the Space.

## What to expect on the first build

Spaces builds the image itself, so the first deploy takes several minutes —
about 100 dependencies plus the ~79MB ONNX embedding model, which is downloaded
during the build and baked in so the running container never needs it.

Watch the **Logs** tab. A successful start ends with:

```
{"event": "assistant_ready", "documents": 8, "mock_llm": true, ...}
```

`documents: 8` is the thing to check. Anything else means the index did not come
through the build.

## Not yet verified

The container is verified; the **hosting** is not, and will not be until it is
actually deployed. Two things specifically:

- **`TRUSTED_PROXY_COUNT=1` assumes exactly one proxy in front.** That matches
  how Spaces is documented to work, but it has not been confirmed against a live
  Space. If `X-Forwarded-For` arrives with more entries, the rate limit would key
  on a proxy address rather than the caller — everyone sharing a bucket again.
  Check a request log after deploying and adjust the count to match what is
  really there.
- **The free tier sleeps when idle.** The first request after a sleep pays the
  container start, which includes loading the ONNX model into memory. Roughly a
  few seconds, not the multi-second-per-query cost v1 had, because the model is
  loaded once at startup rather than per request.
