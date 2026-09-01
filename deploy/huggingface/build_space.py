"""Assemble a Hugging Face Space directory from this repository.

Spaces requires a `Dockerfile` and a `README.md` carrying YAML front matter at
the repository root, which does not match this project's layout. Rather than
keeping a second Dockerfile in the shape Spaces wants -- which would drift from
the one CI actually builds and scans, and the drift would be invisible until a
deployment behaved differently from every test -- the deployable tree is
generated from the real one.

The generated Dockerfile is the repository's own, copied verbatim, with a
clearly marked block of deployment settings appended. Appending rather than
editing keeps the diff between what is tested and what is deployed to exactly
that block, so it can be read in one glance.

Run:  python deploy/huggingface/build_space.py [--output DIR]

Then push the output directory to the Space's git remote. It is a build
artifact, not source, so it is gitignored here.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CARD = Path(__file__).resolve().parent / "space_card.md"
DEFAULT_OUTPUT = REPO_ROOT / "build" / "space"

#: Copied verbatim into the Space, preserving relative paths so the generated
#: Dockerfile needs no rewriting.
CONTENTS = (
    "pyproject.toml",
    "src",
    "data/corpus",
    "docker/requirements-assistant.lock",
)

IGNORED = shutil.ignore_patterns("__pycache__", "*.py[cod]", ".ipynb_checkpoints")

DEPLOYMENT_SETTINGS = """

# ---------------------------------------------------------------------------
# Appended by deploy/huggingface/build_space.py. Everything above is the
# repository's Dockerfile verbatim; everything below is deployment
# configuration for this specific host. Do not edit here -- edit the generator.
# ---------------------------------------------------------------------------

# Spaces terminates TLS and proxies to the container, so every request arrives
# from the proxy's address. Without this, all callers share a single rate-limit
# bucket and one client can exhaust everyone's quota. Declaring the hop count
# makes X-Forwarded-For trustworthy for exactly that many entries; the value
# must match the real topology, since a wrong number is a rate limit that either
# does not apply or applies to the wrong party.
ENV ZEPTO_ASSISTANT_TRUSTED_PROXY_COUNT=1

# Tighter than the local default of 60. The endpoint is public and
# unauthenticated, so the quota is the only thing standing between a bored
# visitor and the service's whole CPU budget.
ENV ZEPTO_ASSISTANT_RATE_LIMIT_REQUESTS=20
ENV ZEPTO_ASSISTANT_RATE_LIMIT_WINDOW_SECONDS=60

# Deliberately no API key, so authentication is left at its default of off. A
# public demo behind a credential nobody has is not a demo. The exposure that
# buys is bounded: the service reads a fixed corpus, writes nothing a caller can
# reach, and holds no user data. Setting it to false explicitly would only
# restate the default and trip the SecretsUsedInArgOrEnv build warning, and a
# deploy artifact that ships a warning teaches people to skim warnings.

# Structured JSON logs, which is what the platform's log viewer can actually
# filter on.
ENV ZEPTO_LOG_JSON=true
"""


def build(output: Path) -> None:
    """Write a complete, pushable Space directory."""
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for relative in CONTENTS:
        source = REPO_ROOT / relative
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            # Bytecode is host-specific and would otherwise be shipped from a
            # developer's machine into a Linux image. The repository .dockerignore
            # already excludes it from a local build; this tree is assembled by
            # hand and gets no such protection.
            shutil.copytree(source, destination, ignore=IGNORED)
        else:
            shutil.copy2(source, destination)

    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    (output / "Dockerfile").write_text(dockerfile + DEPLOYMENT_SETTINGS, encoding="utf-8")

    # The Space card must be README.md at the root, because that is where Spaces
    # reads the front matter from. It doubles as the package readme inside the
    # image, since pyproject points `readme` at that path -- harmless, and
    # noted so the coincidence is not mistaken for a bug later.
    (output / "README.md").write_text(CARD.read_text(encoding="utf-8"), encoding="utf-8")

    (output / ".gitattributes").write_text("* text=auto eol=lf\n", encoding="utf-8")

    print(f"Space assembled at {output}")
    print("\nContents:")
    for path in sorted(output.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(output).as_posix()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output)


if __name__ == "__main__":
    main()
