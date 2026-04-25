"""Smoke-test driver for the examples directory.

Runs every ``examples/*.py`` (excluding ``_lib.py``, ``run_all.py``, and
anything starting with ``_``) in a fresh subprocess, capturing exit code
and elapsed time. Network-touching examples opt in via
``REQUIRES_NETWORK = True`` at module top and are skipped unless the
runner sets ``RUN_NETWORK=1``.

Invoked by ``just examples-smoke`` and the ``examples`` CI job.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent
TIMEOUT_S = 60


def _discover() -> list[Path]:
    return sorted(
        p
        for p in EXAMPLES_DIR.glob("*.py")
        if p.name != "run_all.py" and not p.name.startswith("_")
    )


def _has_network_marker(path: Path) -> bool:
    """True if the example declares ``REQUIRES_NETWORK = True`` at module top."""
    try:
        content = path.read_text()
    except OSError:
        return False
    # Substring scan rather than ast walk — examples are short and the
    # marker is unambiguous when present at the top level.
    return "REQUIRES_NETWORK = True" in content


def _run_one(example: Path, env: dict[str, str]) -> tuple[bool, float, str]:
    """Run *example* in a subprocess. Returns (ok, elapsed_s, status).

    Invoked as ``python -m examples.<modname>`` so ``from examples._lib
    import ...`` resolves; running the file path directly would not add
    the repo root to ``sys.path``.
    """
    start = time.monotonic()
    module_name = f"examples.{example.stem}"
    try:
        completed = subprocess.run(  # noqa: S603 — fixed argv, no shell, internal use
            [sys.executable, "-m", module_name],
            check=False,
            timeout=TIMEOUT_S,
            cwd=EXAMPLES_DIR.parent,
            env=env,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return False, time.monotonic() - start, f"TIMEOUT after {TIMEOUT_S}s"

    elapsed = time.monotonic() - start
    if completed.returncode != 0:
        # Stitch stdout + stderr so failures surface in CI logs without
        # the runner having to re-run them locally.
        tail = (completed.stdout or "") + (completed.stderr or "")
        return False, elapsed, f"exit={completed.returncode}\n{tail.rstrip()}"
    return True, elapsed, "ok"


def main() -> int:
    network = os.environ.get("RUN_NETWORK") == "1"
    env = dict(os.environ)
    failures: list[tuple[str, str]] = []
    skipped: list[str] = []

    examples = _discover()
    sys.stdout.write(f"Discovered {len(examples)} example(s). network={network}\n\n")

    for example in examples:
        rel = example.relative_to(EXAMPLES_DIR.parent)
        if _has_network_marker(example) and not network:
            sys.stdout.write(f"SKIP   {rel}\n")
            skipped.append(str(rel))
            continue
        sys.stdout.write(f"RUN    {rel}\n")
        ok, elapsed, status = _run_one(example, env)
        if ok:
            sys.stdout.write(f"  ok   ({elapsed:.2f}s)\n")
        else:
            sys.stdout.write(f"  FAIL ({elapsed:.2f}s) — {status}\n")
            failures.append((str(rel), status))

    sys.stdout.write("\n")
    sys.stdout.write(f"ran     {len(examples) - len(skipped)}\n")
    sys.stdout.write(f"skipped {len(skipped)}\n")
    sys.stdout.write(f"failed  {len(failures)}\n")

    if failures:
        sys.stdout.write("\nFailures:\n")
        for name, status in failures:
            sys.stdout.write(f"  - {name}\n    {status}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
