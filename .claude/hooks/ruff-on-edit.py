#!/usr/bin/env python3
"""PostToolUse hook: run ruff fix + format on edited Python files.

Mirrors what `.pre-commit-config.yaml` runs at commit time, so Claude's edits
land already-clean and don't trip the pre-commit hook on the next commit.
Failures are non-blocking — ruff prints diagnostics to stderr, the hook
returns 0 so the agent can react to whatever ruff couldn't auto-fix.
"""

from __future__ import annotations

import json
import subprocess
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0

    if data.get("tool_name") not in ("Edit", "Write", "MultiEdit"):
        return 0

    path = (data.get("tool_input") or {}).get("file_path")
    if not isinstance(path, str) or not path.endswith(".py"):
        return 0

    # S603/S607 — `uv` is a trusted dev dependency on $PATH; the only argument
    # we forward is `path` from a Claude Code tool input (not user-supplied
    # shell text), so a partial executable path + no shell is the right call.
    subprocess.run(  # noqa: S603
        ["uv", "run", "ruff", "check", "--fix", "--force-exclude", path],  # noqa: S607
        check=False,
    )
    subprocess.run(  # noqa: S603
        ["uv", "run", "ruff", "format", "--force-exclude", path],  # noqa: S607
        check=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
