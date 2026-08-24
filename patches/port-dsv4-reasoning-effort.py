#!/usr/bin/env python3
"""Preserve explicit low/high/max reasoning effort in the pinned tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path


DEFAULT_TARGET = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/tokenizers/deepseek_v4.py"
)

OLD = '''elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            else:
                reasoning_effort = "high"'''

NEW = '''elif reasoning_effort in ("max", "xhigh"):
                reasoning_effort = "max"
            elif reasoning_effort == "high":
                reasoning_effort = "high"
            else:
                reasoning_effort = "low"'''


def main(path: Path) -> None:
    source = path.read_text()
    if NEW in source:
        print(f"[skip] reasoning-effort mapping already applied: {path}")
        return
    count = source.count(OLD)
    if count != 1:
        raise RuntimeError(f"reasoning-effort mapping: expected one anchor, found {count}")
    path.write_text(source.replace(OLD, NEW, 1))
    compile(path.read_text(), str(path), "exec")
    print(f"[ok] reasoning-effort mapping applied: {path}")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET
    main(target)
