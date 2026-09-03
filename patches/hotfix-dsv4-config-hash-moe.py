#!/usr/bin/env python3
"""Normalize GPTQModel's redundant DeepSeek-V4 ``hash_moe`` layer list.

The pinned Transformers validator does not include ``hash_moe`` in its generic
layer-type enum. The NVIDIA DeepSeek-V4 implementation does not consume
``mlp_layer_types``; it derives the same boundary from ``num_hash_layers``.
Only that exact redundant encoding is removed. Any disagreement fails closed.
"""

from __future__ import annotations

import sys
from pathlib import Path


DEFAULT_PATH = Path(
    "/usr/local/lib/python3.12/dist-packages/vllm/transformers_utils/"
    "configs/deepseek_v4.py"
)
MARK = "# [dsv4-hash-moe-config] validated redundant GPTQModel field"
ANCHOR = """        self.rope_parameters = rope_scaling or rope_parameters
        super().__init__(**kwargs)
"""
REPLACEMENT = f"""        self.rope_parameters = rope_scaling or rope_parameters
        mlp_layer_types = kwargs.get("mlp_layer_types")
        if isinstance(mlp_layer_types, list) and "hash_moe" in mlp_layer_types:
            num_hash_layers = int(kwargs.get("num_hash_layers", 0))
            expected = ["hash_moe"] * num_hash_layers + ["moe"] * (
                len(mlp_layer_types) - num_hash_layers
            )
            if mlp_layer_types != expected:
                raise ValueError(
                    "mlp_layer_types disagrees with num_hash_layers: "
                    f"num_hash_layers={{num_hash_layers}}, values={{mlp_layer_types!r}}"
                )
            kwargs.pop("mlp_layer_types")  {MARK}
        super().__init__(**kwargs)
"""


def patch_text(source: str) -> tuple[str, str]:
    if MARK in source:
        return source, "skipped"
    count = source.count(ANCHOR)
    if count != 1:
        return source, f"drift:anchor={count}"
    updated = source.replace(ANCHOR, REPLACEMENT, 1)
    compile(updated, "deepseek_v4.py", "exec")
    return updated, "applied"


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    if not path.is_file():
        print(f"FATAL: missing {path}", file=sys.stderr)
        return 1
    original = path.read_text()
    updated, status = patch_text(original)
    if status == "applied":
        path.write_text(updated)
    elif status != "skipped":
        print(f"FATAL: {path} {status}", file=sys.stderr)
        return 1
    print(f"[dsv4-hash-moe-config] {status}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
