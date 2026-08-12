#!/usr/bin/env python3
"""Keep InstantTensor's DLPack allocation alive independently of its loader."""

from __future__ import annotations

import sys
from pathlib import Path


def main(path: Path) -> None:
    text = path.read_text()
    lifetime_old = "tensor = tensor_int8.view(torch_dtype).view(*shape)"
    lifetime_new = "tensor = tensor_int8.view(torch_dtype).view(tuple(shape)).clone()"
    if lifetime_new not in text:
        if text.count(lifetime_old) != 1:
            raise RuntimeError(
                "InstantTensor lifetime patch: expected exactly one source anchor, "
                f"found {text.count(lifetime_old)}"
            )
        text = text.replace(lifetime_old, lifetime_new, 1)
        print("[ok]   InstantTensor independent tensor lifetime")
    else:
        print("[skip] InstantTensor independent tensor lifetime")

    # torch 2.11's CUDA caching allocator retains the short-lived clones made
    # above. On a unified-memory GB10 that can make the load peak approach two
    # copies of the model even though the loader itself only owns a 1 GiB ring.
    # Check infrequently, and trim only under real pressure with at least 1 GiB
    # of reclaimable cached blocks. The live model tensors are never released.
    trim_old = "            yield name, tensor\n\n    def get_tensor("
    trim_new = (
        "            yield name, tensor\n"
        "            if (tensor_index & 255) == 255:\n"
        "                free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)\n"
        "                allocated_bytes = torch.cuda.memory_allocated(self.device)\n"
        "                reserved_bytes = torch.cuda.memory_reserved(self.device)\n"
        "                cached_bytes = reserved_bytes - allocated_bytes\n"
        "                if (\n"
        "                    free_bytes * 5 < total_bytes * 2\n"
        "                    and cached_bytes > 1 << 30\n"
        "                ):\n"
        "                    torch.cuda.empty_cache()\n\n"
        "    def get_tensor("
    )
    if trim_new not in text:
        if text.count(trim_old) != 1:
            raise RuntimeError(
                "InstantTensor cache-trim patch: expected exactly one source anchor, "
                f"found {text.count(trim_old)}"
            )
        text = text.replace(trim_old, trim_new, 1)
        print("[ok]   InstantTensor pressure-triggered allocator trim")
    else:
        print("[skip] InstantTensor pressure-triggered allocator trim")

    path.write_text(text)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} INSTANTTENSOR_IMPL")
    main(Path(sys.argv[1]))
