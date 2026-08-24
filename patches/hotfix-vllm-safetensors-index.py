#!/usr/bin/env python3
"""Make vLLM honor the safetensors index when shards contain stale keys.

Hugging Face's ``model.safetensors.index.json`` is authoritative for the shard
that owns each tensor.  The pinned vLLM iterators glob every shard and yield
every physical key, which is normally equivalent.  Some converted checkpoints
retain stale duplicate tensors in an earlier shard, however, so ignoring the
index can load the wrong payload (and even the wrong tensor shape).
"""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text()
    if new in text:
        print(f"[skip] {label}")
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new, 1))
    print(f"[ok]   {label}")


def main(root: Path) -> None:
    weight_utils = root / "model_executor/model_loader/weight_utils.py"

    replace_once(
        weight_utils,
        "        import instanttensor\n",
        "        import instanttensor\n"
        "        from instanttensor._impl import read_safetensors_metadata\n",
        "import InstantTensor metadata reader",
    )

    replace_once(
        weight_utils,
        "def safetensors_weights_iterator(\n",
        "def _safetensors_index_weight_map(\n"
        "    hf_weights_files: list[str],\n"
        ") -> dict[str, str] | None:\n"
        "    if not hf_weights_files:\n"
        "        return None\n"
        "    parents = {Path(filename).parent for filename in hf_weights_files}\n"
        "    if len(parents) != 1:\n"
        "        return None\n"
        "    index_path = next(iter(parents)) / SAFE_WEIGHTS_INDEX_NAME\n"
        "    if not index_path.is_file():\n"
        "        return None\n"
        "    try:\n"
        "        index = json.loads(index_path.read_text())\n"
        "        weight_map = index[\"weight_map\"]\n"
        "    except (OSError, TypeError, ValueError, KeyError):\n"
        "        logger.warning_once(\n"
        "            \"Could not read safetensors index %s; falling back to shard keys\",\n"
        "            index_path,\n"
        "            exc_info=True,\n"
        "        )\n"
        "        return None\n"
        "    if not isinstance(weight_map, dict) or not all(\n"
        "        isinstance(name, str) and isinstance(filename, str)\n"
        "        for name, filename in weight_map.items()\n"
        "    ):\n"
        "        logger.warning_once(\n"
        "            \"Safetensors index %s has an invalid weight_map; \"\n"
        "            \"falling back to shard keys\",\n"
        "            index_path,\n"
        "        )\n"
        "        return None\n"
        "    return weight_map\n\n\n"
        "def _is_indexed_tensor(\n"
        "    weight_map: dict[str, str] | None, name: str, st_file: str\n"
        ") -> bool:\n"
        "    return weight_map is None or weight_map.get(name) == Path(st_file).name\n\n\n"
        "def safetensors_weights_iterator(\n",
        "add safetensors index helpers",
    )

    replace_once(
        weight_utils,
        "    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)\n\n"
        "    fs_type = _get_fs_type(sorted_files)\n",
        "    sorted_files = sorted(hf_weights_files, key=_natural_sort_key)\n"
        "    indexed_weight_map = _safetensors_index_weight_map(sorted_files)\n\n"
        "    fs_type = _get_fs_type(sorted_files)\n",
        "load index for ordinary safetensors iterator",
    )

    replace_once(
        weight_utils,
        "            for name, param in state_dict.items():\n"
        "                if not should_skip_weight(name, local_expert_ids):\n"
        "                    yield name, param\n",
        "            for name, param in state_dict.items():\n"
        "                if not _is_indexed_tensor(indexed_weight_map, name, st_file):\n"
        "                    continue\n"
        "                if not should_skip_weight(name, local_expert_ids):\n"
        "                    yield name, param\n",
        "filter eager safetensors keys through index",
    )

    replace_once(
        weight_utils,
        "                for name in f.keys():  # noqa: SIM118\n"
        "                    if should_skip_weight(name, local_expert_ids):\n"
        "                        continue\n"
        "                    state_dict[name] = f.get_tensor(name)\n",
        "                for name in f.keys():  # noqa: SIM118\n"
        "                    if not _is_indexed_tensor(\n"
        "                        indexed_weight_map, name, st_file\n"
        "                    ):\n"
        "                        continue\n"
        "                    if should_skip_weight(name, local_expert_ids):\n"
        "                        continue\n"
        "                    state_dict[name] = f.get_tensor(name)\n",
        "filter TorchAO safetensors keys through index",
    )

    replace_once(
        weight_utils,
        "                for name in f.keys():  # noqa: SIM118\n"
        "                    if should_skip_weight(name, local_expert_ids):\n"
        "                        continue\n"
        "                    param = f.get_tensor(name)\n"
        "                    yield name, param\n\n\n"
        "def multi_thread_safetensors_weights_iterator(\n",
        "                for name in f.keys():  # noqa: SIM118\n"
        "                    if not _is_indexed_tensor(\n"
        "                        indexed_weight_map, name, st_file\n"
        "                    ):\n"
        "                        continue\n"
        "                    if should_skip_weight(name, local_expert_ids):\n"
        "                        continue\n"
        "                    param = f.get_tensor(name)\n"
        "                    yield name, param\n\n\n"
        "def multi_thread_safetensors_weights_iterator(\n",
        "filter ordinary safetensors keys through index",
    )

    instant_old = '''    device = current_platform.current_device()

    with instanttensor.safe_open(
        hf_weights_files, framework="pt", device=device, process_group=process_group
    ) as f:
        yield from tqdm(
            f.tensors(),
            desc="Loading safetensors using InstantTensor loader",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
            position=tqdm._get_free_pos(),
            total=len(f.keys()),
            mininterval=1.0,
        )
'''
    instant_new = '''    device = current_platform.current_device()
    indexed_weight_map = _safetensors_index_weight_map(hf_weights_files)

    conflicting_shards = False
    if indexed_weight_map is not None:
        # InstantTensor's combined-file iterator intentionally exposes physical
        # shard contents and does not consult the HF index. Read only the small
        # headers here to decide whether the fast combined path is unambiguous.
        for st_file in hf_weights_files:
            _, tensor_metadata, _ = read_safetensors_metadata(st_file)
            if any(
                not _is_indexed_tensor(indexed_weight_map, name, st_file)
                for name in tensor_metadata
            ):
                conflicting_shards = True
                break

    if not conflicting_shards:
        with instanttensor.safe_open(
            hf_weights_files,
            framework="pt",
            device=device,
            process_group=process_group,
        ) as f:
            yield from tqdm(
                f.tensors(),
                desc="Loading safetensors using InstantTensor loader",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
                position=tqdm._get_free_pos(),
                total=len(f.keys()),
                mininterval=1.0,
            )
        return

    logger.warning_once(
        "Checkpoint shards contain tensor keys owned by a different file in "
        "the safetensors index; loading shards separately and honoring the index"
    )

    def indexed_tensors() -> Generator[tuple[str, torch.Tensor], None, None]:
        for st_file in sorted(hf_weights_files, key=_natural_sort_key):
            with instanttensor.safe_open(
                st_file,
                framework="pt",
                device=device,
                process_group=process_group,
            ) as f:
                for name, tensor in f.tensors():
                    if _is_indexed_tensor(indexed_weight_map, name, st_file):
                        yield name, tensor

    yield from tqdm(
        indexed_tensors(),
        desc="Loading indexed safetensors using InstantTensor loader",
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
        position=tqdm._get_free_pos(),
        total=len(indexed_weight_map),
        mininterval=1.0,
    )
'''
    replace_once(
        weight_utils,
        instant_old,
        instant_new,
        "honor safetensors index in InstantTensor iterator",
    )

    compile(weight_utils.read_text(), str(weight_utils), "exec")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
