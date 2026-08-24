#!/usr/bin/env python3
"""Teach the imported EXL3 backend about standard-HF mixed K2/K3 experts."""

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
    exl3 = root / "model_executor/layers/quantization/exl3.py"

    replace_once(
        exl3,
        "        self.rank_sliced_bits_by_layer: dict[int, tuple[int, ...]] = {}\n"
        "        self.standard_fused_moe = False",
        "        self.rank_sliced_bits_by_layer: dict[int, tuple[int, ...]] = {}\n"
        "        self.standard_bits_by_layer: dict[int, tuple[int, ...]] = {}\n"
        "        self.standard_fused_moe = False",
        "store standard mixed-bitrate maps",
    )
    replace_once(
        exl3,
        '        """Use b12x for a standard, unsliced, uniform DeepSeek EXL3 MoE."""\n\n'
        "        if hf_config is None or getattr(hf_config, \"model_type\", None) != \"deepseek_v4\":\n"
        "            return\n"
        "        bits = float(self.bits) if self.bits is not None else 0.0\n"
        "        if bits != int(bits) or int(bits) not in (2, 3, 4, 5, 6):\n"
        "            return\n"
        '        if self.codebook != "mcg" or not self.tensor_storage:\n',
        '        """Use b12x for standard, unsliced DeepSeek EXL3 experts."""\n\n'
        "        if hf_config is None or getattr(hf_config, \"model_type\", None) != \"deepseek_v4\":\n"
        "            return\n"
        '        if self.codebook != "mcg" or not self.tensor_storage:\n',
        "allow a non-integral checkpoint-average bitrate",
    )
    replace_once(
        exl3,
        "        if not all(standard_expert.fullmatch(name) for name in self.tensor_storage):\n"
        "            return\n"
        "        self.standard_fused_moe = True\n"
        "        # GPTQModel emits standard Transformers expert names while NVIDIA's\n"
        "        # DeepSeek-V4 implementation uses the native checkpoint spelling. Keep\n"
        "        # both in the metadata map: source-file lookup still needs the former,\n"
        "        # and FusedMoE construction/codebook validation needs the latter.\n"
        "        num_hidden_layers = int(getattr(hf_config, \"num_hidden_layers\", 0))\n",
        "        if not all(standard_expert.fullmatch(name) for name in self.tensor_storage):\n"
        "            return\n"
        "        # The top-level bits field is the realized checkpoint average. Mixed\n"
        "        # checkpoints keep the executable integer bitrate on every tensor.\n"
        "        # Require all three projections in an expert family to agree so a\n"
        "        # malformed artifact cannot be silently assigned to the wrong tier.\n"
        "        num_hidden_layers = int(getattr(hf_config, \"num_hidden_layers\", 0))\n"
        "        families: dict[tuple[int, int], dict[str, int]] = {}\n"
        "        family_pattern = re.compile(\n"
        "            r\"^(?P<root>model\\.layers\\.(?P<layer>\\d+)|mtp\\.(?P<mtp>\\d+))\"\n"
        "            r\"\\.mlp\\.experts\\.(?P<expert>\\d+)\\.\"\n"
        "            r\"(?P<projection>gate_proj|up_proj|down_proj)$\"\n"
        "        )\n"
        "        for name, entry in self.tensor_storage.items():\n"
        "            match = family_pattern.fullmatch(name)\n"
        "            assert match is not None\n"
        "            layer_index = (\n"
        "                int(match.group(\"layer\"))\n"
        "                if match.group(\"layer\") is not None\n"
        "                else num_hidden_layers + int(match.group(\"mtp\"))\n"
        "            )\n"
        "            raw_bits = entry.get(\"bits_per_weight\", self.bits)\n"
        "            try:\n"
        "                bits = float(raw_bits)\n"
        "            except (TypeError, ValueError):\n"
        "                raise ValueError(\n"
        "                    f\"standard EXL3 tensor {name} has invalid \"\n"
        "                    f\"bits_per_weight={raw_bits!r}\"\n"
        "                ) from None\n"
        "            if bits != int(bits) or int(bits) not in (2, 3, 4, 5, 6):\n"
        "                raise ValueError(\n"
        "                    f\"standard EXL3 tensor {name} requires an integral \"\n"
        "                    f\"K2-K6 bitrate, got {raw_bits!r}\"\n"
        "                )\n"
        "            key = (layer_index, int(match.group(\"expert\")))\n"
        "            families.setdefault(key, {})[match.group(\"projection\")] = int(bits)\n"
        "        by_layer: dict[int, dict[int, int]] = {}\n"
        "        required_projections = {\"gate_proj\", \"up_proj\", \"down_proj\"}\n"
        "        for (layer_index, expert_id), projections in families.items():\n"
        "            if set(projections) != required_projections:\n"
        "                raise ValueError(\n"
        "                    \"standard EXL3 expert family is incomplete: \"\n"
        "                    f\"layer={layer_index}, expert={expert_id}, \"\n"
        "                    f\"projections={sorted(projections)}\"\n"
        "                )\n"
        "            bitrates = set(projections.values())\n"
        "            if len(bitrates) != 1:\n"
        "                raise ValueError(\n"
        "                    \"standard EXL3 expert projections disagree on bitrate: \"\n"
        "                    f\"layer={layer_index}, expert={expert_id}, \"\n"
        "                    f\"bitrates={projections}\"\n"
        "                )\n"
        "            by_layer.setdefault(layer_index, {})[expert_id] = bitrates.pop()\n"
        "        for layer_index, experts in by_layer.items():\n"
        "            expected = list(range(max(experts) + 1))\n"
        "            if sorted(experts) != expected:\n"
        "                raise ValueError(\n"
        "                    f\"standard EXL3 layer {layer_index} has a sparse expert map\"\n"
        "                )\n"
        "            self.standard_bits_by_layer[layer_index] = tuple(\n"
        "                experts[expert_id] for expert_id in expected\n"
        "            )\n"
        "        self.standard_fused_moe = True\n"
        "        # GPTQModel emits standard Transformers expert names while NVIDIA's\n"
        "        # DeepSeek-V4 implementation uses the native checkpoint spelling. Keep\n"
        "        # both in the metadata map: source-file lookup still needs the former,\n"
        "        # and FusedMoE construction/codebook validation needs the latter.\n",
        "derive per-layer expert bitrates from tensor metadata",
    )
    replace_once(
        exl3,
        "    def normalize_standard_weight_name(self, name: str) -> str:\n",
        "    def standard_layer_bitrates(\n"
        "        self, layer_name: str, num_experts: int\n"
        "    ) -> tuple[int, ...]:\n"
        "        match = re.search(r\"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)\", layer_name)\n"
        "        if match is None:\n"
        "            raise ValueError(\n"
        "                f\"cannot resolve standard EXL3 layer index from {layer_name!r}\"\n"
        "            )\n"
        "        layer_index = int(match.group(1))\n"
        "        try:\n"
        "            bitrates = self.standard_bits_by_layer[layer_index]\n"
        "        except KeyError as exc:\n"
        "            raise ValueError(\n"
        "                f\"standard EXL3 bitrate map has no layer {layer_index}\"\n"
        "            ) from exc\n"
        "        if len(bitrates) != num_experts:\n"
        "            raise ValueError(\n"
        "                \"standard EXL3 expert count does not match metadata: \"\n"
        "                f\"layer={layer_index}, metadata={len(bitrates)}, \"\n"
        "                f\"model={num_experts}\"\n"
        "            )\n"
        "        return bitrates\n\n"
        "    def normalize_standard_weight_name(self, name: str) -> str:\n",
        "expose standard layer bitrate lookup",
    )
    replace_once(
        exl3,
        "                if rank_sliced\n"
        "                else (int(self.quant_config.bits),) * num_experts\n",
        "                if rank_sliced\n"
        "                else self.quant_config.standard_layer_bitrates(\n"
        "                    str(layer.layer_name), num_experts\n"
        "                )\n",
        "plan standard experts from per-expert bitrates",
    )
    replace_once(
        exl3,
        "        unsupported = sorted(set(tiers).difference((3, 4, 5, 6)))\n"
        "        if unsupported:\n"
        "            raise NotImplementedError(\n"
        "                \"the installed rank-sliced EXL3 fused runtime accepts only K3-K6; \"\n"
        "                f\"the checkpoint contains K{unsupported[0]} payloads\"\n"
        "            )\n",
        "        unsupported = sorted(set(tiers).difference((2, 3, 4, 5, 6)))\n"
        "        if unsupported:\n"
        "            raise NotImplementedError(\n"
        "                \"the installed mixed EXL3 fused runtime accepts only K2-K6; \"\n"
        "                f\"the checkpoint contains K{unsupported[0]} payloads\"\n"
        "            )\n",
        "enable mixed K2 tiers",
    )
    replace_once(
        exl3,
        "    del weight_name\n"
        "    param.load_exl3_weight(\n"
        "        loaded_weight,\n"
        "        expert_id=expert_id,\n"
        "        shard_id=shard_id,\n"
        "    )\n"
        "    return True if return_success else None\n",
        "    try:\n"
        "        param.load_exl3_weight(\n"
        "            loaded_weight,\n"
        "            expert_id=expert_id,\n"
        "            shard_id=shard_id,\n"
        "        )\n"
        "    except (RuntimeError, ValueError) as exc:\n"
        "        raise type(exc)(\n"
        "            f\"{exc}; weight={weight_name!r}, expert={expert_id}, \"\n"
        "            f\"shard={shard_id!r}, shape={tuple(loaded_weight.shape)}\"\n"
        "        ) from exc\n"
        "    return True if return_success else None\n",
        "retain EXL3 expert weight context in loader errors",
    )

    compile(exl3.read_text(), str(exl3), "exec")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
