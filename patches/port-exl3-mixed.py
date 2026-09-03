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
    replace_once(
        exl3,
        "            trellis_bits=bits,\n"
        "            trellis_tile_config=tile_config,\n"
        "        )\n"
        "        layer.exl3_trellis_weights = api.prepare_weights(\n",
        "            trellis_bits=bits,\n"
        "            trellis_tile_config=tile_config,\n"
        "            trellis_codebook=self.quant_config.codebook,\n"
        "            trellis_rate_granularity=\"uniform\",\n"
        "        )\n"
        "        layer.exl3_trellis_weights = api.prepare_weights(\n",
        "pass validated Trellis format to current B12X",
    )
    replace_once(
        exl3,
        "            source_format=\"exl3_trellis_mcg\",\n"
        "            activation=layer.activation.value,\n"
        "            params_dtype=layer.exl3_params_dtype,\n",
        "            source_format=\"exl3_trellis_mcg\",\n"
        "            activation=layer.activation.value,\n"
        "            # B12X keeps the prepared full-rotation payload in FP16 even\n"
        "            # when live activations are BF16.  The execution binding\n"
        "            # converts/returns the caller dtype independently.\n"
        "            params_dtype=torch.float16,\n",
        "plan Trellis prepared weights with the B12X FP16 storage contract",
    )
    replace_once(
        exl3,
        "        layer.exl3_trellis_weights = api.prepare_weights(\n"
        "            plan=weight_plan,\n"
        "            params_dtype=layer.exl3_params_dtype,\n",
        "        layer.exl3_trellis_weights = api.prepare_weights(\n"
        "            plan=weight_plan,\n"
        "            params_dtype=torch.float16,\n",
        "prepare Trellis weights with the B12X FP16 storage contract",
    )
    replace_once(
        exl3,
        "        max_batched_tokens = int(layer.exl3_max_num_batched_tokens)\n"
        "        prefill_plan_enabled = prefill_trellis and max_batched_tokens > max_trellis_m\n"
        "        max_parity_batch = min(max_batched_tokens, min_trellis_m - 1)\n",
        "        max_batched_tokens = int(layer.exl3_max_num_batched_tokens)\n"
        "        prefill_plan_enabled = prefill_trellis and max_batched_tokens > max_trellis_m\n"
        "        raw_prefill_capacities = os.environ.get(\n"
        "            \"VLLM_EXL3_PREFILL_PLAN_CAPACITIES\", \"512,2560\"\n"
        "        )\n"
        "        try:\n"
        "            requested_prefill_capacities = tuple(\n"
        "                int(value.strip())\n"
        "                for value in raw_prefill_capacities.split(\",\")\n"
        "                if value.strip()\n"
        "            )\n"
        "        except ValueError as exc:\n"
        "            raise ValueError(\n"
        "                \"VLLM_EXL3_PREFILL_PLAN_CAPACITIES must be a comma-separated \"\n"
        "                \"list of positive integers\"\n"
        "            ) from exc\n"
        "        if any(capacity <= 0 for capacity in requested_prefill_capacities):\n"
        "            raise ValueError(\n"
        "                \"VLLM_EXL3_PREFILL_PLAN_CAPACITIES must contain only \"\n"
        "                \"positive integers\"\n"
        "            )\n"
        "        prefill_plan_capacities = (\n"
        "            tuple(\n"
        "                sorted(\n"
        "                    {\n"
        "                        min(capacity, max_batched_tokens)\n"
        "                        for capacity in requested_prefill_capacities\n"
        "                        if capacity > max_trellis_m\n"
        "                    }\n"
        "                    | {max_batched_tokens}\n"
        "                )\n"
        "            )\n"
        "            if prefill_plan_enabled\n"
        "            else ()\n"
        "        )\n"
        "        max_parity_batch = min(max_batched_tokens, min_trellis_m - 1)\n",
        "derive bounded Trellis prefill plan capacities",
    )
    replace_once(
        exl3,
        "            prefill_trellis,\n"
        "            prefill_block_m,\n"
        "            layer.exl3_trellis_tile_config,\n",
        "            prefill_trellis,\n"
        "            prefill_block_m,\n"
        "            prefill_plan_capacities,\n"
        "            layer.exl3_trellis_tile_config,\n",
        "key Trellis runtime by prefill capacities",
    )
    replace_once(
        exl3,
        "        prefill_plan = None\n"
        "        prefill_scratch = None\n"
        "        if prefill_plan_enabled:\n"
        "            prefill_plan, prefill_scratch = _plan_with_scratch(\n"
        "                max_batched_tokens, prefill_block_m\n"
        "            )\n",
        "        prefill_states = []\n"
        "        for capacity in prefill_plan_capacities:\n"
        "            prefill_plan, prefill_scratch = _plan_with_scratch(\n"
        "                capacity, prefill_block_m\n"
        "            )\n"
        "            prefill_states.append((capacity, prefill_plan, prefill_scratch))\n",
        "plan size-bucketed Trellis prefill runtimes",
    )
    replace_once(
        exl3,
        '            "prefill_plan": prefill_plan,\n'
        '            "prefill_scratch": prefill_scratch,\n',
        '            "prefill_states": tuple(prefill_states),\n',
        "store size-bucketed Trellis prefill runtimes",
    )
    replace_once(
        exl3,
        "        prefill_arena_mib = (\n"
        "            0.0\n"
        "            if prefill_scratch is None\n"
        "            else prefill_scratch.numel() * prefill_scratch.element_size() / (1 << 20)\n"
        "        )\n"
        "        logger.info_once(\n"
        "            \"EXL3 rank-sliced runtime planned: Trellis m=%d..%d block_m=%d, \"\n"
        "            \"prefill %s capacity=%d chunk=%d topk=%d\",\n"
        "            min_trellis_m,\n"
        "            max_trellis_m,\n"
        "            block_m,\n"
        "            (\n"
        "                f\"trellis block_m={prefill_block_m} arena={prefill_arena_mib:.1f}MiB\"\n"
        "                if prefill_plan is not None\n"
        "                else \"parity\"\n"
        "            ),\n"
        "            max_batched_tokens,\n"
        "            chunk,\n"
        "            topk,\n"
        "        )\n",
        "        prefill_arena_mib = sum(\n"
        "            scratch.numel() * scratch.element_size()\n"
        "            for _, _, scratch in prefill_states\n"
        "        ) / (1 << 20)\n"
        "        logger.info_once(\n"
        "            \"EXL3 rank-sliced runtime planned: Trellis m=%d..%d block_m=%d, \"\n"
        "            \"prefill %s capacity=%d chunk=%d topk=%d\",\n"
        "            min_trellis_m,\n"
        "            max_trellis_m,\n"
        "            block_m,\n"
        "            (\n"
        "                f\"trellis block_m={prefill_block_m} \"\n"
        "                f\"buckets={prefill_plan_capacities} \"\n"
        "                f\"arena={prefill_arena_mib:.1f}MiB\"\n"
        "                if prefill_states\n"
        "                else \"parity\"\n"
        "            ),\n"
        "            max_batched_tokens,\n"
        "            chunk,\n"
        "            topk,\n"
        "        )\n",
        "report size-bucketed Trellis prefill runtimes",
    )
    replace_once(
        exl3,
        '        if runtime["prefill_plan"] is not None and m > runtime["max_trellis_m"]:\n'
        '            if m > runtime["max_batched_tokens"]:\n'
        "                raise ValueError(\n"
        '                    "EXL3 batch exceeds its planned capacity: "\n'
        '                    f"m={m}, capacity={runtime[\'max_batched_tokens\']}"\n'
        "                )\n"
        '            binding = runtime["api"].bind(\n'
        '                runtime["prefill_plan"],\n'
        '                scratch=runtime["prefill_scratch"],\n'
        "                a=x,\n"
        "                experts=layer.exl3_trellis_weights,\n"
        "                topk_weights=topk_weights,\n"
        "                topk_ids=topk_ids,\n"
        "            )\n"
        '            output = runtime["api"].run(binding=binding)\n'
        "            return output.to(x.dtype)\n",
        '        if runtime["prefill_states"] and m > runtime["max_trellis_m"]:\n'
        '            if m > runtime["max_batched_tokens"]:\n'
        "                raise ValueError(\n"
        '                    "EXL3 batch exceeds its planned capacity: "\n'
        '                    f"m={m}, capacity={runtime[\'max_batched_tokens\']}"\n'
        "                )\n"
        "            prefill_state = next(\n"
        "                state for state in runtime[\"prefill_states\"] if m <= state[0]\n"
        "            )\n"
        "            _, prefill_plan, prefill_scratch = prefill_state\n"
        '            binding = runtime["api"].bind(\n'
        "                prefill_plan,\n"
        "                scratch=prefill_scratch,\n"
        "                a=x,\n"
        "                experts=layer.exl3_trellis_weights,\n"
        "                topk_weights=topk_weights,\n"
        "                topk_ids=topk_ids,\n"
        "            )\n"
        '            output = runtime["api"].run(binding=binding)\n'
        "            return output.to(x.dtype)\n",
        "select the smallest fitting Trellis prefill runtime",
    )

    compile(exl3.read_text(), str(exl3), "exec")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
