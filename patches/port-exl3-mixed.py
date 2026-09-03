#!/usr/bin/env python3
"""Teach EXL3 about standard-HF expert- and projection-mixed K2/K3."""

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
        "        self.standard_projection_bits_by_layer: dict[\n"
        "            int, tuple[tuple[int, int, int], ...]\n"
        "        ] = {}\n"
        "        self.standard_fused_moe = False",
        "store standard mixed-bitrate projection maps",
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
        "        # The top-level bits field is only a checkpoint summary. The executable\n"
        "        # integer bitrate lives on each projection tensor independently.\n"
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
        "        by_layer: dict[int, dict[int, tuple[int, int, int]]] = {}\n"
        "        required_projections = {\"gate_proj\", \"up_proj\", \"down_proj\"}\n"
        "        for (layer_index, expert_id), projections in families.items():\n"
        "            if set(projections) != required_projections:\n"
        "                raise ValueError(\n"
        "                    \"standard EXL3 expert family is incomplete: \"\n"
        "                    f\"layer={layer_index}, expert={expert_id}, \"\n"
        "                    f\"projections={sorted(projections)}\"\n"
        "                )\n"
        "            by_layer.setdefault(layer_index, {})[expert_id] = tuple(\n"
        "                projections[name]\n"
        "                for name in (\"gate_proj\", \"up_proj\", \"down_proj\")\n"
        "            )\n"
        "        for layer_index, experts in by_layer.items():\n"
        "            expected = list(range(max(experts) + 1))\n"
        "            if sorted(experts) != expected:\n"
        "                raise ValueError(\n"
        "                    f\"standard EXL3 layer {layer_index} has a sparse expert map\"\n"
        "                )\n"
        "            projection_bitrates = tuple(\n"
        "                experts[expert_id] for expert_id in expected\n"
        "            )\n"
        "            self.standard_projection_bits_by_layer[layer_index] = (\n"
        "                projection_bitrates\n"
        "            )\n"
        "            if all(len(set(values)) == 1 for values in projection_bitrates):\n"
        "                self.standard_bits_by_layer[layer_index] = tuple(\n"
        "                    values[0] for values in projection_bitrates\n"
        "                )\n"
        "        self.standard_fused_moe = True\n"
        "        # GPTQModel emits standard Transformers expert names while NVIDIA's\n"
        "        # DeepSeek-V4 implementation uses the native checkpoint spelling. Keep\n"
        "        # both in the metadata map: source-file lookup still needs the former,\n"
        "        # and FusedMoE construction/codebook validation needs the latter.\n",
        "derive per-layer projection bitrates from tensor metadata",
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
        "    def standard_layer_projection_bitrates(\n"
        "        self, layer_name: str, num_experts: int\n"
        "    ) -> tuple[tuple[int, int, int], ...]:\n"
        "        match = re.search(r\"(?:^|\\.)layers\\.(\\d+)(?:\\.|$)\", layer_name)\n"
        "        if match is None:\n"
        "            raise ValueError(\n"
        "                f\"cannot resolve standard EXL3 layer index from {layer_name!r}\"\n"
        "            )\n"
        "        layer_index = int(match.group(1))\n"
        "        try:\n"
        "            bitrates = self.standard_projection_bits_by_layer[layer_index]\n"
        "        except KeyError as exc:\n"
        "            raise ValueError(\n"
        "                f\"standard EXL3 projection map has no layer {layer_index}\"\n"
        "            ) from exc\n"
        "        if len(bitrates) != num_experts:\n"
        "            raise ValueError(\n"
        "                \"standard EXL3 projection expert count does not match metadata: \"\n"
        "                f\"layer={layer_index}, metadata={len(bitrates)}, \"\n"
        "                f\"model={num_experts}\"\n"
        "            )\n"
        "        return bitrates\n\n"
        "    def normalize_standard_weight_name(self, name: str) -> str:\n",
        "expose standard layer and projection bitrate lookup",
    )
    replace_once(
        exl3,
        "            layer.exl3_layer_bitrates = (\n"
        "                self.quant_config.rank_sliced_layer_bitrates(str(layer.layer_name))\n"
        "                if rank_sliced\n"
        "                else (int(self.quant_config.bits),) * num_experts\n"
        "            )\n"
        "            layer.exl3_mixed_bitrate = len(set(layer.exl3_layer_bitrates)) > 1\n",
        "            layer.exl3_projection_bitrates = ()\n"
        "            layer.exl3_projection_mixed = False\n"
        "            if rank_sliced:\n"
        "                layer.exl3_layer_bitrates = (\n"
        "                    self.quant_config.rank_sliced_layer_bitrates(\n"
        "                        str(layer.layer_name)\n"
        "                    )\n"
        "                )\n"
        "            else:\n"
        "                projection_bitrates = (\n"
        "                    self.quant_config.standard_layer_projection_bitrates(\n"
        "                        str(layer.layer_name), num_experts\n"
        "                    )\n"
        "                )\n"
        "                layer.exl3_projection_bitrates = projection_bitrates\n"
        "                layer.exl3_projection_mixed = any(\n"
        "                    len(set(values)) > 1 for values in projection_bitrates\n"
        "                )\n"
        "                layer.exl3_layer_bitrates = tuple(\n"
        "                    values[0] for values in projection_bitrates\n"
        "                )\n"
        "            layer.exl3_mixed_bitrate = (\n"
        "                layer.exl3_projection_mixed\n"
        "                or len(set(layer.exl3_layer_bitrates)) > 1\n"
        "            )\n",
        "plan standard experts from per-projection bitrates",
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
        "    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:\n"
        "        if getattr(layer, \"exl3_mixed_bitrate\", False):\n"
        "            self._prepare_mixed_rank_sliced_weights(layer)\n"
        "            return\n",
        "    def _prepare_projection_mixed_weights(\n"
        "        self, layer: RoutedExperts\n"
        "    ) -> None:\n"
        "        api = _load_b12x_fused_moe()\n"
        "        from b12x.moe.fused_moe.trellis import (\n"
        "            ProjectionTrellisTierWeights,\n"
        "        )\n"
        "\n"
        "        num_experts = int(layer.local_num_experts)\n"
        "        hidden_size = int(layer.exl3_hidden_size)\n"
        "        intermediate_size = int(layer.exl3_intermediate_size_per_partition)\n"
        "        projection_bitrates = tuple(\n"
        "            tuple(int(value) for value in values)\n"
        "            for values in layer.exl3_projection_bitrates\n"
        "        )\n"
        "        if (\n"
        "            len(projection_bitrates) != num_experts\n"
        "            or any(len(values) != 3 for values in projection_bitrates)\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"projection-mixed EXL3 metadata must provide gate/up/down \"\n"
        "                f\"rates for {num_experts} experts\"\n"
        "            )\n"
        "        tier_bits = tuple(\n"
        "            sorted({bit for values in projection_bitrates for bit in values})\n"
        "        )\n"
        "        if (\n"
        "            len(tier_bits) != 2\n"
        "            or tier_bits[1] != tier_bits[0] + 1\n"
        "            or any(bit not in (2, 3, 4, 5, 6) for bit in tier_bits)\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"projection-mixed EXL3 requires exactly two consecutive \"\n"
        "                f\"K2-K6 tiers, got {tier_bits}\"\n"
        "            )\n"
        "\n"
        "        w13_param = layer.w13_trellis\n"
        "        w2_param = layer.w2_trellis\n"
        "        if (\n"
        "            tuple(w13_param.exl3_shard_ids) != (\"w1\", \"w3\")\n"
        "            or tuple(w2_param.exl3_shard_ids) != (\"w2\",)\n"
        "        ):\n"
        "            raise ValueError(\n"
        "                \"projection-mixed EXL3 requires w13=(w1,w3) and w2=(w2)\"\n"
        "            )\n"
        "        gate_suh, up_suh = self._rank_sliced_backing(layer, \"w13_suh\")\n"
        "        gate_svh, up_svh = self._rank_sliced_backing(layer, \"w13_svh\")\n"
        "        down_suh = self._rank_sliced_backing(layer, \"w2_suh\")\n"
        "        down_svh = self._rank_sliced_backing(layer, \"w2_svh\")\n"
        "        device = gate_suh.device\n"
        "\n"
        "        def stack_projection(param, expert_ids, shard_id, tail_shape):\n"
        "            tensors = tuple(\n"
        "                param.exl3_tensors[(expert_id, shard_id)]\n"
        "                for expert_id in expert_ids\n"
        "            )\n"
        "            if tensors:\n"
        "                return torch.stack(tensors).contiguous()\n"
        "            return torch.empty(\n"
        "                (0, *tail_shape), dtype=torch.int16, device=device\n"
        "            )\n"
        "\n"
        "        native_tiers = []\n"
        "        membership_counts = []\n"
        "        for bits in tier_bits:\n"
        "            gate_ids = tuple(\n"
        "                expert_id\n"
        "                for expert_id, values in enumerate(projection_bitrates)\n"
        "                if values[0] == bits\n"
        "            )\n"
        "            up_ids = tuple(\n"
        "                expert_id\n"
        "                for expert_id, values in enumerate(projection_bitrates)\n"
        "                if values[1] == bits\n"
        "            )\n"
        "            down_ids = tuple(\n"
        "                expert_id\n"
        "                for expert_id, values in enumerate(projection_bitrates)\n"
        "                if values[2] == bits\n"
        "            )\n"
        "            last = 16 * bits\n"
        "            fc1_tail = (\n"
        "                hidden_size // 16, intermediate_size // 16, last\n"
        "            )\n"
        "            gate = stack_projection(w13_param, gate_ids, \"w1\", fc1_tail)\n"
        "            up = stack_projection(w13_param, up_ids, \"w3\", fc1_tail)\n"
        "            down = stack_projection(\n"
        "                w2_param,\n"
        "                down_ids,\n"
        "                \"w2\",\n"
        "                (intermediate_size // 16, hidden_size // 16, last),\n"
        "            )\n"
        "            w13 = torch.cat((gate, up), dim=0).contiguous()\n"
        "            expected_w13 = (\n"
        "                len(gate_ids) + len(up_ids), *fc1_tail\n"
        "            )\n"
        "            expected_w2 = (\n"
        "                len(down_ids),\n"
        "                intermediate_size // 16,\n"
        "                hidden_size // 16,\n"
        "                last,\n"
        "            )\n"
        "            if tuple(w13.shape) != expected_w13 or tuple(down.shape) != expected_w2:\n"
        "                raise ValueError(\n"
        "                    f\"projection-mixed EXL3 K{bits} geometry mismatch: \"\n"
        "                    f\"w13={tuple(w13.shape)}, w2={tuple(down.shape)}, \"\n"
        "                    f\"expected={expected_w13}/{expected_w2}\"\n"
        "                )\n"
        "            native_tiers.append(\n"
        "                ProjectionTrellisTierWeights(\n"
        "                    bits=bits,\n"
        "                    w13=w13,\n"
        "                    w2=down,\n"
        "                    gate_experts=gate_ids,\n"
        "                    up_experts=up_ids,\n"
        "                    down_experts=down_ids,\n"
        "                )\n"
        "            )\n"
        "            membership_counts.append(\n"
        "                (bits, len(gate_ids), len(up_ids), len(down_ids))\n"
        "            )\n"
        "\n"
        "        intermediate_rotations = torch.cat(\n"
        "            (gate_svh, up_svh, down_suh), dim=1\n"
        "        ).contiguous()\n"
        "        # Leave projection-mixed geometry unresolved: current B12X\n"
        "        # uses K64/N128 for bounded direct decode and K128/N128 for\n"
        "        # expert-packed prefill.  Pinning one tuple here defeats that\n"
        "        # live-shape selector and measurably slows mixed decode.\n"
        "        tile_config = None\n"
        "        weight_plan = api.plan_weights(\n"
        "            quant_modes=\"w4a16\",\n"
        "            source_format=\"exl3_trellis_mcg\",\n"
        "            activation=layer.activation.value,\n"
        "            # Projection-mixed kernels bind live activations directly,\n"
        "            # so their plan dtype must follow vLLM (BF16 on DSV4).\n"
        "            # Unlike uniform trellis_t256, trellis_t256_proj does not\n"
        "            # use the independent FP16 prepared-weight contract.\n"
        "            params_dtype=layer.exl3_params_dtype,\n"
        "            num_experts=num_experts,\n"
        "            hidden_size=hidden_size,\n"
        "            intermediate_size=intermediate_size,\n"
        "            w13_layout=\"trellis_t256_proj\",\n"
        "            w4a16_layout=\"trellis_native\",\n"
        "            trellis_bits=tier_bits[0],\n"
        "            trellis_tile_config=tile_config,\n"
        "            trellis_codebook=self.quant_config.codebook,\n"
        "            trellis_rate_granularity=\"per_expert_projection\",\n"
        "        )\n"
        "        layer.exl3_trellis_weights = api.prepare_weights(\n"
        "            plan=weight_plan,\n"
        "            params_dtype=layer.exl3_params_dtype,\n"
        "            projection_tiers=tuple(native_tiers),\n"
        "            gate_suh=gate_suh,\n"
        "            up_suh=up_suh,\n"
        "            intermediate_rotations=intermediate_rotations,\n"
        "            down_svh=down_svh,\n"
        "        )\n"
        "        layer.exl3_trellis_tile_config = tile_config\n"
        "\n"
        "        for prefix in (\"w13\", \"w2\"):\n"
        "            for suffix in (\"suh\", \"svh\", \"trellis\", \"mcg\", \"mul1\"):\n"
        "                param = getattr(layer, f\"{prefix}_{suffix}\")\n"
        "                param.exl3_tensors.clear()\n"
        "                param.exl3_backing = None\n"
        "        logger.info(\n"
        "            \"EXL3 projection-mixed Trellis %s: \"\n"
        "            \"tiers=(bits,gate,up,down)%s\",\n"
        "            layer.layer_name,\n"
        "            tuple(membership_counts),\n"
        "        )\n"
        "\n"
        "    def _prepare_rank_sliced_weights(self, layer: RoutedExperts) -> None:\n"
        "        if getattr(layer, \"exl3_projection_mixed\", False):\n"
        "            self._prepare_projection_mixed_weights(layer)\n"
        "            return\n"
        "        if getattr(layer, \"exl3_mixed_bitrate\", False):\n"
        "            self._prepare_mixed_rank_sliced_weights(layer)\n"
        "            return\n",
        "prepare projection-native Trellis weights",
    )
    replace_once(
        exl3,
        "        if getattr(layer, \"exl3_mixed_bitrate\", False):\n"
        "            return self._apply_mixed_rank_sliced(\n"
        "                layer,\n"
        "                x,\n"
        "                topk_weights,\n"
        "                topk_ids,\n"
        "            )\n",
        "        if (\n"
        "            getattr(layer, \"exl3_mixed_bitrate\", False)\n"
        "            and not getattr(layer, \"exl3_projection_mixed\", False)\n"
        "        ):\n"
        "            return self._apply_mixed_rank_sliced(\n"
        "                layer,\n"
        "                x,\n"
        "                topk_weights,\n"
        "                topk_ids,\n"
        "            )\n",
        "route projection-mixed execution through unified B12X API",
    )
    replace_once(
        exl3,
        "        api = _load_b12x_fused_moe()\n"
        "\n"
        "        def _plan_with_scratch(plan_max_tokens: int, plan_block_m: int):\n"
        "            caps = api.Caps(\n"
        "                max_tokens=plan_max_tokens,\n"
        "                num_topk=topk,\n"
        "                # vLLM supplies final top-k IDs/weights to bind(); the fused-MoE\n"
        "                # router workspace is unused. A zero route-workspace request\n"
        "                # still lets the W4A16 core derive route_E from weight_E.\n"
        "                route_num_experts=0,\n"
        "                device=x.device,\n"
        "                weight_plan=layer.exl3_trellis_weights.plan,\n"
        "                quant_mode=\"w4a16\",\n"
        "                w4a16_block_size_m=plan_block_m,\n"
        "            )\n",
        "        api = _load_b12x_fused_moe()\n"
        "\n"
        "        def _plan_with_scratch(plan_max_tokens: int, plan_block_m: int):\n"
        "            mixed_launch_caps = {}\n"
        "            if getattr(layer, \"exl3_projection_mixed\", False):\n"
        "                # Plan only the route dtype seen at this vLLM boundary.\n"
        "                # EXL3 supplies per-expert input and output rotations, so\n"
        "                # broadcast specializations are unreachable. B12X's broad\n"
        "                # defaults otherwise compile an unnecessary 8-way product.\n"
        "                if topk_ids.dtype not in (torch.int32, torch.int64):\n"
        "                    raise TypeError(\n"
        "                        \"projection-mixed EXL3 route IDs must be int32 \"\n"
        "                        f\"or int64, got {topk_ids.dtype}\"\n"
        "                    )\n"
        "                mixed_launch_caps = {\n"
        "                    \"mixed_trellis_route_id_dtypes\": (topk_ids.dtype,),\n"
        "                    \"mixed_trellis_broadcast_suh\": (False,),\n"
        "                    \"mixed_trellis_broadcast_svh\": (False,),\n"
        "                }\n"
        "            caps = api.Caps(\n"
        "                max_tokens=plan_max_tokens,\n"
        "                num_topk=topk,\n"
        "                # vLLM supplies final top-k IDs/weights to bind(); the fused-MoE\n"
        "                # router workspace is unused. A zero route-workspace request\n"
        "                # still lets the W4A16 core derive route_E from weight_E.\n"
        "                route_num_experts=0,\n"
        "                device=x.device,\n"
        "                weight_plan=layer.exl3_trellis_weights.plan,\n"
        "                quant_mode=\"w4a16\",\n"
        "                w4a16_block_size_m=plan_block_m,\n"
        "                **mixed_launch_caps,\n"
        "            )\n",
        "limit projection-mixed JIT planning to reachable vLLM variants",
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
