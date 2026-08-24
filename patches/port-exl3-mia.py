#!/usr/bin/env python3
"""Add the narrow EXL3 integration surface to Mia's pinned vLLM tree."""

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
    registry = root / "model_executor/layers/quantization/__init__.py"
    replace_once(
        registry,
        '    "deepseek_v4_fp8",\n    "online",',
        '    "deepseek_v4_fp8",\n    "exl3",\n    "online",',
        "register EXL3 quantization name",
    )
    replace_once(
        registry,
        "    from .experts_int8 import ExpertsInt8Config\n",
        "    from .experts_int8 import ExpertsInt8Config\n"
        "    from .exl3 import Exl3Config\n",
        "import EXL3 config",
    )
    replace_once(
        registry,
        '        "deepseek_v4_fp8": DeepseekV4FP8Config,\n'
        '        "humming": HummingConfig,',
        '        "deepseek_v4_fp8": DeepseekV4FP8Config,\n'
        '        "exl3": Exl3Config,\n'
        '        "humming": HummingConfig,',
        "map EXL3 config",
    )

    model = root / "models/deepseek_v4/nvidia/model.py"
    replace_once(
        model,
        '            prefix=f"{prefix}.experts",\n'
        "            scoring_func=self.scoring_func,",
        '            prefix=f"{prefix}.experts",\n'
        '            ckpt_names=("w1", "w2", "w3"),\n'
        "            scoring_func=self.scoring_func,",
        "declare DSV4 expert checkpoint names",
    )
    replace_once(
        model,
        "        self.config = config\n"
        '        expert_dtype = getattr(config, "expert_dtype", "fp4")',
        "        self.config = config\n"
        "        self.quant_config = vllm_config.quant_config\n"
        '        expert_dtype = getattr(config, "expert_dtype", "fp4")',
        "retain target quantization config",
    )
    replace_once(
        model,
        "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
        '        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])',
        "    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:\n"
        "        normalize = getattr(\n"
        "            self.quant_config,\n"
        '            "normalize_standard_weight_name",\n'
        "            None,\n"
        "        )\n"
        "        if normalize is not None:\n"
        "            weights = ((normalize(name), weight) for name, weight in weights)\n"
        '        loader = AutoWeightsLoader(self, skip_substrs=["mtp."])',
        "normalize standard EXL3 target names",
    )

    dspark = root / "models/deepseek_v4/nvidia/dspark.py"
    replace_once(
        dspark,
        "        config = vllm_config.speculative_config.draft_model_config.hf_config\n"
        "        self.config = config\n",
        "        config = vllm_config.speculative_config.draft_model_config.hf_config\n"
        "        self.config = config\n"
        "        self.quant_config = vllm_config.quant_config\n",
        "retain draft quantization config",
    )
    replace_once(
        dspark,
        "        self.draft_model_config = vllm_config.speculative_config.draft_model_config\n"
        "        self.config = self.draft_model_config.hf_config\n"
        "        self.model = DSparkDeepseekV4Model(\n",
        "        self.draft_model_config = vllm_config.speculative_config.draft_model_config\n"
        "        self.config = self.draft_model_config.hf_config\n"
        "        self.quant_config = vllm_config.quant_config\n"
        "        self.model = DSparkDeepseekV4Model(\n",
        "retain draft wrapper quantization config",
    )
    replace_once(
        dspark,
        "        current_vllm_config = get_current_vllm_config()\n"
        "        self.layers = nn.ModuleList(\n"
        "            [\n"
        "                DeepseekV4DecoderLayer(\n"
        "                    current_vllm_config,\n"
        '                    prefix=maybe_prefix(prefix, f"layers.{self.num_hidden_layers + i}"),\n'
        "                )\n"
        "                for i in range(self.num_dspark_layers)\n"
        "            ]\n"
        "        )",
        "        current_vllm_config = get_current_vllm_config()\n"
        "        target_hf_config = current_vllm_config.model_config.hf_config\n"
        "        target_n_routed_experts = target_hf_config.n_routed_experts\n"
        "        target_hf_config.n_routed_experts = config.n_routed_experts\n"
        "        try:\n"
        "            self.layers = nn.ModuleList(\n"
        "                [\n"
        "                    DeepseekV4DecoderLayer(\n"
        "                        current_vllm_config,\n"
        "                        prefix=maybe_prefix(\n"
        '                            prefix, f"layers.{self.num_hidden_layers + i}"\n'
        "                        ),\n"
        "                    )\n"
        "                    for i in range(self.num_dspark_layers)\n"
        "                ]\n"
        "            )\n"
        "        finally:\n"
        "            target_hf_config.n_routed_experts = target_n_routed_experts",
        "construct draft layers with draft expert count",
    )
    replace_once(
        dspark,
        "        for name, loaded_weight in weights:\n"
        "            mapped = self._remap_dspark_name(name)",
        "        normalize = getattr(\n"
        "            self.quant_config,\n"
        '            "normalize_standard_weight_name",\n'
        "            None,\n"
        "        )\n"
        "        for name, loaded_weight in weights:\n"
        "            original_name = name\n"
        "            if normalize is not None:\n"
        "                name = normalize(name)\n"
        "            mapped = self._remap_dspark_name(name)",
        "normalize standard EXL3 draft names",
    )
    replace_once(
        dspark,
        "                        loaded_weight,\n"
        "                        name_mapped,\n"
        "                        shard_id=shard_id,\n",
        "                        loaded_weight,\n"
        "                        original_name,\n"
        "                        shard_id=shard_id,\n",
        "retain draft checkpoint names in expert loader diagnostics",
    )

    compile(registry.read_text(), str(registry), "exec")
    compile(model.read_text(), str(model), "exec")
    compile(dspark.read_text(), str(dspark), "exec")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} VLLM_ROOT")
    main(Path(sys.argv[1]))
