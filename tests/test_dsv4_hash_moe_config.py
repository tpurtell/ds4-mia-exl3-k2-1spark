from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "hotfix-dsv4-config-hash-moe.py"
SPEC = importlib.util.spec_from_file_location("dsv4_hash_moe_config", PATCH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


SOURCE = '''from typing import Any

class PretrainedConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class DeepseekV4Config(PretrainedConfig):
    def __init__(
        self,
        max_position_embeddings: int = 1048576,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        rope_theta: float = 10000.0,
        **kwargs,
    ):
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.rope_theta = rope_theta
        self.rope_parameters = rope_scaling or rope_parameters
        super().__init__(**kwargs)
'''


class HashMoeConfigPatchTest(unittest.TestCase):
    def _patched_class(self):
        updated, status = MODULE.patch_text(SOURCE)
        self.assertEqual(status, "applied")
        namespace: dict[str, object] = {}
        exec(compile(updated, "deepseek_v4.py", "exec"), namespace)
        return namespace["DeepseekV4Config"], updated

    def test_exact_redundant_list_is_removed(self) -> None:
        cls, _ = self._patched_class()
        config = cls(
            num_hash_layers=3,
            mlp_layer_types=["hash_moe"] * 3 + ["moe"] * 40,
        )
        self.assertEqual(config.kwargs, {"num_hash_layers": 3})

    def test_disagreement_fails_closed(self) -> None:
        cls, _ = self._patched_class()
        with self.assertRaisesRegex(ValueError, "disagrees"):
            cls(
                num_hash_layers=3,
                mlp_layer_types=["hash_moe", "moe", "hash_moe"],
            )

    def test_non_hash_field_is_preserved_for_transformers(self) -> None:
        cls, _ = self._patched_class()
        config = cls(num_hash_layers=0, mlp_layer_types=["moe", "moe"])
        self.assertEqual(config.kwargs["mlp_layer_types"], ["moe", "moe"])

    def test_patch_is_idempotent(self) -> None:
        _, updated = self._patched_class()
        second, status = MODULE.patch_text(updated)
        self.assertEqual(status, "skipped")
        self.assertEqual(second, updated)


if __name__ == "__main__":
    unittest.main()
