from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
VISION_MIXED_REPO = (
    "wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2.2-D2-v1"
)
VISION_MIXED_REVISION = "8347bfb8776287ef2dcab2b46e9f15c655825c3a"
OLD_MIXED_REPO = (
    "wrldsuksgo2mars/DeepSeek-V4-Flash-0731-EXL3-K2.1-D2.2-calibrated-v3"
)
OLD_MIXED_REVISION = "7827301eed170e2a5e394f45a13cc66561c601ed"


class ProjectionMixedRecipeTest(unittest.TestCase):
    def test_profiles_are_immutable_and_model_aware(self) -> None:
        entrypoint = (ROOT / "scripts" / "k2-entrypoint.sh").read_text()
        launcher = (ROOT / "launch.sh").read_text()
        stopper = (ROOT / "stop.sh").read_text()
        for token in (
            "vision-k22|vision-k2.2|vision-k22-d2",
            VISION_MIXED_REPO,
            VISION_MIXED_REVISION,
            "k21-d22|k2.1-d2.2",
            OLD_MIXED_REPO,
            OLD_MIXED_REVISION,
        ):
            self.assertIn(token, entrypoint)
        self.assertIn("vision-k22", launcher)
        self.assertIn("k21-d22", launcher)
        self.assertIn("vision-k22", stopper)
        self.assertIn("k21-d22", stopper)

    def test_projection_rates_are_retained_independently(self) -> None:
        adapter = (ROOT / "patches" / "port-exl3-mixed.py").read_text()
        self.assertIn("standard_projection_bits_by_layer", adapter)
        self.assertIn(
            'for name in (\\"gate_proj\\", \\"up_proj\\", \\"down_proj\\")',
            adapter,
        )
        self.assertIn("layer.exl3_projection_mixed = any", adapter)
        self.assertNotIn("expert projections disagree on bitrate", adapter)

    def test_projection_native_b12x_contract_is_used(self) -> None:
        adapter = (ROOT / "patches" / "port-exl3-mixed.py").read_text()
        for token in (
            "ProjectionTrellisTierWeights",
            'trellis_rate_granularity=\\"per_expert_projection\\"',
            'w13_layout=\\"trellis_t256_proj\\"',
            'w4a16_layout=\\"trellis_native\\"',
            "projection_tiers=tuple(native_tiers)",
            "tile_config = None",
            "gate_experts=gate_ids",
            "up_experts=up_ids",
            "down_experts=down_ids",
            "mixed_trellis_route_id_dtypes",
            "mixed_trellis_broadcast_suh",
            "mixed_trellis_broadcast_svh",
            "params_dtype=layer.exl3_params_dtype",
            'and not getattr(layer, \\"exl3_projection_mixed\\", False)',
        ):
            self.assertIn(token, adapter)

    def test_projection_mixed_vision_is_default(self) -> None:
        entrypoint = (ROOT / "scripts" / "k2-entrypoint.sh").read_text()
        compose = (ROOT / "compose.yaml").read_text()
        self.assertIn("model_kind=${MODEL_KIND:-vision-k22}", entrypoint)
        self.assertIn("MODEL_KIND: ${MODEL_KIND:-vision-k22}", compose)

    def test_vision_projection_mixed_gets_a_bootable_memory_default(self) -> None:
        entrypoint = (ROOT / "scripts" / "k2-entrypoint.sh").read_text()
        launcher = (ROOT / "launch.sh").read_text()
        compose = (ROOT / "compose.yaml").read_text()
        self.assertIn("vision_projection_mixed=1", entrypoint)
        self.assertIn("default_gpu_memory_utilization=0.86", entrypoint)
        self.assertIn('GPU_MEMORY_UTILIZATION:-${default_gpu_memory_utilization}', entrypoint)
        self.assertIn('GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-}"', launcher)
        self.assertIn("GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-}", compose)

    def test_vision_allows_one_three_token_draft_cycle(self) -> None:
        entrypoint = (ROOT / "scripts" / "k2-entrypoint.sh").read_text()
        self.assertIn("minimum_dspark_tokens=3", entrypoint)
        self.assertIn("minimum_dspark_tokens=5", entrypoint)
        self.assertIn("dspark_tokens < minimum_dspark_tokens", entrypoint)
        self.assertIn("vision_model && dspark_tokens % 3 != 0", entrypoint)


if __name__ == "__main__":
    unittest.main()
