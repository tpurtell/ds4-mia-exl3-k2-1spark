from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
VISION_REPO = "wrldsuksgo2mars/DeepSeek-V4-Flash-Vision-Exp-EXL3-K2-v1"
VISION_REVISION = "419697c409cb4157471bcaf68be07dbd151b0a40"
B12X_REVISION = "3fc8d1491d1313c0ca64b2b95772972b7f42ee9d"


class VisionK2RecipeTest(unittest.TestCase):
    def test_image_pins_vision_capable_b12x_head(self) -> None:
        dockerfile = (ROOT / "Dockerfile").read_text()
        self.assertIn(f"ARG B12X_COMMIT={B12X_REVISION}", dockerfile)
        self.assertIn(
            "COPY patches/vision_exp /opt/dspark-patches/vision_exp", dockerfile
        )
        self.assertIn("patches/hotfix-dsv4-vision-exp.py", dockerfile)

    def test_exl3_adapter_passes_current_b12x_trellis_contract(self) -> None:
        adapter = (ROOT / "patches" / "port-exl3-mixed.py").read_text()
        self.assertIn(
            'trellis_codebook=self.quant_config.codebook,', adapter
        )
        self.assertIn('trellis_rate_granularity=\\"uniform\\"', adapter)
        self.assertEqual(adapter.count("params_dtype=torch.float16"), 2)
        self.assertIn(
            "plan Trellis prepared weights with the B12X FP16 storage contract",
            adapter,
        )
        self.assertIn(
            "prepare Trellis weights with the B12X FP16 storage contract",
            adapter,
        )

    def test_exl3_prefill_uses_bounded_size_buckets(self) -> None:
        adapter = (ROOT / "patches" / "port-exl3-mixed.py").read_text()
        self.assertIn("VLLM_EXL3_PREFILL_PLAN_CAPACITIES", adapter)
        self.assertIn('\\"512,2560\\"', adapter)
        self.assertIn("prefill_states.append((capacity, prefill_plan, prefill_scratch))", adapter)
        self.assertIn("if m <= state[0]", adapter)

    def test_vision_k2_profile_is_exactly_pinned(self) -> None:
        entrypoint = (ROOT / "scripts" / "k2-entrypoint.sh").read_text()
        self.assertIn("vision-k2|vision)", entrypoint)
        self.assertIn(VISION_REPO, entrypoint)
        self.assertIn(VISION_REVISION, entrypoint)
        self.assertIn("default_dspark_tokens=6", entrypoint)
        self.assertIn("dspark_tokens % 3", entrypoint)

    def test_vision_runtime_is_fail_closed_and_registers_multimodal_limit(
        self,
    ) -> None:
        entrypoint = (ROOT / "scripts" / "k2-entrypoint.sh").read_text()
        copy = entrypoint.index('cp "${encoding_source}"')
        encoding_hotfix = entrypoint.index("hotfix-encoding-dsv4-issue21.py", copy)
        vision_hotfix = entrypoint.index(
            "hotfix-dsv4-vision-exp.py", encoding_hotfix
        )
        serve = entrypoint.index("exec /usr/local/bin/vllm serve")
        self.assertLess(copy, encoding_hotfix)
        self.assertLess(encoding_hotfix, vision_hotfix)
        self.assertLess(vision_hotfix, serve)
        self.assertIn(
            'limit_mm_args=(--limit-mm-per-prompt "${limit_mm_json}")',
            entrypoint,
        )
        self.assertIn('"${limit_mm_args[@]}"', entrypoint)
        self.assertIn(
            "Vision-Exp encoding/encoding_dsv4.py is missing", entrypoint
        )

    def test_default_remains_calibrated_0731_k2_v1(self) -> None:
        compose = (ROOT / "compose.yaml").read_text()
        env_example = (ROOT / ".env.example").read_text()
        self.assertIn("MODEL_KIND: ${MODEL_KIND:-k2}", compose)
        self.assertIn("MODEL_KIND=k2", env_example)
        self.assertNotIn("DSPARK_TOKENS=5", env_example)

    def test_qualified_xgrammar_fix_is_enabled_by_default(self) -> None:
        compose = (ROOT / "compose.yaml").read_text()
        launcher = (ROOT / "launch.sh").read_text()
        entrypoint = (ROOT / "scripts" / "k2-entrypoint.sh").read_text()
        env_example = (ROOT / ".env.example").read_text()
        token = "DSPARK_ENABLE_ISSUE136_XGRAMMAR_HOTFIX"
        self.assertIn(f"{token}=1", env_example)
        self.assertIn(f"${{{token}:-1}}", compose)
        self.assertIn(f"${{{token}:-1}}", launcher)
        self.assertIn(f'case "${{{token}:-1}}" in', entrypoint)

    def test_empty_values_are_consumed_before_vllm(self) -> None:
        entrypoint = (ROOT / "scripts" / "k2-entrypoint.sh").read_text()
        self.assertIn(
            "dspark_tokens=${DSPARK_TOKENS:-${default_dspark_tokens}}", entrypoint
        )
        self.assertIn(
            "max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE:-}",
            entrypoint,
        )
        self.assertIn(
            "export CUTE_DSL_ARCH=${CUTE_DSL_ARCH:-sm_121a}", entrypoint
        )


if __name__ == "__main__":
    unittest.main()
