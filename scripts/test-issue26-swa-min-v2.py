#!/usr/bin/env python3
"""Unit tests for the issue #26/#36 hybrid-SWA hotfix v2 (no live serve)."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOTFIX = ROOT / "patches" / "hotfix-dsv4-issue26-hybrid-swa-min.py"


def _load():
    spec = importlib.util.spec_from_file_location("hotfix_issue26", HOTFIX)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Issue26SwaMinV2Test(unittest.TestCase):
    def setUp(self):
        self.mod = _load()

    def test_reverts_v1_continue(self):
        src = "prefix\n" + self.mod.V1_INJECT + "suffix\n"
        new, status = self.mod.apply_text(src)
        self.assertEqual(status, "reverted-v1")
        self.assertIn(self.mod.MARK_V2, new)
        self.assertNotIn("must not shrink the hybrid common hit", new)
        self.assertNotIn("if isinstance(spec, SlidingWindowSpec):", new)
        self.assertIn("curr_hit_length = _new_hit_length", new)

    def test_annotates_stock(self):
        src = "prefix\n" + self.mod.STOCK + "suffix\n"
        new, status = self.mod.apply_text(src)
        self.assertEqual(status, "annotated")
        self.assertIn(self.mod.MARK_V2, new)
        self.assertIn("curr_hit_length = _new_hit_length", new)

    def test_idempotent(self):
        src = "prefix\n" + self.mod.V1_INJECT + "suffix\n"
        once, _ = self.mod.apply_text(src)
        twice, status = self.mod.apply_text(once)
        self.assertEqual(status, "v2")
        self.assertEqual(once, twice)

    def test_v1_would_skip_prefill_on_empty_swa(self):
        # Document the v1 control-flow bug: continue before curr_hit_length
        # assign leaves a long MLA candidate in place.
        self.assertIn("continue\n", self.mod.V1_INJECT)
        self.assertLess(
            self.mod.V1_INJECT.index("continue\n"),
            self.mod.V1_INJECT.index("curr_hit_length = _new_hit_length"),
        )
        self.assertNotIn("continue\n", self.mod.V2_BLOCK)


if __name__ == "__main__":
    unittest.main()
