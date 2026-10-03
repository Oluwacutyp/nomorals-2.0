"""Universal wave: Liger-Kernel accelerator module.

Liger is a *free* (Apache-2.0) fused-Triton-kernel drop-in: +20%
throughput, -60% VRAM on transformer training.  These tests pin the
probe logic, the TRL kwarg builder, the axolotl plugin block (keys
copied from the axolotl docs), and the Colab-script wiring — all
without needing liger_kernel installed (the sandbox has no GPU).
"""

from __future__ import annotations

import ast
import tempfile
import unittest
from unittest.mock import patch

from nomorals.training import liger
from nomorals.training.finetune import write_colab_script
from nomorals.training.liger import (
    AXOLOTL_LIGER_PLUGIN,
    axolotl_liger_block,
    liger_available,
    pip_hint,
    trl_sft_kwargs,
)


class LigerProbeTests(unittest.TestCase):
    def test_available_returns_tuple(self) -> None:
        ok, reason = liger_available()
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(reason, str)
        self.assertTrue(reason)

    def test_available_false_without_package(self) -> None:
        with patch.object(liger, "_spec_exists", return_value=False):
            ok, reason = liger_available()
        self.assertFalse(ok)
        self.assertIn("pip install liger-kernel", reason)

    def test_available_true_with_package(self) -> None:
        with patch.object(liger, "_spec_exists", return_value=True):
            ok, reason = liger_available()
        self.assertTrue(ok)
        self.assertIn("fused kernels", reason)

    def test_trl_kwargs_empty_when_missing(self) -> None:
        with patch.object(liger, "_spec_exists", return_value=False):
            self.assertEqual(trl_sft_kwargs(), {})

    def test_trl_kwargs_set_when_present(self) -> None:
        with patch.object(liger, "_spec_exists", return_value=True):
            self.assertEqual(trl_sft_kwargs(), {"use_liger_kernel": True})

    def test_pip_hint_names_package(self) -> None:
        self.assertIn("liger-kernel", pip_hint())


class AxolotlPluginBlockTests(unittest.TestCase):
    def test_plugin_entry_is_documented_path(self) -> None:
        self.assertEqual(AXOLOTL_LIGER_PLUGIN,
                         "axolotl.integrations.liger.LigerPlugin")

    def test_block_enables_the_money_kernels(self) -> None:
        block = axolotl_liger_block()
        text = "\n".join(block)
        self.assertIn(AXOLOTL_LIGER_PLUGIN, text)
        # the fused linear CE is the biggest single VRAM win (logits tensor
        # never materializes) — it must be on
        self.assertIn("liger_fused_linear_cross_entropy: true", text)
        self.assertIn("liger_rms_norm: true", text)
        self.assertIn("liger_glu_activation: true", text)
        self.assertIn("liger_rope: true", text)

    def test_block_does_not_change_loss_semantics(self) -> None:
        # token scaling changes the LOSS MATH, not just the kernels — it
        # must stay an explicit user choice, never a default
        text = "\n".join(axolotl_liger_block())
        self.assertNotIn("liger_use_token_scaling", text)
        self.assertNotIn("cutedsl", text)


class ColabScriptLigerTests(unittest.TestCase):
    def _script(self) -> str:
        with tempfile.TemporaryDirectory() as d:
            path = write_colab_script(d + "/mix", train_file="/tmp/x.jsonl")
            return open(path, encoding="utf-8").read()

    def test_script_is_valid_python(self) -> None:
        ast.parse(self._script())

    def test_probe_ships_inline(self) -> None:
        src = self._script()
        self.assertIn("use_liger_kernel", src)
        self.assertIn("_LIGER_SFT_EXTRA", src)
        self.assertIn("**_LIGER_SFT_EXTRA", src)

    def test_probe_guards_old_trl(self) -> None:
        # an old trl raises TypeError on unknown kwargs — the probe must
        # check the SFTConfig signature before enabling the flag
        src = self._script()
        self.assertIn("inspect.signature", src)
        self.assertIn("SFTConfig.__init__", src)

    def test_pip_line_mentions_liger(self) -> None:
        self.assertIn("liger-kernel", self._script())


if __name__ == "__main__":
    unittest.main()
