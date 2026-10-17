"""Meta-suite (infra spec §9.1): ONE command runs every infrastructure
suite — one per spec section plus the regression suites.

Run it with:  pytest tests/test_infrastructure.py

Each suite runs in its own subprocess (isolated, same as running it
standalone) and must exit 0. A failure names the suite file — run that
file directly for the detail.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest

# One suite per spec section (P0–P5) + the orphan audit + the §9.1
# regression suites. Keep this list in sync with docs/INFRASTRUCTURE_SPEC.md.
SUITES = [
    # P0 errors + config (§8.1–§8.3)
    "test_config_mapping.py",
    "test_config_env.py",
    # P1 one path (§4)
    "test_one_path.py",
    "test_stt_upload_filename.py",
    "test_voice_note_routing.py",
    "test_telegram_audio_inbound.py",
    "test_voice_note_stt_failure.py",
    # P2 commands + outputs (§5 + §6)
    "test_outputs.py",
    "test_command_registry_coverage.py",
    "test_command_stack.py",
    "test_tool_render.py",
    "test_help_sync.py",
    "test_slash_help_routing.py",
    # P3 style (§3)
    "test_style_tuning.py",
    "test_partner_style_system.py",
    "test_personal_bot.py",
    # P4 group controls (§7)
    "test_group_controls.py",
    "test_chat_modes.py",
    # P5 startup + health (§8.4 + §8.6)
    "test_startup_doctor.py",
    # §9.1 regression suites
    "test_partner_runtime.py",
    "test_presence.py",
    "test_presence_voice_note.py",
    "test_music_vocal_tts_error.py",
    "test_game_move_routing.py",
    "test_proposals_unified.py",
    "test_reachability.py",
]


class TestInfrastructureMeta(unittest.TestCase):
    def test_every_suite_file_exists(self):
        here = os.path.dirname(os.path.abspath(__file__))
        missing = [name for name in SUITES
                   if not os.path.isfile(os.path.join(here, name))]
        self.assertEqual(missing, [], f"meta-suite lists missing files: {missing}")

    def test_all_infrastructure_suites_pass(self):
        here = os.path.dirname(os.path.abspath(__file__))
        failures: list[str] = []
        for name in SUITES:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest",
                 os.path.join(here, name), "-q", "-p", "no:warnings"],
                capture_output=True, text=True, timeout=600)
            if proc.returncode != 0:
                tail = "\n".join(
                    proc.stdout.splitlines()[-15:] + proc.stderr.splitlines()[-5:])
                failures.append(f"--- {name} (exit {proc.returncode}) ---\n{tail}")
        self.assertEqual(failures, [],
                         "\n\n".join(failures) or "all suites passed")


if __name__ == "__main__":
    unittest.main()
