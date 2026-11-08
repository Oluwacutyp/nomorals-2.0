"""Installed-product smoke test (Reliability Batch A, audit F05).

The unit suite runs against the source tree, where ``__file__``-relative
data files are visible whether or not the packaging ships them — which
is how the wheel reached users with zero runtime assets and the i18n
loader returned an empty locale list. This test builds the wheel,
installs it into a clean venv with no repo on the path, and asserts the
installed product carries and can load its runtime data assets.

Hermetic: wheel build uses --no-build-isolation (the environment's own
setuptools), the venv install uses --no-index --no-deps. Skips honestly
when the sandbox cannot create a venv or build a wheel at all.

``nomorals/games/data/tournaments/t*.json`` is asserted ABSENT: those
files are runtime-written tournament state, not seed assets.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The audit-class asset inventory, relative to the nomorals/ package.
EXPECTED_ASSETS = (
    # i18n catalogs (16)
    *[f"i18n/catalogs/{loc}.tsv" for loc in (
        "ar", "de", "en", "es", "fr", "ha", "hi", "ig", "it", "ja",
        "nl", "pcm", "pt", "ru", "sw", "yo")],
    # media-edit workflows (6)
    *[f"media_edit/workflows/{name}.json" for name in (
        "img2img", "inpaint", "txt2img", "upscale", "wan22_i2v",
        "wan22_t2v")],
    # domain seed libraries (19)
    *[f"{pkg}/library.json" for pkg in (
        "assets", "commitments", "decisions", "focus", "household",
        "identity", "legacy", "life", "meta", "people", "preferences",
        "publicface", "pulses", "retrieval", "reviews", "scam", "shows",
        "sources", "media/upgrade")],
    # builder templates the scaffolder copies wholesale
    *[f"builders/templates/{kind}/requirements.txt" for kind in (
        "bot", "cli_tool", "dashboard", "rest_api", "telegram_bot",
        "webapp")],
    "builders/templates/dashboard/data.json",
    "builders/templates/bot/bot.py",
    # wisdom (8)
    "wisdom/data/manifest.json",
    "wisdom/data/timeline.json",
    *[f"wisdom/sessions/{name}.json" for name in (
        "astral_prep", "box_breathing", "four_seven_eight", "kapalabhati",
        "nadi_shodhana", "void_state")],
    # brain-eval data (5)
    "agents/braineval/data/listings.json",
    "agents/braineval/data/refusal_patterns.json",
    "agents/braineval/data/cases/capability.jsonl",
    "agents/braineval/data/cases/refusal.jsonl",
    "agents/braineval/data/cases/tools.jsonl",
    # game word seed, training seed, reverse-engineering pin
    "games/data/words.txt",
    "training/seeds/codebeast_seed.jsonl",
    "agents/reverse/pins/unblob_26.6.4.txt",
)

_PROBE = r"""
import json
import sys
from pathlib import Path

import nomorals

root = Path(nomorals.__file__).resolve().parent
if "site-packages" not in str(root):
    raise SystemExit(f"nomorals imported from {root}, not the clean venv")

expected = json.loads(sys.argv[1])
missing = [rel for rel in expected if not (root / rel).is_file()]
if missing:
    raise SystemExit(f"missing installed assets: {missing}")

tournaments = root / "games" / "data" / "tournaments"
if tournaments.is_dir() and list(tournaments.glob("t*.json")):
    raise SystemExit("runtime tournament state leaked into the wheel")

from nomorals.i18n.catalog import load_catalogs

locales = load_catalogs().locales()
if len(locales) != 16:
    raise SystemExit(f"installed i18n loader saw {len(locales)} locales")

from nomorals.builders.scaffold import list_templates

templates = list_templates()
if len(templates) != 6 or any(not t.files for t in templates):
    raise SystemExit("installed builder templates incomplete")

for rel in expected:
    if rel.endswith(".json"):
        json.loads((root / rel).read_text(encoding="utf-8"))
words = (root / "games" / "data" / "words.txt").read_text(
    encoding="utf-8").split()
if len(words) < 100:
    raise SystemExit("installed word seed is empty or truncated")
print("INSTALLED_SMOKE_OK", len(expected), "assets")
"""


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=600, **kw)


class InstalledProductSmokeTest(unittest.TestCase):
    def test_wheel_install_carries_runtime_assets(self) -> None:
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="pkg-smoke-"))
        wheels = tmp / "wheels"
        wheels.mkdir()
        build = _run([sys.executable, "-m", "pip", "wheel", str(REPO),
                      "--no-deps", "--no-build-isolation", "-w",
                      str(wheels)], cwd=tmp)
        wheel = next(wheels.glob("nomorals-*.whl"), None)
        if build.returncode != 0 or wheel is None:
            self.skipTest(
                "wheel build unavailable in this environment: "
                + (build.stderr or build.stdout)[-400:])

        venv = tmp / "venv"
        made = _run([sys.executable, "-m", "venv", str(venv)])
        if made.returncode != 0:
            self.skipTest("venv creation unavailable in this environment: "
                          + made.stderr[-400:])
        venv_python = venv / "bin" / "python"
        if not venv_python.exists():
            venv_python = venv / "Scripts" / "python.exe"
        install = _run([str(venv_python), "-m", "pip", "install",
                        "--no-index", "--no-deps", str(wheel)], cwd=tmp)
        if install.returncode != 0:
            self.fail("clean-venv wheel install failed: "
                      + (install.stderr or install.stdout)[-800:])

        probe = _run([str(venv_python), "-c", _PROBE,
                      json.dumps(list(EXPECTED_ASSETS))], cwd=tmp)
        self.assertEqual(probe.returncode, 0,
                         probe.stderr[-1200:] or probe.stdout[-1200:])
        self.assertIn("INSTALLED_SMOKE_OK", probe.stdout)


if __name__ == "__main__":
    unittest.main()
