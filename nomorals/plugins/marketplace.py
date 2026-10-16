"""Plugin marketplace: list, install, review, rate — with self-tests.

Skip-list correction: the marketplace is NOT gated by tests — tests
are PART of it. Installing a plugin runs the plugin's OWN test suite
in a sandbox FIRST, and rejects the install on failure:

* **Sandbox** — tests run in a subprocess with a timeout, a clean
  environment (no inherited proxy/credential env vars, empty
  PYTHONPATH), and no network (``http_proxy`` /
  ``https_proxy`` stripped; the runner never fetches).
* **Rejection** — failing tests, a timeout, or a missing test runner
  verdict all reject the install BEFORE anything is copied: nothing
  is installed, no receipt is written.
* **Install** — only after tests pass does the install delegate to
  :func:`nomorals.plugins.shipping.install_plugin` (manifest +
  permissions validation, receipts). The marketplace never
  re-implements shipping.

Reviews/ratings live beside the receipts: ``add_review``,
``average_rating``, ``listings`` (installed plugins + rating summary).
Reviews are data — one review per (plugin, reviewer), re-review
overwrites.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .errors import ManifestError
from .shipping import install_plugin as _ship_install
from .shipping import list_plugins as _ship_list
from .shipping import uninstall as _ship_uninstall

_log = get_logger("nomorals.plugins.marketplace")

__all__ = [
    "PluginTestFailed", "Marketplace", "run_plugin_tests",
    "TEST_TIMEOUT_S",
]

#: Default per-plugin test timeout (seconds).
TEST_TIMEOUT_S = 120.0

#: Env vars stripped from the test sandbox (no creds, no proxy, no net).
_STRIPPED_ENV = (
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN", "TELEGRAM_BOT_TOKEN",
)


class PluginTestFailed(Exception):
    """The plugin's own tests failed (or timed out) — install rejected."""


def _find_test_files(root: Path) -> list[Path]:
    tests_dir = root / "tests"
    found: list[Path] = []
    if tests_dir.is_dir():
        found += sorted(tests_dir.rglob("test_*.py"))
        found += sorted(tests_dir.rglob("*_test.py"))
    found += sorted(p for p in root.glob("test_*.py") if p.is_file())
    # Deduplicate, keep deterministic order.
    return sorted(set(found))


def run_plugin_tests(plugin_dir: str | Path, *,
                     timeout_s: float = TEST_TIMEOUT_S) -> dict[str, Any]:
    """Run a plugin's own tests in a sandboxed subprocess.

    Returns ``{"passed": bool, "detail": str, "tests_found": int}``.
    A plugin with NO test files passes vacuously (``tests_found=0``)
    but the result says so honestly — the marketplace records it.
    Raises :exc:`PluginTestFailed` on failure / timeout.
    """
    root = Path(plugin_dir)
    tests = _find_test_files(root)
    if not tests:
        return {"passed": True, "detail": "no test files found (vacuous pass)",
                "tests_found": 0}
    env = {k: v for k, v in os.environ.items() if k not in _STRIPPED_ENV}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # NOTE: PYTHONNOUSERSITE deliberately NOT set - pytest lives in user site.
    env["PYTHONPATH"] = ""  # no caller paths leak into the sandbox
    # No -I / PYTHONNOUSERSITE: isolated mode would strip the user site
    # pytest itself lives in. Empty PYTHONPATH + cwd pinned to the plugin
    # dir + no creds/proxies + timeout is the sandbox boundary.
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short",
           *[str(t) for t in tests]]
    try:
        proc = subprocess.run(cmd, cwd=str(root), env=env,
                              capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        raise PluginTestFailed(
            f"plugin tests timed out after {timeout_s}s — install rejected")
    tail = (proc.stdout + proc.stderr)[-2000:]
    if proc.returncode != 0:
        raise PluginTestFailed(
            f"plugin's own tests failed (exit {proc.returncode}): {tail}")
    return {"passed": True, "detail": tail[-500:], "tests_found": len(tests)}


class Marketplace:
    """Marketplace over the shipping path + reviews/ratings."""

    def __init__(self, plugins_dir: str | Path | None = None) -> None:
        self.plugins_dir = Path(plugins_dir) if plugins_dir else (Path.cwd() / "plugins")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        self._reviews_path = self.plugins_dir / "marketplace_reviews.json"

    # ── reviews / ratings ──────────────────────────────────────────────────
    def _load_reviews(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self._reviews_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _save_reviews(self, reviews: dict[str, dict[str, Any]]) -> None:
        self._reviews_path.write_text(json.dumps(reviews, indent=1, sort_keys=True),
                                      encoding="utf-8")

    def add_review(self, plugin: str, reviewer: str, rating: int,
                   text: str = "") -> dict[str, Any]:
        """Add/overwrite one review. Rating must be 1–5."""
        rating = int(rating)
        if not 1 <= rating <= 5:
            raise ValueError("rating must be 1-5")
        reviews = self._load_reviews()
        entry = reviews.setdefault(plugin, {"reviews": []})
        entry["reviews"] = [r for r in entry["reviews"] if r.get("reviewer") != reviewer]
        entry["reviews"].append({"reviewer": reviewer, "rating": rating,
                                 "text": text, "at": time.time()})
        self._save_reviews(reviews)
        return {"plugin": plugin, "reviewer": reviewer, "rating": rating}

    def average_rating(self, plugin: str) -> float | None:
        reviews = self._load_reviews().get(plugin, {}).get("reviews", [])
        if not reviews:
            return None
        return round(sum(r["rating"] for r in reviews) / len(reviews), 2)

    def reviews(self, plugin: str) -> list[dict[str, Any]]:
        return list(self._load_reviews().get(plugin, {}).get("reviews", []))

    # ── list / install / uninstall ─────────────────────────────────────────
    def listings(self) -> list[dict[str, Any]]:
        """Installed plugins + their rating summary."""
        out = []
        for receipt in _ship_list(self.plugins_dir):
            name = receipt.get("name", "")
            out.append({**receipt, "average_rating": self.average_rating(name),
                        "review_count": len(self.reviews(name))})
        return out

    def install(self, path: str | Path, *, overwrite: bool = False,
                run_tests: bool = True,
                timeout_s: float = TEST_TIMEOUT_S) -> dict[str, Any]:
        """Install a plugin: own-tests first, shipping install second.

        ``run_tests=False`` is the escape hatch for plugins that
        legitimately ship no tests — the receipt records
        ``tests_run=False`` honestly. With tests present and failing,
        :exc:`PluginTestFailed` is raised and NOTHING is installed.
        """
        src = Path(path)
        if not src.exists():
            raise ManifestError(f"plugin path does not exist: {path!r}")
        test_result: dict[str, Any] = {"passed": None, "detail": "tests skipped",
                                       "tests_found": 0}
        if run_tests:
            test_result = run_plugin_tests(src, timeout_s=timeout_s)
        receipt = _ship_install(src, plugins_dir=self.plugins_dir, overwrite=overwrite)
        receipt["tests_run"] = bool(run_tests)
        receipt["tests_found"] = test_result.get("tests_found", 0)
        receipt["tests_passed"] = test_result.get("passed")
        return receipt

    def uninstall(self, name: str, **kwargs: Any) -> bool:
        return _ship_uninstall(name, plugins_dir=self.plugins_dir, **kwargs)
