"""Batch E (façade hardening) regression tests — WHOLE_SYSTEM_AUDIT F13–F16
plus the runtime_media ``_log`` error-path defect.

Every test drives the public façade / ordinary code path named by the
finding, not an isolated helper where the audit asked for the façade:

- F13: ``RenderedTab.reload`` — one authoritative definition; the daemon's
  keyword call shape must work; duplicate-method detection over the
  touched modules.
- F14: ``self_heal`` probe — the timeout must bound the CALLER's wait,
  outstanding probe workers are budgeted, a hung kind is backed off
  instead of re-probed back-to-back, and probe threads are daemons so
  a stuck command can never block process shutdown.
- F15: ``tools.pytest_runner.run_tests`` — the unittest fallback with
  pytest forced absent: selected files, Windows-style paths, node
  selectors, and the honest nothing-ran shape for pytest-style files.
- F16: ``SmartHomeIntegration.browse_media`` — resolves the entity it
  was given and returns the service response.
- runtime_media ``_produce_hook`` — a throwing producer (and a failing
  fingerprint index) must yield the intended error reply, never a
  secondary ``NameError`` from the undefined ``_log``.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# F13 — single authoritative browser reload
# ---------------------------------------------------------------------------

from nomorals.browser import service as S  # noqa: E402


class _FakePage:
    """Playwright-page stand-in: records reload options."""

    def __init__(self, url: str = "http://x/") -> None:
        self.url = url
        self.reload_calls: list[dict] = []

    def title(self) -> str:
        return "X Title"

    def reload(self, **kwargs):
        self.reload_calls.append(kwargs)


def _tab(tmp_path) -> S.RenderedTab:
    tab = S.RenderedTab("t1", "s", tmp_path / "state.json", object())
    tab._page = _FakePage()
    tab.url = "http://x/"
    tab.title = "X Title"  # a loaded tab carries its current title
    return tab


class TestReloadFacade:
    def test_single_reload_with_documented_signature(self):
        defs = [n for n in S.RenderedTab.__dict__.values()
                if getattr(n, "__name__", "") == "reload"]
        assert len(defs) == 1
        params = inspect.signature(S.RenderedTab.reload).parameters
        assert "wait_until" in params and "timeout" in params

    def test_default_reload_drives_driver_and_returns_result(self, tmp_path):
        tab = _tab(tmp_path)
        out = tab.reload()
        assert tab._page.reload_calls == [
            {"wait_until": "domcontentloaded", "timeout": 60_000}]
        assert out["ok"] is True
        assert out["url"] == "http://x/"
        assert out["title"] == "X Title"
        assert out["tab_id"] == "t1"

    def test_keyword_reload_passes_options_through(self, tmp_path):
        # The browser daemon's r_reload op calls exactly this shape; on
        # the shadowed definition it raised TypeError.
        tab = _tab(tmp_path)
        out = tab.reload(wait_until="load", timeout=1)
        assert tab._page.reload_calls == [{"wait_until": "load", "timeout": 1}]
        assert out["ok"] is True

    def test_reload_rejects_unknown_wait_until(self, tmp_path):
        tab = _tab(tmp_path)
        with pytest.raises(S.BrowserError):
            tab.reload(wait_until="whenever")


_TOUCHED_MODULES = [
    "nomorals/browser/service.py",
    "nomorals/browser/daemon.py",
    "nomorals/browser/challenge.py",
    "nomorals/browser/takeover.py",
    "nomorals/core/self_heal.py",
    "nomorals/tools/pytest_runner.py",
    "nomorals/integrations/smarthome_integration.py",
    "nomorals/agents/partner/runtime_media.py",
]

#: Reviewed duplicate-method exceptions: name -> why the override is an
#: interface-preserving extension, not a shadowing. Empty today — the
#: F13 scan found exactly one duplicate (RenderedTab.reload) and it was
#: the defect. Add an entry here ONLY with a written justification.
DUPLICATE_METHOD_EXCEPTIONS: dict[str, str] = {}


def test_no_duplicate_method_definitions_in_touched_modules():
    offenders = []
    for rel in _TOUCHED_MODULES:
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            seen: dict[str, int] = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    seen[item.name] = seen.get(item.name, 0) + 1
            for name, count in seen.items():
                key = f"{rel}:{node.name}.{name}"
                if count > 1 and key not in DUPLICATE_METHOD_EXCEPTIONS:
                    offenders.append(f"{key} x{count}")
    assert offenders == []


# ---------------------------------------------------------------------------
# F14 — bounded self-heal probe supervision
# ---------------------------------------------------------------------------

from nomorals.core import self_heal as sh  # noqa: E402


@pytest.fixture(autouse=True)
def _probe_state_isolation():
    cooldowns = getattr(sh, "_probe_cooldowns", None)
    if isinstance(cooldowns, dict):
        cooldowns.clear()
    yield
    if isinstance(cooldowns, dict):
        cooldowns.clear()


def _parking_handler(park: threading.Event, calls: list):
    def handler(text, chat_key="", message=None):
        calls.append(text)
        park.wait(3)  # backstop; the release sets it sooner in budget tests
        return "a real answer"
    return handler


class TestBoundedProbe:
    def test_timeout_bounds_the_callers_wait(self, monkeypatch):
        monkeypatch.setattr(sh, "_PROBE_TIMEOUT", 0.05)
        park = threading.Event()  # never set during the caller's wait
        calls: list = []
        start = time.monotonic()
        rep = sh._probe_inner(_parking_handler(park, calls), "batche-hang")
        elapsed = time.monotonic() - start
        park.set()
        assert rep.timed_out is True
        assert rep.broken is True and rep.exc is None
        # The caller must return at the timeout, not when the command
        # eventually finishes (the old with-executor shape waited ~3s).
        assert elapsed < 1.5

    def test_probe_threads_are_daemons(self, monkeypatch):
        monkeypatch.setattr(sh, "_PROBE_TIMEOUT", 0.05)
        park = threading.Event()
        calls: list = []
        rep = sh._probe_inner(_parking_handler(park, calls), "batche-daemon")
        workers = [t for t in threading.enumerate()
                   if t.name.startswith("heal-probe")]
        park.set()
        assert rep.timed_out is True
        assert workers, "probe worker thread not found"
        assert all(t.daemon for t in workers)

    def test_hung_kind_is_backed_off_not_reprobed(self, monkeypatch):
        monkeypatch.setattr(sh, "_PROBE_TIMEOUT", 0.05)
        park = threading.Event()
        calls: list = []
        handler = _parking_handler(park, calls)
        first = sh._probe_inner(handler, "batche-backoff")
        assert first.timed_out is True and len(calls) == 1
        start = time.monotonic()
        second = sh._probe_inner(handler, "batche-backoff")
        elapsed = time.monotonic() - start
        park.set()
        assert second.timed_out is True
        assert second.backed_off is True
        assert len(calls) == 1  # no second spawn during cooldown
        assert elapsed < 1.0

    def test_outstanding_probe_budget_is_enforced(self, monkeypatch):
        monkeypatch.setattr(sh, "_PROBE_TIMEOUT", 0.05)
        monkeypatch.setattr(sh, "_PROBE_MAX_OUTSTANDING", 1)
        # Drain any worker a neighbouring test left parked (they free
        # their slots as soon as their handlers return).
        deadline = time.monotonic() + 5
        while getattr(sh, "_probe_outstanding", 0) and \
                time.monotonic() < deadline:
            time.sleep(0.05)
        release = threading.Event()
        calls: list = []
        outcome: dict = {}

        def run_hung():
            outcome["rep"] = sh._probe_inner(
                _parking_handler(release, calls), "batche-budget-a")

        t = threading.Thread(target=run_hung)
        t.start()
        t.join(5)
        assert outcome["rep"].timed_out is True
        # The hung command's worker still holds the only slot: a new
        # probe must fail honestly (machinery report), not queue a
        # second unbounded worker.
        rep = sh._probe_inner(lambda *a, **k: "fine", "batche-budget-b")
        assert rep.broken is True and rep.exc is None
        assert rep.err.startswith("probe machinery failed: RuntimeError: "
                                  "probe budget exhausted")
        release.set()
        # Once the hung command returns, its slot frees and probing
        # works again (poll: the worker releases asynchronously).
        deadline = time.monotonic() + 5
        ok = None
        while time.monotonic() < deadline:
            ok = sh._probe_inner(lambda *a, **k: "a real answer",
                                 "batche-budget-b")
            if not ok.broken:
                break
            time.sleep(0.05)
        assert ok is not None and ok.broken is False

    def test_healthy_and_raising_controls_unchanged(self, monkeypatch):
        monkeypatch.setattr(sh, "_PROBE_TIMEOUT", 2.0)
        healthy = sh._probe_inner(lambda *a, **k: "a real answer",
                                  "batche-healthy")
        assert (healthy.broken, healthy.exc, healthy.err) == (False, None, "")

        def boom(text, chat_key="", message=None):
            raise ValueError("kaput")

        rep = sh._probe_inner(boom, "batche-raising")
        assert rep.broken is True
        assert isinstance(rep.exc, ValueError)
        assert rep.err == "ValueError: kaput"
        assert rep.timed_out is False


# ---------------------------------------------------------------------------
# F15 — pytest_runner unittest fallback
# ---------------------------------------------------------------------------

from nomorals.tools import pytest_runner as pr  # noqa: E402


def _fixture_repo(tmp_path) -> Path:
    root = tmp_path / "repo"
    (root / "tests").mkdir(parents=True)
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "test_sample.py").write_text(textwrap.dedent("""
        import unittest


        class TestSample(unittest.TestCase):
            def test_one(self):
                self.assertEqual(1, 1)

            def test_two(self):
                self.assertTrue(True)
        """))
    (root / "tests" / "test_funcs.py").write_text(
        "def test_plain():\n    assert True\n")
    return root


@pytest.fixture()
def no_pytest(monkeypatch):
    monkeypatch.setattr(pr, "_has_pytest", lambda: False)


class TestUnittestFallback:
    def test_path_to_module_shapes(self):
        assert pr._path_to_module("tests/test_mod.py") == "tests.test_mod"
        assert pr._path_to_module("tests\\test_mod.py") == "tests.test_mod"
        assert pr._path_to_module("./tests/test_mod.py") == "tests.test_mod"
        assert (pr._path_to_module("tests/test_mod.py::TestX::test_y")
                == "tests.test_mod.TestX.test_y")
        assert (pr._path_to_module("tests/test_mod.py::TestX::test_y[p1]")
                == "tests.test_mod.TestX.test_y")

    def test_unittest_cmd_empty_selection(self, tmp_path):
        bare = tmp_path / "bare"
        bare.mkdir()
        assert pr._unittest_cmd(bare, []) is None
        assert pr._unittest_cmd(bare, None) is None

    def test_fallback_runs_selected_file(self, no_pytest, tmp_path):
        root = _fixture_repo(tmp_path)
        res = pr.run_tests(paths=["tests/test_sample.py"], repo=str(root),
                           timeout=60)
        assert res["runner"] == "unittest"
        assert res["ok"] is True
        assert res["ran"] == 2 and res["passed"] == 2

    def test_fallback_windows_style_path(self, no_pytest, tmp_path):
        root = _fixture_repo(tmp_path)
        res = pr.run_tests(paths=["tests\\test_sample.py"], repo=str(root),
                           timeout=60)
        assert res["ok"] is True and res["ran"] == 2

    def test_fallback_node_selector(self, no_pytest, tmp_path):
        root = _fixture_repo(tmp_path)
        res = pr.run_tests(paths=["tests/test_sample.py::TestSample::test_one"],
                           repo=str(root), timeout=60)
        assert res["ok"] is True
        assert res["ran"] == 1 and res["passed"] == 1

    def test_fallback_pytest_style_file_is_honest_nothing_ran(
            self, no_pytest, tmp_path):
        # unittest cannot execute plain pytest functions; the fallback
        # must say nothing ran instead of pretending a pass (F04/F15).
        root = _fixture_repo(tmp_path)
        res = pr.run_tests(paths=["tests/test_funcs.py"], repo=str(root),
                           timeout=60)
        assert res["ok"] is True
        assert res["ran"] == 0 and res["nothing_ran"] is True
        assert "no tests collected" in (res.get("note") or "")


# ---------------------------------------------------------------------------
# F16 — smart-home browse_media resolves its own parameter
# ---------------------------------------------------------------------------

from nomorals.integrations.smarthome_integration import (  # noqa: E402
    SmartHomeIntegration,
)


def _browse_integration(calls: list):
    integration = SmartHomeIntegration(MagicMock(), MagicMock())
    states = [{
        "entity_id": "media_player.living_room",
        "attributes": {"friendly_name": "Living Room Speaker"},
    }]

    async def fake_ha(method, endpoint, payload=None):
        calls.append((method, endpoint, payload))
        if endpoint == "states":
            return states
        return {"response": {"children": ["album-a", "album-b"]}}

    integration._ha_request = fake_ha
    integration._detect_backend = lambda: "homeassistant"
    return integration


class TestBrowseMedia:
    def test_browses_the_named_entity_and_returns_response(self):
        calls: list = []
        integration = _browse_integration(calls)
        out = asyncio.run(integration.browse_media(
            "Living Room Speaker", media_content_type="music",
            media_content_id="library://x"))
        svc = [c for c in calls if c[1].startswith("services/")]
        assert len(svc) == 1
        assert svc[0][1] == ("services/media_player/browse_media"
                             "?return_response=true")
        assert svc[0][2]["entity_id"] == "media_player.living_room"
        assert svc[0][2]["media_content_type"] == "music"
        assert svc[0][2]["media_content_id"] == "library://x"
        assert out == {"response": {"children": ["album-a", "album-b"]}}

    def test_browses_by_entity_id_directly(self):
        calls: list = []
        integration = _browse_integration(calls)
        asyncio.run(integration.browse_media("media_player.living_room"))
        svc = [c for c in calls if c[1].startswith("services/")]
        assert svc[0][2]["entity_id"] == "media_player.living_room"


# ---------------------------------------------------------------------------
# runtime_media _log — producer errors must not become NameErrors
# ---------------------------------------------------------------------------

import importlib  # noqa: E402

from nomorals.agents.partner import runtime_media  # noqa: E402
from nomorals.agents.partner.runtime_media import RuntimeMediaMixin  # noqa: E402

fingerprint_mod = importlib.import_module(  # noqa: E402
    "nomorals.audio.fingerprint")
import nomorals.media.producer as producer_mod  # noqa: E402
import nomorals.media.taste as taste_mod  # noqa: E402


def _produce_rig(monkeypatch, producer_cls):
    store = SimpleNamespace(profile=SimpleNamespace(last_production=None))
    monkeypatch.setattr(taste_mod, "load_taste", lambda ctx=None: store)
    monkeypatch.setattr(taste_mod, "detect_produce_intent",
                        lambda text: "a synthwave beat")
    monkeypatch.setattr(taste_mod, "mood_hint", lambda text: ())
    monkeypatch.setattr(producer_mod, "Producer", producer_cls)
    mixin = RuntimeMediaMixin()
    mixin.context = SimpleNamespace()
    return mixin


class TestProduceHookErrorPath:
    def test_module_logger_exists(self):
        assert hasattr(runtime_media, "_log")

    def test_throwing_producer_returns_error_reply(self, monkeypatch):
        class ExplodingProducer:
            def __init__(self, ctx):
                pass

            def produce(self, ref):
                raise RuntimeError("producer exploded")

        mixin = _produce_rig(monkeypatch, ExplodingProducer)
        out = mixin._produce_hook(SimpleNamespace(text="make a synthwave beat"))
        assert out == "produce error: producer exploded"

    def test_failing_fingerprint_index_still_delivers_reply(self, monkeypatch):
        class GoodProducer:
            def __init__(self, ctx):
                pass

            def produce(self, ref):
                return {"ok": True, "title": "Neon Run", "bpm": 128.0,
                        "key": "A", "mode": "minor", "notes": "driving",
                        "path": "/tmp/neon.wav", "profile": None}

        def bad_index(*a, **k):
            raise RuntimeError("index down")

        monkeypatch.setattr(fingerprint_mod, "index_own", bad_index)
        mixin = _produce_rig(monkeypatch, GoodProducer)
        msg = SimpleNamespace(text="make a synthwave beat",
                              chat=SimpleNamespace(key=""))
        out = mixin._produce_hook(msg)
        assert out is not None and "Neon Run" in out
