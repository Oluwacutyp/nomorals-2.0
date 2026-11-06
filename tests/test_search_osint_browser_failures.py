"""Tail-108 pins: nomorals/search/osint_browser.py census sites.

Every failure degrades to absence (no tool, not-ready, results
already collected), never to a fabricated hit:

* 36 — ``_browser()``: CONVERTED by the Pass 3 targeted wiring.
  The original pin proved the guard was not hypothetical —
  ``nomorals.tools.registry`` exposes only the ``ToolRegistry``
  class and has NO module-level ``get``, so the old lookup raised
  AttributeError on every call, ``_browser()`` returned None
  unconditionally, and both browser OSINT adapters (registered in
  search/sources.py) were dead organs. ``_browser()`` now resolves
  the tool on a real registry INSTANCE — the search context's own
  in production (threaded via sources.build_osint_adapter /
  federated_search), a lazily built standalone registry otherwise —
  and drives it via ``registry.call("browser", ...)``, the
  research/pipeline.py ``_browser_fetch_fallback`` pattern. The
  pins below pin the wired contract: no browser tool on the
  registry -> None; a failed standalone build -> None; calls
  unwrap the Outcome to the tool's result dict, with an Err
  outcome becoming an ``ok: False`` dict.
* 44 — ``_rendered_ok()``: import/probe crash -> (False,
  "browser tool not importable") — fail-closed with the reason
  carried in-band.
* 117 — people adapter: ``browse("close")`` teardown raising after
  results are collected changes nothing; the hits stand.
* 183 — site adapter: the identical close teardown.

At 117/183 the swallow binds nothing and logs nothing, while both
sibling per-engine guards in the same functions bind ``exc`` and
``_log.debug`` it — that asymmetry is the shelved observability
sketch, not a convert.
"""

import types

from nomorals.search import osint_browser as ob
from nomorals.tools import browser as browser_mod
from nomorals.tools import registry as reg


# ── 36: _browser() ───────────────────────────────────────────────

def test_browser_is_none_when_registry_has_no_browser_tool():
    # Converted standing-state pin: the unwired case is now a real
    # registry instance with no "browser" tool registered on it.
    assert not hasattr(reg, "get")  # the old lookup target: still absent
    assert ob._browser(reg.ToolRegistry()) is None


def test_browser_registry_build_crash_returns_none(monkeypatch):
    # A failed standalone-registry build degrades to "unwired".
    def _boom(self):
        raise RuntimeError("registry boom")

    monkeypatch.setattr(reg.ToolRegistry, "register_builtins", _boom)
    monkeypatch.setattr(ob, "_registry_cache", None, raising=False)
    assert ob._browser() is None


def test_browser_calls_through_registry_instance():
    # The returned callable drives registry.call("browser", ...) and
    # unwraps the Outcome to the tool's result dict.
    calls = []

    class _Reg:
        def get(self, name):
            return object() if name == "browser" else None

        def call(self, name, actor="system", **kw):
            calls.append((name, actor, kw))
            return types.SimpleNamespace(
                ok=True, value={"text": "hi"}, error=None)

    browse = ob._browser(_Reg())
    assert browse is not None
    assert browse("text", engine="rendered", session="osint") == {"text": "hi"}
    assert calls == [("browser", "system",
                      {"action": "text", "engine": "rendered",
                       "session": "osint"})]

    class _ErrReg(_Reg):
        def call(self, name, actor="system", **kw):
            return types.SimpleNamespace(
                ok=False, value=None,
                error=types.SimpleNamespace(message="nope"))

    # an Err outcome becomes an ok:False dict, never an exception
    assert ob._browser(_ErrReg())("open", url="https://x") == {
        "ok": False, "error": "nope"}


# ── 44: _rendered_ok() ───────────────────────────────────────────

def test_rendered_ok_crash_fails_closed_with_reason(monkeypatch):
    def _pa():
        raise RuntimeError("playwright boom")

    monkeypatch.setattr(browser_mod, "_playwright_available", _pa)
    assert ob._rendered_ok() == (False, "browser tool not importable")


def test_rendered_ok_passes_through_probe_answer(monkeypatch):
    monkeypatch.setattr(
        browser_mod, "_playwright_available", lambda: (True, "ok"))
    assert ob._rendered_ok() == (True, "ok")


# ── 117 / 183: close teardown ────────────────────────────────────

def _make_browse(close_raises):
    def browse(action, **kw):
        if action == "open":
            return {"ok": True}
        if action == "wait":
            return {}
        if action == "text":
            return {"text": "profile https://example.com/jane here",
                    "url": "https://webmii.com/people?n=Jane+Doe"}
        if action == "extract":
            return {"links": [{
                "href": "https://web.archive.org/web/2020/https://example.com/",
                "text": "cap"}]}
        if action == "close":
            if close_raises:
                raise RuntimeError("close boom")
            return {}
        return {}

    return browse


def _wire(monkeypatch, close_raises):
    monkeypatch.setattr(ob, "_rendered_ok", lambda: (True, ""))
    # _browser gained the registry parameter in the Pass 3 wiring;
    # the seam's shape (module function the adapters consult) is
    # unchanged.
    monkeypatch.setattr(
        ob, "_browser", lambda registry=None: _make_browse(close_raises))


def test_people_adapter_close_crash_keeps_collected_hits(monkeypatch):
    _wire(monkeypatch, close_raises=True)
    hits = ob.BrowserPeopleSearchAdapter().search("Jane Doe", limit=3)
    assert len(hits) >= 1
    _wire(monkeypatch, close_raises=False)
    hits_ok = ob.BrowserPeopleSearchAdapter().search("Jane Doe", limit=3)
    assert len(hits_ok) == len(hits)


def test_site_adapter_close_crash_keeps_collected_hits(monkeypatch):
    _wire(monkeypatch, close_raises=True)
    hits = ob.BrowserSiteSearchAdapter().search("some query", limit=3)
    assert len(hits) >= 1
    _wire(monkeypatch, close_raises=False)
    hits_ok = ob.BrowserSiteSearchAdapter().search("some query", limit=3)
    assert len(hits_ok) == len(hits)
