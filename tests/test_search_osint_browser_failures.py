"""Tail-108 pins: nomorals/search/osint_browser.py census sites.

All four sites are exclusions — every failure degrades to absence
(no tool, not-ready, results already collected), never to a
fabricated hit:

* 36 — ``_browser()``: any registry failure -> None, the module's
  "unwired" value; both adapters then return []. TARGETED-FIX
  CANDIDATE (top of this batch): the pin below proves the guard is
  not hypothetical — ``nomorals.tools.registry`` exposes only the
  ``ToolRegistry`` class and has NO module-level ``get``, so
  ``_browser()`` returns None unconditionally today and both
  browser OSINT adapters (registered in search/sources.py) are
  dead organs. The real spine pattern is a registry INSTANCE
  called via ``registry.call("browser", ...)`` (see
  research/pipeline.py ``_browser_fetch_fallback``). The batch
  pins the behaviour; rewiring is a targeted pass, not a convert —
  the failure shape is absence, not a fabricated success.
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

def test_browser_is_none_with_the_real_registry_module():
    # Standing-state pin: the tools registry module has no
    # module-level ``get`` (only ToolRegistry instances do), so the
    # lookup raises AttributeError inside the guard and _browser()
    # reports "unwired". A future wiring fix SHOULD break this pin.
    assert not hasattr(reg, "get")
    assert ob._browser() is None


def test_browser_registry_crash_returns_none(monkeypatch):
    def _get(name):
        raise RuntimeError("registry boom")

    monkeypatch.setattr(reg, "get", _get, raising=False)
    assert ob._browser() is None


def test_browser_unwraps_registered_tool(monkeypatch):
    sentinel = lambda *a, **k: {}  # noqa: E731
    monkeypatch.setattr(
        reg, "get", lambda name: types.SimpleNamespace(func=sentinel),
        raising=False)
    assert ob._browser() is sentinel
    monkeypatch.setattr(reg, "get", lambda name: sentinel, raising=False)
    assert ob._browser() is sentinel
    monkeypatch.setattr(reg, "get", lambda name: None, raising=False)
    assert ob._browser() is None


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
    monkeypatch.setattr(ob, "_browser", lambda: _make_browse(close_raises))


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
