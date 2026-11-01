"""Tail-85 pins for nomorals/agents/weather.py — 5 exclusions, 0 converts.

Each test pins one census site's fallback behavior plus a control half
proving the except path was taken. The file itself contains no
exception handlers, so it adds zero census sites, and no sys.path
manipulation — tests bind via rootdir from the tree root.

NOT-recording is pinned with the snapshot method: weather.py calls no
failures-ledger record function anywhere (only _log), so both the
agency and agents ledgers must be unchanged across each trigger.

Verdict notes (full grounds in SURVEY.md):
- 895: module-level import of morning_briefing's _Provider /
  BriefingSection -> local fallback classes. The designed standalone
  ladder ("standalone use without the briefing"): the providers
  still build sections; with the real briefing importable they
  subclass the real base (control half, in-process).
- 933 / 972: provider collect() fetch failure -> None. None is the
  _Provider contract's "no section" value — BriefingComposer
  appends only non-None sections, and the composer's own guard
  treats a raising provider identically (warn + skip). The
  unguarded not-ok sibling path in the same methods returns None
  too, so failure == designed value by construction; no weather
  section is ever fabricated.
- 943: forecast enrichment failure -> pass. The section still ships
  with the now-line already collected; the forecast line is an
  optional second line of the same section.
- 967: news fetch failure -> pass ("news optional" at the site).
  news_items stays [] and the overview is called with None; the
  USA section ships without the optional news input.
"""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.agents import weather as wx


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def test_895_briefing_import_fallback_classes():
    before = _snap()
    code = (
        "import sys\n"
        "sys.modules['nomorals.agents.morning_briefing'] = None\n"
        "from nomorals.agents import weather as w\n"
        # fallback classes are defined IN weather.py (standalone use)
        "assert w._Provider.__module__ == 'nomorals.agents.weather'\n"
        "assert w.BriefingSection.__module__ == 'nomorals.agents.weather'\n"
        "assert issubclass(w.WeatherProvider, w._Provider)\n"
        "sec = w.BriefingSection(name='x', title='X', priority=1)\n"
        "assert sec.items == [] and sec.lines == []\n"
        "print('FALLBACK_OK')\n"
    )
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root))
    if Path(sys.executable).name == "pycopy":
        # a copied interpreter binary needs PYTHONHOME to find its
        # stdlib (the verify harness runs tests under such a copy)
        env["PYTHONHOME"] = "/usr"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=str(root), env=env,
        capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "FALLBACK_OK" in out.stdout
    # control in-process: with the real briefing importable, the
    # providers subclass the REAL base
    import nomorals.agents.morning_briefing as mb
    assert wx._Provider is mb._Provider
    assert _snap() == before


class _StubWeather:
    now_exc = None
    fc_exc = None
    now_res = {"ok": True, "text": "Sunny 20C in New York"}
    fc_res = {"ok": True, "text": "Today\nRain tomorrow"}

    def __init__(self, default_place=None):
        pass

    def now(self, place):
        if type(self).now_exc is not None:
            raise type(self).now_exc
        return dict(type(self).now_res)

    def forecast(self, place, days=1):
        if type(self).fc_exc is not None:
            raise type(self).fc_exc
        return dict(type(self).fc_res)


def _reset_stub():
    _StubWeather.now_exc = None
    _StubWeather.fc_exc = None


def test_933_weather_now_failure_is_no_section():
    before = _snap()
    _reset_stub()
    _StubWeather.now_exc = RuntimeError("now boom")
    with mock.patch.object(wx, "Weather", _StubWeather):
        sec = wx.WeatherProvider().collect(SimpleNamespace(), 0.0)
    assert sec is None
    _reset_stub()
    with mock.patch.object(wx, "Weather", _StubWeather):
        sec = wx.WeatherProvider().collect(SimpleNamespace(), 0.0)
    assert sec is not None and sec.name == "weather"
    assert any("Sunny" in ln for ln in sec.lines)
    assert _snap() == before


def test_943_forecast_failure_keeps_now_section():
    before = _snap()
    _reset_stub()
    _StubWeather.fc_exc = RuntimeError("forecast boom")
    with mock.patch.object(wx, "Weather", _StubWeather):
        sec = wx.WeatherProvider().collect(SimpleNamespace(), 0.0)
    # the section still ships with the now-line only
    assert sec is not None
    assert len(sec.lines) == 1 and "Sunny" in sec.lines[0]
    _reset_stub()
    with mock.patch.object(wx, "Weather", _StubWeather):
        sec = wx.WeatherProvider().collect(SimpleNamespace(), 0.0)
    assert sec is not None and len(sec.lines) == 2
    assert _snap() == before


class _StubUSA:
    exc = None
    seen = []
    res = {"ok": True, "text": "USA\nAlerts line",
           "top_events": [{"event": "Storm"}]}

    def overview(self, news_items=None):
        type(self).seen.append(news_items)
        if type(self).exc is not None:
            raise type(self).exc
        return dict(type(self).res)


class _BoomNews:
    def __init__(self, ctx):
        raise RuntimeError("news boom")


class _OkNews:
    def __init__(self, ctx):
        pass

    def recent(self, limit=40):
        return [{"title": "T1", "summary": "S1"}]


def test_967_news_failure_is_optional_input():
    before = _snap()
    _StubUSA.exc = None
    _StubUSA.seen = []
    with mock.patch.object(wx, "USASituations", _StubUSA):
        with mock.patch("nomorals.agents.news.NewsAgent", _BoomNews):
            sec = wx.USASituationsProvider().collect(
                SimpleNamespace(), 0.0)
    # news is optional: the section ships, overview saw news_items=None
    assert sec is not None and sec.name == "usa"
    assert _StubUSA.seen == [None]
    # control: working news reaches the overview as items
    _StubUSA.seen = []
    with mock.patch.object(wx, "USASituations", _StubUSA):
        with mock.patch("nomorals.agents.news.NewsAgent", _OkNews):
            sec = wx.USASituationsProvider().collect(
                SimpleNamespace(), 0.0)
    assert sec is not None
    assert _StubUSA.seen == [[{"title": "T1", "summary": "S1"}]]
    assert _snap() == before


def test_972_usa_overview_failure_is_no_section():
    before = _snap()
    _StubUSA.exc = RuntimeError("overview boom")
    _StubUSA.seen = []
    with mock.patch.object(wx, "USASituations", _StubUSA):
        with mock.patch("nomorals.agents.news.NewsAgent", _OkNews):
            sec = wx.USASituationsProvider().collect(
                SimpleNamespace(), 0.0)
    assert sec is None
    _StubUSA.exc = None
    with mock.patch.object(wx, "USASituations", _StubUSA):
        with mock.patch("nomorals.agents.news.NewsAgent", _OkNews):
            sec = wx.USASituationsProvider().collect(
                SimpleNamespace(), 0.0)
    assert sec is not None and sec.name == "usa"
    assert _snap() == before
