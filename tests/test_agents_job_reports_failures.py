"""Pass tail-96 pins: nomorals/agents/job_reports.py — 4 sites, all exclusions.

217 / 223 — ``_system_universes``: each lazy cross-layer import is
guarded separately; a failed import leaves that universe empty.
EXCLUSION: the digest degrades to counts only (``_cycle_section``
prints the bare count when the total is unknown, and
``_stood_down_names`` returns [] on an empty universe) — the module's
stated rule is "it never invents a total". Names are context, not
content; the counts in the report come from the cycle value itself.

529 — ``job_report_doc``: the fallback Doc's own construction
(``str(value)`` for overflow_text) crashes -> a bare title-only Doc.
EXCLUSION: last-resort render guard inside a "Never raises"
presentation function. The PRIMARY builder failure one frame up is
already recorded (``_log.debug`` with exc_info — that handler is not
a census site); this handler only covers a value so hostile that
even stringifying it for the fallback explodes. What ships is a
minimal but truthful Doc (right title, ✅, no invented summary).

673 — ``summarize_last_result`` crash -> "". EXCLUSION: a "Never
raises" view helper producing one display line for status views;
failure yields an empty line, never a broken view or a fabricated
outcome line.
"""

import sys
import types
from collections.abc import Mapping

from nomorals.agency.expansion import EXPANSION_SYSTEMS
from nomorals.agency.systems import SYSTEM_NAMES
from nomorals.agents import job_reports as jr


# ── 217 / 223: one universe's import fails, the other survives ───

def _broken_module(name):
    mod = types.ModuleType(name)

    def _getattr(attr):
        raise RuntimeError(f"{name} unavailable (injected)")
    mod.__getattr__ = _getattr
    return mod


def test_217_core_universe_import_failure_degrades_to_counts(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.agency.systems",
                        _broken_module("nomorals.agency.systems"))
    core, expansion = jr._system_universes()
    assert core == ()
    # the other guard is independent: expansion names still load
    assert expansion == tuple(str(n) for n in EXPANSION_SYSTEMS)


def test_223_expansion_universe_import_failure_degrades_to_counts(
        monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.agency.expansion",
                        _broken_module("nomorals.agency.expansion"))
    core, expansion = jr._system_universes()
    assert expansion == ()
    assert core == tuple(str(n) for n in SYSTEM_NAMES)


def test_217_223_universes_load_normally_as_control():
    core, expansion = jr._system_universes()
    assert core == tuple(str(n) for n in SYSTEM_NAMES)
    assert expansion == tuple(str(n) for n in EXPANSION_SYSTEMS)
    # and with no universe, stood-down names are never guessed
    assert jr._stood_down_names((), ["a"], 1) == []


# ── 529: fallback Doc construction crash -> bare Doc ─────────────

class _ExplodingMap(Mapping):
    def __getitem__(self, key):
        raise RuntimeError("hostile mapping")

    def __iter__(self):
        raise RuntimeError("hostile mapping")

    def __len__(self):
        return 0

    def __str__(self):
        raise RuntimeError("hostile mapping")


def test_529_hostile_value_yields_bare_fallback_doc():
    doc = jr.job_report_doc("zzz_tool", _ExplodingMap())
    assert doc.title == "scheduled: zzz_tool"
    assert doc.icon == "✅"
    assert doc.tone == "ok"
    assert doc.summary == ""


def test_529_mid_tier_fallback_keeps_summary_and_overflow():
    def bad_builder(tool, value, *, title):
        raise RuntimeError("builder exploded")
    jr._BUILDERS["zzz_boom"] = bad_builder
    try:
        doc = jr.job_report_doc("zzz_boom", {"a": 1})
    finally:
        del jr._BUILDERS["zzz_boom"]
    # the first failure lands here: summary + overflow still render
    assert doc.summary.endswith("finished.")
    assert "'a': 1" in doc.overflow_text


# ── 673: hostile job row -> empty display line ───────────────────

class _HostileJob(dict):
    def get(self, *a, **k):
        raise RuntimeError("row exploded")


def test_673_hostile_job_row_summarizes_to_empty():
    assert jr.summarize_last_result(_HostileJob()) == ""


def test_673_normal_rows_still_summarize_as_control():
    assert jr.summarize_last_result(
        {"last_result": "news:\nok: yes\n42 items fetched"}) \
        == "✅ 42 items fetched"
    failed = jr.summarize_last_result(
        {"last_result": "job failed: kaput"})
    assert failed.startswith("❌")
    assert "kaput" in failed
