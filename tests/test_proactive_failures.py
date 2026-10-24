"""Pins for Pass #110: proactive failure ledger.

Every converted site is driven for real; the one deliberate exclusion
(_parse_date's ISO-fallback probe) is pinned as the subsystem's sole
remaining scanner finding.
"""

import asyncio
from pathlib import Path

import pytest

from nomorals.agents.proactive import (
    ProactiveEngine,
    RelationshipCadence,
    Suggestion,
)
from nomorals.agents.proactive_failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)
from nomorals.tools.error_scan import scan


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(100)]


# -- ledger basics --------------------------------------------------------


def test_ledger_record_and_readback():
    record_failure("pin.site", ValueError("boom"))
    assert failure_count() == 1
    (entry,) = recent_failures()
    assert entry.site == "pin.site"
    assert entry.exc_type == "ValueError"
    assert entry.message == "boom"
    assert "pin.site" in format_failures()
    assert "ValueError: boom" in format_failures()


def test_ledger_empty_state():
    assert recent_failures() == []
    assert failure_count() == 0
    assert format_failures() == "no proactive failures recorded."


def test_ledger_bound():
    for _ in range(250):
        record_failure("pin.flood", RuntimeError("x"))
    assert failure_count() == 250
    assert len(recent_failures(1000)) == 200
    assert all(f.site == "pin.flood" for f in recent_failures(1000))


def test_ledger_newest_first():
    record_failure("pin.first", ValueError("1"))
    record_failure("pin.second", ValueError("2"))
    assert [f.site for f in recent_failures()] == ["pin.second", "pin.first"]


def test_ledger_unprintable_exc():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("nope")

    record_failure("pin.badstr", BadStr())
    (entry,) = recent_failures()
    assert entry.exc_type == "BadStr"
    assert "unprintable" in entry.message


def test_direct_import_surface():
    import nomorals.agents.proactive_failures as mod

    for name in ("Failure", "record_failure", "recent_failures",
                 "failure_count", "clear_failures", "format_failures"):
        assert name in mod.__all__
        assert hasattr(mod, name)


# -- real converted sites --------------------------------------------------


class _RaisingId:
    @property
    def id(self):
        raise RuntimeError("id boom")

    def label(self):
        return "x"


class _RaisingSignature:
    def signature(self):
        raise RuntimeError("sig boom")


class _HealthyIdentity:
    def __init__(self):
        self.id = "ident-1"

    def label(self):
        return "Ident One"

    def signature(self):
        return "Ident One"


def _engine_with_stub_suggestion():
    engine = ProactiveEngine(memory=None, calendar=None)

    async def _stub(user_id, context):
        return [Suggestion(suggestion_id="s1", text="t",
                           suggestion_type="pattern", action="a",
                           confidence=0.9, priority=1)]

    engine._suggestion_generators = [_stub]
    return engine


def test_identity_tag_failure_recorded():
    engine = _engine_with_stub_suggestion()
    engine._identity = _RaisingId()
    out = asyncio.run(engine.get_suggestions(user_id="u1"))
    assert len(out) == 1
    assert "identity_id" not in out[0].metadata
    assert _sites() == ["get_suggestions.identity_tag"]


def test_identity_tag_healthy_no_record():
    engine = _engine_with_stub_suggestion()
    engine._identity = _HealthyIdentity()
    out = asyncio.run(engine.get_suggestions(user_id="u1"))
    assert out[0].metadata["identity_id"] == "ident-1"
    assert out[0].metadata["identity_label"] == "Ident One"
    assert recent_failures() == []


def test_set_identity_failure_recorded():
    engine = ProactiveEngine(memory=None, calendar=None)

    def _raising_setattr(self, name, value):
        if name == "_identity":
            raise RuntimeError("set boom")
        object.__setattr__(self, name, value)

    orig = ProactiveEngine.__setattr__
    ProactiveEngine.__setattr__ = _raising_setattr
    try:
        assert engine.set_identity(_HealthyIdentity()) is False
    finally:
        ProactiveEngine.__setattr__ = orig
    assert _sites() == ["set_identity"]


def test_clear_identity_failure_recorded():
    engine = ProactiveEngine(memory=None, calendar=None)
    engine._identity = _HealthyIdentity()

    def _raising_setattr(self, name, value):
        if name == "_identity":
            raise RuntimeError("clear boom")
        object.__setattr__(self, name, value)

    orig = ProactiveEngine.__setattr__
    ProactiveEngine.__setattr__ = _raising_setattr
    try:
        assert engine.clear_identity() is None
    finally:
        ProactiveEngine.__setattr__ = orig
    assert _sites() == ["clear_identity"]


def test_identity_signature_failure_recorded():
    engine = ProactiveEngine(memory=None, calendar=None)
    engine._identity = _RaisingSignature()
    assert engine.identity_signature() == "Devon"
    assert _sites() == ["identity_signature"]


def test_index_order_failure_recorded(tmp_path):
    # people_dir is a FILE: INDEX.md under it cannot be read (OSError).
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    cadence = RelationshipCadence(people_dir=blocker)
    assert cadence._index_order() == []
    assert _sites() == ["_index_order"]


def test_record_contact_glob_failure_recorded(tmp_path, monkeypatch):
    cadence = RelationshipCadence(people_dir=tmp_path)

    def _raising_glob(self, pattern):
        raise OSError("glob boom")

    monkeypatch.setattr(Path, "glob", _raising_glob)
    path = cadence.record_contact("Nobody Here", ts=1_700_000_000)
    assert path.exists()  # fallback still writes the new page
    assert _sites() == ["record_contact.glob"]


def test_record_contact_idempotency_read_failure_recorded(tmp_path, monkeypatch):
    cadence = RelationshipCadence(people_dir=tmp_path)
    first = cadence.record_contact("Ada", ts=1_700_000_000)
    assert first.exists()

    def _raising_read_text(self, *args, **kwargs):
        raise OSError("read boom")

    monkeypatch.setattr(Path, "read_text", _raising_read_text)
    second = cadence.record_contact("Ada", ts=1_700_000_000)
    assert second == first  # same page, write still proceeds
    assert _sites() == ["record_contact.idempotency_read"]


# -- subsystem scan remainder ----------------------------------------------
# The deliberate exclusion: _parse_date's ISO-fallback probe. A failed
# ISO parse is the normal signal that sends the loop to the next format,
# not a failure — recording it would spam the ring on every non-ISO date
# string. It is the subsystem's ONLY remaining finding, by rule and
# context (not by line number, which shifts).


def test_subsystem_scan_remainder_is_only_the_deliberate_exclusion():
    report = scan(["nomorals/agents/proactive.py",
                   "nomorals/agents/proactive_failures.py"])
    by_file: dict[str, list] = {}
    for f in report.findings:
        by_file.setdefault(f.file.split("nomorals/")[-1], []).append(f)
    assert by_file.get("agents/proactive_failures.py", []) == []
    rest = by_file.get("agents/proactive.py", [])
    assert len(rest) == 1
    assert rest[0].rule == "E103"
    assert "except ValueError" in rest[0].context
