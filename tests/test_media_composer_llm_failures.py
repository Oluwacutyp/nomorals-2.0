"""Pass tail-115 pins for nomorals/media/composer_llm.py — 0 converts / 3 exclusions.

43 (brain_available): a crashing router stats_snapshot answers False
— the probe's honest negative; compose_song_spec then takes the
designed algorithmic path the module docstring promises ("When no
brain is available... an algorithmic composer builds the SongSpec").

107 (_extract_json): the first json.loads is rung one of a parse
ladder, not a swallow — on failure the regex block-extraction rung
runs, and None comes back only when both rungs fail. Flagged-accepted
(typed guard, no noqa; stays in the flagged scan).

126 (_taste_summary): FIXED by the assembly-audit truth-fix pass
(AA-46).  The guard used to import ``get_taste`` from
nomorals.media.taste — a function that did not exist anywhere in the
repo — so the guard fired on every call and the composition prompt
NEVER received the owner's taste line.  ``_taste_summary`` now reads
the real API, ``load_taste(context) -> TasteStore``, and summarises
``profile.genre_weights`` / ``profile.energy_weights``; a taste
store that fails to load degrades to the same neutral line but logs
at error level (loud in the journal, never a silent swallow).  The
pins below assert the NEW contract and replace the standing-state
pins that were written to break when the wiring was fixed.

No except handlers appear in this file (AST-checked).
"""
import nomorals.media.taste as taste_mod
from nomorals.media import composer_llm as cl


class _Router:
    def __init__(self, fn):
        self._fn = fn

    def stats_snapshot(self):
        return self._fn()


class _Ctx:
    pass


def _boom():
    raise RuntimeError("snapshot boom")


def test_43_snapshot_crash_means_brain_unavailable():
    ctx = _Ctx()
    ctx.router = _Router(_boom)
    assert cl.brain_available(ctx) is False


def test_43_healthy_active_router_control():
    ctx = _Ctx()
    ctx.router = _Router(lambda: {"active": "groq-llama"})
    assert cl.brain_available(ctx) is True


def test_43_mock_router_is_not_a_brain():
    ctx = _Ctx()
    ctx.router = _Router(lambda: {"active": "mock"})
    assert cl.brain_available(ctx) is False


def test_43_missing_snapshot_control():
    ctx = _Ctx()
    ctx.router = object()
    assert cl.brain_available(ctx) is False


def test_107_embedded_json_extracted_by_second_rung():
    assert cl._extract_json('prose {"b": 2} tail') == {"b": 2}


def test_107_direct_json_first_rung():
    assert cl._extract_json('{"a": 1}') == {"a": 1}


def test_107_garbage_returns_none():
    assert cl._extract_json("no json at all") is None


def test_126_taste_wiring_is_live(tmp_path):
    # AA-46: the summary comes from the real taste store — a real
    # taste.json under the context's workspace dir drives the line.
    import json
    from types import SimpleNamespace

    store_dir = tmp_path / ".nomorals"
    store_dir.mkdir()
    (store_dir / "taste.json").write_text(json.dumps({
        "genre_weights": {"afrobeats": 1.5, "jazz": 0.5},
        "energy_weights": {"driving": 1.0},
    }), encoding="utf-8")
    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    assert cl._taste_summary(ctx) == (
        "likes: afrobeats, jazz; energy: driving")


def test_126_empty_profile_reads_neutral(tmp_path):
    # An EMPTY profile (no taste learned yet) honestly reads as the
    # neutral line — distinct from a load failure.
    from types import SimpleNamespace

    ctx = SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    assert cl._taste_summary(ctx) == "unknown — write to the request"


def test_126_injected_api_control(monkeypatch):
    from types import SimpleNamespace

    profile = SimpleNamespace(
        genre_weights={"afrobeats": 2.0, "jazz": 1.0},
        energy_weights={}, tracks=[])
    monkeypatch.setattr(
        taste_mod, "load_taste",
        lambda ctx: SimpleNamespace(profile=profile))
    assert cl._taste_summary(_Ctx()) == "likes: afrobeats, jazz"


def test_126_raising_taste_source_degrades_loudly(monkeypatch, caplog):
    # A taste store that fails to load must be LOUD (error-level log
    # the journal can see), not the old invisible swallow.
    import logging

    monkeypatch.setattr(taste_mod, "load_taste", lambda ctx: _boom())
    with caplog.at_level(logging.ERROR,
                         logger="nomorals.media.composer_llm"):
        assert cl._taste_summary(_Ctx()) == "unknown — write to the request"
    assert any("taste" in r.message.lower() and r.levelno >= logging.ERROR
               for r in caplog.records)
