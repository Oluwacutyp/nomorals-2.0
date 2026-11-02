"""Pass tail-115 pins for nomorals/media/composer_llm.py — 0 converts / 3 exclusions.

43 (brain_available): a crashing router stats_snapshot answers False
— the probe's honest negative; compose_song_spec then takes the
designed algorithmic path the module docstring promises ("When no
brain is available... an algorithmic composer builds the SongSpec").

107 (_extract_json): the first json.loads is rung one of a parse
ladder, not a swallow — on failure the regex block-extraction rung
runs, and None comes back only when both rungs fail. Flagged-accepted
(typed guard, no noqa; stays in the flagged scan).

126 (_taste_summary): DEAD ORGAN, pinned as standing state. The guard
imports ``get_taste`` from nomorals.media.taste — a function that does
not exist anywhere in the repo (the real API is
``load_taste(context) -> TasteStore``) — so the guard fires on every
call and the composition prompt NEVER receives the owner's taste
line; it always gets the documented default "unknown — write to the
request". The guard itself degrades honestly (exclusion in-batch);
the missing wiring is a targeted fix, absence-shaped, not a convert.
The standing-state pin below is written to break when the wiring is
fixed.

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


def test_126_standing_state_taste_wiring_is_dead():
    # get_taste does not exist in nomorals.media.taste — the import
    # inside the guard raises on every call. Breaks when fixed.
    assert not hasattr(taste_mod, "get_taste")
    assert cl._taste_summary(_Ctx()) == "unknown — write to the request"


def test_126_injected_api_control(monkeypatch):
    monkeypatch.setattr(
        taste_mod, "get_taste",
        lambda ctx: {"liked_styles": ["afrobeats", "jazz"]},
        raising=False)
    assert cl._taste_summary(_Ctx()) == "likes: afrobeats, jazz"


def test_126_raising_taste_source_degrades_to_default(monkeypatch):
    monkeypatch.setattr(taste_mod, "get_taste", lambda ctx: _boom(),
                        raising=False)
    assert cl._taste_summary(_Ctx()) == "unknown — write to the request"
