"""Failure pins for nomorals/audio/overview.py (tail-67).

Every site is an adjudicated EXCLUSION: the swallow is the module's
documented "Never raises" design and the failure value is identical
to the function's genuine-empty / unavailable value. Each pin proves
the guarded operation was attempted (spy) and that the failure value
equals the honest control value.
"""
from pathlib import Path

import pytest

from nomorals.audio import overview as ov


class RaisingConn:
    """Store connection whose execute always dies (spy)."""

    def __init__(self):
        self.attempts = 0

    def execute(self, *a, **k):
        self.attempts += 1
        raise RuntimeError("store boom")

    def commit(self):
        raise RuntimeError("store boom")


def _made_overview(tmp_path):
    store = ov.OverviewStore(str(tmp_path / "ov.db"))
    res = ov.make_overview(
        [{"title": "T", "text": "gold rose sharply [S1] today"}],
        "deep-dive",
        script_fn=lambda p: "A: gold rose sharply [S1] today\n"
                            "B: indeed [S1]",
        render_audio=False, store=store)
    assert res["ok"] is True
    return store, res["overview"]


# 336 — _audio_duration: probe failure -> 0.0, the same value as
# "ffprobe missing" (measurement unavailable); both consumers in
# _render_audio branch on falsy to their designed fallbacks.
def test_336_audio_duration_probe_failure_is_unavailable(monkeypatch):
    calls = []

    def boom_run(*a, **k):
        calls.append(a)
        raise RuntimeError("ffprobe died")

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ffprobe")
    monkeypatch.setattr("subprocess.run", boom_run)
    assert ov._audio_duration(Path("x.wav")) == 0.0
    assert len(calls) == 1  # the probe was really attempted
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert ov._audio_duration(Path("x.wav")) == 0.0  # identical value


# 411 — OverviewStore.save: store failure -> False, the function's
# honest failure value (never a fake True).
def test_411_save_failure_returns_false(tmp_path):
    store, overview = _made_overview(tmp_path)
    assert store.get(overview.overview_id) is not None  # control: works
    rc = RaisingConn()
    store._db = rc
    assert store.save(overview) is False
    assert rc.attempts >= 1  # the INSERT was really attempted


# 431 — OverviewStore.get: store failure -> None, identical to a
# genuinely missing id (the documented AudioOverview | None contract).
def test_431_get_failure_returns_none(tmp_path):
    store, overview = _made_overview(tmp_path)
    assert store.get("missing-id") is None  # control: genuine absence
    rc = RaisingConn()
    store._db = rc
    assert store.get(overview.overview_id) is None
    assert rc.attempts >= 1


# 442 — OverviewStore.list: store failure -> [], identical to a
# genuinely empty store.
def test_442_list_failure_returns_empty(tmp_path):
    store, _overview = _made_overview(tmp_path)
    assert len(store.list()) == 1  # control: works
    empty = ov.OverviewStore(str(tmp_path / "empty.db"))
    assert empty.list() == []  # control: genuine empty
    rc = RaisingConn()
    store._db = rc
    assert store.list() == []
    assert rc.attempts >= 1


# 575 — suggest_questions: a dying llm_fn lands on [], the same value
# as an overview with no script (the function's designed empty).
def test_575_suggest_questions_llm_failure_returns_empty(tmp_path):
    _store, overview = _made_overview(tmp_path)
    calls = []

    def boom_llm(prompt):
        calls.append(prompt)
        raise RuntimeError("llm died")

    assert ov.suggest_questions(overview, llm_fn=boom_llm) == []
    assert len(calls) == 1  # the model was really asked
    empty_ov = ov.AudioOverview(
        overview_id="e", title="t", format="brief", lang="en",
        script=ov.OverviewScript(format="brief", lang="en", lines=[]))
    assert ov.suggest_questions(empty_ov) == []  # identical value


# 605 — export_markdown: a broken overview object lands on "", the
# never-raises display fallback; a healthy overview renders.
def test_605_export_markdown_failure_returns_empty_str(tmp_path):
    _store, overview = _made_overview(tmp_path)
    assert overview.title in ov.export_markdown(overview)  # control

    class BadOv:
        @property
        def title(self):
            raise RuntimeError("broken ov")

    assert ov.export_markdown(BadOv()) == ""
