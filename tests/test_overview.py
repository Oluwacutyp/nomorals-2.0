"""Tests for nomorals/audio/overview.py — interactive audio overviews.

All offline: mock script_fn / voice_fn / llm_fn. No real TTS, no real LLM.
"""
import os
import tempfile

import pytest

from nomorals.audio.overview import (
    FORMATS,
    AudioOverview,
    Chapter,
    InteractiveSession,
    OverviewScript,
    OverviewStore,
    ScriptLine,
    _ground_script,
    _parse_script,
    _parse_sources,
    control_overview,
    make_overview,
)


def _db():
    return os.path.join(tempfile.mkdtemp(), "ov.db")


def _mock_script(prompt):
    return (
        "CHAPTER: Rents\n"
        "A: Welcome back — today we are digging into Lagos rents [S1].\n"
        "B: Right, the big number is that Yaba averages 2.1 million a year [S1].\n"
        "A: And Lekki pushes past 7 million for a 3-bed [S2].\n"
        "B: So Yaba is the value play for renters [S1].\n"
        "A: Exactly — that wraps the rent picture.\n"
    )


_SOURCES = [
    {"title": "Yaba report", "text": "Yaba 2-bed averages 2.1 million naira per year."},
    {"title": "Lekki report", "text": "Lekki 3-bed rents past 7 million naira per year."},
]


def _wav(speaker, text, **kwargs):
    # mock voice_fn: one distinct wav per host
    path = os.path.join(tempfile.mkdtemp(), f"{speaker}.wav")
    open(path, "wb").write(b"RIFF....WAVE")
    return path


def _mock_llm(prompt):
    if "CANNOT" in prompt or "question" in prompt.lower():
        return "Yaba averages 2.1 million [S1]."
    return _mock_script(prompt)


# ── parsing ───────────────────────────────────────────────────────────────

def test_parse_script_lines_and_chapters():
    lines = _parse_script(_mock_script(""))
    assert len(lines) == 5
    assert lines[0].speaker == "A"
    assert lines[0].cites == [1]
    assert lines[0].chapter == "Rents"
    assert all(isinstance(l, ScriptLine) for l in lines)


def test_parse_script_ignores_garbage():
    lines = _parse_script("C: wrong host\n\njust a stage direction\nA: ok [S1]")
    assert len(lines) == 1
    assert lines[0].speaker == "A"


def test_parse_sources():
    srcs = _parse_sources("Yaba :: rents are high ;; Lekki :: rents higher")
    assert len(srcs) == 2
    assert srcs[0]["title"] == "Yaba"
    assert "rents are high" in srcs[0]["text"]


def test_parse_sources_bare_text():
    srcs = _parse_sources("just some notes")
    assert len(srcs) == 1 and srcs[0]["title"] == "source"


# ── grounding ─────────────────────────────────────────────────────────────

def test_ground_script_keeps_cited():
    lines = _parse_script(_mock_script(""))
    script = _ground_script(lines, 2)
    assert script is not None
    assert len(script.lines) == 4  # the uncited closer is dropped
    assert script.dropped == 1
    assert script.chapters[0].title == "Rents"


def test_ground_script_refuses_mostly_uncited():
    lines = [ScriptLine(speaker="A", text="hi there"),
             ScriptLine(speaker="B", text="hello")]
    assert _ground_script(lines, 2) is None


def test_ground_script_rejects_bad_cite_index():
    lines = [ScriptLine(speaker="A", text="claim [S9]")]
    assert _ground_script(lines, 2) is None


# ── make_overview ─────────────────────────────────────────────────────────

def test_make_overview_no_sources():
    res = make_overview([], "deep-dive", script_fn=_mock_script)
    assert res["ok"] is False and "no sources" in res["reason"]


def test_make_overview_no_script_fn():
    res = make_overview(_SOURCES, "deep-dive")
    assert res["ok"] is False and "no script generator" in res["reason"]


def test_make_overview_bad_format():
    res = make_overview(_SOURCES, "opera", script_fn=_mock_script)
    assert res["ok"] is False and "unknown format" in res["reason"]


def test_make_overview_script_only():
    res = make_overview(_SOURCES, "deep-dive", script_fn=_mock_script,
                        render_audio=False, store=OverviewStore(db_path=_db()))
    assert res["ok"] is True
    ov: AudioOverview = res["overview"]
    assert len(ov.script.lines) == 4
    assert ov.script.dropped == 1
    assert {ln.speaker for ln in ov.script.lines} == {"A", "B"}
    assert not ov.has_audio()
    assert "script kept" in res["note"] or "skipped" in res["note"]


def test_make_overview_renders_audio_mock_voices(tmp_path, monkeypatch):
    # fake the render pieces: voice_fn returns real wav files, ffmpeg concat mocked
    import nomorals.audio.overview as ov_mod

    def fake_render(script, **kwargs):
        return {"ok": True, "path": str(tmp_path / "ov.wav"),
                "chapters": [Chapter(title="Rents", start_s=0.0)]}

    monkeypatch.setattr(ov_mod, "_render_audio", fake_render)
    res = make_overview(_SOURCES, "debate", script_fn=_mock_script,
                        voice_fn=_wav, store=OverviewStore(db_path=_db()))
    assert res["ok"] is True
    ov = res["overview"]
    assert ov.script.chapters[0].title == "Rents"
    assert "audio rendered" in res["note"]


def test_make_overview_never_raises():
    res = make_overview(None, None, script_fn=lambda p: (_ for _ in ()).throw(RuntimeError("x")))
    assert res["ok"] is False


def test_make_overview_lang_alias():
    res = make_overview(_SOURCES, "brief", lang="ekiti",
                        script_fn=_mock_script, render_audio=False)
    assert res["ok"] is True
    assert res["overview"].lang == "yo-ekiti"


def test_formats():
    assert set(FORMATS) == {"deep-dive", "debate", "brief"}


# ── store ─────────────────────────────────────────────────────────────────

def test_store_roundtrip():
    store = OverviewStore(db_path=_db())
    res = make_overview(_SOURCES, "deep-dive", script_fn=_mock_script,
                        render_audio=False, store=store)
    ov = res["overview"]
    got = store.get(ov.overview_id)
    assert got is not None and got.title == ov.title
    assert len(got.script.lines) == 4
    assert got.sources[0]["title"] == "Yaba report"
    assert store.latest().overview_id == ov.overview_id
    assert len(store.list()) == 1


def test_store_missing():
    store = OverviewStore(db_path=_db())
    assert store.get("nope") is None
    assert store.latest() is None


# ── interactive ───────────────────────────────────────────────────────────

def _overview_with_script():
    res = make_overview(_SOURCES, "deep-dive", script_fn=_mock_script,
                        render_audio=False)
    assert res["ok"]
    return res["overview"]


def test_interactive_ask_grounded():
    ov = _overview_with_script()
    sess = InteractiveSession(ov, llm_fn=_mock_llm, voice_fn=_wav)
    ans = sess.ask("what does Yaba cost?")
    assert ans["ok"] is True
    assert ans["refused"] is False
    assert "2.1" in ans["answer"]
    assert ans["audio"].endswith(".wav")
    assert len(sess.history) == 1


def test_interactive_ask_refused_not_confabulated():
    ov = _overview_with_script()

    def refuse_llm(prompt):
        return "CANNOT_ANSWER"

    sess = InteractiveSession(ov, llm_fn=refuse_llm, voice_fn=_wav)
    ans = sess.ask("what is the capital of Mars?")
    assert ans["ok"] is True
    assert ans["refused"] is True
    assert "can't answer" in ans["answer"]
    assert ans["audio"] == ""  # no audio for a refusal


def test_interactive_no_session():
    ov = _overview_with_script()
    sess = InteractiveSession(ov)  # no llm_fn
    ans = sess.ask("hello?")
    assert ans["ok"] is False and "unavailable" in ans["reason"]


def test_interactive_empty_question():
    ov = _overview_with_script()
    sess = InteractiveSession(ov, llm_fn=_mock_llm)
    assert sess.ask("  ")["ok"] is False


# ── chat ──────────────────────────────────────────────────────────────────

def test_chat_help():
    out = control_overview("")
    assert "/overview make" in out


def test_chat_make():
    store = OverviewStore(db_path=_db())
    out = control_overview(
        "make deep-dive Yaba :: 2-bed averages 2.1m ;; Lekki :: 3-bed past 7m",
        script_fn=_mock_script, store=store)
    assert "overview ready" in out


def test_chat_make_no_sources():
    out = control_overview("make deep-dive", script_fn=_mock_script,
                           store=OverviewStore(db_path=_db()))
    assert "couldn't build it" in out


def test_chat_list_empty():
    out = control_overview("list", store=OverviewStore(db_path=_db()))
    assert "no overviews" in out


def test_chat_list_and_ask():
    store = OverviewStore(db_path=_db())
    control_overview("make brief Notes :: the rent is 2.1m [S1] ok",
                     script_fn=_mock_script, store=store)
    listed = control_overview("list", store=store)
    assert "saved overviews" in listed
    out = control_overview("ask what is the rent?", llm_fn=_mock_llm,
                           voice_fn=_wav, store=store)
    assert "2.1" in out or "can't answer" in out


def test_chat_voices():
    assert "Adaeze" in control_overview("voices")


def test_chat_unknown():
    assert "/overview make" in control_overview("frobnicate")


def test_chat_never_raises():
    assert isinstance(control_overview(None), str)
    assert isinstance(control_overview("make"), str)
