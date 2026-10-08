"""Tests for nomorals.games.film — VOD film study. All offline."""

import os
import tempfile

import pytest

from nomorals.games.film import (
    FilmBreakdown,
    FilmStore,
    FingerprintStore,
    GAMES,
    PlayNote,
    _coaching_summary,
    _normalize,
    _parse_moment_lines,
    analyze_vod,
    control_film,
    export_breakdown,
)


@pytest.fixture()
def home():
    d = tempfile.mkdtemp()
    old = os.environ.get("NOMORALS_HOME")
    os.environ["NOMORALS_HOME"] = d
    yield d
    if old is None:
        os.environ.pop("NOMORALS_HOME", None)
    else:
        os.environ["NOMORALS_HOME"] = old


def _vision_factory(responses):
    """vision(image_path, question) -> str from a rotating response list."""
    calls = {"n": 0}

    def vision(image_path, question):
        calls["n"] += 1
        return responses[(calls["n"] - 1) % len(responses)]

    vision.calls = calls
    return vision


def _img(path):
    from PIL import Image
    Image.new("RGB", (64, 64), "green").save(path)
    return path


# ── parsing ──────────────────────────────────────────────────────────

def test_parse_moment_lines():
    ms, hs = _parse_moment_lines(
        "MISTAKE — overextended alone in mid\n"
        "GOOD PLAY — clean headshot on the entry\n")
    assert ms == ["overextended alone in mid"]
    assert hs == ["clean headshot on the entry"]


def test_parse_moment_lines_none_filtered():
    ms, hs = _parse_moment_lines("MISTAKE — none\nGOOD PLAY — none visible\n")
    assert ms == [] and hs == []


def test_parse_moment_lines_garbage():
    ms, hs = _parse_moment_lines("lorem ipsum\nno colon here")
    assert ms == [] and hs == []


def test_normalize_maps_to_pattern():
    assert _normalize("overextended alone pushing mid", "valorant") == "overextend"
    assert _normalize("crosshair placement too low on angles", "valorant") == "crosshair"
    assert _normalize("something totally unrelated", "valorant") == ""


# ── analyze_vod ────────────────────────────────────────────────────────

def test_analyze_image_valorant(home):
    img = _img(os.path.join(home, "frame.jpg"))
    vision = _vision_factory([
        "MISTAKE — overextended alone pushing mid\n"
        "GOOD PLAY — clean headshot on the entry frag",
    ])
    bd = analyze_vod(img, "valorant", vision=vision)
    assert bd.available
    assert len(bd.mistakes) == 1
    assert len(bd.highlights) == 1
    assert bd.mistakes[0].fix  # normalized pattern → concrete fix
    assert "overextend" in bd.mistakes[0].text.lower()


def test_analyze_capped_at_three(home):
    img = _img(os.path.join(home, "frame.jpg"))
    vision = _vision_factory([
        "MISTAKE — mistake one\nGOOD PLAY — good one",
        "MISTAKE — mistake two\nGOOD PLAY — good two",
        "MISTAKE — mistake three\nGOOD PLAY — good three",
        "MISTAKE — mistake four\nGOOD PLAY — good four",
    ])
    bd = analyze_vod(img, "valorant", vision=vision)
    assert len(bd.mistakes) <= 3
    assert len(bd.highlights) <= 3


def test_analyze_dedupes_repeated_notes(home):
    img = _img(os.path.join(home, "frame.jpg"))
    vision = _vision_factory([
        "MISTAKE — overextended alone\nGOOD PLAY — none",
    ])
    bd = analyze_vod(img, "valorant", vision=vision)
    texts = [m.text.lower() for m in bd.mistakes]
    assert len(texts) == len(set(texts))


def test_analyze_timestamps_present(home):
    img = _img(os.path.join(home, "frame.jpg"))
    vision = _vision_factory(["MISTAKE — bad peek\nGOOD PLAY — nice trade"])
    bd = analyze_vod(img, "lol", vision=vision)
    assert bd.mistakes[0].label == "00:00"  # image → timestamp 0


def test_analyze_unknown_game():
    bd = analyze_vod("/tmp/x.jpg", "chess")
    assert not bd.available
    assert "unknown game" in bd.note


def test_analyze_missing_video():
    bd = analyze_vod("/tmp/does-not-exist.mp4", "valorant",
                     vision=lambda p, q: "")
    assert not bd.available
    assert "no video found" in bd.note


def test_analyze_no_vision_path(home):
    img = _img(os.path.join(home, "frame.jpg"))
    bd = analyze_vod(img, "valorant",
                     vision=lambda p, q: "blurry nothing")
    assert not bd.available


def test_analyze_vision_raises(home):
    img = _img(os.path.join(home, "frame.jpg"))

    def boom(p, q):
        raise RuntimeError("nope")

    bd = analyze_vod(img, "valorant", vision=boom)
    assert not bd.available


def test_analyze_workout_needs_exercise():
    bd = analyze_vod("/tmp/x.mp4", "workout")
    assert not bd.available
    assert "exercise" in bd.note


def test_coaching_summary():
    ms = [PlayNote(kind="mistake", text="Overextended alone")]
    hs = [PlayNote(kind="highlight", text="Clean entry frag")]
    s = _coaching_summary("valorant", ms, hs)
    assert "overextended" in s.lower() and "entry frag" in s.lower()


# ── export ─────────────────────────────────────────────────────────────

def test_export_breakdown():
    bd = FilmBreakdown(
        id="film_1", game="valorant",
        mistakes=[PlayNote(kind="mistake", text="Overextended",
                           timestamp=95.0, fix="Stay with team.")],
        highlights=[PlayNote(kind="highlight", text="Nice clutch",
                             timestamp=180.0)],
        coaching="drill the top mistake first.")
    out = export_breakdown(bd)
    assert "01:35" in out and "03:00" in out
    assert "Overextended" in out and "Nice clutch" in out


def test_format_includes_disclaimer():
    bd = FilmBreakdown(id="film_1", game="valorant",
                       mistakes=[PlayNote(kind="mistake", text="x")])
    assert "not professional advice" in bd.format()


# ── stores ─────────────────────────────────────────────────────────────

def test_film_store_roundtrip(home):
    store = FilmStore()
    bd = FilmBreakdown(
        id="film_abc", game="valorant", video="/tmp/m.mp4",
        mistakes=[PlayNote(kind="mistake", text="Overextended",
                           timestamp=60.0, fix="fix it")],
        highlights=[PlayNote(kind="highlight", text="Good")],
        coaching="coach me")
    assert store.save(bd)
    got = store.get("film_abc")
    assert got is not None and got.mistakes[0].text == "Overextended"
    assert got.mistakes[0].fix == "fix it"
    rows = store.list(game="valorant")
    assert any(r["id"] == "film_abc" for r in rows)


def test_fingerprint_recurring_pattern(home):
    fps = FingerprintStore()
    for ts in (480.0, 500.0):
        bd = FilmBreakdown(
            id=f"film_{int(ts)}", game="valorant", available=True,
            mistakes=[PlayNote(kind="mistake", text="overextended alone",
                               timestamp=ts)])
        assert fps.ingest("owner", bd) == 1
    pats = fps.fingerprint("owner", game="valorant")
    assert len(pats) == 1
    assert "minute 8" in pats[0]
    assert "overextend" in pats[0]


def test_fingerprint_needs_two_occurrences(home):
    fps = FingerprintStore()
    bd = FilmBreakdown(
        id="film_1", game="lol", available=True,
        mistakes=[PlayNote(kind="mistake", text="poor map awareness",
                           timestamp=120.0)])
    fps.ingest("owner", bd)
    assert fps.fingerprint("owner") == []


def test_fingerprint_ignores_unavailable(home):
    fps = FingerprintStore()
    bd = FilmBreakdown(id="film_1", game="valorant", available=False)
    assert fps.ingest("owner", bd) == 0


# ── chat ───────────────────────────────────────────────────────────────

def test_chat_games():
    out = control_film("games")
    assert "valorant" in out and "workout" in out


def test_chat_analyze_image(home):
    img = _img(os.path.join(home, "frame.jpg"))
    vision = _vision_factory([
        "MISTAKE — overextended alone\nGOOD PLAY — clean entry"])
    out = control_film(f"analyze {img} valorant", vision=vision)
    assert "VOD breakdown" in out
    assert "overextended" in out.lower()


def test_chat_analyze_usage():
    assert "usage" in control_film("analyze").lower()
    assert "usage" in control_film("analyze /tmp/x.mp4").lower()


def test_chat_fingerprint_empty(home):
    out = control_film("fingerprint")
    assert "2+ VODs" in out or "no recurring" in out.lower()


def test_chat_list_empty(home):
    assert "no vod" in control_film("list").lower()


def test_chat_export_missing(home):
    assert "no breakdown" in control_film("export nope").lower()


def test_chat_never_raises():
    for tail in ("", "help", "analyze", "export", "list", "fingerprint",
                 "bogus verb here", "analyze \x00"):
        assert isinstance(control_film(tail), str)


def test_games_constant():
    assert set(GAMES) == {"valorant", "lol", "bgmi", "sport", "workout"}
