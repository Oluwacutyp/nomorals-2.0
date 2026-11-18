"""Video briefs (Summarize.tech, 52-websites #39): a long video as a
sectioned brief — timestamped sections with headings, each section's
summary and key points, an overall TLDR, every section deep-linking to
its moment, and the brief naming which transcript it was built from.

The proof, per the report: a fixture long video with a known transcript
-> section boundaries align to the transcript's timestamps, each section
summary contains its section's planted key facts and no others;
captionless video falls back to transcription; a video with neither
reports that honestly instead of inventing a summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nomorals.media.video_brief import (
    VideoBrief,
    VideoBriefStore,
    brief_video,
    build_brief,
    deep_link,
    format_brief_reply,
    parse_video_ref,
    parse_vtt,
    section_words,
)

# ── fixture transcript ──────────────────────────────────────────────────
#
# A 12-minute talk in three ~5-minute sections. Each section's planted
# fact sentence repeats its distinctive words across the section, so
# the extractive scorer ranks it top — and each section's vocabulary
# is disjoint from the others, so no summary can carry another
# section's facts.


def _words_for(sentences: list[tuple[float, str]]) -> list[dict[str, object]]:
    """Sentences as (start_seconds, text) -> word-timed words."""
    words: list[dict[str, object]] = []
    for start, text in sentences:
        t = start
        for token in text.split():
            words.append({"start": t, "end": t + 0.4, "text": token})
            t += 0.45
    return words


SECTION_1 = [
    (2.0, "Welcome to this talk about fusion energy and the future of power."),
    (20.0, "The fusion reactor reaches plasma temperatures hotter than "
           "the core of the sun."),
    (45.0, "Magnetic confinement holds the plasma steady inside the "
           "fusion reactor chamber."),
    (80.0, "The plasma inside the fusion reactor must stay stable for "
           "the reaction to continue."),
    (120.0, "Engineers monitor the plasma temperature every second of "
            "the fusion reactor run."),
    (200.0, "The fusion reactor team checks the magnetic confinement "
            "coils before every plasma shot."),
    (280.0, "That is the first part: the fusion reactor, the plasma, "
            "and magnetic confinement."),
]

SECTION_2 = [
    (305.0, "Next we turn to ocean tides and the power hidden in the sea."),
    (330.0, "The tidal lagoon generates clean electricity from the rise "
            "and fall of the ocean tide."),
    (380.0, "Tidal turbines spin as the ocean tide flows through the lagoon gates."),
    (450.0, "The lagoon stores ocean tide energy and releases it at peak demand."),
    (520.0, "Engineers measure the tidal lagoon output across every ocean tide cycle."),
    (580.0, "That is the second part: the tidal lagoon, the ocean tide, "
            "and clean electricity."),
]

SECTION_3 = [
    (605.0, "Finally we look at desert solar farms and the grid that carries them."),
    (630.0, "The desert solar array feeds the national grid through "
            "high voltage lines across the sand."),
    (680.0, "Solar panels in the desert array track the sun from dawn until dusk."),
    (700.0, "The national grid balances the desert solar array output with "
            "evening demand."),
]

FIXTURE_WORDS = _words_for(SECTION_1 + SECTION_2 + SECTION_3)

VTT_FIXTURE = """WEBVTT

1
00:00:02.000 --> 00:00:05.500
Hello <b>world</b>, welcome to the talk.

2
00:00:20.000 --> 00:00:24.000
The fusion reactor
reaches plasma temperatures.
"""


# ── refs ────────────────────────────────────────────────────────────────


def test_parse_video_ref_youtube_forms() -> None:
    for ref in (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
        "dQw4w9WgXcQ",
    ):
        got = parse_video_ref(ref)
        assert got["kind"] == "youtube", ref
        assert got["video_id"] == "dQw4w9WgXcQ", ref


def test_parse_video_ref_local_and_garbage() -> None:
    got = parse_video_ref("/home/owner/talk.mp4")
    assert got["kind"] == "local"
    assert got["path"] == "/home/owner/talk.mp4"
    got = parse_video_ref("not a video ref at all")
    assert got["kind"] == "unknown"


def test_deep_link() -> None:
    assert deep_link("dQw4w9WgXcQ", 305.0) == (
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=305")


# ── VTT ─────────────────────────────────────────────────────────────────


def test_parse_vtt() -> None:
    words = parse_vtt(VTT_FIXTURE)
    assert words[0].text == "Hello"
    assert words[0].start == pytest.approx(2.0)
    # words inside one cue share its span evenly: 6 words over 3.5s
    assert words[0].end == pytest.approx(2.5833, abs=0.01)
    texts = [w.text for w in words]
    assert "world," in texts
    assert "fusion" in texts
    assert "reactor" in texts
    # tags stripped, never carried into the words
    assert not any("<" in t for t in texts)
    # cue 2 spans two lines: 6 words over 4s, "reactor" is the third
    reactor = next(w for w in words if w.text == "reactor")
    assert reactor.start == pytest.approx(21.33, abs=0.01)
    assert reactor.end == pytest.approx(22.0)


def test_parse_vtt_malformed_never_raises() -> None:
    assert parse_vtt("") == []
    assert parse_vtt("WEBVTT\n\nnot a cue at all\n") == []
    assert parse_vtt("00:99:99.000 --> 00:00:01.000\nbroken time\n") == []


# ── sectioning ──────────────────────────────────────────────────────────


def test_section_words_align_to_timestamps() -> None:
    windows = section_words(FIXTURE_WORDS, window_s=300.0)
    assert [round(w[0], 1) for w in windows] == [0.0, 300.0, 600.0]
    # ends are the last word's end in the window — the transcript's own
    assert [w[1] for w in windows] == [
        pytest.approx(285.8, abs=0.1),
        pytest.approx(586.25, abs=0.1),
        pytest.approx(705.35, abs=0.1)]
    # every word lands in exactly one window
    total = sum(len(w[2]) for w in windows)
    assert total == len(FIXTURE_WORDS)


def test_section_words_merges_short_tail() -> None:
    words = _words_for([
        (2.0, "First part words here."),
        (305.0, "Second part words here."),
        (605.0, "A tiny tail."),
    ])
    windows = section_words(words, window_s=300.0, min_tail_s=60.0)
    assert len(windows) == 2
    assert windows[1][1] == pytest.approx(606.3, abs=0.1)


# ── the brief itself ────────────────────────────────────────────────────


def test_build_brief_sections_carry_their_own_facts() -> None:
    brief = build_brief(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        FIXTURE_WORDS, title="A Talk in Three Parts",
        transcript_source="captions", language="en")
    assert brief.ok
    assert len(brief.sections) == 3
    starts = [round(s.start, 1) for s in brief.sections]
    assert starts == [0.0, 300.0, 600.0]
    # section 1 names fusion, never tides or solar
    first = brief.sections[0]
    assert "fusion" in first.summary.lower()
    assert "plasma" in first.summary.lower()
    assert "tidal" not in first.summary.lower()
    assert "solar" not in first.summary.lower()
    # section 2 names tides, never fusion or solar
    second = brief.sections[1]
    assert "tidal" in second.summary.lower()
    assert "lagoon" in second.summary.lower()
    assert "fusion" not in second.summary.lower()
    assert "solar" not in second.summary.lower()
    # section 3 names solar, never fusion or tides
    third = brief.sections[2]
    assert "solar" in third.summary.lower()
    assert "fusion" not in third.summary.lower()
    assert "tidal" not in third.summary.lower()
    # every section deep-links to its moment
    for section in brief.sections:
        assert section.deep_link.endswith(f"&t={int(section.start)}")
    # headings are grounded in the section's own words
    heading = brief.sections[0].heading.lower()
    assert any(term in heading for term in
               ("fusion", "magnetic", "plasma", "confinement"))
    # the brief names which transcript it was built from
    assert brief.transcript_source == "captions"
    assert brief.title == "A Talk in Three Parts"


def test_build_brief_tldr_and_duration() -> None:
    brief = build_brief(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        FIXTURE_WORDS, title="A Talk in Three Parts",
        transcript_source="captions", language="en")
    assert brief.overall_tldr
    assert len(brief.overall_tldr.split()) <= 60
    # duration is the transcript's own span — last word's end
    assert brief.duration == pytest.approx(705.35, abs=1.0)


def test_build_brief_empty_words_is_honest() -> None:
    brief = build_brief("https://www.youtube.com/watch?v=dQw4w9WgXcQ", [])
    assert not brief.ok
    assert "no transcript" in brief.reason.lower()
    assert brief.sections == []


def test_format_reply_names_source_and_links() -> None:
    brief = build_brief(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        FIXTURE_WORDS, title="A Talk in Three Parts",
        transcript_source="captions", language="en")
    text = format_brief_reply(brief)
    assert "A Talk in Three Parts" in text
    assert "caption track" in text
    assert "00:00" in text
    assert "05:00" in text
    assert "10:00" in text
    assert "watch?v=dQw4w9WgXcQ&t=300" in text
    assert brief.overall_tldr.split(".")[0] in text


# ── acquisition: captions → ASR → honest refusal ───────────────────────


def test_brief_video_uses_captions_first(tmp_path: Path) -> None:
    def captions(video_id: str) -> dict[str, object]:
        return {"words": FIXTURE_WORDS, "language": "en",
                "manual": True, "title": "A Talk in Three Parts"}

    result = brief_video(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        caption_fetcher=captions,
        transcribe_fn=lambda path: (_ for _ in ()).throw(
            AssertionError("ASR must not run when captions exist")),
        store=VideoBriefStore(tmp_path / "video_briefs.json"))
    assert result.ok
    assert result.transcript_source == "captions"
    assert len(result.sections) == 3


def test_brief_video_falls_back_to_transcription(tmp_path: Path) -> None:
    calls: list[str] = []

    def no_captions(video_id: str) -> dict[str, object]:
        return {"words": None, "reason": "no captions on this video"}

    def download(ref: str) -> str:
        calls.append("download")
        return "/tmp/fake-talk.mp4"

    def transcribe(path: str) -> list[dict[str, object]]:
        calls.append("transcribe")
        return FIXTURE_WORDS

    result = brief_video(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        caption_fetcher=no_captions, download_fn=download,
        transcribe_fn=transcribe,
        store=VideoBriefStore(tmp_path / "video_briefs.json"))
    assert result.ok
    assert result.transcript_source == "asr"
    assert calls == ["download", "transcribe"]


def test_brief_video_neither_reports_honestly() -> None:
    result = brief_video(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        caption_fetcher=lambda vid: {"words": None,
                                     "reason": "no captions on this video"},
        download_fn=lambda ref: (_ for _ in ()).throw(
            RuntimeError("yt-dlp is not installed")),
        transcribe_fn=lambda path: (_ for _ in ()).throw(
            AssertionError("must not transcribe without a file")))
    assert not result.ok
    assert "captions" in result.reason.lower()
    assert "download" in result.reason.lower()
    assert result.sections == []
    assert result.overall_tldr == ""


def test_brief_video_local_path_skips_download(tmp_path: Path) -> None:
    result = brief_video(
        "/home/owner/talk.mp4",
        caption_fetcher=lambda vid: (_ for _ in ()).throw(
            AssertionError("local files have no caption track to fetch")),
        download_fn=lambda ref: (_ for _ in ()).throw(
            AssertionError("local files are not downloaded")),
        transcribe_fn=lambda path: FIXTURE_WORDS,
        store=VideoBriefStore(tmp_path / "video_briefs.json"))
    assert result.ok
    assert result.transcript_source == "asr"


# ── translation ─────────────────────────────────────────────────────────


def test_brief_video_translates_foreign_transcript() -> None:
    spanish = _words_for([
        (2.0, "Bienvenidos a esta charla sobre la energía de fusión."),
        (20.0, "El reactor de fusión alcanza temperaturas de plasma "
               "más altas que el núcleo del sol."),
        (45.0, "El confinamiento magnético mantiene el plasma estable "
               "dentro del reactor de fusión."),
    ])

    def fake_translate(text: str, *, source: str, target: str) -> str:
        return f"[EN] {text}"

    brief = build_brief(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        spanish, title="Charla sobre fusión",
        transcript_source="captions", language="es",
        translate_fn=fake_translate)
    assert brief.ok
    assert brief.translated
    assert brief.title == "Charla sobre fusión"
    assert brief.overall_tldr.startswith("[EN]")
    assert brief.sections[0].summary.startswith("[EN]")
    assert "es" in brief.limits[0] or "Spanish" in " ".join(brief.limits)


def test_brief_video_keeps_original_when_translation_unavailable() -> None:
    spanish = _words_for([
        (2.0, "Bienvenidos a esta charla sobre la energía de fusión."),
    ])
    brief = build_brief(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        spanish, title="Charla sobre fusión",
        transcript_source="captions", language="es",
        translate_fn=lambda text, source, target: None)
    assert brief.ok
    assert not brief.translated
    assert any("translat" in limit.lower() for limit in brief.limits)


# ── store ───────────────────────────────────────────────────────────────


def test_store_round_trip(tmp_path: Path) -> None:
    store = VideoBriefStore(tmp_path / "video_briefs.json")
    brief = build_brief(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        FIXTURE_WORDS, title="A Talk in Three Parts",
        transcript_source="captions", language="en")
    assert brief.ok
    store.save(brief)
    listed = store.list()
    assert len(listed) == 1
    assert listed[0]["title"] == "A Talk in Three Parts"
    got = store.get(brief.brief_id)
    assert got is not None
    assert got["title"] == "A Talk in Three Parts"
    assert len(got["sections"]) == 3
    # the file on disk is the brief itself, readable as JSON
    raw = json.loads((tmp_path / "video_briefs.json").read_text())
    assert brief.brief_id in raw["briefs"]


def test_store_missing_is_none(tmp_path: Path) -> None:
    store = VideoBriefStore(tmp_path / "video_briefs.json")
    assert store.get("nope") is None
    assert store.list() == []


def test_brief_result_is_json_safe() -> None:
    brief = build_brief(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        FIXTURE_WORDS, title="A Talk in Three Parts",
        transcript_source="captions", language="en")
    json.dumps(brief.to_dict())
    assert VideoBrief.from_dict(brief.to_dict()).title == brief.title
