"""Mock screens Phase 4 (FakeDetail port, census §9.2,
§16.3): screen recordings — the flagship FakeDetail
behaviour.  A chat records messages revealing one by one
with typing dots and voice notes at true length; lists
arrive row by row; status/story segments fill; calls
advance their timer; long pages scroll; static screens
hold.

Network-free: renders are local Pillow work, voice
fixtures are generated WAVs, video is local ffmpeg via
Motion Studio's frame pipe, and the delivery seam is a
recording stub — the phase-2b/3 suites' idiom.

H.264 is lossy, so frame comparisons carry a small
per-channel tolerance instead of pretending a decoded
frame is byte-identical to the still.
"""
from __future__ import annotations

import io
import math
import shutil
import subprocess
import wave
from array import array
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops

from nomorals.agents.partner.runtime_mock_screens import (
    RuntimeMockScreensMixin,
)
from nomorals.media.motion_studio._core import probe_duration
from nomorals.social.chat.mock_screens import (
    CHAT_DESIGNS,
    DeviceChrome,
    Message,
    MockScreensError,
    MockScreenService,
    Participant,
    ProfileItem,
    RecordingError,
    Screen,
    TimelineSpec,
    build_call_recording,
    build_chat_recording,
    build_comment_recording,
    build_list_recording,
    build_profile_recording,
    build_status_recording,
    build_story_recording,
    record,
)
from nomorals.social.chat.mock_screens.authoring import (
    demo_call_screen,
    demo_comment_screen,
    demo_list_screen,
    demo_profile_screen,
    demo_screen,
    demo_status_screen,
    demo_story_screen,
    parse_mock_tail,
)
from nomorals.social.chat.mock_screens.render import (
    render_screen,
)
from nomorals.social.chat.mock_screens.render_screens import (
    render_call_screen,
    render_list_screen,
    render_status_screen,
)
from nomorals.social.chat.mock_screens.render_social import (
    render_comment_screen,
    render_profile_screen,
    render_story_screen,
)

# ── fixtures ─────────────────────────────────────────────


def _wav_bytes(seconds: float = 2.0, hz: int = 440,
               rate: int = 16000) -> bytes:
    """A real voice note: a sine tone as 16-bit mono WAV."""
    frames = array(
        "h",
        (int(12000 * math.sin(2 * math.pi * hz * i / rate))
         for i in range(int(seconds * rate))))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(frames.tobytes())
    return buf.getvalue()


def _wav_file(tmp_path, seconds: float = 2.0) -> Path:
    path = tmp_path / ("voice-" + str(seconds) + ".wav")
    path.write_bytes(_wav_bytes(seconds))
    return path


def _send_stub():
    class _Send:
        def __init__(self):
            self.calls = []

        def __call__(self, ref, caption=""):
            self.calls.append((ref, caption))
            return SimpleNamespace(ok=True, error="")

    return _Send()


class _Runtime(RuntimeMockScreensMixin):
    def __init__(self, tmp_path, operator=True):
        self._tmp = tmp_path
        self._operator = operator
        self.send = _send_stub()

    def _is_operator(self, message):
        return self._operator

    def _expression_root(self):
        return self._tmp

    def _makers_audience(self, message):
        return "personal"

    def _expression_send(self, message, ref, caption=""):
        return self.send(ref, caption)


def _message():
    return SimpleNamespace(
        chat=SimpleNamespace(kind="dm"), sender="owner")


def _voice_screen(wav: Path) -> Screen:
    return Screen(
        platform="whatsapp", title="Adaeze",
        subtitle="online",
        participants=[Participant("me", "You"),
                      Participant("adaeze", "Adaeze")],
        messages=[
            Message("d1", "system", kind="divider",
                    text="Today"),
            Message("t1", "me", text="listen to this",
                    time="9:41 AM", ticks="read"),
            Message("v1", "adaeze", kind="voice",
                    time="9:42 AM", media=str(wav)),
            Message("t2", "adaeze", text="did you hear it?",
                    time="9:43 AM"),
        ],
        chrome=DeviceChrome(style="android", clock="9:41"))


def _extract_frames(mp4: Path, out_dir: Path) -> list:
    """Every decoded frame, in order, as PIL images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg is required for Phase 4 tests"
    proc = subprocess.run(
        [ffmpeg, "-v", "error", "-y", "-i", str(mp4),
         str(out_dir / "f%03d.png")],
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr
    return [Image.open(p).convert("RGB")
            for p in sorted(out_dir.glob("f*.png"))]


def _mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    """Mean per-channel absolute difference between two
    frames of the same size (H.264 is lossy: exact
    equality is the wrong assertion for decoded video)."""
    assert a.size == b.size, (a.size, b.size)
    # an RGB histogram has 768 bins (one 256-bin range
    # per channel), so weigh the difference in grayscale
    # where the bin index IS the pixel value.
    diff = ImageChops.difference(a, b).convert("L")
    hist = diff.histogram()
    total = sum(hist)
    weighted = sum(i * n for i, n in enumerate(hist))
    return weighted / max(1, total)


def _assert_frame_like(frame, still_png, tol: float = 5.0):
    still = Image.open(io.BytesIO(still_png)).convert("RGB")
    assert _mean_abs_diff(frame, still) <= tol, (
        "decoded frame differs from the still by "
        + str(_mean_abs_diff(frame, still)))


# ── the timeline spec is data ────────────────────────────


def test_spec_rejects_bad_values():
    with pytest.raises(RecordingError):
        TimelineSpec(fps=0)
    with pytest.raises(RecordingError):
        TimelineSpec(reveal_hold_s=-1.0)


def test_record_refuses_an_empty_timeline(tmp_path):
    from nomorals.social.chat.mock_screens.animation import (
        Recording,
    )
    rec = Recording(kind="chat", spec=TimelineSpec(),
                    clips=[], size=(800, 1600))
    with pytest.raises(RecordingError):
        record(rec, tmp_path / "empty.mp4")


# ── chats: reveal, typing, voice at true length ─────────


def test_chat_timeline_is_data_not_sleeps():
    rec = build_chat_recording(demo_screen("whatsapp"))
    assert rec.kind == "chat"
    assert rec.size == (800, 1864)
    labels = [c["label"] for c in rec.timeline]
    assert labels[0] == "opening"
    assert labels[-1] == "ending"
    assert labels.count("typing") == 12  # 3 incoming runs × 4 phases
    assert labels.count("reveal m2") == 1
    # 0.5 opening + 9 × 0.7 reveals + 3 × 1.2 typing + 1.0 end
    assert rec.duration_s == pytest.approx(
        0.5 + 9 * 0.7 + 3 * 1.2 + 1.0)


def test_chat_recording_is_a_playable_mp4(tmp_path):
    rec = build_chat_recording(demo_screen("whatsapp"))
    out = tmp_path / "chat.mp4"
    result = record(rec, out)
    assert out.exists() and out.stat().st_size > 0
    assert result["frames"] == round(
        rec.duration_s * rec.spec.fps)
    assert result["duration_s"] == pytest.approx(
        rec.duration_s, abs=0.35)
    assert probe_duration(out) == pytest.approx(
        rec.duration_s, abs=0.35)
    frames = _extract_frames(out, tmp_path / "frames")
    assert len(frames) == result["frames"]
    assert all(f.size == tuple(rec.size) for f in frames)


def test_chat_recording_ends_on_the_previewed_screen(tmp_path):
    screen = demo_screen("whatsapp")
    rec = build_chat_recording(screen)
    out = tmp_path / "chat.mp4"
    record(rec, out)
    frames = _extract_frames(out, tmp_path / "frames")
    still, drawn = render_screen(screen)
    _assert_frame_like(frames[-1], still)
    # the opening frame shows the empty conversation —
    # it differs from the finished screen — while the
    # input bar stays pinned at the bottom of the frame
    assert _mean_abs_diff(frames[0], frames[-1]) > 5.0
    bar = drawn["regions"]["input_bar"]
    box = (int(bar[0]), int(bar[1]),
           int(bar[0] + bar[2]), int(bar[1] + bar[3]))
    still_img = Image.open(io.BytesIO(still)).convert("RGB")
    assert _mean_abs_diff(
        frames[0].crop(box), still_img.crop(box)) <= 5.0


def test_voice_note_plays_at_its_true_duration(tmp_path):
    wav = _wav_file(tmp_path, 2.0)
    screen = _voice_screen(wav)
    rec = build_chat_recording(screen)
    # 0.5 + 4 reveals × 0.7 + 1.2 typing + 2.0 voice
    # playback + 0.3 tail + 1.0 end
    assert rec.duration_s == pytest.approx(
        0.5 + 4 * 0.7 + 1.2 + 2.0 + 0.3 + 1.0)
    voice = [c for c in rec.timeline
             if c["label"].startswith("voice v1")]
    assert len(voice) == 8  # 2.0s in 0.25s steps
    assert sum(c["end_s"] - c["start_s"] for c in voice) \
        == pytest.approx(2.0)
    out = tmp_path / "voice.mp4"
    result = record(rec, out)
    assert result["duration_s"] == pytest.approx(7.8, abs=0.35)
    frames = _extract_frames(out, tmp_path / "frames")
    assert len(frames) == result["frames"]


def test_voice_playhead_colors_the_played_bars(tmp_path):
    wav = _wav_file(tmp_path, 2.0)
    screen = _voice_screen(wav)
    accent = CHAT_DESIGNS["whatsapp"]["light"]["accent"]
    counts = {}
    for frac in (0.0, 0.5, 1.0):
        data, drawn = render_screen(
            screen, voice_progress={"v1": frac})
        entry = next(e for e in drawn["elements"]
                     if e["id"] == "v1")
        box = entry["media_box"]
        img = Image.open(io.BytesIO(data)).convert("RGB")
        region = img.crop((int(box[0]), int(box[1]),
                           int(box[0] + box[2]),
                           int(box[1] + box[3])))
        counts[frac] = sum(
            1 for px in region.getdata() if px[:3] == accent)
    assert counts[0.0] == 0  # unplayed: no accent bars
    assert counts[0.5] > counts[0.0]
    assert counts[1.0] > counts[0.5]


def test_still_render_is_unchanged_by_the_playhead():
    screen = demo_screen("whatsapp")
    plain, _ = render_screen(screen)
    empty_progress, _ = render_screen(
        screen, voice_progress={})
    assert plain == empty_progress


def test_broken_voice_media_fails_honestly(tmp_path):
    screen = _voice_screen(tmp_path / "missing.ogg")
    with pytest.raises(RecordingError,
                       match="could not be used"):
        build_chat_recording(screen)


def test_broken_image_media_fails_honestly():
    screen = Screen(
        platform="whatsapp", title="Adaeze",
        participants=[Participant("me", "You"),
                      Participant("adaeze", "Adaeze")],
        messages=[
            Message("i1", "adaeze", kind="image",
                    caption="gone", media="/nonexistent/x.png"),
        ],
        chrome=DeviceChrome(style="android"))
    with pytest.raises(RecordingError):
        build_chat_recording(screen)


def test_no_media_screen_still_records(tmp_path):
    screen = Screen(
        platform="telegram", title="Adaeze",
        participants=[Participant("me", "You"),
                      Participant("adaeze", "Adaeze")],
        messages=[
            Message("t1", "adaeze", text="hi",
                    time="9:41 AM"),
            Message("t2", "me", text="hello",
                    time="9:42 AM", ticks="read"),
        ],
        chrome=DeviceChrome(style="android"))
    rec = build_chat_recording(screen)
    out = tmp_path / "plain.mp4"
    result = record(rec, out)
    assert result["duration_s"] == pytest.approx(
        rec.duration_s, abs=0.35)


# ── lists ────────────────────────────────────────────────


def test_list_recording_rows_arrive_one_by_one(tmp_path):
    screen = demo_list_screen("whatsapp-list")
    rec = build_list_recording(screen)
    assert rec.kind == "list"
    assert len(rec.timeline) == 1 + len(screen.rows) + 1
    assert rec.duration_s == pytest.approx(
        0.5 + len(screen.rows) * 0.7 + 1.0)
    out = tmp_path / "list.mp4"
    record(rec, out)
    frames = _extract_frames(out, tmp_path / "frames")
    still, _d = render_list_screen(screen)
    _assert_frame_like(frames[-1], still)
    assert _mean_abs_diff(frames[0], frames[-1]) > 5.0


# ── status / story: the segment model ────────────────────


def test_status_recording_fills_to_the_previewed_point(
        tmp_path):
    screen = demo_status_screen("whatsapp-status")
    assert (screen.segments, screen.index, screen.played) \
        == (4, 1, 0.45)
    rec = build_status_recording(screen)
    # 0.5 opening + 5.0 first segment + 5.0 × 0.45 second
    # + 1.0 ending
    assert rec.duration_s == pytest.approx(0.5 + 5.0 + 2.25 + 1.0)
    out = tmp_path / "status.mp4"
    result = record(rec, out)
    assert result["duration_s"] == pytest.approx(
        rec.duration_s, abs=0.35)
    frames = _extract_frames(out, tmp_path / "frames")
    still, _d = render_status_screen(screen)
    _assert_frame_like(frames[-1], still)


def test_story_recording_advances_segments(tmp_path):
    screen = demo_story_screen("instagram-story")
    assert (screen.segments, screen.index, screen.played) \
        == (3, 1, 0.4)
    rec = build_story_recording(screen)
    assert rec.duration_s == pytest.approx(0.5 + 5.0 + 2.0 + 1.0)
    out = tmp_path / "story.mp4"
    record(rec, out)
    frames = _extract_frames(out, tmp_path / "frames")
    still, drawn = render_story_screen(screen)
    _assert_frame_like(frames[-1], still)
    # the segment advance shows in the progress bar —
    # the medium is one photo, so the whole-frame
    # difference is small and the bar is the signal
    prog = drawn["regions"]["progress"]
    pbox = (int(prog[0]), int(prog[1]),
            int(prog[0] + prog[2]), int(prog[1] + prog[3]))
    assert _mean_abs_diff(
        frames[0].crop(pbox), frames[-1].crop(pbox)) > 1.0


# ── calls ────────────────────────────────────────────────


def test_call_recording_advances_the_timer(tmp_path):
    screen = demo_call_screen("call")
    assert screen.state == "ongoing"
    assert screen.status == "02:15"
    rec = build_call_recording(screen)
    assert rec.duration_s == pytest.approx(0.5 + 4.0 + 1.0)
    out = tmp_path / "call.mp4"
    record(rec, out)
    frames = _extract_frames(out, tmp_path / "frames")
    still, _d = render_call_screen(
        replace(screen, status="02:19"))
    _assert_frame_like(frames[-1], still)


def test_incoming_call_recording_holds_ringing():
    screen = demo_call_screen("call-incoming")
    assert screen.state == "incoming"
    rec = build_call_recording(screen)
    assert rec.duration_s == pytest.approx(0.5 + 4.0 + 1.0)
    assert len(rec.timeline) == 3


# ── scrolling pages ──────────────────────────────────────


def test_profile_scroll_records_top_to_bottom(tmp_path):
    screen = demo_profile_screen("instagram-profile")
    items = [ProfileItem("i" + str(i), kind="photo")
             for i in range(18)]
    screen = replace(screen, items=items)
    still, drawn = render_profile_screen(screen)
    full = Image.open(io.BytesIO(still)).convert("RGB")
    assert full.height > 1600  # taller than one phone screen
    rec = build_profile_recording(screen)
    assert rec.kind == "scroll"
    assert rec.size == (800, 1600)
    assert rec.duration_s == pytest.approx(
        0.5 + 0.6 + 4.0 + 0.6 + 1.0)
    out = tmp_path / "profile.mp4"
    record(rec, out)
    frames = _extract_frames(out, tmp_path / "frames")
    regions = drawn["regions"]
    pin_top = int(regions["status_bar"][1]
                  + regions["status_bar"][3])
    pin_bottom = int(regions["nav"][3])
    # chrome stays pinned: the top and bottom strips are
    # identical between the first and last frame
    assert _mean_abs_diff(
        frames[0].crop((0, 0, 800, pin_top)),
        frames[-1].crop((0, 0, 800, pin_top))) <= 5.0
    assert _mean_abs_diff(
        frames[0].crop((0, 1600 - pin_bottom, 800, 1600)),
        frames[-1].crop((0, 1600 - pin_bottom, 800,
                         1600))) <= 5.0
    # the last frame's window shows the bottom of the
    # page (the rows the still carries at its foot)
    max_offset = full.height - 1600
    assert _mean_abs_diff(
        frames[-1].crop((0, pin_top, 800,
                         1600 - pin_bottom)),
        full.crop((0, pin_top + max_offset, 800,
                   full.height - pin_bottom))) <= 5.0
    assert _mean_abs_diff(frames[0], frames[-1]) > 5.0


def test_short_page_records_as_a_hold():
    screen = demo_profile_screen("instagram-profile")
    still, _d = render_profile_screen(screen)
    full = Image.open(io.BytesIO(still)).convert("RGB")
    assert full.height <= 1600
    rec = build_profile_recording(screen)
    assert len(rec.timeline) == 1
    assert rec.size == tuple(full.size)


def test_comment_sheet_scroll_pins_its_header(tmp_path):
    screen = demo_comment_screen()
    rec = build_comment_recording(screen)
    assert rec.kind == "scroll"
    out = tmp_path / "comments.mp4"
    record(rec, out)
    frames = _extract_frames(out, tmp_path / "frames")
    assert _mean_abs_diff(frames[0], frames[-1]) > 5.0
    # the sheet's count header stays pinned: the top
    # strip is identical between the first and last frame
    still, drawn = render_comment_screen(screen)
    top = drawn["elements"][0]["box"]
    pin = int(top[1])
    assert _mean_abs_diff(
        frames[0].crop((0, 0, 800, pin)),
        frames[-1].crop((0, 0, 800, pin))) <= 5.0


# ── the /mock record surface ─────────────────────────────


def test_parse_record_forms():
    screen, mode = parse_mock_tail("record demo whatsapp")
    assert mode == "record-demo"
    screen, mode = parse_mock_tail(
        "record demo whatsapp-list")
    assert mode == "record-demo-list"
    screen, mode = parse_mock_tail(
        "record whatsapp me: hi ;; Adaeze: hello")
    assert mode == "record-built"
    with pytest.raises(MockScreensError):
        parse_mock_tail("record record demo whatsapp")
    with pytest.raises(MockScreensError):
        parse_mock_tail("record")


def test_record_command_delivers_a_video(tmp_path):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock(
        "mock", "record demo whatsapp", _message())
    assert reply == ""
    assert len(runtime.send.calls) == 1
    ref, caption = runtime.send.calls[0]
    assert ref.kind == "video"
    assert ref.mime == "video/mp4"
    assert "recording" in caption
    assert Path(ref.path).exists()
    assert probe_duration(ref.path) == pytest.approx(11.4,
                                                     abs=0.35)


def test_record_list_command_delivers_a_video(tmp_path):
    runtime = _Runtime(tmp_path)
    reply = runtime._control_mock(
        "mock", "record demo whatsapp-list", _message())
    assert reply == ""
    ref, _caption = runtime.send.calls[0]
    assert ref.kind == "video"
    assert Path(ref.path).exists()


def test_record_command_is_owner_gated(tmp_path):
    runtime = _Runtime(tmp_path, operator=False)
    reply = runtime._control_mock(
        "mock", "record demo whatsapp", _message())
    assert reply == "that one's just for the owner."
    assert runtime.send.calls == []


def test_service_record_chat_delivers(tmp_path):
    service = MockScreenService(media_dir=tmp_path / "media")
    out = service.record_chat(
        demo_screen("telegram"), send=_send_stub())
    assert out["sent"] is True
    assert out["kind"] == "chat"
    assert Path(out["path"]).exists()
    assert out["duration_s"] == pytest.approx(
        out["expected_s"], abs=0.35)


def test_service_record_honours_a_broken_voice(tmp_path):
    service = MockScreenService(media_dir=tmp_path / "media")
    screen = _voice_screen(tmp_path / "missing.ogg")
    out = service.record_chat(screen, send=_send_stub())
    assert out["sent"] is False
    assert "could not be used" in out["reason"]


def test_service_recording_cache_is_content_keyed(tmp_path):
    service = MockScreenService(media_dir=tmp_path / "media")
    one = demo_screen("whatsapp")
    first = service.record_chat(one)
    assert first["sent"] is False
    assert Path(first["path"]).exists()
    again = service.record_chat(one)
    assert again["path"] == first["path"]
    # the same screen reuses the encode: the file is
    # untouched by the second call
    assert Path(again["path"]).stat().st_mtime_ns == \
        Path(first["path"]).stat().st_mtime_ns
    # a different screen — same shape, different text —
    # must never share the first screen's file
    other = replace(one, title=one.title + " (2)")
    changed = service.record_chat(other)
    assert changed["path"] != first["path"]
    assert Path(changed["path"]).exists()


# ── census accounting ────────────────────────────────────

#: Phase 4 tools the census's §9.2 animation table names
CENSUS_RECORDED = {
    "WhatsApp chat": "chat",
    "Instagram DM": "chat",
    "WhatsApp Status": "status",
    "Instagram Story": "story",
    "Instagram Chat List": "list",
    "Telegram Chat List": "list",
    "Profiles / comment sheets": "scroll",
}

#: census-named scope held back, with its census reason —
#: never filled by guessing
CENSUS_REMAINDERS = {
    # Telegram list swipe-by-swipe scrolling with pauses
    # (§9.2): a distinct choreography from row arrival;
    # shipped as row arrival, the scroll form named here.
    "telegram-list swipe scroll": "census §9.2 names "
    "swipe-by-swipe scrolling with pauses for the Telegram "
    "list; this phase ships row arrival for all lists",
    # Recorded video has no sound (census §5.9): voice
    # notes play visually at true length; audio is not
    # muxed into the mp4.
    "recorded audio": "census §5.9: recorded video has "
    "no sound — the voice playhead advances for the true "
    "duration, the mp4 stays silent",
}
