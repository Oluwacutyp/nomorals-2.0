"""Acceptance tests for the live voice loop (Prompt 10).

Covers the energy VAD (calibration, no onset clipping, prebuffer), the
single-reader mic capture thread, output shaping, consent/stats stores,
encrypted audio retention, wav splitting, the bridge STT adapter, and full
session runs with fake hardware (barge-in, end words, consent refusal).
"""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

from nomorals.voice.session import (
    CHUNK_BYTES,
    ConsentStore,
    EnergyVAD,
    MicCapture,
    SessionReport,
    StatsStore,
    VoiceSession,
    _rms_int16,
    decrypt_kept_audio,
    int_to_words,
    make_bridge_stt,
    read_wav_bytes,
    speakify,
    split_wav,
    stt_supports_partial,
    write_wav_bytes,
)


def _silence() -> bytes:
    return b"\x00" * CHUNK_BYTES


def _tone(amplitude: int = 3000) -> bytes:
    # constant-amplitude int16 chunk: RMS == amplitude, well above threshold
    return (amplitude.to_bytes(2, "little", signed=True)) * (CHUNK_BYTES // 2)


class FakeMic:
    sample_rate = 16000

    def __init__(self, chunks):
        self._chunks = list(chunks)
        self.closed = False

    def read_chunk(self):
        if not self._chunks:
            return None
        return self._chunks.pop(0)

    def close(self):
        self.closed = True


class FakeSpeaker:
    """playing() True `true_count` times, then False."""

    def __init__(self, true_count: int = 0):
        self._n = true_count
        self.stopped = []
        self.played = []

    def play(self, wav_path):
        self.played.append(wav_path)
        return 1

    def playing(self, token):
        if self._n > 0:
            self._n -= 1
            return True
        return False

    def stop(self, token):
        self.stopped.append(token)

    def close(self):
        pass


def _fake_tts(tmpdir):
    def _tts(text, profile):
        path = os.path.join(tmpdir, f"tts-{len(os.listdir(tmpdir))}.wav")
        write_wav_bytes(path, _tone() * 10)  # 300ms of tone
        return {"path": path}

    return _tts


class TestEnergyVAD(unittest.TestCase):
    def test_silence_then_speech(self):
        vad = EnergyVAD(silence_ms=300, min_speech_ms=90)
        for _ in range(9):
            self.assertEqual(vad.observe(_silence()), "undecided")
        # 300ms of quiet (10 chunks) → declared silence
        self.assertEqual(vad.observe(_silence()), "silence")
        vad.reset()
        # 90ms of speech = 3 chunks at 30ms
        self.assertEqual(vad.observe(_tone()), "undecided")
        self.assertEqual(vad.observe(_tone()), "undecided")
        self.assertEqual(vad.observe(_tone()), "speech")

    def test_calibrate_learns_loud_room(self):
        vad = EnergyVAD()
        # build 400-RMS ambient chunks
        ambient = b"".join(
            [(400).to_bytes(2, "little", signed=True)] * (CHUNK_BYTES // 2))
        self.assertGreater(_rms_int16(ambient), 300)
        vad.calibrate([ambient] * 10)
        # floor must sit above the room: 400-RMS is now silence
        self.assertFalse(vad.is_speech(ambient))
        self.assertGreater(vad.noise_floor, 200.0)

    def test_speech_never_raises_floor(self):
        """Regression: loud speech must not drag the threshold up behind
        itself and clip the start of the utterance."""
        vad = EnergyVAD()
        before = vad.noise_floor
        for _ in range(30):
            self.assertTrue(vad.is_speech(_tone()))
        self.assertLessEqual(vad.noise_floor, before)
        # ...and speech still classifies as speech afterwards
        self.assertTrue(vad.is_speech(_tone()))

    def test_quiet_room_floor_relaxes_down(self):
        vad = EnergyVAD(floor=2000.0)
        for _ in range(50):
            vad.is_speech(_silence())
        self.assertLess(vad.noise_floor, 2000.0)


class TestPrebuffer(unittest.TestCase):
    def test_onset_is_preserved(self):
        # 16 calibration + 5 silence + 6 tone + 30 silence + end
        chunks = [_silence()] * 21 + [_tone()] * 6 + [_silence()] * 30
        mic = FakeMic(chunks)
        capture = MicCapture(mic)
        capture.start()
        try:
            vad = EnergyVAD(silence_ms=800)
            for _ in range(16):
                capture.listen_next()
            session = VoiceSession(
                think=lambda t: "", stt=lambda p: "", tts=lambda t, p: {},
                mic=mic, data_dir=tempfile.mkdtemp())
            heard = session._listen_utterance(vad, capture)
        finally:
            capture.stop()
        self.assertIsNotNone(heard)
        pcm, _ts = heard
        # 6 speech chunks + 10 prebuffer chunks (300ms) worth of audio;
        # without the prebuffer we'd only get the 6 post-debounce chunks.
        self.assertGreaterEqual(len(pcm), 6 * CHUNK_BYTES)
        self.assertLessEqual(len(pcm), 16 * CHUNK_BYTES)


class TestMicCapture(unittest.TestCase):
    def test_single_reader_gets_everything_in_order(self):
        chunks = [_silence()] * 5 + [_tone()] * 3
        mic = FakeMic(chunks)
        capture = MicCapture(mic)
        capture.start()
        try:
            got = [capture.listen_next() for _ in range(8)]
            self.assertIsNone(capture.listen_next())  # stream end
        finally:
            capture.stop()
        self.assertEqual(got, chunks)

    def test_poll_timeout_returns_none(self):
        mic = FakeMic([])  # immediate stream end
        capture = MicCapture(mic)
        capture.start()
        try:
            self.assertIsNone(capture.listen_next())
            self.assertIsNone(capture.poll(timeout=0.05))
        finally:
            capture.stop()

    def test_stream_end_is_sticky(self):
        """After the terminal None, listen_next returns None at once —
        it must never block on a dead thread's empty queue."""
        mic = FakeMic([_silence()])
        capture = MicCapture(mic)
        capture.start()
        try:
            self.assertIsNotNone(capture.listen_next())
            self.assertIsNone(capture.listen_next())   # terminal
            self.assertIsNone(capture.listen_next())   # sticky, no hang
            # the barge-in poll must not swallow the end marker either
            self.assertIsNone(capture.poll(timeout=0.05))
            self.assertIsNone(capture.listen_next())
        finally:
            capture.stop()


class TestSpeakify(unittest.TestCase):
    def test_code_blocks_become_a_pointer(self):
        plan = speakify("here:\n```python\nx = 1\n```\ndone")
        self.assertNotIn("x = 1", plan.spoken)
        self.assertIn("code snippet", plan.spoken)
        self.assertTrue(plan.had_code)
        self.assertIn("x = 1", plan.full)

    def test_naira_amounts_read_as_words(self):
        plan = speakify("it costs ₦50,000")
        self.assertIn("fifty thousand naira", plan.spoken)

    def test_long_text_truncates_with_pointer(self):
        plan = speakify("First sentence. " + "word " * 500, cap_secs=60)
        self.assertTrue(plan.truncated)
        self.assertIn("full version to chat", plan.spoken)

    def test_lists_get_cadence(self):
        plan = speakify("- apples\n- oranges")
        self.assertIn("first, apples.", plan.spoken)
        self.assertIn("second, oranges.", plan.spoken)

    def test_int_to_words(self):
        self.assertEqual(int_to_words(0), "zero")
        self.assertEqual(int_to_words(42), "forty-two")
        self.assertEqual(int_to_words(1500), "one thousand five hundred")


class TestConsentAndStats(unittest.TestCase):
    def test_consent_grant_revoke(self):
        with tempfile.TemporaryDirectory() as d:
            store = ConsentStore(d)
            self.assertFalse(store.consented("mic1"))
            store.grant("mic1")
            self.assertTrue(store.consented("mic1"))
            # survives reload
            self.assertTrue(ConsentStore(d).consented("mic1"))
            store.revoke("mic1")
            self.assertFalse(store.consented("mic1"))
            self.assertFalse(ConsentStore(d).consented("mic1"))

    def test_stats_summary_percentiles(self):
        with tempfile.TemporaryDirectory() as d:
            store = StatsStore(d)
            report = SessionReport(session_id="s1", turns=2, barge_ins=1,
                                   latencies_ms=[100.0, 200.0, 300.0, 400.0])
            store.record_session(report)
            summary = StatsStore(d).summary()
            self.assertEqual(summary["sessions"], 1)
            self.assertEqual(summary["turns"], 2)
            self.assertEqual(summary["barge_ins"], 1)
            self.assertEqual(summary["ear_to_ear_ms_p50"], 300.0)
            self.assertEqual(summary["ear_to_ear_ms_p95"], 400.0)


class TestSplitWav(unittest.TestCase):
    def test_short_wav_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "short.wav")
            write_wav_bytes(path, _silence() * 100)  # 3s
            self.assertEqual(split_wav(path, max_secs=120), [path])

    def test_long_wav_splits(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "long.wav")
            # 130s of silence at 16kHz int16 mono
            write_wav_bytes(path, b"\x00" * (16000 * 130 * 2))
            segs = split_wav(path, max_secs=120)
            self.assertEqual(len(segs), 2)
            total = sum(os.path.getsize(s) for s in segs)
            self.assertGreater(total, 16000 * 125 * 2)


class TestEncryptedRetention(unittest.TestCase):
    def test_encrypted_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            key = os.urandom(32)
            session = VoiceSession(
                think=lambda t: "", stt=lambda p: "", tts=lambda t, p: {},
                data_dir=d, keep_audio=True, audio_key=key)
            pcm = _tone() * 20
            kept = session._store_utterance(pcm, turn=0)
            self.assertTrue(kept.endswith(".wav.enc"))
            self.assertTrue(Path(kept).exists())
            # no plaintext copy left behind
            self.assertEqual(list(Path(d).rglob("utt-*.wav")), [])
            self.assertEqual(decrypt_kept_audio(kept, key), pcm)
            self.assertEqual(session.purge_audio(), 1)
            self.assertFalse(Path(kept).exists())

    def test_keep_audio_without_key_stays_plain(self):
        with tempfile.TemporaryDirectory() as d:
            session = VoiceSession(
                think=lambda t: "", stt=lambda p: "", tts=lambda t, p: {},
                data_dir=d, keep_audio=True)
            kept = session._store_utterance(_tone() * 5, turn=0)
            self.assertTrue(kept.endswith(".wav"))
            pcm_back, _rate = read_wav_bytes(kept)
            self.assertEqual(pcm_back, _tone() * 5)


class TestBridgeSTT(unittest.TestCase):
    def test_async_bridge_adapts_to_sync(self):
        class FakeBridge:
            async def transcribe_voice(self, path, *, language="en"):
                await asyncio.sleep(0)
                self.seen = (path, language)
                return "hello world"

        bridge = FakeBridge()
        stt = make_bridge_stt(bridge, language="yo")
        self.assertEqual(stt("/tmp/x.wav"), "hello world")
        self.assertEqual(bridge.seen, ("/tmp/x.wav", "yo"))

    def test_partial_transcript_honesty(self):
        stt = make_bridge_stt(object())
        self.assertFalse(stt_supports_partial(stt))
        self.assertFalse(stt_supports_partial(lambda p: ""))


def _session_script(*chunks):
    return list(chunks)


class TestSessionRun(unittest.TestCase):
    def _run(self, mic_chunks, stt_texts, speaker_true=0, max_turns=1,
             keep_audio=False):
        tmp = tempfile.mkdtemp()
        mic = FakeMic(mic_chunks)
        speaker = FakeSpeaker(true_count=speaker_true)
        delivered = []
        stt_calls = {"n": 0}

        def _stt(path):
            stt_calls["n"] += 1
            return stt_texts[min(stt_calls["n"] - 1, len(stt_texts) - 1)]

        session = VoiceSession(
            think=lambda t: f"reply to {t}",
            stt=_stt,
            tts=_fake_tts(tmp),
            mic=mic,
            speaker=speaker,
            deliver_text=lambda tid, text: delivered.append((tid, text)),
            data_dir=tmp,
            keep_audio=keep_audio,
        )
        report = session.run(max_turns=max_turns,
                             ask_consent=lambda: True)
        return session, report, delivered, speaker, tmp

    def _utterance(self):
        # 5 silence + 6 tone (speech) + 30 silence (>= 26 silence chunks)
        return [_silence()] * 5 + [_tone()] * 6 + [_silence()] * 30

    def test_one_turn_happy_path(self):
        chunks = [_silence()] * 16 + self._utterance()
        session, report, delivered, speaker, tmp = self._run(
            chunks, ["hello"], max_turns=1)
        self.assertEqual(report.turns, 1)
        self.assertEqual(report.end_reason, "completed")
        self.assertEqual(report.error, "")
        # voice summarizes, chat gets the full text
        self.assertEqual(len(delivered), 1)
        self.assertIn("reply to hello", delivered[0][1])
        self.assertTrue(speaker.played)
        self.assertFalse(session.is_recording)
        self.assertEqual(len(report.latencies_ms), 1)

    def test_no_consent_no_recording(self):
        tmp = tempfile.mkdtemp()
        session = VoiceSession(
            think=lambda t: "", stt=lambda p: "", tts=lambda t, p: {},
            data_dir=tmp)
        report = session.run(max_turns=1, ask_consent=lambda: False)
        self.assertEqual(report.end_reason, "no-consent")
        self.assertEqual(report.turns, 0)

    def test_end_word_hangs_up(self):
        chunks = [_silence()] * 16 + self._utterance()
        session, report, delivered, _sp, _tmp = self._run(
            chunks, ["goodbye"], max_turns=0)
        self.assertEqual(report.end_reason, "user-exit")
        self.assertEqual(report.turns, 0)
        self.assertEqual(delivered, [])

    def test_barge_in_counts_and_continues(self):
        chunks = (
            [_silence()] * 16
            + self._utterance()          # turn 1
            + [_tone()] * 8             # owner talks over the reply
            + [_silence()] * 5
            + self._utterance()          # turn 2 after the barge
        )
        session, report, delivered, speaker, tmp = self._run(
            chunks, ["hello", "goodbye"], speaker_true=12, max_turns=0)
        self.assertEqual(report.barge_ins, 1)
        self.assertEqual(report.turns, 1)
        self.assertEqual(report.end_reason, "user-exit")
        self.assertTrue(speaker.stopped)  # playback was actually cut

    def test_is_recording_reflects_state(self):
        tmp = tempfile.mkdtemp()
        session = VoiceSession(
            think=lambda t: "", stt=lambda p: "", tts=lambda t, p: {},
            data_dir=tmp)
        self.assertFalse(session.is_recording)
        session._set_state("listening")
        self.assertTrue(session.is_recording)
        session._set_state("thinking")
        self.assertFalse(session.is_recording)
        session._set_state("speaking")
        self.assertTrue(session.is_recording)


class TestSpokenCommand(unittest.TestCase):
    """A voice note that transcribes to /command takes the typed path."""

    def _brain(self, transcript: str):
        from nomorals.agents.partner_runtime import PartnerBrain

        brain = PartnerBrain.__new__(PartnerBrain)
        brain._transcribe_media_note = (
            lambda media: f"voice note (transcribed) — {transcript}"
            if transcript else "")
        return brain

    def _msg(self, **kwargs):
        from nomorals.social.chat.base import ChatMessage, ChatRef, MediaRef

        return ChatMessage(chat=ChatRef(platform="t", chat_id="c"),
                           incoming=True, **kwargs)

    def test_command_transcript_returned(self):
        from nomorals.agents.partner_runtime import PartnerBrain
        from nomorals.social.chat.base import MediaRef

        brain = self._brain("/speak hello there")
        msg = self._msg(text="", media=[MediaRef(path="/tmp/x.ogg",
                                                 kind="voice")])
        self.assertEqual(PartnerBrain._spoken_command(brain, msg),
                         "/speak hello there")

    def test_chatter_transcript_ignored(self):
        from nomorals.agents.partner_runtime import PartnerBrain
        from nomorals.social.chat.base import MediaRef

        brain = self._brain("what's the weather like")
        msg = self._msg(text="", media=[MediaRef(path="/tmp/x.ogg",
                                                 kind="audio")])
        self.assertEqual(PartnerBrain._spoken_command(brain, msg), "")

    def test_no_media_noop(self):
        from nomorals.agents.partner_runtime import PartnerBrain

        brain = self._brain("/speak hi")
        msg = self._msg(text="plain typed text")
        self.assertEqual(PartnerBrain._spoken_command(brain, msg), "")


if __name__ == "__main__":
    unittest.main()
