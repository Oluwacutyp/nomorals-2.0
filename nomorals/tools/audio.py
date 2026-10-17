"""Voice I/O: text-to-speech and speech-to-text.

Two planes, both optional — the system runs with no audio at all:

**TTS** tries local engines first (no account, no network):
``espeak-ng`` (Termux: ``pkg install espeak-ng``), ``pico2wave``, macOS ``say``,
the ``edge-tts`` Python package, then an OpenAI-compatible ``/audio/speech``
endpoint when one is configured.

**STT** mirrors that: an OpenAI-compatible ``/audio/transcriptions``
endpoint (whisper-1 et al.) or a local ``whisper.cpp`` binary.

Generated audio lands in ``<NM_HOME>/<audio.audio_dir>`` and the ``speak``
tool returns the path — the chat layer decides whether to send it as a
file (``gateway.send_file``) or just report the path.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..core.text import safe_filename

__all__ = ["detect_tts_engines", "tts", "stt", "audio_info", "register",
           "VoiceTagger", "VOICE_TAGGER_NATIVE_ENGINES",
           "wav_to_voice_ogg"]

_log = get_logger(__name__)

_TTS_TIMEOUT = 120.0
_STT_TIMEOUT = 300.0

# Audio MIME by file extension for STT uploads. Telegram voice notes are
# .ogg Opus; sending them as application/octet-stream makes some
# endpoints reject the part as an unknown binary type.
_AUDIO_MIME_BY_EXT = {
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".wav": "audio/wav",
    ".wave": "audio/wav",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".webm": "audio/webm",
    ".flac": "audio/flac",
    ".aac": "audio/aac",
}

# ── TTS pacing + expressive tags ──────────────────────────────────────────
# /tts goes through this module (speak tool), NOT the neural UniversalTTS
# engine (/speak, /voice say). These engines have no native tag support, so
# tags must never reach them literally — "[laugh]" spoken as words is a
# bug, not a feature. Rules:
#   * [pause] / [pause:MS]  -> real silence (default 500ms for bare [pause])
#   * [laugh] [cough] [sigh] ... (sound tags) -> DROPPED (these engines
#     cannot laugh; use /speak for real laughs via Chatterbox/Bark)
#   * [happy] [whisper] ... (emotion tags) -> stripped (no emotion control)
#   * unknown [bracket text] -> left as literal text (never eat real words)
# Pacing profiles insert breathing room at sentence/comma/paragraph
# boundaries. "tight" = legacy behaviour (no added pauses).

#: Bare [pause] with no :MS means this much silence.
DEFAULT_PAUSE_MS = 500

#: Sentence/comma/paragraph pause (ms) per pacing profile.
PACE_PROFILES: dict[str, dict[str, int]] = {
    "tight": {"sentence": 0, "comma": 0, "paragraph": 0},
    "natural": {"sentence": 250, "comma": 120, "paragraph": 500},
    "slow": {"sentence": 450, "comma": 200, "paragraph": 900},
}

#: Words whose trailing period never ends a sentence (no pause after).
_PACING_ABBREV = frozenset({
    "mr", "mrs", "ms", "dr", "st", "sr", "jr", "vs", "etc",
    "e.g", "i.e", "fig", "no", "vol", "pp", "u.s", "u.k", "a.m", "p.m",
})

_STRICT_TAG_RE = re.compile(r"\[([a-z_]+)(?::(\d+))?\]")

# Words-per-minute for the local engines at speed 1.0.
_ESPEAK_BASE_WPM = 175


def _known_tts_tags() -> tuple[frozenset[str], frozenset[str]]:
    """(emotion_tags, sound_tags) from the shared TagProcessor vocabulary.

    Lazy import: voice/tts.py is heavy; tools/audio.py must stay
    importable without it.
    """
    from ..voice.tts import TagProcessor

    return frozenset(TagProcessor.EMOTION_TAGS), frozenset(TagProcessor.SOUND_TAGS)


def chunk_tts_text(text: str, pace: str = "natural") -> tuple[list[tuple[str, int]], list[str]]:
    """Split TTS input into speakable chunks with pauses.

    Returns ``(chunks, dropped_sounds)`` where ``chunks`` is a list of
    ``(text, pause_after_ms)``. All known tags are consumed (never left
    in the speakable text); unknown ``[bracket text]`` is preserved.
    Sentence/comma/paragraph pauses come from the ``pace`` profile;
    explicit ``[pause:MS]`` adds its silence at that position.

    Pure — no I/O, fully testable.
    """
    profile = PACE_PROFILES.get((pace or "natural").lower())
    if profile is None:
        raise ToolError(
            f"unknown pace {pace!r} — use one of "
            f"{', '.join(sorted(PACE_PROFILES))}")
    emotion_tags, sound_tags = _known_tts_tags()
    chunks: list[tuple[str, int]] = []
    dropped: list[str] = []
    pending_pause = 0  # explicit [pause] before any text = leading silence

    def _emit(run: str, extra_pause: int) -> None:
        """Split one tag-free text run into pacing chunks."""
        nonlocal pending_pause
        if not run.strip():
            return
        # Paragraph breaks first (blank line = paragraph pause).
        for para_i, para in enumerate(re.split(r"\n\s*\n", run)):
            if para_i > 0 and chunks:
                text, pause = chunks[-1]
                chunks[-1] = (text, pause + profile["paragraph"])
            elif para_i > 0 and pending_pause == 0 and not chunks:
                pending_pause = profile["paragraph"]
            _split_sentences(para, extra_pause if para_i == len(re.split(r"\n\s*\n", run)) - 1 else 0)

    def _split_sentences(para: str, extra_pause: int) -> None:
        nonlocal pending_pause
        # Split after sentence-ending punctuation followed by space +
        # uppercase/quote/EOS, skipping abbreviations; commas/semicolons/
        # colons/em-dashes get the comma pause.
        tokens = re.split(r"(\n+)", para)
        for tok in tokens:
            if not tok.strip():
                continue
            if tok.startswith("\n"):
                continue
            pos = 0
            # Find split points: (index_after_boundary, kind)
            splits: list[tuple[int, str]] = []
            for m in re.finditer(r"([.!?…]+|[;,:\u2014])\s+", tok):
                kind = "sentence" if m.group(1)[0] in ".!?\u2026" else "comma"
                if kind == "sentence":
                    # Abbreviation check: word before the period.
                    before = tok[:m.start()]
                    word = re.split(r"[\s\"'(\u201c]+", before)[-1].lower().rstrip(".")
                    if word in _PACING_ABBREV:
                        continue
                    # Initial chain (J. R. R. Tolkien): a single letter
                    # followed by another "X. " is not a sentence end.
                    # The pronoun "I." does NOT match (next word has no
                    # period), so "I went home. I left." still splits.
                    if (len(word) == 1
                            and re.match(r"[A-Za-z]\.\s",
                                         tok[m.end():])):
                        continue
                    # Lowercase starts DO split: casual text
                    # ("hello. world") is the common case.
                splits.append((m.end(), kind))
            last = 0
            for end, kind in splits:
                piece = tok[last:end].strip()
                if piece:
                    pause = profile[kind]
                    if pending_pause:
                        # Leading explicit pause merges into the first chunk's
                        # preceding silence: emit as its own silence chunk.
                        chunks.append(("", pending_pause))
                        pending_pause = 0
                    chunks.append((piece, pause))
                last = end
            tail = tok[last:].strip()
            if tail:
                if pending_pause:
                    chunks.append(("", pending_pause))
                    pending_pause = 0
                chunks.append((tail, extra_pause))

    pos = 0
    for m in _STRICT_TAG_RE.finditer(text or ""):
        run = text[pos:m.start()]
        name, ms = m.group(1), m.group(2)
        if name == "pause":
            _emit(run, 0)
            pause_ms = int(ms) if ms else DEFAULT_PAUSE_MS
            if chunks:
                t, p = chunks[-1]
                chunks[-1] = (t, p + pause_ms)
            else:
                pending_pause += pause_ms
        elif name in sound_tags:
            _emit(run, 0)
            dropped.append(name)
        elif name in emotion_tags:
            _emit(run, 0)
            # Emotion tags carry no audio on these engines — strip silently.
        else:
            # Unknown tag: leave the bracket text in the speakable output.
            _emit(run + m.group(0), 0)
        pos = m.end()
    _emit(text[pos:], 0)
    if pending_pause:
        chunks.insert(0, ("", pending_pause))
    # Merge adjacent chunks with no pause between them (avoids one API
    # call per clause when pacing is tight).
    merged: list[tuple[str, int]] = []
    for t, p in chunks:
        if t == "" and merged:
            tt, pp = merged[-1]
            merged[-1] = (tt, pp + p)
        elif t == "" and not merged:
            merged.append((t, p))
        elif merged and merged[-1][1] == 0 and merged[-1][0]:
            tt, _ = merged[-1]
            merged[-1] = (tt + " " + t, p)
        else:
            merged.append((t, p))
    return [(t, p) for t, p in merged if t.strip() or p > 0], dropped


def _concat_wav_chunks(wav_blobs: list[bytes], pauses_ms: list[int],
                       sample_rate: int, leading_silence_ms: int = 0) -> bytes:
    """Concatenate WAV blobs (same params) with silence gaps, stdlib only.

    ``pauses_ms[i]`` is the silence after chunk ``i`` (last entry ignored).
    ``leading_silence_ms`` is silence before the first chunk (from leading
    ``[pause]`` tags). All chunks must share channels/samplewidth/framerate;
    the first chunk's params win.
    """
    import io

    out = io.BytesIO()
    params = None
    with wave.open(out, "wb") as wout:
        for i, blob in enumerate(wav_blobs):
            with wave.open(io.BytesIO(blob), "rb") as win:
                if params is None:
                    params = win.getparams()
                    wout.setparams(params)
                    if leading_silence_ms > 0:
                        n = int(sample_rate * leading_silence_ms / 1000)
                        wout.writeframes(
                            b"\x00" * (n * params.nchannels * params.sampwidth))
                frames = win.readframes(win.getnframes())
            wout.writeframes(frames)
            if i < len(pauses_ms) and pauses_ms[i] > 0:
                n_silence = int(sample_rate * pauses_ms[i] / 1000)
                n_channels = params.nchannels if params else 1
                sampwidth = params.sampwidth if params else 2
                wout.writeframes(b"\x00" * (n_silence * n_channels * sampwidth))
    return out.getvalue()


def _fold_empty_chunks(chunks: list[tuple[str, int]]) -> tuple[list[tuple[str, int]], int]:
    """Fold empty-text chunks (pure [pause] tags) into neighbours.

    Returns ``(text_chunks, leading_silence_ms)``: an empty chunk's pause
    joins the previous text chunk's pause-after, or becomes leading
    silence when it precedes all text.
    """
    text_chunks: list[tuple[str, int]] = []
    leading = 0
    carry = 0
    for ctext, pause_ms in chunks:
        if not ctext.strip():
            carry += pause_ms
            continue
        if carry:
            if text_chunks:
                t, p = text_chunks[-1]
                text_chunks[-1] = (t, p + carry)
            else:
                leading += carry
            carry = 0
        text_chunks.append((ctext, pause_ms))
    if carry and text_chunks:
        t, p = text_chunks[-1]
        text_chunks[-1] = (t, p + carry)
    elif carry:
        leading += carry
    return text_chunks, leading


def _wav_sample_rate(blob: bytes) -> int:
    import io

    with wave.open(io.BytesIO(blob), "rb") as w:
        return w.getframerate()


#: What each /tts engine does with tags — the honest capability table.
#: Shared by the /tts engines report and help. "pause" = [pause]/[pause:MS]
#: becomes real silence; "sound" = [laugh]/[cough]/... dropped (no engine
#: here can laugh); "emotion" = [happy]/... stripped. Native-tag engines
#: (Chatterbox/Bark) live under /speak, not /tts.
TTS_ENGINE_TAG_SUPPORT: dict[str, dict[str, str]] = {
    "espeak_ng": {
        "pause": "SSML <break time>",
        "sound": "dropped — espeak-ng cannot laugh",
        "emotion": "stripped",
        "speed": "-s (wpm)",
        "gap": "-g (word gap, 10ms units)",
    },
    "pico2wave": {
        "pause": "silence spliced between chunks",
        "sound": "dropped — cannot laugh",
        "emotion": "stripped",
        "speed": "not supported",
        "gap": "not supported",
    },
    "say": {
        "pause": "[[slnc N]]",
        "sound": "dropped — cannot laugh",
        "emotion": "stripped",
        "speed": "-r (wpm)",
        "gap": "not supported",
    },
    "edge_tts": {
        "pause": "SSML <break time>",
        "sound": "dropped — cannot laugh",
        "emotion": "stripped",
        "speed": "SSML prosody rate",
        "gap": "not supported",
    },
    "openai_compat": {
        "pause": "silence spliced between wav chunks",
        "sound": "dropped — API cannot laugh",
        "emotion": "stripped",
        "speed": "API speed param (0.25-4.0)",
        "gap": "not supported",
    },
}


#: Engines whose text pipeline passes bracket tags through natively
#: (Chatterbox turbo/nano speak [laugh]/[chuckle]/[cough]; Bark speaks
#: its own tag vocabulary). Everything else gets tags stripped /
#: converted by the chunker — VoiceTagger must not emit native-only
#: tags for them. Unknown engines default to chunked (safe).
VOICE_TAGGER_NATIVE_ENGINES: frozenset[str] = frozenset({"chatterbox", "bark"})

#: Laughter cues -> [laugh] on native-tag engines. The cue words stay
#: (never eaten); the tag is inserted AFTER them. "lol" etc. are
#: matched whole-word only (no "lollipop" misfire).
_LAUGH_CUES_RE = re.compile(r"\b(?:haha+|hehe+|lol|lmao|rofl|ha\s+ha)\b",
                            re.IGNORECASE)

#: Laughter emoji -> [laugh] on native-tag engines, removed on chunked
#: engines (not speakable words; espeak would mangle them).
_LAUGH_EMOJI: tuple[str, ...] = ("😂", "🤣", "😹")

#: Hesitation words -> a comma-length pause after them. "er" is
#: deliberately excluded (too easy to misfire on "ER" / names).
_HESITATION_RE = re.compile(r"\b(?:um|uh|hmm)\b(?![.!?,;:])", re.IGNORECASE)

#: Markdown emphasis markers stripped (the words stay): *word*, **word**,
#: _word_. Lookarounds keep snake_case identifiers and math (2*3) intact.
_MARKDOWN_EMPHASIS_RES = (
    re.compile(r"(?<!\w)\*\*([^*\n]+?)\*\*(?!\w)"),
    re.compile(r"(?<!\w)\*([^*\n]+?)\*(?!\w)"),
    re.compile(r"(?<!\w)_([^_\n]+?)_(?!\w)"),
)


class VoiceTagger:
    """Auto-apply TTS tags based on the active voice engine.

    The owner writes plain text; the tagger inserts engine-appropriate
    tags so pacing/expression happens without manual ``[pause]`` /
    ``[laugh]`` markup:

    * **Native-tag engines** (Chatterbox turbo/nano, Bark): insert
      ``[pause:MS]`` at sentence / comma / paragraph / ellipsis /
      em-dash boundaries (pace profile supplies the lengths),
      ``[pause]`` after hesitation words, and ``[laugh]`` after
      laughter cues (``haha``, ``lol``, ``😂`` ...).
    * **Chunked engines** (espeak-ng, pico2wave, say, edge-tts,
      OpenAI-compat, XTTS, ...): the pacing chunker already inserts
      sentence/comma/paragraph/ellipsis/em-dash pauses, so the tagger
      only adds what the chunker misses — hesitation pauses — plus
      cleanup (markdown markers stripped, laughter emoji removed).
      Sound tags are NEVER emitted here (these engines cannot laugh;
      the cue words stay and are spoken as words).

    Rules (all engines):

    * Explicit user tags are PRESERVED and never duplicated: a trailing
      auto ``[pause]`` / ``[laugh]`` is dropped when the next explicit
      tag is the same kind.
    * Real words are never eaten. Unknown ``[bracket text]`` is not a
      tag and passes through untouched.
    * CAPS emphasis is intentionally NOT auto-tagged (acronym
      ambiguity — NASA is not shouting); ``!`` / ``?`` need no tag
      (engines handle the intonation natively). Write explicit
      ``[shouting]`` when shouting is meant.
    * ``tag()`` returns ``(tagged_text, applied)`` where ``applied``
      counts only newly inserted tags (``["pause×3", "laugh×1"]``) —
      explicit tags are not counted.
    """

    def __init__(self, engine: str = "auto", pace: str = "natural", *,
                 native_tags: bool | None = None) -> None:
        profile = PACE_PROFILES.get((pace or "natural").lower())
        if profile is None:
            raise ToolError(
                f"unknown pace {pace!r} — use one of "
                f"{', '.join(sorted(PACE_PROFILES))}")
        self.engine = (engine or "auto").lower()
        self.pace = (pace or "natural").lower()
        self._profile = profile
        if native_tags is None:
            native_tags = self.engine in VOICE_TAGGER_NATIVE_ENGINES
        self.native_tags = bool(native_tags)

    # ── public ────────────────────────────────────────────────────────
    def tag(self, text: str) -> tuple[str, list[str]]:
        """Insert engine-appropriate tags into plain text.

        Returns ``(tagged_text, applied)``. Pure — no I/O.
        """
        if not (text or "").strip():
            return text or "", []
        in_counts = self._count_tags(text)
        out_parts: list[str] = []
        for kind, content in self._split_explicit(text):
            if kind == "tag":
                out_parts.append(content)
                continue
            out_parts.append(self._tag_segment(content))
        # Strip a trailing auto [pause]/[laugh] that would duplicate the
        # next explicit tag (the explicit tag covers the break/laugh).
        out_parts = self._strip_trailing_dupes(out_parts)
        tagged = "".join(out_parts)
        # Insertions leave double spaces ("[pause:250]  word") — collapse
        # runs of spaces (newlines untouched; TTS text is space-insensitive).
        tagged = re.sub(r"  +", " ", tagged)
        out_counts = self._count_tags(tagged)
        applied = sorted(
            f"{name}×{out_counts[name] - in_counts.get(name, 0)}"
            for name in out_counts
            if out_counts[name] - in_counts.get(name, 0) > 0)
        return tagged, applied

    # ── explicit-tag handling ───────────────────────────────────────────
    @staticmethod
    def _split_explicit(text: str) -> list[tuple[str, str]]:
        """Split into ("text", run) / ("tag", "[name(:ms)]") pieces."""
        parts = _STRICT_TAG_RE.split(text or "")
        out: list[tuple[str, str]] = [("text", parts[0])]
        i = 1
        while i + 2 < len(parts):
            name, ms = parts[i], parts[i + 1] if i + 1 < len(parts) else ""
            full = f"[{name}]" if not ms else f"[{name}:{ms}]"
            out.append(("tag", full))
            out.append(("text", parts[i + 2] if i + 2 < len(parts) else ""))
            i += 3
        return out

    @staticmethod
    def _count_tags(text: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in _STRICT_TAG_RE.finditer(text or ""):
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        return counts

    @staticmethod
    def _strip_trailing_dupes(parts: list[str]) -> list[str]:
        """Drop an auto tag trailing a text piece when the next explicit
        tag is the same kind (no duplication across the boundary)."""
        out = list(parts)
        for i in range(len(out) - 1):
            if out[i + 1].startswith("[pause") or out[i + 1] in (
                    "[laugh]", "[laughs]"):
                kind = "pause" if out[i + 1].startswith("[pause") else "laugh"
                if kind == "pause":
                    # Replace with one space (the explicit tag that follows
                    # still needs its word boundary).
                    out[i] = re.sub(r"\s*\[pause(?::\d+)?\]\s*$", " ", out[i])
                else:
                    out[i] = re.sub(r"\s*\[(?:laugh|laughs)\]\s*$", " ", out[i])
        return out

    # ── per-segment tagging ─────────────────────────────────────────────
    @staticmethod
    def _ptag(pause_ms: int) -> str:
        """Pause tag with surrounding spaces (callers skip zero pauses,
        so the original punctuation stays untouched on tight)."""
        return f" [pause:{pause_ms}] "

    def _tag_segment(self, seg: str) -> str:
        if not seg.strip():
            return seg
        # 1. Markdown emphasis markers stripped (words stay).
        for rx in _MARKDOWN_EMPHASIS_RES:
            seg = rx.sub(r"\1", seg)
        p = self._profile
        if self.native_tags:
            # 2. Paragraph breaks -> paragraph pause.
            if p["paragraph"] > 0:
                seg = re.sub(r"\n\s*\n", self._ptag(p["paragraph"]), seg)
            # 3. Ellipsis -> sentence pause.
            if p["sentence"] > 0:
                seg = re.sub(r"\.{3,}|…", self._ptag(p["sentence"]), seg)
                # 5. Sentence ends -> sentence pause (abbrev/initial aware).
                seg = self._insert_sentence_pauses(seg, p["sentence"])
            if p["comma"] > 0:
                # 4. Em-dash -> comma pause.
                seg = re.sub(r"\s*—\s*", self._ptag(p["comma"]), seg)
                # 6. Comma / semicolon / colon -> comma pause.
                seg = re.sub(r"([,;:])( +)",
                             lambda m: f"{m.group(1)}{self._ptag(p['comma'])}", seg)
                # 7. Hesitation -> comma pause.
                seg = _HESITATION_RE.sub(
                    lambda m: f"{m.group(0)}{self._ptag(p['comma'])}", seg)
            # 8. Laughter words -> [laugh] after the cue.
            seg = _LAUGH_CUES_RE.sub(lambda m: f"{m.group(0)} [laugh]", seg)
            # 9. Laughter emoji -> [laugh].
            for emoji in _LAUGH_EMOJI:
                seg = seg.replace(emoji, " [laugh] ")
        else:
            # Chunked engines: the pacing chunker already handles
            # sentence/comma/paragraph/ellipsis/em-dash pauses. Only
            # hesitation pauses + laughter-emoji cleanup are new here.
            if p["comma"] > 0:
                seg = _HESITATION_RE.sub(
                    lambda m: f"{m.group(0)}{self._ptag(p['comma'])}", seg)
            for emoji in _LAUGH_EMOJI:
                seg = seg.replace(emoji, "")
        # 10. Dedupe consecutive pause tags (sum) / identical sound tags.
        return self._dedupe(seg)

    def _insert_sentence_pauses(self, seg: str, pause_ms: int) -> str:
        """Insert [pause:MS] after sentence-ending punctuation + space.

        Same abbreviation / initial-chain rules as the pacing chunker:
        "Mr. Smith" and "J. R. R. Tolkien" do not split; lowercase
        starts ("hello. world") do. A zero pause inserts nothing.
        """
        if pause_ms <= 0:
            return seg
        out: list[str] = []
        last = 0
        for m in re.finditer(r"[.!?]+\s+", seg):
            before = seg[:m.start()]
            word = re.split(r"[\s\"'(\u201c]+", before)[-1].lower().rstrip(".")
            if word in _PACING_ABBREV:
                continue
            if (len(word) == 1
                    and re.match(r"[A-Za-z]\.\s", seg[m.end():])):
                continue
            out.append(seg[last:m.end()])
            out.append(f"[pause:{pause_ms}] ")
            last = m.end()
        out.append(seg[last:])
        return "".join(out)

    @staticmethod
    def _dedupe(seg: str) -> str:
        """Collapse consecutive pause tags (sum ms) / identical sound tags."""
        for _ in range(4):  # runs to a fixed point; 3+ in a row need >1 pass
            merged = re.sub(
                r"\[pause(?::(\d+))?\]\s*\[pause(?::(\d+))?\]",
                lambda m: f"[pause:{int(m.group(1) or DEFAULT_PAUSE_MS) + int(m.group(2) or DEFAULT_PAUSE_MS)}]",
                seg)
            merged = re.sub(
                r"\[(laugh|laughs|chuckle|cough|sigh)\]\s*\[\1\]",
                r"[\1]", merged)
            if merged == seg:
                return seg
            seg = merged
        return seg


def _escape_ssml(text: str) -> str:
    from xml.sax.saxutils import escape

    return escape(text, {'"': "&quot;", "'": "&apos;"})


def _espeak_ssml(chunks: list[tuple[str, int]]) -> str:
    """Chunks -> one SSML document for espeak-ng (-m).

    Sentence/comma/paragraph/explicit pauses become <break time>.
    Text is XML-escaped; tags are already stripped by chunking.
    """
    parts: list[str] = []
    for ctext, pause_ms in chunks:
        if ctext.strip():
            parts.append(_escape_ssml(ctext))
        if pause_ms > 0:
            parts.append(f'<break time="{pause_ms}ms"/>')
    return ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xml:lang="en-US">' + "".join(parts) + "</speak>")


def _tts_espeak_ssml(chunks: list[tuple[str, int]], voice: str, out: Path,
                     *, speed: float, gap_ms: int = 0) -> None:
    """espeak-ng via SSML: -m interprets <break>; -s sets speed (wpm);
    -g sets the word gap in 10ms units. Speed is -s ONLY (no <prosody
    rate> — passing both would double-apply the rate)."""
    ssml = _espeak_ssml(chunks)
    argv = [_ESPEAK, "-m"]
    if voice:
        argv += ["-v", voice]
    wpm = max(80, min(450, round(_ESPEAK_BASE_WPM * speed)))
    argv += ["-s", str(wpm)]
    if gap_ms > 0:
        argv += ["-g", str(max(1, gap_ms // 10))]
    argv += ["-w", str(out), ssml]
    proc = subprocess.run(argv, capture_output=True, timeout=_TTS_TIMEOUT, check=False)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise ToolError(
            f"espeak_ng failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout or b'').decode('utf-8', 'replace')[:300]}"
        )

#: engine name -> how to invoke it. argv is a template; {out} and {voice} are
#: substituted, the text is always a final argv element (never a shell string).
_ESPEAK = "espeak-ng"
_PICO = "pico2wave"
_SAY = "say"


def detect_tts_engines() -> list[str]:
    """Which TTS engines are actually usable on this machine, in priority order."""
    found: list[str] = []
    if shutil.which(_ESPEAK):
        found.append("espeak_ng")
    if shutil.which(_PICO):
        found.append("pico2wave")
    if shutil.which(_SAY):
        found.append("say")
    try:
        import edge_tts  # noqa: F401

        found.append("edge_tts")
    except ImportError:  # noqa: E103 - optional TTS backend probe
        pass
    return found


def normalize_stt_provider(provider: str) -> str:
    """Normalize a user-facing STT provider name to a ``detect_stt`` value.

    ``api`` is what ``NM_VOICE_STT_PROVIDER=api`` means in .env: use the
    configured OpenAI-compatible STT API endpoint. Anything unrecognized
    falls through unchanged (``detect_stt`` returns None for it).
    """
    return "openai_compat" if (provider or "").strip().lower() == "api" else provider


def detect_stt(provider: str = "auto", *, base_url: str = "", api_key: str = "") -> str | None:
    """Resolve which STT provider will actually work; None = nothing available."""
    provider = normalize_stt_provider(provider)
    if provider and provider != "auto":
        if provider == "openai_compat":
            return "openai_compat" if (base_url or "") and (api_key or "") else None
        if provider == "whisper_cpp":
            return "whisper_cpp" if _whisper_cpp_binary() else None
        return None
    # auto: prefer a configured endpoint (higher quality), fall back to local
    if (base_url or "") and (api_key or ""):
        return "openai_compat"
    if _whisper_cpp_binary():
        return "whisper_cpp"
    return None


def _whisper_cpp_binary() -> str | None:
    env_bin = os.environ.get("NM_WHISPER_CPP_BIN", "").strip()
    if env_bin and Path(env_bin).exists():
        return env_bin
    for name in ("whisper-cpp", "whisper.cpp", "main"):
        path = shutil.which(name)
        if path:
            return path
    return None


def _whisper_cpp_model() -> str | None:
    model = os.environ.get("NM_WHISPER_CPP_MODEL", "").strip()
    if model and Path(model).exists():
        return model
    return None


# ── TTS ──────────────────────────────────────────────────────────────────────


def _tts_local(text: str, engine: str, voice: str, out: Path,
               *, speed: float = 1.0) -> None:
    if engine == "espeak_ng":
        argv = [_ESPEAK]
        if voice:
            argv += ["-v", voice]
        # espeak speed: words per minute (80-450), default 175.
        wpm = max(80, min(450, round(_ESPEAK_BASE_WPM * speed)))
        argv += ["-s", str(wpm), "-w", str(out), text]
    elif engine == "pico2wave":
        argv = [_PICO, "-w", str(out), text]
    elif engine == "say":
        argv = [_SAY, "-o", str(out), "aiff"]
        if voice:
            argv += ["-v", voice]
        # macOS say rate: words per minute.
        argv += ["-r", str(max(90, min(300, round(175 * speed))))]
        argv.append(text)
    else:
        raise ToolError(f"unknown local TTS engine {engine!r}")
    proc = subprocess.run(argv, capture_output=True, timeout=_TTS_TIMEOUT, check=False)
    if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
        raise ToolError(
            f"{engine} failed (exit {proc.returncode}): "
            f"{(proc.stderr or proc.stdout or b'').decode('utf-8', 'replace')[:300]}"
        )


def _tts_edge(text: str, voice: str, out: Path, *,
              chunks: list[tuple[str, int]] | None = None,
              speed: float = 1.0) -> None:
    """edge-tts: single call. With chunks, SSML carries <break> pauses and
    a <prosody rate> — edge-tts treats input starting with <speak> as SSML."""
    import asyncio

    import edge_tts

    if chunks is not None:
        pct = round((speed - 1.0) * 100)
        rate = f"{pct:+d}%"
        parts: list[str] = []
        for ctext, pause_ms in chunks:
            if ctext.strip():
                parts.append(_escape_ssml(ctext))
            if pause_ms > 0:
                parts.append(f'<break time="{pause_ms}ms"/>')
        ssml = (f'<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
                f'xml:lang="en-US"><prosody rate="{rate}">'
                + "".join(parts) + "</prosody></speak>")
        payload = ssml
    else:
        payload = text
    voice_id = voice or None
    if voice_id:
        coro = edge_tts.Communicate(payload, voice_id).save(str(out))
    else:
        coro = edge_tts.Communicate(payload).save(str(out))
    asyncio.run(asyncio.wait_for(coro, timeout=_TTS_TIMEOUT))
    if not out.exists() or out.stat().st_size == 0:
        raise ToolError("edge_tts produced no audio")


def _tts_openai_compat(text: str, voice: str, out: Path, *, base_url: str,
                       api_key: str, model: str, speed: float = 1.0,
                       response_format: str = "mp3") -> None:
    from ..core.http import HttpClient

    base = base_url.rstrip("/")
    if not base.endswith("/v1") and "/v1" not in base:
        base = base + "/v1"
    client = HttpClient(timeout=_TTS_TIMEOUT)
    payload: dict[str, Any] = {
        "model": model or "tts-1",
        "voice": voice or "alloy",
        "input": text[:4000],
        "speed": max(0.25, min(4.0, speed)),
    }
    if response_format != "mp3":
        payload["response_format"] = response_format
    response = client.post_json(
        f"{base}/audio/speech",
        payload,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if not response.ok:
        raise ToolError(f"TTS API {response.status}: {response.text[:300]}")
    out.write_bytes(response.body)
    if out.stat().st_size == 0:
        raise ToolError("TTS API returned an empty file")


# Microsoft / edge-tts voice ids ("en-NG-EzinneNeural",
# "en-US-AriaNeural") — language-REGION-NameNeural. The local engines
# (espeak-ng, pico2wave, macOS say) have no such voices.
_EDGE_VOICE_RE = re.compile(r"^[a-z]{2,3}-[A-Z]{2}-.+Neural$", re.IGNORECASE)
# OpenAI voice names — likewise meaningless to the local engines.
_OPENAI_VOICES = frozenset({
    "alloy", "ash", "ballad", "cedar", "coral", "echo", "fable", "marin",
    "nova", "onyx", "sage", "shimmer", "verse",
})
_LOCAL_TTS_ENGINES = frozenset({"espeak_ng", "pico2wave", "say"})


def _voice_for_engine(engine: str, voice: str) -> str:
    """Voice string actually valid for ``engine``.

    The local engines don't understand Microsoft/edge-tts voice ids
    or OpenAI voice names — passing one verbatim (espeak-ng's ``-v``,
    say's ``-v``) makes the engine reject the call and the whole TTS
    fails. The voice-note reply path used to do exactly that: its
    default voice *is* an edge-tts id, so on an espeak-ng machine
    every voice reply died in TTS and downgraded to text. Drop
    foreign ids for local engines so the engine default speaks;
    edge-tts / OpenAI-compatible engines keep their ids untouched,
    and a voice that already looks local (``en``, ``en-gb``,
    ``Samantha``) passes through unchanged.
    """
    if not voice or engine not in _LOCAL_TTS_ENGINES:
        return voice
    if _EDGE_VOICE_RE.match(voice) or voice.lower() in _OPENAI_VOICES:
        _log.debug("dropping voice %r for local TTS engine %s: not a %s "
                   "voice — using the engine default", voice, engine, engine)
        return ""
    return voice


def tts(
    text: str,
    *,
    engine: str = "auto",
    voice: str = "",
    out_dir: str | Path,
    base_url: str = "",
    api_key: str = "",
    model: str = "",
    timeout: float = _TTS_TIMEOUT,
    speed: float = 1.0,
    pace: str = "natural",
    gap_ms: int = 0,
    auto_tags: bool = True,
) -> dict[str, Any]:
    """Synthesize speech to a file.

    Returns ``{path, engine, format, bytes, seconds, speed, pace, gap_ms,
    pauses_ms, dropped_sounds, pauses_dropped, gap_ignored, auto_tags,
    auto_tags_applied}``.

    Tags: ``[pause]`` / ``[pause:MS]`` insert real silence; sound tags
    (``[laugh]`` ...) are dropped — these engines cannot laugh, use
    ``/speak`` for real laughs; emotion tags are stripped; unknown
    ``[bracket text]`` is preserved. Tag text is never spoken literally.
    ``pace`` (tight | natural | slow) adds sentence/comma/paragraph
    breathing room; ``speed`` scales the speaking rate (0.25–4.0);
    ``gap_ms`` is the espeak-ng word gap (``-g``); other engines ignore it.

    ``auto_tags`` (default ON): :class:`VoiceTagger` inserts
    engine-appropriate tags from plain-text cues before synthesis
    (hesitation pauses on every engine; sentence/comma/paragraph
    pauses and ``[laugh]`` only where the engine renders them
    natively). Explicit user tags are always preserved and never
    duplicated. ``auto_tags_applied`` lists what was inserted.
    """
    text = (text or "").strip()
    if not text:
        raise ToolError("tts needs text")
    speed = max(0.25, min(4.0, float(speed)))
    gap_ms = max(0, int(gap_ms))
    engines = detect_tts_engines()
    if engine in ("", "auto"):
        engine = engines[0] if engines else ("openai_compat" if base_url and api_key else "")
        if not engine:
            raise ToolError(
                "no TTS engine available — Termux: pkg install espeak-ng "
                "(or set an OpenAI-compatible endpoint for API TTS)"
            )
    elif engine not in engines and engine != "openai_compat":
        available = ", ".join(engines) or "none"
        raise ToolError(f"TTS engine {engine!r} not available (found: {available})")
    auto_tags_applied: list[str] = []
    if auto_tags:
        text, auto_tags_applied = VoiceTagger(engine, pace).tag(text)
    chunks, dropped_sounds = chunk_tts_text(text, pace)
    speakable = " ".join(t for t, _ in chunks if t.strip())
    if not speakable:
        raise ToolError(
            "tts needs speakable text — input was only tags "
            f"(dropped: {', '.join(dropped_sounds) or 'none'})")
    total_pause_ms = sum(p for _, p in chunks)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-engine voice handling (see _voice_for_engine): an edge-tts /
    # OpenAI voice id must never reach a local engine's -v. /tts passes
    # no voice, so it is unaffected by this.
    voice = _voice_for_engine(engine, voice)

    ext = "mp3" if engine in {"edge_tts", "openai_compat"} else (
        "aiff" if engine == "say" else "wav")
    out = out_dir / f"tts-{int(time.time() * 1000)}.{ext}"
    started = time.perf_counter()
    pauses_dropped = False
    try:
        if engine == "edge_tts":
            _tts_edge(speakable, voice, out, chunks=chunks, speed=speed)
        elif engine == "openai_compat":
            if total_pause_ms > 0:
                try:
                    _tts_openai_chunked(chunks, voice, out, base_url=base_url,
                                        api_key=api_key, model=model, speed=speed)
                    ext = "wav"
                except ToolError:
                    # Endpoint rejected wav format (or a chunk failed):
                    # fall back to one mp3 call, tags stripped, no pauses.
                    pauses_dropped = True
                    _tts_openai_compat(speakable, voice, out, base_url=base_url,
                                       api_key=api_key, model=model, speed=speed)
            else:
                _tts_openai_compat(speakable, voice, out, base_url=base_url,
                                   api_key=api_key, model=model, speed=speed)
        elif engine == "espeak_ng":
            # SSML path: <break> carries every pause (sentence/comma/
            # paragraph/explicit) in ONE call; -s speed, -g word gap.
            _tts_espeak_ssml(chunks, voice, out, speed=speed, gap_ms=gap_ms)
        elif engine == "pico2wave":
            if total_pause_ms > 0:
                _tts_local_chunked(chunks, engine, voice, out, speed=speed)
            else:
                _tts_local(speakable, engine, voice, out, speed=speed)
        elif engine == "say":
            _tts_say_with_pauses(chunks, voice, out, speed=speed)
        else:
            _tts_local(speakable, engine, voice, out, speed=speed)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"TTS timed out after {timeout}s") from exc
    return {
        "path": str(out),
        "engine": engine,
        "format": ext,
        "bytes": out.stat().st_size,
        "seconds": round(time.perf_counter() - started, 2),
        "speed": speed,
        "pace": pace,
        "gap_ms": gap_ms,
        "pauses_ms": 0 if pauses_dropped else total_pause_ms,
        "dropped_sounds": dropped_sounds,
        "pauses_dropped": pauses_dropped,
        "gap_ignored": bool(gap_ms and engine != "espeak_ng"),
        "auto_tags": auto_tags,
        "auto_tags_applied": auto_tags_applied,
    }


def _tts_openai_chunked(chunks: list[tuple[str, int]], voice: str, out: Path,
                        *, base_url: str, api_key: str, model: str,
                        speed: float) -> None:
    """API TTS with real silence: per-chunk WAV calls, stdlib concat."""
    import tempfile

    text_chunks, leading = _fold_empty_chunks(chunks)
    if not text_chunks:
        raise ToolError("TTS API returned no audio for any chunk")
    wav_blobs: list[bytes] = []
    pauses: list[int] = []
    with tempfile.TemporaryDirectory(prefix="tts-chunks-") as tmp:
        for i, (ctext, pause_ms) in enumerate(text_chunks):
            part = Path(tmp) / f"part-{i}.wav"
            _tts_openai_compat(ctext, voice, part, base_url=base_url,
                               api_key=api_key, model=model, speed=speed,
                               response_format="wav")
            wav_blobs.append(part.read_bytes())
            pauses.append(pause_ms)
    sample_rate = _wav_sample_rate(wav_blobs[0])
    out.write_bytes(_concat_wav_chunks(wav_blobs, pauses, sample_rate,
                                       leading_silence_ms=leading))


def _tts_local_chunked(chunks: list[tuple[str, int]], engine: str, voice: str,
                       out: Path, *, speed: float) -> None:
    """Local wav engines with real silence: per-chunk WAV + stdlib concat."""
    import tempfile

    text_chunks, leading = _fold_empty_chunks(chunks)
    if not text_chunks:
        raise ToolError(f"{engine} produced no audio for any chunk")
    wav_blobs: list[bytes] = []
    pauses: list[int] = []
    with tempfile.TemporaryDirectory(prefix="tts-chunks-") as tmp:
        for i, (ctext, pause_ms) in enumerate(text_chunks):
            part = Path(tmp) / f"part-{i}.wav"
            _tts_local(ctext, engine, voice, part, speed=speed)
            wav_blobs.append(part.read_bytes())
            pauses.append(pause_ms)
    sample_rate = _wav_sample_rate(wav_blobs[0])
    out.write_bytes(_concat_wav_chunks(wav_blobs, pauses, sample_rate,
                                       leading_silence_ms=leading))


def _tts_say_with_pauses(chunks: list[tuple[str, int]], voice: str, out: Path,
                         *, speed: float) -> None:
    """macOS `say`: single call, `[[slnc N]]` carries the pauses natively."""
    parts: list[str] = []
    for ctext, pause_ms in chunks:
        if ctext.strip():
            parts.append(ctext)
        if pause_ms > 0:
            parts.append(f"[[slnc {pause_ms}]]")
    _tts_local(" ".join(parts), "say", voice, out, speed=speed)


# ── STT ──────────────────────────────────────────────────────────────────────


def wav_to_voice_ogg(path: str | Path) -> tuple[str, str]:
    """Convert a TTS audio file to OGG/Opus for voice-note bubbles.

    Voice notes on WhatsApp and Telegram are OGG/Opus (``voice`` kind /
    ``voice_note=True``), NOT WAV files. Sending the raw ``tts-*.wav``
    as a file shows up as a document, not a playable voice-note bubble.

    Returns ``(path, mime)``. When ffmpeg is missing or conversion
    fails, returns the ORIGINAL file + its mime — the caller must say
    so honestly (sent as a file, not a voice note), never pretend a
    file is a voice note.
    """
    import os
    import shutil
    import subprocess

    src = str(path)
    if src.lower().endswith((".ogg", ".opus")):
        return src, "audio/ogg"
    mime_by_ext = {".wav": "audio/wav", ".mp3": "audio/mpeg",
                   ".aiff": "audio/aiff", ".m4a": "audio/mp4"}
    fallback_mime = mime_by_ext.get(os.path.splitext(src)[1].lower(),
                                    "audio/wav")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src, fallback_mime
    ogg_path = os.path.splitext(src)[0] + ".ogg"
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", src, "-c:a", "libopus", "-b:a", "64k",
             ogg_path],
            check=True, capture_output=True, timeout=120)
        if os.path.isfile(ogg_path):
            return ogg_path, "audio/ogg"
    except Exception:  # noqa: BLE001 - caller falls back honestly
        pass
    return src, fallback_mime


def _to_wav(path: Path) -> Path:
    """whisper.cpp wants wav; convert with ffmpeg when the input isn't one."""
    if path.suffix.lower() in {".wav", ".wave"}:
        return path
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ToolError(
            f"{path.suffix} audio needs ffmpeg to convert to wav "
            "(Termux: pkg install ffmpeg) or send a .wav"
        )
    converted = path.with_suffix(".wav")
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-ar", "16000", "-ac", "1", str(converted)],
        capture_output=True, timeout=_STT_TIMEOUT, check=False,
    )
    if proc.returncode != 0 or not converted.exists():
        raise ToolError(
            "ffmpeg conversion failed: "
            f"{proc.stderr.decode('utf-8', 'replace')[:300]}"
        )
    return converted


def _stt_whisper_cpp(path: Path, language: str) -> str:
    binary = _whisper_cpp_binary()
    model = _whisper_cpp_model()
    if not binary:
        raise ToolError("whisper.cpp binary not found (set NM_WHISPER_CPP_BIN)")
    if not model:
        raise ToolError("no whisper.cpp model (set NM_WHISPER_CPP_MODEL to a .bin path)")
    wav = _to_wav(path)
    argv = [binary, "-m", model, "-f", str(wav)]
    if language and language != "auto":
        argv += ["-l", language]
    proc = subprocess.run(argv, capture_output=True, timeout=_STT_TIMEOUT, check=False)
    if proc.returncode != 0:
        raise ToolError(
            f"whisper.cpp failed (exit {proc.returncode}): "
            f"{(proc.stderr or b'').decode('utf-8', 'replace')[:300]}"
        )
    return re.sub(r"\[[^]]*\]", "", proc.stdout.decode("utf-8", "replace")).strip()


# Groq / OpenAI-compatible transcription endpoints decide the file type
# from the MULTIPART FILENAME EXTENSION, not the MIME. A filename with
# no supported extension — e.g. the MTProto download
# ``tg-<millis>-0.bin`` (extensionless temp name + ".bin" appended by
# the adapter) — is rejected with HTTP 400 unsupported_audio_format
# even though the bytes are valid .ogg Opus. The MIME fix alone was
# necessary but NOT sufficient; the filename extension is what Groq
# checks. This is the set Groq accepts (from its 400 error message).
_GROQ_SUPPORTED_EXTS = frozenset({
    ".flac", ".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".ogg", ".opus",
    ".wav", ".webm",
})


def _sniff_audio_ext(path: Path) -> str:
    """Detect the audio container from magic bytes. "" when unknown.

    Used to name the upload correctly when the on-disk filename lies
    (``.bin`` / extensionless temp download). Never trusts the suffix.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(16)
    except OSError:
        return ""
    if head.startswith(b"OggS"):
        return ".ogg"
    if head.startswith(b"RIFF") and head[8:12] == b"WAVE":
        return ".wav"
    if head.startswith(b"fLaC"):
        return ".flac"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        return ".webm"
    if head.startswith(b"ID3") or (len(head) >= 2 and head[0] == 0xFF
                                   and (head[1] & 0xE0) == 0xE0):
        return ".mp3"
    if len(head) >= 8 and head[4:8] == b"ftyp":
        return ".m4a"
    return ""


def _to_ogg(path: Path) -> Path:
    """Convert a genuinely unsupported audio format to OGG/Opus.

    Only for formats Groq cannot take at all (AAC, AMR, AIFF, ...).
    Relabelling those bytes with a supported extension would be lying
    about the content — convert, don't hope. Raises ToolError when
    ffmpeg is unavailable or the conversion fails.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ToolError(
            f"{path.suffix or 'extensionless'} audio is not a format Groq "
            "accepts and ffmpeg is unavailable to convert it "
            "(Termux: pkg install ffmpeg)"
        )
    converted = path.with_suffix(".ogg")
    proc = subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-c:a", "libopus", "-b:a", "64k",
         str(converted)],
        capture_output=True, timeout=_STT_TIMEOUT, check=False,
    )
    if proc.returncode != 0 or not converted.exists():
        raise ToolError(
            "ffmpeg conversion to ogg failed: "
            f"{proc.stderr.decode('utf-8', 'replace')[:300]}"
        )
    return converted


def _stt_upload_target(path: Path) -> tuple[Path, str]:
    """Resolve (file to upload, multipart filename) for STT.

    The multipart filename ALWAYS carries a Groq-supported extension
    matching the actual audio format:
    - suffix already supported → on-disk name unchanged;
    - misnamed file (``.bin`` / extensionless) → magic-byte sniff
      supplies the real extension (``.ogg`` for Telegram voice notes);
    - genuinely unsupported format (AAC, AMR, AIFF, unknown) →
      converted to OGG/Opus first, never sent with a lying extension.
    """
    suffix = path.suffix.lower()
    if suffix in _GROQ_SUPPORTED_EXTS:
        return path, path.name
    sniffed = _sniff_audio_ext(path)
    if sniffed in _GROQ_SUPPORTED_EXTS:
        return path, path.stem + sniffed
    converted = _to_ogg(path)
    return converted, converted.name


def _stt_openai_compat(
    path: Path, language: str, *, base_url: str, api_key: str, model: str
) -> str:
    from ..core.http import HttpClient

    base = base_url.rstrip("/")
    if not base.endswith("/v1") and "/v1" not in base:
        base = base + "/v1"
    client = HttpClient(timeout=_STT_TIMEOUT)
    # Groq doesn't serve OpenAI's "whisper-1" — use their turbo model.
    default_model = ("whisper-large-v3-turbo" if "groq.com" in base
                     else "whisper-1")
    # The upload filename must carry a supported extension matching the
    # actual format: Groq decides the type from the filename, not the
    # MIME. A misnamed file (``.bin`` temp download) gets its real
    # extension from magic bytes; an unsupported format is converted
    # to OGG first. The MIME follows the upload filename, not the
    # on-disk suffix.
    upload_path, upload_name = _stt_upload_target(path)
    mime = _AUDIO_MIME_BY_EXT.get(Path(upload_name).suffix.lower(),
                                 "application/octet-stream")
    response = client.post_multipart(
        f"{base}/audio/transcriptions",
        fields={"model": model or default_model, **({"language": language} if language and language != "auto" else {})},
        files=[("file", str(upload_path), mime, upload_name)],
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if not response.ok:
        raise ToolError(f"STT API {response.status}: {response.text[:300]}")
    data = response.json()
    return str(data.get("text") or "").strip()


def stt(
    path: str | Path,
    *,
    provider: str = "auto",
    language: str = "en",
    base_url: str = "",
    api_key: str = "",
    model: str = "",
) -> dict[str, Any]:
    """Transcribe an audio file. Returns {text, provider, seconds}."""
    source = Path(path).expanduser()
    if not source.exists():
        raise ToolError(f"no such audio file: {source}")
    resolved = detect_stt(provider, base_url=base_url, api_key=api_key)
    if not resolved:
        raise ToolError(
            "no STT backend available — configure an OpenAI-compatible endpoint "
            "(NM_AUDIO_STT_BASE_URL + NM_AUDIO_STT_API_KEY) or a whisper.cpp "
            "binary (NM_WHISPER_CPP_BIN + NM_WHISPER_CPP_MODEL)"
        )
    started = time.perf_counter()
    if resolved == "whisper_cpp":
        text = _stt_whisper_cpp(source, language)
    else:
        text = _stt_openai_compat(source, language, base_url=base_url,
                                  api_key=api_key, model=model)
    return {
        "text": text,
        "provider": resolved,
        "seconds": round(time.perf_counter() - started, 2),
        "file": str(source),
        "bytes": source.stat().st_size,
    }


# ── info ─────────────────────────────────────────────────────────────────────


def audio_info(path: str | Path) -> dict[str, Any]:
    """Format, size, and duration (wav exact; others estimated)."""
    source = Path(path).expanduser()
    if not source.exists():
        raise ToolError(f"no such file: {source}")
    size = source.stat().st_size
    suffix = source.suffix.lower()
    info: dict[str, Any] = {"path": str(source), "bytes": size, "format": suffix.lstrip(".") or "unknown"}
    if suffix in {".wav", ".wave"}:
        try:
            with wave.open(str(source), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate() or 1
                info["duration"] = round(frames / rate, 2)
                info["channels"] = handle.getnchannels()
                info["sample_rate"] = rate
        except (wave.Error, EOFError):
            info["duration"] = None
    elif suffix == ".mp3":
        info["duration"] = round(size / 132300, 2)  # ~128 kbps estimate
        info["duration_note"] = "estimated (128kbps)"
    elif suffix in {".ogg", ".opus", ".m4a", ".aac", ".flac", ".aiff", ".aif"}:
        info["duration"] = None
    return info


# ── tool registration ────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    """Attach the speak / transcribe / audio_info tools to a registry."""
    context = registry.context
    settings = getattr(context, "settings", None) if context is not None else None
    audio_settings = getattr(settings, "audio", None) if settings else None
    llm_settings = getattr(settings, "llm", None) if settings else None

    def audio_dir() -> Path:
        rel = getattr(audio_settings, "audio_dir", "audio") or "audio"
        if settings is not None and hasattr(settings, "resolve"):
            return Path(settings.resolve(rel))
        return Path(rel)

    def _api_pair() -> tuple[str, str]:
        # Explicit audio settings win, then the LLM's per-provider keys
        # (groq_api_key etc. — populated from NM_GROQ_API_KEY and friends),
        # then the raw provider env vars the LLM broker itself uses.
        # Base URL and key stay paired — never mix providers.
        base = getattr(audio_settings, "stt_base_url", "") or ""
        key = getattr(audio_settings, "stt_api_key", "") or ""
        if base and key:
            return base, key
        if llm_settings is not None:
            for prefix in ("groq", "openai", "openrouter"):
                pkey = getattr(llm_settings, f"{prefix}_api_key", "") or ""
                pbase = getattr(llm_settings, f"{prefix}_base_url", "") or ""
                if pkey and pbase:
                    return pbase, pkey
            base = getattr(llm_settings, "base_url", "") or ""
            key = getattr(llm_settings, "api_key", "") or ""
            if base and key:
                return base, key
        if os.environ.get("OPENAI_API_KEY", ""):
            return (os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    os.environ["OPENAI_API_KEY"])
        if os.environ.get("GROQ_API_KEY", ""):
            return (os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                    os.environ["GROQ_API_KEY"])
        if os.environ.get("NM_GROQ_API_KEY", ""):
            return ("https://api.groq.com/openai/v1",
                    os.environ["NM_GROQ_API_KEY"])
        return "", ""

    def api_base() -> str:
        return _api_pair()[0]

    def api_key() -> str:
        return _api_pair()[1]

    def _check_net() -> None:
        checker = getattr(context, "check", None)
        if checker is not None:
            decision = checker(Capability.NET_OUT)
            if not getattr(decision, "allowed", True):
                raise ToolError(f"network not granted: {getattr(decision, 'reason', '')}")

    @registry.register(
        "speak",
        description=(
            "text-to-speech: synthesize audio to a file "
            "(espeak-ng / pico2wave / say / edge-tts / OpenAI-compatible, auto-detected). "
            "Tags: [pause] / [pause:MS] insert real silence; sound tags "
            "([laugh] [cough] ...) are dropped (these engines cannot laugh — "
            "use tts_say for real laughs); emotion tags are stripped; "
            "unknown [bracket text] is preserved. pace = tight | natural | slow; "
            "speed = speaking rate (0.25-4.0). auto_tags (default on) inserts "
            "engine-appropriate tags from plain-text cues (hesitation pauses "
            "everywhere; [laugh]/sentence pauses only where the engine "
            "renders them); explicit tags are preserved, never duplicated."
        ),
        capability=Capability.EXEC_SHELL,
        parameters={
            "text": "str — what to say",
            "engine": "str (optional) — auto | espeak_ng | pico2wave | say | edge_tts | openai_compat",
            "voice": "str (optional) — engine voice name",
            "speed": "float (optional, default 1.0) — speaking rate 0.25-4.0",
            "pace": "str (optional, default natural) — tight | natural | slow sentence/comma/paragraph pauses",
            "gap_ms": "int (optional, default 0) — espeak-ng word gap in ms (-g); ignored by other engines",
            "auto_tags": "bool (optional, default true) — auto-apply engine-appropriate tags from plain-text cues",
        },
    )
    def speak(text: str, *, engine: str = "auto", voice: str = "",
              speed: float = 1.0, pace: str = "natural",
              gap_ms: int = 0, auto_tags: bool = True) -> dict[str, Any]:
        base, key = api_base(), api_key()
        if engine in ("auto", "") and base and key and not detect_tts_engines():
            engine = "openai_compat"
        if engine == "openai_compat":
            _check_net()
        return tts(
            text, engine=engine, voice=voice, out_dir=audio_dir(),
            base_url=base, api_key=key,
            model=getattr(audio_settings, "stt_model", "") or "tts-1",
            speed=speed, pace=pace, gap_ms=gap_ms, auto_tags=auto_tags,
        )

    @registry.register(
        "transcribe",
        description=(
            "speech-to-text: transcribe an audio file "
            "(OpenAI-compatible /audio/transcriptions or local whisper.cpp)"
        ),
        capability=Capability.EXEC_SHELL,
        parameters={
            "path": "str — audio file (workspace-relative or absolute)",
            "language": "str (optional, default en; 'auto' to detect)",
            "provider": "str (optional) — auto | openai_compat | whisper_cpp",
        },
    )
    def transcribe(path: str, *, language: str = "en", provider: str = "auto") -> dict[str, Any]:
        from .filesystem import safe_media_path

        # voice notes land in the chat media dir (outside the workspace
        # sandbox by design) — safe_media_path allows Devon's own
        # downloads without opening the sandbox.
        target = safe_media_path(context, path, must_exist=True)
        base, key = api_base(), api_key()
        # The caller's "auto" means "use the configured default": a
        # NM_VOICE_STT_PROVIDER setting (now actually mapped) wins over
        # the hardcoded default. "api" normalizes to openai_compat.
        effective = provider
        if (effective or "auto") == "auto":
            configured = getattr(audio_settings, "stt_provider", "") or ""
            if configured.strip() and configured.strip().lower() != "auto":
                effective = configured
        effective = normalize_stt_provider(effective)
        if effective == "openai_compat" or (effective == "auto" and base and key):
            _check_net()
        # Groq doesn't serve OpenAI's whisper-1 — pick the model for
        # the endpoint (empty = let stt() choose the provider default).
        stt_model = getattr(audio_settings, "stt_model", "") or ""
        if not stt_model and "groq.com" in base:
            stt_model = "whisper-large-v3-turbo"
        return stt(
            target, provider=effective, language=language,
            base_url=base, api_key=key, model=stt_model,
        )

    @registry.register(
        "audio_info",
        description="audio file metadata: format, size, duration (wav exact, mp3 estimated)",
        capability=Capability.FS_READ,
        parameters={"path": "str — audio file path"},
    )
    def audio_info_tool(path: str) -> dict[str, Any]:
        from .filesystem import safe_path

        return audio_info(safe_path(context, path, must_exist=True))

    # ── neural TTS (the owner's universal_tts.py, integrated) ────────────────

    def voices_dir() -> str:
        return str(audio_dir() / "voices")

    @registry.register(
        "tts_say",
        description=(
            "Neural text-to-speech (Bark / XTTS v2 / Kokoro, lazy-loaded): "
            "synthesizes text with inline emotion/pause tags "
            "([happy] [whisper] [laughs] [pause:300]) into a WAV voice note. "
            "voice = a registered profile name; mood bridges the partner "
            "mood system into tags."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "text": "str — what to say (tags allowed)",
            "voice": "str (optional) — registered voice profile name",
            "backend": "str (optional) — auto | bark | xtts | kokoro",
            "mood": "str (optional) — happy | sad | angry | tired | …",
            "mood_level": "int 0-5 (optional, 5)",
            "out_name": "str (optional) — output file name",
        },
    )
    def tts_say(text: str, *, voice: str = "", backend: str = "",
                mood: str = "", mood_level: str = "", out_name: str = "") -> dict[str, Any]:
        from ..voice.tts import UniversalTTS, available_backends

        text = (text or "").strip()
        if not text:
            raise ToolError("tts_say needs text")
        backend_name = (backend or "").strip() or \
            getattr(audio_settings, "neural_tts", "") or "auto"
        try:
            level = int(mood_level) if str(mood_level).strip() else 5
        except ValueError:
            level = 5
        engine = UniversalTTS(backend=backend_name, voices_dir=voices_dir())
        out_path = ""
        if out_name:
            # out_name is caller-controlled: sanitise before it becomes
            # a filename ("a:b" / "x/y" raises OSError on Windows).
            out_path = str(audio_dir() / f"{safe_filename(out_name)}.wav")
        try:
            return engine.speak(
                text, voice_name=voice or None, out_path=out_path,
                mood=mood, mood_level=level)
        except RuntimeError as exc:
            have = available_backends()
            hint = f" available: {', '.join(have)}" if have else ""
            raise ToolError(f"tts backend unavailable: {exc}{hint}") from exc

    @registry.register(
        "tts_voices",
        description=(
            "Manage TTS voice profiles: list, add a reference clip for "
            "cloning, register a built-in preset, or remove. "
            "Clone operations are audit-logged."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "action": "str — list | add | preset | remove",
            "name": "str — profile name (add/preset/remove)",
            "path": "str — reference audio file (add)",
            "preset_id": "str — built-in voice id (preset)",
            "language": "str (optional, en)",
        },
    )
    def tts_voices(action: str, *, name: str = "", path: str = "",
                   preset_id: str = "",
                   language: str = "") -> dict[str, Any]:
        from ..voice.tts import VoiceLibrary

        action = (action or "list").strip().lower()
        lib = VoiceLibrary(voices_dir())
        if action == "list":
            return {"voices": lib.list()}
        if action == "add":
            if not name:
                raise ToolError("tts_voices add needs a name")
            sample = path
            if sample and not os.path.isabs(sample):
                try:
                    from .filesystem import safe_path

                    sample = str(safe_path(context, sample, must_exist=True))
                except ToolError:
                    raise ToolError(f"voice sample not found: {path!r}") from None
            if not sample:
                # profile without reference audio (preset-style)
                from ..voice.tts import VoiceProfile

                profile = VoiceProfile(
                    name=name, language=language or "en")
                lib.profiles[name] = profile
                lib._save_index()
                return {"ok": True, "profile": profile.to_dict()}
            profile = lib.upload_voice(
                name, sample,
                language=language or "en")
            profile.audit_clone("tts_voices")
            return {"ok": True, "profile": profile.to_dict()}
        if action == "preset":
            if not name or not preset_id:
                raise ToolError("tts_voices preset needs name + preset_id")
            profile = lib.register_preset(
                name, preset_id, language=language or "en")
            return {"ok": True, "profile": profile.to_dict()}
        if action == "remove":
            return {"ok": lib.remove(name or "")}
        raise ToolError(f"unknown tts_voices action {action!r}")

    # ── voice design + morphing (God-tier voice lab) ─────────────────────

    # ── god-tier voice: sing / long-form / rich ──────────────────────────

    @registry.register(
        "voice_sing",
        description=(
            "Sing a melody with DiffSinger (open-source neural singing): "
            "melody = 'C4:0.5:hello D4:0.5:world' (note:seconds:lyric). "
            "Needs a DiffSinger voicebank in ~/.nomorals/svs/ and "
            "onnxruntime. rvc_model swaps in the user's timbre."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "melody": "str — 'C4:0.5:hello D4:0.5:world' or MIDI numbers",
            "rvc_model": "str (optional) — RVC model for the singer's timbre",
            "out_name": "str (optional) — output file name",
        },
    )
    def voice_sing(melody: str, *, rvc_model: str = "",
                   out_name: str = "") -> dict[str, Any]:
        from ..voice.tts import UniversalTTS
        if not (melody or "").strip():
            raise ToolError("voice_sing needs a melody")
        engine = UniversalTTS(backend="auto", voices_dir=voices_dir())
        out_path = str(audio_dir() / f"{safe_filename(out_name or 'song')}.wav") \
            if out_name else ""
        try:
            return engine.sing(melody, rvc_model=rvc_model,
                               out_path=out_path)
        except RuntimeError as exc:
            raise ToolError(f"singing unavailable: {exc}") from exc

    @registry.register(
        "voice_long",
        description=(
            "Long-form TTS without voice drift: chapters, audiobooks, "
            "briefings. Sentence-boundary chunking, rolling prosody "
            "context, crossfade stitching, degenerate-chunk regeneration."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "text": "str — the full text (any length)",
            "voice": "str (optional) — registered voice profile name",
            "mood": "str (optional, neutral)",
            "out_name": "str (optional) — output file name",
        },
    )
    def voice_long(text: str, *, voice: str = "", mood: str = "",
                   out_name: str = "") -> dict[str, Any]:
        from ..voice.tts import UniversalTTS
        if not (text or "").strip():
            raise ToolError("voice_long needs text")
        engine = UniversalTTS(backend="auto", voices_dir=voices_dir())
        out_path = str(audio_dir() / f"{safe_filename(out_name or 'longform')}.wav") \
            if out_name else ""
        try:
            return engine.speak_long(text, voice_name=voice or None,
                                     out_path=out_path,
                                     mood=mood or "neutral")
        except RuntimeError as exc:
            raise ToolError(f"long-form TTS failed: {exc}") from exc

    @registry.register(
        "voice_rich",
        description=(
            "God-tier one-call voice: neural emotion ('[said angrily]'), "
            "accent ('british'), procedural ambience ('light rain'), room "
            "reverb ('hall'), and mastering. Every stage reports its "
            "honest tier."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "text": "str — what to say",
            "voice": "str (optional) — registered voice profile name",
            "direction": "str (optional) — '[said angrily in British accent]'",
            "accent": "str (optional) — british | nigerian | french | …",
            "ambient": "str (optional) — light rain | crowd | wind | …",
            "room": "str (optional) — small | room | hall | cave | phone",
            "out_name": "str (optional) — output file name",
        },
    )
    def voice_rich(text: str, *, voice: str = "", direction: str = "",
                   accent: str = "", ambient: str = "", room: str = "",
                   out_name: str = "") -> dict[str, Any]:
        from ..voice.tts import UniversalTTS
        if not (text or "").strip():
            raise ToolError("voice_rich needs text")
        engine = UniversalTTS(backend="auto", voices_dir=voices_dir())
        out_path = str(audio_dir() / f"{safe_filename(out_name or 'rich')}.wav") \
            if out_name else ""
        try:
            return engine.speak_rich(
                text, voice_name=voice or None, out_path=out_path,
                direction=direction, accent=accent,
                ambient=ambient, room=room)
        except RuntimeError as exc:
            raise ToolError(f"voice_rich failed: {exc}") from exc

    @registry.register(
        "design_voice",
        description=(
            "Design a voice from a natural description "
            "(\"deep warm narrator\", \"bright young female\", "
            "\"raspy old sailor\"): shapes a base voice with honest DSP "
            "and registers it in the catalogue. No neural voice model needed."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "name": "str — name for the designed voice",
            "description": "str — natural-language voice description",
            "base_voice": "str (optional) — catalogue voice to shape from",
        },
    )
    def design_voice_tool(name: str, description: str,
                          base_voice: str = "") -> dict[str, Any]:
        from ..voice.design import design_voice
        if not (name or "").strip() or not (description or "").strip():
            raise ToolError("design_voice needs a name and a description")
        return design_voice(name.strip(), description,
                            base_voice=(base_voice or "").strip(),
                            voices_dir=voices_dir())

    @registry.register(
        "morph_voices",
        description=(
            "Morph two catalogue voices into a new blended voice "
            "(pitch-interpolated DSP morph, honest about its limits). "
            "ratio 0.0 = all voice_a, 1.0 = all voice_b."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "name": "str — name for the morphed voice",
            "voice_a": "str — first catalogue voice",
            "voice_b": "str — second catalogue voice",
            "ratio": "float (optional, 0.5) — blend ratio",
        },
    )
    def morph_voices_tool(name: str, voice_a: str, voice_b: str,
                          ratio: str = "") -> dict[str, Any]:
        from ..voice.design import morph_voices
        try:
            r = float(ratio) if str(ratio).strip() else 0.5
        except ValueError:
            r = 0.5
        return morph_voices((name or "").strip(), (voice_a or "").strip(),
                            (voice_b or "").strip(), ratio=r,
                            voices_dir=voices_dir())

    # ── voice biometrics (owner recognition signal) ──────────────────────

    @registry.register(
        "verify_owner_voice",
        description=(
            "Check whether an audio clip sounds like the enrolled owner "
            "voice (acoustic voice-print, no model). Returns match verdict "
            "+ confidence. Enroll first via action='enroll'."
        ),
        capability=Capability.FS_READ,
        parameters={
            "action": "str — verify | enroll | status",
            "path": "str — audio file (verify/enroll)",
        },
    )
    def verify_owner_voice_tool(action: str,
                                path: str = "") -> dict[str, Any]:
        from ..voice import biometrics as _bio
        from .filesystem import safe_media_path
        action = (action or "verify").strip().lower()
        if action == "status":
            return {"enrolled": _bio.owner_enrolled(voices_dir())}
        if action == "enroll":
            target = safe_media_path(context, path, must_exist=True)
            return _bio.register_owner_voice(str(target),
                                             voices_dir=voices_dir())
        target = safe_media_path(context, path, must_exist=True)
        return _bio.verify_voice(str(target), voices_dir=voices_dir())

    # ── voice conversation mode ──────────────────────────────────────────

    @registry.register(
        "voice_chat_mode",
        description=(
            "Turn a chat into a voice conversation: incoming voice notes "
            "are transcribed, answered by the brain, and replied as voice "
            "notes automatically. action = on | off | status | list."
        ),
        capability=Capability.FS_WRITE,
        parameters={
            "action": "str — on | off | status | list",
            "chat_key": "str — chat key (on/off/status)",
        },
    )
    def voice_chat_mode_tool(action: str,
                             chat_key: str = "") -> dict[str, Any]:
        from ..voice import conversation as _conv
        action = (action or "status").strip().lower()
        key = (chat_key or "").strip()
        if action == "on":
            if not key:
                raise ToolError("voice_chat_mode on needs a chat_key")
            _conv.set_voice_mode(key, True, voices_dir())
            return {"ok": True, "chat": key, "voice_mode": True}
        if action == "off":
            if not key:
                raise ToolError("voice_chat_mode off needs a chat_key")
            _conv.set_voice_mode(key, False, voices_dir())
            return {"ok": True, "chat": key, "voice_mode": False}
        if action == "list":
            return {"chats": _conv.list_voice_chats(voices_dir())}
        if not key:
            raise ToolError("voice_chat_mode status needs a chat_key")
        return {"chat": key,
                "voice_mode": _conv.voice_mode_on(key, voices_dir())}
