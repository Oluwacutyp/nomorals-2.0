"""Expose audio — thin wrappers over the real audio engines.

Delegates, never rebuilds:

* Native Shazam — ``nomorals/audio/fingerprint.py``: Wang-2003
  local-maximum constellation fingerprinting, local sqlite DB.
  No API, no key, no network. Scope is honest: it identifies audio
  already in the *local* database (Devon's productions, the user's
  library) — it is NOT a global commercial catalogue.
* Transcript-as-timeline — ``nomorals/audio/edit.py``:
  ``transcript_edit`` / ``remove_fillers``.
* Book characters — ``nomorals/audio/characters.py``: spoiler
  discipline is structural (memories past the cutoff never reach the
  model). :func:`talk_to_character` adds a ``spoiler_level`` param:
  the effective cutoff is ``min(character cutoff, spoiler_level)``,
  so the character can never reveal beyond the chapter the reader
  has reached.
* Backends by profile — ``nomorals/audio/pipeline.py``:
  ``EngineRegistry`` picks STT/TTS backends from
  ``resource_profile`` (termux / laptop / workstation / pc).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

__all__ = [
    "fingerprint_audio",
    "identify_audio",
    "match_batch_audio",
    "analyze_audio",
    "remove_fillers_from",
    "transcript_as_timeline",
    "talk_to_character",
    "character_prompt",
    "pick_backends",
    "backend_for_profile",
]


# ── native Shazam (Wang-2003, local only) ──────────────────────────────────


def fingerprint_audio(path: str, *, peak_mode: str = "localmax",
                      max_hashes: int | None = None,
                      max_seconds: float | None = None) -> dict[str, Any]:
    """Fingerprint ``path`` with Shazam-style constellation hashes.

    ``peak_mode="localmax"`` is the true Wang-2003 local-maximum
    filter (``fingerprint.py:405``). Honest fallback, from the
    underlying docstring: without numpy, or when the local-max pass
    finds no peaks, it falls back to the per-band strongest-peak
    pass — the returned ``peak_mode_requested`` records what was
    asked, not what the FFT path silently chose. Never raises.
    """
    from .fingerprint import fingerprint as _fingerprint_fn

    try:
        kw: dict[str, Any] = {"peak_mode": peak_mode}
        if max_hashes is not None:
            kw["max_hashes"] = max_hashes
        if max_seconds is not None:
            kw["max_seconds"] = max_seconds
        hashes = _fingerprint_fn(path, **kw)
        return {
            "ok": True,
            "path": str(path),
            "peak_mode_requested": peak_mode,
            "hashes": hashes,
            "hash_count": len(hashes),
            "note": "Local fingerprint only. Identification needs a local "
                    "FingerprintDB (see identify_audio) — this is not a "
                    "global commercial catalogue.",
        }
    except Exception as exc:  # noqa: BLE001 - wrapper is never-raising
        return {"ok": False, "path": str(path), "reason": str(exc)}


def identify_audio(path: str, *, db: Any = None,
                   threshold: float | None = None) -> dict[str, Any]:
    """One call: "what is this audio?" → match against the local DB.

    Delegates to ``fingerprint.identify``. ``{"ok": False}`` means no
    confident local match — never a guessed title. Audio Devon has
    never heard and that is not in the local library cannot be
    identified by this path; that is a coverage limit, not a bug.
    """
    from .fingerprint import identify as _identify

    kw: dict[str, Any] = {}
    if db is not None:
        kw["db"] = db
    if threshold is not None:
        kw["threshold"] = threshold
    return _identify(path, **kw)


def match_batch_audio(paths: Sequence[str], *, db: Any = None,
                      threshold: float | None = None) -> list[dict[str, Any]]:
    """Identify many clips at once (batch Shazam, per-item ``ok``)."""
    from .fingerprint import match_batch as _match_batch

    kw: dict[str, Any] = {}
    if db is not None:
        kw["db"] = db
    if threshold is not None:
        kw["threshold"] = threshold
    return _match_batch(list(paths), **kw)


def analyze_audio(path: str, *, max_seconds: float | None = None) -> dict[str, Any]:
    """Native acoustic analysis (tempo, key, chord, tuning, quality flags).

    Delegates to ``fingerprint.analyze``. Failures land in
    ``ok/reason`` — never raised, never faked.
    """
    from .fingerprint import analyze as _analyze

    kw: dict[str, Any] = {}
    if max_seconds is not None:
        kw["max_seconds"] = max_seconds
    return _analyze(path, **kw).to_dict()


# ── transcript-as-timeline ─────────────────────────────────────────────────


def transcript_as_timeline(audio: str, *, language: str = "en",
                           transcriber: Any = None) -> dict[str, Any]:
    """Transcribe ``audio`` to a word-timed editable transcript.

    Delegates to ``edit.transcript_edit``. Returns ``ok=False`` when
    no transcriber is available — never an empty transcript passed
    off as a transcription.
    """
    from .edit import transcript_edit as _transcript_edit

    kw: dict[str, Any] = {}
    if transcriber is not None:
        kw["transcriber"] = transcriber
    result = _transcript_edit(audio, language=language, **kw)
    if result is None or not getattr(result, "words", None):
        return {"ok": False, "audio": str(audio),
                "reason": "could not transcribe — no transcriber available "
                          "(install faster-whisper) or audio unreadable"}
    data = result.to_dict()
    data["ok"] = True
    data["audio"] = str(audio)
    return data


def remove_fillers_from(audio: str, *, lang: str = "en",
                        fillers: Sequence[str] | None = None,
                        pad: float = 0.2,
                        words: Any = None,
                        transcriber: Any = None,
                        out_dir: Any = None) -> dict[str, Any]:
    """Cut filler words ("um", "uh", "you know", "ẹẹm"…) from ``audio``.

    Delegates to ``edit.remove_fillers``. No fillers found → the
    original path is returned with a note; never a fake edit.
    """
    from .edit import remove_fillers as _remove_fillers

    kw: dict[str, Any] = {"lang": lang, "pad": pad}
    if fillers is not None:
        kw["fillers"] = fillers
    if words is not None:
        kw["words"] = words
    if transcriber is not None:
        kw["transcriber"] = transcriber
    if out_dir is not None:
        kw["out_dir"] = out_dir
    return _remove_fillers(audio, **kw)


# ── book characters, structural spoiler discipline ─────────────────────────


def _effective_cutoff(character: Any, spoiler_level: int | None) -> int:
    """Effective chapter cap: ``min(character cutoff, spoiler_level)``.

    ``spoiler_level`` is the chapter the reader has reached. ``None``
    means "the character's own cutoff". The cap only ever moves
    *down* — a caller can restrict what the character reveals, never
    extend it past the character's own knowledge.
    """
    cap = int(character.knows_up_to())
    if spoiler_level is not None:
        try:
            cap = min(cap, max(1, int(spoiler_level)))
        except (TypeError, ValueError):
            pass
    return max(1, cap)


def character_prompt(character: Any, *, spoiler_level: int | None = None) -> str:
    """The character's system prompt, capped at ``spoiler_level``.

    Delegates to ``Character.to_system_prompt`` with the effective
    cutoff. The prompt states the cutoff as identity ("It is
    currently chapter N"), and only memories ``<= N`` are rendered
    into it — later chapters are never loaded.
    """
    return character.to_system_prompt(
        current_chapter=_effective_cutoff(character, spoiler_level))


def talk_to_character(character: Any, question: str, *,
                      spoiler_level: int | None = None,
                      llm_fn: Callable[[str, str], str] | None = None,
                      history: list[dict[str, str]] | None = None,
                      voice_fn: Callable[..., str | None] | None = None) -> Any:
    """Interview a book character without spoilers past ``spoiler_level``.

    ``spoiler_level`` is the chapter the reader has reached (int).
    The character answers from memories ``<= min(its cutoff,
    spoiler_level)`` — it genuinely does not know anything later, so
    it cannot leak it even if asked directly. ``llm_fn`` is required:
    without a dialogue engine the call is refused honestly (the
    underlying ``talk_to`` never fakes a character's voice).

    ``spoiler_level`` must be >= 1 when given; anything else is
    refused rather than silently clamped to a revealing value.
    """
    from .characters import TalkResult as _TalkResult
    from .characters import talk_to as _talk_to

    if spoiler_level is not None:
        try:
            level = int(spoiler_level)
        except (TypeError, ValueError):
            return _TalkResult(
                False, character=getattr(character, "name", ""),
                reason=f"spoiler_level must be an int chapter, got {spoiler_level!r}")
        if level < 1:
            return _TalkResult(
                False, character=getattr(character, "name", ""),
                reason=f"spoiler_level must be >= 1, got {level}")
    effective = _effective_cutoff(character, spoiler_level)
    return _talk_to(character, question, current_chapter=effective,
                          llm_fn=llm_fn, history=history, voice_fn=voice_fn)


# ── backends by resource_profile ───────────────────────────────────────────


def pick_backends(profile: str = "", *, audience: str = "private",
                  available_stt: Sequence[str] | None = None,
                  available_tts: Sequence[str] | None = None) -> dict[str, Any]:
    """Pick STT/TTS backends for a ``resource_profile``.

    Delegates to ``pipeline.EngineRegistry``. Unknown profiles fall
    back to ``"pc"`` (the registry's own rule). ``audience="public"``
    structurally removes non-commercial backends (XTTS) from the TTS
    candidates before any selection happens.
    """
    from .pipeline import EngineRegistry

    registry = EngineRegistry(profile)
    return {
        "profile": registry.profile,
        "requested_profile": profile or "",
        "audience": audience,
        "stt": registry.pick_stt(available_stt),
        "tts": registry.pick_tts(audience, available_tts),
        "stt_candidates": registry.stt_candidates(),
        "tts_candidates": registry.tts_candidates(audience),
        "describe": registry.describe(),
    }


def backend_for_profile(profile: str = "") -> dict[str, Any]:
    """Full backend table for a profile (STT/TTS orders, audience split)."""
    from .pipeline import EngineRegistry

    return EngineRegistry(profile).describe()
