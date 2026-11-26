"""SongRecon — the shared song-identification pipeline.

One implementation of "what is this audio?" used by every surface:
``/sham`` (and its ``/songrecon`` alias) in chat, the ``songrecon``
brain tool, and ``nm audio identify``. The staging is the Doc 24
order — Devon's own fingerprint memory first, her melody catalog
for covers and hums second, the AudD external catalogue last —
and every reply is built from the receipt, never from a guess.

The render helpers below are the exact ``/sham`` wordings (moved
here verbatim so every surface says it the same way); the SR1
receipt tests pin them. Two vocabulary rules hold everywhere:
a local hit reports its coherent-offset fraction (an uncalibrated
alignment measure, never a probability), and the word "confidence"
never appears in user-facing recognition copy.
"""

from __future__ import annotations

import os
from typing import Any

from .failures import record_failure
from .fingerprint import (
    describe_audio,
    external_recognition_receipt,
    match_local_db,
    no_match_receipt,
)

__all__ = [
    "recognize",
    "enroll_track",
    "catalog_stats",
    "receipt_of",
    "local_match_reply",
    "melodic_hypotheses",
    "external_match_reply",
    "unmatched_reply",
]


def receipt_of(local: dict[str, Any]) -> dict[str, Any]:
    """Pull the SongRecon receipt out of a local match result."""
    receipt = (local or {}).get("receipt")
    return receipt if isinstance(receipt, dict) else {}


def local_match_reply(local: dict[str, Any]) -> str:
    """Render a local hit from its SongRecon receipt."""
    receipt = receipt_of(local)
    artist = receipt.get("artist") or local.get("artist") \
        or "unknown artist"
    title = receipt.get("title") or local.get("title") \
        or "unknown title"
    strength = receipt.get("match_strength") \
        or local.get("match_strength") or {}
    value = strength.get("value", local.get("score", 0))
    numerator = strength.get("numerator")
    denominator = strength.get("denominator")
    if numerator is not None and denominator is not None:
        measure = (f"coherent-offset fraction {value} — "
                   f"{numerator} of {denominator} query landmark "
                   "hashes aligned at one reference offset "
                   "(uncalibrated; not a probability)")
    else:
        measure = (f"coherent-offset fraction {value} — query "
                   "landmark hashes aligned at one reference "
                   "offset (uncalibrated; not a probability)")
    lines = [f"🎵 I know this one — {artist} — {title}",
             f"(recognized from my own library; {measure})"]
    windows = receipt.get("evidence_windows") \
        or local.get("evidence_windows") or []
    if windows:
        window = windows[0]
        lines.append(
            "evidence window: query "
            f"{window.get('query_start_s')}–"
            f"{window.get('query_end_s')}s ↔ recording "
            f"{window.get('reference_start_s')}–"
            f"{window.get('reference_end_s')}s")
    alternatives = receipt.get("alternatives") \
        or local.get("alternatives") or []
    if alternatives:
        rendered = []
        for alt in alternatives[:3]:
            alt_strength = alt.get("match_strength") or {}
            alt_value = alt_strength.get(
                "value", alt.get("score", ""))
            rendered.append(
                f"{alt.get('artist') or 'unknown artist'} — "
                f"{alt.get('title') or 'unknown title'} "
                f"(coherent-offset fraction {alt_value})")
        lines.append("other local candidates: "
                     + "; ".join(rendered))
    lines.append(f"/play {artist} {title}")
    return "\n".join(lines)


def melodic_hypotheses(audio_path: str) -> str | None:
    """SR3 melodic step: ranked hypotheses after a class-1 miss.

    Voice notes are the hum intake — a hummed, sung or covered
    query is matched by tune against the enrolled melody
    catalog. Returns the rendered reply, or None when no
    hypothesis clears the floor (the AudD chain then proceeds
    exactly as before SR3). Never raises; hypotheses never
    render as identifications and carry no /play line.
    """
    try:
        from .melodic import (
            match_melody_db,
            render_hypothesis_lines,
        )
        res = match_melody_db(audio_path)
    except Exception as exc:  # noqa: BLE001
        record_failure(
            "audio.songrecon.melodic_hypotheses", exc)
        return None
    hypotheses = res.get("hypotheses") or []
    if not res.get("ok") or not hypotheses:
        return None
    lines = ["🎶 no recording match in my fingerprint catalog — "
             "but the melody itself points to enrolled work(s):"]
    lines.extend(render_hypothesis_lines(hypotheses))
    lines.append("AudD external catalogue was not searched "
                 "(a local melodic hypothesis was found first).")
    return "\n".join(lines)


def external_match_reply(
        res: dict[str, Any],
        local: dict[str, Any]) -> str:
    """Render an AudD hit, explicitly labelled as fallback."""
    receipt = external_recognition_receipt(
        res, local_receipt=receipt_of(local))
    artist = receipt.get("artist") or "Unknown artist"
    title = receipt.get("title") or "Unknown title"
    lines = [f"🎵 {artist} — {title}",
             "Identified by AudD — external catalogue fallback "
             "(not Devon's local fingerprint catalog)."]
    windows = receipt.get("evidence_windows") or []
    if windows and windows[0].get("provider_timecode"):
        lines.append(
            f"AudD timecode: {windows[0]['provider_timecode']}")
    album = receipt.get("album") or ""
    release = receipt.get("release_date") or ""
    if album:
        detail = album
        if release:
            detail += f" ({release[:4]})"
        lines.append(f"💿 {detail}")
    if receipt.get("song_link"):
        lines.append(f"🔗 {receipt['song_link']}")
    lines.append(f"/play {artist} {title}")
    return "\n".join(lines)


def unmatched_reply(
        local: dict[str, Any], *, lead: str, reason: str,
        external_status: str, heard: str,
        external_searched: bool = False) -> str:
    """Render an honest non-match with the actual search scope."""
    receipt = receipt_of(local)
    final_receipt = no_match_receipt(
        receipt, reason=reason,
        attempted_sources=(
            ["devon_local_fingerprint"]
            + (["audd"] if external_searched else [])))
    tracks = final_receipt.get("catalog_track_count")
    if tracks is None:
        catalog = local.get("catalog") or {}
        tracks = catalog.get("tracks", "unknown number of")
    if receipt.get("status") == "unavailable":
        local_reason = receipt.get("reason") \
            or local.get("reason") or "local matching unavailable"
        search_line = (
            "Devon's local fingerprint catalog was not searched "
            f"({local_reason}); {external_status}.")
    else:
        search_line = (
            "Searched Devon's local fingerprint catalog "
            f"({tracks} indexed track(s)); {external_status}.")
    lines = [lead, search_line]
    if reason:
        lines.append(reason)
    lines.append(f"👂 but here's what I hear: {heard}")
    return "\n".join(lines)


def recognize(
        path: str | os.PathLike[str], *,
        db: Any = None,
        audd: Any = None,
        audd_unavailable: str = "") -> dict[str, Any]:
    """One call: "what is this audio?" → receipt-carrying result.

    The Doc 24 staging, shared by chat, brain tool and CLI:

    1. **Devon's own memory** — local fingerprint match against her
       own productions + the indexed library. Zero API, zero cost.
    2. **Devon's melodic memory (SR3)** — when no recording match
       lands, the melody catalog answers the different question
       "is this that song": covers, re-performances and hummed
       voice notes get ranked hypotheses (never identifications).
    3. **AudD** — the global commercial catalogue, only when the
       caller supplies a working connector (``audd``). Pass
       ``audd_unavailable="vault_locked"`` when the credential
       vault could not be opened, or ``"not_configured"`` when no
       AudD setup exists; the reply then says so honestly instead
       of claiming a search that never happened.
    4. **Honest ears** — when neither can name it, the reply says
       what she actually hears (tempo, key, loudness) instead of
       shrugging.

    ``audd`` is an object with a ``recognize(path)`` method (the
    AudD connector — see :func:`nomorals.connectors.audd.open_audd`).
    Never raises.
    """
    path = str(path or "")
    if not path:
        text = (
            "nothing to identify — reply to a voice note or audio "
            "message with /sham, or send audio with /sham as the caption.")
        return {"ok": False, "stage": "no_audio", "text": text,
                "receipt": {}, "artist": "", "title": "",
                "reason": "no audio given"}
    try:
        local = match_local_db(path, db=db)
    except Exception as exc:  # noqa: BLE001
        record_failure("audio.songrecon.recognize", exc)
        local = {"ok": False, "reason": f"local match failed: {exc}",
                 "receipt": {}}
    if local.get("ok"):
        receipt = receipt_of(local)
        return {"ok": True, "stage": "local",
                "text": local_match_reply(local),
                "receipt": receipt,
                "artist": str(receipt.get("artist")
                              or local.get("artist") or ""),
                "title": str(receipt.get("title")
                             or local.get("title") or ""),
                "reason": ""}
    melodic_reply = melodic_hypotheses(path)
    if melodic_reply is not None:
        return {"ok": True, "stage": "melodic",
                "text": melodic_reply,
                "receipt": receipt_of(local),
                "artist": "", "title": "", "reason": ""}
    heard = describe_audio(path)
    if audd is None:
        if audd_unavailable == "vault_locked":
            return {"ok": False, "stage": "unmatched",
                    "text": unmatched_reply(
                        local,
                        lead="couldn't name that track",
                        reason=("the credential vault is locked, so AudD "
                                "could not be searched"),
                        external_status=("AudD external catalogue was not "
                                         "searched"),
                        heard=heard),
                    "receipt": receipt_of(local),
                    "artist": "", "title": "",
                    "reason": "vault locked"}
        return {"ok": False, "stage": "unmatched",
                "text": unmatched_reply(
                    local,
                    lead="couldn't name that track",
                    reason=("music recognition isn't set up (it's one free "
                            "AudD API key away)"),
                    external_status=("AudD external catalogue was not "
                                     "searched"),
                    heard=heard),
                "receipt": receipt_of(local),
                "artist": "", "title": "",
                "reason": "AudD not configured"}
    try:
        res = audd.recognize(path)
    except Exception as exc:  # noqa: BLE001
        record_failure("audio.songrecon.recognize", exc)
        return {"ok": False, "stage": "unmatched",
                "text": unmatched_reply(
                    local,
                    lead="couldn't complete recognition",
                    reason=str(exc),
                    external_status=("AudD external catalogue was "
                                     "searched"),
                    external_searched=True,
                    heard=heard),
                "receipt": receipt_of(local),
                "artist": "", "title": "",
                "reason": str(exc)}
    if not isinstance(res, dict):
        res = {"ok": False, "reason": "AudD returned no result"}
    if not res.get("ok"):
        if res.get("needs_key"):
            # Surface the connector's own reason — it carries the
            # free-API setup guidance (dashboard URL, 300 free
            # requests). A bare "no key configured" strands the user.
            reason = str(res.get("reason") or "no AudD key configured")
            return {"ok": False, "stage": "unmatched",
                    "text": unmatched_reply(
                        local,
                        lead="couldn't name that track",
                        reason=reason,
                        external_status=("AudD external catalogue was not "
                                         "searched"),
                        heard=heard),
                    "receipt": receipt_of(local),
                    "artist": "", "title": "",
                    "reason": reason}
        reason = str(res.get("reason")
                     or "could not identify that audio")
        if "no match in AudD" in reason:
            return {"ok": False, "stage": "unmatched",
                    "text": unmatched_reply(
                        local,
                        lead="no reliable match found",
                        reason=("neither catalog accepted a recording "
                                "match for this clip"),
                        external_status=("AudD external catalogue was "
                                         "searched"),
                        external_searched=True,
                        heard=heard),
                    "receipt": receipt_of(local),
                    "artist": "", "title": "",
                    "reason": reason}
        return {"ok": False, "stage": "unmatched",
                "text": unmatched_reply(
                    local,
                    lead="couldn't complete recognition",
                    reason=reason,
                    external_status=("AudD external catalogue was "
                                     "searched"),
                    external_searched=True,
                    heard=heard),
                "receipt": receipt_of(local),
                "artist": "", "title": "",
                "reason": reason}
    receipt = external_recognition_receipt(
        res, local_receipt=receipt_of(local))
    return {"ok": True, "stage": "external",
            "text": external_match_reply(res, local),
            "receipt": receipt,
            "artist": str(receipt.get("artist") or ""),
            "title": str(receipt.get("title") or ""),
            "reason": ""}


def enroll_track(
        path: str | os.PathLike[str], *,
        title: str = "",
        artist: str = "local",
        source: str = "library",
        db: Any = None) -> dict[str, Any]:
    """Index a track into the local recognition memory.

    Same enrollment the ``/audio fingerprint`` verb runs: the
    fingerprint goes into Devon's local catalog, and the work's
    melody is enrolled alongside it (SR3) so covers and hums of
    it can be hypothesized later. Never raises.
    """
    from pathlib import Path as _Path

    from .fingerprint import default_fingerprint_db
    from .melodic import default_melody_catalog

    try:
        store = db or default_fingerprint_db()
        res = store.add_track(
            path, title=title or _Path(str(path)).stem,
            artist=artist, source=source)
    except Exception as exc:  # noqa: BLE001
        record_failure("audio.songrecon.enroll_track", exc)
        return {"ok": False, "reason": f"enroll failed: {exc}",
                "text": f"🎛️ enroll failed: {exc}"}
    if not res.get("ok"):
        return {"ok": False, "reason": str(res.get("reason", "failed")),
                "text": f"🎛️ {res.get('reason', 'failed')}"}
    text = (f"🎛️ fingerprinted '{res.get('title')}' — "
            f"{res.get('hashes')} hashes in local memory")
    try:
        work = default_melody_catalog().enroll_fingerprint_track(
            store, int(res.get("track_id") or 0), path)
    except Exception as exc:  # noqa: BLE001
        record_failure("audio.songrecon.enroll_track", exc)
        work = {"ok": False, "reason": f"melody enroll failed: {exc}"}
    if work.get("ok"):
        text += (f"; melody enrolled for cover/hum matching "
                 f"({work.get('notes')} notes)")
    else:
        text += (f"; melody not enrolled "
                 f"({work.get('reason', 'extraction failed')})")
    out = dict(res)
    out["text"] = text
    out["melody"] = work
    return out


def catalog_stats(db: Any = None) -> dict[str, Any]:
    """Local recognition catalog at a glance. Never raises."""
    from .fingerprint import default_fingerprint_db

    try:
        store = db or default_fingerprint_db()
        snap = store.catalog_snapshot()
        return {"ok": True,
                "tracks": int(snap.get("tracks") or 0),
                "hashes": int(snap.get("hashes") or 0),
                "version": str(snap.get("version") or ""),
                "text": (f"🎛️ local fingerprint catalog: "
                         f"{int(snap.get('tracks') or 0)} track(s), "
                         f"{int(snap.get('hashes') or 0)} hashes "
                         f"({str(snap.get('version') or '')})")}
    except Exception as exc:  # noqa: BLE001
        record_failure("audio.songrecon.catalog_stats", exc)
        return {"ok": False, "reason": f"catalog stats failed: {exc}",
                "tracks": 0, "hashes": 0, "version": "",
                "text": f"🎛️ catalog unavailable ({exc})"}
