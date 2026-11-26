"""SongRecon brain tool — identify the song in an audio file.

The brain-callable face of the same recognition ``/sham`` runs in
chat: Devon's local fingerprint catalog first, melodic hypotheses
for covers and hums second, the AudD external catalogue last —
every answer built from the receipt, never from a guess. A local
hit reports its coherent-offset fraction (an uncalibrated
alignment measure, never a probability); the word "confidence"
never appears in recognition copy.
"""

from __future__ import annotations

from typing import Any

from ..core.failures import record_failure


def register(registry: Any) -> None:
    """Attach the SongRecon tool to a registry."""
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "songrecon",
        description=(
            "SongRecon — identify the song in an audio file or voice "
            "note, the same recognition /sham runs in chat. Actions: "
            "identify <audio> (full pipeline: Devon's local "
            "fingerprint catalog first, then melodic hypotheses for "
            "covers and hums, then the AudD external catalogue; "
            "honest receipts, never a guess), enroll <audio> [title] "
            "[artist] (index a track into the local catalog so it is "
            "recognized next time), match <audio> (local catalog "
            "only), lineage <audio> (sample/lineage scan: regions + "
            "originality band), catalog (local catalog stats)."
        ),
        capability=Capability.MODEL_CALL,
        parameters={
            "action": "str — identify|enroll|match|lineage|catalog "
                      "(default identify)",
            "audio": "str — audio file path for identify|enroll|"
                     "match|lineage",
            "title": "str — title for enroll",
            "artist": "str — artist for enroll",
        },
    )
    def songrecon(action: str = "identify", audio: str = "",
                  title: str = "", artist: str = "") -> dict[str, Any]:
        action = (action or "identify").strip().lower()
        audio = (audio or "").strip()
        try:
            if action == "identify":
                return _identify(audio, context)
            if action == "enroll":
                return _enroll(audio, title, artist)
            if action == "match":
                return _match(audio)
            if action == "lineage":
                return _lineage(audio)
            if action == "catalog":
                return _catalog()
            return {"ok": False, "reason": f"unknown songrecon action '{action}'",
                    "text": f"unknown songrecon action '{action}'"}
        except Exception as exc:  # noqa: BLE001 - never raises
            record_failure("tools.songrecon.songrecon", exc)
            return {"ok": False, "reason": str(exc),
                    "text": f"songrecon failed: {exc}"}


def _identify(audio: str, context: Any) -> dict[str, Any]:
    if not audio:
        return {"ok": False, "reason": "audio is required",
                "text": "songrecon identify needs an audio file path"}
    from ..audio.songrecon import recognize
    from ..connectors.audd import open_audd

    db = getattr(context, "db", None) if context is not None else None
    audd, unavailable = open_audd(db, context, actor="songrecon")
    return recognize(audio, audd=audd, audd_unavailable=unavailable)


def _enroll(audio: str, title: str, artist: str) -> dict[str, Any]:
    if not audio:
        return {"ok": False, "reason": "audio is required",
                "text": "songrecon enroll needs an audio file path"}
    from ..audio.songrecon import enroll_track

    res = enroll_track(audio, title=title or "",
                       artist=(artist or "local").strip() or "local")
    out = {"ok": bool(res.get("ok")), "text": str(res.get("text") or ""),
           "title": str(res.get("title") or ""),
           "hashes": res.get("hashes"),
           "melody": res.get("melody") or {}}
    if not res.get("ok"):
        out["reason"] = str(res.get("reason") or "enroll failed")
    return out


def _match(audio: str) -> dict[str, Any]:
    if not audio:
        return {"ok": False, "reason": "audio is required",
                "text": "songrecon match needs an audio file path"}
    from ..audio.fingerprint import match_local_db
    from ..audio.songrecon import catalog_stats, local_match_reply

    local = match_local_db(audio)
    if local.get("ok"):
        return {"ok": True, "stage": "local",
                "text": local_match_reply(local),
                "artist": str(local.get("artist") or ""),
                "title": str(local.get("title") or ""),
                "score": local.get("score"),
                "receipt": local.get("receipt") or {}}
    stats = catalog_stats()
    reason = str(local.get("reason") or "no local match")
    return {"ok": False, "stage": "no_match",
            "text": (f"no local match for '{audio}' ({reason}) — local "
                     f"catalog holds {stats.get('tracks', 0)} track(s); "
                     "use identify for the full pipeline "
                     "(melody, then AudD)."),
            "reason": reason,
            "receipt": local.get("receipt") or {}}


def _lineage(audio: str) -> dict[str, Any]:
    if not audio:
        return {"ok": False, "reason": "audio is required",
                "text": "songrecon lineage needs an audio file path"}
    from ..audio.lineage import render_lineage_lines, scan_lineage_db

    res = scan_lineage_db(audio)
    if not res.get("ok"):
        return {"ok": False, "reason": str(res.get("reason") or "scan failed"),
                "text": (f"lineage scan unavailable: "
                         f"{res.get('reason', 'no match')}")}
    lines = render_lineage_lines(res)
    return {"ok": True, "stage": "lineage",
            "text": "\n".join(lines), "lines": lines,
            "result": {k: v for k, v in res.items()}}


def _catalog() -> dict[str, Any]:
    from ..audio.songrecon import catalog_stats

    stats = catalog_stats()
    return {"ok": bool(stats.get("ok")), "stage": "catalog",
            "text": str(stats.get("text") or ""),
            "tracks": stats.get("tracks", 0),
            "hashes": stats.get("hashes", 0),
            "version": stats.get("version", ""),
            "reason": str(stats.get("reason") or "")}
