"""The GoldenSwing desk draft landing (WS3-P7).

GoldenSwing's /desk card is content evidence for the cooperative:
GS emits the card (PNG + caption) as a nomorals-bundle (kind
``gs-desk-card``, written by ``goldenswing/alerts/desk_bundle.py``)
and this module is the Devon side of the handoff.  When a bundle
artifact lands in Devon's artifact store — a file arriving in
chat, an artifact import — the landing recognizes the bundle by
its CONTRACT (the manifest kind), never by filename, and creates
exactly ONE draft per GS day in the review queue at
``pending_review``: the caption is the draft's text, the
GS-rendered PNG is attached as the draft's media (stored as an
artifact, byte-for-byte as GS rendered it — never re-rendered),
and the provenance metadata names the bundle and the GS day.

Reader discipline mirrors the WS2-P7 journal-state reader
(``goals/providers.py``): the manifest kind is checked first,
``verify_bundle`` runs BEFORE any member is read, and a bundle
that fails verification lands NO draft and names its failure —
the landing never falls back to an older bundle or a partial
read.  A bundle that is not a desk bundle is not an error: it is
simply not this module's business.
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

#: The bundle kind the GS desk writer records in its manifest —
#: one string, both repos (goldenswing/alerts/desk_bundle.py).
GS_DESK_BUNDLE_KIND = "gs-desk-card"

#: Member names inside the desk bundle (the writer's contract).
DESK_PNG_MEMBER = "desk.png"
DESK_CAPTION_MEMBER = "caption.txt"
DESK_FAMILY_MEMBER = "manifest.json"

_LANDINGS_DDL = """
CREATE TABLE IF NOT EXISTS desk_draft_landings (
  day TEXT PRIMARY KEY,
  draft_id TEXT NOT NULL DEFAULT '',
  png_artifact_id TEXT NOT NULL DEFAULT '',
  bundle_sha TEXT NOT NULL DEFAULT '',
  landed_at REAL NOT NULL DEFAULT 0
);
"""


def _member_named(members: dict[str, bytes],
                  basename: str) -> bytes | None:
    """One member's bytes: exact arcname, else basename equality —
    never a substring guess (the providers.py discipline)."""
    if basename in members:
        return members[basename]
    for name, data in members.items():
        if Path(name).name == basename:
            return data
    return None


def inspect_desk_bundle(data: bytes) -> dict[str, Any]:
    """Classify + verify one candidate bundle's bytes.

    Returns a dict: ``desk`` False when the bytes are not a desk
    bundle at all (not an archive, no contract manifest, another
    kind — none of this module's business); ``desk`` True with
    ``ok`` False and named ``problems`` when the bundle CLAIMS
    the desk kind but fails verification or is missing a member;
    ``desk`` True and ``ok`` True with ``day`` / ``caption`` /
    ``png`` when it may land.  Member contents are read only
    after ``verify_bundle`` passes.
    """
    from ..archives import open_bundle, read_bundle_members, verify_bundle

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "candidate.zip"
        path.write_bytes(bytes(data))
        try:
            opened = open_bundle(path)
        except Exception as exc:  # noqa: BLE001 - not a bundle at all
            _log.debug("desk landing: not an openable bundle: %s", exc)
            return {"desk": False}
        manifest = opened.get("manifest") or {}
        if str(manifest.get("kind") or "") != GS_DESK_BUNDLE_KIND:
            return {"desk": False}
        verdict = verify_bundle(path)
        if not verdict.get("ok"):
            return {"desk": True, "ok": False,
                    "problems": [str(p) for p in
                                 verdict.get("problems") or []]}
        members = read_bundle_members(path)
    png = _member_named(members, DESK_PNG_MEMBER)
    caption_raw = _member_named(members, DESK_CAPTION_MEMBER)
    family_raw = _member_named(members, DESK_FAMILY_MEMBER)
    problems: list[str] = []
    if png is None:
        problems.append(f"desk bundle carries no {DESK_PNG_MEMBER}"
                        " member")
    if caption_raw is None:
        problems.append(f"desk bundle carries no"
                        f" {DESK_CAPTION_MEMBER} member")
    family: dict[str, Any] = {}
    if family_raw is not None:
        try:
            parsed = json.loads(family_raw.decode("utf-8"))
            if isinstance(parsed, dict):
                family = parsed
        except (ValueError, UnicodeDecodeError):
            problems.append("desk bundle family manifest is not"
                            " readable JSON")
    day = str(family.get("day") or "")
    if not day:
        problems.append("desk bundle family manifest names no day")
    if problems:
        return {"desk": True, "ok": False, "problems": problems}
    return {"desk": True, "ok": True, "problems": [],
            "day": day,
            "caption": caption_raw.decode("utf-8", "replace").strip(),
            "png": png, "family": family}


class DeskLandingStore:
    """Which GS days have already landed a draft (runtime-created
    table on the shared Database — the Batch B pattern).  One
    draft per GS day: a re-sent bundle for a landed day is a
    named no-op, never a duplicate draft."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_LANDINGS_DDL)

    def landed(self, day: str) -> dict[str, Any] | None:
        row = self.db.query_one(
            "SELECT * FROM desk_draft_landings WHERE day = ?",
            (day,))
        return dict(row) if row else None

    def record(self, *, day: str, draft_id: str,
               png_artifact_id: str, bundle_sha: str) -> None:
        self.db.execute(
            "INSERT INTO desk_draft_landings (day, draft_id,"
            " png_artifact_id, bundle_sha, landed_at)"
            " VALUES (?,?,?,?,?)",
            (day, draft_id, png_artifact_id, bundle_sha,
             time.time()))


def land_desk_draft(db: Any, queue: Any, store: Any, data: bytes,
                    *, source: str = "") -> dict[str, Any]:
    """Land one landed bundle's bytes as a desk draft.

    ``db`` is the context database, ``queue`` the DraftQueue,
    ``store`` the ArtifactStore the bundle arrived through.  The
    result dict says exactly what happened: ``landed`` with the
    new ``draft_id``; ``already`` when the GS day already has its
    draft; ``error`` naming the verification problems when a desk
    bundle failed (no draft exists for it); or ``desk`` False
    when the bytes were never a desk bundle.
    """
    import hashlib

    inspected = inspect_desk_bundle(data)
    if not inspected.get("desk"):
        return {"landed": False, "desk": False}
    if not inspected.get("ok"):
        problems = "; ".join(inspected.get("problems") or [])
        _log.warning("desk bundle refused: %s", problems)
        return {"landed": False, "desk": True,
                "error": f"desk bundle failed verification: "
                         f"{problems}"}
    day = str(inspected["day"])
    landings = DeskLandingStore(db)
    prior = landings.landed(day)
    if prior is not None:
        return {"landed": False, "desk": True, "already": True,
                "day": day, "draft_id": prior.get("draft_id", "")}
    png_art = store.put(
        inspected["png"], type="image", creator="gs_desk_bundle",
        mime="image/png",
        metadata={"origin": "gs_desk_bundle", "gs_day": day,
                  "filename": DESK_PNG_MEMBER},
        provenance={"source_type": "goldenswing",
                    "source_id": f"desk:{day}",
                    "verification_state": "verified"})
    draft = queue.create_draft(
        str(inspected["caption"]), platforms=None,
        metadata={
            "origin": "gs_desk_bundle",
            "gs_day": day,
            "bundle_kind": GS_DESK_BUNDLE_KIND,
            "arrival": source,
            "media": [{"artifact": png_art.id, "mime": "image/png",
                       "name": DESK_PNG_MEMBER}],
        })
    queue.propose(draft.id)
    landings.record(
        day=day, draft_id=draft.id, png_artifact_id=png_art.id,
        bundle_sha=hashlib.sha256(bytes(data)).hexdigest())
    return {"landed": True, "desk": True, "day": day,
            "draft_id": draft.id, "png_artifact_id": png_art.id}


__all__ = [
    "DESK_CAPTION_MEMBER", "DESK_FAMILY_MEMBER", "DESK_PNG_MEMBER",
    "GS_DESK_BUNDLE_KIND", "DeskLandingStore", "inspect_desk_bundle",
    "land_desk_draft",
]
