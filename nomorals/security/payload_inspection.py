"""Payload inspection — the anticrash organ (DESIGN §17 +
defense brief §1; owner doctrine 2026-11-03: "you can't defend
without knowing how to attack").

One feature-based inspector for hostile chat payloads, built
from the attack taxonomy as STRUCTURAL features — ratios,
densities, counts — never keyword or string lists (the standing
no-hardcoded-routing law applies to detectors too).  Three
consumers share it: the group moderation pipeline (a sensor
ahead of the Moderator), self-protection on Devon's linked
accounts, and the future file/media malware scanner (register
§11 — same extractors).

Honest scope: Devon cannot stop another member's phone from
rendering a payload.  What she controls is exposure time —
delete-fast in groups she admins, and on her own surfaces the
payload never reaches memory or the brain.

Pure stdlib, no Devon imports: the organ is extractable as-is.
Consumers supply the normalized envelope (producers live in the
platform adapters) plus a small context dict for window state
the organ cannot see (roster size, sticker velocity).
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

# ── thresholds (named constants — data, calibratable) ──────────
# Each feature ramps 0 → 1 between its soft (flag) and hard
# (quarantine) line.  A single class at 1.0 quarantines alone:
# bombs are never averaged away.
THRESHOLDS: dict[str, tuple[float, float]] = {
    # length bomb
    "chars": (4000.0, 12000.0),
    "bytes": (12000.0, 48000.0),
    "lines": (300.0, 2000.0),
    "max_line": (2000.0, 8000.0),
    # combining-char / zalgo
    "combining_ratio": (0.35, 0.80),
    "combining_max_run": (12.0, 40.0),
    "combining_total": (400.0, 4000.0),
    # zero-width / control flood
    "invisible_count": (200.0, 1500.0),
    "invisible_ratio": (0.20, 0.60),
    # entity / mention flood
    "mentions": (15.0, 40.0),
    "mention_roster_ratio": (0.25, 0.60),
    "entities": (60.0, 200.0),
    "entity_density": (8.0, 25.0),          # per 100 chars
    "custom_emoji": (30.0, 120.0),
    # quote-chain bomb
    "quote_depth": (8.0, 20.0),
    "quoted_chars": (8000.0, 30000.0),
    # media-metadata bomb
    "media_max_dim": (32768.0, 131072.0),
    "media_seconds": (86400.0, 604800.0),   # 24 h → 7 d
    "media_count": (30.0, 100.0),
    "media_frames": (512.0, 4096.0),
    # sticker flood
    "sticker_velocity": (6.0, 15.0),        # per window, consumer-fed
    "sticker_repeats": (5.0, 12.0),         # consumer-fed
    "sticker_bytes": (500_000.0, 5_000_000.0),
}

#: Below this aggregate a payload flows on untouched; at or
#: above it the payload is flagged (allowed, audited, counted);
#: a class subscore of 1.0 quarantines outright.
FLAG_SCORE = 0.40

_CLASS_OF_FEATURE = {
    "chars": "length", "bytes": "length", "lines": "length",
    "max_line": "length",
    "combining_ratio": "combining", "combining_max_run": "combining",
    "combining_total": "combining",
    "invisible_count": "invisible", "invisible_ratio": "invisible",
    "mentions": "entity", "mention_roster_ratio": "entity",
    "entities": "entity", "entity_density": "entity",
    "custom_emoji": "entity",
    "quote_depth": "quote", "quoted_chars": "quote",
    "media_max_dim": "media", "media_seconds": "media",
    "media_count": "media", "media_frames": "media",
    "sticker_velocity": "sticker", "sticker_repeats": "sticker",
    "sticker_bytes": "sticker",
}

CLASS_REASONS = {
    "length": "oversized payload",
    "combining": "combining-character flood (zalgo density)",
    "invisible": "invisible/control character flood",
    "entity": "mention/entity flood",
    "quote": "quote-chain bomb",
    "media": "absurd media metadata",
    "sticker": "sticker flood",
    "signature": "known crash-payload shape (learned signature)",
}

_INVISIBLE = frozenset(
    [chr(c) for c in (0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF)]
    + [chr(c) for c in range(0x202A, 0x202F)]      # bidi embeds
    + [chr(c) for c in range(0x2066, 0x206A)]      # bidi isolates
)


def _is_invisible(ch: str) -> bool:
    if ch in _INVISIBLE:
        return True
    code = ord(ch)
    if code < 0x20 and ch not in ("\n", "\t", "\r"):
        return True                                  # C0 controls
    if 0x7F <= code <= 0x9F:
        return True                                  # DEL + C1
    return False


# ── the envelope ────────────────────────────────────────────────

@dataclass
class MessageEnvelope:
    """The normalized message shape every producer fills and
    every defense consumer reads (brief §1.3).  Absent data is
    None/empty — never invented."""
    platform: str = ""
    chat_key: str = ""
    sender_id: str = ""
    message_id: str = ""
    text: str = ""
    entities: list[dict[str, Any]] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    media: dict[str, Any] | None = None
    forward: dict[str, Any] | None = None
    quote_depth: int = 0
    quoted_chars: int = 0
    sticker: dict[str, Any] | None = None
    raw_len: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform, "chat_key": self.chat_key,
            "sender_id": self.sender_id,
            "message_id": self.message_id, "text": self.text,
            "entities": self.entities, "mentions": self.mentions,
            "media": self.media, "forward": self.forward,
            "quote_depth": self.quote_depth,
            "quoted_chars": self.quoted_chars,
            "sticker": self.sticker, "raw_len": self.raw_len,
        }


def normalize_envelope(raw: dict[str, Any] | None, *,
                       platform: str = "", chat_key: str = "",
                       sender_id: str = "", message_id: str = "",
                       text: str = "") -> MessageEnvelope:
    """Coerce a producer's raw dict (or None) into an envelope.
    Defensive by construction: untrusted adapter payloads with
    wrong types degrade to defaults, never to exceptions."""
    raw = raw if isinstance(raw, dict) else {}
    entities: list[dict[str, Any]] = []
    for ent in raw.get("entities") or []:
        if isinstance(ent, dict) and ent.get("type"):
            entities.append({
                "type": str(ent.get("type") or ""),
                "offset": int(ent.get("offset") or 0),
                "length": int(ent.get("length") or 0),
                "user_id": str(ent.get("user_id") or ""),
                "url": str(ent.get("url") or ""),
            })
    mentions: list[str] = []
    for m in raw.get("mentions") or []:
        m = str(m or "")
        if m and m not in mentions:
            mentions.append(m)
    media = raw.get("media")
    if isinstance(media, dict):
        media = {
            "kind": str(media.get("kind") or ""),
            "width": media.get("width"),
            "height": media.get("height"),
            "seconds": media.get("seconds"),
            "bytes": media.get("bytes"),
            "mime": str(media.get("mime") or ""),
            "ext": str(media.get("ext") or ""),
            "frames": media.get("frames"),
            "count": media.get("count"),
        }
    else:
        media = None
    forward = raw.get("forward")
    if isinstance(forward, dict):
        forward = {
            "is_forwarded": bool(forward.get("is_forwarded")),
            "score": forward.get("score"),
            "origin_kind": str(forward.get("origin_kind") or ""),
            "origin_name": str(forward.get("origin_name") or ""),
            "sender_chat_id": str(forward.get("sender_chat_id")
                                  or ""),
            "is_automatic": bool(forward.get("is_automatic")),
        }
    else:
        forward = None
    sticker = raw.get("sticker")
    if isinstance(sticker, dict):
        sticker = {
            "animated": bool(sticker.get("animated")),
            "bytes": sticker.get("bytes"),
            "pack": str(sticker.get("pack") or ""),
            "key": str(sticker.get("key") or ""),
        }
    else:
        sticker = None
    body = raw.get("text")
    body = body if isinstance(body, str) else text
    return MessageEnvelope(
        platform=str(raw.get("platform") or platform),
        chat_key=str(raw.get("chat_key") or chat_key),
        sender_id=str(raw.get("sender_id") or sender_id),
        message_id=str(raw.get("message_id") or message_id),
        text=body,
        entities=entities, mentions=mentions, media=media,
        forward=forward,
        quote_depth=max(0, int(raw.get("quote_depth") or 0)),
        quoted_chars=max(0, int(raw.get("quoted_chars") or 0)),
        sticker=sticker,
        raw_len=max(0, int(raw.get("raw_len") or len(body))),
    )


# ── feature extraction ──────────────────────────────────────────

def _num(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


_EXT_FAMILY = {
    "jpg": "image", "jpeg": "image", "png": "image", "gif": "image",
    "webp": "image", "bmp": "image", "svg": "image",
    "mp4": "video", "mov": "video", "mkv": "video", "webm": "video",
    "avi": "video",
    "mp3": "audio", "ogg": "audio", "opus": "audio", "wav": "audio",
    "m4a": "audio", "flac": "audio",
    "pdf": "doc", "zip": "archive", "rar": "archive", "7z": "archive",
    "exe": "binary", "apk": "binary", "dmg": "binary", "msi": "binary",
    "bat": "binary", "sh": "binary", "com": "binary", "scr": "binary",
}


def _type_mismatch(media: dict[str, Any]) -> float:
    """Declared extension family vs declared MIME family: a
    structural lie about what the file is (1.0) or not (0.0)."""
    ext = str(media.get("ext") or "").lower().lstrip(".")
    mime = str(media.get("mime") or "").lower()
    if not ext or not mime or "/" not in mime:
        return 0.0
    mime_family = {"image": "image", "video": "video",
                   "audio": "audio"}.get(mime.split("/", 1)[0])
    if mime_family is None:
        mime_family = {"application/pdf": "doc",
                       "application/zip": "archive"}.get(mime, "")
    ext_family = _EXT_FAMILY.get(ext, "")
    if not mime_family or not ext_family:
        return 0.0
    return 0.0 if mime_family == ext_family else 1.0


def extract_features(env: MessageEnvelope, *,
                     context: dict[str, Any] | None = None
                     ) -> dict[str, float]:
    """Every structural feature the classes score on.  Pure:
    same envelope + context → same features."""
    context = context or {}
    text = env.text or ""
    chars = len(text)
    marks = 0
    max_run = 0
    run = 0
    invisible = 0
    graphemes = 0
    for ch in text:
        cat = unicodedata.category(ch)
        if cat in ("Mn", "Me"):
            marks += 1
            run += 1
            if run > max_run:
                max_run = run
            continue
        run = 0
        graphemes += 1
        if _is_invisible(ch):
            invisible += 1
    base = max(1, graphemes)
    lines = text.split("\n")
    media = env.media or {}
    width = _num(media.get("width"))
    height = _num(media.get("height"))
    roster = _num(context.get("roster_size"))
    mention_n = len(env.mentions)
    sticker = env.sticker or {}
    features = {
        "chars": float(chars),
        "bytes": float(len(text.encode("utf-8", "replace"))),
        "lines": float(len(lines)) if text else 0.0,
        "max_line": float(max((len(ln) for ln in lines),
                              default=0)),
        "graphemes": float(graphemes),
        "combining_total": float(marks),
        "combining_ratio": marks / base,
        "combining_max_run": float(max_run),
        "invisible_count": float(invisible),
        "invisible_ratio": invisible / max(1, chars),
        "entities": float(len(env.entities)),
        # Per-100-chars with a 100-char floor: a rate is only
        # meaningful over a real span of text — one entity on a
        # 1-char message is not a density attack.
        "entity_density": (100.0 * len(env.entities)
                           / max(100, chars)),
        "custom_emoji": float(sum(
            1 for e in env.entities
            if e.get("type") in ("custom_emoji", "text_mention"))),
        "mentions": float(mention_n),
        "mention_roster_ratio": (mention_n / roster
                                 if roster > 0 else 0.0),
        "quote_depth": float(env.quote_depth),
        "quoted_chars": float(env.quoted_chars),
        "media_max_dim": max(width, height),
        "media_seconds": _num(media.get("seconds")),
        "media_count": _num(media.get("count")) if media else 0.0,
        "media_frames": _num(media.get("frames")),
        "media_mismatch": _type_mismatch(media) if media else 0.0,
        "sticker_velocity": _num(context.get("sticker_velocity")),
        "sticker_repeats": _num(context.get("sticker_repeats")),
        "sticker_bytes": _num(sticker.get("bytes")),
    }
    return features


# ── scoring + verdict ───────────────────────────────────────────

def _ramp(value: float, soft: float, hard: float) -> float:
    if value <= soft:
        return 0.0
    if value >= hard:
        return 1.0
    return (value - soft) / (hard - soft)


@dataclass
class PayloadVerdict:
    action: str                  # "allow" | "flag" | "quarantine"
    score: float                 # 0..1 aggregate (max-led blend)
    classes: dict[str, float]    # per-class subscores
    features: dict[str, float]   # raw features (audit + learning)
    signature: str               # bucketed feature fingerprint
    reason: str                  # one human line

    def to_dict(self) -> dict[str, Any]:
        return {"action": self.action, "score": round(self.score, 3),
                "classes": {k: round(v, 3)
                            for k, v in self.classes.items()},
                "features": self.features,
                "signature": self.signature, "reason": self.reason}


def fingerprint(features: dict[str, float]) -> str:
    """Bucketed feature-vector fingerprint: per scored feature,
    a band digit (0 below soft, 1 in the ramp, 2 at/over hard)
    plus an order-of-magnitude digit.  Two payloads of the same
    SHAPE share a fingerprint; their content is never stored."""
    parts = []
    for name in sorted(THRESHOLDS):
        soft, hard = THRESHOLDS[name]
        value = float(features.get(name) or 0.0)
        band = 0 if value <= soft else (2 if value >= hard else 1)
        mag = 0 if value <= 0 else min(
            6, int(math.log10(value)) + 1)
        parts.append(f"{name}:{band}{mag}")
    digest = hashlib.sha256(
        "|".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def inspect_payload(env: MessageEnvelope, *,
                    context: dict[str, Any] | None = None,
                    signatures: "SignatureStore | None" = None
                    ) -> PayloadVerdict:
    """Score one envelope.  An active learned signature
    short-circuits to quarantine; otherwise the worst class
    leads and the rest only add weight — never dilute it."""
    features = extract_features(env, context=context)
    sig = fingerprint(features)
    if signatures is not None and signatures.is_active(sig):
        return PayloadVerdict(
            action="quarantine", score=1.0,
            classes={"signature": 1.0}, features=features,
            signature=sig, reason=CLASS_REASONS["signature"])
    classes: dict[str, float] = {}
    for name, (soft, hard) in THRESHOLDS.items():
        sub = _ramp(float(features.get(name) or 0.0), soft, hard)
        cls = _CLASS_OF_FEATURE[name]
        if sub > classes.get(cls, 0.0):
            classes[cls] = sub
    if features.get("media_mismatch"):
        classes["media"] = max(classes.get("media", 0.0), 0.85)
    classes = {k: v for k, v in classes.items() if v > 0}
    if not classes:
        return PayloadVerdict(
            action="allow", score=0.0, classes={}, features=features,
            signature=sig, reason="")
    ranked = sorted(classes.values(), reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else 0.0
    score = min(1.0, top + 0.25 * second)
    worst = max(classes, key=lambda k: classes[k])
    if top >= 1.0:
        action = "quarantine"
    elif score >= FLAG_SCORE:
        action = "flag"
    else:
        action = "allow"
    return PayloadVerdict(
        action=action, score=score, classes=classes,
        features=features, signature=sig,
        reason=CLASS_REASONS[worst] if action != "allow" else "")


# ── signature learning ──────────────────────────────────────────

PROMOTE_SENDERS = 3          # distinct senders before a shape
                             # enforces on its own
DECAY_DAYS = 90.0            # unseen actives demote to candidate


class SignatureStore:
    """Learned crash-payload shapes: bucketed fingerprints with
    sighting counts — never message content.  A fingerprint
    starts as a count-only CANDIDATE; promotion (≥3 distinct
    senders, or the owner) makes it ACTIVE (matching shapes
    quarantine on sight); actives unseen for DECAY_DAYS demote
    back; an admin false-positive correction demotes instantly
    and makes the shape re-earn its promotion.

    Persistence follows the house rule: an unreadable store is
    quarantined aside (``quarantined`` flag set, file preserved)
    and the store starts empty — derived data is rebuilt by
    observation, never silently merged with a torn file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.data: dict[str, dict[str, Any]] = {}
        self.quarantined = False
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.data = {}
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) \
                    or not isinstance(raw.get("signatures"), dict):
                raise ValueError("signature store shape is wrong")
            self.data = raw["signatures"]
        except (OSError, ValueError):
            stamp = int(time.time())
            try:
                self.path.replace(
                    self.path.with_suffix(f".corrupt-{stamp}"))
            except OSError as exc:
                _log.debug("payload_inspection.load failed", exc_info=True)
            self.data = {}
            self.quarantined = True

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"signatures": self.data}, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def _entry(self, signature: str) -> dict[str, Any]:
        return self.data.setdefault(signature, {
            "count": 0, "senders": [], "first_seen": 0.0,
            "last_seen": 0.0, "active": False, "promoted_by": "",
            "false_positives": 0})

    def _decay(self, entry: dict[str, Any], now: float) -> None:
        if entry.get("active") and entry.get("last_seen") \
                and now - float(entry["last_seen"]) \
                > DECAY_DAYS * 86400:
            entry["active"] = False
            entry["promoted_by"] = ""

    def observe(self, signature: str, sender_id: str, *,
                now: float | None = None) -> dict[str, Any]:
        """One sighting of a shape.  Returns the entry after any
        promotion the sighting earned."""
        now = now if now is not None else time.time()
        entry = self._entry(signature)
        self._decay(entry, now)
        entry["count"] = int(entry.get("count") or 0) + 1
        entry["last_seen"] = now
        if not entry.get("first_seen"):
            entry["first_seen"] = now
        senders = entry.setdefault("senders", [])
        if sender_id and sender_id not in senders:
            senders.append(sender_id)
            del senders[:-20]
        if not entry.get("active") \
                and len([s for s in senders if s]) >= PROMOTE_SENDERS:
            entry["active"] = True
            entry["promoted_by"] = "sightings"
        self.save()
        return entry

    def is_active(self, signature: str, *,
                  now: float | None = None) -> bool:
        entry = self.data.get(signature)
        if not entry:
            return False
        self._decay(entry, now if now is not None
                    else time.time())
        return bool(entry.get("active"))

    def promote(self, signature: str, *, by: str = "owner"
                ) -> dict[str, Any]:
        entry = self._entry(signature)
        entry["active"] = True
        entry["promoted_by"] = by
        entry["last_seen"] = time.time()
        self.save()
        return entry

    def demote(self, signature: str, *, reason: str = ""
               ) -> dict[str, Any]:
        """Admin correction: the shape was a false positive.
        Demotion is instant and the shape must re-earn promotion
        from a clean sender set."""
        entry = self._entry(signature)
        entry["active"] = False
        entry["promoted_by"] = ""
        entry["senders"] = []
        entry["false_positives"] = int(
            entry.get("false_positives") or 0) + 1
        entry["demote_reason"] = reason[:200]
        self.save()
        return entry

    def candidates(self) -> list[dict[str, Any]]:
        """Count-only shapes for the owner digest, newest first,
        in plain data (the caller words them)."""
        rows = []
        for sig, entry in self.data.items():
            if entry.get("active"):
                continue
            rows.append({
                "signature": sig,
                "count": int(entry.get("count") or 0),
                "senders": len([s for s in
                                entry.get("senders") or [] if s]),
                "last_seen": float(entry.get("last_seen") or 0),
                "false_positives": int(
                    entry.get("false_positives") or 0)})
        return sorted(rows, key=lambda r: -r["last_seen"])
