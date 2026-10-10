"""Music distribution pipeline: generate → master → distribute as one flow.

Build-map #60. Phase 1: Devon prepares everything (metadata, splits,
AI disclosure, cover spec, submission checklist) — the USER clicks
submit on the distributor. Phase 2 (API automation) is documented as
hooks, gated on legal review.

Three non-negotiables:
1. Royalty splits are MANDATORY — they must sum to 100. Collaboration
   without splits is a friendship-ender.
2. AI disclosure is REQUIRED on every release — never omitted. AI
   disclosure rules are hardening across distributors.
3. Phase 1 never uploads. Devon prepares; the human submits.

Legal weather (also surfaced in packet output):
- LANDR excludes AI tracks from Content ID — route accordingly.
- UMG v. DistroKid is active; distributor terms change. Every packet
  carries a "check current terms" reminder.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field

from ..core.logging_setup import get_logger

_log = get_logger("nomorals.media")

#: The AI disclosure statement. REQUIRED on every release — never omitted.
AI_DISCLOSURE = "AI-generated instrumental + AI vocals"

#: Platforms that matter, Nigerian platforms included.
PLATFORM_DEFAULTS: tuple[str, ...] = (
    "spotify", "apple_music", "boomplay", "audiomack",
)

#: Known platform display names.
PLATFORM_NAMES: dict[str, str] = {
    "spotify": "Spotify",
    "apple_music": "Apple Music",
    "boomplay": "Boomplay",
    "audiomack": "Audiomack",
    "youtube_music": "YouTube Music",
    "deezer": "Deezer",
    "tiktok": "TikTok",
    "triller": "Triller",
}


class DistributionError(RuntimeError):
    """Raised when a release can't be prepared (bad splits, missing audio)."""


# ───────────────────────── splits ───────────────────────────────────────────

def validate_splits(splits: dict[str, float | int]) -> dict[str, float]:
    """Validate royalty splits. MUST sum to 100 — never optional.

    Raises DistributionError on: empty, non-numeric, negative, or a
    total that isn't 100.
    """
    if not splits:
        raise DistributionError(
            "royalty splits are mandatory — 'collaboration without splits "
            "is a friendship-ender'. Provide e.g. artist=70, producer=30.")
    cleaned: dict[str, float] = {}
    for name, share in splits.items():
        name = (name or "").strip()
        if not name:
            raise DistributionError("split holder names can't be blank")
        try:
            val = float(share)
        except (TypeError, ValueError):
            raise DistributionError(
                f"split share for {name!r} isn't a number: {share!r}")
        if val < 0:
            raise DistributionError(
                f"split share for {name!r} can't be negative: {val}")
        cleaned[name] = val
    total = sum(cleaned.values())
    if abs(total - 100.0) > 0.01:
        raise DistributionError(
            f"royalty splits must sum to 100, got {total:g} "
            f"({', '.join(f'{k}={v:g}' for k, v in cleaned.items())})")
    return cleaned


def parse_splits(text: str) -> dict[str, float] | None:
    """Parse 'artist=70, producer=30' → dict. None when unparseable."""
    out: dict[str, float] = {}
    for part in (text or "").split(","):
        part = part.strip()
        if not part or "=" not in part:
            return None
        name, _, val = part.partition("=")
        name, val = name.strip(), val.strip().rstrip("%")
        try:
            out[name] = float(val)
        except ValueError:
            return None
    return out or None


# ───────────────────────── legal weather ────────────────────────────────────

def legal_notes() -> list[str]:
    """Current legal weather. Surfaced in every release packet."""
    return [
        "🤖 AI disclosure is REQUIRED on every upload — this packet includes "
        f"it: \"{AI_DISCLOSURE}\". AI disclosure rules are hardening; "
        "never omit it.",
        "⚠️ LANDR excludes AI-generated tracks from Content ID — if Content "
        "ID matters for this release, route mastering elsewhere or accept "
        "the exclusion.",
        "📜 UMG v. DistroKid is active and distributor terms change — check "
        "the CURRENT terms of your chosen distributor before submitting.",
        "🎵 TuneCore for serious releases; Boomy-pattern (casual) for "
        "quick drops. This packet is distributor-agnostic.",
    ]


# ───────────────────────── release packet ───────────────────────────────────

@dataclass
class ReleasePacket:
    """Everything a distributor needs. Phase 1: Devon prepares, you submit."""
    packet_id: str
    title: str
    artist: str
    song_path: str
    mastered_path: str = ""
    isrc: str = ""  # placeholder — distributor assigns or you request one
    platforms: tuple[str, ...] = PLATFORM_DEFAULTS
    splits: dict[str, float] = field(default_factory=dict)
    ai_disclosure: str = AI_DISCLOSURE
    cover_spec: str = ""
    created_ts: float = 0.0
    phase: int = 1  # 1 = prepared, human submits. 2 = API automation (gated)

    def platform_display(self) -> list[str]:
        return [PLATFORM_NAMES.get(p, p) for p in self.platforms]

    def master_targets(self) -> dict[str, dict[str, float]]:
        """Per-platform loudness targets for the master.

        Streaming services normalize to ~−14 LUFS integrated; clubs/
        DJ promos want hotter (−8). Returns platform → {"lufs",
        "true_peak_db"} so the mastering step can aim per destination
        instead of shipping one master everywhere.
        """
        out: dict[str, dict[str, float]] = {}
        for p in self.platforms:
            spec = PLATFORM_MASTER.get(p, PLATFORM_MASTER["default"])
            out[p] = dict(spec)
        return out

    def master_plan_text(self) -> str:
        """God-tier mastering plan: one target row per platform."""
        from .style import theme as _theme
        th = _theme()
        lines = [th.banner("Mastering Plan", self.title)]
        for plat, spec in self.master_targets().items():
            name = PLATFORM_NAMES.get(plat, plat)
            lines.append(
                f"  {th.bullet} {name:12s} "
                f"{spec['lufs']:+.0f} LUFS integrated, "
                f"true peak {spec['true_peak_db']:+.1f} dB")
        lines.append(th.status_line(
            True, "targets set",
            "master to the HOTTEST target, let services turn down"))
        return "\n".join(lines)


#: Per-platform mastering targets (integrated LUFS, true peak dBTP).
#: Streaming normalizes everything down to its target — mastering
#: hotter just gets turned down; mastering quieter stays quiet.
PLATFORM_MASTER: dict[str, dict[str, float]] = {
    "spotify": {"lufs": -14.0, "true_peak_db": -1.0},
    "apple": {"lufs": -16.0, "true_peak_db": -1.0},
    "youtube": {"lufs": -14.0, "true_peak_db": -1.0},
    "tiktok": {"lufs": -14.0, "true_peak_db": -1.0},
    "instagram": {"lufs": -14.0, "true_peak_db": -1.0},
    "audiomack": {"lufs": -14.0, "true_peak_db": -1.0},
    "boomplay": {"lufs": -14.0, "true_peak_db": -1.0},
    "club": {"lufs": -8.0, "true_peak_db": -0.3},
    "dj_promo": {"lufs": -9.0, "true_peak_db": -0.5},
    "default": {"lufs": -14.0, "true_peak_db": -1.0},
}


def cover_art_spec(title: str, artist: str) -> str:
    """What the cover art needs — a spec the artist approves, not AI slop."""
    return (
        f"Cover art for “{title}” by {artist}:\n"
        "• 3000×3000px minimum (distributor requirement)\n"
        "• No URLs, no social handles, no pricing text on the art\n"
        f"• Must include the AI disclosure line in the release metadata: "
        f"\"{AI_DISCLOSURE}\"\n"
        "• Tip: a bold single image beats a busy collage at thumbnail size."
    )


def submission_checklist(packet: ReleasePacket) -> list[str]:
    """The human's click-through list. Phase 1 ends here."""
    plats = ", ".join(packet.platform_display())
    splits = ", ".join(f"{k}: {v:g}%" for k, v in packet.splits.items())
    return [
        f"1. Master: {packet.mastered_path or packet.song_path} "
        "(16-bit/44.1kHz WAV minimum)",
        f"2. Metadata: title “{packet.title}”, artist “{packet.artist}”, "
        f"ISRC: {packet.isrc or '(distributor assigns)'}",
        f"3. AI disclosure on the upload form: \"{packet.ai_disclosure}\"",
        f"4. Platforms: {plats}",
        f"5. Royalty splits locked: {splits}",
        "6. Cover art: 3000×3000px, no URLs/handles/pricing",
        "7. Check the distributor's CURRENT terms (see legal notes)",
        "8. Submit — then paste the release link back here and I'll "
        "track it.",
    ]


def prepare(song_path: str, *, title: str, artist: str,
            platforms: tuple[str, ...] | list[str] | None = None,
            splits: dict[str, float | int],
            ledger: Any = None) -> ReleasePacket:
    """Prepare a release packet. Phase 1 — never uploads anything.

    Validates splits (must sum to 100), stamps the AI disclosure,
    records the split agreement in the finance ledger
    (category="royalty_split") so future earnings split automatically.
    """
    if not song_path or not os.path.isfile(song_path):
        raise DistributionError(
            f"song audio not found: {song_path!r} — render the song first "
            "(/music full).")
    title = (title or "").strip() or "untitled"
    artist = (artist or "").strip() or "unknown artist"
    clean_splits = validate_splits(splits)
    plats = tuple(p.strip().lower() for p in (platforms or PLATFORM_DEFAULTS)
                  if p and p.strip())
    if not plats:
        plats = PLATFORM_DEFAULTS

    packet = ReleasePacket(
        packet_id=uuid.uuid4().hex[:12],
        title=title,
        artist=artist,
        song_path=song_path,
        isrc="",  # distributor assigns
        platforms=plats,
        splits=clean_splits,
        ai_disclosure=AI_DISCLOSURE,  # never omitted
        cover_spec=cover_art_spec(title, artist),
        created_ts=time.time(),
        phase=1,
    )
    # Record the split agreement — future earnings split automatically.
    if ledger is not None:
        try:
            ledger.log(
                100,  # nominal — the agreement, not money moving
                category="royalty_split",
                note=(f"royalty split agreement: “{title}” by {artist} — "
                      + ", ".join(f"{k}={v:g}%" for k, v in
                                  clean_splits.items())
                      + f" [packet {packet.packet_id}]"),
                kind="spend",
                source="distribute",
            )
        except Exception:  # noqa: BLE001 — packet must not fail on logging
            _log.debug("split ledger write failed", exc_info=True)
    _log.info("release packet %s prepared for “%s”", packet.packet_id, title)
    return packet


def format_packet(packet: ReleasePacket) -> str:
    """Render the packet as the chat message the human acts on."""
    lines = [
        f"📦 **Release packet** `{packet.packet_id}`",
        f"🎵 “{packet.title}” — {packet.artist}",
        f"🤖 AI disclosure (required): \"{packet.ai_disclosure}\"",
        "",
        "**Submission checklist:**",
    ]
    lines.extend(f"  {item}" for item in submission_checklist(packet))
    lines.append("")
    lines.append("**Legal weather:**")
    lines.extend(f"  {note}" for note in legal_notes())
    lines.append("")
    lines.append("Phase 1: everything's prepared — YOU click submit on the "
                 "distributor. Paste the release link back here when it's live.")
    return "\n".join(lines)


# ───────────────────────── conversational draft ─────────────────────────────

@dataclass
class ReleaseDraft:
    """Multi-step /distribute walk-through state, keyed by chat."""
    step: str = "title"  # title → artist → platforms → splits → confirm
    title: str = ""
    artist: str = ""
    platforms: tuple[str, ...] = PLATFORM_DEFAULTS
    splits: dict[str, float] = field(default_factory=dict)
    song_path: str = ""


_pending: dict[str, ReleaseDraft] = {}


def start_draft(chat_key: str, song_path: str = "") -> ReleaseDraft:
    draft = ReleaseDraft(song_path=song_path)
    _pending[chat_key] = draft
    return draft


def pending_draft(chat_key: str) -> ReleaseDraft | None:
    return _pending.get(chat_key)


def clear_draft(chat_key: str) -> None:
    _pending.pop(chat_key, None)


def draft_prompt(draft: ReleaseDraft) -> str:
    """The next question for the current step."""
    if draft.step == "title":
        return "📦 Let's prep this release. What's the **song title**?"
    if draft.step == "artist":
        return f"🎤 Title: “{draft.title}”. Who's the **artist**?"
    if draft.step == "platforms":
        plats = ", ".join(PLATFORM_NAMES.get(p, p)
                          for p in PLATFORM_DEFAULTS)
        return (f"🎤 Artist: {draft.artist}. Which **platforms**? "
                f"(comma-separated, or 'all' for: {plats})")
    if draft.step == "splits":
        return ("💰 Platforms set. Now the **royalty splits** — MANDATORY, "
                "must sum to 100.\n"
                "Format: `artist=70, producer=30`")
    if draft.step == "confirm":
        splits = ", ".join(f"{k}={v:g}%" for k, v in draft.splits.items())
        plats = ", ".join(PLATFORM_NAMES.get(p, p)
                          for p in draft.platforms)
        return (f"📦 Ready to build the packet?\n"
                f"🎵 “{draft.title}” — {draft.artist}\n"
                f"📡 {plats}\n"
                f"💰 {splits}\n"
                f"🤖 AI disclosure: \"{AI_DISCLOSURE}\"\n\n"
                "Reply **yes** to build it, or **cancel**.")
    return "📦 Release draft — reply **cancel** to stop."


def advance_draft(draft: ReleaseDraft, text: str) -> str:
    """Advance the draft one step. Returns the next prompt or 'done'/'cancel'."""
    text = (text or "").strip()
    if text.lower() in ("cancel", "stop", "nevermind", "never mind"):
        return "cancel"
    if draft.step == "title":
        if not text:
            return draft_prompt(draft)
        draft.title = text
        draft.step = "artist"
    elif draft.step == "artist":
        if not text:
            return draft_prompt(draft)
        draft.artist = text
        draft.step = "platforms"
    elif draft.step == "platforms":
        low = text.lower()
        if low in ("all", "default", "defaults"):
            draft.platforms = PLATFORM_DEFAULTS
        else:
            wanted = [p.strip().lower().replace(" ", "_")
                      for p in text.split(",") if p.strip()]
            # accept display names too ("apple music" → apple_music)
            known = {**PLATFORM_NAMES,
                     **{v.lower(): k for k, v in PLATFORM_NAMES.items()}}
            plats = tuple(known.get(p, p) for p in wanted)
            draft.platforms = plats or PLATFORM_DEFAULTS
        draft.step = "splits"
    elif draft.step == "splits":
        parsed = parse_splits(text)
        if parsed is None:
            return ("Couldn't parse that. Format: `artist=70, producer=30` "
                    "(must sum to 100).")
        try:
            draft.splits = validate_splits(parsed)
        except DistributionError as exc:
            return f"{exc}\nTry again: `artist=70, producer=30`"
        draft.step = "confirm"
    elif draft.step == "confirm":
        if text.lower() in ("yes", "y", "build", "go", "confirm"):
            return "done"
        return "cancel"
    return draft_prompt(draft)


def parse_distribute_request(text: str) -> dict | None:
    """/distribute [song path or topic] — starts the release walk-through.

    Returns {"song_path": ...} or None when it isn't a distribute request.
    """
    raw = (text or "").strip()
    low = raw.lower()
    if low.startswith("/distribute"):
        return {"song_path": raw[len("/distribute"):].strip()}
    return None


# ───────────────────────── tool registration ────────────────────────────────

def register(registry: Any) -> None:
    from ..core.policy import Capability
    from ..core.errors import ToolError

    @registry.register(
        "distribute",
        description=(
            "Music distribution pipeline (Phase 1: prepare, human submits). "
            "action=prepare (song_path, title, artist, platforms, splits) | "
            "legal (legal weather) | checklist (packet → submission list). "
            "Splits MANDATORY and must sum to 100. AI disclosure always "
            "included. Never uploads — Phase 2 API automation gated on "
            "legal review."
        ),
        capability=Capability.FS_WRITE,
    )
    def distribute(action: str = "legal", song_path: str = "",
                   title: str = "", artist: str = "",
                   platforms: str = "", splits: str = "") -> dict[str, Any]:
        action = (action or "legal").lower()
        if action == "legal":
            return {"legal_notes": legal_notes(),
                    "ai_disclosure": AI_DISCLOSURE}
        if action == "checklist":
            return {"usage": ("build a packet first: "
                              "action=prepare, song_path=..., title=..., "
                              "artist=..., splits='artist=70, producer=30'")}
        if action == "prepare":
            from ..finance.ledger import Ledger
            plats = [p.strip() for p in platforms.split(",") if p.strip()]
            parsed = parse_splits(splits)
            if parsed is None:
                return {"ok": False,
                        "error": ("unparseable splits — use "
                                  "'artist=70, producer=30' (must sum to 100)")}
            try:
                packet = prepare(
                    song_path, title=title, artist=artist,
                    platforms=plats or None, splits=parsed,
                    ledger=Ledger())
            except DistributionError as exc:
                return {"ok": False, "error": str(exc)}
            return {
                "ok": True,
                "packet_id": packet.packet_id,
                "title": packet.title,
                "artist": packet.artist,
                "platforms": list(packet.platforms),
                "splits": packet.splits,
                "ai_disclosure": packet.ai_disclosure,
                "checklist": submission_checklist(packet),
                "legal_notes": legal_notes(),
                "phase": 1,
                "note": ("Phase 1: packet prepared — the human submits. "
                         "Never auto-uploads."),
            }
        raise ToolError(f"unknown action {action!r}")


# ───────────────────────── Phase 2 hooks (documented, gated) ───────────────

def phase2_hooks() -> dict[str, str]:
    """Phase 2 automation hooks — DO NOT call without legal review.

    Each hook documents what full automation would need. Phase 2 is
    gated on: (1) legal review of distributor terms (UMG v. DistroKid
    active), (2) the owner's explicit go-ahead, (3) real API credentials.
    """
    return {
        "mastering": ("LANDR/Masterchannel API: upload WAV → mastered "
                      "master. NOTE: LANDR excludes AI tracks from Content ID."),
        "distributor": ("TuneCore/DistroKid API: submit packet + AI "
                        "disclosure + splits. Terms change — re-check before "
                        "every automated submit."),
        "splits_payout": ("split ledger (category=royalty_split) already "
                          "records the agreement; Phase 2 wires payouts "
                          "through the money primitive (#49) with the "
                          "biometric gate."),
        "gate": ("legal review + owner go-ahead + credentials. "
                 "Until then, Phase 1: Devon prepares, human submits."),
    }
