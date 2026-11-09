"""WS3 P4 — the DayRecord: the owner's day, kept and made into drafts.

The headline flow: he tells Devon what he did today — in pieces,
whenever, by chat or voice — Devon keeps the day, and on his word
makes something out of it: per-platform draft posts in his voice,
landed in the review queue for his decision.  Nothing about the day
is ever auto-posted; the #43 rule (explicit intent or review
approval) governs every outward step, unchanged.

The record is ONE new store (:class:`DayStore`, runtime-created
tables on the Batch B pattern — ``migrations.py`` stays frozen):

* ``day_fragments`` — APPEND-ONLY.  A fragment (his words, with
  timestamp + source) is never overwritten or edited; the day is a
  log, like everything else he trusts.
* ``day_sections`` — the derived sections, recomputed snapshots:
  what the organs that already watch his day saw (goal classes,
  the GoldenSwing bundle, missions, arrived artifacts).  Derived,
  so recomputation replaces them; the fragments are the truth.
* ``day_derivations`` — one receipt per derivation run: the
  contract it ran under, the drafts it landed, the drafts the
  skeptic blocked.

Days are keyed by owner-local date — Nigeria-anchored via
:func:`nomorals.pulses.timezone.resolve_owner_tz` (Africa/Lagos by
default), the same anchor the morning briefing's day runs on.

The evening derivation is a CELL on the shipped delegation
substrate (``agents/delegation.py``), not a bare function:

* researcher (``day-researcher``) — pulls the section receipts
  into the contract's claim ledger as evidence;
* planner — a real function here: platforms × the day's material,
  angles per platform;
* drafter (``day-drafter``) — the brain, voice-shaped; on a dead
  brain an honest template fallback (marked as such in the draft
  metadata, never disguised);
* skeptic (``day-skeptic``) — every number in a draft must trace
  to a record receipt (his own fragments, or a section's measured
  values).  An untraced number is challenged in the ledger and
  the draft is BLOCKED — it never reaches the queue;
* verifier (``day-verifier``, ≠ worker — the store enforces it) —
  verifies the evidence, checks platform fit + voice signals,
  signs off; the contract completes only through the store's own
  completion gate;
* delivery steward — lands surviving drafts in ``DraftQueue`` as
  ``pending_review``, where the wired review surface (WS3 P2)
  takes over.

Intake/planner/learning are not contract roles here by design:
intake is the ``/day`` surface + the brain's ``day`` tool (the
dynamic-understanding route — no keyword router), and learning is
the measurement spine's (WS3 P1) — day drafts carry their origin
in metadata, so their outcomes feed the same briefs calibration
as every other post.

The derivation fires BOTH ways (census §4/§11 defaults): on his
word, and automatically at 19:00 owner-local — the repo's evening
slot hour (the evening wrap's "day wrap + tomorrow") — via the
durable ``day derivation`` job (:func:`ensure_day_derivation_job`,
installed at partner boot beside the other standing jobs).  The
automatic run is idempotent per day, drafts-only (#43 unchanged),
and honours a per-day opt-out recorded on the record itself
(``/day skip`` / the tool's ``skip`` action): a skipped day is
never auto-derived, though his explicit word afterwards still is.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .morning_briefing import _Provider

_log = get_logger(__name__)

__all__ = [
    "DAY_PLATFORMS",
    "EVENING_JOB",
    "EVENING_TIME",
    "DayBookendProvider",
    "DaySection",
    "DayStore",
    "DerivationResult",
    "brain_llm_fn",
    "build_sections",
    "day_key",
    "day_window",
    "derive_day",
    "ensure_day_derivation_job",
    "export_day",
    "owner_tz",
    "register",
    "render_day",
    "run_evening_derivation",
    "today_key",
    "yesterday_key",
]

#: The durable evening job + its hour — 19:00 owner-local, the
#: repo's evening slot (``BRIEFING_SLOTS["evening"]`` default_time,
#: the day-wrap hour), bookending the 06:00 morning anchor.
EVENING_JOB = "day derivation"
EVENING_TIME = "19:00"

#: The cell's role names on the delegation ledger.  Verifier ≠
#: worker is enforced by the store, not by convention.
RESEARCHER = "day-researcher"
DRAFTER = "day-drafter"
SKEPTIC = "day-skeptic"
VERIFIER = "day-verifier"
STEWARD = "day-steward"

#: Platforms a day derivation drafts for, flagship first (census
#: §4: a LinkedIn post, an X post, a WhatsApp-status-sized line).
DAY_PLATFORMS = ("linkedin", "x", "whatsapp")

#: Fragment sources the record accepts (census §4: chat / voice /
#: artifact).  Anything else is stored as chat rather than refused —
#: the day is his log, not a schema court.
FRAGMENT_SOURCES = ("chat", "voice", "artifact")


# ── day keying (Nigeria-anchored) ────────────────────────────────────

def owner_tz() -> tuple[str, Any]:
    """(name, tzinfo) for the owner's day — Africa/Lagos by default,
    honouring the same env/config the pulses anchor reads."""
    from ..pulses.timezone import resolve_owner_tz

    name, tz, _src = resolve_owner_tz()
    return name, tz


def day_key(ts: float, tz: Any = None) -> str:
    """The owner-local calendar date (``YYYY-MM-DD``) containing ``ts``."""
    if tz is None:
        _name, tz = owner_tz()
    return datetime.fromtimestamp(float(ts), tz).strftime("%Y-%m-%d")


def day_window(day: str, tz: Any = None) -> tuple[float, float]:
    """[start, end) epoch bounds of an owner-local day."""
    if tz is None:
        _name, tz = owner_tz()
    start = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=tz)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


def today_key(now: float | None = None) -> str:
    return day_key(time.time() if now is None else now)


def yesterday_key(now: float | None = None) -> str:
    _name, tz = owner_tz()
    ts = time.time() if now is None else now
    yday = datetime.fromtimestamp(ts, tz) - timedelta(days=1)
    return yday.strftime("%Y-%m-%d")


# ── the store ────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS day_fragments (
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    id           TEXT NOT NULL UNIQUE,
    day          TEXT NOT NULL,
    ts           REAL NOT NULL,
    source       TEXT NOT NULL DEFAULT 'chat',
    text         TEXT NOT NULL,
    artifact_ref TEXT NOT NULL DEFAULT '',
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_day_fragments_day
    ON day_fragments (day, ts);
CREATE TABLE IF NOT EXISTS day_sections (
    day          TEXT NOT NULL,
    section      TEXT NOT NULL,
    body_json    TEXT NOT NULL DEFAULT '{}',
    computed_at  REAL NOT NULL,
    PRIMARY KEY (day, section)
);
CREATE TABLE IF NOT EXISTS day_derivations (
    id           TEXT PRIMARY KEY,
    day          TEXT NOT NULL,
    contract_id  TEXT NOT NULL DEFAULT '',
    drafts_json  TEXT NOT NULL DEFAULT '[]',
    blocked_json TEXT NOT NULL DEFAULT '[]',
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_day_derivations_day
    ON day_derivations (day, created_at);
CREATE TABLE IF NOT EXISTS day_meta (
    day             TEXT PRIMARY KEY,
    skip_derivation INTEGER NOT NULL DEFAULT 0,
    skip_note       TEXT NOT NULL DEFAULT '',
    updated_at      REAL NOT NULL
);
"""


class DayStore:
    """The day record over the context database (Batch B pattern:
    tables are created at construction, never by a migration)."""

    def __init__(self, db: Any) -> None:
        self.db = db
        self.db.executescript(_SCHEMA)

    # -- fragments: append-only, never overwritten -------------------

    def add_fragment(self, *, text: str, ts: float | None = None,
                     source: str = "chat", day: str = "",
                     artifact_ref: str = "") -> dict[str, Any]:
        """Append one fragment to its day.  There is deliberately
        no update or delete path — a fragment, once told, stands."""
        text = (text or "").strip()
        if not text:
            raise ValueError("a day fragment needs words")
        ts = time.time() if ts is None else float(ts)
        day = day or day_key(ts)
        source = source if source in FRAGMENT_SOURCES else "chat"
        frag = {
            "id": new_id("dayfrag"),
            "day": day,
            "ts": ts,
            "source": source,
            "text": text,
            "artifact_ref": artifact_ref or "",
            "created_at": time.time(),
        }
        self.db.execute(
            "INSERT INTO day_fragments (id, day, ts, source, text,"
            " artifact_ref, created_at) VALUES (?,?,?,?,?,?,?)",
            (frag["id"], frag["day"], frag["ts"], frag["source"],
             frag["text"], frag["artifact_ref"], frag["created_at"]))
        return frag

    def fragments(self, day: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM day_fragments WHERE day = ?"
            " ORDER BY ts ASC, seq ASC", (day,))
        return [dict(r) for r in rows]

    # -- derived sections: recomputed snapshots -----------------------

    def save_sections(self, day: str,
                      sections: list[DaySection]) -> None:
        now = time.time()
        for sec in sections:
            self.db.execute(
                "INSERT INTO day_sections (day, section, body_json,"
                " computed_at) VALUES (?,?,?,?)"
                " ON CONFLICT(day, section) DO UPDATE SET"
                " body_json = excluded.body_json,"
                " computed_at = excluded.computed_at",
                (day, sec.name, json.dumps(sec.to_dict()), now))

    def sections(self, day: str) -> list[DaySection]:
        rows = self.db.query(
            "SELECT * FROM day_sections WHERE day = ?", (day,))
        out = []
        for row in rows:
            try:
                out.append(DaySection.from_dict(
                    json.loads(row["body_json"] or "{}")))
            except (ValueError, TypeError) as exc:
                _log.debug("day section %s/%s unreadable: %s",
                           day, row["section"], exc)
        return out

    # -- derivation receipts -------------------------------------------

    def record_derivation(self, *, day: str, contract_id: str,
                          drafts: list[dict[str, Any]],
                          blocked: list[dict[str, Any]]) -> dict[str, Any]:
        receipt = {
            "id": new_id("dayderive"),
            "day": day,
            "contract_id": contract_id or "",
            "drafts_json": json.dumps(drafts),
            "blocked_json": json.dumps(blocked),
            "created_at": time.time(),
        }
        self.db.execute(
            "INSERT INTO day_derivations (id, day, contract_id,"
            " drafts_json, blocked_json, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (receipt["id"], receipt["day"], receipt["contract_id"],
             receipt["drafts_json"], receipt["blocked_json"],
             receipt["created_at"]))
        return receipt

    def derivations(self, day: str) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT * FROM day_derivations WHERE day = ?"
            " ORDER BY created_at ASC", (day,))
        out = []
        for row in rows:
            item = dict(row)
            item["drafts"] = json.loads(item.pop("drafts_json") or "[]")
            item["blocked"] = json.loads(item.pop("blocked_json") or "[]")
            out.append(item)
        return out

    # -- per-day flags (the derivation opt-out) ------------------------

    def set_skip(self, day: str, *, skip: bool = True,
                 note: str = "") -> dict[str, Any]:
        """Record (or clear) the owner's instruction to skip a day's
        automatic derivation.  The skip binds the evening job only —
        his explicit "make something of today" afterwards still runs,
        because his word in the moment outranks a standing flag."""
        self.db.execute(
            "INSERT INTO day_meta (day, skip_derivation, skip_note,"
            " updated_at) VALUES (?,?,?,?)"
            " ON CONFLICT(day) DO UPDATE SET"
            " skip_derivation = excluded.skip_derivation,"
            " skip_note = excluded.skip_note,"
            " updated_at = excluded.updated_at",
            (day, 1 if skip else 0, note or "", time.time()))
        return self.meta(day)

    def meta(self, day: str) -> dict[str, Any]:
        rows = self.db.query(
            "SELECT * FROM day_meta WHERE day = ?", (day,))
        if not rows:
            return {"day": day, "skip_derivation": False, "skip_note": ""}
        row = dict(rows[0])
        row["skip_derivation"] = bool(row.get("skip_derivation"))
        return row

    def skipped(self, day: str) -> bool:
        return bool(self.meta(day).get("skip_derivation"))


# ── derived sections ─────────────────────────────────────────────────

@dataclass
class DaySection:
    """One derived section of a day record: what one watching organ
    saw.  ``present`` False means the organ had nothing for the day
    — ``note`` then says so in words; a section is never padded."""

    name: str
    title: str
    present: bool
    lines: list[str] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "title": self.title,
            "present": self.present, "lines": list(self.lines),
            "numbers": list(self.numbers),
            "evidence": dict(self.evidence), "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DaySection:
        return cls(
            name=str(data.get("name") or ""), title=str(data.get("title") or ""),
            present=bool(data.get("present")),
            lines=[str(x) for x in data.get("lines") or []],
            numbers=[float(x) for x in data.get("numbers") or []],
            evidence=dict(data.get("evidence") or {}),
            note=str(data.get("note") or ""))


_NUM_TOKEN = re.compile(r"[-+]?[$₦€£]?\d[\d,]*(?:\.\d+)?%?")


def numbers_in(text: str) -> list[float]:
    """Every numeric value a text states, normalised (currency,
    commas, percent and sign stripped to the bare value)."""
    out: list[float] = []
    for token in _NUM_TOKEN.findall(text or ""):
        cleaned = token.replace(",", "").replace("%", "")
        cleaned = cleaned.strip("$₦€£")
        try:
            out.append(float(cleaned))
        except ValueError:
            continue
    return out


# ── section builders ─────────────────────────────────────────────────
#
# Each builder reads ONE watching organ and is failure-contained: a
# broken organ yields an absent section whose note names the failure,
# never a crashed derivation and never invented content.

def _fragments_section(store: DayStore, day: str) -> DaySection:
    frags = store.fragments(day)
    _name, tz = owner_tz()
    lines = []
    for frag in frags:
        stamp = datetime.fromtimestamp(frag["ts"], tz).strftime("%H:%M")
        lines.append(f"{stamp} ({frag['source']}) — {frag['text']}")
    numbers: list[float] = []
    for frag in frags:
        numbers.extend(numbers_in(frag["text"]))
    return DaySection(
        name="fragments", title="In your words",
        present=bool(frags), lines=lines, numbers=numbers,
        evidence={"fragments": len(frags),
                  "ids": [f["id"] for f in frags]},
        note="" if frags else "nothing told about this day yet")


def _goals_section(db: Any, tracker: Any, end: float) -> DaySection:
    sec = DaySection(name="goals", title="Goals", present=False)
    try:
        from ..goals import GoalTracker
        from ..goals.surface import (
            OWNER_USER_ID,
            goals_context_snapshot,
            run_sync,
        )

        tracker = tracker or GoalTracker(db)
        entries = goals_context_snapshot(
            tracker, OWNER_USER_ID, limit=8, now=end - 1.0)
        # An inactive tracker reads as: goals exist but none active.
        active = run_sync(tracker.list_goals(
            OWNER_USER_ID, status="active", limit=50))
        if not active:
            sec.note = "no goals on the tracker yet"
            return sec
        lines: list[str] = []
        numbers: list[float] = []
        for entry in entries:
            pct = entry.get("percent")
            title = str(entry.get("title") or "goal")
            if pct is None:
                lines.append(f"{title} — no reading yet")
                continue
            pct = float(pct)
            numbers.append(pct)
            if entry.get("basis") == "measured":
                lines.append(
                    f"{title} — {pct:.0f}% (measured · {entry.get('source')})")
            else:
                lines.append(f"{title} — {pct:.0f}% (self-reported)")
        sec.present = bool(lines)
        sec.lines = lines
        sec.numbers = numbers
        sec.evidence = {"goals": len(active), "readings": len(numbers)}
        if not lines:
            sec.note = "goals exist but none produced a reading"
        return sec
    except Exception as exc:  # noqa: BLE001 - contained per section
        sec.note = f"goal readings unavailable: {type(exc).__name__}: {exc}"
        return sec


def _trading_section(db: Any, day: str, start: float,
                     end: float) -> DaySection:
    """The GoldenSwing day, through the WS2 P7 bundle discipline:
    the newest journal-state bundle, VERIFIED before any member is
    read; STATE.md prose never parsed; the day window is the
    difference of the reader's own journal stats at the window's
    edges — no second parser is written here."""
    sec = DaySection(name="trading", title="Trading (GoldenSwing)",
                     present=False)
    try:
        from ..goals import providers

        located = providers._locate_gs_state_bundle(db)
        if located is None:
            sec.note = ("no GoldenSwing state bundle in the artifact"
                        " store — the trading day is not visible here")
            return sec
        artifact = located["artifact"]
        if not located.get("verified"):
            problems = list(located.get("problems") or [])
            sec.note = ("the newest GoldenSwing bundle failed"
                        " verification (" + "; ".join(problems[:3]) + ")"
                        " — no reading taken from it")
            sec.evidence = {"artifact_id": artifact.id,
                            "bundle_verified": False}
            return sec
        found = providers._member_named(
            located.get("members") or {}, providers.GS_JOURNAL_BASENAME)
        if found is None:
            sec.note = "the verified bundle carries no trade journal"
            return sec
        _arcname, journal = found
        manifest = located.get("manifest") or {}
        # The reader's stats are cumulative from `since`: the day
        # window [start, end) is stats-from-start minus
        # stats-from-end — no second parser, the reader's own
        # semantics differenced at the window's edges.
        from_start = providers._gs_journal_stats(journal, since=start)
        from_end = providers._gs_journal_stats(journal, since=end)
        closes = int(from_start["closed_trades"]
                     - from_end["closed_trades"])
        realized = round(float(from_start["realized_pnl"]
                               - from_end["realized_pnl"]), 2)
        bundle_day = providers._bundle_day(manifest.get("created_at"))
        where = f"the verified GoldenSwing bundle of {bundle_day}" \
            if bundle_day else "a verified GoldenSwing bundle"
        if closes:
            lines = [
                f"{closes} trade(s) closed that day in {where};"
                f" realized {realized:+.2f} on the day",
            ]
        else:
            lines = [f"no trades closed that day in {where}"]
        sec.present = True
        sec.lines = lines
        sec.numbers = [float(closes), realized]
        sec.evidence = {
            "artifact_id": artifact.id,
            "bundle_verified": True,
            "bundle_day": bundle_day,
            "journal_records": int(from_start["journal_records"]),
            "closed_trades_day": closes,
            "realized_pnl_day": realized,
            "unreadable_lines": int(from_start["unreadable_lines"]),
        }
        return sec
    except Exception as exc:  # noqa: BLE001 - contained per section
        sec.note = f"trading reading unavailable: {type(exc).__name__}: {exc}"
        return sec


def _missions_section(db: Any, start: float, end: float) -> DaySection:
    sec = DaySection(name="missions", title="Missions finished",
                     present=False)
    try:
        from ..missions.mission import MissionStatus, MissionStore

        done = MissionStore(db).list(status=MissionStatus.DONE, limit=200)
        finished = []
        for mission in done:
            stamp = mission.finished_at or mission.updated_at or 0.0
            if start <= float(stamp) < end:
                finished.append(mission)
        if not finished:
            sec.note = "no missions finished that day"
            return sec
        lines = []
        for mission in finished[:8]:
            label = (mission.name or mission.goal or "mission").strip()
            lines.append(f"finished: {label[:80]}")
        sec.present = True
        sec.lines = lines
        sec.numbers = [float(len(finished))]
        sec.evidence = {"missions_finished": len(finished)}
        return sec
    except Exception as exc:  # noqa: BLE001 - contained per section
        sec.note = f"mission record unavailable: {type(exc).__name__}: {exc}"
        return sec


def _artifacts_section(db: Any, artifact_store: Any, start: float,
                       end: float) -> DaySection:
    sec = DaySection(name="artifacts", title="Files that arrived",
                     present=False)
    try:
        store = artifact_store
        if store is None:
            from ..goals.providers import _bundle_artifact_store

            store = _bundle_artifact_store(db)
        if store is None:
            sec.note = "no artifact store attached"
            return sec
        arrived = [a for a in store.search(limit=200)
                   if start <= float(a.created_at or 0.0) < end]
        if not arrived:
            sec.note = "no files arrived that day"
            return sec
        lines = []
        for art in arrived[:8]:
            meta = art.metadata or {}
            label = (meta.get("filename") or meta.get("name")
                     or art.type or "file")
            origin = str(meta.get("origin") or "")
            suffix = " (sent in chat)" if origin == "chat_inbound" else ""
            lines.append(f"{label}{suffix}")
        sec.present = True
        sec.lines = lines
        sec.numbers = [float(len(arrived))]
        sec.evidence = {"artifacts": len(arrived),
                        "ids": [a.id for a in arrived[:8]]}
        return sec
    except Exception as exc:  # noqa: BLE001 - contained per section
        sec.note = f"artifact record unavailable: {type(exc).__name__}: {exc}"
        return sec


def build_sections(context: Any, day: str, *, tracker: Any = None,
                   artifact_store: Any = None) -> list[DaySection]:
    """Compute (and persist) every derived section for ``day``.

    The day's own keying carries the window: owner-local midnight
    to owner-local midnight.  Sections are snapshots — recomputing
    replaces them; fragments underneath never move."""
    db = getattr(context, "db", None)
    _name, tz = owner_tz()
    start, end = day_window(day, tz)
    store = DayStore(db) if db is not None else None
    if db is None or store is None:
        return [DaySection(
            name=name, title=name, present=False,
            note="no database attached — the day record is unavailable")
            for name in ("fragments", "goals", "trading", "missions",
                         "artifacts")]
    sections = [
        _fragments_section(store, day),
        _goals_section(db, tracker, end),
        _trading_section(db, day, start, end),
        _missions_section(db, start, end),
        _artifacts_section(db, artifact_store, start, end),
    ]
    store.save_sections(day, sections)
    return sections


def _allowed_numbers(day: str, sections: list[DaySection],
                     fragments: list[dict[str, Any]]) -> set[float]:
    """Every number a draft may state: the day's own date parts,
    the fragments' numbers, and each section's receipt values."""
    allowed: set[float] = set()
    for part in day.split("-"):
        with suppress(ValueError):
            allowed.add(float(int(part)))
    for frag in fragments:
        allowed.update(numbers_in(frag["text"]))
    for sec in sections:
        allowed.update(float(n) for n in sec.numbers)
    allowed.add(float(len(fragments)))
    return allowed


def _untraced_numbers(text: str, allowed: set[float]) -> list[float]:
    """Numbers in ``text`` with no record receipt — the skeptic's
    whole case.  A value counts as traced when it (or its rounded
    form) is among the receipts."""
    rounded = {round(v, 2) for v in allowed}
    bad: list[float] = []
    for value in numbers_in(text):
        if round(value, 2) not in rounded and value not in bad:
            bad.append(value)
    return bad


# ── the derivation cell ──────────────────────────────────────────────

@dataclass
class DerivationResult:
    day: str
    contract_id: str = ""
    drafts: list[dict[str, Any]] = field(default_factory=list)
    blocked: list[dict[str, Any]] = field(default_factory=list)
    sections: list[DaySection] = field(default_factory=list)
    note: str = ""


_PLATFORM_ANGLES = {
    "linkedin": ("the day as build-in-public: what got done, what"
                 " it means, one honest lesson"),
    "x": "one tight post: the single sharpest moment of the day",
    "whatsapp": "one short status line: the day in a breath",
}


def _plan(sections: list[DaySection],
          fragments: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The planner step — a real function, not a contract role:
    every wired draft platform gets an angle over the day's
    material; LinkedIn leads (the flagship surface)."""
    return [{"platform": platform, "angle": _PLATFORM_ANGLES[platform]}
            for platform in DAY_PLATFORMS]


def _voice_brief() -> tuple[Any, str]:
    """(profile, brief) — the owner's VoiceProfile when one is
    trained; an honest plain brief when it is not."""
    try:
        from ..social.voice import VoiceProfile

        profile = VoiceProfile.load()
        if getattr(profile, "n_posts", 0):
            return profile, profile.describe()
        return profile, ("plain, direct, first person; short"
                         " sentences; no hype, no hashtag spam")
    except Exception as exc:  # noqa: BLE001 - voice is a rung, not a gate
        _log.debug("voice profile unavailable: %s", exc)
        return None, ("plain, direct, first person; short sentences;"
                      " no hype, no hashtag spam")


def _draft_prompt(platform: str, angle: str, day: str,
                  fragments: list[dict[str, Any]],
                  sections: list[DaySection], brief: str) -> str:
    frag_lines = "\n".join(f"- {f['text']}" for f in fragments) \
        or "- (nothing told)"
    fact_lines = []
    for sec in sections:
        if sec.name == "fragments" or not sec.present:
            continue
        for line in sec.lines[:3]:
            fact_lines.append(f"- [{sec.name}] {line}")
    facts = "\n".join(fact_lines) or "- (no organ readings)"
    return (
        f"You are drafting a {platform} post for the owner, in his"
        f" voice, from his day record for {day}.\n"
        f"Voice: {brief}\n\n"
        f"The day, in his words:\n{frag_lines}\n\n"
        f"What the day's organs recorded:\n{facts}\n\n"
        "HARD RULES:\n"
        "- Use ONLY numbers that appear above. Never invent a"
        " figure, a count, or a percentage.\n"
        "- Write ordinals and small counts as words, not digits.\n"
        "- One post, ready to paste. No preamble, no surrounding"
        " quotes.\n"
        f"Angle: {angle}\n")


def _template_draft(platform: str, fragments: list[dict[str, Any]],
                    sections: list[DaySection]) -> str:
    """The honest fallback drafter (content_pipeline precedent):
    composed ONLY from record content — fragments verbatim and
    section receipts — so every number in it traces by
    construction.  Drafts made this way are marked in metadata."""
    texts = [f["text"] for f in fragments]
    facts = []
    for sec in sections:
        if sec.name == "fragments" or not sec.present:
            continue
        facts.extend(sec.lines[:2])
    if platform == "linkedin":
        parts = list(texts[:3])
        if facts:
            parts.append("From the record:")
            parts.extend(f"• {line}" for line in facts[:3])
        parts.append("Building in public, one day at a time.")
        return "\n\n".join(parts)
    if platform == "x":
        head = texts[0] if texts else ""
        out = head[:240]
        if facts and len(out) + len(facts[0]) + 3 <= 280:
            out = f"{out} — {facts[0]}"
        return out[:280]
    # whatsapp: one short status line
    head = (texts[0] if texts else "")[:160]
    return f"today: {head}" if head else "today, in the record"


def derive_day(context: Any, day: str, *, queue: Any,
               llm_fn: Callable[[str], str] | None = None,
               now: float | None = None, tracker: Any = None,
               artifact_store: Any = None,
               platforms: tuple[str, ...] = DAY_PLATFORMS
               ) -> DerivationResult:
    """Run the evening derivation for ``day`` as a delegation cell.

    Lands surviving drafts in ``queue`` as pending_review and
    returns the full receipt.  Raises only when the record itself
    is unreachable (no database) — callers answer that in words."""
    from .delegation import DelegationStore

    db = getattr(context, "db", None)
    if db is None:
        raise ValueError("the day record needs the context database")
    store = DayStore(db)
    fragments = store.fragments(day)
    sections = build_sections(context, day, tracker=tracker,
                              artifact_store=artifact_store)
    result = DerivationResult(day=day, sections=sections)
    if not fragments and not any(s.present for s in sections):
        result.note = (f"nothing recorded for {day} yet — tell me"
                       " about the day first (/day tell …)")
        return result
    allowed = _allowed_numbers(day, sections, fragments)
    contracts = DelegationStore(db)
    contract = contracts.create_contract(
        goal=(f"turn the {day} day record into per-platform draft"
              " posts the owner can review"),
        deliverable="per-platform drafts landed in the review queue",
        worker=DRAFTER, verifier=VERIFIER, skeptic=SKEPTIC,
        required_evidence=("day-sections", "draft"),
        consequential=False)
    result.contract_id = contract.id
    cid = contract.id

    # Researcher: the section receipts enter the claim ledger, and
    # the verifier checks each against the PERSISTED sections (a
    # read-back, not the researcher's word).
    persisted = {s.name: s for s in store.sections(day)}
    for sec in sections:
        summary = (sec.lines[0] if sec.present and sec.lines
                   else sec.note or "no reading")
        claim = contracts.submit_evidence(
            cid, kind="day-sections",
            claim=f"{sec.title}: {summary}", producer=RESEARCHER,
            evidence_ref=f"day:{day}:section:{sec.name}")
        held = persisted.get(sec.name)
        if held is not None and held.present == sec.present:
            contracts.verify_claim(
                claim.id, validator=VERIFIER, verdict="supported")

    profile, brief = _voice_brief()
    from ..social.tone import adapt_tone
    from ..social.voice import apply_voice, virality_score

    plan = [p for p in _plan(sections, fragments)
            if p["platform"] in platforms]
    landed: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for step in plan:
        platform = step["platform"]
        origin = "brain"
        text = ""
        if llm_fn is not None:
            try:
                text = str(llm_fn(_draft_prompt(
                    platform, step["angle"], day, fragments,
                    sections, brief)) or "").strip()
            except Exception as exc:  # noqa: BLE001 - fallback is honest
                _log.warning("day drafter brain failed (%s): %s",
                             platform, exc)
                text = ""
        if not text:
            origin = "template"
            text = _template_draft(platform, fragments, sections)
        try:
            if profile is not None:
                text = apply_voice(text, profile, llm_fn=llm_fn)
        except Exception as exc:  # noqa: BLE001 - voice never gates
            _log.debug("apply_voice failed for %s: %s", platform, exc)
        try:
            text = adapt_tone(text, platform, llm_fn=llm_fn)
        except Exception as exc:  # noqa: BLE001 - raw draft still lands
            _log.debug("adapt_tone failed for %s: %s", platform, exc)
        claim = contracts.submit_evidence(
            cid, kind="draft",
            claim=(f"{platform} draft for {day} ({len(text)} chars,"
                   f" drafter: {origin})"),
            producer=DRAFTER, evidence_ref=f"day:{day}:draft:{platform}")
        # Skeptic: every number traces to a receipt, or the draft
        # is challenged in the ledger and never lands.
        bad = _untraced_numbers(text, allowed)
        if bad:
            challenge = contracts.challenge(
                cid, challenger=SKEPTIC,
                claim=(f"{platform} draft states number(s)"
                       f" {', '.join(str(b) for b in bad)} with no"
                       " receipt in the day record"),
                target_claim_id=claim.id)
            contracts.resolve_challenge(
                challenge.id, resolver=VERIFIER,
                verdict="contradicted",
                note="untraced numbers never land")
            blocked.append({
                "platform": platform, "numbers": bad,
                "reason": "numbers with no receipt in the day record"})
            continue
        try:
            scored = virality_score(text, platform=platform)
            fit = (f"virality {scored.score:.0f} ({scored.grade});"
                   f" checked for {platform}")
        except Exception as exc:  # noqa: BLE001 - fit check is evidence
            fit = f"virality check unavailable ({type(exc).__name__})"
        contracts.verify_claim(
            claim.id, validator=VERIFIER, verdict="supported")
        _log.debug("day verifier: %s draft for %s — %s; every"
                   " number traced to the record", platform, day, fit)
        # Delivery steward: land it in the review queue.  Nothing
        # posts — P2's review surface owns every outward step.
        draft = queue.create_draft(
            text, platforms=[platform],
            metadata={
                "origin": "day-derivation",
                "day": day,
                "platform": platform,
                "drafter": origin,
                "source_fragments": [f["id"] for f in fragments],
                "sections": [s.name for s in sections if s.present],
                "contract_id": cid,
            })
        queue.propose(draft.id)
        landed.append({"draft_id": draft.id, "platform": platform,
                       "content": text})
    result.drafts = landed
    result.blocked = blocked
    if landed:
        delivery = contracts.submit_evidence(
            cid, kind="draft",
            claim=(f"delivery: {len(landed)} draft(s) landed in the"
                   f" review queue for {day} ("
                   + ", ".join(d["platform"] for d in landed) + ")"
                   + (f"; {len(blocked)} blocked by the skeptic"
                      if blocked else "")),
            producer=STEWARD, evidence_ref=f"day:{day}:delivery")
        contracts.verify_claim(
            delivery.id, validator=VERIFIER, verdict="supported")
        contracts.sign_off(
            cid, verifier=VERIFIER,
            note="sections verified, claims traced, drafts landed")
        decision = contracts.complete(cid, by=VERIFIER)
        if not decision.ok:
            result.note = ("drafts landed, but the derivation"
                           " contract could not complete: "
                           + "; ".join(decision.reasons or []))
    elif blocked:
        result.note = ("every draft was blocked by the skeptic —"
                       " numbers with no receipt in the record:"
                       " nothing was queued")
    store.record_derivation(day=day, contract_id=cid,
                            drafts=[{"draft_id": d["draft_id"],
                                     "platform": d["platform"]}
                                    for d in landed],
                            blocked=blocked)
    if not result.note:
        result.note = (f"{len(landed)} draft(s) from {day} are in"
                       " /drafts for your review")
    return result


# ── the automatic evening run ────────────────────────────────────────

def brain_llm_fn(context: Any) -> Callable[[str], str] | None:
    """The drafter's brain seam: ``brain_for(context).complete``
    adapted to the plain prompt->text callable the cell uses.
    None when no brain is available (the template fallback then
    drafts, marked as such)."""
    try:
        from ..llm.brain import brain_for

        brain = brain_for(context)
    except Exception as exc:  # noqa: BLE001 - fallback is a real rung
        _log.debug("day drafter: no brain available: %s", exc)
        return None

    def _fn(prompt: str) -> str:
        resp = brain.complete(prompt, task_kind="creative")
        return str(getattr(resp, "text", "") or "")

    return _fn


def run_evening_derivation(context: Any, *, now: float | None = None,
                           queue: Any = None,
                           llm_fn: Callable[[str], str] | None = None,
                           tracker: Any = None,
                           artifact_store: Any = None
                           ) -> dict[str, Any]:
    """The automatic 19:00 run: derive today's drafts — unless the
    day was skipped at his word, holds no fragments, or was already
    derived (the run is idempotent per day, so a catch-up refire
    never doubles the queue).  Drafts only; #43 unchanged."""
    db = getattr(context, "db", None)
    if db is None:
        return {"ok": False, "day": "",
                "error": "no database attached — the day record"
                         " is unavailable"}
    day = today_key(now)
    store = DayStore(db)
    if store.skipped(day):
        return {"ok": True, "day": day, "skipped": True, "drafts": 0,
                "note": f"derivation for {day} was skipped at the"
                        " owner's word"}
    if not store.fragments(day):
        return {"ok": True, "day": day, "empty": True, "drafts": 0,
                "note": f"nothing was told about {day} — no derivation"}
    if store.derivations(day):
        return {"ok": True, "day": day, "already_derived": True,
                "drafts": 0,
                "note": f"{day} was already derived — its drafts"
                        " are in /drafts"}
    own_queue = queue is None
    if own_queue:
        from ..social.drafts import DraftQueue

        queue = DraftQueue()
    try:
        result = derive_day(
            context, day, queue=queue,
            llm_fn=llm_fn if llm_fn is not None
            else brain_llm_fn(context),
            now=now, tracker=tracker, artifact_store=artifact_store)
    finally:
        if own_queue:
            queue.close()
    return {"ok": True, "day": day, "drafts": len(result.drafts),
            "blocked": len(result.blocked),
            "platforms": [d["platform"] for d in result.drafts],
            "contract_id": result.contract_id, "note": result.note}


def ensure_day_derivation_job(scheduler: Any) -> list[dict[str, Any]]:
    """Register the durable evening derivation job (idempotent by
    name, the social/jobs.py pattern): daily at 19:00 owner-local —
    the repo's evening slot hour — firing the ``day`` tool's
    ``evening`` action on the durable scheduler."""
    out: list[dict[str, Any]] = []
    have = {j.get("name") for j in scheduler.list_jobs()}
    if EVENING_JOB in have:
        return [{"name": EVENING_JOB, "already_scheduled": True}]
    tz_name, _tz = owner_tz()
    job = scheduler.add(
        EVENING_JOB, f"daily {EVENING_TIME}", "tool",
        {"tool": "day", "args": {"action": "evening"}},
        timezone=tz_name, missed_fire_policy="fire_now",
        overlap_policy="skip", run_timeout_s=1800, routine=False)
    out.append({"name": EVENING_JOB, "scheduled": True,
                "job_id": job.get("id")})
    return out


# ── rendering + export ───────────────────────────────────────────────

_SECTION_ICONS = {
    "fragments": "🗣",
    "goals": "🎯",
    "trading": "📈",
    "missions": "🛠",
    "artifacts": "📎",
}


def render_day(context: Any, day: str, *, tracker: Any = None,
               artifact_store: Any = None) -> str:
    """The day, whole, as one message: his words first, then what
    the organs saw, then what was made of it."""
    db = getattr(context, "db", None)
    if db is None:
        return "the day record is unavailable — no database attached"
    store = DayStore(db)
    tz_name, _tz = owner_tz()
    sections = build_sections(context, day, tracker=tracker,
                              artifact_store=artifact_store)
    lines = [f"🌅 your day — {day} ({tz_name})"]
    if store.skipped(day):
        lines.append("⏭ tonight's automatic derivation is skipped"
                     " for this day (your word stands until you"
                     " unskip it)")
    for sec in sections:
        icon = _SECTION_ICONS.get(sec.name, "•")
        if sec.present:
            lines.append(f"\n{icon} {sec.title}")
            lines.extend(f"  {line}" for line in sec.lines[:10])
        else:
            lines.append(f"\n{icon} {sec.title} — {sec.note}")
    for receipt in store.derivations(day):
        made = receipt.get("drafts") or []
        blocked = receipt.get("blocked") or []
        if made:
            names = ", ".join(d.get("platform", "") for d in made)
            lines.append(f"\n✍️ drafts made from this day: {names}"
                         " — waiting in /drafts for your review")
        if blocked:
            lines.append(f"\n🛡 {len(blocked)} draft(s) were blocked"
                         " by the fact-check (numbers with no"
                         " receipt in the record)")
    return "\n".join(lines)


def export_day(context: Any, day: str, *, tracker: Any = None,
               artifact_store: Any = None) -> Any:
    """The day record as an artifact (WS1 family): the rendered
    record, kept in the store he already searches."""
    db = getattr(context, "db", None)
    if db is None:
        return None
    store = artifact_store
    if store is None:
        from ..goals.providers import _bundle_artifact_store

        store = _bundle_artifact_store(db)
    if store is None:
        return None
    text = render_day(context, day, tracker=tracker,
                      artifact_store=store)
    return store.put_text(
        text, type="day-record", creator="dayrecord",
        metadata={"day": day, "title": f"day record {day}"})


# ── the morning bookend ──────────────────────────────────────────────

class DayBookendProvider(_Provider):
    """Morning-briefing bookend (WS3 census §4): yesterday, in his
    words, plus how many day-drafts still wait for his decision.
    Absent when there is no yesterday record — the briefing never
    pads.  Registered with the composer the weather way (lazy
    append at composer construction)."""

    name = "day_bookend"
    title = "🌅 Yesterday, in your words"
    priority = 25  # beside the draft-review provider (23)

    def collect(self, ctx: Any, since: float):
        try:
            return self._collect(ctx)
        except Exception as exc:  # noqa: BLE001 - a provider never
            _log.debug("day bookend failed: %s", exc)  # breaks the
            return None                           # briefing

    def _collect(self, ctx: Any, since: float = 0.0):
        from .morning_briefing import BriefingSection

        db = getattr(ctx, "db", None)
        if db is None:
            return None
        store = DayStore(db)
        yday = yesterday_key()
        frags = store.fragments(yday)
        if not frags:
            return None
        _name, tz = owner_tz()
        first = frags[0]
        stamp = datetime.fromtimestamp(first["ts"], tz).strftime("%H:%M")
        excerpt = first["text"]
        if len(excerpt) > 140:
            excerpt = excerpt[:137] + "…"
        lines = [f"“{excerpt}” — {stamp}, the first of"
                 f" {len(frags)} note(s) yesterday"]
        pending = 0
        try:
            from ..social.drafts import DraftQueue, reviewable_drafts

            queue = DraftQueue()
            try:
                pending = len(reviewable_drafts(queue))
            finally:
                queue.close()
        except Exception as exc:  # noqa: BLE001 - count is a garnish
            _log.debug("day bookend: draft count unavailable: %s", exc)
        if pending:
            lines.append(f"{pending} draft(s) are waiting for your"
                         " review — /drafts")
        return BriefingSection(
            name=self.name, title=self.title, priority=self.priority,
            source="day record",
            items=[{"title": lines[0], "kind": "day_bookend"}],
            lines=lines)


# ── the brain tool ───────────────────────────────────────────────────

def register(registry: Any) -> None:
    """The ``day`` tool — the brain's route into the day record.

    This is how narration in ordinary owner chat reaches the
    record: the brain's dynamic understanding recognises a
    day-telling (or a "make something of today", or a "skip
    tonight") and calls this tool — the description below is the
    recognition surface.  There is no keyword router anywhere in
    the path.
    """
    context = registry.context

    @registry.register(
        "day",
        description=(
            "The owner's DAY RECORD: an append-only log of what he"
            " did today, plus drafts made from it. Use action=tell"
            " when he narrates his day in ordinary chat (what he"
            " did, shipped, decided, saw — in his own words, one"
            " call per piece), with source=voice when the narration"
            " arrived as a voice note. Use action=derive when he"
            " asks to make something of the day ('make something"
            " of today', 'turn today into posts') — it drafts"
            " per-platform posts into the review queue; nothing"
            " posts. Use action=skip when he says to skip today's"
            " derivation (the automatic evening run then leaves"
            " the day alone), action=unskip to undo that."
            " action=show answers 'what did I do today/yesterday'"
            " from the record; action=export files the day as an"
            " artifact; action=evening is the scheduled evening"
            " run (normally fired by the durable job, not by chat)."
        ),
        capability="memory.write",
        parameters={
            "action": ("str — tell | show | derive | evening |"
                       " skip | unskip | export"),
            "text": "str — for tell: the fragment, in his words",
            "source": "str — tell source: chat | voice | artifact"
                      " (default chat)",
            "day": "str — YYYY-MM-DD (default: today, owner-local)",
        },
    )
    def _day(action: str = "show", *, text: str = "",
             source: str = "chat", day: str = "") -> dict[str, Any]:
        db = getattr(context, "db", None)
        if db is None:
            return {"ok": False,
                    "error": "the day record is unavailable — no"
                             " database attached"}
        store = DayStore(db)
        action = (action or "show").strip().lower()
        target = day.strip() or today_key()
        if action == "tell":
            try:
                frag = store.add_fragment(
                    text=text, source=source, day=target)
            except ValueError as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "day": frag["day"],
                    "fragments": len(store.fragments(frag["day"])),
                    "note": "kept in the day record"}
        if action == "show":
            return {"ok": True, "day": target,
                    "text": render_day(context, target)}
        if action == "skip":
            store.set_skip(target, skip=True,
                           note="skipped at the owner's word")
            return {"ok": True, "day": target, "skipped": True,
                    "note": "tonight's automatic derivation will"
                            " leave this day alone"}
        if action == "unskip":
            store.set_skip(target, skip=False)
            return {"ok": True, "day": target, "skipped": False}
        if action == "export":
            art = export_day(context, target)
            if art is None:
                return {"ok": False,
                        "error": "no artifact store attached"}
            return {"ok": True, "day": target,
                    "artifact": art.uri,
                    "note": "the day record is filed as an artifact"}
        if action in ("derive", "evening"):
            from ..social.drafts import DraftQueue

            queue = DraftQueue()
            try:
                if action == "evening":
                    out = run_evening_derivation(
                        context, queue=queue,
                        llm_fn=brain_llm_fn(context))
                    return out
                if not store.fragments(target) and not any(
                        s.present for s in build_sections(
                            context, target)):
                    return {"ok": False, "day": target,
                            "error": f"nothing recorded for {target}"
                                     " yet — nothing to derive"}
                result = derive_day(
                    context, target, queue=queue,
                    llm_fn=brain_llm_fn(context))
            finally:
                queue.close()
            return {"ok": True, "day": target,
                    "drafts": [d["platform"] for d in result.drafts],
                    "blocked": [b["platform"]
                                for b in result.blocked],
                    "note": result.note}
        return {"ok": False,
                "error": f"unknown day action {action!r}"}
