"""Regulatory change monitoring — "tell me when the rule changes."

Build-map #80 (Thomson Reuters Regulatory Intelligence / Compliance.ai
pattern). Nobody offers CBN/SEC/NDPA/FIRS change monitoring as a
consumer/SMB product in Nigeria; this module closes that gap.

Source reliability first: every source entry is an *official* regulator
domain (verified October 2026) — no fabricated URLs, no scraped-blog
guessing. Live fetching is an opt-in injectable seam (``fetcher``): by
default the module does no network I/O at all. The expert-in-the-loop
pattern means every alert carries the official source link and a "verify
consequential changes at the source" note — the AI flags, the human
confirms.

Positioning (factual, not moralizing): legal *information*, never legal
*advice*. Alerts explain what changed and link the official source; they
never tell the owner what to do. :data:`DISCLAIMER` rides every output.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .contracts import DISCLAIMER, information_only_check
from ..core.logging_setup import get_logger

__all__ = [
    "DISCLAIMER",
    "REGULATORS",
    "REGULATORY_CALENDAR",
    "ALERT_STYLES",
    "Regulator",
    "RegulatoryItem",
    "RegWatch",
    "RegulatoryWatch",
    "OBLIGATION_CONTROLS",
    "controls_for",
    "alert_text",
    "impact_score",
    "upcoming_deadlines",
    "sync_calendar_to_portfolio",
    "check_all",
    "ensure_schedule",
    "control_regwatch",
    "REGCHECK_ACTION",
    "REGCHECK_CRON",
]

_log = get_logger(__name__)

#: Scheduler action + cron for the daily regulatory scan.
REGCHECK_ACTION = "regulatory_check"
REGCHECK_CRON = "0 7 * * *"  # 07:00 daily, before the morning briefing

_DEFAULT_DB = os.path.join(
    os.path.expanduser("~"), ".nomorals", "legal", "regulatory.db"
)


# ── source registry (official domains only; no fabricated URLs) ──────

@dataclass
class Regulator:
    """One monitored regulator: official identity, no invented endpoints."""
    code: str
    name: str
    site: str
    blurb: str
    typical_topics: list[str] = field(default_factory=list)


REGULATORS: dict[str, Regulator] = {
    "CBN": Regulator(
        code="CBN",
        name="Central Bank of Nigeria",
        site="https://www.cbn.gov.ng",
        blurb="Circulars, supervision guidelines, payments-system rules. "
              "Monitor: Documents → Circulars on the official site.",
        typical_topics=["fintech licensing", "payments", "AML",
                        "foreign exchange", "cash policy", "BVN"],
    ),
    "SEC": Regulator(
        code="SEC",
        name="Securities and Exchange Commission (Nigeria)",
        site="https://sec.gov.ng",
        blurb="Capital-market rules, collective investment schemes, "
              "digital-asset (crypto) regulation.",
        typical_topics=["securities", "crypto", "fundraising",
                        "investment advisers"],
    ),
    "NDPA": Regulator(
        code="NDPA",
        name="Nigeria Data Protection Commission (NDPC)",
        site="https://ndpc.gov.ng",
        blurb="NDP Act 2023 enforcement: circulars, audit returns, DPCO "
              "licensing, penalties. Official commission site.",
        typical_topics=["data protection", "audit returns", "DPO",
                        "penalties", "compliance organisations"],
    ),
    "FIRS": Regulator(
        code="FIRS",
        name="Federal Inland Revenue Service",
        site="https://www.firs.gov.ng",
        blurb="Tax circulars, filing deadlines, withholding-tax changes, "
              "e-invoicing rules.",
        typical_topics=["tax", "VAT", "withholding tax", "e-invoicing",
                        "filing deadlines"],
    ),
    "CAC": Regulator(
        code="CAC",
        name="Corporate Affairs Commission",
        site="https://www.cac.gov.ng",
        blurb="Company registration, annual returns, post-incorporation "
              "filings, beneficial-ownership rules.",
        typical_topics=["annual returns", "incorporation", "company filing",
                        "beneficial ownership"],
    ),
    "NCC": Regulator(
        code="NCC",
        name="Nigerian Communications Commission",
        site="https://www.ncc.gov.ng",
        blurb="Telecom licensing, SIM registration, data and tariff rules, "
              "VAS/aggregator regulation.",
        typical_topics=["telecom licensing", "SIM registration", "tariffs",
                        "data regulation"],
    ),
    "SON": Regulator(
        code="SON",
        name="Standards Organisation of Nigeria",
        site="https://son.gov.ng",
        blurb="Product standards, SONCAP certification for imports, MANCAP "
              "for manufactured goods.",
        typical_topics=["product standards", "SONCAP", "MANCAP",
                        "certification"],
    ),
}

#: Alert output styles.
ALERT_STYLES = ("full", "brief")

#: Recurring regulatory deadlines — the compliance calendar. (month, day,
#: regulator, title). Year-agnostic; resolved against the current year.
REGULATORY_CALENDAR: tuple[tuple[int, int, str, str], ...] = (
    (3, 31, "NDPA", "NDPC Compliance Audit Return (CAR) filing deadline"),
    (1, 1, "CAC", "CAC annual returns window opens (file early in the year)"),
    (12, 31, "CAC", "CAC annual returns — year-end cutoff reminder"),
    (1, 31, "FIRS", "VAT returns rhythm check — monthly filing discipline"),
    (6, 30, "FIRS", "Companies income tax — 6-month post-year-end checkpoint"),
)

#: Words that mark an item as needing the owner's attention *today*.
_URGENT_WORDS = frozenset(
    "prohibit prohibit(s) prohibited ban banned suspend suspended revoke "
    "revoked penalty penalties fine fines enforcement immediate immediately "
    "take effect sanction sanctions freeze frozen deadline".split()
)


def _norm_reg(code: str) -> str:
    c = (code or "").strip().upper()
    # Accept "NDPC" as an alias for the NDPA watch key.
    return "NDPA" if c == "NDPC" else c


def _looks_urgent(text: str) -> bool:
    words = set(re.findall(r"[a-z]+", (text or "").lower()))
    return bool(words & _URGENT_WORDS)


# ── items ─────────────────────────────────────────────────────────────

@dataclass
class RegulatoryItem:
    """One regulatory update. ``source_url`` is always the official
    regulator domain or an explicit item URL supplied by the fetcher —
    never invented."""
    id: str
    regulator: str
    title: str
    summary: str = ""
    source_url: str = ""
    ref_no: str = ""
    published: float = 0.0
    topics: list[str] = field(default_factory=list)
    urgency: str | None = None  # None → auto-detect; "normal" | "urgent"
    effective_date: float = 0.0  # when the change takes effect (0 = unknown)
    updated: bool = False  # True when this is a change to a seen item

    def __post_init__(self) -> None:
        if self.urgency in ("normal", "urgent"):
            return  # caller set it explicitly
        if _looks_urgent(f"{self.title} {self.summary}"):
            self.urgency = "urgent"
        else:
            self.urgency = "normal"

    def digest(self) -> str:
        """Content hash for change detection (TR Regulatory Intelligence
        pattern: detect changes over time, not just new items)."""
        import hashlib
        h = hashlib.sha1()
        h.update(f"{self.title}\n{self.summary}\n{self.ref_no}".encode(
            "utf-8", "replace"))
        return h.hexdigest()[:16]


def item_from_dict(d: dict) -> Optional[RegulatoryItem]:
    """Build an item from a fetcher's dict. Returns None on bad input —
    a broken source can never break the watch."""
    try:
        reg = _norm_reg(str(d.get("regulator", "")))
        if reg not in REGULATORS:
            return None
        title = str(d.get("title", "")).strip()
        if not title:
            return None
        url = str(d.get("source_url", "")).strip()
        if not url:
            url = REGULATORS[reg].site  # fall back to the official homepage
        return RegulatoryItem(
            id=str(d.get("id") or uuid.uuid4().hex[:12]),
            regulator=reg,
            title=title,
            summary=str(d.get("summary", "") or ""),
            source_url=url,
            ref_no=str(d.get("ref_no", "") or ""),
            published=float(d.get("published") or 0.0),
            topics=[str(t) for t in (d.get("topics") or []) if str(t).strip()],
            urgency=(str(d.get("urgency") or "").strip().lower()
                     if str(d.get("urgency") or "").strip().lower()
                     in ("normal", "urgent") else None),
            effective_date=float(d.get("effective_date") or 0.0),
        )
    except Exception:  # noqa: BLE001 — bad source data, not a crash
        return None


# ── obligation → control mapping ───────────────────────────────────────

@dataclass
class Control:
    """Obligation-to-control mapping (MetricStream pattern): the rule
    requires X → here's the template → here's the checklist."""
    obligation: str
    template: str
    checklist: list[str] = field(default_factory=list)
    regulators: list[str] = field(default_factory=list)


OBLIGATION_CONTROLS: dict[str, Control] = {
    "data protection audit": Control(
        obligation="NDP Act 2023: data controllers/processors of major "
                   "importance must file a Compliance Audit Return (CAR) "
                   "with the NDPC, latest by 31 March each year, and engage "
                   "a licensed DPCO.",
        template="CAR template (NDPC) + internal audit worksheet",
        checklist=[
            "Confirm whether you process data of 'major importance' "
            "(high volume / sensitive data).",
            "Appoint and register a Data Protection Officer.",
            "Engage a licensed Data Protection Compliance Organisation.",
            "Run a DPIA on high-risk processing.",
            "File the audit return not later than 31 March.",
        ],
        regulators=["NDPA"],
    ),
    "tax filing": Control(
        obligation="FIRS: companies income tax returns due within 6 months "
                   "of financial year-end; VAT monthly; withholding tax "
                   "remittance on qualifying payments.",
        template="FIRS filing calendar + TIN verification checklist",
        checklist=[
            "Confirm your TIN and tax office are current.",
            "Reconcile VAT on monthly supplies.",
            "Remit withholding tax deducted from vendors.",
            "File companies income tax within 6 months of year-end.",
        ],
        regulators=["FIRS"],
    ),
    "fintech licensing": Control(
        obligation="CBN: payments/fintech activity requires the right "
                   "licence category (PSP, MMO, PSB, etc.); AML baseline "
                   "standards apply to all licensed institutions.",
        template="CBN licence-category checklist + AML baseline standards",
        checklist=[
            "Match your activity to a CBN licence category.",
            "Confirm baseline AML automated-solution standards.",
            "Keep transaction monitoring and reporting current.",
            "Track new supervision circulars for your category.",
        ],
        regulators=["CBN"],
    ),
    "securities offering": Control(
        obligation="SEC: public offers and collective investment schemes "
                   "need SEC approval/registration; digital-asset rules "
                   "cover VASPs operating in Nigeria.",
        template="SEC registration checklist",
        checklist=[
            "Confirm whether your raise is a public offer.",
            "Check VASP licensing status for digital-asset activity.",
            "Verify adviser/dealer registration where applicable.",
        ],
        regulators=["SEC"],
    ),
    "annual returns": Control(
        obligation="CAC: every registered company/business name must file "
                   "annual returns each year and keep its records current "
                   "(directors, address, beneficial ownership).",
        template="CAC annual-returns filing checklist",
        checklist=[
            "Confirm your RC/BN number and filing status on the CAC portal.",
            "Update directors, registered address, and share capital if changed.",
            "Declare beneficial owners where required.",
            "File the annual return and keep the receipt.",
        ],
        regulators=["CAC"],
    ),
    "telecom licensing": Control(
        obligation="NCC: telecom and value-added services need the right "
                   "licence class; SIM registration and tariff/data rules "
                   "apply to consumer-facing services.",
        template="NCC licence-class checklist",
        checklist=[
            "Match your service to an NCC licence class.",
            "Confirm SIM-registration compliance for subscriber services.",
            "Check current tariff and data-rollover rules.",
        ],
        regulators=["NCC"],
    ),
    "product standards": Control(
        obligation="SON: regulated products need SONCAP certification "
                   "(imports) or MANCAP (locally manufactured) before sale "
                   "in Nigeria.",
        template="SON certification checklist",
        checklist=[
            "Confirm whether your product category is regulated.",
            "Obtain SONCAP for imports / MANCAP for local manufacture.",
            "Keep test reports and certificates on file for inspection.",
        ],
        regulators=["SON"],
    ),
}


def controls_for(item: RegulatoryItem) -> list[Control]:
    """Map an item's topics to actionable control checklists."""
    out: list[Control] = []
    hay = " ".join(item.topics + [item.title, item.summary]).lower()
    for key, control in OBLIGATION_CONTROLS.items():
        key_hit = any(w in hay for w in key.split())
        reg_hit = item.regulator in control.regulators and any(
            t in hay for t in ("audit", "tax", "licen", "offer", "compliance")
        )
        if key_hit or reg_hit:
            out.append(control)
    return out


def impact_score(item: RegulatoryItem,
                 profile: dict | None = None) -> int:
    """Impact score 0–100 for an item against a business profile.

    Compliance.ai pattern: not just "relevant", but *how much it matters*.
    Urgency is the base; profile-keyword matches raise it; an update to a
    previously seen rule (item.updated) raises it further.
    """
    try:
        score = 55 if item.urgency == "urgent" else 25
        hay = " ".join([item.title, item.summary] + item.topics).lower()
        if profile:
            keywords = [str(k).lower() for k in
                        (profile.get("keywords") or [])
                        + [profile.get("business_type") or ""]
                        + (profile.get("sectors") or [])]
            keywords = [k for k in keywords if k.strip()]
            hits = sum(1 for k in keywords if k in hay)
            score += min(30, hits * 10)
        if item.updated:
            score += 10
        if item.effective_date and item.effective_date > 0:
            import time as _t
            days_out = (item.effective_date - _t.time()) / 86400
            if 0 <= days_out <= 30:
                score += 10  # takes effect soon
        return max(0, min(100, score))
    except Exception:  # noqa: BLE001
        return 0


# ── regulatory calendar ─────────────────────────────────────────────

def upcoming_deadlines(months: int = 6,
                       now: float | None = None) -> list[dict]:
    """Recurring regulatory deadlines in the next ``months``.

    Returns [{month, day, regulator, title, date_ts}]. Never raises.
    """
    try:
        import datetime
        now = now if now is not None else time.time()
        today = datetime.date.fromtimestamp(now)
        out: list[dict] = []
        for month, day, reg, title in REGULATORY_CALENDAR:
            for year in (today.year, today.year + 1):
                try:
                    d = datetime.date(year, month, day)
                except ValueError:
                    continue
                ts = datetime.datetime(year, month, day).timestamp()
                if today <= d <= today + datetime.timedelta(
                        days=int(months * 30.44)):
                    out.append({"month": month, "day": day,
                                "regulator": reg, "title": title,
                                "date_ts": ts})
                    break
        out.sort(key=lambda e: e["date_ts"])
        return out
    except Exception:  # noqa: BLE001
        return []


def sync_calendar_to_portfolio(portfolio: Any,
                               months: int = 6) -> int:
    """Track upcoming regulatory deadlines as portfolio obligations.

    Cross-module wiring: the contract portfolio becomes the single
    obligation surface — contract dates AND regulatory deadlines.
    Returns the number of deadlines tracked. Never raises.
    """
    try:
        if portfolio is None:
            return 0
        import datetime
        count = 0
        existing = {(o.description, int(o.due_at))
                    for o in portfolio.obligations(include_done=True)}
        for entry in upcoming_deadlines(months):
            desc = (f"[{entry['regulator']}] {entry['title']} — "
                    f"{datetime.date.fromtimestamp(entry['date_ts']).isoformat()}")
            key = (desc, int(entry["date_ts"]))
            if key in existing:
                continue
            oid = portfolio.track_obligation("", desc, entry["date_ts"])
            if oid:
                count += 1
                existing.add(key)
        return count
    except Exception:  # noqa: BLE001
        return 0


# ── the watch store ───────────────────────────────────────────────────

@dataclass
class RegWatch:
    """One monitoring watch: regulator + optional topic filter."""
    id: str
    regulator: str
    topics: list[str] = field(default_factory=list)
    created_at: float = 0.0

    def label(self) -> str:
        if self.topics:
            return f"{self.regulator} ({', '.join(self.topics)})"
        return f"{self.regulator} (all updates)"


class RegulatoryWatch:
    """Persistent regulatory watches + dedupe + relevance filtering.

    Live fetching is an injectable seam: ``check(fetcher=...)`` calls your
    fetcher per regulator; with no fetcher the watch simply reports that
    live fetch is not configured (honest, not silent). Never raises.
    """

    def __init__(self, db_path: str = "") -> None:
        path = db_path or _DEFAULT_DB
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS reg_watches (
                       id TEXT PRIMARY KEY, regulator TEXT,
                       topics_json TEXT, created_at REAL)"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS reg_seen (
                       item_id TEXT PRIMARY KEY, watch_id TEXT,
                       seen_at REAL, digest TEXT DEFAULT '',
                       dismissed INTEGER DEFAULT 0)"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS reg_items (
                       item_id TEXT PRIMARY KEY, regulator TEXT, title TEXT,
                       summary TEXT, source_url TEXT, ref_no TEXT,
                       published REAL DEFAULT 0, effective_date REAL DEFAULT 0,
                       topics_json TEXT, urgency TEXT DEFAULT 'normal')"""
            )
            self._migrate()
            self._db.commit()
        except Exception:  # noqa: BLE001 — degraded mode
            _log.warning("regulatory db unavailable; watches in-memory only")
            self._db = sqlite3.connect(":memory:")
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS reg_watches (
                       id TEXT PRIMARY KEY, regulator TEXT,
                       topics_json TEXT, created_at REAL)"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS reg_seen (
                       item_id TEXT PRIMARY KEY, watch_id TEXT,
                       seen_at REAL, digest TEXT DEFAULT '',
                       dismissed INTEGER DEFAULT 0)"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS reg_items (
                       item_id TEXT PRIMARY KEY, regulator TEXT, title TEXT,
                       summary TEXT, source_url TEXT, ref_no TEXT,
                       published REAL DEFAULT 0, effective_date REAL DEFAULT 0,
                       topics_json TEXT, urgency TEXT DEFAULT 'normal')"""
            )
        import json
        self._json = json

    def _migrate(self) -> None:
        """Add sweep columns to older databases. Never raises."""
        try:
            cols = {r[1] for r in self._db.execute(
                "PRAGMA table_info(reg_seen)").fetchall()}
            if "digest" not in cols:
                self._db.execute(
                    "ALTER TABLE reg_seen ADD COLUMN digest TEXT DEFAULT ''")
            if "dismissed" not in cols:
                self._db.execute(
                    "ALTER TABLE reg_seen ADD COLUMN dismissed INTEGER DEFAULT 0")
        except Exception:  # noqa: BLE001
            _log.debug("regulatory migration skipped", exc_info=True)

    # ── watch management ──────────────────────────────────────────

    def watch(self, regulator: str,
              topics: list[str] | None = None) -> Optional[RegWatch]:
        """Add a watch. Returns None (never raises) on a bad regulator."""
        reg = _norm_reg(regulator)
        if reg not in REGULATORS:
            return None
        clean = [t.strip().lower() for t in (topics or []) if t.strip()]
        w = RegWatch(id="reg_" + uuid.uuid4().hex[:8], regulator=reg,
                     topics=clean, created_at=time.time())
        self._db.execute(
            "INSERT INTO reg_watches VALUES (?, ?, ?, ?)",
            (w.id, w.regulator, self._json.dumps(w.topics), w.created_at),
        )
        self._db.commit()
        return w

    def unwatch(self, watch_id: str) -> bool:
        cur = self._db.execute("DELETE FROM reg_watches WHERE id = ?",
                               (watch_id,))
        self._db.commit()
        return cur.rowcount > 0

    def list_watches(self) -> list[RegWatch]:
        out: list[RegWatch] = []
        for row in self._db.execute(
                "SELECT * FROM reg_watches ORDER BY created_at"):
            out.append(RegWatch(
                id=row["id"], regulator=row["regulator"],
                topics=self._json.loads(row["topics_json"] or "[]"),
                created_at=row["created_at"]))
        return out

    # ── relevance filter ──────────────────────────────────────────

    @staticmethod
    def relevant(item: RegulatoryItem, watch: RegWatch,
                 profile: dict | None = None) -> bool:
        """Relevance filter: watch topics AND/OR the business profile must
        match the item. A watch with no topics matches everything from
        that regulator; a profile match rescues cross-regulator items."""
        hay = " ".join([item.title, item.summary] + item.topics).lower()
        if watch.topics:
            topic_hit = any(t in hay for t in watch.topics)
        else:
            topic_hit = True
        if topic_hit:
            return True
        if profile:
            keywords = [str(k).lower() for k in
                        (profile.get("keywords") or [])
                        + [profile.get("business_type") or ""]
                        + (profile.get("sectors") or [])]
            keywords = [k for k in keywords if k.strip()]
            return any(k in hay for k in keywords)
        return False

    # ── check ─────────────────────────────────────────────────────

    def check(self, *,
              fetcher: Optional[Callable[[str], list[dict]]] = None,
              profile: dict | None = None,
              now: float | None = None) -> list[RegulatoryItem]:
        """Run one scan across all watches. Returns new, relevant items.

        ``fetcher(regulator_code) -> [dict]`` is the live-source seam
        (RSS/web). Without it the check honestly reports no sources.
        """
        now = now if now is not None else time.time()
        watches = self.list_watches()
        if not watches:
            return []
        if fetcher is None:
            _log.info("regulatory check: no fetcher configured — skipping")
            return []
        new_items: list[RegulatoryItem] = []
        for watch in watches:
            try:
                raw = fetcher(watch.regulator) or []
            except Exception:  # noqa: BLE001 — one dead source != dead scan
                _log.warning("regulatory fetch failed for %s", watch.regulator)
                continue
            for d in raw:
                item = item_from_dict(d)
                if item is None:
                    continue
                if item.regulator != watch.regulator:
                    continue
                if not self.relevant(item, watch, profile):
                    continue
                row = self._db.execute(
                    "SELECT digest, dismissed FROM reg_seen WHERE item_id = ?",
                    (item.id,)).fetchone()
                if row is not None and (row["dismissed"] or 0):
                    continue  # owner dismissed this item — never alert again
                digest = item.digest()
                if row is None:
                    self._db.execute(
                        "INSERT OR IGNORE INTO reg_seen "
                        "(item_id, watch_id, seen_at, digest, dismissed) "
                        "VALUES (?, ?, ?, ?, 0)",
                        (item.id, watch.id, now, digest))
                    new_items.append(item)
                elif (row["digest"] or "") != digest:
                    # TR Regulatory Intelligence pattern: the item changed
                    # since we last saw it — alert on the update.
                    self._db.execute(
                        "UPDATE reg_seen SET digest = ?, seen_at = ? "
                        "WHERE item_id = ?",
                        (digest, now, item.id))
                    item.updated = True
                    new_items.append(item)
                # unchanged items: no alert
        self._db.commit()
        return sorted(new_items,
                      key=lambda i: (i.urgency != "urgent", -i.published))

    def dismiss(self, item_id: str) -> bool:
        """Dismiss an item — never alert on it again (relevance feedback).

        The RegTech pattern: owner feedback tunes future relevance.
        """
        try:
            cur = self._db.execute(
                "UPDATE reg_seen SET dismissed = 1 WHERE item_id = ?",
                (item_id,))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    def upcoming_effective(self, days: float = 90,
                           now: float | None = None) -> list[RegulatoryItem]:
        """Items with a known effective date in the next ``days``.

        Pending-change visibility (TR Regulatory Intelligence pattern):
        "this rule takes effect on <date>". Populated by ``note_item()``.
        Never raises.
        """
        try:
            now = now if now is not None else time.time()
            rows = self._db.execute(
                """SELECT * FROM reg_items
                   WHERE effective_date > ? AND effective_date <= ?
                   ORDER BY effective_date""",
                (now, now + days * 86400)).fetchall()
            out = []
            for r in rows:
                try:
                    out.append(RegulatoryItem(
                        id=r["item_id"], regulator=r["regulator"],
                        title=r["title"], summary=r["summary"] or "",
                        source_url=r["source_url"] or "",
                        ref_no=r["ref_no"] or "",
                        published=r["published"] or 0.0,
                        topics=self._json.loads(r["topics_json"] or "[]"),
                        urgency=r["urgency"] or None,
                        effective_date=r["effective_date"] or 0.0))
                except Exception:  # noqa: BLE001 — one bad row != dead query
                    continue
            return out
        except Exception:  # noqa: BLE001
            return []

    def note_item(self, item: RegulatoryItem) -> bool:
        """Record an item directly (e.g. from a manual scan). Returns True
        when the item is new or changed (i.e. alert-worthy). Never raises.
        """
        try:
            if item.regulator not in REGULATORS or not item.title:
                return False
            row = self._db.execute(
                "SELECT digest, dismissed FROM reg_seen WHERE item_id = ?",
                (item.id,)).fetchone()
            if row is not None and (row["dismissed"] or 0):
                return False
            digest = item.digest()
            self._db.execute(
                """INSERT OR REPLACE INTO reg_items
                       (item_id, regulator, title, summary, source_url,
                        ref_no, published, effective_date, topics_json,
                        urgency)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (item.id, item.regulator, item.title[:500],
                 item.summary[:2000], item.source_url[:500],
                 item.ref_no[:120], item.published,
                 item.effective_date,
                 self._json.dumps(item.topics[:20]),
                 item.urgency or "normal"))
            if row is None:
                self._db.execute(
                    "INSERT OR IGNORE INTO reg_seen "
                    "(item_id, watch_id, seen_at, digest, dismissed) "
                    "VALUES (?, ?, ?, ?, 0)",
                    (item.id, "", time.time(), digest))
                self._db.commit()
                return True
            if (row["digest"] or "") != digest:
                self._db.execute(
                    "UPDATE reg_seen SET digest = ?, seen_at = ? "
                    "WHERE item_id = ?", (digest, time.time(), item.id))
                self._db.commit()
                item.updated = True
                return True
            self._db.commit()
            return False
        except Exception:  # noqa: BLE001
            return False


# ── alerts ──────────────────────────────────────────────────────────

def alert_text(item: RegulatoryItem, *,
               profile: dict | None = None,
               style: str = "full") -> str:
    """Plain-language regulatory alert. Information, never advice.

    "CBN changed X — here's what it means for you, here's what to do."
    The "what to do" is: read the official source, check the control
    checklist, verify consequential changes with a professional.

    ``style`` is "full" (default) or "brief" (one screen).
    """
    style = (style or "full").lower()
    if style not in ALERT_STYLES:
        style = "full"
    reg = REGULATORS.get(item.regulator)
    reg_name = reg.name if reg else item.regulator
    badge = "🚨 URGENT" if item.urgency == "urgent" else "📜"
    updated_tag = " (UPDATED — this rule changed since the last alert)" \
        if item.updated else ""
    score = impact_score(item, profile)
    lines = [f"{badge} {item.regulator}: {item.title}{updated_tag}"]
    if style == "brief":
        lines.append(f"Impact: {score}/100")
        if item.ref_no:
            lines.append(f"Ref: {item.ref_no}")
        lines.append(f"Source: {item.source_url or (reg.site if reg else '')}")
        lines.append(DISCLAIMER)
        return "\n".join(lines)
    if item.ref_no:
        lines.append(f"Ref: {item.ref_no}")
    lines.append(f"Impact score: {score}/100")
    lines.append("")
    if item.effective_date:
        import datetime
        eff = datetime.date.fromtimestamp(
            item.effective_date).isoformat()
        lines.append(f"Takes effect: {eff}")
        lines.append("")
    if profile and profile.get("business_type"):
        lines.append(
            f"Why this may matter to you: you operate as "
            f"{profile['business_type']} — this update touches "
            f"{', '.join(item.topics) or 'general compliance'}.")
        lines.append("")
    lines.append(
        "What changed (per the official source): " +
        (item.summary or item.title))
    lines.append("")
    controls = controls_for(item)
    if controls:
        c = controls[0]
        lines.append(f"Related obligation: {c.obligation}")
        lines.append(f"Template: {c.template}")
        lines.append("Checklist:")
        for step in c.checklist:
            lines.append(f"  • {step}")
        lines.append("")
    lines.append(f"Official source: {item.source_url or (reg.site if reg else '')}")
    lines.append("Verify consequential changes at the source before acting.")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


# ── scheduler seam ──────────────────────────────────────────────────

def check_all(store: RegulatoryWatch | None = None,
              *,
              fetcher: Optional[Callable[[str], list[dict]]] = None,
              profile: dict | None = None,
              sender: Optional[Callable[[str], Any]] = None) -> list[str]:
    """Host entry point: run one regulatory scan, send alerts, return them.

    Urgent items go out immediately; normal items are bundled. Never
    raises."""
    try:
        s = store or RegulatoryWatch()
        items = s.check(fetcher=fetcher, profile=profile)
        alerts = [alert_text(i, profile=profile) for i in items]
        if sender is not None and alerts:
            for a in alerts:
                try:
                    sender(a)
                except Exception:  # noqa: BLE001 — one bad send != lost batch
                    _log.warning("regulatory alert send failed")
        return alerts
    except Exception:  # noqa: BLE001
        _log.debug("regulatory check_all failed", exc_info=True)
        return []


def ensure_schedule(scheduler: Any) -> bool:
    """Register the daily regulatory cron. Idempotent-ish."""
    try:
        import asyncio

        async def _ensure() -> bool:
            jobs: list[Any] = []
            try:
                jobs = scheduler.list_jobs()  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            for j in jobs or []:
                params = getattr(j, "parameters", {}) or {}
                if getattr(j, "action", "") == REGCHECK_ACTION:
                    return True
            await scheduler.schedule_cron(
                task_id="regulatory-daily",
                cron_expr=REGCHECK_CRON,
                action=REGCHECK_ACTION,
                parameters={},
            )
            return True

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            loop.create_task(_ensure())
            return True
        return False
    except Exception:  # noqa: BLE001
        return False


# ── chat ────────────────────────────────────────────────────────────

def _usage() -> str:
    regs = ", ".join(f"{r.code} ({r.name})" for r in REGULATORS.values())
    return (
        "/regwatch — regulatory change monitoring: CBN, SEC, NDPA (data "
        "protection), FIRS. Get a plain-language alert when rules affecting "
        "your business change.\n"
        f"Regulators: {regs}\n"
        "Usage:\n"
        "  /regwatch add CBN [fintech, AML] — watch a regulator (+ topics)\n"
        "  /regwatch list — your watches\n"
        "  /regwatch remove <id> — stop a watch\n"
        "  /regwatch check — run a scan now\n"
        "  /regwatch calendar [months] — upcoming recurring deadlines\n"
        "  /regwatch effective [days] — rules taking effect soon\n"
        "  /regwatch dismiss <item id> — never alert on an item again\n"
        "  /regwatch regulators — official source list\n"
        "Owner only. Legal information, never legal advice; Devon is not a "
        "lawyer.\n" + DISCLAIMER
    )


def control_regwatch(tail: str, context: Any = None, chat: Any = None,
                     sender_id: str = "", sender: str = "") -> str:
    """/regwatch — regulatory watches. Owner-only; never raises."""
    try:
        rest = (tail or "").strip()
        store = getattr(context, "regulatory_store", None) \
            if context is not None else None
        s = store if isinstance(store, RegulatoryWatch) else RegulatoryWatch()
        parts = rest.split()
        if not parts or parts[0].lower() in ("help", "?"):
            return _usage()

        cmd = parts[0].lower()

        if cmd == "regulators":
            return ("Official monitored sources:\n" + "\n".join(
                f"• {r.code} — {r.name}\n  {r.site}\n  {r.blurb}"
                for r in REGULATORS.values()
            ) + "\n\n" + DISCLAIMER)

        if cmd == "add":
            if len(parts) < 2:
                return "Usage: /regwatch add CBN [topics…]\n" + _usage()
            reg = _norm_reg(parts[1])
            if reg not in REGULATORS:
                return (f"Unknown regulator '{parts[1]}'. Try: "
                        f"{', '.join(REGULATORS)}.\n" + DISCLAIMER)
            topics = parts[2:] if len(parts) > 2 else []
            w = s.watch(reg, topics)
            if w is None:
                return "Couldn't create that watch.\n" + DISCLAIMER
            return (f"👀 Watching {w.label()}. You'll get a plain-language "
                    f"alert when relevant {w.regulator} updates appear "
                    f"(id: {w.id}).\n" + DISCLAIMER)

        if cmd == "list":
            watches = s.list_watches()
            if not watches:
                return ("No regulatory watches yet. "
                        "/regwatch add CBN fintech — to start.\n" + DISCLAIMER)
            return ("Your regulatory watches:\n" + "\n".join(
                f"• {w.label()}  (id: {w.id})" for w in watches
            ) + "\n\n" + DISCLAIMER)

        if cmd in ("remove", "rm", "delete"):
            if len(parts) < 2:
                return "Usage: /regwatch remove <id>\n" + DISCLAIMER
            ok = s.unwatch(parts[1])
            return (f"Watch {parts[1]} removed." if ok
                    else f"No watch with id {parts[1]}.") + "\n" + DISCLAIMER

        if cmd == "check":
            profile = getattr(context, "business_profile", None) \
                if context is not None else None
            fetcher = getattr(context, "regulatory_fetcher", None) \
                if context is not None else None
            items = s.check(fetcher=fetcher, profile=profile)
            if fetcher is None:
                return ("Live regulatory fetching isn't configured on this "
                        "device yet — watches are saved, and scans run once "
                        "a source fetcher is attached.\n" + DISCLAIMER)
            if not items:
                return ("Scan complete — nothing new from your watched "
                        "regulators.\n" + DISCLAIMER)
            out = [f"📡 {len(items)} new regulatory update(s):"]
            for i in items:
                out.append(alert_text(i, profile=profile))
            return "\n\n".join(out)

        if cmd == "calendar":
            months = 6
            if len(parts) > 1:
                try:
                    months = max(1, min(24, int(parts[1])))
                except ValueError:
                    pass
            entries = upcoming_deadlines(months)
            if not entries:
                return ("No recurring deadlines in the next "
                        f"{months} months.\n" + DISCLAIMER)
            lines = [f"📅 Regulatory calendar (next {months} months):"]
            for e in entries:
                import datetime
                d = datetime.date.fromtimestamp(e["date_ts"]).isoformat()
                lines.append(f"  • {d} — [{e['regulator']}] {e['title']}")
            return "\n".join(lines) + "\n\n" + DISCLAIMER

        if cmd == "effective":
            days = 90.0
            if len(parts) > 1:
                try:
                    days = max(1.0, float(parts[1]))
                except ValueError:
                    pass
            items = s.upcoming_effective(days)
            if not items:
                return (f"No tracked rules taking effect in the next "
                        f"{int(days)} days.\n" + DISCLAIMER)
            lines = [f"⏳ Rules taking effect (next {int(days)} days):"]
            for i in items[:10]:
                import datetime
                d = datetime.date.fromtimestamp(
                    i.effective_date).isoformat()
                lines.append(f"  • {d} — {i.regulator}: {i.title}")
            return "\n".join(lines) + "\n\n" + DISCLAIMER

        if cmd == "dismiss":
            if len(parts) < 2:
                return "Usage: /regwatch dismiss <item id>\n" + DISCLAIMER
            ok = s.dismiss(parts[1])
            return (f"Item {parts[1]} dismissed — it won't alert again."
                    if ok else f"No tracked item {parts[1]}.") + "\n" + DISCLAIMER

        return "Unknown subcommand.\n" + _usage()
    except Exception as e:  # noqa: BLE001 — never raise from chat
        return f"Regwatch hit an error ({e}). {DISCLAIMER}"
