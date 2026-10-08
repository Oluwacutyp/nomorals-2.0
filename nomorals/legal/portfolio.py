"""Contract portfolio — all of the owner's contracts as one obligation
tracker.

Build-map #79 (repository intelligence, Ironclad/LinkSquares pattern).
#76 reviews single contracts; this module ingests those reviews and
tracks obligations, key dates, renewals and SLA terms across all of
them: "your tenancy renews in 60 days, the rent-increase clause allows
no more than the agreed notice period."

Positioning (factual, not moralizing): legal *information*, never legal
*advice*. The portfolio states what the paperwork says and when things
are due; it never tells the owner what to do, never performs like a
lawyer, and carries :data:`DISCLAIMER` on chat outputs. Every proactive
alert is a neutral date reminder, not a recommendation.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .contracts import (
    DISCLAIMER,
    Review,
    detect_contract_type,
    review_contract,
)
from ..core.logging_setup import get_logger

__all__ = [
    "DISCLAIMER",
    "Contract",
    "Obligation",
    "TimelineEvent",
    "ContractPortfolio",
    "check_all",
    "ensure_schedule",
    "control_contracts",
    "PORTFOLIO_CHECK_ACTION",
    "PORTFOLIO_CRON",
]

_log = get_logger(__name__)

#: Scheduler action + cron for the proactive renewal/deadline check.
PORTFOLIO_CHECK_ACTION = "portfolio_check"
PORTFOLIO_CRON = "0 8 * * *"  # 08:00 daily

_DEFAULT_DB = os.path.join(
    os.path.expanduser("~"), ".nomorals", "legal", "portfolio.db"
)

# ── date extraction (never raises, never fabricates) ─────────────────


_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_DATE_PATTERNS = (
    # 2026-12-01 / 2026/12/01
    re.compile(r"(?P<y>\d{4})[-/](?P<m>\d{1,2})[-/](?P<d>\d{1,2})"),
    # 01/12/2026 / 1-12-26  (day-first, Nigerian convention)
    re.compile(r"(?P<d>\d{1,2})[-/](?P<m>\d{1,2})[-/](?P<y>\d{2,4})"),
    # 1st December 2026 / December 1, 2026
    re.compile(
        r"(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+(?P<mon>[a-z]+),?\s+(?P<y>\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?P<mon>[a-z]+)\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y>\d{4})",
        re.IGNORECASE,
    ),
)

_DATE_KEYWORDS = (
    ("renewal", re.compile(r"renew|expir|end of (the )?term|lapse", re.IGNORECASE)),
    ("notice", re.compile(r"notice|notify|inform.{0,10}(in )?writing", re.IGNORECASE)),
    ("payment", re.compile(r"pay|invoice|rent due|fee due", re.IGNORECASE)),
    ("commencement", re.compile(r"commenc|start date|effective date|begin", re.IGNORECASE)),
    ("termination", re.compile(r"terminat|cancel|end.{0,10}agreement", re.IGNORECASE)),
)


def _to_ts(year: int, month: int, day: int) -> float | None:
    try:
        if year < 100:
            year += 2000 if year < 70 else 1900
        import calendar
        if not (1 <= month <= 12):
            return None
        if not (1 <= day <= calendar.monthrange(year, month)[1]):
            return None
        import datetime
        return datetime.datetime(year, month, day).timestamp()
    except Exception:  # noqa: BLE001
        return None


def _parse_dates(text: str) -> list[tuple[float, str]]:
    """Extract (timestamp, surrounding context) pairs from free text.

    Never raises; returns [] on garbage input.
    """
    try:
        found: list[tuple[float, str]] = []
        seen: set[float] = set()
        for pat in _DATE_PATTERNS:
            for m in pat.finditer(text or ""):
                gd = m.groupdict()
                try:
                    y = int(gd["y"])
                    mon = gd.get("mon")
                    if mon is not None:
                        month = _MONTHS.get(mon.lower(), 0)
                    else:
                        month = int(gd["m"])
                    day = int(gd["d"])
                except (ValueError, TypeError, KeyError):
                    continue
                ts = _to_ts(y, month, day)
                if ts is None or ts in seen:
                    continue
                seen.add(ts)
                start = max(0, m.start() - 60)
                end = min(len(text), m.end() + 60)
                found.append((ts, text[start:end]))
        return found
    except Exception:  # noqa: BLE001
        return []


def _classify_date(context: str) -> str:
    """Best-effort kind label for a date from its surrounding context."""
    for kind, pat in _DATE_KEYWORDS:
        if pat.search(context or ""):
            return kind
    return "deadline"


_DURATION_RE = re.compile(
    r"(?P<n>\d+(?:\.\d+)?)\s*(?P<u>day|week|month|year)s?", re.IGNORECASE
)


def _derive_expiry(doc_text: str, dates: list[tuple[float, str]]) -> float | None:
    """Expiry = commencement date + stated term length, when both found."""
    comm = next((ts for ts, ctx in dates if _classify_date(ctx) == "commencement"), None)
    if comm is None:
        return None
    term: float | None = None
    for m in _DURATION_RE.finditer(doc_text or ""):
        ctx = doc_text[max(0, m.start() - 60): m.end() + 20]
        if re.search(r"term|tenancy|duration|period of|for a period", ctx, re.IGNORECASE):
            try:
                n = float(m.group("n"))
            except ValueError:
                continue
            u = m.group("u").lower()
            months = {"day": 1 / 30, "week": 1 / 4.345, "month": 1, "year": 12}[u]
            term = n * months
            break
    if term is None:
        return None
    return comm + term * 30 * 86400


# ── models ──────────────────────────────────────────────────────────


@dataclass
class Obligation:
    id: str
    contract_id: str
    kind: str  # renewal | expiry | notice | payment | deadline | manual | sla
    description: str
    due_at: float
    done: bool = False
    done_at: float = 0.0
    source: str = ""  # "extracted" | "manual" | "sla"

    def overdue(self, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return not self.done and self.due_at < now

    def due_within(self, days: float, now: float | None = None) -> bool:
        now = now if now is not None else time.time()
        return not self.done and self.due_at <= now + days * 86400


@dataclass
class Contract:
    id: str
    name: str
    contract_type: str
    parties: str = ""
    grade: str = ""
    score: int = 0
    findings_count: int = 0
    created_at: float = 0.0

    def headline(self) -> str:
        grade = f" ({self.grade})" if self.grade else ""
        parties = f" — {self.parties}" if self.parties else ""
        return f"📄 {self.name}{grade}{parties}"


@dataclass
class TimelineEvent:
    due_at: float
    contract_name: str
    kind: str
    description: str
    days_left: int = 0
    obligation_id: str = ""

    def headline(self) -> str:
        if self.days_left < 0:
            when = f"{-self.days_left}d overdue"
        elif self.days_left == 0:
            when = "due today"
        else:
            when = f"in {self.days_left}d"
        return f"• {self.contract_name}: {self.description} — {when}"


@dataclass
class SLAStatus:
    contract_id: str
    contract_name: str
    metric: str
    target: str
    breached: bool
    note: str = ""


# ── portfolio ───────────────────────────────────────────────────────


def _now() -> float:
    return time.time()


class ContractPortfolio:
    """All of the owner's contracts, with obligations tracked.

    SQLite-backed, never raises on chat paths. Legal information only —
    the portfolio states dates and paperwork facts, never advice.
    """

    def __init__(self, db_path: str = "") -> None:
        try:
            path = db_path or _DEFAULT_DB
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS contracts (
                       id TEXT PRIMARY KEY, name TEXT, contract_type TEXT,
                       parties TEXT, grade TEXT, score INTEGER,
                       findings_count INTEGER, created_at REAL)"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS obligations (
                       id TEXT PRIMARY KEY, contract_id TEXT, kind TEXT,
                       description TEXT, due_at REAL, done INTEGER DEFAULT 0,
                       done_at REAL DEFAULT 0, source TEXT DEFAULT 'extracted')"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS sla_terms (
                       contract_id TEXT, metric TEXT, target TEXT,
                       breached INTEGER DEFAULT 0, note TEXT DEFAULT '',
                       PRIMARY KEY (contract_id, metric))"""
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_obl_due ON obligations(due_at)"
            )
            self._db.commit()
            self._ok = True
        except Exception as exc:  # noqa: BLE001 — fail-closed, never raise
            _log.warning("ContractPortfolio unavailable: %s", exc)
            self._db = None  # type: ignore[assignment]
            self._ok = False

    # — ingestion —

    def add_contract(
        self,
        review: Review,
        *,
        doc_text: str = "",
        name: str = "",
        parties: str = "",
    ) -> Contract | None:
        """Ingest a #76 review (+ optional raw text for date extraction).

        Extracts obligations/key dates from the document text; stores the
        grade + findings count from the review. Returns None on failure
        (never raises).
        """
        if not self._ok:
            return None
        try:
            cid = "ctr_" + uuid.uuid4().hex[:10]
            name = (name or "").strip() or f"{review.contract_type.title()} contract"
            contract = Contract(
                id=cid, name=name[:120], contract_type=review.contract_type,
                parties=(parties or "").strip()[:200],
                grade=review.grade, score=review.score,
                findings_count=len(review.findings), created_at=_now(),
            )
            self._db.execute(
                """INSERT INTO contracts (id, name, contract_type, parties,
                       grade, score, findings_count, created_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (contract.id, contract.name, contract.contract_type,
                 contract.parties, contract.grade, contract.score,
                 contract.findings_count, contract.created_at),
            )
            if doc_text:
                self._extract_obligations(cid, doc_text, review.contract_type)
            self._db.commit()
            return contract
        except Exception:  # noqa: BLE001
            _log.debug("add_contract failed", exc_info=True)
            return None

    def add_raw(self, doc_text: str, *, name: str = "",
                contract_type: str = "auto",
                parties: str = "") -> tuple[Contract, Review] | None:
        """Convenience: review the text with #76, then ingest it."""
        try:
            review = review_contract(doc_text, contract_type)
            ctype = review.contract_type
        except Exception:  # noqa: BLE001
            return None
        contract = self.add_contract(review, doc_text=doc_text,
                                     name=name or f"{ctype.title()} contract",
                                     parties=parties)
        if contract is None:
            return None
        return contract, review

    def _extract_obligations(self, contract_id: str, doc_text: str,
                             contract_type: str) -> None:
        """Date → obligation extraction. Best effort, never raises."""
        try:
            dates = _parse_dates(doc_text)
            now = _now()
            for ts, ctx in dates:
                if ts < now - 365 * 86400:  # ignore ancient dates
                    continue
                kind = _classify_date(ctx)
                desc = self._obligation_desc(kind, ctx)
                self._insert_obligation(
                    contract_id, kind, desc, ts, source="extracted")
            expiry = _derive_expiry(doc_text, dates)
            if expiry and expiry > now:
                self._insert_obligation(
                    contract_id, "expiry",
                    "Contract term ends (derived from commencement + term)",
                    expiry, source="extracted")
        except Exception:  # noqa: BLE001
            _log.debug("obligation extraction failed", exc_info=True)

    @staticmethod
    def _obligation_desc(kind: str, ctx: str) -> str:
        snippet = re.sub(r"\s+", " ", (ctx or "").strip())[:120]
        labels = {
            "renewal": "Renewal/expiry-related date in the contract",
            "notice": "Notice deadline mentioned",
            "payment": "Payment-related date mentioned",
            "commencement": "Contract commencement date",
            "termination": "Termination-related date mentioned",
            "deadline": "Date mentioned in the contract",
        }
        base = labels.get(kind, "Date mentioned in the contract")
        return f"{base}: {snippet}" if snippet else base

    def _insert_obligation(self, contract_id: str, kind: str,
                           description: str, due_at: float,
                           source: str = "extracted") -> str:
        oid = "obl_" + uuid.uuid4().hex[:10]
        self._db.execute(
            """INSERT INTO obligations (id, contract_id, kind, description,
                   due_at, done, done_at, source)
               VALUES (?,?,?,?,?,0,0,?)""",
            (oid, contract_id, kind, description[:400], due_at, source),
        )
        return oid

    # — queries —

    def list_contracts(self) -> list[Contract]:
        if not self._ok:
            return []
        try:
            rows = self._db.execute(
                "SELECT * FROM contracts ORDER BY created_at DESC").fetchall()
            return [Contract(id=r["id"], name=r["name"],
                             contract_type=r["contract_type"],
                             parties=r["parties"] or "",
                             grade=r["grade"] or "", score=r["score"] or 0,
                             findings_count=r["findings_count"] or 0,
                             created_at=r["created_at"] or 0.0)
                    for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def get_contract(self, contract_id: str) -> Contract | None:
        if not self._ok:
            return None
        try:
            r = self._db.execute(
                "SELECT * FROM contracts WHERE id = ?", (contract_id,)).fetchone()
            if r is None:
                return None
            return Contract(id=r["id"], name=r["name"],
                            contract_type=r["contract_type"],
                            parties=r["parties"] or "",
                            grade=r["grade"] or "", score=r["score"] or 0,
                            findings_count=r["findings_count"] or 0,
                            created_at=r["created_at"] or 0.0)
        except Exception:  # noqa: BLE001
            return None

    def obligations(self, contract_id: str = "",
                    *, include_done: bool = False) -> list[Obligation]:
        if not self._ok:
            return []
        try:
            if contract_id:
                rows = self._db.execute(
                    "SELECT * FROM obligations WHERE contract_id = ? "
                    "ORDER BY due_at", (contract_id,)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM obligations ORDER BY due_at").fetchall()
            out = []
            for r in rows:
                if r["done"] and not include_done:
                    continue
                out.append(Obligation(
                    id=r["id"], contract_id=r["contract_id"], kind=r["kind"],
                    description=r["description"] or "",
                    due_at=r["due_at"] or 0.0, done=bool(r["done"]),
                    done_at=r["done_at"] or 0.0, source=r["source"] or ""))
            return out
        except Exception:  # noqa: BLE001
            return []

    def timeline(self, *, include_done: bool = False) -> list[TimelineEvent]:
        """All upcoming obligations/dates across all contracts, sorted."""
        if not self._ok:
            return []
        try:
            now = _now()
            names = {c.id: c.name for c in self.list_contracts()}
            names.setdefault("", "Personal")
            events = []
            for o in self.obligations(include_done=include_done):
                days_left = int((o.due_at - now) // 86400)
                events.append(TimelineEvent(
                    due_at=o.due_at,
                    contract_name=names.get(o.contract_id, o.contract_id),
                    kind=o.kind, description=o.description,
                    days_left=days_left, obligation_id=o.id))
            events.sort(key=lambda e: e.due_at)
            return events
        except Exception:  # noqa: BLE001
            return []

    def needs_attention(self, days: float = 60) -> list[tuple[Contract, list[Obligation]]]:
        """Contracts with obligations due within ``days`` (or overdue).

        Contract-less obligations (personal deadlines tracked without a
        contract) surface under a synthetic "Personal" contract.
        """
        if not self._ok:
            return []
        try:
            now = _now()
            cutoff = now + days * 86400
            contracts = {c.id: c for c in self.list_contracts()}
            personal = Contract(id="", name="Personal", contract_type="")
            contracts.setdefault("", personal)
            hits: dict[str, list[Obligation]] = {}
            for o in self.obligations():
                if o.due_at <= cutoff and o.contract_id in contracts:
                    hits.setdefault(o.contract_id, []).append(o)
            out = []
            for cid, obls in hits.items():
                obls.sort(key=lambda o: o.due_at)
                out.append((contracts[cid], obls))
            out.sort(key=lambda pair: min(o.due_at for o in pair[1]))
            return out
        except Exception:  # noqa: BLE001
            return []

    # — manual obligations —

    def track_obligation(self, contract_id: str, description: str,
                         due_at: float) -> str | None:
        """Track a manual obligation ('quarterly tax filing due')."""
        if not self._ok:
            return None
        try:
            if self.get_contract(contract_id) is None:
                # allow contract-less obligations (personal deadlines)
                contract_id = ""
            oid = self._insert_obligation(
                contract_id, "manual", (description or "").strip()[:400],
                float(due_at), source="manual")
            self._db.commit()
            return oid
        except Exception:  # noqa: BLE001
            return None

    def mark_done(self, obligation_id: str) -> bool:
        if not self._ok:
            return False
        try:
            cur = self._db.execute(
                "UPDATE obligations SET done = 1, done_at = ? WHERE id = ?",
                (_now(), obligation_id))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    # — SLA monitoring —

    def track_sla(self, contract_id: str, metric: str, target: str) -> bool:
        """Record an SLA term for a service agreement (e.g. uptime 99.9%)."""
        if not self._ok:
            return False
        try:
            self._db.execute(
                """INSERT OR REPLACE INTO sla_terms
                       (contract_id, metric, target, breached, note)
                   VALUES (?,?,?,0,'')""",
                (contract_id, (metric or "").strip()[:120],
                 (target or "").strip()[:120]))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def report_sla_breach(self, contract_id: str, metric: str,
                          note: str = "") -> bool:
        """Mark an SLA term as breached (information recorded, not advice)."""
        if not self._ok:
            return False
        try:
            cur = self._db.execute(
                "UPDATE sla_terms SET breached = 1, note = ? "
                "WHERE contract_id = ? AND metric = ?",
                ((note or "").strip()[:400], contract_id,
                 (metric or "").strip()[:120]))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    def sla_status(self, contract_id: str = "") -> list[SLAStatus]:
        if not self._ok:
            return []
        try:
            if contract_id:
                rows = self._db.execute(
                    "SELECT * FROM sla_terms WHERE contract_id = ?",
                    (contract_id,)).fetchall()
            else:
                rows = self._db.execute("SELECT * FROM sla_terms").fetchall()
            names = {c.id: c.name for c in self.list_contracts()}
            return [SLAStatus(
                contract_id=r["contract_id"],
                contract_name=names.get(r["contract_id"], r["contract_id"]),
                metric=r["metric"], target=r["target"],
                breached=bool(r["breached"]), note=r["note"] or "")
                for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def sla_breaches(self) -> list[SLAStatus]:
        return [s for s in self.sla_status() if s.breached]

    # — summaries —

    def summary(self) -> str:
        contracts = self.list_contracts()
        if not contracts:
            return ("📁 Contract portfolio is empty.\n"
                    "Add one with: /contracts add <name> | <paste the contract text>\n\n"
                    + DISCLAIMER)
        lines = [f"📁 Contract portfolio — {len(contracts)} contract(s)"]
        attention = self.needs_attention(60)
        flagged = {c.id for c, _ in attention}
        for c in contracts:
            mark = " ⚠️" if c.id in flagged else ""
            lines.append(c.headline() + mark)
        if attention:
            lines.append("")
            lines.append("Needs attention (next 60 days):")
            for c, obls in attention:
                for o in obls[:2]:
                    lines.append(f"  ⚠️ {c.name}: {o.description[:80]}")
        else:
            lines.append("")
            lines.append("Nothing due in the next 60 days.")
        lines.append("")
        lines.append(DISCLAIMER)
        return "\n".join(lines)

    def attention_text(self, days: float = 60) -> str:
        attention = self.needs_attention(days)
        if not attention:
            return (f"✅ Nothing needs attention in the next {int(days)} days.\n\n"
                    + DISCLAIMER)
        lines = [f"⚠️ Needs attention (next {int(days)} days):"]
        for c, obls in attention:
            lines.append("")
            lines.append(c.headline())
            for o in obls:
                if o.overdue():
                    when = "OVERDUE"
                else:
                    d = max(0, int((o.due_at - _now()) // 86400))
                    when = "due today" if d == 0 else f"in {d}d"
                lines.append(f"  • {o.description[:90]} — {when}")
        lines.append("")
        lines.append(DISCLAIMER)
        return "\n".join(lines)

    # — proactive check (scheduler host entry point) —

    def due_alerts(self, days: float = 14) -> list[str]:
        """Neutral date reminders for the scheduler. Information only."""
        alerts = []
        for c, obls in self.needs_attention(days):
            for o in obls:
                if o.overdue():
                    alerts.append(
                        f"⏰ Overdue: {c.name} — {o.description[:100]}")
                else:
                    d = max(0, int((o.due_at - _now()) // 86400))
                    when = "today" if d == 0 else f"in {d} day(s)"
                    alerts.append(
                        f"⏰ {c.name}: {o.description[:100]} — due {when}")
        for s in self.sla_breaches():
            note = f" ({s.note})" if s.note else ""
            alerts.append(
                f"⏰ SLA breach recorded: {s.contract_name} — "
                f"{s.metric} vs target {s.target}{note}")
        return alerts


# ── scheduler seam ──────────────────────────────────────────────────


def check_all(portfolio: ContractPortfolio | None = None,
              days: float = 14) -> list[str]:
    """Host entry point: run the portfolio check once, return alerts."""
    try:
        p = portfolio or ContractPortfolio()
        return p.due_alerts(days)
    except Exception:  # noqa: BLE001
        _log.debug("portfolio check_all failed", exc_info=True)
        return []


def ensure_schedule(scheduler: Any) -> bool:
    """Register the daily portfolio cron. Idempotent-ish."""
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
                if getattr(j, "action", "") == PORTFOLIO_CHECK_ACTION:
                    return True
            await scheduler.schedule_cron(
                task_id="contracts-portfolio-daily",
                cron_expr=PORTFOLIO_CRON,
                action=PORTFOLIO_CHECK_ACTION,
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
        return asyncio.run(_ensure())
    except Exception:  # noqa: BLE001
        _log.debug("ensure_schedule failed", exc_info=True)
        return False


# ── chat control ────────────────────────────────────────────────────


def _usage() -> str:
    return (
        "/contracts — portfolio summary (all contracts, attention flags)\n"
        "/contracts attention [days] — what needs attention soon\n"
        "/contracts add <name> | <paste the contract text> — review + ingest\n"
        "/contracts obligations <contract id> — open obligations for one contract\n"
        "/contracts track <contract id> | <description> | <YYYY-MM-DD> — manual obligation\n"
        "/contracts done <obligation id> — mark an obligation done\n"
        "/contracts sla <contract id> <metric> | <target> — track an SLA term\n"
        "/contracts breach <contract id> <metric> | [note] — record an SLA breach\n"
        "/contracts timeline — all upcoming dates across contracts"
    )


def _parse_date_arg(s: str) -> float | None:
    """Parse YYYY-MM-DD (or anything _parse_dates finds). Never raises."""
    try:
        s = (s or "").strip()
        m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
        if m:
            return _to_ts(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        dates = _parse_dates(s)
        return dates[0][0] if dates else None
    except Exception:  # noqa: BLE001
        return None


def control_contracts(tail: str, context: Any = None, chat: Any = None,
                      sender_id: str = "", sender: str = "") -> str:
    """/contracts — contract portfolio. Owner-only; never raises."""
    try:
        portfolio = ContractPortfolio()
        rest = (tail or "").strip()
        if not rest or rest.lower() == "help":
            return _usage() + "\n\n" + DISCLAIMER
        low = rest.lower()
        if low == "timeline":
            events = portfolio.timeline()
            if not events:
                return "No obligations tracked yet.\n\n" + DISCLAIMER
            lines = ["🗓️ Contract timeline:"]
            lines += [e.headline() for e in events[:20]]
            if len(events) > 20:
                lines.append(f"…and {len(events) - 20} more.")
            return "\n".join(lines) + "\n\n" + DISCLAIMER
        if low.startswith("attention"):
            parts = rest.split()
            days = 60.0
            if len(parts) > 1:
                try:
                    days = max(1.0, float(parts[1]))
                except ValueError:
                    days = 60.0
            return portfolio.attention_text(days)
        if low.startswith("add "):
            body = rest[4:].strip()
            if "|" in body:
                name, text = body.split("|", 1)
            else:
                name, text = "", body
            name, text = name.strip(), text.strip()
            if len(text) < 80:
                return ("I need the contract text to review it — paste the "
                        "clauses after the command.\n" + _usage())
            result = portfolio.add_raw(text, name=name or "Contract")
            if result is None:
                return "Couldn't ingest that contract.\n\n" + DISCLAIMER
            contract, review = result
            obls = portfolio.obligations(contract.id)
            lines = [
                f"📥 Added: {contract.headline()}",
                f"Grade: {review.grade} ({review.score}/100), "
                f"{review.needs_attention} clause(s) flagged.",
                f"Tracking {len(obls)} date(s)/obligation(s) from the text.",
                f"ID: {contract.id}",
            ]
            return "\n".join(lines) + "\n\n" + DISCLAIMER
        if low.startswith("obligations "):
            cid = rest[len("obligations "):].strip()
            contract = portfolio.get_contract(cid)
            if contract is None:
                return "No contract with that ID. See /contracts for the list."
            obls = portfolio.obligations(cid)
            if not obls:
                return f"{contract.name}: no open obligations.\n\n{DISCLAIMER}"
            lines = [f"📋 {contract.name} — open obligations:"]
            for o in obls:
                when = "OVERDUE" if o.overdue() else ""
                lines.append(f"  • [{o.id}] {o.description[:90]} {when}")
            return "\n".join(lines) + "\n\n" + DISCLAIMER
        if low.startswith("track "):
            body = rest[len("track "):].strip()
            parts = [p.strip() for p in body.split("|")]
            if len(parts) != 3:
                return ("Usage: /contracts track <contract id> | <description> | "
                        "<YYYY-MM-DD>\n" + _usage())
            cid, desc, datestr = parts
            ts = _parse_date_arg(datestr)
            if ts is None:
                return f"Couldn't read that date: {datestr}"
            oid = portfolio.track_obligation(cid, desc, ts)
            if oid is None:
                return "Couldn't track that obligation."
            return f"📌 Tracked: {desc[:80]} — ID {oid}\n\n{DISCLAIMER}"
        if low.startswith("done "):
            oid = rest[len("done "):].strip()
            if portfolio.mark_done(oid):
                return f"✅ Marked done: {oid}"
            return "No obligation with that ID."
        if low.startswith("sla "):
            body = rest[len("sla "):].strip()
            first, sep, target = body.partition("|")
            toks = first.split()
            cid = toks[0] if toks else ""
            metric = " ".join(toks[1:]).strip()
            target = target.strip()
            if not sep or not cid or not metric or not target:
                return ("Usage: /contracts sla <contract id> <metric> | <target>\n"
                        + _usage())
            if portfolio.track_sla(cid, metric, target):
                return f"📊 SLA tracked for {cid}: {metric} vs {target}"
            return "Couldn't track that SLA term."
        if low.startswith("breach "):
            body = rest[len("breach "):].strip()
            parts = [p.strip() for p in body.split("|", 1)]
            first = parts[0].split()
            if len(first) < 2:
                return ("Usage: /contracts breach <contract id> <metric> | [note]\n"
                        + _usage())
            cid, metric = first[0], " ".join(first[1:])
            note = parts[1] if len(parts) > 1 else ""
            if portfolio.report_sla_breach(cid, metric, note):
                return (f"📊 SLA breach recorded: {metric} (contract {cid}). "
                        f"This is a recorded fact from your report — not legal "
                        f"advice.\n\n{DISCLAIMER}")
            return "Couldn't record that — check the contract ID and metric."
        return portfolio.summary()
    except Exception as e:  # noqa: BLE001 — never raise from chat
        return f"Contract portfolio hit an error ({e}).\n\n{DISCLAIMER}"
