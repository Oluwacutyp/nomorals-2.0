"""Comment → DM → lead social-sales loop (build-map #100).

The ManyChat pattern: a keyword trigger on post comments
("comment 'PRICE'") fires an auto-DM flow → lead capture → CRM
entry → follow-up sequence (via the #98 send engine).

Gradient (#8) semantics: autonomous outreach is conservative
(only keyword-matched commenters, WhatsApp legs priced through
#68 cost-awareness).  An EXPLICIT owner command — "DM everyone
who commented" — always executes, regardless of the gradient.

Every function never raises.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

# ── module-level defaults ──────────────────────────────────────────────────

_WHATSAPP_SERVICE_KOBO = 1400    # ₦14 service message (Nigeria, #68)
_WHATSAPP_MARKETING_KOBO = 8400  # ₦84 marketing message (Nigeria, #68)

_DOMAINS = ("instagram", "facebook", "tiktok", "telegram", "whatsapp")


def _default_db() -> str:
    try:
        from pathlib import Path
        p = Path.home() / ".nomorals" / "social" / "leads.db"
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    except Exception:  # noqa: BLE001
        return ":memory:"


# ── records ────────────────────────────────────────────────────────────────

@dataclass
class CommentTrigger:
    trigger_id: str = ""
    platform: str = ""
    post_id: str = ""
    keyword: str = ""
    dm_template: str = ""
    followup_template: str = ""
    active: bool = True
    created_at: float = 0.0
    # ManyChat-pattern extensions (loaded from trigger_meta — see below).
    public_reply: str = ""  # public comment reply ("Sent, check your DMs")
    qualifier: str = ""     # exactly one qualifying question after delivery


#: Keywords people type anyway — they fire on ordinary comments and flood
#: the funnel with false positives (ManyChat playbook rule).
_GENERIC_KEYWORDS = frozenset({
    "yes", "info", "link", "ok", "hi", "hey", "dm", "me", "please",
    "interested", "details", "price", "how",
})


def keyword_quality(keyword: str) -> dict[str, object]:
    """Grade a trigger keyword per the ManyChat keyword rules.

    Good: one short specific word nobody types by accident ("VAULT",
    "PLAN"). Bad: generic words ("info", "link"), emoji (spam filters
    dislike them in replies, reads as bait), multi-word phrases people
    embed in sentences. Returns {"grade", "warnings"} — grade is
    "good" | "risky" | "bad".
    """
    kw = (keyword or "").strip()
    warnings: list[str] = []
    if not kw:
        return {"grade": "bad", "warnings": ["keyword is empty"]}
    words = kw.split()
    if len(words) > 2:
        warnings.append("multi-word keywords fire inside ordinary sentences — use one word")
    if kw.lower() in _GENERIC_KEYWORDS:
        warnings.append(f"'{kw}' is typed in ordinary comments — expect false positives")
    if any(ord(c) > 0x2500 for c in kw):
        warnings.append("emoji/symbol keywords read as bait and trip spam filters")
    if len(kw) > 12:
        warnings.append("long keywords get misspelled — keep it short")
    if any(c in kw for c in ".,!?;:"):
        warnings.append("punctuation in keywords is fragile — plain word only")
    grade = "good" if not warnings else ("bad" if len(warnings) > 1 else "risky")
    return {"grade": grade, "warnings": warnings}


@dataclass
class Lead:
    lead_id: str = ""
    name: str = ""
    platform: str = ""
    contact: str = ""
    interest: str = ""
    source_post: str = ""
    status: str = "new"          # new | dmed | followed_up | converted | dead
    expected_value_kobo: int = 0
    created_at: float = 0.0


@dataclass
class CommentEvent:
    platform: str = ""
    post_id: str = ""
    commenter: str = ""
    commenter_contact: str = ""
    text: str = ""


# ── store ──────────────────────────────────────────────────────────────────

class LeadStore:
    """Triggers + leads + CRM entries. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = db_path or _default_db()
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS lead_triggers
                   (trigger_id TEXT PRIMARY KEY, platform TEXT, post_id TEXT,
                    keyword TEXT, dm_template TEXT, followup_template TEXT,
                    active INTEGER, created_at REAL)"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS leads
                   (lead_id TEXT PRIMARY KEY, name TEXT, platform TEXT,
                    contact TEXT, interest TEXT, source_post TEXT,
                    status TEXT, expected_value_kobo INTEGER, created_at REAL)"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS comment_seen
                   (platform TEXT, post_id TEXT, commenter TEXT,
                    trigger_id TEXT, ts REAL,
                    PRIMARY KEY (platform, post_id, commenter, trigger_id))"""
            )
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS lead_comments
                   (platform TEXT, post_id TEXT, commenter TEXT,
                    contact TEXT, text TEXT, ts REAL)"""
            )
            # Side table for trigger extensions (public_reply, qualifier)
            # — the lead_triggers schema stays frozen for compat.
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS trigger_meta
                   (trigger_id TEXT, key TEXT, value TEXT,
                    PRIMARY KEY (trigger_id, key))"""
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            self._db = None

    # — triggers —
    def add_trigger(self, platform: str, post_id: str, keyword: str,
                    dm_template: str = "", followup_template: str = "") -> CommentTrigger | None:
        try:
            platform = (platform or "").strip().lower()
            post_id = (post_id or "").strip()
            keyword = (keyword or "").strip().upper()
            if not platform or not post_id or not keyword or self._db is None:
                return None
            if platform not in _DOMAINS:
                return None
            tid = "trig_" + uuid.uuid4().hex[:8]
            self._db.execute(
                "INSERT INTO lead_triggers VALUES (?,?,?,?,?,?,?,?)",
                (tid, platform, post_id, keyword,
                 dm_template or _default_dm_template(keyword),
                 followup_template, 1, time.time()),
            )
            self._db.commit()
            return self.get_trigger(tid)
        except Exception:  # noqa: BLE001
            return None

    def get_trigger(self, trigger_id: str) -> CommentTrigger | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM lead_triggers WHERE trigger_id = ?", (trigger_id,)
            ).fetchone()
            return self._row_trigger(row) if row else None
        except Exception:  # noqa: BLE001
            return None

    def list_triggers(self, platform: str = "") -> list[CommentTrigger]:
        try:
            if self._db is None:
                return []
            if platform:
                rows = self._db.execute(
                    "SELECT * FROM lead_triggers WHERE platform = ? ORDER BY created_at DESC",
                    (platform.strip().lower(),)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM lead_triggers ORDER BY created_at DESC").fetchall()
            return [self._row_trigger(r) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def stop_trigger(self, trigger_id: str) -> bool:
        try:
            if self._db is None or not trigger_id:
                return False
            cur = self._db.execute(
                "UPDATE lead_triggers SET active = 0 WHERE trigger_id = ?", (trigger_id,))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    def match_triggers(self, platform: str, post_id: str, text: str) -> list[CommentTrigger]:
        """Active triggers whose keyword appears in the comment text."""
        try:
            if self._db is None:
                return []
            rows = self._db.execute(
                "SELECT * FROM lead_triggers WHERE platform = ? AND post_id = ? AND active = 1",
                ((platform or "").strip().lower(), (post_id or "").strip())).fetchall()
            upper = (text or "").upper()
            return [self._row_trigger(r) for r in rows if r["keyword"] and r["keyword"] in upper]
        except Exception:  # noqa: BLE001
            return []

    # — comment log (for explicit-override "DM everyone who commented") —
    def log_comment(self, ev: CommentEvent) -> bool:
        try:
            if self._db is None or not ev.post_id:
                return False
            self._db.execute(
                "INSERT OR IGNORE INTO lead_comments VALUES (?,?,?,?,?,?)",
                (ev.platform, ev.post_id, ev.commenter, ev.commenter_contact,
                 (ev.text or "")[:500], time.time()))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def commenters(self, post_id: str, platform: str = "") -> list[dict]:
        try:
            if self._db is None:
                return []
            if platform:
                rows = self._db.execute(
                    "SELECT DISTINCT commenter, contact FROM lead_comments "
                    "WHERE post_id = ? AND platform = ?",
                    (post_id, platform.strip().lower())).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT DISTINCT commenter, contact FROM lead_comments WHERE post_id = ?",
                    (post_id,)).fetchall()
            return [{"commenter": r["commenter"], "contact": r["contact"]} for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def already_dmed(self, platform: str, post_id: str, commenter: str,
                     trigger_id: str) -> bool:
        try:
            if self._db is None:
                return False
            row = self._db.execute(
                "SELECT 1 FROM comment_seen WHERE platform = ? AND post_id = ? "
                "AND commenter = ? AND trigger_id = ?",
                (platform, post_id, commenter, trigger_id)).fetchone()
            return row is not None
        except Exception:  # noqa: BLE001
            return False

    def mark_dmed(self, platform: str, post_id: str, commenter: str, trigger_id: str) -> None:
        try:
            if self._db is None:
                return
            self._db.execute(
                "INSERT OR IGNORE INTO comment_seen VALUES (?,?,?,?,?)",
                (platform, post_id, commenter, trigger_id, time.time()))
            self._db.commit()
        except Exception:  # noqa: BLE001
            pass

    # — leads / CRM —
    def capture_lead(self, name: str, platform: str, contact: str,
                     interest: str, source_post: str,
                     expected_value_kobo: int = 0) -> Lead | None:
        try:
            if self._db is None or not contact:
                return None
            lid = "lead_" + uuid.uuid4().hex[:8]
            self._db.execute(
                "INSERT INTO leads VALUES (?,?,?,?,?,?,?,?,?)",
                (lid, (name or "").strip()[:120], (platform or "").strip().lower(),
                 contact.strip()[:200], (interest or "").strip()[:300],
                 (source_post or "").strip()[:200], "new",
                 max(0, int(expected_value_kobo or 0)), time.time()))
            self._db.commit()
            return self.get_lead(lid)
        except Exception:  # noqa: BLE001
            return None

    def get_lead(self, lead_id: str) -> Lead | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM leads WHERE lead_id = ?", (lead_id,)).fetchone()
            return self._row_lead(row) if row else None
        except Exception:  # noqa: BLE001
            return None

    def list_leads(self, status: str = "") -> list[Lead]:
        try:
            if self._db is None:
                return []
            if status:
                rows = self._db.execute(
                    "SELECT * FROM leads WHERE status = ? ORDER BY created_at DESC",
                    (status.strip().lower(),)).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM leads ORDER BY created_at DESC").fetchall()
            return [self._row_lead(r) for r in rows]
        except Exception:  # noqa: BLE001
            return []

    def set_lead_status(self, lead_id: str, status: str) -> bool:
        try:
            if self._db is None or not lead_id:
                return False
            cur = self._db.execute(
                "UPDATE leads SET status = ? WHERE lead_id = ?",
                ((status or "").strip().lower(), lead_id))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    # — rows —
    def _row_trigger(self, r: sqlite3.Row) -> CommentTrigger:
        trig = CommentTrigger(
            trigger_id=r["trigger_id"], platform=r["platform"], post_id=r["post_id"],
            keyword=r["keyword"], dm_template=r["dm_template"] or "",
            followup_template=r["followup_template"] or "",
            active=bool(r["active"]), created_at=float(r["created_at"] or 0))
        trig.public_reply = self.get_trigger_meta(trig.trigger_id, "public_reply")
        trig.qualifier = self.get_trigger_meta(trig.trigger_id, "qualifier")
        return trig

    def set_trigger_meta(self, trigger_id: str, key: str, value: str) -> bool:
        """Store a trigger extension (public_reply, qualifier)."""
        try:
            if self._db is None or not trigger_id or not key:
                return False
            self._db.execute(
                "INSERT OR REPLACE INTO trigger_meta VALUES (?,?,?)",
                (trigger_id, key, value or ""))
            self._db.commit()
            return True
        except Exception:  # noqa: BLE001
            return False

    def get_trigger_meta(self, trigger_id: str, key: str) -> str:
        try:
            if self._db is None or not trigger_id or not key:
                return ""
            row = self._db.execute(
                "SELECT value FROM trigger_meta WHERE trigger_id = ? AND key = ?",
                (trigger_id, key)).fetchone()
            return str(row["value"] or "") if row else ""
        except Exception:  # noqa: BLE001
            return ""

    def set_public_reply(self, trigger_id: str, template: str) -> bool:
        """The public comment reply posted alongside the DM.

        ManyChat's highest-ROI setting: some DMs never arrive, so the
        public reply ("Sent — check your DMs 👀") tells the commenter to
        look AND boosts the post in the algorithm. Template supports
        {name} and {keyword}.
        """
        if template and "{name}" not in template and "check your dm" not in template.lower():
            _log.debug("public reply for %s has no {name} or DM pointer", trigger_id)
        return self.set_trigger_meta(trigger_id, "public_reply", template or "")

    def set_qualifier(self, trigger_id: str, question: str) -> bool:
        """Exactly ONE qualifying question, asked after the delivery DM.

        The playbook: delivery message → one question ("Are you training
        right now, or getting back into it?") → intent answers route to
        the lead form. Anything longer is a survey and people leave.
        """
        return self.set_trigger_meta(trigger_id, "qualifier", (question or "").strip())

    def funnel_stats(self, trigger_id: str) -> dict[str, object]:
        """Per-trigger conversion funnel: comments → DMs → lead statuses.

        Answers "is this trigger pulling its weight?" with real numbers.
        """
        try:
            if self._db is None:
                return {}
            trig = self.get_trigger(trigger_id)
            if trig is None:
                return {}
            comments = self._db.execute(
                "SELECT COUNT(DISTINCT commenter) AS n FROM lead_comments "
                "WHERE platform = ? AND post_id = ?",
                (trig.platform, trig.post_id)).fetchone()
            dmed = self._db.execute(
                "SELECT COUNT(*) AS n FROM comment_seen WHERE trigger_id = ?",
                (trigger_id,)).fetchone()
            statuses = self._db.execute(
                "SELECT status, COUNT(*) AS n FROM leads "
                "WHERE source_post = ? GROUP BY status",
                (trig.post_id,)).fetchall()
            by_status = {r["status"]: int(r["n"]) for r in statuses}
            n_comments = int(comments["n"] or 0)
            n_dmed = int(dmed["n"] or 0)
            return {
                "trigger_id": trigger_id, "keyword": trig.keyword,
                "comments": n_comments, "dmed": n_dmed,
                "dm_rate": round(n_dmed / n_comments, 3) if n_comments else 0.0,
                "leads_by_status": by_status,
                "converted": by_status.get("converted", 0),
                "qualifier": trig.qualifier,
                "public_reply": bool(trig.public_reply),
            }
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _row_lead(r: sqlite3.Row) -> Lead:
        return Lead(
            lead_id=r["lead_id"], name=r["name"] or "", platform=r["platform"] or "",
            contact=r["contact"] or "", interest=r["interest"] or "",
            source_post=r["source_post"] or "", status=r["status"] or "new",
            expected_value_kobo=int(r["expected_value_kobo"] or 0),
            created_at=float(r["created_at"] or 0))


def _default_dm_template(keyword: str) -> str:
    return (
        f"Hey {{name}}! 👋 You commented '{keyword}' on our post — "
        "here are the details: {interest}. Want me to send the full breakdown?"
    )


# ── DM flow ────────────────────────────────────────────────────────────────

def render_dm(template: str, lead: Lead, extra: dict | None = None) -> str:
    """Fill a DM template. Never raises."""
    try:
        text = template or _default_dm_template("")
        vars_ = {"name": lead.name or "there", "platform": lead.platform,
                 "interest": lead.interest, "contact": lead.contact}
        if extra:
            vars_.update(extra)
        for k, v in vars_.items():
            text = text.replace("{" + k + "}", str(v))
        return text
    except Exception:  # noqa: BLE001
        return template or ""


def _whatsapp_cost_kobo(is_marketing: bool) -> int:
    return _WHATSAPP_MARKETING_KOBO if is_marketing else _WHATSAPP_SERVICE_KOBO


def handle_comment(store: LeadStore, ev: CommentEvent, *,
                   dm_sender: Callable[[str, str, str], bool] | None = None,
                   llm_fn: Callable[[str], str] | None = None,
                   followup_engine=None) -> list[dict]:
    """A comment arrived → match triggers → DM flow → lead capture.

    Returns per-trigger results: {trigger_id, dmed, lead_id, reason}.
    WhatsApp legs are priced through #68 cost-awareness via ``worth_it``.
    Never raises.
    """
    results: list[dict] = []
    try:
        if store is None or ev is None:
            return results
        store.log_comment(ev)
        triggers = store.match_triggers(ev.platform, ev.post_id, ev.text)
        for trig in triggers:
            if store.already_dmed(ev.platform, ev.post_id, ev.commenter, trig.trigger_id):
                results.append({"trigger_id": trig.trigger_id, "dmed": False,
                                "reason": "already DMed"})
                continue
            lead = store.capture_lead(
                name=ev.commenter, platform=ev.platform,
                contact=ev.commenter_contact or ev.commenter,
                interest=f"keyword '{trig.keyword}' on post {ev.post_id}",
                source_post=ev.post_id)
            if lead is None:
                results.append({"trigger_id": trig.trigger_id, "dmed": False,
                                "reason": "lead capture failed"})
                continue
            body = render_dm(trig.dm_template, lead)
            if llm_fn is not None:
                try:
                    personal = (llm_fn(f"Personalize this DM opener for {lead.name}: {body}") or "").strip()
                    if personal:
                        body = personal
                except Exception:  # noqa: BLE001
                    pass
            # #68 cost-awareness on WhatsApp legs.
            ok = True
            cost_note = ""
            if ev.platform == "whatsapp":
                cost = _whatsapp_cost_kobo(is_marketing=True)
                try:
                    from .whatsapp_cost import worth_it
                    if not worth_it(lead.expected_value_kobo, cost):
                        ok = False
                        cost_note = (f" (₦{cost/100:,.0f} WhatsApp marketing message not "
                                     "worth it — expected value too low)")
                except Exception:  # noqa: BLE001
                    pass
            sent = False
            if ok and dm_sender is not None:
                try:
                    sent = bool(dm_sender(ev.platform, lead.contact, body))
                except Exception:  # noqa: BLE001
                    sent = False
            elif ok and dm_sender is None:
                sent = True  # dry-run: flow resolved, no sender configured
            if sent:
                store.mark_dmed(ev.platform, ev.post_id, ev.commenter, trig.trigger_id)
                store.set_lead_status(lead.lead_id, "dmed")
                # Follow-up sequence via the #98 send engine.
                if followup_engine is not None and trig.followup_template:
                    try:
                        followup_engine.enqueue(
                            lead.contact, "__lead_followup__",
                            {"name": lead.name, "interest": lead.interest},
                            "whatsapp" if ev.platform == "whatsapp" else "smtp")
                    except Exception:  # noqa: BLE001
                        pass
            # The public reply rides back to the caller (the adapter posts
            # it as a comment reply): "Sent — check your DMs" both fixes
            # undelivered DMs and feeds the algorithm. Supports
            # {name} and {keyword}.
            public = ""
            if trig.public_reply:
                public = (trig.public_reply
                          .replace("{name}", lead.name or "there")
                          .replace("{keyword}", trig.keyword))
            result: dict[str, object] = {
                "trigger_id": trig.trigger_id, "dmed": sent,
                "lead_id": lead.lead_id,
                "reason": ("sent" if sent else "blocked" + cost_note),
            }
            if public:
                result["public_reply"] = public
            if sent and trig.qualifier:
                # Exactly one qualifying question after the delivery DM —
                # the caller appends it; longer flows are surveys.
                result["qualifier"] = trig.qualifier.replace(
                    "{name}", lead.name or "there")
            results.append(result)
        return results
    except Exception:  # noqa: BLE001
        return results


def dm_everyone(store: LeadStore, post_id: str, *,
                dm_sender: Callable[[str, str, str], bool] | None = None,
                llm_fn: Callable[[str], str] | None = None,
                template: str = "",
                platform: str = "") -> dict:
    """EXPLICIT-OVERRIDE: owner said "DM everyone who commented".

    Executes regardless of the #8 gradient — an explicit command is
    always executed.  WhatsApp legs still priced through #68.
    Never raises.
    """
    out = {"dmed": 0, "skipped": 0, "details": []}
    try:
        if store is None or not post_id:
            return out
        for c in store.commenters(post_id, platform):
            ev = CommentEvent(platform=platform or "instagram", post_id=post_id,
                              commenter=c["commenter"], commenter_contact=c["contact"],
                              text="(explicit broadcast)")
            lead = store.capture_lead(name=c["commenter"], platform=ev.platform,
                                      contact=c["contact"] or c["commenter"],
                                      interest="explicit broadcast to all commenters",
                                      source_post=post_id)
            if lead is None:
                out["skipped"] += 1
                continue
            body = render_dm(template or "Hey {name}! 👋 Here's what you asked about — {interest}.",
                             lead)
            if llm_fn is not None:
                try:
                    personal = (llm_fn(f"Personalize this DM for {lead.name}: {body}") or "").strip()
                    if personal:
                        body = personal
                except Exception:  # noqa: BLE001
                    pass
            ok = True
            if ev.platform == "whatsapp":
                try:
                    from .whatsapp_cost import worth_it
                    ok = worth_it(lead.expected_value_kobo,
                                  _whatsapp_cost_kobo(is_marketing=True))
                except Exception:  # noqa: BLE001
                    ok = True  # explicit command: cost gate warns, does not veto
            sent = True
            if dm_sender is not None:
                try:
                    sent = bool(dm_sender(ev.platform, lead.contact, body))
                except Exception:  # noqa: BLE001
                    sent = False
            if sent:
                store.set_lead_status(lead.lead_id, "dmed")
                out["dmed"] += 1
            else:
                out["skipped"] += 1
            out["details"].append({"contact": lead.contact, "dmed": sent,
                                  "cost_ok": ok})
        return out
    except Exception:  # noqa: BLE001
        return out


# ── evergreen lead magnets (#44) ───────────────────────────────────────────

def mark_lead_magnet(store: LeadStore, trigger_id: str, recycle: bool = True) -> bool:
    """Flag a trigger as an evergreen lead magnet: every recycled post
    (#44) with this trigger keeps pulling leads. Never raises."""
    try:
        if store is None or not trigger_id or store._db is None:
            return False
        store._db.execute(
            "CREATE TABLE IF NOT EXISTS lead_magnets (trigger_id TEXT PRIMARY KEY, ts REAL)")
        store._db.execute(
            "INSERT OR REPLACE INTO lead_magnets VALUES (?, ?)", (trigger_id, time.time()))
        store._db.commit()
        return True
    except Exception:  # noqa: BLE001
        return False


def is_lead_magnet(store: LeadStore, trigger_id: str) -> bool:
    try:
        if store is None or not trigger_id or store._db is None:
            return False
        row = store._db.execute(
            "SELECT 1 FROM lead_magnets WHERE trigger_id = ?", (trigger_id,)).fetchone()
        return row is not None
    except Exception:  # noqa: BLE001
        return False


# ── singleton + chat ───────────────────────────────────────────────────────

_STORE: LeadStore | None = None


def get_store(db_path: str = "") -> LeadStore:
    global _STORE
    if _STORE is None or db_path:
        _STORE = LeadStore(db_path=db_path)
    return _STORE


def _usage() -> str:
    return (
        "📣 /leads — comment → DM → lead loop\n"
        "  /leads trigger <post_id> <KEYWORD> [platform] — arm a keyword trigger\n"
        "  /leads triggers — list armed triggers\n"
        "  /leads stop <trigger_id> — disarm\n"
        "  /leads list [status] — CRM leads\n"
        "  /leads dm <contact> — DM one lead now\n"
        "  /leads everyone <post_id> — EXPLICIT: DM every commenter\n"
        "  /leads comment <platform> <post_id> <commenter> <contact> <text> — simulate a comment\n"
        "WhatsApp legs are priced through #68 (₦14/₦84). Explicit commands always execute."
    )


def control_leads(tail: str, context=None, chat=None, **kwargs) -> str:
    """/leads — social-sales loop. Owner-only at dispatch; never raises."""
    try:
        store: LeadStore = kwargs.get("store") or get_store()
        rest = (tail or "").strip()
        if not rest or rest.lower() in ("help", "?"):
            return _usage()
        low = rest.lower()

        if low == "triggers" or low.startswith("triggers "):
            trigs = store.list_triggers()
            if not trigs:
                return "no triggers armed. /leads trigger <post_id> <KEYWORD> [platform]"
            lines = ["📣 armed triggers:"]
            for t in trigs[:20]:
                mag = " 🧲" if is_lead_magnet(store, t.trigger_id) else ""
                state = "on" if t.active else "off"
                lines.append(f"  • {t.trigger_id} [{state}]{mag} {t.platform}:{t.post_id} — '{t.keyword}'")
            return "\n".join(lines)

        if low.startswith("trigger"):
            parts = rest[7:].strip().split()
            if len(parts) < 2:
                return "usage: /leads trigger <post_id> <KEYWORD> [platform]"
            post_id, keyword = parts[0], parts[1]
            platform = (parts[2] if len(parts) > 2 else "instagram").lower()
            trig = store.add_trigger(platform, post_id, keyword)
            if trig is None:
                return f"couldn't arm that trigger (platform must be one of: {', '.join(_DOMAINS)})."
            return (f"📣 trigger armed — comment '{trig.keyword}' on {trig.platform}:{trig.post_id}\n"
                    f"→ auto-DM flow → lead capture. id {trig.trigger_id}")

        if low.startswith("stop"):
            tid = rest[4:].strip()
            return "trigger disarmed." if store.stop_trigger(tid) else "no such trigger."

        if low.startswith("everyone"):
            post_id = rest[8:].strip().split()[0] if rest[8:].strip() else ""
            if not post_id:
                return "usage: /leads everyone <post_id>"
            out = dm_everyone(store, post_id)
            return (f"📣 explicit broadcast → {out['dmed']} DMed, {out['skipped']} skipped "
                    f"(cost gate still priced every WhatsApp leg).")

        if low.startswith("comment"):
            parts = rest[7:].strip().split(None, 4)
            if len(parts) < 5:
                return "usage: /leads comment <platform> <post_id> <commenter> <contact> <text>"
            ev = CommentEvent(platform=parts[0], post_id=parts[1], commenter=parts[2],
                              commenter_contact=parts[3], text=parts[4])
            res = handle_comment(store, ev)
            if not res:
                return "no trigger matched that comment."
            lines = []
            for r in res:
                lines.append(f"  • {r['trigger_id']}: {'DMed' if r['dmed'] else 'not sent'} — {r['reason']}")
            return "comment handled:\n" + "\n".join(lines)

        if low.startswith("list"):
            status = rest[4:].strip()
            leads = store.list_leads(status)
            if not leads:
                return "no leads yet."
            lines = [f"👥 leads ({len(leads)}):"]
            for l in leads[:25]:
                lines.append(f"  • {l.lead_id} [{l.status}] {l.name or l.contact} — {l.interest[:40]}")
            return "\n".join(lines)

        if low.startswith("dm"):
            contact = rest[2:].strip()
            if not contact:
                return "usage: /leads dm <contact>"
            lead = store.capture_lead(name=contact, platform="manual", contact=contact,
                                      interest="manual DM", source_post="manual")
            if lead is None:
                return "couldn't capture that lead."
            return (f"lead {lead.lead_id} captured for {contact}. "
                    "DM sender goes here when a platform adapter is attached (dry-run OK).")

        return _usage()
    except Exception:  # noqa: BLE001
        return "leads hiccup — try again."
