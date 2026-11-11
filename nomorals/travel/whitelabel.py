"""White-label travel assistants — sell Devon-travel B2B2C (build-map #73).

GuideGeek's model, Devon's infrastructure: per-client travel assistants
with isolated knowledge (destination guides, airline policies) and shared
travel infra (#70 Duffel, #71 watchers, #72 itineraries).

The business rules:

* **Isolation is structural.** Each client's knowledge lives in its own
  SQLite index file (``~/.nomorals/travel/knowledge/<client_id>.db``) —
  client A can never see client B's documents, because there is no shared
  index to leak through.
* **Billing is per client.** Every client runs through #68's
  :class:`~nomorals.social.whatsapp_cost.CostAwareSender` with the
  client's own budget — cost per client is tracked and billable.
* **Infra is shared.** Duffel booking, price watchers, and itinerary
  builders are the same code for every client — the moat compounds.

The pilot template is VIKI-for-Nigeria: a WhatsApp airline bot
(search → book → check-in → flight status) in Nigerian context —
₦ prices, local airlines, Pidgin-friendly copy.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "TravelClient",
    "TravelClientStore",
    "client_knowledge",
    "answer",
    "viki_template",
    "register",
    "TEMPLATES",
    "template_for",
    "hotel_concierge_template",
    "tour_operator_template",
    "car_rental_template",
    "log_client_event",
    "client_analytics",
    "route_to_human",
    "client_onboarding",
    "list_knowledge",
    "ESCALATION_PATTERNS",
]

_DEFAULT_DB = os.path.expanduser("~/.nomorals/travel/clients.db")
_KNOWLEDGE_DIR = os.path.expanduser("~/.nomorals/travel/knowledge")


# ── client record ───────────────────────────────────────────────────────────

@dataclass
class TravelClient:
    """One white-label travel assistant."""
    id: str
    name: str
    whatsapp_number: str = ""
    branding: dict = field(default_factory=dict)  # name, greeting, tone, colors
    cost_client: str = ""          # #68 CostTracker client key (defaults to id)
    created_at: float = 0.0
    active: bool = True
    channels: list[str] = field(default_factory=list)  # whatsapp, instagram…
    languages: list[str] = field(default_factory=list)  # en, yo, pcm…

    @property
    def budget_key(self) -> str:
        return self.cost_client or self.id


#: The store's final-shape DDL — the seam mapping is derived from
#: it, so the two can never drift. The native path reaches the same
#: shape via its CREATE + the channels/languages ALTERs below; those
#: ALTERs stay native-path concerns, and pre-ALTER file rows import
#: with the mapping's defaults for the columns they lack.
_DDL = (
    "CREATE TABLE IF NOT EXISTS travel_clients ("
    " id TEXT PRIMARY KEY, name TEXT, whatsapp_number TEXT,"
    " branding_json TEXT, cost_client TEXT,"
    " created_at REAL, active INTEGER DEFAULT 1,"
    " channels_json TEXT, languages_json TEXT);"
    "CREATE TABLE IF NOT EXISTS client_events ("
    " id INTEGER PRIMARY KEY AUTOINCREMENT,"
    " client_id TEXT NOT NULL, at REAL NOT NULL,"
    " kind TEXT NOT NULL, meta_json TEXT DEFAULT '{}');"
    "CREATE INDEX IF NOT EXISTS idx_ce_client "
    "ON client_events (client_id, at)"
)


class TravelClientStore:
    """SQLite registry of white-label travel clients. Never raises.

    When a document store is resolved (Mongo program P5 — an explicit
    ``store`` or the boot ambient), the same API runs over the seam
    instead (registered in storage.sidecars as travel-whitelabel):
    clients are documents in ``travel-whitelabel__travel_clients``;
    the AUTOINCREMENT event log rides ``put_raw`` with ids allocated
    by the port's atomic counter. The file's existing rows are
    imported once, through the same mapping, at first construction.
    The per-client knowledge files are a separate DocumentIndex-owned
    surface and are not part of this store on either rung.
    """

    def __init__(self, db_path: str = "", *, store: Any = None) -> None:
        path = db_path or _DEFAULT_DB
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._db: sqlite3.Connection | None = None
        self._port: Any = None
        self._imported_max_event = 0
        from ..storage.sidecars import sidecar_port
        port = sidecar_port("travel-whitelabel", _DDL,
                            ("travel_clients",), path, store=store)
        if port is not None:
            self._import_into(port)
            self._port = port
            return
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS travel_clients (
                   id TEXT PRIMARY KEY, name TEXT, whatsapp_number TEXT,
                   branding_json TEXT, cost_client TEXT,
                   created_at REAL, active INTEGER DEFAULT 1)"""
        )
        # GuideGeek lives on 4 surfaces in 15+ languages — so do our clients.
        for _ddl in (
            "ALTER TABLE travel_clients ADD COLUMN channels_json TEXT",
            "ALTER TABLE travel_clients ADD COLUMN languages_json TEXT",
        ):
            try:
                self._db.execute(_ddl)
            except Exception:  # noqa: BLE001 — already migrated
                pass
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS client_events (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   client_id TEXT NOT NULL, at REAL NOT NULL,
                   kind TEXT NOT NULL, meta_json TEXT DEFAULT '{}')"""
        )
        self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_ce_client "
            "ON client_events (client_id, at)"
        )
        self._db.commit()

    def _import_into(self, port: Any) -> None:
        """The one-time lazy file import (see SidecarPort): clients go
        through their RowTable; event rows keep their file ids via
        put_raw, and the id counter is seeded from the imported
        maximum so seam-allocated ids continue the file's sequence."""

        def write_clients(rows: list[dict[str, Any]]) -> int:
            table = port.tables["travel_clients"]
            for row in rows:
                table.put(row)
            return len(rows)

        def write_events(rows: list[dict[str, Any]]) -> int:
            for row in rows:
                eid = int(row["id"])
                self._imported_max_event = max(self._imported_max_event, eid)
                port.put_raw("client_events", str(eid), {
                    "id": eid, "client_id": row["client_id"],
                    "at": row["at"], "kind": row["kind"],
                    "meta_json": row["meta_json"]})
            return len(rows)

        def seed_counter(_counts: dict[str, int]) -> None:
            doc = port.store.get(port.meta_collection, "counters") or {}
            if int(doc.get("client_events") or 0) < self._imported_max_event:
                port.store.put(port.meta_collection, "counters", {
                    **doc, "client_events": self._imported_max_event})

        port.ensure_imported(
            {"travel_clients": write_clients,
             "client_events": write_events},
            after_import=seed_counter)

    @staticmethod
    def _row_to_client(row: Any) -> TravelClient:
        import json
        cols = set(row.keys())
        return TravelClient(
            id=row["id"], name=row["name"],
            whatsapp_number=row["whatsapp_number"] or "",
            branding=json.loads(row["branding_json"] or "{}"),
            cost_client=row["cost_client"] or "",
            created_at=row["created_at"], active=bool(row["active"]),
            channels=json.loads(row["channels_json"] or '["whatsapp"]')
            if "channels_json" in cols else ["whatsapp"],
            languages=json.loads(row["languages_json"] or '["en"]')
            if "languages_json" in cols else ["en"],
        )

    def create(self, name: str, *, whatsapp_number: str = "",
               branding: dict | None = None,
               cost_client: str = "",
               channels: list[str] | None = None,
               languages: list[str] | None = None) -> TravelClient | None:
        """Create and persist a client. Returns the client, or None
        when the INSERT/commit failed — never a phantom client that
        a later get()/list() can't find."""
        import json
        client = TravelClient(
            id="tcl_" + uuid.uuid4().hex[:10],
            name=(name or "Unnamed").strip(),
            whatsapp_number=whatsapp_number.strip(),
            branding=dict(branding or {}),
            cost_client=(cost_client or "").strip(),
            created_at=time.time(),
            channels=list(channels or ["whatsapp"]),
            languages=list(languages or ["en"]),
        )
        try:
            if self._port is not None:
                self._port.tables["travel_clients"].put({
                    "id": client.id, "name": client.name,
                    "whatsapp_number": client.whatsapp_number,
                    "branding_json": json.dumps(client.branding),
                    "cost_client": client.cost_client,
                    "created_at": client.created_at, "active": 1,
                    "channels_json": json.dumps(client.channels),
                    "languages_json": json.dumps(client.languages)})
                return client
            self._db.execute(
                "INSERT INTO travel_clients (id, name, whatsapp_number,"
                " branding_json, cost_client, created_at, active,"
                " channels_json, languages_json)"
                " VALUES (?,?,?,?,?,?,1,?,?)",
                (client.id, client.name, client.whatsapp_number,
                 json.dumps(client.branding), client.cost_client,
                 client.created_at, json.dumps(client.channels),
                 json.dumps(client.languages)),
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("travel client create failed", exc_info=True)
            return None
        return client

    def get(self, client_id: str) -> TravelClient | None:
        try:
            if self._port is not None:
                row = self._port.tables["travel_clients"].get(client_id)
                return self._row_to_client(row) if row else None
            row = self._db.execute(
                "SELECT * FROM travel_clients WHERE id = ?",
                (client_id,)).fetchone()
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        return self._row_to_client(row)

    def list(self, *, active_only: bool = True) -> list[TravelClient]:
        try:
            if self._port is not None:
                rows = self._port.tables["travel_clients"].find()
                if active_only:
                    rows = [r for r in rows if int(r["active"] or 0) == 1]
                rows.sort(key=lambda r: r["created_at"] or 0,
                          reverse=True)
                return [self._row_to_client(r) for r in rows]
            q = "SELECT * FROM travel_clients"
            if active_only:
                q += " WHERE active = 1"
            rows = self._db.execute(q + " ORDER BY created_at DESC").fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [self._row_to_client(r) for r in rows]

    def deactivate(self, client_id: str) -> bool:
        try:
            if self._port is not None:
                table = self._port.tables["travel_clients"]
                if table.get(client_id) is None:
                    return False
                return table.update(client_id, {"active": 0})
            cur = self._db.execute(
                "UPDATE travel_clients SET active = 0 WHERE id = ?",
                (client_id,))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False

    # ── client events (the analytics feed) ──────────────────────────

    def log_event(self, client_id: str, kind: str,
                  meta: dict | None = None) -> None:
        """Append one event row. Raises on failure — the module-level
        wrapper owns the never-raises contract, as it does natively."""
        import json
        if self._port is not None:
            eid = self._port.next_id("client_events")
            self._port.put_raw("client_events", str(eid), {
                "id": eid, "client_id": client_id, "at": time.time(),
                "kind": kind, "meta_json": json.dumps(meta or {})})
            return
        self._db.execute(
            "INSERT INTO client_events (client_id, at, kind, meta_json)"
            " VALUES (?, ?, ?, ?)",
            (client_id, time.time(), kind, json.dumps(meta or {})))
        self._db.commit()

    def event_counts(self, client_id: str,
                     cutoff: float) -> dict[str, int]:
        """{kind: count} for one client since ``cutoff`` — the
        analytics GROUP BY, re-expressed over finds on the seam."""
        if self._port is not None:
            counts: dict[str, int] = {}
            for r in self._port.find_raw(
                    "client_events", {"client_id": client_id}):
                if float(r["at"]) >= cutoff:
                    counts[r["kind"]] = counts.get(r["kind"], 0) + 1
            return counts
        rows = self._db.execute(
            "SELECT kind, COUNT(*) AS n FROM client_events"
            " WHERE client_id = ? AND at >= ? GROUP BY kind",
            (client_id, cutoff)).fetchall()
        return {r["kind"]: r["n"] for r in rows}


# ── per-client knowledge (structural isolation) ─────────────────────────────

def _knowledge_path(client_id: str) -> str:
    safe = "".join(c for c in client_id if c.isalnum() or c in "_-")
    os.makedirs(_KNOWLEDGE_DIR, exist_ok=True)
    return os.path.join(_KNOWLEDGE_DIR, safe + ".db")


def _registry_path(client_id: str) -> str:
    """Sidecar registry of what's IN a client's knowledge (titles + dates).

    The .db file is DocumentIndex-owned; this JSON is the table of contents.
    """
    safe = "".join(c for c in client_id if c.isalnum() or c in "_-")
    os.makedirs(_KNOWLEDGE_DIR, exist_ok=True)
    return os.path.join(_KNOWLEDGE_DIR, safe + ".docs.json")


def list_knowledge(client_id: str) -> list[dict]:
    """Table of contents of a client's corpus. Never raises."""
    import json
    try:
        with open(_registry_path(client_id), encoding="utf-8") as fh:
            docs = json.load(fh)
        return docs if isinstance(docs, list) else []
    except Exception:  # noqa: BLE001
        return []


def client_knowledge(client_id: str):
    """The DocumentIndex for one client — its own file, no shared state."""
    from ..documents.index import DocumentIndex
    from ..storage.db import Database
    return DocumentIndex(Database(_knowledge_path(client_id)))


def add_knowledge(client_id: str, title: str, text: str) -> bool:
    """Inject one document into a client's corpus. Never raises."""
    try:
        import json
        from ..documents.model import Document, Section
        idx = client_knowledge(client_id)
        doc = Document(id="doc_" + uuid.uuid4().hex[:8], title=title,
                       format="text", source="travelclient",
                       created_at=time.time())
        doc.sections = [Section(heading="", text=text)]
        idx.add(doc)
        # table of contents for onboarding/listing
        try:
            docs = list_knowledge(client_id)
            docs.append({"title": title.strip(),
                         "added_at": time.time(),
                         "chars": len(text or "")})
            with open(_registry_path(client_id), "w",
                      encoding="utf-8") as fh:
                json.dump(docs, fh)
        except Exception:  # noqa: BLE001
            pass
        return True
    except Exception:  # noqa: BLE001
        _log.debug("knowledge add failed", exc_info=True)
        return False


def answer(client_id: str, question: str, *, limit: int = 3) -> dict:
    """Grounded Q&A over ONE client's corpus. Never crosses clients."""
    try:
        idx = client_knowledge(client_id)
        hits = idx.search(question, limit=limit)
        return {"client_id": client_id, "question": question,
                "hits": hits, "grounded": bool(hits)}
    except Exception:  # noqa: BLE001
        return {"client_id": client_id, "question": question,
                "hits": [], "grounded": False}


# ── billing per client (#68) ────────────────────────────────────────────────

def client_sender(client_id: str,
                  sender: Callable[[str, str], bool] | None = None,
                  *,
                  category: str = "service",
                  db_path: str = ""):
    """A #68 CostAwareSender scoped to one client's budget."""
    from ..social.whatsapp_cost import CostAwareSender, CostTracker
    tracker = CostTracker(db_path=db_path)
    base = sender or (lambda phone, text: False)
    return CostAwareSender(base, tracker=tracker, client=client_id,
                           category=category)


def client_spend_report(client_id: str, *, db_path: str = "") -> str:
    """What this client has cost on WhatsApp."""
    from ..social.whatsapp_cost import CostTracker
    try:
        tracker = CostTracker(db_path=db_path)
        return tracker.weekly_spend(client=client_id)
    except Exception as exc:  # noqa: BLE001
        assert isinstance(exc, Exception)
        return f"💸 spend data unavailable for {client_id}."


# ── client analytics (the B2B product surface) ──────────────────────────────

#: Event kinds tracked per client.
EVENT_QUERY = "query"
EVENT_BOOKING = "booking"
EVENT_ESCALATION = "escalation"
EVENT_KNOWLEDGE_HIT = "knowledge_hit"
EVENT_KNOWLEDGE_MISS = "knowledge_miss"


def log_client_event(client_id: str, kind: str,
                     meta: dict | None = None, *,
                     db_path: str = "") -> bool:
    """Record one client event. Never raises."""
    try:
        store = TravelClientStore(db_path)
        store.log_event(client_id, kind, meta)
        return True
    except Exception:  # noqa: BLE001
        _log.debug("client event log failed", exc_info=True)
        return False


def client_analytics(client_id: str, *, days: int = 30,
                     db_path: str = "") -> str:
    """Queries, bookings, escalations, knowledge grounding, spend — one view."""
    try:
        store = TravelClientStore(db_path)
        cutoff = time.time() - days * 86400
        counts = store.event_counts(client_id, cutoff)
    except Exception:  # noqa: BLE001
        return f"no analytics for {client_id}."
    queries = counts.get(EVENT_QUERY, 0)
    hits = counts.get(EVENT_KNOWLEDGE_HIT, 0)
    misses = counts.get(EVENT_KNOWLEDGE_MISS, 0)
    ground_rate = (hits / (hits + misses) * 100
                   if hits + misses else 0.0)
    lines = [f"📊 {client_id} — last {days}d"]
    lines.append(f"  queries: {queries}")
    lines.append(f"  bookings: {counts.get(EVENT_BOOKING, 0)}")
    lines.append(f"  escalations: {counts.get(EVENT_ESCALATION, 0)}")
    lines.append(f"  knowledge grounding: {ground_rate:.0f}%"
                 + (f" ({hits} hit / {misses} miss)" if hits + misses else ""))
    spend = client_spend_report(client_id, db_path=db_path)
    lines.append(f"  {spend}")
    return "\n".join(lines)


# ── escalation (the human-curation layer) ───────────────────────────────────

#: Signals that the bot should hand the chat to a human agent.
ESCALATION_PATTERNS = (
    r"\bhuman\b", r"\bagent\b", r"\bcomplaint\b", r"\brefund\b",
    r"\bcancel\s+(my\s+)?booking\b", r"\bscam\b", r"\bfraud\b",
    r"\bmanager\b", r"\blawyer\b", r"\buseless\b", r"\bstupid\b",
    r"\btalk\s+to\s+someone\b", r"\breal\s+person\b",
)


def route_to_human(text: str) -> tuple[bool, str]:
    """Should this message escalate? → (yes, matched reason). Pure."""
    import re
    low = (text or "").lower()
    for pat in ESCALATION_PATTERNS:
        m = re.search(pat, low)
        if m:
            return True, f"matched '{m.group(0)}'"
    if "!" in low and len(low) < 60 and any(
            w in low for w in ("angry", "furious", "terrible", "worst",
                               "disgusting")):
        return True, "angry short message"
    return False, ""


# ── onboarding checklist ────────────────────────────────────────────────────

def client_onboarding(client_id: str, *, db_path: str = "") -> str:
    """Readiness report for a new client: knowledge? channel? billing?"""
    store = TravelClientStore(db_path)
    client = store.get(client_id)
    if client is None:
        return "no such client."
    docs = list_knowledge(client_id)
    checks: list[tuple[str, bool, str]] = [
        ("knowledge base", bool(docs),
         f"{len(docs)} doc(s)" if docs else "empty — add docs first"),
        ("channel", bool(client.whatsapp_number or client.channels),
         ", ".join(client.channels) or "whatsapp"),
        ("languages", bool(client.languages),
         ", ".join(client.languages)),
        ("billing", True, f"budget key {client.budget_key}"),
    ]
    lines = [f"🚀 onboarding: {client.name} (`{client.id}`)"]
    ready = True
    for label, ok, detail in checks:
        lines.append(f"  {'✅' if ok else '⬜'} {label}: {detail}")
        ready = ready and ok
    lines.append("ready to serve ✅" if ready
                 else "finish the ⬜ items, then go live.")
    return "\n".join(lines)


# ── VIKI Nigeria template (the pilot) ───────────────────────────────────────

#: Nigerian domestic airlines for the template's inventory context.
NG_AIRLINES = (
    "Air Peace", "Arik Air", "Dana Air", "Ibom Air", "Max Air",
    "Overland Airways", "ValueJet", "Green Africa",
)


def viki_template(airline: str = "ValueJet") -> dict:
    """Pre-built Nigerian airline WhatsApp bot template.

    The booking lifecycle on WhatsApp/Telegram: search → book →
    check-in → flight status. Nigerian context throughout: ₦ prices,
    local airlines, Pidgin-friendly copy.
    """
    return {
        "name": f"{airline} Assistant (VIKI template)",
        "channel": "whatsapp",
        "flows": [
            {"id": "search",
             "trigger": "user asks for flights, e.g. 'Lagos to Abuja Friday'",
             "steps": ["parse route + date (Pidgin-friendly: 'Wetin be di date?')",
                       "search Duffel (#70) → cheapest 3 options in ₦",
                       "reply with options + [Book] buttons"]},
            {"id": "book",
             "trigger": "user taps [Book]",
             "steps": ["re-price the offer (never ticket a stale fare)",
                       "collect passenger name + phone",
                       "confirmation gate: itinerary + ₦ price → confirm",
                       "create order → PNR → send ticket summary"]},
            {"id": "checkin",
             "trigger": "'check me in' / 24h before departure",
             "steps": ["look up booking by PNR",
                       "proactive nudge 24h before: 'Check-in don open ✈️'",
                       "return boarding pass link"]},
            {"id": "status",
             "trigger": "'where is my flight?' / flight number",
             "steps": ["FlightAware status lookup",
                       "delays proactively pushed, not just answered"]},
        ],
        "context": {
            "currency": "NGN (₦)",
            "airlines": list(NG_AIRLINES),
            "tone": "warm, Pidgin-friendly, never stiff",
            "greeting": ("Wetin dey! 👋 Na me be {airline} assistant. "
                         "I fit find flights, book tickets, check you in. "
                         "Where you wan go?").format(airline=airline),
        },
        "billing": "every message through #68 CostAwareSender (client budget)",
        "infra": ["duffel (#70)", "watchers (#71)", "itinerary (#72)"],
    }


# ── template registry (beyond airlines) ─────────────────────────────────────

def hotel_concierge_template(hotel: str = "Lagos Continental") -> dict:
    """WhatsApp concierge for a hotel: rooms, dining, local experiences."""
    return {
        "name": f"{hotel} Concierge",
        "channel": "whatsapp",
        "flows": [
            {"id": "rooms",
             "trigger": "user asks about rooms / rates / availability",
             "steps": ["parse dates + guests",
                       "quote in ₦ with total-first display (#75)",
                       "offer [Reserve] with confirmation gate"]},
            {"id": "dining",
             "trigger": "'where should we eat?' / cuisine request",
             "steps": ["in-house restaurants first, then local picks",
                       "take a reservation: date, time, party size"]},
            {"id": "experiences",
             "trigger": "'what's there to do?'",
             "steps": ["curated local experiences from the knowledge base",
                       "day-plan suggestion with realistic pacing"]},
            {"id": "service",
             "trigger": "complaints / requests (extra towels, late checkout)",
             "steps": ["log as a service ticket",
                       "escalate to a human when route_to_human() fires"]},
        ],
        "context": {
            "currency": "NGN (₦)",
            "tone": "warm, hospitable, Pidgin-friendly",
            "greeting": (f"Welcome to {hotel}! 🏨 I fit help with rooms, "
                         "restaurants, and showing you around town. "
                         "Wetin you need?"),
        },
        "billing": "every message through #68 CostAwareSender (client budget)",
        "infra": ["itinerary (#72)", "knowledge base"],
    }


def tour_operator_template(operator: str = "Naija Trails") -> dict:
    """Booking + guiding assistant for a tour operator / DMO."""
    return {
        "name": f"{operator} Tours",
        "channel": "whatsapp",
        "flows": [
            {"id": "discover",
             "trigger": "'where can I go?' / vibe request",
             "steps": ["ask: pace, budget, interests (GuideGeek discovery)",
                       "suggest 3 tours with ₦ prices, total-first"]},
            {"id": "book",
             "trigger": "user picks a tour",
             "steps": ["confirm date + party size",
                       "confirmation gate: itinerary + ₦ total → confirm",
                       "issue booking reference"]},
            {"id": "dayof",
             "trigger": "tour day",
             "steps": ["morning briefing: meeting point, what to bring",
                       "live help during the tour"]},
        ],
        "context": {
            "currency": "NGN (₦)",
            "tone": "adventurous, warm, Pidgin-friendly",
            "greeting": (f"You wan explore? 🌍 Na {operator} be this — "
                         "tours, hidden gems, proper Naija experiences. "
                         "Wetin dey your mind?"),
        },
        "billing": "every message through #68 CostAwareSender (client budget)",
        "infra": ["watchers (#71)", "itinerary (#72)", "knowledge base"],
    }


def car_rental_template(company: str = "Avis Nigeria") -> dict:
    """Fleet booking + pickup assistant for a car rental company."""
    return {
        "name": f"{company} Rentals",
        "channel": "whatsapp",
        "flows": [
            {"id": "quote",
             "trigger": "user asks for a car",
             "steps": ["parse pickup/dropoff dates + location",
                       "quote per car class in ₦/day, cheapest first"]},
            {"id": "book",
             "trigger": "user picks a class",
             "steps": ["driver's licence + phone",
                       "confirmation gate: dates + ₦ total → confirm",
                       "pickup instructions + reference"]},
            {"id": "extend",
             "trigger": "'extend my rental'",
             "steps": ["look up booking by reference",
                       "re-quote the extra days, confirm, extend"]},
        ],
        "context": {
            "currency": "NGN (₦)",
            "tone": "efficient, warm, Pidgin-friendly",
            "greeting": (f"Need wheels? 🚗 {company} dey here — tell me "
                         "pickup date and location, I go sort you."),
        },
        "billing": "every message through #68 CostAwareSender (client budget)",
        "infra": ["itinerary (#72)"],
    }


#: Every white-label vertical we ship. GuideGeek monetizes via DMOs;
#: we monetize via whoever runs travel on chat.
TEMPLATES: dict[str, Any] = {
    "airline": viki_template,
    "hotel": hotel_concierge_template,
    "tours": tour_operator_template,
    "car": car_rental_template,
}


def template_for(kind: str, name: str = "") -> dict:
    """Build a client template by vertical. Unknown kind → airline."""
    fn = TEMPLATES.get((kind or "").lower(), viki_template)
    return fn(name) if name else fn()


# ── chat wiring ─────────────────────────────────────────────────────────────

def _control_travelclient(self, tail: str, *, chat_key: str = "") -> str:
    """White-label travel clients. Owner-only.

    /travelclient create <name> [whatsapp]   new client assistant
    /travelclient knowledge <id> <title> | <text>   inject a doc
    /travelclient docs <id>                   list a client's knowledge
    /travelclient ask <id> <question>        grounded Q&A (client's corpus)
    /travelclient spend <id>                  WhatsApp spend report
    /travelclient stats <id>                  queries/bookings/escalations
    /travelclient onboard <id>                readiness checklist
    /travelclient templates                   available verticals
    /travelclient template <kind> [name]      preview a vertical template
    /travelclient escalate <text>             test escalation detection
    /travelclient list                        all clients
    /travelclient viki [airline]              VIKI Nigeria template
    """
    store = getattr(self, "_travelclient_store", None)
    if store is None:
        store = TravelClientStore()
        self._travelclient_store = store
    rest = (tail or "").strip()
    if not rest or rest == "list":
        clients = store.list()
        if not clients:
            return ("No travel clients yet. "
                    "`/travelclient create <name>` to start one.")
        return "\n".join(
            f"• {c.name} (`{c.id}`){' — inactive' if not c.active else ''}"
            for c in clients)
    low = rest.lower()
    if low == "templates":
        return ("📦 verticals: " + ", ".join(sorted(TEMPLATES)) +
                "\n`/travelclient template <kind> [name]` to preview.")
    if low.startswith("template "):
        parts = rest[9:].split(None, 1)
        kind = parts[0] if parts else "airline"
        name = parts[1] if len(parts) > 1 else ""
        tpl = template_for(kind, name)
        flows = "\n".join(f"  {i+1}. {f['id']}: {f['trigger']}"
                          for i, f in enumerate(tpl["flows"]))
        return (f"📦 {tpl['name']}\n{flows}\n"
                f"Greeting: {tpl['context']['greeting']}")
    if low.startswith("viki"):
        airline = rest[4:].strip() or "ValueJet"
        tpl = viki_template(airline)
        flows = "\n".join(f"  {i+1}. {f['id']}: {f['trigger']}"
                          for i, f in enumerate(tpl["flows"]))
        return (f"✈️ {tpl['name']}\n{flows}\n"
                f"Greeting: {tpl['context']['greeting']}")
    if low.startswith("create "):
        name = rest[7:].strip()
        c = store.create(name)
        if c is None:
            return ("❌ couldn't create that travel client — the save "
                    "failed, so nothing was stored. Try again.")
        return (f"✅ travel client `{c.name}` created (`{c.id}`).\n"
                f"Add knowledge: `/travelclient knowledge {c.id} <title> | <text>`\n"
                f"Readiness: `/travelclient onboard {c.id}`")
    if low.startswith("knowledge "):
        parts = rest[10:].split(None, 1)
        if len(parts) < 2 or "|" not in parts[1]:
            return "Usage: `/travelclient knowledge <id> <title> | <text>`"
        title, _, text = parts[1].partition("|")
        ok = add_knowledge(parts[0], title.strip(), text.strip())
        return "📚 knowledge added." if ok else "Couldn't add that doc."
    if low.startswith("docs "):
        docs = list_knowledge(rest[5:].strip())
        if not docs:
            return "No docs in this client's knowledge yet."
        return "📚 knowledge:\n" + "\n".join(
            f"  • {d.get('title', '?')} ({d.get('chars', 0)} chars)"
            for d in docs)
    if low.startswith("ask "):
        parts = rest[4:].split(None, 1)
        if len(parts) < 2:
            return "Usage: `/travelclient ask <id> <question>`"
        res = answer(parts[0], parts[1])
        if not res["grounded"]:
            return "No grounded answer in this client's corpus."
        top = res["hits"][0]
        return f"📖 {top.get('title', '')}\n{top.get('snippet', '')}"
    if low.startswith("spend "):
        return client_spend_report(rest[6:].strip())
    if low.startswith("stats "):
        return client_analytics(rest[6:].strip())
    if low.startswith("onboard "):
        return client_onboarding(rest[8:].strip())
    if low.startswith("escalate "):
        yes, reason = route_to_human(rest[9:])
        return (f"🚨 escalates ({reason})" if yes
                else "✅ no escalation — bot handles it.")
    return ("`/travelclient create <name>` | `knowledge <id> <title> | <text>` "
            "| `ask <id> <question>` | `spend <id>` | `list` | `viki [airline]`")


def register(registry: Any) -> None:
    """Register the travel-client tool (owner surface)."""
    from ..core.policy import Capability

    @registry.register(
        "travelclient",
        description=("White-label travel clients: create <name> | "
                     "knowledge <id> <title> | <text> | docs <id> | "
                     "ask <id> <q> | spend <id> | stats <id> | onboard <id> | "
                     "templates | template <kind> [name] | "
                     "escalate <text> | list | viki [airline]."),
        capabilities=[Capability("travel.whitelabel")],
    )
    def _travelclient_tool(ctx: Any, action: str = "list",
                           **kw: Any) -> str:
        store = TravelClientStore()
        if action == "create":
            c = store.create(str(kw.get("name", "Unnamed")))
            if c is None:
                return "create failed: the client could not be saved"
            return f"created {c.id}"
        if action == "list":
            return "\n".join(f"{c.id}: {c.name}" for c in store.list())
        if action == "viki":
            return str(viki_template(str(kw.get("airline", "ValueJet"))))
        if action == "templates":
            return ",".join(sorted(TEMPLATES))
        if action == "template":
            return str(template_for(str(kw.get("kind", "airline")),
                                    str(kw.get("name", ""))))
        if action == "stats":
            return client_analytics(str(kw.get("client_id", "")))
        if action == "onboard":
            return client_onboarding(str(kw.get("client_id", "")))
        return "unknown action"
