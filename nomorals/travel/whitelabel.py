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

    @property
    def budget_key(self) -> str:
        return self.cost_client or self.id


class TravelClientStore:
    """SQLite registry of white-label travel clients. Never raises."""

    def __init__(self, db_path: str = "") -> None:
        path = db_path or _DEFAULT_DB
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._db = sqlite3.connect(path)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS travel_clients (
                   id TEXT PRIMARY KEY, name TEXT, whatsapp_number TEXT,
                   branding_json TEXT, cost_client TEXT,
                   created_at REAL, active INTEGER DEFAULT 1)"""
        )
        self._db.commit()

    def create(self, name: str, *, whatsapp_number: str = "",
               branding: dict | None = None,
               cost_client: str = "") -> TravelClient:
        import json
        client = TravelClient(
            id="tcl_" + uuid.uuid4().hex[:10],
            name=(name or "Unnamed").strip(),
            whatsapp_number=whatsapp_number.strip(),
            branding=dict(branding or {}),
            cost_client=(cost_client or "").strip(),
            created_at=time.time(),
        )
        try:
            self._db.execute(
                "INSERT INTO travel_clients VALUES (?,?,?,?,?,?,1)",
                (client.id, client.name, client.whatsapp_number,
                 json.dumps(client.branding), client.cost_client,
                 client.created_at),
            )
            self._db.commit()
        except Exception:  # noqa: BLE001
            _log.debug("travel client create failed", exc_info=True)
        return client

    def get(self, client_id: str) -> TravelClient | None:
        import json
        try:
            row = self._db.execute(
                "SELECT * FROM travel_clients WHERE id = ?",
                (client_id,)).fetchone()
        except Exception:  # noqa: BLE001
            return None
        if not row:
            return None
        return TravelClient(
            id=row["id"], name=row["name"],
            whatsapp_number=row["whatsapp_number"] or "",
            branding=json.loads(row["branding_json"] or "{}"),
            cost_client=row["cost_client"] or "",
            created_at=row["created_at"], active=bool(row["active"]),
        )

    def list(self, *, active_only: bool = True) -> list[TravelClient]:
        import json
        try:
            q = "SELECT * FROM travel_clients"
            if active_only:
                q += " WHERE active = 1"
            rows = self._db.execute(q + " ORDER BY created_at DESC").fetchall()
        except Exception:  # noqa: BLE001
            return []
        return [TravelClient(
            id=r["id"], name=r["name"],
            whatsapp_number=r["whatsapp_number"] or "",
            branding=json.loads(r["branding_json"] or "{}"),
            cost_client=r["cost_client"] or "",
            created_at=r["created_at"], active=bool(r["active"]),
        ) for r in rows]

    def deactivate(self, client_id: str) -> bool:
        try:
            cur = self._db.execute(
                "UPDATE travel_clients SET active = 0 WHERE id = ?",
                (client_id,))
            self._db.commit()
            return cur.rowcount > 0
        except Exception:  # noqa: BLE001
            return False


# ── per-client knowledge (structural isolation) ─────────────────────────────

def _knowledge_path(client_id: str) -> str:
    safe = "".join(c for c in client_id if c.isalnum() or c in "_-")
    os.makedirs(_KNOWLEDGE_DIR, exist_ok=True)
    return os.path.join(_KNOWLEDGE_DIR, safe + ".db")


def client_knowledge(client_id: str):
    """The DocumentIndex for one client — its own file, no shared state."""
    from ..documents.index import DocumentIndex
    from ..storage.db import Database
    return DocumentIndex(Database(_knowledge_path(client_id)))


def add_knowledge(client_id: str, title: str, text: str) -> bool:
    """Inject one document into a client's corpus. Never raises."""
    try:
        from ..documents.model import Document, Section
        idx = client_knowledge(client_id)
        doc = Document(id="doc_" + uuid.uuid4().hex[:8], title=title,
                       format="text", source="travelclient",
                       created_at=time.time())
        doc.sections = [Section(heading="", text=text)]
        idx.add(doc)
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
    except Exception:  # noqa: BLE001
        return f"💸 no spend data for {client_id}."


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


# ── chat wiring ─────────────────────────────────────────────────────────────

def _control_travelclient(self, tail: str, *, chat_key: str = "") -> str:
    """White-label travel clients. Owner-only.

    /travelclient create <name> [whatsapp]   new client assistant
    /travelclient knowledge <id> <title> | <text>   inject a doc
    /travelclient ask <id> <question>        grounded Q&A (client's corpus)
    /travelclient spend <id>                  WhatsApp spend report
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
        return (f"✅ travel client `{c.name}` created (`{c.id}`).\n"
                f"Add knowledge: `/travelclient knowledge {c.id} <title> | <text>`")
    if low.startswith("knowledge "):
        parts = rest[10:].split(None, 1)
        if len(parts) < 2 or "|" not in parts[1]:
            return "Usage: `/travelclient knowledge <id> <title> | <text>`"
        title, _, text = parts[1].partition("|")
        ok = add_knowledge(parts[0], title.strip(), text.strip())
        return "📚 knowledge added." if ok else "Couldn't add that doc."
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
    return ("`/travelclient create <name>` | `knowledge <id> <title> | <text>` "
            "| `ask <id> <question>` | `spend <id>` | `list` | `viki [airline]`")


def register(registry: Any) -> None:
    """Register the travel-client tool (owner surface)."""
    from ..core.policy import Capability

    @registry.register(
        "travelclient",
        description=("White-label travel clients: create <name> | "
                     "knowledge <id> <title> | <text> | ask <id> <q> | "
                     "spend <id> | list | viki [airline]."),
        capabilities=[Capability("travel.whitelabel")],
    )
    def _travelclient_tool(ctx: Any, action: str = "list",
                           **kw: Any) -> str:
        store = TravelClientStore()
        if action == "create":
            c = store.create(str(kw.get("name", "Unnamed")))
            return f"created {c.id}"
        if action == "list":
            return "\n".join(f"{c.id}: {c.name}" for c in store.list())
        if action == "viki":
            return str(viki_template(str(kw.get("airline", "ValueJet"))))
        return "unknown action"
