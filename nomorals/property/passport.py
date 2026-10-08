"""Verify-once rental passport + true-cost calculator (build-map #94).

The passport is the owner's reusable tenant profile: KYC summary, income
*band* (never an exact figure), rental history, and references. Identity
documents stay in the vault — this module stores vault *references*
(document IDs + labels) only. It never opens a vault, never copies a
secret, never touches ``nomorals/accounts/vault.py``.

The true-cost calculator parses a listing's fee language and fills gaps
with Lagos market norms, so the owner sees the real move-in number before
getting emotionally attached: "₦2.5m/yr headline = ₦3.35m real move-in."

Kwaba-pattern affordability: rent is affordable when the monthly rent is
at most 33% of monthly income (band midpoint used as the income estimate).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import uuid
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)

__all__ = [
    "INCOME_BANDS",
    "AGENCY_PCT",
    "LEGAL_PCT",
    "AFFORDABILITY_RATIO",
    "RentalHistory",
    "Reference",
    "DocRef",
    "RentalPassport",
    "PassportStore",
    "CostBreakdown",
    "true_cost",
    "can_afford",
    "affordability",
    "control_passport",
    "control_truecost",
    "register",
]

# ── norms ─────────────────────────────────────────────────────────────

#: Income bands: key → (low, high) monthly ₦. ``None`` high = open-ended.
#: Bands, never exact figures — the passport attests a band.
INCOME_BANDS: dict[str, tuple[int, int | None]] = {
    "under-100k": (0, 100_000),
    "100k-250k": (100_000, 250_000),
    "250k-500k": (250_000, 500_000),
    "500k-1m": (500_000, 1_000_000),
    "1m-2m": (1_000_000, 2_000_000),
    "2m-5m": (2_000_000, 5_000_000),
    "5m+": (5_000_000, None),
}

#: Lagos market norms for the fees listings often omit.
AGENCY_PCT = 10.0   # agency fee is typically 10% of annual rent
LEGAL_PCT = 10.0    # legal/documentation agreement fee, typically 10%

#: Kwaba-pattern affordability: monthly rent ≤ 33% of monthly income.
AFFORDABILITY_RATIO = 0.33


def _default_db() -> str:
    return os.path.join(os.path.expanduser("~"), ".nomorals", "property",
                        "passport.db")


# ── money parsing ─────────────────────────────────────────────────────

_MONEY_RE = re.compile(
    r"[₦n]\s*([\d,.]+)\s*(m|million|k|thousand)?\b", re.IGNORECASE)
_BARE_RE = re.compile(r"\b([\d,]{5,})\s*(naira|ngn)?\b", re.IGNORECASE)


def _parse_kobo(text: str) -> int:
    """₦2,500,000 / N2.5m / 2,500,000 naira → kobo. 0 when unparseable."""
    t = (text or "").lower()
    m = _MONEY_RE.search(t)
    if not m:
        m = _BARE_RE.search(t)
    if not m:
        return 0
    try:
        num = float(m.group(1).replace(",", ""))
    except (ValueError, IndexError):
        return 0
    mult = {"m": 1_000_000, "million": 1_000_000,
            "k": 1_000, "thousand": 1_000}.get(
                (m.group(2) or "").lower() if len(m.groups()) > 1 else "", 1)
    return int(num * mult * 100)


def _naira(kobo: int) -> str:
    return f"₦{kobo // 100:,}"


def _fee_kobo(text: str, keywords: list[str], base_kobo: int) -> tuple[int, bool]:
    """Extract a fee from listing text.

    Returns (kobo, explicit) — explicit=False means the value came from a
    "%" figure rather than a stated amount; the caller decides norms.
    Looks for the keyword, then a percentage or a money amount in the
    surrounding window (percentage may also precede the keyword).
    """
    t = (text or "").lower()
    for kw in keywords:
        for m in re.finditer(re.escape(kw), t):
            # % right after the keyword ("agency fee 10%")…
            after = t[m.end():m.end() + 60]
            pm = re.search(r"(\d+(?:\.\d+)?)\s*%", after)
            if pm and base_kobo > 0:
                try:
                    return int(base_kobo * float(pm.group(1)) / 100), True
                except ValueError:
                    pass
            # …or tight before it ("10% agency fee") — 12 chars only, so a
            # percentage belonging to another fee can't leak in.
            before = t[max(0, m.start() - 12):m.start()]
            pm = re.search(r"(\d+(?:\.\d+)?)\s*%\s*$", before)
            if pm and base_kobo > 0:
                try:
                    return int(base_kobo * float(pm.group(1)) / 100), True
                except ValueError:
                    pass
            amt = _parse_kobo(after)
            if amt > 0:
                return amt, True
    return 0, False


# ── passport data model ───────────────────────────────────────────────

@dataclass
class RentalHistory:
    address: str = ""
    landlord: str = ""
    years: str = ""            # e.g. "2022–2024"
    note: str = ""


@dataclass
class Reference:
    name: str = ""
    relationship: str = ""     # "former landlord", "employer", ...
    contact_ref: str = ""      # vault/contact reference ID — never a raw number


@dataclass
class DocRef:
    """A vault document *reference*. The vault is never opened here."""
    vault_id: str = ""         # e.g. "vault:doc_nin_2024"
    label: str = ""            # e.g. "National ID"


@dataclass
class RentalPassport:
    passport_id: str = ""
    owner_name: str = ""
    kyc_summary: str = ""
    income_band: str = ""      # INCOME_BANDS key — band, never exact
    history: list[RentalHistory] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    doc_refs: list[DocRef] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0


class PassportStore:
    """SQLite passport storage. Never raises. Stores vault references only."""

    def __init__(self, db_path: str = "") -> None:
        self._db: sqlite3.Connection | None = None
        try:
            path = db_path or _default_db()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._db = sqlite3.connect(path)
            self._db.row_factory = sqlite3.Row
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS passports (
                       id TEXT PRIMARY KEY, owner_name TEXT, kyc_summary TEXT,
                       income_band TEXT, created_at REAL, updated_at REAL)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS history (
                       id TEXT PRIMARY KEY, passport_id TEXT, address TEXT,
                       landlord TEXT, years TEXT, note TEXT)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS refs (
                       id TEXT PRIMARY KEY, passport_id TEXT, name TEXT,
                       relationship TEXT, contact_ref TEXT)""")
            self._db.execute(
                """CREATE TABLE IF NOT EXISTS doc_refs (
                       id TEXT PRIMARY KEY, passport_id TEXT,
                       vault_id TEXT, label TEXT)""")
            self._db.commit()
        except Exception:  # noqa: BLE001 — a bad DB path is an empty store
            _log.warning("passport: db unavailable, running empty",
                         exc_info=True)
            self._db = None

    # ── assembly ──

    def generate(self, owner_name: str = "", *,
                 memory: dict | None = None,
                 income_band: str = "",
                 kyc_summary: str = "") -> RentalPassport | None:
        """Assemble a passport. ``memory`` is an optional dict-like with
        owner facts (owner_name, income_band, kyc_summary, rental_history).
        Vault documents are referenced by ID via ``add_doc_ref`` — never
        opened here.
        """
        try:
            mem = memory or {}
            name = (owner_name or str(mem.get("owner_name", "") or "")).strip()
            band = (income_band or str(mem.get("income_band", "") or "")
                    ).strip().lower()
            if band and band not in INCOME_BANDS:
                band = ""
            kyc = (kyc_summary or str(mem.get("kyc_summary", "") or "")
                   ).strip()[:500]
            p = RentalPassport(
                passport_id="pp_" + uuid.uuid4().hex[:8],
                owner_name=name or "owner",
                kyc_summary=kyc,
                income_band=band,
                created_at=time.time(), updated_at=time.time())
            for h in (mem.get("rental_history") or [])[:10]:
                if isinstance(h, dict):
                    p.history.append(RentalHistory(
                        address=str(h.get("address", ""))[:200],
                        landlord=str(h.get("landlord", ""))[:120],
                        years=str(h.get("years", ""))[:40],
                        note=str(h.get("note", ""))[:200]))
            if self._db is None:
                return p
            self._db.execute(
                "INSERT INTO passports VALUES (?, ?, ?, ?, ?, ?)",
                (p.passport_id, p.owner_name, p.kyc_summary, p.income_band,
                 p.created_at, p.updated_at))
            for h in p.history:
                self._db.execute(
                    "INSERT INTO history VALUES (?, ?, ?, ?, ?, ?)",
                    ("h_" + uuid.uuid4().hex[:8], p.passport_id, h.address,
                     h.landlord, h.years, h.note))
            self._db.commit()
            return p
        except Exception:  # noqa: BLE001
            _log.warning("passport generate failed", exc_info=True)
            return None

    def get(self, passport_id: str) -> RentalPassport | None:
        try:
            if self._db is None:
                return None
            row = self._db.execute(
                "SELECT * FROM passports WHERE id = ?",
                (passport_id,)).fetchone()
            if row is None:
                return None
            p = RentalPassport(passport_id=row["id"],
                               owner_name=row["owner_name"] or "",
                               kyc_summary=row["kyc_summary"] or "",
                               income_band=row["income_band"] or "",
                               created_at=row["created_at"] or 0.0,
                               updated_at=row["updated_at"] or 0.0)
            for h in self._db.execute(
                    "SELECT * FROM history WHERE passport_id = ?",
                    (p.passport_id,)):
                p.history.append(RentalHistory(
                    address=h["address"] or "", landlord=h["landlord"] or "",
                    years=h["years"] or "", note=h["note"] or ""))
            for r in self._db.execute(
                    "SELECT * FROM refs WHERE passport_id = ?",
                    (p.passport_id,)):
                p.references.append(Reference(
                    name=r["name"] or "", relationship=r["relationship"] or "",
                    contact_ref=r["contact_ref"] or ""))
            for d in self._db.execute(
                    "SELECT * FROM doc_refs WHERE passport_id = ?",
                    (p.passport_id,)):
                p.doc_refs.append(DocRef(vault_id=d["vault_id"] or "",
                                         label=d["label"] or ""))
            return p
        except Exception:  # noqa: BLE001
            return None

    def list(self) -> list[RentalPassport]:
        try:
            if self._db is None:
                return []
            ids = [r["id"] for r in self._db.execute(
                "SELECT id FROM passports ORDER BY updated_at DESC")]
            out = []
            for pid in ids:
                p = self.get(pid)
                if p is not None:
                    out.append(p)
            return out
        except Exception:  # noqa: BLE001
            return []

    def default(self) -> RentalPassport | None:
        items = self.list()
        return items[0] if items else None

    # ── mutation ──

    def _touch(self, passport_id: str) -> None:
        if self._db is not None:
            self._db.execute("UPDATE passports SET updated_at = ? WHERE id = ?",
                             (time.time(), passport_id))
            self._db.commit()

    def set_income_band(self, passport_id: str, band: str) -> bool:
        try:
            band = (band or "").strip().lower()
            if band not in INCOME_BANDS or self._db is None:
                return False
            self._db.execute("UPDATE passports SET income_band = ? WHERE id = ?",
                             (band, passport_id))
            self._touch(passport_id)
            return True
        except Exception:  # noqa: BLE001
            return False

    def set_kyc(self, passport_id: str, summary: str) -> bool:
        try:
            if self._db is None:
                return False
            self._db.execute(
                "UPDATE passports SET kyc_summary = ? WHERE id = ?",
                ((summary or "").strip()[:500], passport_id))
            self._touch(passport_id)
            return True
        except Exception:  # noqa: BLE001
            return False

    def add_history(self, passport_id: str, address: str, landlord: str = "",
                    years: str = "", note: str = "") -> bool:
        try:
            if self._db is None or not self.get(passport_id):
                return False
            self._db.execute(
                "INSERT INTO history VALUES (?, ?, ?, ?, ?, ?)",
                ("h_" + uuid.uuid4().hex[:8], passport_id,
                 (address or "").strip()[:200], (landlord or "").strip()[:120],
                 (years or "").strip()[:40], (note or "").strip()[:200]))
            self._touch(passport_id)
            return True
        except Exception:  # noqa: BLE001
            return False

    def add_reference(self, passport_id: str, name: str, relationship: str = "",
                      contact_ref: str = "") -> bool:
        """contact_ref is a vault/contact reference ID — never a raw number."""
        try:
            if self._db is None or not self.get(passport_id):
                return False
            self._db.execute(
                "INSERT INTO refs VALUES (?, ?, ?, ?, ?)",
                ("r_" + uuid.uuid4().hex[:8], passport_id,
                 (name or "").strip()[:120],
                 (relationship or "").strip()[:80],
                 (contact_ref or "").strip()[:120]))
            self._touch(passport_id)
            return True
        except Exception:  # noqa: BLE001
            return False

    def add_doc_ref(self, passport_id: str, vault_id: str,
                    label: str = "") -> bool:
        """Store a vault *reference* only. The vault is never opened here."""
        try:
            if self._db is None or not self.get(passport_id):
                return False
            self._db.execute(
                "INSERT INTO doc_refs VALUES (?, ?, ?, ?)",
                ("d_" + uuid.uuid4().hex[:8], passport_id,
                 (vault_id or "").strip()[:160],
                 (label or "").strip()[:120]))
            self._touch(passport_id)
            return True
        except Exception:  # noqa: BLE001
            return False

    # ── export ──

    def export_text(self, passport_id: str) -> str:
        """Shareable tenant summary. Contains bands and reference IDs —
        no exact income, no secrets, no raw contacts."""
        p = self.get(passport_id)
        if p is None:
            return "no such passport."
        lines = ["🪪 Rental passport — verify once, share anywhere"]
        if p.owner_name:
            lines.append(f"Name: {p.owner_name}")
        if p.kyc_summary:
            lines.append(f"KYC: {p.kyc_summary}")
        if p.income_band:
            low, high = INCOME_BANDS[p.income_band]
            band_txt = (f"₦{low:,}/mo+" if high is None
                        else f"₦{low:,}–₦{high:,}/mo")
            lines.append(f"Income band: {band_txt} (attested, not exact)")
        if p.history:
            lines.append("Rental history:")
            for h in p.history:
                bit = f"  • {h.address}" if h.address else "  •"
                if h.years:
                    bit += f" ({h.years})"
                if h.landlord:
                    bit += f" — landlord: {h.landlord}"
                lines.append(bit)
        if p.references:
            lines.append("References:")
            for r in p.references:
                bit = f"  • {r.name}"
                if r.relationship:
                    bit += f" ({r.relationship})"
                if r.contact_ref:
                    bit += f" — ref: {r.contact_ref}"
                lines.append(bit)
        if p.doc_refs:
            lines.append("Verified documents (vault-held, shared by ID):")
            for d in p.doc_refs:
                lines.append(f"  • {d.label or 'document'} — {d.vault_id}")
        return "\n".join(lines)


# ── true cost ─────────────────────────────────────────────────────────

@dataclass
class CostBreakdown:
    headline_kobo: int = 0       # the listed rent
    rent_kobo: int = 0
    agency_kobo: int = 0
    legal_kobo: int = 0
    service_kobo: int = 0        # service charge
    inspection_kobo: int = 0
    caution_kobo: int = 0        # refundable deposit
    other: list[tuple[str, int]] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)

    @property
    def total_kobo(self) -> int:
        return (self.rent_kobo + self.agency_kobo + self.legal_kobo
                + self.service_kobo + self.inspection_kobo
                + self.caution_kobo
                + sum(k for _, k in self.other))

    def format(self) -> str:
        lines = ["🏠 True cost — headline vs real move-in"]
        lines.append(f"Headline rent: {_naira(self.headline_kobo)}/yr")
        if self.agency_kobo:
            lines.append(f"+ Agency: {_naira(self.agency_kobo)}")
        if self.legal_kobo:
            lines.append(f"+ Legal: {_naira(self.legal_kobo)}")
        if self.service_kobo:
            lines.append(f"+ Service charge: {_naira(self.service_kobo)}")
        if self.inspection_kobo:
            lines.append(f"+ Inspection: {_naira(self.inspection_kobo)}")
        if self.caution_kobo:
            lines.append(f"+ Caution deposit: {_naira(self.caution_kobo)}"
                         " (refundable)")
        for label, k in self.other:
            lines.append(f"+ {label}: {_naira(k)}")
        lines.append(f"= Real move-in: {_naira(self.total_kobo)}")
        if self.assumptions:
            lines.append("Assumptions: " + "; ".join(self.assumptions))
        return "\n".join(lines)


_AGENCY_KEYS = ["agency fee", "agency", "commission"]
_LEGAL_KEYS = ["legal fee", "legal", "documentation fee", "agreement fee"]
_SERVICE_KEYS = ["service charge", "service fee", "estate dues"]
_INSPECTION_KEYS = ["inspection fee", "inspection"]
_CAUTION_KEYS = ["caution fee", "caution deposit", "refundable deposit"]


def true_cost(listing_text: str) -> CostBreakdown:
    """Parse a listing's fee language → real move-in cost.

    Stated fees win; gaps are filled with Lagos norms (agency 10%,
    legal 10%). Never raises.
    """
    bd = CostBreakdown()
    try:
        text = listing_text or ""
        rent = _parse_kobo(text)
        bd.headline_kobo = rent
        bd.rent_kobo = rent

        agency, explicit = _fee_kobo(text, _AGENCY_KEYS, rent)
        if explicit:
            bd.agency_kobo = agency
        elif rent > 0:
            bd.agency_kobo = int(rent * AGENCY_PCT / 100)
            bd.assumptions.append(
                f"agency at {AGENCY_PCT:g}% Lagos norm (not stated)")

        legal, explicit = _fee_kobo(text, _LEGAL_KEYS, rent)
        if explicit:
            bd.legal_kobo = legal
        elif rent > 0:
            bd.legal_kobo = int(rent * LEGAL_PCT / 100)
            bd.assumptions.append(
                f"legal at {LEGAL_PCT:g}% Lagos norm (not stated)")

        service, _ = _fee_kobo(text, _SERVICE_KEYS, rent)
        bd.service_kobo = service

        inspection, _ = _fee_kobo(text, _INSPECTION_KEYS, rent)
        bd.inspection_kobo = inspection

        caution, _ = _fee_kobo(text, _CAUTION_KEYS, rent)
        bd.caution_kobo = caution
        return bd
    except Exception:  # noqa: BLE001
        _log.warning("true_cost failed", exc_info=True)
        return CostBreakdown()


# ── affordability ─────────────────────────────────────────────────────

def affordability(rent_kobo: int, income_band: str) -> dict:
    """Kwaba-pattern pre-check. Returns detail; never raises."""
    try:
        band = (income_band or "").strip().lower()
        if band not in INCOME_BANDS or rent_kobo <= 0:
            return {"ok": False, "reason": "need a valid income band "
                    "(/passport income <band>) and a rent figure"}
        low, high = INCOME_BANDS[band]
        mid_monthly = low if high is None else (low + high) / 2
        monthly_rent = rent_kobo / 100 / 12
        ratio = monthly_rent / mid_monthly if mid_monthly > 0 else 9.0
        ok = ratio <= AFFORDABILITY_RATIO
        return {
            "ok": ok,
            "monthly_rent": int(monthly_rent),
            "income_mid_monthly": int(mid_monthly),
            "ratio": round(ratio, 2),
            "max_affordable_annual": int(
                mid_monthly * AFFORDABILITY_RATIO * 12),
            "verdict": ("affordable — within the 33% rule"
                        if ok else
                        "tight — rent exceeds 33% of the income band midpoint"),
        }
    except Exception:  # noqa: BLE001
        return {"ok": False, "reason": "couldn't assess affordability"}


def can_afford(rent_kobo: int, income_band: str) -> bool:
    """Monthly rent ≤ 33% of the income band's midpoint. Never raises."""
    try:
        return bool(affordability(rent_kobo, income_band).get("ok"))
    except Exception:  # noqa: BLE001
        return False


# ── chat ──────────────────────────────────────────────────────────────

def _usage() -> str:
    return (
        "🪪 /passport — verify-once rental passport (owner only)\n"
        "  generate [name] — assemble from memory\n"
        "  show [id] — display the passport\n"
        "  income <band> — set income band: "
        + ", ".join(INCOME_BANDS) + "\n"
        "  history add <address> | <landlord> | <years>\n"
        "  ref add <name> | <relationship> | <contact-ref-id>\n"
        "  doc add <vault-doc-id> | <label> — vault reference only, "
        "never the document\n"
        "  export [id] — shareable tenant summary")


def _memory_facts(context) -> dict:
    """Best-effort owner facts from context. Never raises."""
    try:
        mem = getattr(context, "memory", None)
        if isinstance(mem, dict):
            return {k: mem.get(k) for k in
                    ("owner_name", "income_band", "kyc_summary",
                     "rental_history") if mem.get(k)}
        get = getattr(mem, "get", None)
        if callable(get):
            out = {}
            for k in ("owner_name", "income_band", "kyc_summary",
                      "rental_history"):
                try:
                    v = get(k)
                except Exception:  # noqa: BLE001
                    v = None
                if v:
                    out[k] = v
            return out
    except Exception:  # noqa: BLE001
        pass
    return {}


def control_passport(tail: str, context=None, chat=None, **kwargs) -> str:
    """/passport — rental passport. Owner-only; never raises."""
    try:
        store = PassportStore()
        rest = (tail or "").strip()
        if not rest or rest.lower() in ("help", "?"):
            return _usage()
        low = rest.lower()

        if low.startswith("generate"):
            name = rest[8:].strip()
            p = store.generate(name, memory=_memory_facts(context))
            if p is None:
                return "couldn't assemble the passport — try again."
            return ("🪪 passport ready (" + p.passport_id + ").\n"
                    + store.export_text(p.passport_id)
                    + "\n\nAdd history/refs/docs with: "
                      "/passport history add … | ref add … | doc add …")

        if low.startswith("show") or low.startswith("export"):
            parts = rest.split(None, 1)
            p = (store.get(parts[1].strip()) if len(parts) > 1
                 else store.default())
            if p is None:
                return "no passport yet — /passport generate first."
            return store.export_text(p.passport_id)

        if low.startswith("income"):
            band = rest[6:].strip().lower()
            p = store.default()
            if p is None:
                return "no passport yet — /passport generate first."
            if band not in INCOME_BANDS:
                return ("unknown band. pick one: " + ", ".join(INCOME_BANDS))
            store.set_income_band(p.passport_id, band)
            return f"income band set to {band} (attested, never exact)."

        if low.startswith("history add"):
            body = rest[len("history add"):].strip()
            parts = [x.strip() for x in body.split("|")]
            p = store.default()
            if p is None:
                return "no passport yet — /passport generate first."
            if not parts or not parts[0]:
                return ("usage: /passport history add "
                        "<address> | <landlord> | <years>")
            ok = store.add_history(
                p.passport_id, parts[0],
                parts[1] if len(parts) > 1 else "",
                parts[2] if len(parts) > 2 else "")
            return ("rental history added." if ok
                    else "couldn't add that entry.")

        if low.startswith("ref add"):
            body = rest[len("ref add"):].strip()
            parts = [x.strip() for x in body.split("|")]
            p = store.default()
            if p is None:
                return "no passport yet — /passport generate first."
            if not parts or not parts[0]:
                return ("usage: /passport ref add "
                        "<name> | <relationship> | <contact-ref-id>")
            ok = store.add_reference(
                p.passport_id, parts[0],
                parts[1] if len(parts) > 1 else "",
                parts[2] if len(parts) > 2 else "")
            return ("reference added (stored as a reference ID)."
                    if ok else "couldn't add that reference.")

        if low.startswith("doc add"):
            body = rest[len("doc add"):].strip()
            parts = [x.strip() for x in body.split("|")]
            p = store.default()
            if p is None:
                return "no passport yet — /passport generate first."
            if not parts or not parts[0]:
                return ("usage: /passport doc add "
                        "<vault-doc-id> | <label>")
            ok = store.add_doc_ref(
                p.passport_id, parts[0],
                parts[1] if len(parts) > 1 else "")
            return ("document reference stored — the vault stays closed."
                    if ok else "couldn't add that reference.")

        return _usage()
    except Exception:  # noqa: BLE001
        _log.warning("passport control failed", exc_info=True)
        return "passport hit a snag — try again."


def control_truecost(tail: str, context=None, chat=None, **kwargs) -> str:
    """/truecost — real move-in cost from listing text. Owner-only."""
    try:
        rest = (tail or "").strip()
        if not rest or rest.lower() in ("help", "?"):
            return ("🏠 /truecost <paste listing text>\n"
                    "  parses rent + fees, fills gaps with Lagos norms, "
                    "shows the real move-in number.\n"
                    "  /truecost afford <listing text> — also checks against "
                    "your passport income band.")
        afford = False
        if rest.lower().startswith("afford "):
            afford = True
            rest = rest[7:].strip()
        bd = true_cost(rest)
        if bd.headline_kobo <= 0:
            return "couldn't find a rent figure in that text."
        out = bd.format()
        if afford:
            store = PassportStore()
            p = store.default()
            band = p.income_band if p else ""
            if not band:
                out += ("\n\nset an income band first: "
                        "/passport income <band>")
            else:
                a = affordability(bd.headline_kobo, band)
                out += (f"\n\n💰 affordability ({band}): "
                        f"{_naira(a['monthly_rent'] * 100)}/mo rent vs "
                        f"{_naira(a['income_mid_monthly'] * 100)}/mo band "
                        f"midpoint — {a['verdict']}.")
        return out
    except Exception:  # noqa: BLE001
        _log.warning("truecost control failed", exc_info=True)
        return "true-cost hit a snag — try again."


def register(registry) -> None:
    """Tool-registry hook."""
    try:
        registry.register("passport", control_passport,
                          "Verify-once rental passport — reusable tenant "
                          "profile (KYC summary, income band, rental history, "
                          "references, vault doc references).")
        registry.register("truecost", control_truecost,
                          "True rental move-in cost — parses listing fees, "
                          "fills gaps with Lagos norms.")
    except Exception:  # noqa: BLE001
        _log.debug("passport register failed", exc_info=True)
