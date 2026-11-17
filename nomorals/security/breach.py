"""Breach exposure: HaveIBeenPwned checks + watcher (52-websites #23).

Two halves, both honest about what they can know:

* **Password check** — Pwned Passwords k-anonymity: the SHA-1 is
  computed locally, only the first 5 hash characters are sent to the
  range service, suffixes are matched client-side. The password
  itself — and the full hash — never leave the box, are never stored
  and never logged. Free, no key.
* **Account check** — the official HIBP v3 API (breachedaccount /
  breacheddomain), key-gated (``NM_OSINT_HIBP_KEY``). Full breach
  objects, so every hit names the breach, its date and the data
  classes exposed. Without a key this refuses in words — never a
  guessed answer.
* **Watcher** — the owner's own addresses (his call which), checked
  daily; a *new* breach naming a watched address pages him once
  through the Notifier. Clean days are silent.

The 14B-account corpus is HIBP's asset; Devon's answers ride that
public service or nothing — she cannot independently know a breach
happened. Her own-side value is the k-anonymity password check and
the watcher, both honest scopes.

Layering: this module is L3 (``security/``) — stdlib + ``core`` only.
The brain tools (``tools/breach.py``) and the OSINT email check
(``tools/osint_people.breach_check``) are thin surfaces over it.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "BREACH_WATCH_JOB",
    "BREACH_WATCH_SPEC",
    "BREACH_WATCH_TOOL",
    "BreachRecord",
    "BreachWatchStore",
    "check_account",
    "check_password",
    "ensure_breach_watch_job",
    "format_breach_reply",
    "run_breach_watch_tick_once",
]

_PWNED_RANGE_URL = "https://api.pwnedpasswords.com/range/{prefix}"
_HIBP_ACCOUNT_URL = "https://haveibeenpwned.com/api/v3/breachedaccount/{target}"
_HIBP_DOMAIN_URL = "https://haveibeenpwned.com/api/v3/breacheddomain/{target}"
_UA = "Mozilla/5.0 (compatible; NoMoralsBreach/1.0; +personal research)"

BREACH_WATCH_JOB = "breach watch"
BREACH_WATCH_TOOL = "__breach_watch_tick__"
#: Daily 05:00 local — the 02:00 slot is the backup drill's, 03:00 the
#: AV sweep's, 04:00 the airport refresh's (1st only). One watched
#: address is one HIBP call; a handful of addresses is far inside the
#: free 5k/month tier.
BREACH_WATCH_SPEC = "cron 0 5 * * *"
_STATE_NAME = "breach_watch.json"

#: A fetcher takes (url, headers) and returns (status, body text).
Fetcher = Callable[[str, dict[str, str] | None], tuple[int, str]]


# ── records ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BreachRecord:
    """One breach naming one target: name, date, what was exposed."""

    name: str
    date: str = ""
    data_classes: tuple[str, ...] = ()
    source: str = "haveibeenpwned"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "date": self.date,
            "data_classes": list(self.data_classes),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> BreachRecord:
        classes = row.get("data_classes") or []
        return cls(
            name=str(row.get("name") or ""),
            date=str(row.get("date") or ""),
            data_classes=tuple(str(c) for c in classes),
            source=str(row.get("source") or "haveibeenpwned"),
        )

    @classmethod
    def from_hibp(cls, row: dict[str, Any]) -> BreachRecord:
        classes = row.get("DataClasses") or []
        date = str(row.get("BreachDate") or row.get("PublishedDate") or "")[:10]
        return cls(
            name=str(row.get("Name") or row.get("Title") or ""),
            date=date,
            data_classes=tuple(str(c) for c in classes),
        )


# ── transport ──────────────────────────────────────────────────────────────


def _default_fetcher(url: str,
                     headers: dict[str, str] | None = None) -> tuple[int, str]:
    """The live fetcher: Devon's HTTP client, one GET."""
    from ..core.http import HttpClient

    resp = HttpClient().get(url, headers=headers or {})
    return (resp.status, resp.text)


def _fetch(fetcher: Fetcher | None) -> Fetcher:
    return fetcher if fetcher is not None else _default_fetcher


def _hibp_key_from_context(context: Any) -> str:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return ""
    osint = getattr(settings, "osint", None)
    if osint is None:
        return ""
    return str(getattr(osint, "hibp_key", "") or "")


# ── password check (k-anonymity) ───────────────────────────────────────────


def check_password(password: str, *,
                    fetcher: Fetcher | None = None) -> dict[str, Any]:
    """Has this password appeared in a known breach?

    SHA-1 locally; only the first 5 hex characters travel to
    ``api.pwnedpasswords.com/range/<prefix>``; suffixes match
    client-side. The password and the full hash never leave this
    function — they are not stored, not logged, not returned.
    Never raises: a dead network is ``ok: False`` with the error.
    """
    password = password or ""
    if not password:
        return {"ok": False, "kind": "password", "pwned": False,
                "count": 0, "prefix": "",
                "error": "breach check needs a password"}
    digest = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    prefix, suffix = digest[:5], digest[5:]
    try:
        status, body = _fetch(fetcher)(
            _PWNED_RANGE_URL.format(prefix=prefix),
            {"User-Agent": _UA, "Accept": "text/plain"})
    except Exception as exc:  # noqa: BLE001 - dead network is a fact
        return {"ok": False, "kind": "password", "pwned": False,
                "count": 0, "prefix": prefix,
                "error": f"pwned-passwords lookup failed: {exc}"}
    if status != 200:
        return {"ok": False, "kind": "password", "pwned": False,
                "count": 0, "prefix": prefix,
                "error": f"pwned-passwords returned {status}"}
    count = 0
    for line in body.splitlines():
        if ":" not in line:
            continue
        got, _, num = line.strip().partition(":")
        if got.strip().upper() == suffix:
            with contextlib.suppress(ValueError):
                count = int(num.strip())
            break
    return {"ok": True, "kind": "password", "pwned": count > 0,
            "count": count, "prefix": prefix, "error": ""}


# ── account check (official HIBP v3) ───────────────────────────────────────


def check_account(target: str, *, key: str = "",
                  fetcher: Fetcher | None = None) -> dict[str, Any]:
    """Has this email/domain appeared in a known breach?

    Full breach objects (no truncate): every hit names the breach,
    its date and the data classes exposed. 404 means clean. Without
    a key this refuses in words and names ``NM_OSINT_HIBP_KEY``.
    Never raises.
    """
    target = (target or "").strip().lower()
    if not target:
        return {"ok": False, "kind": "account", "target": target,
                "pwned": False, "count": 0, "breaches": [],
                "error": "breach check needs an email or domain"}
    key = (key or "").strip()
    if not key:
        return {"ok": False, "kind": "account", "target": target,
                "pwned": False, "count": 0, "breaches": [],
                "error": ("no HIBP API key set — add NM_OSINT_HIBP_KEY to "
                          "~/.nomorals/.env (free, 5k calls/month, "
                          "non-commercial)")}
    is_domain = "@" not in target
    template = _HIBP_DOMAIN_URL if is_domain else _HIBP_ACCOUNT_URL
    url = template.format(target=urllib.parse.quote(target))
    try:
        status, body = _fetch(fetcher)(url, {
            "User-Agent": _UA, "hibp-api-key": key,
            "Accept": "application/json",
        })
    except Exception as exc:  # noqa: BLE001 - dead network is a fact
        return {"ok": False, "kind": "account", "target": target,
                "pwned": False, "count": 0, "breaches": [],
                "error": f"HIBP lookup failed: {exc}"}
    if status == 404:
        return {"ok": True, "kind": "account", "target": target,
                "pwned": False, "count": 0, "breaches": [],
                "note": "not found in HIBP data", "error": ""}
    if status != 200:
        return {"ok": False, "kind": "account", "target": target,
                "pwned": False, "count": 0, "breaches": [],
                "error": f"HIBP returned {status}"}
    try:
        data = json.loads(body)
    except ValueError as exc:
        return {"ok": False, "kind": "account", "target": target,
                "pwned": False, "count": 0, "breaches": [],
                "error": f"HIBP returned unparseable data: {exc}"}
    rows = data if isinstance(data, list) else data.get("Breaches", [])
    records = [BreachRecord.from_hibp(b) for b in rows
               if isinstance(b, dict) and (b.get("Name") or b.get("Title"))]
    return {"ok": True, "kind": "account", "target": target,
            "pwned": bool(records), "count": len(records),
            "breaches": [r.to_dict() for r in records],
            "note": "breach names, dates and data classes — the official "
                    "API never exposes credential content",
            "error": ""}


# ── watch store ────────────────────────────────────────────────────────────


def _default_state_path(context: Any) -> Path | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(settings, "home", None)
    if not home:
        return None
    return Path(home) / "data" / _STATE_NAME


class BreachWatchStore:
    """Durable watched-address list + seen-breach journal (atomic JSON).

    The journal stores breach NAMES only — never a password, never a
    hash, never credential content of any kind.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    # -- persistence ------------------------------------------------------
    def _load(self) -> dict[str, Any]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - no state is a fact, not an error
            return {"watched": [], "seen": {}, "last_check": None}
        if not isinstance(raw, dict):
            return {"watched": [], "seen": {}, "last_check": None}
        watched = [str(w).strip().lower() for w in raw.get("watched", [])
                   if str(w).strip()]
        seen: dict[str, list[str]] = {}
        for addr, names in (raw.get("seen") or {}).items():
            if isinstance(names, list):
                seen[str(addr).strip().lower()] = [str(n) for n in names]
        return {"watched": sorted(set(watched)), "seen": seen,
                "last_check": raw.get("last_check")}

    def _save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    # -- watched list -----------------------------------------------------
    def list_watched(self) -> list[str]:
        return list(self._load()["watched"])

    def add(self, address: str) -> bool:
        address = (address or "").strip().lower()
        if not address or "@" not in address:
            return False
        state = self._load()
        if address in state["watched"]:
            return False
        state["watched"] = sorted(set(state["watched"]) | {address})
        self._save(state)
        return True

    def remove(self, address: str) -> bool:
        address = (address or "").strip().lower()
        state = self._load()
        if address not in state["watched"]:
            return False
        state["watched"] = [w for w in state["watched"] if w != address]
        state["seen"].pop(address, None)
        self._save(state)
        return True

    # -- seen journal ------------------------------------------------------
    def seen_names(self, address: str) -> list[str]:
        return list(self._load()["seen"].get(
            (address or "").strip().lower(), []))

    def new_breaches(self, address: str,
                     records: list[BreachRecord]) -> list[BreachRecord]:
        seen = set(self.seen_names(address))
        return [r for r in records if r.name and r.name not in seen]

    def record(self, address: str, records: list[BreachRecord]) -> None:
        address = (address or "").strip().lower()
        if not address:
            return
        state = self._load()
        names = sorted(set(state["seen"].get(address, []))
                       | {r.name for r in records if r.name})
        state["seen"][address] = names
        self._save(state)

    def mark_checked(self, now: float | None = None) -> None:
        state = self._load()
        state["last_check"] = float(now if now is not None else time.time())
        self._save(state)

    def last_check(self) -> float | None:
        value = self._load()["last_check"]
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None


# ── watcher tick ───────────────────────────────────────────────────────────


def run_breach_watch_tick_once(
    *,
    context: Any = None,
    notifier: Any = None,
    fetcher: Fetcher | None = None,
    state_path: str | Path | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """One breach-watch tick. Never raises.

    Checks every watched address; a NEW breach naming one pages the
    owner once through the Notifier; a clean day — or a day with
    nothing new — is silent. Without an HIBP key the tick reports
    ``ok: False`` and publishes nothing.
    """
    now_f = float(now if now is not None else time.time())
    summary: dict[str, Any] = {"ok": True, "checked": [], "new": [],
                               "skipped": "", "error": ""}
    try:
        path = (Path(state_path) if state_path is not None
                else _default_state_path(context))
        if path is None:
            return {**summary, "ok": False,
                    "error": "no home to read the watch list from"}
        store = BreachWatchStore(path)
        watched = store.list_watched()
        if not watched:
            summary["skipped"] = "no watched addresses yet"
            return summary
        key = _hibp_key_from_context(context)
        if not key:
            return {**summary, "ok": False,
                    "error": ("no HIBP API key set — add NM_OSINT_HIBP_KEY "
                              "to ~/.nomorals/.env")}
        new: list[dict[str, str]] = []
        for address in watched:
            result = check_account(address, key=key, fetcher=fetcher)
            if not result.get("ok"):
                summary["error"] = str(result.get("error") or "")
                continue
            summary["checked"].append(address)
            records = [BreachRecord.from_dict(b)
                       for b in result.get("breaches", [])]
            fresh = store.new_breaches(address, records)
            if fresh:
                store.record(address, records)
                for rec in fresh:
                    new.append({"address": address, "breach": rec.name,
                                "date": rec.date})
        store.mark_checked(now_f)
        summary["new"] = sorted({row["breach"] for row in new})
        if new and notifier is not None:
            lines = ["new breach(es) on a watched address:"]
            for row in new:
                when = f" ({row['date']})" if row["date"] else ""
                lines.append(f"  {row['address']}: {row['breach']}{when}")
            notifier.publish("breach_watch",
                             title="new breach on a watched address",
                             body="\n".join(lines))
        return summary
    except Exception as exc:  # noqa: BLE001 - tick must never raise
        return {**summary, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def ensure_breach_watch_job(scheduler: Any, *,
                            spec: str = BREACH_WATCH_SPEC) -> Any | None:
    """Register the daily breach-watch job; idempotent by name.

    A job with the same name but a stale spec is replaced; a job that
    already matches is returned untouched. Best-effort — a scheduler
    that cannot store the job returns None rather than raising.
    """
    if scheduler is None:
        return None
    try:
        for job in scheduler.list_jobs() or []:
            if isinstance(job, dict):
                name, job_spec, job_id = (job.get("name"), job.get("spec"),
                                          job.get("id"))
            else:
                name = getattr(job, "name", None)
                job_spec = getattr(job, "spec", None)
                job_id = getattr(job, "id", None)
            if name != BREACH_WATCH_JOB:
                continue
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                scheduler.remove(job_id)
            break
        return scheduler.add(
            BREACH_WATCH_JOB, spec, "tool",
            {"tool": BREACH_WATCH_TOOL, "args": {}}, routine=True)
    except Exception:
        return None


# ── reply rendering ────────────────────────────────────────────────────────


def format_breach_reply(result: dict[str, Any]) -> str:
    """One chat/CLI rendering for any breach-check result."""
    if not isinstance(result, dict) or not result.get("ok"):
        error = str((result or {}).get("error") or "breach check failed")
        return f"breach check failed: {error}"
    kind = str(result.get("kind") or "account")
    if kind == "password":
        count = int(result.get("count") or 0)
        if result.get("pwned"):
            return (f"🔓 this password has appeared in {count:,} known "
                    f"breach(es) — do not use it.")
        return ("✅ this password has not appeared in any known breach "
                "(Pwned Passwords, k-anonymity — only 5 hash characters "
                "ever left this box).")
    target = str(result.get("target") or "")
    breaches = result.get("breaches") or []
    if not result.get("pwned") or not breaches:
        return (f"✅ {target} — clean, not found in HaveIBeenPwned data.")
    lines = [f"🔓 {target} appears in {len(breaches)} breach(es):"]
    for row in breaches:
        rec = row if isinstance(row, BreachRecord) else BreachRecord.from_dict(row)
        when = f" — {rec.date}" if rec.date else ""
        classes = (", ".join(rec.data_classes)
                   if rec.data_classes else "data classes not listed")
        lines.append(f"  • {rec.name}{when} — exposed: {classes}")
    return "\n".join(lines)
