"""Round collectors — providers are data, collection is generic.

A crash-data provider is a :class:`ProviderSpec`: where its round
history lives, how a round is shaped when it arrives, what auth it
needs.  The collectors below are the generic machinery over those
specs — a JSON history endpoint shape and a CSV/JSONL file import —
plus the open-sources posture: every fetch outcome is recorded
against the shared source registry (kind ``betting``) so sources
that serve data rise, sources that wall or die sink, and new hosts
encountered on the path enter as candidates, never as trusted feeds.

Honesty note (build-time, 2026-10-22): the big books' crash histories
are served inside JS apps / behind sessions from most networks, so a
spec can be correctly shaped and still return nothing until it runs
on a host with the owner's session or a reachable network path.  The
collector reports that as a failed/empty fetch with the reason — it
never invents rounds.
"""

from __future__ import annotations

import csv
import io
import json
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .crash import CrashRound

__all__ = [
    "ProviderSpec", "SEED_PROVIDERS", "provider_by_id",
    "collect_provider", "collect_all", "import_rounds_file",
    "parse_rounds_payload",
]

HttpGet = Callable[[str, dict], tuple[int, bytes]]


def _urllib_get(url: str, headers: dict) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (nomorals betdesk)",
        **(headers or {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read() or b""


@dataclass(frozen=True)
class ProviderSpec:
    """One crash-data source, declaratively.

    ``history_url`` may carry ``{limit}``.  ``field_map`` names where
    each CrashRound field sits in one payload record (dotted paths for
    nested JSON).  ``rounds_path`` is the dotted path to the record
    list inside a JSON payload ("" = the payload IS the list).
    """

    id: str                      # "stake" | "bustabit" | ...
    display: str
    game: str                    # "crash" | "aviator" | ...
    domain: str = ""             # source-registry domain
    history_url: str = ""
    rounds_path: str = ""
    field_map: tuple = ()        # ((crashround_field, payload_path), ...)
    auth: str = "none"           # none | api_key | session
    house_edge: float = 0.01
    notes: str = ""


#: Seed providers.  Books are SEEDS under the open-sources law — the
#: registry grows from outcomes and candidates, never stops here.
SEED_PROVIDERS: tuple[ProviderSpec, ...] = (
    ProviderSpec(
        id="bustabit", display="Bustabit", game="crash",
        domain="bustabit.com",
        notes="The original crash game; provably-fair hash chain with "
              "published salts — the lab's chain verifier targets it. "
              "Live history rides its websocket app; the desk's CSV "
              "import covers exported logs until a session collector "
              "is wired on an owner host."),
    ProviderSpec(
        id="stake", display="Stake Originals", game="crash",
        domain="stake.com", auth="session",
        notes="Stake's own Crash + Limbo family. Round history lives "
              "behind the account session — collection arms when the "
              "desk runs with the owner's Stake session."),
    ProviderSpec(
        id="sportybet", display="SportyBet", game="aviator",
        domain="sportybet.com", auth="session",
        notes="Spribe Aviator on SportyBet (NG). The public site "
              "bot-walls datacenter networks (403 at build time); "
              "collection arms on the owner's host/session."),
    ProviderSpec(
        id="msport", display="MSport", game="aviator",
        domain="msport.com", auth="session",
        notes="Spribe Aviator on MSport. Session-shaped like the "
              "other books."),
    ProviderSpec(
        id="football", display="Football.com", game="aviator",
        domain="football.com", auth="session",
        notes="Aviator + crash family on Football.com (NG). "
              "Session-shaped."),
)


def provider_by_id(pid: str) -> Optional[ProviderSpec]:
    for p in SEED_PROVIDERS:
        if p.id == pid:
            return p
    return None


def _dig(obj: Any, path: str) -> Any:
    cur = obj
    for part in (path or "").split("."):
        if not part:
            continue
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if idx < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    return cur


def _coerce_ts(value: Any) -> float:
    """Epoch seconds from epoch s / epoch ms / ISO-8601 text."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        return v / 1000.0 if v > 1e12 else v
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        v = float(text)
        return v / 1000.0 if v > 1e12 else v
    except ValueError:
        pass
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return 0.0


def parse_rounds_payload(spec: ProviderSpec, payload: Any) -> list[CrashRound]:
    """Normalize one fetched payload into CrashRounds via the spec's
    field map.  Records without a round id or a bust multiplier are
    dropped — a half-known round poisons statistics silently."""
    records = _dig(payload, spec.rounds_path) if spec.rounds_path else payload
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list):
        return []
    fmap = dict(spec.field_map)
    out: list[CrashRound] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        def grab(field_name: str, default: Any = None) -> Any:
            path = fmap.get(field_name)
            return _dig(rec, path) if path else rec.get(field_name, default)
        rid = grab("round_id")
        crash = grab("crash")
        if rid is None or crash is None:
            continue
        try:
            crash_f = float(crash)
        except (TypeError, ValueError):
            continue
        out.append(CrashRound(
            provider=spec.id, game=spec.game, round_id=str(rid),
            ts=_coerce_ts(grab("ts")), crash=crash_f,
            game_hash=str(grab("game_hash") or ""),
            prev_hash=str(grab("prev_hash") or ""),
            seed_reveal=str(grab("seed_reveal") or ""),
            source=f"collector:{spec.id}"))
    return out


def collect_provider(spec: ProviderSpec, http_get: HttpGet = _urllib_get,
                     limit: int = 500,
                     registry: Any = None) -> dict[str, Any]:
    """Fetch + normalize one provider's recent rounds.

    Returns a report dict — never raises for a network-shaped
    failure; the failure is the report (and lands on the source
    registry as a recorded outcome).
    """
    report: dict[str, Any] = {"provider": spec.id, "game": spec.game,
                              "rounds": [], "ok": False, "detail": ""}
    if not spec.history_url:
        report["detail"] = ("no keyless history endpoint configured for "
                            "this provider — arms via session collection "
                            "or file import")
        return report
    url = spec.history_url.replace("{limit}", str(limit))
    try:
        status, body = http_get(url, {})
    except Exception as exc:  # noqa: BLE001 - network shapes vary
        report["detail"] = f"fetch failed: {exc}"
        _record(registry, spec, False, report["detail"])
        return report
    if status != 200:
        report["detail"] = f"HTTP {status} from {spec.domain or url}"
        _record(registry, spec, False, report["detail"])
        return report
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
    except ValueError:
        report["detail"] = "payload was not JSON"
        _record(registry, spec, False, report["detail"])
        return report
    rounds = parse_rounds_payload(spec, payload)
    report["rounds"] = rounds
    report["ok"] = True
    report["detail"] = f"{len(rounds)} rounds parsed"
    _record(registry, spec, bool(rounds), report["detail"])
    return report


def _record(registry: Any, spec: ProviderSpec, ok: bool, detail: str) -> None:
    """Record a fetch outcome on the shared source registry, when one
    is bound — the open-sources growth seam.  Never raises."""
    if registry is None or not spec.domain:
        return
    try:
        desc = registry.for_domain(spec.domain)
        if desc is not None:
            registry.record_outcome(desc, ok, error_class="" if ok else "fetch")
    except Exception:  # noqa: BLE001 - registry trouble never breaks collection
        pass


def collect_all(store: Any, specs: tuple = SEED_PROVIDERS,
                http_get: HttpGet = _urllib_get,
                registry: Any = None) -> dict[str, Any]:
    """Run every provider collector and land what arrived in the store."""
    reports = []
    total_added = 0
    for spec in specs:
        rep = collect_provider(spec, http_get=http_get, registry=registry)
        rounds = rep.pop("rounds", [])
        if rounds:
            res = store.append(rounds)
            rep["stored"] = res
            total_added += res.get("added", 0)
        reports.append(rep)
    return {"providers": reports, "added": total_added,
            "at": time.time()}


def import_rounds_file(store: Any, path: str, provider: str,
                       game: str) -> dict[str, Any]:
    """Import an exported round history (CSV or JSONL) into the store.

    CSV columns are matched by name (case-insensitive): a round id
    column (round_id / id / game / #), a bust column (crash / bust /
    multiplier / cashed), a timestamp column (ts / time / date /
    datetime) in epoch or ISO form, optional hash columns.  This is
    the desk's guaranteed data door: whatever an owner can export
    from a book or a tracker, the lab can study.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    rounds: list[CrashRound] = []
    if path.endswith(".jsonl") or text.lstrip().startswith("{"):
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.setdefault("provider", provider)
            d.setdefault("game", game)
            r = CrashRound.from_dict(d)
            r.source = f"import:{path.rsplit('/', 1)[-1]}"
            rounds.append(r)
    else:
        reader = csv.DictReader(io.StringIO(text))
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}

        def pick(*names: str) -> Optional[str]:
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        c_id = pick("round_id", "id", "game_id", "round", "#")
        c_crash = pick("crash", "bust", "multiplier", "cashed_out",
                       "busted_at", "crash_point")
        c_ts = pick("ts", "time", "date", "datetime", "timestamp")
        c_hash = pick("hash", "game_hash")
        c_prev = pick("prev_hash", "previous_hash")
        c_seed = pick("seed", "seed_reveal", "salt")
        if c_crash is None:
            return {"ok": False, "detail": "no bust/multiplier column found",
                    "added": 0}
        for i, row in enumerate(reader):
            try:
                crash = float(str(row.get(c_crash, "")).replace("x", "")
                              .strip() or 0)
            except ValueError:
                continue
            if crash <= 0:
                continue
            rid = str(row.get(c_id, "")).strip() if c_id else ""
            rounds.append(CrashRound(
                provider=provider, game=game,
                round_id=rid or f"import-{i}",
                ts=_coerce_ts(row.get(c_ts) if c_ts else None),
                crash=crash,
                game_hash=str(row.get(c_hash, "") or "") if c_hash else "",
                prev_hash=str(row.get(c_prev, "") or "") if c_prev else "",
                seed_reveal=str(row.get(c_seed, "") or "") if c_seed else "",
                source=f"import:{path.rsplit('/', 1)[-1]}"))
    res = store.append(rounds)
    return {"ok": True, "parsed": len(rounds), **res}
