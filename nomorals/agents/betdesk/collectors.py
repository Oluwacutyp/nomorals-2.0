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

Bustabit verdict (Pass 8, verified live 2026-10-22): there is NO
keyless HTTP history endpoint.  The root answers 200 with the SPA
shell; ``/api/games`` and ``/api/history`` return that same shell
(client-side routing fallback); the socket.io polling path returns
the shell too.  Bustabit's history rides its websocket app — a
session/socket collector is future work on an owner host.  The
desk's door for bustabit rounds today is the file import below,
whose column matching already speaks the bustabit export shape
(``game_id`` / ``busted_at`` / ``hash`` / ``date``).

Pass 8 also adds the sports fixture collectors at the bottom of this
module: the analyst's own fetchers (ESPN keyless; the Odds API and
football-data when their provisioned keys exist) driven as desk
collectors that LAND what they fetch in the analyst's BetStore —
fixtures and finished results — with every outcome recorded on the
registry against the same hosts the desk's seeds name.
"""

from __future__ import annotations

import csv
import io
from .failures import record_failure
import json
import os
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from .crash import CrashRound

__all__ = [
    "ProviderSpec", "SEED_PROVIDERS", "provider_by_id",
    "collect_provider", "collect_all", "import_rounds_file",
    "parse_rounds_payload",
    "collect_sports", "default_sports_fetchers",
    "fetch_football_data_results",
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
              "Live history rides its websocket app (Pass 8 verified: "
              "no keyless HTTP history endpoint exists — root and "
              "/api/* paths all answer the SPA shell); the desk's CSV "
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
        _record(registry, spec, False, report["detail"], status=status)
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


def _record(registry: Any, spec: ProviderSpec, ok: bool, detail: str,
            status: int = 0) -> None:
    """Record a fetch outcome on the shared source registry, when one
    is bound — the open-sources growth seam, in the registry's own
    vocabulary: a served fetch is an outcome; an HTTP 401/403 is a
    WALL (``record_blocked`` — bot-blocked, never dead); a 429 is a
    rate limit (``record_rate_limited``); anything else is a plain
    failure.  Never raises."""
    if registry is None or not spec.domain:
        return
    try:
        desc = registry.for_domain(spec.domain)
        if desc is None:
            return
        if ok:
            registry.record_outcome(desc, True)
        elif status in (401, 403):
            registry.record_blocked(desc)
        elif status == 429:
            registry.record_rate_limited(desc)
        else:
            registry.record_outcome(desc, False, error_class="fetch")
    except Exception as e:  # registry trouble never breaks collection
        record_failure("collectors._record", e)
        pass


def _record_domain(registry: Any, domain: str, ok: bool,
                   status: int = 0) -> None:
    """The sports collectors' outcome seam — same vocabulary as
    :func:`_record`, keyed by host.  Never raises."""
    if registry is None or not domain:
        return
    try:
        desc = registry.for_domain(domain)
        if desc is None:
            return
        if ok:
            registry.record_outcome(desc, True)
        elif status in (401, 403):
            registry.record_blocked(desc)
        elif status == 429:
            registry.record_rate_limited(desc)
        else:
            registry.record_outcome(desc, False, error_class="fetch")
    except Exception as e:
        record_failure("collectors._record_domain", e)
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


# ── sports fixture collectors (Pass 8) ───────────────────────────────
#
# The crash collectors above study rounds; these land the analyst's
# fixture feed.  The desk's sports leg previously only RENDERED a
# digest of ESPN fixtures — nothing was ever stored, and the two
# keyed fetchers (the Odds API, football-data) had no driver at all.
# Here they run as collectors with the same discipline as the crash
# side: fetch, land in the store the desk already defines (the
# analyst's BetStore history — played results are what its ensemble
# learns from), record the outcome on the registry, report honestly
# (a keyed feed without its key is a skipped feed with the reason,
# never a silent zero).

#: feed id -> source-registry domain (the desk's seeds name the same
#: hosts, so a fetch outcome and a health record are about one host).
_SPORTS_FEED_DOMAINS = {
    "espn": "site.api.espn.com",
    "the-odds-api": "the-odds-api.com",
    "football-data": "football-data.org",
}


def default_sports_fetchers(environ: Any = None) -> list:
    """The analyst fetchers a collection cycle drives: ESPN always
    (keyless); the keyed fetchers only when their provisioned keys
    are in the environment (the fetchers read env per construction,
    so a key provisioned minutes ago is live with no restart)."""
    from ..sports_bet import EspnFetcher, FootballDataFetcher, \
        TheOddsApiFetcher
    env = environ if environ is not None else os.environ
    fetchers: list = [EspnFetcher()]
    if str(env.get("ODDS_API_KEY", "") or "").strip():
        fetchers.append(TheOddsApiFetcher())
    if str(env.get("FOOTBALL_DATA_API_KEY", "") or "").strip():
        fetchers.append(FootballDataFetcher())
    return fetchers


def _fixture_key(fx: Any) -> tuple:
    return (fx.home, fx.away, fx.league, fx.date)


def _land_fixtures(store: Any, entries: list) -> dict[str, int]:
    """Record fetched (Fixture, odds) pairs into the BetStore,
    skipping fixtures already in the history — a collection cycle
    is idempotent, and a fixture first seen as upcoming is updated
    in place when it later arrives played (the result replaces the
    fixture row, so the analyst learns the outcome, not a dup).
    Rows are located by key per entry (the store trims its history,
    so a cached index map could go stale mid-cycle)."""
    landed = updated = skipped = 0
    for fx, snaps in entries:
        key = _fixture_key(fx)
        existing = store.fixtures()
        idx = next((i for i, f in enumerate(existing)
                    if _fixture_key(f) == key), None)
        if idx is None:
            store.record(fx, snaps)
            landed += 1
        elif fx.played and not existing[idx].played:
            store.history[idx] = {
                "fixture": asdict(fx),
                "odds": [asdict(o) for o in (snaps or [])],
            }
            store.analyst.ingest(fx)
            store.save()
            updated += 1
        else:
            skipped += 1
    return {"landed": landed, "updated": updated, "skipped": skipped}


def fetch_football_data_results(api_key: str, league: str = "PL",
                                limit: int = 100,
                                http_get: Any = None) -> list:
    """Finished football-data matches as played Fixtures — the
    results half of the feed (the analyst's FootballDataFetcher
    fetches SCHEDULED only).  Same endpoint family, same auth
    header; scores land as home_goals/away_goals so BetStore.record
    ingests them into the ensemble on landing."""
    from ..sports_bet import Fixture, _http_json
    get = http_get or _http_json
    data = get(
        f"https://api.football-data.org/v4/competitions/{league}/matches"
        f"?status=FINISHED&limit={limit}",
        headers={"X-Auth-Token": api_key})
    out = []
    for m in data.get("matches", [])[:limit]:
        score = (m.get("score") or {}).get("fullTime") or {}
        home_goals, away_goals = score.get("home"), score.get("away")
        if home_goals is None or away_goals is None:
            continue
        out.append(Fixture(
            home=m["homeTeam"]["name"], away=m["awayTeam"]["name"],
            league=league, date=m.get("utcDate", ""),
            home_goals=int(home_goals), away_goals=int(away_goals)))
    return out


def collect_sports(store: Any = None, fetchers: Any = None,
                   registry: Any = None, environ: Any = None,
                   leagues: tuple = ("PL",)) -> dict[str, Any]:
    """One sports collection cycle: drive every available fetcher,
    land fixtures (+ football-data results) in the BetStore, record
    outcomes on the registry.  Returns a per-feed report — never
    raises for a feed-shaped failure; the failure is the report.

    ``store`` defaults to the analyst's own BetStore (the same one
    the desk's sports scan reads).  Keyed feeds absent from
    ``fetchers`` because their key is missing are reported as
    skipped with the reason — visible in /desk collect, not silent.
    """
    from ..sports_bet import BetStore
    env = environ if environ is not None else os.environ
    store = store if store is not None else BetStore()
    if fetchers is None:
        fetchers = default_sports_fetchers(env)
    driven = {getattr(f, "name", "") for f in fetchers}
    report: dict[str, Any] = {"feeds": [], "landed": 0, "updated": 0,
                              "at": time.time()}
    for name, env_key in (("the-odds-api", "ODDS_API_KEY"),
                          ("football-data", "FOOTBALL_DATA_API_KEY")):
        if name not in driven:
            report["feeds"].append({
                "feed": name, "domain": _SPORTS_FEED_DOMAINS[name],
                "ok": False, "skipped": True,
                "detail": f"{env_key} not set — provision {name} "
                          f"to arm this feed"})
    for fetcher in fetchers:
        name = getattr(fetcher, "name", fetcher.__class__.__name__)
        domain = _SPORTS_FEED_DOMAINS.get(name, "")
        row: dict[str, Any] = {"feed": name, "domain": domain,
                               "ok": False, "detail": ""}
        try:
            entries = fetcher.fetch(limit=40)
        except Exception as exc:  # noqa: BLE001 - a feed reports
            row["detail"] = f"fetch failed: {exc}"
            _record_domain(registry, domain, False)
            report["feeds"].append(row)
            continue
        res = _land_fixtures(store, entries)
        row.update(res)
        row["fetched"] = len(entries)
        row["ok"] = True
        row["detail"] = (f"{len(entries)} fixtures fetched, "
                         f"{res['landed']} new landed")
        _record_domain(registry, domain, True)
        report["feeds"].append(row)
        report["landed"] += res["landed"]
        report["updated"] += res["updated"]
    # football-data RESULTS: finished matches are the analyst's
    # training food — fetched whenever the key exists, landed played.
    fd_key = str(env.get("FOOTBALL_DATA_API_KEY", "") or "").strip()
    if fd_key:
        row = {"feed": "football-data-results",
               "domain": _SPORTS_FEED_DOMAINS["football-data"],
               "ok": False, "detail": ""}
        try:
            played: list = []
            for league in leagues:
                played.extend(fetch_football_data_results(
                    fd_key, league=league))
            res = _land_fixtures(store, [(fx, []) for fx in played])
            row.update(res)
            row["fetched"] = len(played)
            row["ok"] = True
            row["detail"] = (f"{len(played)} finished matches fetched, "
                             f"{res['landed']} new landed, "
                             f"{res['updated']} results landed")
            _record_domain(registry, row["domain"], True)
            report["landed"] += res["landed"]
            report["updated"] += res["updated"]
        except Exception as exc:  # noqa: BLE001
            row["detail"] = f"results fetch failed: {exc}"
            _record_domain(registry, row["domain"], False)
        report["feeds"].append(row)
    report["history_total"] = len(store.fixtures())
    return report
