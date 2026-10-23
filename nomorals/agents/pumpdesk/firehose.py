"""Intake — the launch firehose and its supervision.

Three sources, one sink (design §4.1):

* **PumpPortal** websocket (free subs: new-token + migration) —
  the primary intake;
* **public Solana RPC** ``logsSubscribe`` on the pump program with
  stdlib Anchor decode — the always-free floor, and the trade feed
  for watched mints (per-curve ``logsSubscribe``);
* **pump.fun frontend API** — polled backfill / candidate discovery
  (the desk's watch cycle drives it; see :func:`fetch_frontend`).

Supervision is the point: one consumer, reconnect with exponential
backoff, heartbeat, and failover ordered by the ONE source
registry's health (a stood-down/disabled backend is skipped, never
retried in a hot loop).  Feed outcomes are recorded into the
registry — the health loop grades the backends, this module grows
no bespoke health code.

Volume discipline lives here too: the create firehose is thousands
of launches a day, so :class:`Prefilter` decides — cheaply, from
the launch record alone — what earns dossier work.  Everything is
ledgered; almost nothing is dossiered.  That asymmetry is the
module's cost model.
"""

from __future__ import annotations

import json
import queue
import random
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Optional

from . import chain
from .wsclient import (
    TRANSPORT_DENIED_CLASS, TransportDeniedError, WebSocketClient,
    WSClosed)

__all__ = [
    "PUMPPORTAL_WS", "Prefilter", "frontend_candidate",
    "fetch_frontend", "fetch_coin_row", "base_backoff",
    "PumpPortalBackend", "RpcLogBackend", "Firehose",
    "TransportDeniedError", "TRANSPORT_DENIED_CLASS",
]

PUMPPORTAL_WS = "wss://pumpportal.fun/api/data"
FRONTEND_API = "https://frontend-api-v3.pump.fun"

Emit = Callable[[str, dict], None]


# ── pre-filter discipline ─────────────────────────────────────────────

class Prefilter:
    """Cheap launch triage: what earns a dossier.

    Dossier-worthy: a watched dev launched it, the mint is watched,
    or it already graduated (migration seen).  Everything else is
    ledger-only until traction evidence arrives another way
    (frontend candidate poll, an owner request)."""

    def __init__(self, watched_devs: set[str] | None = None,
                 watched_mints: set[str] | None = None):
        self.watched_devs = watched_devs or set()
        self.watched_mints = watched_mints or set()

    def decide(self, launch: dict[str, Any]) -> str:
        if launch.get("mint") in self.watched_mints:
            return "dossier"
        if launch.get("creator") in self.watched_devs:
            return "dossier"
        return "ledger"

    def decide_migration(self, mint: str) -> str:
        return "dossier"  # a graduation always earns the full look


#: Frontend candidate gate: a recent coin earns a dossier look when
#: its curve (computed from the API's own reserve fields, in SOL —
#: never the API's USD figure) is already meaningfully filled.
FRONTEND_MIN_CURVE_PCT = 0.25


def frontend_candidate(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Normalize one frontend-API coin row into a launch-shaped
    record when it passes the candidate gate; None otherwise."""
    try:
        state = chain.BondingCurveState(
            virtual_token_reserves=int(row.get("virtual_token_reserves") or 0),
            virtual_sol_reserves=int(row.get("virtual_sol_reserves") or 0),
            real_token_reserves=int(row.get("real_token_reserves") or 0),
            real_sol_reserves=int(row.get("real_sol_reserves") or 0),
            token_total_supply=int(row.get("total_supply") or 0),
            complete=bool(row.get("complete")))
    except (TypeError, ValueError):
        return None
    if state.token_total_supply <= 0:
        return None
    pct, _basis = chain.curve_pct(state)
    if pct < FRONTEND_MIN_CURVE_PCT and not state.complete:
        return None
    return {
        "mint": row.get("mint", ""),
        "creator": row.get("creator", ""),
        "name": row.get("name", ""), "symbol": row.get("symbol", ""),
        "uri": row.get("metadata_uri", ""),
        "bonding_curve": row.get("bonding_curve", ""),
        # the coin row declares the token's socials + platform
        # comment count — the attention verifier reads both from
        # the launch ledger (recurrence + comment velocity)
        "twitter": row.get("twitter", "") or "",
        "telegram": row.get("telegram", "") or "",
        "website": row.get("website", "") or "",
        "reply_count": int(row.get("reply_count") or 0),
        "created_ts": (float(row.get("created_timestamp") or 0) / 1000.0),
        "source": "frontend-api",
        "curve_state": state.to_dict(),
        "curve_pct_seen": pct,
    }


def fetch_frontend(http_get: Callable, limit: int = 50,
                   sort: str = "market_cap") -> list[dict[str, Any]]:
    """Recent coins from the frontend API (backfill/discovery).
    ``http_get(url, headers) -> (status, text)`` — the house shape.
    The UA resolves through the desk's ONE config surface
    (sources.desk_user_agent), never a per-call literal."""
    from .sources import desk_user_agent
    url = (f"{FRONTEND_API}/coins?offset=0&limit={int(limit)}"
           f"&sort={sort}&order=DESC&includeNsfw=true")
    status, text = http_get(url, {"User-Agent": desk_user_agent()})
    if status != 200:
        raise IOError(f"frontend api answered HTTP {status}")
    rows = json.loads(text)
    return rows if isinstance(rows, list) else []


#: Pages of the market-cap list the single-coin lookup scans.
#: The frontend API has no by-mint GET (verified live: the detail
#: route 404s), so a coin row is found by scanning the list the
#: desk already polls — bounded, and a miss is reported as a miss.
COIN_SCAN_PAGES = 4


def fetch_coin_row(http_get: Callable, mint: str,
                   pages: int = COIN_SCAN_PAGES,
                   page_size: int = 50) -> Optional[dict[str, Any]]:
    """The frontend API's current coin row for one mint (reply
    count, declared socials, creation time), or None when the
    mint is not in the scanned pages."""
    from .sources import desk_user_agent
    for page in range(max(1, int(pages))):
        url = (f"{FRONTEND_API}/coins?offset={page * int(page_size)}"
               f"&limit={int(page_size)}&sort=market_cap&order=DESC"
               f"&includeNsfw=true")
        status, text = http_get(url, {"User-Agent": desk_user_agent()})
        if status != 200:
            raise IOError(f"frontend api answered HTTP {status}")
        rows = json.loads(text)
        if not isinstance(rows, list) or not rows:
            return None
        for row in rows:
            if isinstance(row, dict) and row.get("mint") == mint:
                return row
    return None


def _urllib_get(url: str, headers: dict,
                timeout: float = 20.0) -> tuple[int, str]:
    """The desk's plain-GET transport.  ``timeout`` is a hard
    ceiling the caller sizes from the source's cycle budget.

    A transport denial (builtin PermissionError — the OS refusing
    the connection, WSAEACCES on Windows — bare or wrapped as a
    URLError's reason) is re-raised as TransportDeniedError: the
    raw class name is the house taxonomy's POLICY denial, and a
    fetch-layer refusal must never be recorded under it."""
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except PermissionError as exc:
        raise TransportDeniedError(
            f"transport denied fetching {url}: {exc}") from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, PermissionError):
            raise TransportDeniedError(
                f"transport denied fetching {url}: "
                f"{exc.reason}") from exc
        raise


# ── backoff ───────────────────────────────────────────────────────────

def base_backoff(attempt: int) -> float:
    """Exponential reconnect schedule: 2, 4, 8, ... capped 300s.
    Jitter is applied by the supervisor, not here (test-pinnable)."""
    return min(300.0, 2.0 * (2 ** max(0, attempt)))


# ── stream backends ───────────────────────────────────────────────────

class _Control:
    """The consumer loop's control channel: watch-set changes and
    stop requests arrive here (thread-safe)."""

    def __init__(self):
        self.q: "queue.SimpleQueue[tuple]" = queue.SimpleQueue()

    def watch(self, mint: str, curve: str) -> None:
        self.q.put(("watch", mint, curve))

    def unwatch(self, mint: str) -> None:
        self.q.put(("unwatch", mint))

    def stop(self) -> None:
        self.q.put(("stop",))


def _default_ws_client(url: str) -> WebSocketClient:
    """The production socket factory: the desk's configured UA,
    resolved through the ONE config surface."""
    from .sources import desk_user_agent
    return WebSocketClient(url, user_agent=desk_user_agent())


class PumpPortalBackend:
    """PumpPortal subs: token creation + migration events (free),
    plus the metered trade streams when an API key is provisioned.

    The key (sources.pumpportal_api_key — the vault→env bridge,
    read per serve by the Firehose) rides the socket URL as
    ``?api-key=`` — PumpPortal's own convention for its keyed
    endpoints.  Keyless, the backend serves exactly the free
    subscriptions it always did; the key is never logged."""

    name = "pumpportal"
    domain = "pumpportal.fun"

    def __init__(self, client_factory: Callable[[str], Any] = None,
                 api_key: str = ""):
        self._factory = client_factory or _default_ws_client
        self.api_key = api_key or ""

    @property
    def ws_url(self) -> str:
        if self.api_key:
            return f"{PUMPPORTAL_WS}?api-key={self.api_key}"
        return PUMPPORTAL_WS

    def serve(self, emit: Emit, ctl: _Control,
              heartbeat_s: float = 120.0) -> None:
        ws = self._factory(self.ws_url)
        ws.connect()
        try:
            ws.send_text(json.dumps({"method": "subscribeNewToken"}))
            ws.send_text(json.dumps({"method": "subscribeMigration"}))
            last_msg = time.time()
            while True:
                try:
                    raw = ws.recv(timeout=1.0)
                    last_msg = time.time()
                except (TimeoutError, socket_timeout()):
                    raw = None
                # control channel
                stopped = False
                while True:
                    try:
                        cmd = ctl.q.get_nowait()
                    except queue.Empty:
                        break
                    if cmd[0] == "stop":
                        stopped = True
                if stopped:
                    return
                if raw is None:
                    if time.time() - last_msg > heartbeat_s:
                        raise WSClosed("heartbeat lost (pumpportal silent)")
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(msg, dict) or "message" in msg:
                    continue  # subscription acks carry no event
                tx = msg.get("txType")
                if tx == "create":
                    emit("launch", {
                        "mint": msg.get("mint", ""),
                        "creator": msg.get("traderPublicKey", ""),
                        "name": msg.get("name", ""),
                        "symbol": msg.get("symbol", ""),
                        "uri": msg.get("uri", ""),
                        "bonding_curve": msg.get("bondingCurveKey", ""),
                        "signature": msg.get("signature", ""),
                        "portal_initial_buy": msg.get("initialBuy"),
                        # declared socials ride the create payload —
                        # the attention verifier's recurrence leg
                        # reads them from the launch ledger
                        "twitter": msg.get("twitter", "") or "",
                        "telegram": msg.get("telegram", "") or "",
                        "website": msg.get("website", "") or "",
                        "source": "pumpportal", "ts": time.time(),
                    })
                elif tx in ("migrate", "migration"):
                    emit("migration", {"mint": msg.get("mint", ""),
                                       "source": "pumpportal",
                                       "ts": time.time()})
        finally:
            ws.close()


def socket_timeout():
    """socket.timeout is TimeoutError on modern Python; the tuple
    form keeps the except clause honest across versions."""
    import socket as _s
    return _s.timeout


class RpcLogBackend:
    """Public-RPC logsSubscribe: program-wide (creates floor) plus
    one subscription per watched mint's bonding curve (its trades).
    All events are stdlib Anchor decodes of ``Program data:`` lines.
    """

    name = "rpc-logs"
    domain = "api.mainnet-beta.solana.com"

    def __init__(self, rpc_http_endpoint: str,
                 client_factory: Callable[[str], Any] = None,
                 max_watched: int = 30):
        self.http_endpoint = rpc_http_endpoint
        self._factory = client_factory or _default_ws_client
        self.max_watched = max_watched
        self._watched: dict[str, str] = {}   # mint -> curve

    @property
    def ws_url(self) -> str:
        url = self.http_endpoint
        if url.startswith("https://"):
            return "wss://" + url[len("https://"):]
        if url.startswith("http://"):
            return "ws://" + url[len("http://"):]
        return url

    def serve(self, emit: Emit, ctl: _Control,
              heartbeat_s: float = 150.0) -> None:
        ws = self._factory(self.ws_url)
        ws.connect()
        sub_ids: dict[int, str] = {}       # subscription id -> role
        pending: dict[int, str] = {}       # request id -> role
        req = [0]

        def _send_sub(params: list, role: str) -> None:
            req[0] += 1
            pending[req[0]] = role
            ws.send_text(json.dumps({
                "jsonrpc": "2.0", "id": req[0],
                "method": "logsSubscribe", "params": params}))

        try:
            _send_sub([{"mentions": [chain.PUMP_PROGRAM]},
                       {"commitment": "confirmed"}], "program")
            for mint, curve in list(self._watched.items()):
                _send_sub([{"mentions": [curve]},
                           {"commitment": "confirmed"}], f"mint:{mint}")
            last_msg = time.time()
            last_ping = time.time()
            while True:
                try:
                    raw = ws.recv(timeout=1.0)
                    last_msg = time.time()
                except (TimeoutError, socket_timeout()):
                    raw = None
                while True:
                    try:
                        cmd = ctl.q.get_nowait()
                    except queue.Empty:
                        break
                    if cmd[0] == "stop":
                        return
                    if cmd[0] == "watch" and len(self._watched) < self.max_watched:
                        _, mint, curve = cmd
                        if mint not in self._watched:
                            self._watched[mint] = curve
                            _send_sub([{"mentions": [curve]},
                                       {"commitment": "confirmed"}],
                                      f"mint:{mint}")
                    if cmd[0] == "unwatch":
                        self._watched.pop(cmd[1], None)
                if time.time() - last_ping > 25.0:
                    ws.ping()
                    last_ping = time.time()
                if raw is None:
                    if time.time() - last_msg > heartbeat_s:
                        raise WSClosed("heartbeat lost (rpc silent)")
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if "id" in msg and "result" in msg:
                    role = pending.pop(int(msg["id"]), "")
                    if role:
                        sub_ids[int(msg["result"])] = role
                    continue
                if msg.get("method") != "logsNotification":
                    continue
                params = msg.get("params") or {}
                role = sub_ids.get(int(params.get("subscription") or -1), "")
                value = (params.get("result") or {}).get("value") or {}
                slot = int((params.get("result") or {}).get("context", {})
                           .get("slot") or 0)
                signature = value.get("signature", "")
                if value.get("err"):
                    continue
                events = chain.decode_log_lines(value.get("logs") or [])
                for ev in events:
                    kind = ev.get("event")
                    if kind == "CreateEvent":
                        emit("launch", {
                            "mint": ev.get("mint", ""),
                            "creator": ev.get("creator", ""),
                            "name": ev.get("name", ""),
                            "symbol": ev.get("symbol", ""),
                            "uri": ev.get("uri", ""),
                            "bonding_curve": ev.get("bonding_curve", ""),
                            "signature": signature, "slot": slot,
                            "source": "rpc-logs", "ts": time.time()})
                    elif kind == "TradeEvent":
                        if role.startswith("mint:") or \
                                ev.get("mint") in self._watched:
                            ev2 = dict(ev)
                            ev2["signature"] = signature
                            ev2["slot"] = slot
                            ev2["source"] = "rpc-logs"
                            emit("trade", ev2)
                    elif kind == "CompleteEvent":
                        emit("complete", {"mint": ev.get("mint", ""),
                                          "signature": signature,
                                          "slot": slot,
                                          "source": "rpc-logs",
                                          "ts": time.time()})
                    elif kind == "CompletePumpAmmMigrationEvent":
                        emit("migration_raw", {
                            "signature": signature, "slot": slot,
                            "source": "rpc-logs", "ts": time.time(),
                            "logs": value.get("logs") or []})
        finally:
            ws.close()


# ── supervisor ────────────────────────────────────────────────────────

class Firehose:
    """Owns the active backend; fails over in registry-health order.

    The consumer runs in a daemon thread (``start``) under the
    runtime, or synchronously in bounded slices for jobs/tests
    (``run_slice``).  Outcomes land in the ONE registry: a connect +
    first message is a success; an abnormal drop is a failure with
    its class — the health loop (stand-down 3 / disable 10) decides
    what gets skipped next round."""

    #: Grace a slice worker gets beyond its slice window to honour
    #: the stop signal before it is declared stuck and abandoned.
    SLICE_GRACE_S = 5.0

    def __init__(self, emit: Emit, registry: Any = None,
                 rpc_http_endpoint: str =
                 "https://api.mainnet-beta.solana.com",
                 client_factory: Callable[[str], Any] = None,
                 jitter: Callable[[], float] = None):
        self.emit = emit
        self.registry = registry
        self.rpc_http_endpoint = rpc_http_endpoint
        self._factory = client_factory
        self._jitter = jitter or (lambda: random.uniform(0.5, 1.0))
        self.ctl = _Control()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.stats = {"connects": 0, "drops": 0, "events": 0,
                      "backend": "", "last_event_at": 0.0,
                      "last_error": ""}
        #: every worker thread this firehose has spawned and not
        #: yet reaped (the standing consumer + slice workers).
        #: Python cannot kill a thread, so a worker that ignores
        #: its stop signal is ABANDONED — daemon, signalled, never
        #: awaited or reused — and stays visible here until it
        #: actually exits, so a leak is a number, not a rumour.
        self._workers: set[threading.Thread] = set()
        self._workers_lock = threading.Lock()

    # ── worker accounting ─────────────────────────────────────────

    def _track_worker(self, t: threading.Thread) -> None:
        with self._workers_lock:
            self._workers.add(t)

    def _prune_workers(self) -> None:
        with self._workers_lock:
            self._workers = {t for t in self._workers if t.is_alive()}

    def live_workers(self) -> int:
        """Worker threads currently alive (dead ones are reaped
        from the ledger on every read).  A healthy desk returns to
        its baseline after every cycle — the leak test pins it."""
        self._prune_workers()
        with self._workers_lock:
            return len(self._workers)

    def _backends(self) -> list:
        from .sources import pumpportal_api_key
        out = []
        pp = PumpPortalBackend(self._factory,
                               api_key=pumpportal_api_key())
        rpc = RpcLogBackend(self.rpc_http_endpoint, self._factory)
        rpc._watched = getattr(self, "_watched_seed", {})
        for backend in (pp, rpc):
            if self.registry is not None:
                desc = self.registry.for_domain(backend.domain)
                if desc is not None and not self.registry.is_active(desc):
                    continue  # stood-down/disabled: the loop has spoken
            out.append(backend)
        return out

    def _drain_stops(self) -> None:
        """Drop stale stop commands (a bounded slice's timer may
        leave one behind); watch commands are preserved."""
        keep = []
        while True:
            try:
                cmd = self.ctl.q.get_nowait()
            except queue.Empty:
                break
            if cmd[0] != "stop":
                keep.append(cmd)
        for cmd in keep:
            self.ctl.q.put(cmd)

    def _record(self, domain: str, ok: bool, error_class: str = "") -> None:
        if self.registry is None:
            return
        desc = self.registry.for_domain(domain)
        if desc is not None:
            self.registry.record_outcome(desc, ok, error_class=error_class)

    def run_slice(self, seconds: float = 20.0) -> dict[str, Any]:
        """One bounded consume window on the best healthy backend.
        Returns a report (never raises): the job-cycle shape.

        The serve loop runs in a supervised worker thread, joined
        with the slice window plus a grace period.  A worker that
        does not honour the stop signal inside that bound is
        declared stuck: the failure is recorded
        (``worker-stuck``), stop is signalled again, and the
        daemon worker is ABANDONED — never awaited, never reused;
        the next slice spawns a fresh worker.  A stuck fetch can
        therefore cost one worker, never the desk (pre-fix, the
        serve loop ran on the caller's thread and a wedged
        backend wedged the whole watch job with it)."""
        backends = self._backends()
        if not backends:
            return {"ok": False, "detail": "all intake backends are "
                    "stood down or disabled in the registry"}
        backend = backends[0]
        self.stats["backend"] = backend.name
        self._drain_stops()
        got = {"n": 0}

        def _emit(kind: str, payload: dict) -> None:
            got["n"] += 1
            self.stats["events"] += 1
            self.stats["last_event_at"] = time.time()
            self.emit(kind, payload)

        timer = threading.Timer(seconds, self.ctl.stop)
        timer.daemon = True
        timer.start()
        outcome: dict[str, Any] = {}

        def _serve() -> None:
            try:
                backend.serve(_emit, self.ctl)
                outcome["ok"] = True
            except Exception as exc:  # noqa: BLE001 - a drop is data
                outcome["error"] = exc

        worker = threading.Thread(target=_serve, daemon=True,
                                  name="pumpdesk-firehose-slice")
        self._track_worker(worker)
        worker.start()
        try:
            worker.join(timeout=max(0.1, float(seconds))
                        + self.SLICE_GRACE_S)
            if worker.is_alive():
                self.ctl.stop()
                self.stats["drops"] += 1
                self.stats["last_error"] = (
                    "WorkerStuck: backend serve ignored its stop "
                    "signal past the slice budget; worker abandoned")
                self._record(backend.domain, False,
                             error_class="worker-stuck")
                return {"ok": False, "backend": backend.name,
                        "events": got["n"],
                        "detail": self.stats["last_error"],
                        "worker_stuck": True}
            if "error" in outcome:
                exc = outcome["error"]
                self.stats["drops"] += 1
                self.stats["last_error"] = \
                    f"{type(exc).__name__}: {exc}"
                self._record(backend.domain, False,
                             error_class=type(exc).__name__)
                return {"ok": False, "backend": backend.name,
                        "events": got["n"],
                        "detail": self.stats["last_error"]}
            self._record(backend.domain, True)
            return {"ok": True, "backend": backend.name,
                    "events": got["n"]}
        finally:
            timer.cancel()

    def start(self) -> bool:
        """Run the supervised consumer in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()

        def _loop() -> None:
            attempt = 0
            while not self._stop.is_set():
                backends = self._backends()
                if not backends:
                    self._stop.wait(60.0)
                    attempt = 0
                    continue
                backend = backends[attempt % len(backends)]
                self.stats["backend"] = backend.name
                self.stats["connects"] += 1
                self._drain_stops()
                try:
                    backend.serve(self._counting_emit, self.ctl)
                    attempt = 0
                except Exception as exc:  # noqa: BLE001
                    self.stats["drops"] += 1
                    self.stats["last_error"] = (
                        f"{type(exc).__name__}: {exc}")
                    self._record(backend.domain, False,
                                 error_class=type(exc).__name__)
                    attempt += 1
                    delay = base_backoff(min(attempt, 8)) * self._jitter()
                    self._stop.wait(delay)
                else:
                    self._record(backend.domain, True)

        self._thread = threading.Thread(target=_loop, daemon=True,
                                        name="pumpdesk-firehose")
        self._track_worker(self._thread)
        self._thread.start()
        return True

    def _counting_emit(self, kind: str, payload: dict) -> None:
        self.stats["events"] += 1
        self.stats["last_event_at"] = time.time()
        self.emit(kind, payload)

    def watch(self, mint: str, curve: str) -> None:
        self._watched_seed = getattr(self, "_watched_seed", {})
        self._watched_seed[mint] = curve
        self.ctl.watch(mint, curve)

    def stop(self) -> None:
        self._stop.set()
        self.ctl.stop()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
