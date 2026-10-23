"""The pump desk — Devon's standing pump.fun intelligence organ.

Mirrors the betting desk's shape (design §2.1): a context-bound
facade over the firehose intake, the dossier builder, the de-rug
gate, and the self-scoring call ledger, with its feeds on the ONE
source registry and its jobs in the scheduler's desk idiom.

Doctrine (the owner's, non-negotiable): the desk studies, scores,
and signals.  It contains NO buying or selling machinery, no
wallet keypair handling, nothing that signs a transaction —
execution is a separate owner word and a separate design (P3).
The desk's edge is information quality and an auditable track
record, never speed: it does not pretend to win block-0 against
co-located snipers (design §4).
"""

from __future__ import annotations

import time
from typing import Any, Optional

from . import chain, dossier as _dossier, firehose as _firehose
from . import graph as _graph, learn as _learn, shelf as _shelf
from . import outcomes as _outcomes, score as _score, watch as _watch
from .calls import CallEngine, calibration, render_call
from .store import PumpStore, pumpdesk_dir

__all__ = ["PumpDesk", "render_status", "render_calls"]


class PumpDesk:
    """One desk instance.  ``context`` binds the Notifier (call
    delivery + health alerts) and the registry's KV store; CLI and
    tests run context-free with an in-memory registry."""

    _streams: dict[str, Any] = {}   # desk dir -> Firehose (process-wide)

    def __init__(self, context: Any = None, desk_dir: str = "",
                 store: Optional[PumpStore] = None,
                 notifier: Any = None, rpc: Any = None,
                 registry: Any = None,
                 attention_search: Any = None,
                 attention_platform: Any = None,
                 attention_dex: Any = None):
        self.context = context
        self._attention_search = attention_search
        self._attention_platform = attention_platform
        self._attention_dex = attention_dex
        self.dir = desk_dir or pumpdesk_dir()
        self.store = store or PumpStore(self.dir)
        if notifier is None and context is not None:
            try:
                from ..notifier import Notifier
                notifier = Notifier(context)
            except Exception:  # noqa: BLE001 - delivery is optional kit
                notifier = None
        self.notifier = notifier
        self.calls = CallEngine(self.dir, notifier=notifier)
        self._registry = registry
        self._rpc = rpc
        self._firehose: Optional[_firehose.Firehose] = None
        self._prefilter = _firehose.Prefilter(
            self.store.watched_devs(), set(self.store.dossiers()))
        self._last_watch_at = 0.0

    # ── shared organs (lazy) ──────────────────────────────────────

    @property
    def registry(self) -> Any:
        if self._registry is None:
            from .sources import (ensure_pump_sources,
                                  pump_registry_for_context)
            self._registry = pump_registry_for_context(self.context)
            ensure_pump_sources(self._registry)
        return self._registry

    @property
    def rpc(self) -> Any:
        if self._rpc is None:
            from .sources import desk_source_budget_s, rpc_routes
            reg = self.registry

            def _outcome_for(domain: str) -> Any:
                def _outcome(ok: bool, latency: float,
                             error_class: str = "") -> None:
                    desc = reg.for_domain(domain)
                    if desc is not None:
                        reg.record_outcome(desc, ok, latency_s=latency,
                                           error_class=error_class)
                return _outcome

            clients = [
                chain.SolanaRPC(
                    route["endpoint"],
                    on_outcome=_outcome_for(route["domain"]),
                    budget_s=desk_source_budget_s())
                for route in rpc_routes(reg)
            ]
            self._rpc = chain.RoutedRPC(clients)
        return self._rpc

    # ── feed-source state (resilience pass) ───────────────────────

    #: Registry health values that mean a source is DOWN.
    _DOWN_HEALTH = ("stood-down", "disabled", "dead")
    #: Health values that mean a source is serving impaired.
    _IMPAIRED_HEALTH = ("bot-blocked", "rate-limited", "unverifiable")

    @staticmethod
    def _error_class_of(exc: BaseException) -> str:
        """The registry error class for a fetch failure: a
        transport denial records as ``transport-denied`` on every
        path (never the raw ``PermissionError`` — the house
        taxonomy's policy-denial name); everything else records
        its real class."""
        if isinstance(exc, chain.TransportDeniedError):
            return chain.TRANSPORT_DENIED_CLASS
        return type(exc).__name__

    def _record_source(self, domain: str, ok: bool,
                       error_class: str = "") -> None:
        """Record one real fetch outcome for a registry source by
        domain.  Guarded: registry trouble never sinks a cycle."""
        try:
            desc = self.registry.for_domain(domain)
            if desc is not None:
                self.registry.record_outcome(
                    desc, ok, error_class=error_class)
        except Exception:  # noqa: BLE001 - recording is advisory
            pass

    def _reset_source_budgets(self) -> None:
        """Fresh per-cycle time budget for every RPC route
        (duck-typed: injected RPC doubles without budgets are
        skipped)."""
        rpc = self._rpc
        if rpc is not None:
            reset = getattr(rpc, "reset_budgets", None)
            if callable(reset):
                reset()

    def _budgeted_http_get(self) -> Any:
        """An ``http_get(url, headers)`` closure whose TOTAL spend
        is capped at the frontend source's cycle budget — multi-
        page scans included.  Past the cap it fails fast with
        TimeoutError instead of paging a stalled source forever:
        the source degrades for this cycle, the desk moves on."""
        from .sources import desk_source_budget_s
        budget = desk_source_budget_s()
        spent = {"s": 0.0}

        def _get(url: str, headers: dict) -> tuple[int, str]:
            left = budget - spent["s"]
            if left <= 0:
                raise TimeoutError(
                    f"frontend source budget exhausted "
                    f"({budget:.0f}s this cycle) — the source "
                    f"degrades, the desk fails over")
            t0 = time.time()
            try:
                return _firehose._urllib_get(
                    url, headers, timeout=max(1.0, left))
            finally:
                spent["s"] += time.time() - t0

        return _get

    def _dex_http_get(self) -> Any:
        """The DexScreener transport with outcome recording: every
        real read lands in the ONE registry for
        api.dexscreener.com — success when the API answers 200
        (a 'no pair' answer included: the source SERVED), failure
        with its real class otherwise.  This is the recording the
        grader's reads never had, which is how DexScreener could
        sit at ok 0 / fail 0 / last_seen 0.0 while a root probe
        wore it as dead.  The transport timeout is the source's
        cycle budget."""
        from .sources import desk_source_budget_s

        def _get(url: str, headers: dict) -> tuple[int, str]:
            try:
                status, text = _outcomes.default_http_get(
                    url, headers,
                    timeout=max(1.0, desk_source_budget_s()))
            except Exception as exc:  # noqa: BLE001 - recorded, re-raised
                self._record_source(
                    "api.dexscreener.com", False,
                    error_class=self._error_class_of(exc))
                raise
            self._record_source(
                "api.dexscreener.com", status == 200,
                error_class="" if status == 200 else f"http-{status}")
            return status, text

        return _get

    def _dex_snapshot(self, mint: str) -> Optional[dict]:
        """Call-grading snapshot over the recording transport."""
        return _outcomes.token_snapshot(mint, self._dex_http_get())

    def _source_states(self) -> list[dict[str, Any]]:
        """Per-source feed state from the ONE registry, in the
        desk's vocabulary: ok / degraded / down / unprobed, each
        with its reason.  A source with zero recorded attempts is
        ``unprobed`` — never down (it has not failed; it has not
        been tried)."""
        from .sources import pump_status
        out: list[dict[str, Any]] = []
        for row in pump_status(self.registry)["sources"]:
            stats = row.get("stats") or {}
            attempts = sum(int(stats.get(k) or 0) for k in
                           ("ok", "fail", "blocked", "rate_limited",
                            "no_result"))
            health = row.get("health", "unknown")
            last_err = str(stats.get("last_error_class") or "")
            if health in self._DOWN_HEALTH:
                state = "down"
            elif attempts == 0:
                state = "unprobed"
            elif last_err or health in self._IMPAIRED_HEALTH:
                state = "degraded"
            else:
                state = "ok"
            reason = ""
            if state in ("down", "degraded"):
                reason = last_err or health
            out.append({
                "domain": row.get("domain", ""), "name": row.get("name", ""),
                "health": health, "active": row.get("active", True),
                "attempts": attempts, "ok": int(stats.get("ok") or 0),
                "fail": int(stats.get("fail") or 0),
                "last_error_class": last_err, "state": state,
                "reason": reason})
        return out

    def _desk_state(self, states: list[dict[str, Any]]) -> dict[str, Any]:
        """The desk-level feed state.  STAND-DOWN is explicit and
        named: at least one source is down AND nothing is serving
        (every source is down or never attempted) — the desk says
        so instead of emitting a misleading all-feeds blob or
        partial garbage.  Recovery is automatic: the state is
        recomputed from the registry on every cycle/status, so
        the first success anywhere lifts it."""
        down = [s for s in states if s["state"] == "down"]
        degraded = [s for s in states if s["state"] == "degraded"]
        serving = [s for s in states if s["state"] in ("ok", "degraded")]
        if down and not serving:
            name = "stand-down"
        elif down or degraded:
            name = "degraded"
        else:
            name = "ok"
        return {"state": name, "down": down, "degraded": degraded,
                "sources": states}

    def _journal_desk_state(self, desk_state: dict[str, Any],
                            at: float) -> None:
        """Journal desk feed-state TRANSITIONS to the store's
        health ledger (stand-down entered, recovered, degraded —
        each with the per-source reasons at the time).  Steady
        state journals nothing.  Never raises."""
        try:
            events = self.store.health_events()
            last = events[-1].get("state") if events else None
            current = desk_state["state"]
            if last == current:
                return
            self.store.record_health_event({
                "at": at, "state": current, "from": last,
                "sources": [
                    {"domain": s["domain"], "health": s["health"],
                     "state": s["state"], "reason": s["reason"]}
                    for s in desk_state["sources"]],
            })
        except Exception:  # noqa: BLE001 - journaling never sinks
            pass

    @property
    def firehose(self) -> _firehose.Firehose:
        if self._firehose is None:
            from .sources import rpc_endpoint_from_env
            endpoint, _ = rpc_endpoint_from_env()
            self._firehose = _firehose.Firehose(
                self._ingest, registry=self.registry,
                rpc_http_endpoint=endpoint)
            for mint in self.store.dossiers():
                # watched set = dossiered mints (their curve accounts)
                try:
                    self._firehose.watch(
                        mint, chain.bonding_curve_pda(mint))
                except Exception:  # noqa: BLE001
                    pass
        return self._firehose

    # ── intake: the firehose sink ─────────────────────────────────

    def _ingest(self, kind: str, payload: dict[str, Any]) -> None:
        """Firehose events land here.  Ledger first (cheap, always);
        dossier work only where the pre-filter says it is earned."""
        try:
            if kind == "launch":
                mint = payload.get("mint", "")
                if not mint:
                    return
                is_new = self.store.record_launch(payload)
                if is_new and self._prefilter.decide(payload) == "dossier":
                    self.refresh_dossier(mint)
            elif kind == "trade":
                self.store.record_trade(payload)
            elif kind == "migration":
                mint = payload.get("mint", "")
                if mint:
                    self.store.update_launch(mint, graduated=True)
                    self.refresh_dossier(mint)
            elif kind == "complete":
                mint = payload.get("mint", "")
                if mint and self.store.dossier(mint):
                    self.refresh_dossier(mint)
            elif kind == "migration_raw":
                self._resolve_migration_raw(payload)
        except Exception:  # noqa: BLE001 - one bad event never kills
            pass  # the firehose; the drop is visible in desk stats

    def _resolve_migration_raw(self, payload: dict[str, Any]) -> None:
        """An RPC-log migration event whose mint the decoder could
        not name: scan the raw log payload for a known mint's bytes
        (recent launches), then confirm by pool existence in the
        dossier refresh — the event alone never graduates a token."""
        import base64
        known = {l["mint"]: l for l in self.store.launches()[-2000:]
                 if l.get("mint")}
        if not known:
            return
        for line in payload.get("logs") or []:
            if "Program data:" not in line:
                continue
            try:
                raw = base64.b64decode(
                    line.split("Program data:", 1)[1].strip())
            except Exception:  # noqa: BLE001
                continue
            for mint in known:
                try:
                    if chain.b58decode(mint) in raw:
                        self.store.update_launch(mint, graduated=True)
                        self.refresh_dossier(mint)
                        return
                except ValueError:
                    continue

    # ── dossiers ──────────────────────────────────────────────────

    def refresh_dossier(self, mint: str) -> dict[str, Any]:
        """Build/refresh one dossier, then backfill its block-0
        trades from the chain when the create slot is known and the
        ledger has none yet (bounded: the create slot's signatures
        on the curve account only)."""
        d = _dossier.build_dossier(self.store, self.rpc, mint,
                                   now=time.time())
        launch = self.store.launch(mint) or {}
        # first-sight facts the outcome classifier + curve basis use
        state = (d.get("curve") or {}).get("state") or {}
        updates: dict[str, Any] = {}
        if d.get("curve") and not launch.get("birth_price"):
            updates["birth_price"] = d["curve"]["price_sol"] * 1e9
        if (state and int(state.get("real_sol_reserves") or 0) == 0
                and not launch.get("initial_real_token_reserves")):
            updates["initial_real_token_reserves"] = \
                state.get("real_token_reserves")
        if updates:
            self.store.update_launch(mint, **updates)
            launch = self.store.launch(mint) or launch
        slot = int(launch.get("slot") or 0)
        if not slot and launch.get("signature"):
            # PumpPortal names the create signature but not its slot;
            # the transaction itself carries the slot (one read).
            try:
                tx = self.rpc.get_transaction(launch["signature"])
                slot = int((tx or {}).get("slot") or 0)
                if slot:
                    self.store.update_launch(mint, slot=slot)
            except Exception:  # noqa: BLE001 - backfill is best-effort
                slot = 0
        if slot and not self.store.trades(mint):
            self._backfill_slot_trades(mint, slot)
            d = _dossier.build_dossier(self.store, self.rpc, mint,
                                       now=time.time())
        smart = _watch.smart_confluence(self.store, mint,
                                        self.store.watched_smart())
        if smart:
            d["smart_hits"] = smart
        # P2: the wallet graph's track-record confluence supersedes
        # the label count when the graph covers this mint's buyers
        graph = self.store.graph()
        if graph.get("wallets"):
            conf = _graph.graph_confluence(self.store, mint,
                                           graph=graph)
            if conf["buyers_seen"]:
                d["smart"] = conf
        if smart or d.get("smart"):
            self.store.save_dossier(d)
        return d

    def _backfill_slot_trades(self, mint: str, create_slot: int) -> int:
        """Decode the create slot's transactions on the curve
        account into the trade ledger (block-0 absorption evidence).
        Bounded to the first page of signatures; fail-soft."""
        try:
            curve = chain.bonding_curve_pda(mint)
            sigs = self.rpc.get_signatures(curve, limit=100)
        except Exception:  # noqa: BLE001
            return 0
        added = 0
        for row in sigs:
            if int(row.get("slot") or 0) != create_slot:
                continue
            if row.get("err"):
                continue
            try:
                tx = self.rpc.get_transaction(row.get("signature", ""))
            except Exception:  # noqa: BLE001
                continue
            logs = ((tx or {}).get("meta") or {}).get("logMessages") or []
            for ev in chain.decode_log_lines(logs):
                if ev.get("event") == "TradeEvent" and ev.get("mint") == mint:
                    ev["signature"] = row.get("signature", "")
                    ev["slot"] = create_slot
                    ev["source"] = "rpc-backfill"
                    self.store.record_trade(ev)
                    added += 1
        return added

    # ── cycles ────────────────────────────────────────────────────

    def watch_cycle(self, deliver: bool = True,
                    stream_seconds: float = 0.0) -> dict[str, Any]:
        """One watch cycle: intake (a bounded stream slice when
        asked, plus the frontend candidate poll), dossier refresh
        for everything the pre-filter promoted, gate evaluation,
        calls, and the health report.  Every leg reports its own
        outcome; a dead leg never kills the cycle."""
        out: dict[str, Any] = {"at": time.time()}
        self._reset_source_budgets()
        # 1. frontend candidate poll (backfill/discovery) — one
        #    source, one hard budget: the poll spends at most the
        #    source's cycle budget, so a wedged frontend degrades
        #    this leg only.
        try:
            rows = _firehose.fetch_frontend(self._budgeted_http_get(),
                                            limit=50)
            promoted = 0
            for row in rows:
                cand = _firehose.frontend_candidate(row)
                if cand is None:
                    continue
                mint = cand["mint"]
                if self.store.launch(mint) is None:
                    self.store.record_launch(cand)
                self.refresh_dossier(mint)
                promoted += 1
            out["frontend"] = {"rows": len(rows), "promoted": promoted}
            self._record_source("frontend-api-v3.pump.fun", True)
        except Exception as exc:  # noqa: BLE001 - a leg reports
            out["frontend"] = {"error": f"{type(exc).__name__}: {exc}"}
            self._record_source("frontend-api-v3.pump.fun", False,
                                error_class=self._error_class_of(exc))
        # 2. bounded stream slice (jobs/tests); the runtime's thread
        #    consumer is the standing intake — a slice here also
        #    serves hosts where the thread never started.
        if stream_seconds > 0:
            out["stream"] = self.firehose.run_slice(stream_seconds)
        # 3. refresh + evaluate every dossiered mint; publish calls
        calls, reasons = [], []
        params = _learn.current_params(self.store)
        for mint in sorted(self.store.dossiers()):
            d = self.refresh_dossier(mint)
            self._attention_check(d)
            verdict = _score.evaluate(
                d, smart_wallets=self.store.watched_smart(),
                params=params)
            if verdict.verdict == "CALL_CANDIDATE":
                call = self.calls.build_call(d, verdict)
                if call is not None and not self._already_called(mint):
                    calls.append(self.calls.publish(call,
                                                    deliver=deliver))
                elif call is not None:
                    reasons.append(f"{mint[:8]}: already called this "
                                   "window — the ledger does not repeat "
                                   "itself")
            else:
                reasons.append(f"{mint[:8]}: {verdict.verdict} — "
                               f"{verdict.evidence[:160]}")
            d["verdict"] = verdict.to_dict()
            self.store.save_dossier(d)
        out["calls"] = calls
        out["no_call_reasons"] = reasons
        # 4. watched-actor activity (digest-grade, reported not paged)
        out["watch_activity"] = _watch.watch_activity(
            self.store, since=self._last_watch_at)
        self._last_watch_at = out["at"]
        # 5. feed state: per-source degradation is named in the
        #    report; a total outage is the explicit stand-down
        #    state (journaled on transition), never a blob.
        states = self._source_states()
        desk_state = self._desk_state(states)
        self._journal_desk_state(desk_state, at=out["at"])
        out["sources"] = states
        out["degraded_sources"] = [
            s["domain"] for s in states
            if s["state"] in ("degraded", "down")]
        out["stand_down"] = (
            {"state": "stand-down",
             "reasons": {s["domain"]: (s["reason"] or s["health"])
                         for s in desk_state["down"]}}
            if desk_state["state"] == "stand-down" else None)
        # 6. module health → alert policy (hints, Pass-7 discipline)
        out["health"] = self._health_report(states)
        return out

    #: Attention verdicts are cached on the dossier this long
    #: before the verifier re-gathers off-chain evidence.
    ATTENTION_TTL_S = 1800.0

    def _attention_check(self, d: dict[str, Any],
                         force: bool = False) -> dict[str, Any]:
        """Run/refresh the attention verifier for one dossier
        (cached on the dossier; a fresh verdict rides into the
        score's manipulation component via dossier["attention"]).
        Only dossiers past the sample floors earn the search —
        attention evidence for a 3-trade token decides nothing."""
        from . import attention as _attention
        now = time.time()
        cached = d.get("attention") or {}
        if (not force and cached
                and now - float(cached.get("checked_at") or 0)
                < self.ATTENTION_TTL_S):
            return cached
        trades = d.get("trades") or {}
        if (int(trades.get("trades", 0)) < _score.MIN_TRADES
                and not force):
            return cached
        search_fn = self._attention_search
        if search_fn is None:
            search_fn = _attention.default_search_fn(self.context)
        # platform + DexScreener legs: explicit doubles win; the
        # live defaults bind only on a real runtime context, so
        # context-free runs (tests, offline CLI) keep the legs as
        # named gaps instead of reaching for the network.
        platform_fn = self._attention_platform
        if platform_fn is None and self.context is not None:
            platform_fn = _attention.default_platform_fn(
                self._budgeted_http_get())
        dex_fn = self._attention_dex
        if dex_fn is None and self.context is not None:
            dex_fn = _attention.default_dex_fn(self._dex_http_get())
        att = _attention.verify_attention(
            d, search_fn=search_fn, store=self.store, now=now,
            platform_fn=platform_fn, dex_fn=dex_fn)
        d["attention"] = att
        self.store.save_dossier(d)
        return att

    def _already_called(self, mint: str) -> bool:
        now = time.time()
        return any(c.mint == mint and c.expires_at > now
                   for c in self.calls.log.calls())

    def study_cycle(self) -> dict[str, Any]:
        """The slower cycle: grade due calls, recompute dev
        profiles, classify launch outcomes — the desk learning
        from its own journal.  P2 completes the loop: the wallet
        graph is rebuilt from the trade ledger, the scoring
        parameters recalibrate against the graded calls (every
        attempt journaled — adjust, hold, or INSUFFICIENT_DATA,
        never silent drift: design §4.4.6), and the dossiers and
        wallet records that earned it are shelved into the
        Library."""
        out: dict[str, Any] = {"at": time.time()}
        self._reset_source_budgets()
        out["scoring"] = self.calls.score_due(
            self.store, snapshot_fn=self._dex_snapshot)
        profiles = _dossier.refresh_dev_profiles(self.store)
        out["dev_profiles"] = len(profiles)
        out["serial_ruggers"] = sum(1 for p in profiles.values()
                                    if p.get("serial_rugger"))
        for p in profiles.values():
            if p.get("serial_rugger"):
                self.store.watch_dev(p["creator"],
                                     note="auto: serial rugger")
        graph = _graph.build_wallet_graph(self.store)
        out["graph"] = {
            "wallets": len(graph.get("wallets", {})),
            "graded": sum(1 for w in graph.get("wallets", {}).values()
                          if w.get("verdict") != "unknown"),
            "clusters": len(graph.get("clusters", [])),
        }
        out["learning"] = _learn.recalibrate(
            self.store, self.calls.log.calls())
        out["shelved"] = _shelf.shelve_notable(
            self.store, context=self.context)
        return out

    # ── P2 views ──────────────────────────────────────────────────

    def wallet_view(self, wallet: str) -> dict[str, Any]:
        rec = _graph.wallet_record(self.store, wallet)
        return {"wallet": wallet, "record": rec,
                "rendered": _graph.render_wallet(rec, wallet)}

    def graph_view(self) -> dict[str, Any]:
        graph = self.store.graph()
        return {"graph": {"wallets": len(graph.get("wallets", {})),
                          "clusters": graph.get("clusters", []),
                          "built_at": graph.get("built_at", 0.0)},
                "rendered": _graph.render_graph(graph)}

    def attention_view(self, mint: str) -> dict[str, Any]:
        from . import attention as _attention
        d = self.store.dossier(mint) or self.refresh_dossier(mint)
        att = self._attention_check(d, force=True)
        return {"mint": mint, "attention": att,
                "rendered": _attention.render_attention(att)}

    def learn_view(self) -> dict[str, Any]:
        return {"params": _learn.current_params(self.store),
                "events": self.store.param_events(),
                "rendered": _learn.render_learning(self.store)}

    def shelf_view(self, query: str) -> dict[str, Any]:
        hits = _shelf.search_shelf(self.store, query)
        return {"query": query, "hits": hits,
                "rendered": _shelf.render_shelf_hits(hits, query)}

    # ── health (Pass-7 alert discipline) ──────────────────────────

    def _health_report(self, states: Any = None) -> dict[str, Any]:
        """Feed health from the registry; a newly-dead backend is
        published WITH hints — the policy decides PAGE vs DIGEST;
        closures/recurrence are the policy's business, not ours.
        A total outage publishes as the desk's STAND-DOWN (one
        named state with per-source reasons), not as N separate
        feed deaths."""
        from .sources import pump_status
        st = pump_status(self.registry)
        if states is None:
            states = self._source_states()
        desk_state = self._desk_state(states)
        dead = [s for s in states if s["state"] == "down"]
        reported = []
        if self.notifier is not None and \
                desk_state["state"] == "stand-down":
            try:
                reasons = "; ".join(
                    f"{s['domain']}: {s['reason'] or s['health']}"
                    for s in dead)
                res = self.notifier.publish(
                    "pumpdesk",
                    "pump desk STAND-DOWN: all feeds down",
                    f"every feed source is down or unattempted — "
                    f"the desk has stood down and will recover "
                    f"automatically when any feed answers. "
                    f"per-source: {reasons}",
                    alert_subject="pumpdesk stand-down",
                    alert_hints={"failed": True,
                                 "subject": "pumpdesk stand-down"})
                reported.append({"domain": "stand-down",
                                 "tier": res.get("tier", "")})
            except Exception:  # noqa: BLE001 - alerting never sinks
                pass
        elif dead and self.notifier is not None:
            by_domain = {s["domain"]: s for s in st["sources"]}
            for s in dead:
                row = by_domain.get(s["domain"], {})
                try:
                    res = self.notifier.publish(
                        "pumpdesk",
                        f"pump desk feed {s['health']}: {s['domain']}",
                        f"the {s['name']} intake backend is "
                        f"{s['health']} in the source registry — the "
                        f"desk is running on its remaining feeds. "
                        f"stats: {row.get('stats', {})}",
                        alert_subject=f"pumpdesk feed {s['domain']}",
                        alert_hints={"failed": True,
                                     "subject": s["domain"]})
                    reported.append({"domain": s["domain"],
                                     "tier": res.get("tier", "")})
                except Exception:  # noqa: BLE001 - alerting never sinks
                    pass
        return {"sources": st["sources"], "dead": dead,
                "desk_state": desk_state["state"],
                "alerted": reported}

    # ── stream supervision ────────────────────────────────────────

    def ensure_stream(self) -> bool:
        """Start the standing firehose consumer (process-wide, one
        per desk dir).  Best-effort — the watch cycle's slices keep
        the desk fed where a thread cannot run."""
        existing = PumpDesk._streams.get(self.dir)
        if existing is not None:
            return False
        fh = self.firehose
        PumpDesk._streams[self.dir] = fh
        return fh.start()

    # ── status ────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        from .sources import pump_status
        calls = self.calls.log.calls()
        graph = self.store.graph()
        wallets = graph.get("wallets", {})
        states = self._source_states()
        desk_state = self._desk_state(states)
        firehose_stats: dict[str, Any] = {}
        if self._firehose is not None:
            firehose_stats = dict(self.firehose.stats)
            firehose_stats["workers"] = self.firehose.live_workers()
        return {
            "launches_seen": len(self.store.launches()),
            "dossiers": len(self.store.dossiers()),
            "dev_profiles": len(self.store.dev_profiles()),
            "serial_ruggers": sum(
                1 for p in self.store.dev_profiles().values()
                if p.get("serial_rugger")),
            "calls": calibration(calls),
            "graph": {
                "wallets": len(wallets),
                "smart": sum(1 for w in wallets.values()
                             if w.get("verdict") == "smart"),
                "losing": sum(1 for w in wallets.values()
                              if w.get("verdict") == "losing"),
                "clusters": len(graph.get("clusters", [])),
            },
            "learning": {
                "params": _learn.current_params(self.store),
                "events": len(self.store.param_events()),
            },
            "shelf_items": len(self.store.knowledge()),
            "sources": pump_status(self.registry),
            "desk_state": {
                "state": desk_state["state"],
                "degraded_sources": [
                    s["domain"] for s in states
                    if s["state"] in ("degraded", "down")],
                "down_reasons": {
                    s["domain"]: (s["reason"] or s["health"])
                    for s in desk_state["down"]},
                "sources": states,
            },
            "firehose": firehose_stats,
            "execution": "none — signals only (P2); no wallet, no "
                         "signing, no orders",
        }


def _dex_snapshot(mint: str) -> Optional[dict]:
    return _outcomes.token_snapshot(mint)


def render_status(st: dict[str, Any]) -> str:
    L = [f"🎰 pump desk — {st.get('launches_seen', 0)} launches seen, "
         f"{st.get('dossiers', 0)} dossiers, "
         f"{st.get('dev_profiles', 0)} dev profiles "
         f"({st.get('serial_ruggers', 0)} serial ruggers flagged)"]
    cal = st.get("calls") or {}
    if cal.get("calls_total"):
        hr = cal.get("hit_rate")
        L.append(f"calls: {cal['calls_scored']}/{cal['calls_total']} "
                 f"graded, hit rate "
                 f"{hr:.0%}" if hr is not None else
                 f"calls: {cal['calls_total']} published, none graded yet")
        L.append(f"  vs graduation base rate "
                 f"{cal.get('base_rate_graduation', 0):.1%} of all "
                 f"launches; no-data misses: "
                 f"{cal.get('misses_no_data', 0)}")
        for cls, row in (cal.get("by_class") or {}).items():
            L.append(f"  class {cls}: {row['wins']}/{row['calls']} "
                     f"({(row.get('hit_rate') or 0):.0%})")
    else:
        L.append("calls: none published yet — nothing has passed the "
                 "de-rug gate (that is the gate working, not silence)")
    fh = st.get("firehose") or {}
    if fh:
        L.append(f"firehose: backend {fh.get('backend') or 'idle'}, "
                 f"{fh.get('events', 0)} events, {fh.get('drops', 0)} "
                 f"drops")
    g = st.get("graph") or {}
    if g:
        L.append(f"wallet graph: {g.get('wallets', 0)} wallets "
                 f"({g.get('smart', 0)} smart, {g.get('losing', 0)} "
                 f"losing), {g.get('clusters', 0)} cohort(s)")
    learn = st.get("learning") or {}
    if learn:
        params = learn.get("params") or {}
        L.append(f"learning: parameter v{params.get('version', 1)}, "
                 f"{learn.get('events', 0)} recalibration record(s); "
                 f"shelf: {st.get('shelf_items', 0)} knowledge "
                 f"item(s)")
    src = (st.get("sources") or {}).get("sources", [])
    if src:
        L.append("sources: " + ", ".join(
            f"{s['domain']} [{s['health']}]" for s in src))
    ds = st.get("desk_state") or {}
    if ds.get("state") == "stand-down":
        reasons = ds.get("down_reasons") or {}
        detail = "; ".join(f"{d}: {r}" for d, r in reasons.items())
        L.append("⛔ STAND-DOWN — all feeds down; the desk is "
                 "serving nothing and will recover automatically "
                 "when any feed answers"
                 + (f" ({detail})" if detail else ""))
    elif ds.get("state") == "degraded":
        names = ", ".join(ds.get("degraded_sources") or [])
        L.append(f"⚠️ degraded feeds: {names} — the desk is "
                 f"serving from its remaining sources")
    L.append(str(st.get("execution", "")))
    return "\n".join(L)


def render_calls(calls: list[dict[str, Any]]) -> str:
    if not calls:
        return ("no calls in the ledger yet — the desk publishes only "
                "off a CALL_CANDIDATE verdict, and grades every call "
                "it makes")
    lines = []
    for c in calls[-20:]:
        state = ("pending" if c.get("scored") is None else
                 c.get("outcome", ""))
        lines.append(
            f"{c.get('symbol') or c.get('mint', '')[:8]}: "
            f"entry {c.get('entry_mc_sol', 0):.1f} SOL mcap → "
            f"target {c.get('target_mc_sol', 0):.1f} — {state}"
            + (f" (peak {c.get('max_favorable', 0)}x, via "
               f"{c.get('graded_via')})" if c.get("scored") is not None
               else ""))
    return "\n".join(lines)
