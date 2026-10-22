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
                 registry: Any = None):
        self.context = context
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
            from .sources import rpc_endpoint_from_env
            endpoint, _src = rpc_endpoint_from_env()
            reg = self.registry

            def _outcome(ok: bool, latency: float, error_class: str = "",
                         _endpoint: str = endpoint) -> None:
                from urllib.parse import urlparse
                host = urlparse(_endpoint).hostname or ""
                desc = reg.for_domain(host)
                if desc is not None:
                    reg.record_outcome(desc, ok, latency_s=latency,
                                       error_class=error_class)

            self._rpc = chain.SolanaRPC(endpoint, on_outcome=_outcome)
        return self._rpc

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
        # 1. frontend candidate poll (backfill/discovery)
        try:
            rows = _firehose.fetch_frontend(_firehose._urllib_get,
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
            desc = self.registry.for_domain("frontend-api-v3.pump.fun")
            if desc is not None:
                self.registry.record_outcome(desc, True)
        except Exception as exc:  # noqa: BLE001 - a leg reports
            out["frontend"] = {"error": f"{type(exc).__name__}: {exc}"}
            desc = self.registry.for_domain("frontend-api-v3.pump.fun")
            if desc is not None:
                self.registry.record_outcome(
                    desc, False, error_class=type(exc).__name__)
        # 2. bounded stream slice (jobs/tests); the runtime's thread
        #    consumer is the standing intake — a slice here also
        #    serves hosts where the thread never started.
        if stream_seconds > 0:
            out["stream"] = self.firehose.run_slice(stream_seconds)
        # 3. refresh + evaluate every dossiered mint; publish calls
        calls, reasons = [], []
        for mint in sorted(self.store.dossiers()):
            d = self.refresh_dossier(mint)
            verdict = _score.evaluate(
                d, smart_wallets=self.store.watched_smart())
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
        # 5. module health → alert policy (hints, Pass-7 discipline)
        out["health"] = self._health_report()
        return out

    def _already_called(self, mint: str) -> bool:
        now = time.time()
        return any(c.mint == mint and c.expires_at > now
                   for c in self.calls.log.calls())

    def study_cycle(self) -> dict[str, Any]:
        """The slower cycle: grade due calls, recompute dev
        profiles, classify launch outcomes — the desk learning from
        its own journal (weight recalibration is journaled here as a
        proposal record, never silent drift: design §4.4.6)."""
        out: dict[str, Any] = {"at": time.time()}
        out["scoring"] = self.calls.score_due(
            self.store, snapshot_fn=_dex_snapshot)
        profiles = _dossier.refresh_dev_profiles(self.store)
        out["dev_profiles"] = len(profiles)
        out["serial_ruggers"] = sum(1 for p in profiles.values()
                                    if p.get("serial_rugger"))
        for p in profiles.values():
            if p.get("serial_rugger"):
                self.store.watch_dev(p["creator"],
                                     note="auto: serial rugger")
        return out

    # ── health (Pass-7 alert discipline) ──────────────────────────

    def _health_report(self) -> dict[str, Any]:
        """Feed health from the registry; a newly-dead backend is
        published WITH hints — the policy decides PAGE vs DIGEST;
        closures/recurrence are the policy's business, not ours."""
        from .sources import pump_status
        st = pump_status(self.registry)
        dead = [s for s in st["sources"]
                if s["health"] in ("disabled", "stood-down", "dead")]
        reported = []
        if dead and self.notifier is not None:
            for s in dead:
                try:
                    res = self.notifier.publish(
                        "pumpdesk",
                        f"pump desk feed {s['health']}: {s['domain']}",
                        f"the {s['name']} intake backend is "
                        f"{s['health']} in the source registry — the "
                        f"desk is running on its remaining feeds. "
                        f"stats: {s['stats']}",
                        alert_subject=f"pumpdesk feed {s['domain']}",
                        alert_hints={"failed": True,
                                     "subject": s["domain"]})
                    reported.append({"domain": s["domain"],
                                     "tier": res.get("tier", "")})
                except Exception:  # noqa: BLE001 - alerting never sinks
                    pass
        return {"sources": st["sources"], "dead": dead,
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
        return {
            "launches_seen": len(self.store.launches()),
            "dossiers": len(self.store.dossiers()),
            "dev_profiles": len(self.store.dev_profiles()),
            "serial_ruggers": sum(
                1 for p in self.store.dev_profiles().values()
                if p.get("serial_rugger")),
            "calls": calibration(calls),
            "sources": pump_status(self.registry),
            "firehose": (self.firehose.stats
                         if self._firehose is not None else {}),
            "execution": "none — signals only (P1); no wallet, no "
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
    src = (st.get("sources") or {}).get("sources", [])
    if src:
        L.append("sources: " + ", ".join(
            f"{s['domain']} [{s['health']}]" for s in src))
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
