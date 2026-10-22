"""The betting desk — Devon's standing betting-study organ.

The desk EXTENDS the sports analyst (``agents/sports_bet.py``, the
ensemble that answers one-off "analyze this match" questions) with
the things a desk does that an analyst doesn't:

* **continuous study** — scheduled cycles that collect crash rounds,
  re-run the pattern lab, refresh the Library, and scan the sports
  analyst's fixture feed for value (analysis only);
* **a crash pattern lab** (:mod:`.lab`) testing the owner's time
  hypothesis and its neighbours, per game, with verdicts in writing;
* **signal calls** (:mod:`.signals`) — timed alerts with >= 2
  minutes of lead, issued only off lab-confirmed time structure;
* **a Library** — one structured record per (provider, game): sample
  sizes, the tests run, the verdict with its evidence, and what
  would change it.  Findings persist across restarts and cycles;
* **self-provisioning** — its keyed data services live in the real
  provisioning registry (:mod:`.sources`), and its data sources in
  the shared source registry, kind ``betting``.

Doctrine (the owner's, non-negotiable): the desk studies, signals,
proposes.  It contains NO bet-placement path and never will — real
money moves only on the owner's explicit word, outside this code.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from . import lab as _lab
from .collectors import SEED_PROVIDERS, collect_all, import_rounds_file
from .crash import CrashStore, betdesk_dir
from .signals import SignalEngine

__all__ = ["BettingDesk", "render_status", "render_library"]


class BettingDesk:
    """One desk instance.  ``context`` is the runtime context when
    bound (Notifier delivery); CLI/tests run context-free."""

    def __init__(self, context: Any = None, desk_dir: str = "",
                 store: Optional[CrashStore] = None,
                 notifier: Any = None):
        self.context = context
        self.dir = desk_dir or betdesk_dir()
        os.makedirs(self.dir, exist_ok=True)
        self.store = store or CrashStore(self.dir)
        if notifier is None and context is not None:
            try:
                from ..notifier import Notifier
                notifier = Notifier(context)
            except Exception:  # noqa: BLE001 - delivery is optional kit
                notifier = None
        self.signals = SignalEngine(self.dir, notifier=notifier)
        self._registry = None

    # ── shared source registry (lazy; the ONE registry) ──────────────

    @property
    def registry(self) -> Any:
        if self._registry is None:
            from ...media.source_registry import DEFAULT_REGISTRY
            from .sources import ensure_betting_sources
            self._registry = DEFAULT_REGISTRY
            ensure_betting_sources(self._registry)
        return self._registry

    # ── library ───────────────────────────────────────────────────────

    def _library_path(self) -> str:
        return os.path.join(self.dir, "library.json")

    def library(self) -> dict[str, Any]:
        """The desk Library: per-game study records, keyed
        ``provider/game``."""
        if os.path.exists(self._library_path()):
            try:
                with open(self._library_path(), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError):
                pass
        return {}

    def _save_library(self, lib: dict[str, Any]) -> None:
        tmp = self._library_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(lib, fh, indent=1)
        os.replace(tmp, self._library_path())

    # ── study cycle ───────────────────────────────────────────────────

    def analyze_all(self) -> dict[str, Any]:
        """Run the lab over every game in the store; refresh Library."""
        lib = self.library()
        verdicts: dict[str, Any] = {}
        for provider, game, _n in self.store.games():
            rounds = self.store.rounds(provider, game)
            spec_edge = 0.01
            for spec in SEED_PROVIDERS:
                if spec.id == provider and spec.game == game:
                    spec_edge = spec.house_edge
                    break
            v = _lab.analyze_game(rounds, provider=provider, game=game,
                                  house_edge=spec_edge)
            key = f"{provider}/{game}"
            verdicts[key] = v.to_dict()
            rec = lib.get(key, {})
            rec.update(v.to_dict())
            rec["updated_at"] = time.time()
            rec["chain"] = _lab.verify_hash_chain(rounds)
            lib[key] = rec
        self._save_library(lib)
        return verdicts

    def study(self, deliver_signals: bool = False) -> dict[str, Any]:
        """One full desk cycle: collect -> import inbox -> analyze ->
        (optionally) call signals -> sports scan.  Every leg reports
        its own outcome; a dead leg never kills the cycle."""
        out: dict[str, Any] = {"at": time.time()}
        # 1. collect from providers (registry outcomes recorded inside)
        try:
            out["collect"] = collect_all(self.store, registry=self.registry)
        except Exception as exc:  # noqa: BLE001 - a leg reports, not raises
            out["collect"] = {"error": f"{type(exc).__name__}: {exc}"}
        # 2. import anything the owner dropped in the inbox
        out["imports"] = self.import_inbox()
        # 3. lab + library
        out["verdicts"] = self.analyze_all()
        # 4. signals off the fresh verdicts
        out["signals"] = self.run_signals(deliver=deliver_signals)
        # 5. sports leg: the analyst's fixture scan (analysis only)
        out["sports"] = self.sports_scan()
        return out

    def import_inbox(self) -> list[dict[str, Any]]:
        """Import CSV/JSONL files dropped in ``<desk>/inbox/``.

        File name convention: ``<provider>__<game>.csv`` (or .jsonl).
        Imported files move to ``inbox/done/`` — a file is studied
        once, never re-read every cycle.
        """
        inbox = os.path.join(self.dir, "inbox")
        if not os.path.isdir(inbox):
            return []
        done_dir = os.path.join(inbox, "done")
        results = []
        for name in sorted(os.listdir(inbox)):
            path = os.path.join(inbox, name)
            if not os.path.isfile(path):
                continue
            stem = name.rsplit(".", 1)[0]
            parts = stem.split("__", 1)
            if len(parts) != 2:
                results.append({"file": name, "ok": False,
                                "detail": "name must be "
                                          "<provider>__<game>.csv"})
                continue
            try:
                res = import_rounds_file(self.store, path, parts[0],
                                         parts[1])
                res["file"] = name
                if res.get("ok"):
                    os.makedirs(done_dir, exist_ok=True)
                    os.replace(path, os.path.join(done_dir, name))
                results.append(res)
            except Exception as exc:  # noqa: BLE001
                results.append({"file": name, "ok": False,
                                "detail": f"{type(exc).__name__}: {exc}"})
        return results

    # ── signals ───────────────────────────────────────────────────────

    def run_signals(self, deliver: bool = True) -> dict[str, Any]:
        """Evaluate every Library verdict for a call; publish the
        callable ones; score any due calls.  No callable verdict ->
        an explicit no-call report, never silence."""
        lib = self.library()
        calls, reasons = [], []
        for key, rec in sorted(lib.items()):
            v = _verdict_from_record(rec)
            decision = self.signals.evaluate(v)
            if decision["call"] is not None:
                calls.append(self.signals.publish(decision["call"],
                                                  deliver=deliver))
            else:
                reasons.append(decision["reason"])
        return {"published": calls, "no_call_reasons": reasons,
                "scoring": self.signals.score_due(self.store)}

    # ── sports leg (the existing analyst, driven continuously) ───────

    def sports_scan(self, top_n: int = 3) -> dict[str, Any]:
        """The analyst's fixture digest as a desk leg: upcoming
        fixtures across the majors, value-ranked.  Fail-soft — the
        digest already renders its own honest error string when the
        feed is unreachable."""
        try:
            from ..sports_bet import BetStore, fixture_digest
            digest = fixture_digest(BetStore(), league_text="",
                                    top_n=top_n)
            return {"ok": True, "digest": digest}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False,
                    "detail": f"{type(exc).__name__}: {exc}"}

    # ── status ────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        from .sources import betting_status, ensure_provisioning
        ensure_provisioning()
        lib = self.library()
        games = []
        for provider, game, n in self.store.games():
            rec = lib.get(f"{provider}/{game}", {})
            games.append({
                "provider": provider, "game": game, "rounds": n,
                "verdict": rec.get("verdict", "NOT_STUDIED"),
                "callable_structure": rec.get("callable_structure", ""),
                "evidence": rec.get("evidence", ""),
            })
        scoring = self.signals.score_due(self.store)
        try:
            sources = betting_status(self.registry)
        except Exception as exc:  # noqa: BLE001
            sources = {"error": f"{type(exc).__name__}: {exc}"}
        return {
            "rounds_total": self.store.count(),
            "games": games,
            "signals": scoring,
            "min_lead_seconds": self.signals.min_lead,
            "sources": sources,
            "provisioning": _provisioning_view(),
            "library_records": len(lib),
        }


def _verdict_from_record(rec: dict[str, Any]) -> Any:
    """Rebuild the lab verdict fields the signal engine reads."""
    from types import SimpleNamespace
    v = SimpleNamespace(
        provider=rec.get("provider", ""), game=rec.get("game", ""),
        verdict=rec.get("verdict", "INSUFFICIENT_DATA"),
        callable_structure=rec.get("callable_structure", ""),
        best_hours=list(rec.get("best_hours", [])),
        hour_rates={int(k): float(vv)
                    for k, vv in (rec.get("hour_rates") or {}).items()},
        base_success_rate=float(rec.get("base_success_rate", 0.0)),
        target=float(rec.get("target", _lab.DEFAULT_TARGET)),
        evidence=rec.get("evidence", ""))
    return v


def _provisioning_view() -> list[dict[str, Any]]:
    """Provisioning state for the desk's keyed services — presence
    only, never values (the provisioning module's own audit rule)."""
    import os as _os
    out = []
    for service, env_key in (("odds-api", "ODDS_API_KEY"),
                             ("football-data", "FOOTBALL_DATA_API_KEY")):
        out.append({"service": service, "env_key": env_key,
                    "key_present": bool(_os.environ.get(env_key, "").strip())})
    return out


# ── rendering ──────────────────────────────────────────────────────────

def render_status(st: dict[str, Any]) -> str:
    L = [f"🃏 betting desk — {st.get('rounds_total', 0)} crash rounds "
         f"in the ledger, {st.get('library_records', 0)} library records"]
    for g in st.get("games", []):
        L.append(f"  {g['provider']}/{g['game']}: {g['rounds']} rounds — "
                 f"{g['verdict']}"
                 + (f" (callable: {g['callable_structure']})"
                    if g.get("callable_structure") else ""))
        if g.get("evidence"):
            L.append(f"    {g['evidence'][:220]}")
    sig = st.get("signals", {})
    if sig.get("calls_total"):
        L.append(f"signals: {sig['calls_scored']}/{sig['calls_total']} "
                 f"calls scored, hit rate "
                 f"{(sig.get('hit_rate') or 0):.0%}")
    else:
        L.append("signals: no calls published yet (nothing has validated "
                 "as callable — see /desk library)")
    L.append(f"signal lead floor: {st.get('min_lead_seconds', 120)}s")
    for p in st.get("provisioning", []):
        L.append(f"key {p['env_key']}: "
                 f"{'present' if p['key_present'] else 'MISSING — provision ' + p['service']}")
    src = (st.get("sources") or {}).get("sources", [])
    if src:
        L.append("sources: " + ", ".join(
            f"{s['domain']} [{s['health']}]" for s in src))
    return "\n".join(L)


def render_library(lib: dict[str, Any]) -> str:
    if not lib:
        return ("the desk library is empty — no game has been studied "
                "yet (collect rounds first: /desk study)")
    blocks = []
    for key, rec in sorted(lib.items()):
        lines = [f"📚 {key} — {rec.get('verdict', '?')} "
                 f"({rec.get('n_rounds', 0)} rounds, "
                 f"{rec.get('n_timed', 0)} timestamped)"]
        if rec.get("evidence"):
            lines.append(f"  evidence: {rec['evidence'][:400]}")
        if rec.get("what_would_change"):
            lines.append(f"  what would change it: "
                         f"{rec['what_would_change'][:200]}")
        for t in rec.get("tests", []):
            if t.get("skipped"):
                lines.append(f"  - {t['name']}: skipped ({t['skipped']})")
            else:
                lines.append(f"  - {t['name']}: p={t['p_value']:.3g} — "
                             f"{t.get('detail', '')[:120]}")
        chain = rec.get("chain") or {}
        if chain.get("checked"):
            lines.append(f"  provably-fair chain: {chain['ok']}/"
                         f"{chain['checked']} links ok")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
