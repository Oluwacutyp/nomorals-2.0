"""Desk storage — the pump desk's memory between cycles.

Append-only JSONL ledgers (the house posture: appends never
rewrite history; dedupe happens on keys at read time) plus small
JSON state files:

* ``launches.jsonl`` — every create the firehose saw, one row per
  mint: the cheapest possible record (mint, creator, slot, name,
  the create-time curve snapshot when one was captured).  Thousands
  of rows a day is FINE — full dossiers are built only for
  candidates, which is the firehose's pre-filter discipline.
* ``trades.jsonl`` — decoded TradeEvents for watched mints only.
* ``observations.jsonl`` — curve snapshots per mint over time: the
  series call scoring reads for excursion / first-touch truth.
* ``dossiers.json`` — the current dossier per mint.
* ``dev_profiles.json`` — per-creator launch history + outcomes.
* ``watchlists.json`` — dev wallets + smart wallets the desk tracks.
* ``wallet_graph.json`` — the P2 smart-money graph: per-wallet
  realized-PnL records + cohort clusters, rebuilt from the trade
  ledger by the study cycle.
* ``score_params.json`` / ``score_params.jsonl`` — the current
  scoring parameter version + the append-only recalibration
  history (the learning loop's journal: every adjustment, and
  every refusal to adjust, is a record).
* ``knowledge.jsonl`` — the desk's Library shelf: dossiers and
  wallet records shelved as knowledge items with provenance
  (teachings-item shape), retrievable after the desk forgets the
  dossier itself.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

__all__ = ["pumpdesk_dir", "PumpStore"]


def pumpdesk_dir() -> str:
    """Desk data directory: ``NOMORALS_PUMPDESK_DIR`` or the config
    home (the betdesk_dir pattern)."""
    override = os.environ.get("NOMORALS_PUMPDESK_DIR", "").strip()
    if override:
        return override
    return os.path.join(os.path.expanduser("~"), ".config", "nomorals",
                        "pumpdesk")


def _append_jsonl(path: str, row: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


class PumpStore:
    def __init__(self, desk_dir: str = ""):
        self.dir = desk_dir or pumpdesk_dir()
        os.makedirs(self.dir, exist_ok=True)
        self._launches_path = os.path.join(self.dir, "launches.jsonl")
        self._trades_path = os.path.join(self.dir, "trades.jsonl")
        self._obs_path = os.path.join(self.dir, "observations.jsonl")
        self._dossiers_path = os.path.join(self.dir, "dossiers.json")
        self._devs_path = os.path.join(self.dir, "dev_profiles.json")
        self._watch_path = os.path.join(self.dir, "watchlists.json")
        self._graph_path = os.path.join(self.dir, "wallet_graph.json")
        self._params_path = os.path.join(self.dir, "score_params.json")
        self._params_log_path = os.path.join(self.dir,
                                             "score_params.jsonl")
        self._knowledge_path = os.path.join(self.dir, "knowledge.jsonl")
        self._health_log_path = os.path.join(self.dir,
                                             "health_events.jsonl")

    # ── launches ──────────────────────────────────────────────────

    def record_launch(self, launch: dict[str, Any]) -> bool:
        """Append a create record.  False when the mint is already
        recorded (a launch happens once; re-delivery is deduped)."""
        if self.launch(launch.get("mint", "")) is not None:
            return False
        _append_jsonl(self._launches_path, launch)
        return True

    def launches(self) -> list[dict[str, Any]]:
        seen: dict[str, dict[str, Any]] = {}
        for row in _read_jsonl(self._launches_path):
            mint = row.get("mint")
            if mint and mint not in seen:
                seen[mint] = row
        return list(seen.values())

    def launch(self, mint: str) -> Optional[dict[str, Any]]:
        for row in self.launches():
            if row.get("mint") == mint:
                return row
        return None

    def launches_by_creator(self, creator: str) -> list[dict[str, Any]]:
        return [r for r in self.launches() if r.get("creator") == creator]

    def update_launch(self, mint: str, **fields: Any) -> None:
        """Merge outcome fields into a launch row (graduated,
        outcome class).  Launches are few enough per creator to
        rewrite the file on outcome updates — appends stay the rule
        for the hot path."""
        rows = _read_jsonl(self._launches_path)
        changed = False
        for row in rows:
            if row.get("mint") == mint:
                row.update(fields)
                changed = True
        if changed:
            tmp = self._launches_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            os.replace(tmp, self._launches_path)

    # ── trades (watched mints only) ───────────────────────────────

    def record_trade(self, trade: dict[str, Any]) -> None:
        _append_jsonl(self._trades_path, trade)

    def trades(self, mint: str = "") -> list[dict[str, Any]]:
        rows = _read_jsonl(self._trades_path)
        if mint:
            rows = [r for r in rows if r.get("mint") == mint]
        return rows

    # ── observations ──────────────────────────────────────────────

    def record_observation(self, mint: str, ts: float,
                           price_lamports: float, mc_lamports: float,
                           curve_pct_value: float,
                           source: str) -> None:
        _append_jsonl(self._obs_path, {
            "mint": mint, "ts": float(ts),
            "price": float(price_lamports), "mc": float(mc_lamports),
            "curve_pct": float(curve_pct_value), "source": source})

    def observations(self, mint: str, since: float = 0.0,
                       until: float = 0.0) -> list[dict[str, Any]]:
        rows = [r for r in _read_jsonl(self._obs_path)
                if r.get("mint") == mint and r.get("ts", 0) >= since
                and (not until or r.get("ts", 0) <= until)]
        return sorted(rows, key=lambda r: r.get("ts", 0))

    # ── dossiers ──────────────────────────────────────────────────

    def _load_json(self, path: str) -> dict[str, Any]:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return data
            except (OSError, ValueError):
                pass
        return {}

    def _save_json(self, path: str, data: dict[str, Any]) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, path)

    def save_dossier(self, dossier: dict[str, Any]) -> None:
        all_d = self._load_json(self._dossiers_path)
        all_d[dossier["mint"]] = dossier
        self._save_json(self._dossiers_path, all_d)

    def dossier(self, mint: str) -> Optional[dict[str, Any]]:
        return self._load_json(self._dossiers_path).get(mint)

    def dossiers(self) -> dict[str, dict[str, Any]]:
        return self._load_json(self._dossiers_path)

    # ── dev profiles ──────────────────────────────────────────────

    def dev_profiles(self) -> dict[str, Any]:
        return self._load_json(self._devs_path)

    def save_dev_profiles(self, profiles: dict[str, Any]) -> None:
        self._save_json(self._devs_path, profiles)

    # ── watchlists ────────────────────────────────────────────────

    def watchlists(self) -> dict[str, Any]:
        data = self._load_json(self._watch_path)
        data.setdefault("devs", {})
        data.setdefault("smart", {})
        return data

    def _save_watchlists(self, data: dict[str, Any]) -> None:
        self._save_json(self._watch_path, data)

    def watch_dev(self, wallet: str, note: str = "") -> None:
        data = self.watchlists()
        data["devs"][wallet] = {"note": note, "added_at": time.time()}
        self._save_watchlists(data)

    def watch_smart(self, wallet: str, note: str = "") -> None:
        data = self.watchlists()
        data["smart"][wallet] = {"note": note, "added_at": time.time()}
        self._save_watchlists(data)

    def unwatch(self, wallet: str) -> bool:
        data = self.watchlists()
        gone = data["devs"].pop(wallet, None) is not None
        gone = (data["smart"].pop(wallet, None) is not None) or gone
        if gone:
            self._save_watchlists(data)
        return gone

    def watched_devs(self) -> set[str]:
        return set(self.watchlists().get("devs", {}))

    def watched_smart(self) -> set[str]:
        return set(self.watchlists().get("smart", {}))

    # ── wallet graph (P2) ─────────────────────────────────────────

    def save_graph(self, graph: dict[str, Any]) -> None:
        self._save_json(self._graph_path, graph)

    def graph(self) -> dict[str, Any]:
        data = self._load_json(self._graph_path)
        data.setdefault("wallets", {})
        data.setdefault("clusters", [])
        return data

    # ── scoring parameters + recalibration journal (P2) ───────────

    def score_params(self) -> Optional[dict[str, Any]]:
        data = self._load_json(self._params_path)
        return data or None

    def save_score_params(self, params: dict[str, Any]) -> None:
        self._save_json(self._params_path, params)

    def record_param_event(self, event: dict[str, Any]) -> None:
        _append_jsonl(self._params_log_path, event)

    def param_events(self) -> list[dict[str, Any]]:
        return _read_jsonl(self._params_log_path)

    # ── feed-health journal (resilience pass) ─────────────────────

    def record_health_event(self, event: dict[str, Any]) -> None:
        """Append one desk feed-state transition (stand-down
        entered / recovered, with the per-source reasons at the
        time).  Append-only like every desk ledger: the desk's
        state history is auditable, never rewritten."""
        _append_jsonl(self._health_log_path, event)

    def health_events(self) -> list[dict[str, Any]]:
        return _read_jsonl(self._health_log_path)

    # ── knowledge shelf (P2) ──────────────────────────────────────

    def shelve_knowledge(self, item: dict[str, Any]) -> bool:
        """Append one knowledge item to the shelf.  Append-only
        like every desk ledger: re-shelving the same key appends a
        newer version and reads take the latest (see
        :meth:`knowledge`), so a dossier's shelf record tracks its
        outcome instead of freezing at first sight."""
        _append_jsonl(self._knowledge_path, item)
        return True

    def knowledge(self) -> list[dict[str, Any]]:
        """The shelf, deduped by key (latest wins)."""
        seen: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in _read_jsonl(self._knowledge_path):
            key = row.get("_key") or row.get("id", "")
            if key not in seen:
                order.append(key)
            seen[key] = row
        return [seen[k] for k in order]
