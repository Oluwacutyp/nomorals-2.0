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
