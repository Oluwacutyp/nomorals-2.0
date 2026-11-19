"""Persistent, resumable engagement state.

Port of ``engine/state.py`` from elementalsouls/Claude-BugHunter
(MIT, Copyright (c) 2026 Sachin Sharma), same state shape and
methods. Devon difference: the base directory is Devon's home
data dir (``<home>/data/bughunter/engagements/<name>/``) instead
of upstream's ``~/.bughunter-engagements`` — the engine never
writes into another agent's home, and the layout stays the same
so a run is auditable and resumable:

  <base>/<name>/
    state.json      surface items, worklist, tested set, candidate
                    + confirmed findings, phase
    engine.log      append-only run log
    evidence/       per-finding evidence files
    report.md       generated at the end
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger

__all__ = ["Engagement", "engagements_root"]

_log = get_logger(__name__)


def _now() -> str:
    # timezone.utc, not datetime.UTC: EC2 runs Python <= 3.11 and this
    # box's 3.12.3 build lacks datetime.UTC (verified).
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def engagements_root(home: str | Path | None = None) -> Path:
    """Where engagement state lives under Devon's home."""
    base = Path(home or Path.home())
    return base / "data" / "bughunter" / "engagements"


class Engagement:
    """One engagement's durable state, loaded or fresh."""

    def __init__(self, base: str | Path, name: str) -> None:
        self.dir = Path(base) / name
        self.name = name
        (self.dir / "evidence").mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.log_path = self.dir / "engine.log"
        self.state = self._load()
        self.state.setdefault("rejected", [])

    DEFAULTS: dict[str, Any] = {
        "targets": [], "surface": [], "tested": [], "candidates": [],
        "confirmed": [], "coverage": {}, "coverage_by_item": {},
    }

    def _fresh_defaults(self) -> dict[str, Any]:
        # Fresh copies — never alias the mutable class-level
        # DEFAULTS into instance state.
        return {k: (dict(v) if isinstance(v, dict) else list(v))
                for k, v in self.DEFAULTS.items()}

    def _load(self) -> dict[str, Any]:
        if self.state_path.is_file():
            s = json.loads(self.state_path.read_text(encoding="utf-8"))
            for k, v in self._fresh_defaults().items():
                s.setdefault(k, v)  # forward-compat with pre-ledger runs
            return s
        return {"name": self.name, "created": _now(), "phase": "init",
                **self._fresh_defaults()}

    def save(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2),
                       encoding="utf-8")
        tmp.replace(self.state_path)

    def log(self, msg: str) -> None:
        line = f"{_now()}  {msg}"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        _log.info("bughunter[%s] %s", self.name, msg)

    # ── phase ────────────────────────────────────────────────
    def set_phase(self, phase: str) -> None:
        self.state["phase"] = phase
        self.save()

    # ── surface (discovered, scope-checked endpoints/params) ──
    def add_surface(self, items: list[dict[str, Any]]) -> int:
        seen = {(s["url"], s.get("param", ""), s.get("vuln_class", ""))
                for s in self.state["surface"]}
        added = 0
        for it in items:
            key = (it["url"], it.get("param", ""),
                   it.get("vuln_class", ""))
            if key not in seen:
                self.state["surface"].append(it)
                seen.add(key)
                added += 1
        self.save()
        return added

    def worklist(self) -> list[dict[str, Any]]:
        """Ranked, not-yet-tested surface items."""
        tested = set(self.state["tested"])
        items = [s for s in self.state["surface"]
                 if self._key(s) not in tested]
        return sorted(items, key=lambda s: -s.get("priority", 0))

    @staticmethod
    def _key(s: dict[str, Any]) -> str:
        return (f"{s['url']}|{s.get('param', '')}"
                f"|{s.get('vuln_class', '')}")

    def mark_tested(self, item: dict[str, Any],
                    candidate: dict[str, Any] | None = None,
                    coverage: str | None = None, note: str = "") -> None:
        k = self._key(item)
        if k not in self.state["tested"]:
            if candidate is not None:
                candidate["ts"] = _now()
                self.state["candidates"].append(candidate)
            self.state["tested"].append(k)
            if coverage is not None:
                self._set_item_coverage(item, coverage, note)
            self.save()

    # ── findings ─────────────────────────────────────────────
    def add_candidate(self, finding: dict[str, Any]) -> None:
        finding["ts"] = _now()
        self.state["candidates"].append(finding)
        self.save()

    def confirm(self, finding: dict[str, Any],
                verdict: dict[str, Any]) -> None:
        finding = {**finding, "verdict": verdict,
                   "confirmed_ts": _now()}
        self.state["confirmed"].append(finding)
        self.save()

    # ── coverage ledger (writer: hunt) ───────────────────────
    # A count of tested items cannot answer "which classes were
    # never tried on host X". This can, and the report renders its
    # caveats from it.
    def _set_item_coverage(self, item: dict[str, Any], verdict: str,
                           note: str) -> None:
        from .closure import VERDICTS
        if verdict not in VERDICTS:
            raise ValueError(
                f"unknown coverage verdict {verdict!r}; expected one of "
                f"{sorted(VERDICTS)}")
        self.state["coverage_by_item"][self._key(item)] = {
            "verdict": verdict, "note": note, "ts": _now()}

    def record_coverage(self, item: dict[str, Any], verdict: str,
                        note: str = "") -> None:
        self._set_item_coverage(item, verdict, note)
        self.save()

    def evidence_path(self, fname: str) -> Path:
        return self.dir / "evidence" / fname

    # ── progress ─────────────────────────────────────────────
    def summary(self) -> dict[str, Any]:
        return {"surface": len(self.state["surface"]),
                "tested": len(self.state["tested"]),
                "candidates": len(self.state["candidates"]),
                "confirmed": len(self.state["confirmed"]),
                "phase": self.state["phase"]}


def state_selftest() -> bool:
    """Upstream's self-test, verbatim in cases (tmpdir-rooted)."""
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    try:
        e = Engagement(d, "demo")
        assert e.add_surface([
            {"url": "http://t/a", "param": "id", "vuln_class": "idor",
             "priority": 5},
            {"url": "http://t/a", "param": "id",
             "vuln_class": "idor"}]) == 1  # dedup
        e.add_surface([{"url": "http://t/b", "param": "q",
                        "vuln_class": "xss", "priority": 9}])
        wl = e.worklist()
        assert wl[0]["url"] == "http://t/b"  # higher priority first
        e.mark_tested(wl[0])
        assert all(s["url"] != "http://t/b" or s.get("param") != "q"
                   for s in e.worklist())
        e.add_candidate({"url": "http://t/a", "vuln_class": "idor",
                         "evidence": "x"})
        e.confirm(e.state["candidates"][0], {"real": True})
        # Resume: a fresh handle on the same dir sees persisted state.
        e2 = Engagement(d, "demo")
        assert e2.summary() == {"surface": 2, "tested": 1,
                                "candidates": 1, "confirmed": 1,
                                "phase": "init"}
        return True
    finally:
        shutil.rmtree(d, ignore_errors=True)
