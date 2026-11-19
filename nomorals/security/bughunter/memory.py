"""Autopilot ledger: cross-engagement capture + skip decisions.

Port of ``engine/memory.py`` from elementalsouls/Claude-BugHunter
(MIT, Copyright (c) 2026 Sachin Sharma), verbatim in semantics.

Lets the hunt loop skip provably-wasteful agent calls:
re-confirming a known finding, or re-testing a vuln_class that
never pays off on this tech stack. Capture is harvested from
the engine's own state; skip is gated (off unless the engine
passes use_memory, i.e. token-saving modes only).

Storage (Devon): ``<home>/data/bughunter/memory/`` (override
with ``$DEVON_BUGHUNTER_MEMORY_DIR``), never
``~/.claude/bughunter/memory``:

  findings.jsonl       one row per confirmed finding (cross-target)
  negatives.jsonl      (host, stack_sig, vuln_class) hunted -> not confirmed
  targets/<host>.json  per-target rollup

Stdlib-only; imports nothing from sibling bughunter modules.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows
    msvcrt = None

from ...core.logging_setup import get_logger

__all__ = [
    "SCHEMA_VERSION",
    "DEAD_CLASS_THRESHOLD",
    "memory_root",
    "record_finding",
    "record_run",
    "skip_decision",
    "rollup",
    "gc",
    "memory_selftest",
]

_log = get_logger(__name__)

SCHEMA_VERSION = 1
MAX_BYTES = 10 * 1024 * 1024        # rotate a JSONL past 10 MB
KEEP = 3                            # backups kept: .1 (newest) .. .3
DEAD_CLASS_THRESHOLD = 5           # negatives before a (stack,class) is "dead"


def memory_root(home: str | Path | None = None) -> Path:
    """Ledger root: env override, else under Devon's home."""
    env = os.environ.get("DEVON_BUGHUNTER_MEMORY_DIR")
    if env:
        return Path(env).expanduser()
    return Path(home or Path.home()) / "data" / "bughunter" / "memory"


def _paths(root: str | Path | None = None) -> Path:
    path = Path(root) if root is not None else memory_root()
    (path / "targets").mkdir(parents=True, exist_ok=True)
    return path


def _now() -> str:
    # timezone.utc, not datetime.UTC: EC2 runs Python <= 3.11 and this
    # box's 3.12.3 build lacks datetime.UTC (verified) — see
    # nomorals/commerce/commands.py for the same convention.
    return datetime.now(timezone.utc).strftime(  # noqa: UP017
        "%Y-%m-%dT%H:%M:%SZ")


def _host(url: str) -> str:
    u = url or ""
    if "://" not in u:
        u = "//" + u
    return (urlparse(u).hostname or "").lower().rstrip(".")


def _stack_sig(tech_stack: Any) -> str:
    return "|".join(sorted({t.lower() for t in (tech_stack or []) if t}))


def _rotate_if_needed(path: str | Path, max_bytes: int = MAX_BYTES,
                      keep: int = KEEP) -> None:
    path = Path(path)
    try:
        if path.stat().st_size < max_bytes:
            return
    except FileNotFoundError:
        return
    oldest = Path(f"{path}.{keep}")
    if oldest.exists():
        oldest.unlink()
    for i in range(keep - 1, 0, -1):
        s, d = Path(f"{path}.{i}"), Path(f"{path}.{i + 1}")
        if s.exists():
            s.replace(d)
    path.replace(f"{path}.1")


def _append(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    line = json.dumps(obj, separators=(",", ":")) + "\n"
    lock = Path(str(path) + ".lock")
    lfd = os.open(lock, os.O_WRONLY | os.O_CREAT, 0o644)
    try:
        if fcntl:
            fcntl.flock(lfd, fcntl.LOCK_EX)
        elif msvcrt:
            with contextlib.suppress(OSError):
                msvcrt.locking(lfd, msvcrt.LK_LOCK, 1)
        _rotate_if_needed(path)  # inside the lock: no rename race
        with open(path, "ab") as f:
            f.write(line.encode("utf-8"))
    finally:
        if fcntl:
            fcntl.flock(lfd, fcntl.LOCK_UN)
        elif msvcrt:
            with contextlib.suppress(OSError):
                msvcrt.locking(lfd, msvcrt.LK_UNLCK, 1)
        os.close(lfd)


def _read(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return out
    with open(path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                record = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                out.append(record)
    return out


def _valid_finding(e: dict[str, Any]) -> bool:
    if not (e.get("host") and e.get("url") and e.get("vuln_class")):
        return False
    try:
        datetime.fromisoformat(str(e["ts"]).replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError):
        return False
    return isinstance(e.get("tech_stack", []), list)


def record_finding(engagement: str, host: str, tech_stack: Any,
                   finding: dict[str, Any], verdict: dict[str, Any],
                   *, root: str | Path | None = None) -> bool:
    """Capture one confirmed finding. Dedup on (host,url,param,vuln_class)."""
    base = _paths(root)
    fp = base / "findings.jsonl"
    entry = {
        "schema_version": SCHEMA_VERSION, "ts": _now(),
        "engagement": engagement,
        "host": host or _host(finding.get("url", "")),
        "url": finding.get("url", ""),
        "param": finding.get("param", ""),
        "vuln_class": finding.get("vuln_class", ""),
        "technique": finding.get("technique", ""),
        "severity": (verdict or {}).get("severity")
                   or finding.get("severity", ""),
        "verdict_reason": (verdict or {}).get("reason", ""),
        "tech_stack": list(tech_stack or []),
        "evidence_ref": finding.get("evidence_ref", ""),
    }
    if not _valid_finding(entry):
        _log.warning("bughunter memory: skipping invalid finding %r",
                     entry.get("url"))
        return False
    key = (entry["host"], entry["url"], entry["param"],
           entry["vuln_class"])
    for e in _read(fp):
        if (e.get("host"), e.get("url"), e.get("param"),
                e.get("vuln_class")) == key:
            return False
    _append(fp, entry)
    return True


def _parse_key(k: str) -> tuple[str, str]:
    """State 'url|param|class' -> (host, vuln_class)."""
    url, _, rest = k.partition("|")
    _param, _, cls = rest.partition("|")
    return _host(url), cls


def _write_rollup(root: Path, host: str, tech_stack: Any,
                  tested: Any, confirmed: Any) -> None:
    p = root / "targets" / f"{host}.json"
    prev: dict[str, Any] = {}
    if p.exists():
        try:
            prev = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            prev = {}
    hc_conf = [f for f in confirmed if _host(f.get("url", "")) == host]
    doc = {
        "schema_version": SCHEMA_VERSION, "host": host,
        "first_seen": prev.get("first_seen", _now()), "last_seen": _now(),
        "sessions": prev.get("sessions", 0) + 1,
        "tech_stack": list(tech_stack or []),
        "tested": [k for k in tested if _parse_key(k)[0] == host],
        "confirmed": [{"url": f.get("url"), "param": f.get("param", ""),
                       "vuln_class": f.get("vuln_class"),
                       "severity": f.get("severity", "")}
                      for f in hc_conf],
    }
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(p)


def record_run(engagement: str, tech_stack: Any, tested: Any,
               confirmed: Any,
               *, root: str | Path | None = None) -> None:
    """End-of-run capture: negatives for tested-but-unconfirmed classes + per-host rollup."""
    base = _paths(root)
    sig = _stack_sig(tech_stack)
    tested_hc = {_parse_key(k) for k in tested}
    confirmed_hc = {(_host(f.get("url", "")), f.get("vuln_class", ""))
                    for f in confirmed}
    neg = base / "negatives.jsonl"
    for host, cls in tested_hc:
        if cls and (host, cls) not in confirmed_hc:
            _append(neg, {"schema_version": SCHEMA_VERSION, "ts": _now(),
                          "host": host, "stack_sig": sig,
                          "vuln_class": cls, "engagement": engagement})
    for host in {h for h, _ in tested_hc} | {h for h, _ in confirmed_hc}:
        if host:
            _write_rollup(base, host, tech_stack, tested, confirmed)


def skip_decision(host: str, tech_stack: Any, item: dict[str, Any],
                  *, root: str | Path | None = None) -> dict[str, Any]:
    """Should the hunt loop skip this worklist item? Off-path returns skip=False."""
    base = _paths(root)
    url, param = item.get("url", ""), item.get("param", "")
    cls = item.get("vuln_class", "")
    # 1) known-confirmed -> carry the finding forward, no agent
    for e in _read(base / "findings.jsonl"):
        if (e.get("host"), e.get("url"), e.get("param"),
                e.get("vuln_class")) == (host, url, param, cls):
            carry = {"url": url, "param": param, "vuln_class": cls,
                     "severity": e.get("severity", ""),
                     "evidence": e.get("verdict_reason", ""),
                     "verdict_reason": e.get("verdict_reason", ""),
                     "verdict": {"real": True,
                                 "severity": e.get("severity", ""),
                                 "reason": e.get("verdict_reason", ""),
                                 "from_memory": True}}
            return {"skip": True, "reason": "known-confirmed",
                    "carry": carry}
    # 2) dead-class: >= threshold negatives for (stack_sig, cls) AND
    # zero confirmations for it
    sig = _stack_sig(tech_stack)
    negatives = _read(base / "negatives.jsonl")
    distinct = {
        (entry.get("engagement") or index, entry.get("host"))
        for index, entry in enumerate(negatives)
        if entry.get("stack_sig") == sig
        and entry.get("vuln_class") == cls
    }
    negs = len(distinct)
    if negs >= DEAD_CLASS_THRESHOLD:
        confirmed_here = any(
            _stack_sig(e.get("tech_stack", [])) == sig
            and e.get("vuln_class") == cls
            for e in _read(base / "findings.jsonl"))
        if not confirmed_here:
            return {"skip": True,
                    "reason": f"dead-class ({negs} negatives, "
                              "0 confirmed)",
                    "carry": None}
    return {"skip": False, "reason": "", "carry": None}


def rollup(host: str, *,
           root: str | Path | None = None) -> dict[str, Any] | None:
    """Per-host rollup, or None when the host was never engaged."""
    p = _paths(root) / "targets" / f"{host}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def gc(action: str = "report", max_mb: int = 10, keep: int = KEEP,
       root: str | Path | None = None) -> list[dict[str, Any]]:
    """report | rotate | purge-backups over every *.jsonl in the ledger dir."""
    base = Path(root) if root is not None else memory_root()
    rows = []
    for dirpath, _dirs, names in os.walk(base):
        for n in sorted(names):
            if not n.endswith(".jsonl"):
                continue
            live = Path(dirpath) / n
            backups = [Path(f"{live}.{i}") for i in range(1, keep + 1)
                       if Path(f"{live}.{i}").exists()]
            live_sz = live.stat().st_size if live.exists() else 0
            total = live_sz + sum(b.stat().st_size for b in backups)
            if action == "rotate" and live_sz >= max_mb * 1024 * 1024:
                _rotate_if_needed(live, max_bytes=max_mb * 1024 * 1024,
                                  keep=keep)
            if action == "purge-backups":
                for b in backups:
                    b.unlink()
            rows.append({"file": str(live), "live_bytes": live_sz,
                         "total_bytes": total, "backups": len(backups)})
    return rows


def memory_selftest() -> bool:
    """Upstream's self-test, verbatim in cases (tmpdir-rooted)."""
    import shutil
    import tempfile
    d = tempfile.mkdtemp()
    try:
        f = {"url": "http://a.com/x", "param": "id", "vuln_class": "idor",
             "severity": "high"}
        assert record_finding("e1", "a.com", ["nextjs"], f,
                              {"reason": "pii", "severity": "high"},
                              root=d) is True
        assert record_finding("e1", "a.com", ["nextjs"], f,
                              {"reason": "pii"}, root=d) is False  # dedup
        record_run("e1", ["nextjs"],
                   ["http://a.com/y|q|xss", "http://a.com/x|id|idor"],
                   [{"url": "http://a.com/x", "vuln_class": "idor"}],
                   root=d)
        negs = _read(Path(d) / "negatives.jsonl")
        # xss tested, not confirmed -> negative
        assert any(n["vuln_class"] == "xss" for n in negs)
        # idor confirmed -> no negative
        assert not any(n["vuln_class"] == "idor" for n in negs)
        assert json.loads(
            (Path(d) / "targets" / "a.com.json").read_text()
        )["sessions"] == 1
        # skip_decision: known-confirmed carries the finding
        res = skip_decision("a.com", ["nextjs"],
                            {"url": "http://a.com/x", "param": "id",
                             "vuln_class": "idor"}, root=d)
        assert res["skip"] and res["reason"] == "known-confirmed" \
            and res["carry"]["verdict"]["real"] is True
        # Unseen class is never skipped.
        assert skip_decision("a.com", ["nextjs"],
                             {"url": "http://a.com/x", "param": "id",
                              "vuln_class": "rce"},
                             root=d)["skip"] is False
        # Dead-class only after >= threshold negatives, zero confirmations.
        for index in range(DEAD_CLASS_THRESHOLD):
            record_run(f"e-{index}", ["django"],
                       ["http://b.com/z|q|csrf"], [], root=d)
        assert skip_decision("b.com", ["django"],
                             {"url": "http://b.com/z", "param": "q",
                              "vuln_class": "csrf"},
                             root=d)["skip"] is True
        # A confirmation on that (stack,class) revives it (not dead).
        record_finding("e", "b.com", ["django"],
                       {"url": "http://b.com/z", "param": "q",
                        "vuln_class": "csrf", "severity": "low"},
                       {"reason": "x"}, root=d)
        assert skip_decision("b.com", ["django"],
                             {"url": "http://b.com/z", "param": "q",
                              "vuln_class": "csrf"},
                             root=d)["reason"] == "known-confirmed"
        return True
    finally:
        shutil.rmtree(d, ignore_errors=True)
