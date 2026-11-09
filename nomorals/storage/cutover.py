"""Cutover machinery — the SQLite → MongoDB primary flip (Mongo P6).

The final phase of the MongoDB program. P0–P5 built the seam, the
carry, the parity gate, the domain ports and the sidecar federation;
this module is the *maintenance flow* that turns an install over:

    preflight → carry → parity gate → flip → smoke      (+ rollback)

Design rules, each one a census-named failure kept impossible:

* **Preflight refuses.** The target mongod must be reachable, a
  replica-set member, a writable primary, and cache-capped for the
  host (census §5.1: WiredTiger cache 0.25–0.5 GB on the 2 GB EC2
  box — an uncapped mongod OOMs its own host). And every residue
  sidecar store (storage/sidecars.py) that holds file data must carry
  a recorded *disposition* — ``exported``, ``ports-later-accepted``
  or ``owner-data-risk`` — before anything moves. Residue data is
  never silently stranded by a flip.
* **The flip needs a gate PASS on record.** The flow runs P2's parity
  gate itself and flips only on a recorded PASS for the current
  mapping version; absence of a record is a refusal, per P2's rule.
* **The flip is a config change, not a file edit.** It writes the
  ``[storage]`` section of the install's config through
  :func:`flip_config` (atomic write, reload through
  ``load_settings``, effective-value assertion — environment
  overrides that would silently defeat the file are named).
* **The source is never written by the carry.** ``carry()`` reads the
  SQLite database and writes only the target; the parity checksums in
  the rehearsal prove the source byte-state (per-collection digests)
  is identical before and after a cutover. Rollback is therefore
  just the config flip back — Mongo keeps its copy, untouched.
* **Every run is recorded** in the source database
  (``cutover_runs``), and smoke runs against a *fresh* boot of the
  flipped configuration: storage resolution reports Mongo LIVE, a KV
  round-trip lands through the seam, and one ported domain
  (conversations) round-trips through its production API.

Dry-run mode walks preflight + carry (dry) + an *unrecorded,
informational* gate, and reports the flip and smoke as skipped —
the configuration is never touched.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..core.config import load_settings
from ..core.logging_setup import get_logger
from .carry import MAPPING_VERSION, carry, parity_gate_status, run_parity_gate
from .docstore import DEFAULT_MONGO_URI
from .sidecars import SIDECAR_STORES

__all__ = [
    "DISPOSITIONS",
    "MAX_MONGO_CACHE_BYTES",
    "record_disposition",
    "dispositions",
    "residue_report",
    "preflight",
    "flip_config",
    "smoke_test",
    "run_cutover",
    "rollback_cutover",
    "cutover_status",
]

_log = get_logger(__name__)

#: The named dispositions a residue store's data can carry into a
#: cutover. ``exported`` — its data was carried/exported elsewhere
#: first; ``ports-later-accepted`` — it stays on its file after the
#: flip, its seam port knowingly queued; ``owner-data-risk`` — the
#: owner accepts the file being left behind as-is.
DISPOSITIONS = ("exported", "ports-later-accepted", "owner-data-risk")

#: WiredTiger cache cap the preflight enforces (census §5.1): the
#: target host has ~2 GB of RAM; mongod's cache must stay at or below
#: half a gigabyte or the database becomes the host's OOM risk.
MAX_MONGO_CACHE_BYTES = 512 * 1024 * 1024  # 0.5 GiB

_DISPOSITIONS_TABLE = "cutover_dispositions"
_RUNS_TABLE = "cutover_runs"

#: Directories under the install home that never hold residue stores:
#: payload stores of other machinery (snapshot/backup payloads are
#: verified by their own machinery; session/media trees are not
#: sidecar databases).
_SKIP_DIRS = {"blobs", "snapshots", "backups", "whatsapp_sessions",
              "media", "node_modules", "__pycache__"}

_DB_TOKEN = re.compile(r"([A-Za-z0-9_.-]+\.db)\b")


# ── disposition + run records (in the source database) ───────────────────

def _ensure_tables(db: Any) -> None:
    db.execute(
        f"CREATE TABLE IF NOT EXISTS {_DISPOSITIONS_TABLE} ("
        " store TEXT PRIMARY KEY, disposition TEXT NOT NULL,"
        " note TEXT NOT NULL DEFAULT '', at REAL NOT NULL)")
    db.execute(
        f"CREATE TABLE IF NOT EXISTS {_RUNS_TABLE} ("
        " id TEXT PRIMARY KEY, at REAL NOT NULL, action TEXT NOT NULL,"
        " dry_run INTEGER NOT NULL DEFAULT 0, ok INTEGER NOT NULL"
        " DEFAULT 0, detail TEXT NOT NULL DEFAULT '{}')")


def record_disposition(db: Any, store: str, disposition: str,
                       note: str = "") -> dict[str, Any]:
    """Record the cutover disposition of one residue store.

    Only residue-verdict stores take dispositions — anything else is a
    category error and raises. The record lives in the source
    database, so it survives into (and is auditable after) the flip.
    """
    spec = SIDECAR_STORES.get(store)
    if spec is None:
        raise ValueError(f"unknown sidecar store: {store!r}")
    if spec.verdict != "residue":
        raise ValueError(
            f"store {store!r} has verdict {spec.verdict!r}; only "
            "residue stores carry cutover dispositions")
    if disposition not in DISPOSITIONS:
        raise ValueError(
            f"disposition must be one of {DISPOSITIONS}, "
            f"got {disposition!r}")
    _ensure_tables(db)
    at = time.time()
    db.execute(
        f"INSERT OR REPLACE INTO {_DISPOSITIONS_TABLE}"
        " (store, disposition, note, at) VALUES (?, ?, ?, ?)",
        (store, disposition, note, at))
    return {"store": store, "disposition": disposition, "note": note,
            "at": at}


def dispositions(db: Any) -> dict[str, dict[str, Any]]:
    """All recorded residue dispositions, by store name."""
    _ensure_tables(db)
    out: dict[str, dict[str, Any]] = {}
    for row in db.query(
            f"SELECT * FROM {_DISPOSITIONS_TABLE} ORDER BY store"):
        out[row["store"]] = {
            "disposition": row["disposition"],
            "note": row["note"],
            "at": float(row["at"]),
        }
    return out


def _record_run(db: Any, action: str, *, dry_run: bool, ok: bool,
                detail: dict[str, Any]) -> None:
    _ensure_tables(db)
    db.execute(
        f"INSERT INTO {_RUNS_TABLE} (id, at, action, dry_run, ok, detail)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (uuid.uuid4().hex, time.time(), action, 1 if dry_run else 0,
         1 if ok else 0, json.dumps(detail, default=str)))


def _recent_runs(db: Any, limit: int = 10) -> list[dict[str, Any]]:
    _ensure_tables(db)
    runs = []
    for row in db.query(
            f"SELECT * FROM {_RUNS_TABLE} ORDER BY at DESC LIMIT ?",
            (limit,)):
        runs.append({
            "at": float(row["at"]), "action": row["action"],
            "dry_run": bool(row["dry_run"]), "ok": bool(row["ok"]),
            "detail": json.loads(row["detail"] or "{}"),
        })
    return runs


# ── the residue probe ────────────────────────────────────────────────────

def _candidate_filenames(spec: Any) -> set[str]:
    """Filenames a residue store's data may live under: the file its
    registry reason names, any ``*.db`` literal in its owner modules'
    sources, its modules' stems (plain and directory-qualified), its
    registry name's tail, and its tables' names."""
    names: set[str] = set()
    names.update(m.group(1) for m in _DB_TOKEN.finditer(spec.reason))
    package = Path(__file__).resolve().parent.parent
    for module in spec.modules:
        names.add(Path(module).stem + ".db")
        parent = Path(module).parent.name
        if parent:
            names.add(f"{parent}_{Path(module).stem}.db")
        source = package / module
        try:
            text = source.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        names.update(m.group(1) for m in _DB_TOKEN.finditer(text))
    names.add(spec.name.rsplit("-", 1)[-1] + ".db")
    names.add(spec.name.replace("-", "_") + ".db")
    for table in spec.tables:
        names.add(table + ".db")
    return names


def _file_rows(path: Path, tables: tuple[str, ...]) -> int | None:
    """Total rows across ``tables`` in a SQLite file, or None when the
    file cannot be read as one."""
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            present = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
            total = 0
            for table in tables:
                if table in present:
                    total += int(conn.execute(
                        f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
            return total
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def residue_report(home: str | os.PathLike[str],
                   db_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Probe the install home for residue-store data files.

    Returns ``{"stores": {name: {"files": [{path, bytes, rows}],
    "has_data": bool}}, "unattributed": [{path, bytes}]}`` — every
    ``*.db`` under home (minus payload/media/session trees and the
    central database itself) attributed to a residue store by the
    candidate-name evidence in :func:`_candidate_filenames`, or
    listed as unattributed: an unowned database file on cutover day
    is a finding in its own right.
    """
    home_path = Path(home)
    central = Path(db_path).resolve()
    found: list[Path] = []
    if home_path.exists():
        for path in sorted(home_path.rglob("*.db")):
            parts = set(path.relative_to(home_path).parts)
            if parts & _SKIP_DIRS:
                continue
            if any(p.startswith(".restore-") for p in parts):
                continue
            if path.resolve() == central:
                continue
            found.append(path)
    specs = {name: spec for name, spec in SIDECAR_STORES.items()
             if spec.verdict == "residue"}
    candidates = {name: _candidate_filenames(spec)
                  for name, spec in specs.items()}
    stores: dict[str, Any] = {
        name: {"files": [], "has_data": False} for name in specs}
    attributed: set[Path] = set()
    for path in found:
        owners = [name for name, names in candidates.items()
                  if path.name in names]
        entry = {"path": str(path), "bytes": path.stat().st_size}
        for name in owners:
            info = dict(entry)
            info["rows"] = _file_rows(path, specs[name].tables)
            stores[name]["files"].append(info)
            stores[name]["has_data"] = True
            attributed.add(path)
    unattributed = [{"path": str(p), "bytes": p.stat().st_size}
                    for p in found if p not in attributed]
    return {"stores": stores, "unattributed": unattributed}


# ── preflight ────────────────────────────────────────────────────────────

def preflight(db: Any, settings: Any, target: Any, *,
              max_cache_bytes: int = MAX_MONGO_CACHE_BYTES,
              ) -> dict[str, Any]:
    """Verify the cutover may proceed. Never mutates anything.

    Checks: target reachable; target is a replica-set member; target
    is a writable primary; WiredTiger cache capped at
    ``max_cache_bytes``; source database present; and every residue
    store with file data carries a recorded disposition.
    """
    checks: dict[str, bool] = {}
    health = target.health()
    checks["target_reachable"] = bool(health.ok)
    facts: dict[str, Any] = {}
    if hasattr(target, "server_facts"):
        facts = target.server_facts()
    checks["replica_set"] = bool(facts.get("set_name"))
    checks["writable_primary"] = bool(facts.get("is_writable_primary"))
    cache = facts.get("wiredtiger_cache_bytes")
    checks["cache_capped"] = bool(
        cache is not None and 0 < int(cache) <= max_cache_bytes)
    checks["source_present"] = Path(settings.db_path).exists()
    residue = residue_report(settings.home_path, settings.db_path)
    recorded = dispositions(db)
    missing = sorted(
        name for name, info in residue["stores"].items()
        if info["has_data"] and name not in recorded)
    checks["residue_dispositions"] = not missing
    ok = all(checks.values())
    return {
        "ok": ok,
        "checks": checks,
        "health": {"ok": bool(health.ok), "state": health.state,
                   "detail": health.detail},
        "facts": facts,
        "max_cache_bytes": max_cache_bytes,
        "residue": residue,
        "dispositions": recorded,
        "missing_dispositions": missing,
    }


# ── the config flip ──────────────────────────────────────────────────────

def _toml_string(value: str) -> str:
    """A TOML basic string for ``value`` (JSON string syntax is a
    valid subset for the values this flow writes)."""
    return json.dumps(value)


def _set_storage_keys(text: str, values: dict[str, str]) -> str:
    """Return config text with the ``[storage]`` keys set: existing
    keys replaced in place inside the section, missing keys appended
    to it, the section created at the end when absent. Everything
    outside the section is preserved byte-for-byte."""
    lines = text.splitlines()
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if start is None and stripped == "[storage]":
                start = i
            elif start is not None:
                end = i
                break
    if start is None:
        out = list(lines)
        if out and out[-1].strip():
            out.append("")
        out.append("[storage]")
        out.extend(f"{k} = {_toml_string(v)}" for k, v in values.items())
        return "\n".join(out) + "\n"
    section = list(lines[start + 1:end])
    remaining = dict(values)
    for i, line in enumerate(section):
        stripped = line.strip()
        for key in list(remaining):
            if re.match(rf"{re.escape(key)}\s*=", stripped):
                indent = line[:len(line) - len(line.lstrip())]
                section[i] = f"{indent}{key} = {_toml_string(remaining.pop(key))}"
                break
    for key, value in remaining.items():
        section.append(f"{key} = {_toml_string(value)}")
    return "\n".join(lines[:start + 1] + section + lines[end:]) + "\n"


def _env_culprits(settings_home: Path, key: str) -> list[str]:
    """Where an environment override for a storage key lives, if one
    does: the process environment (``NM_*``) or the install's ``.env``
    file — both outrank the TOML file in ``load_settings`` precedence,
    so either would silently defeat a config flip."""
    culprits = []
    if key in os.environ:
        culprits.append(f"{key} is set in the process environment")
    env_file = settings_home / ".env"
    if env_file.exists():
        try:
            for line in env_file.read_text(
                    encoding="utf-8", errors="replace").splitlines():
                if line.strip().startswith(f"{key}="):
                    culprits.append(f"{key} is set in {env_file}")
                    break
        except OSError:
            pass
    return culprits


def flip_config(settings: Any, *, backend: str,
                mongo_uri: str | None = None,
                mongo_db: str | None = None) -> dict[str, Any]:
    """Write the storage flip into the install's config and prove it
    took effect.

    The write is atomic (temp file + ``os.replace``) and touches only
    the ``[storage]`` section. The result is then reloaded through
    :func:`load_settings`; if the effective backend is not the one
    written, the flip reports ``effective: False`` with the overriding
    sources named in ``culprits`` — a flip that did not take is a
    failure, never a silent no-op.
    """
    if settings.config_file:
        path = Path(settings.config_file)
    else:
        path = Path(settings.home_path) / "config.toml"
    values: dict[str, str] = {"backend": backend}
    if mongo_uri is not None:
        values["mongo_uri"] = mongo_uri
    if mongo_db is not None:
        values["mongo_db"] = mongo_db
    text = ""
    if path.exists():
        text = path.read_text(encoding="utf-8")
    new_text = _set_storage_keys(text, values)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    os.replace(tmp, path)
    fresh = load_settings(config_file=path, env=os.environ)
    effective = fresh.storage.backend == backend
    culprits: list[str] = []
    if not effective:
        culprits = _env_culprits(Path(settings.home_path), "NM_DB_BACKEND")
        if not culprits:
            culprits = ["the reloaded configuration does not carry the "
                        "written backend and no environment override "
                        "was found — inspect the config file chain"]
    return {"path": str(path), "backend": backend,
            "effective": effective, "culprits": culprits,
            "mongo_uri": fresh.storage.mongo_uri,
            "mongo_db": fresh.storage.mongo_db}


# ── smoke ────────────────────────────────────────────────────────────────

def _effective_settings(settings: Any) -> Any:
    """The settings a fresh boot of this install would load: when the
    install has a config file, reload it (a flip lives there, not in
    the caller's already-loaded object); otherwise the given object
    is the configuration."""
    path = None
    if getattr(settings, "config_file", ""):
        path = Path(settings.config_file)
    else:
        candidate = Path(settings.home_path) / "config.toml"
        if candidate.exists():
            path = candidate
    if path is not None and path.exists():
        return load_settings(config_file=path, env=os.environ)
    return settings


def smoke_test(settings: Any) -> dict[str, Any]:
    """Boot the (possibly just-flipped) configuration fresh and prove
    the storage resolution works end to end: Mongo LIVE in the status
    surface, a KV round-trip through the seam, and one ported domain
    (conversations) round-tripping through its production API. Probe
    rows are removed afterwards."""
    from .conversations import ConversationLog
    from .db import open_database
    from .docstore import attach_document_store, storage_backend_status
    from .kv import KVStore

    settings = _effective_settings(settings)
    checks: dict[str, bool] = {}
    detail: dict[str, Any] = {}
    db, _recovered, _note = open_database(settings.db_path)
    try:
        resolution = attach_document_store(db, settings)
        detail["resolution"] = resolution.to_dict()
        checks["resolution_mongo"] = resolution.active == "mongo"
        store = getattr(db, "document_store", None)
        checks["mongo_live"] = bool(
            store is not None and store.health().ok)
        status = storage_backend_status(settings, db)
        detail["status"] = status
        checks["status_reports_mongo"] = \
            status.get("active_backend") == "mongo"
        if store is None:
            checks["kv_round_trip"] = False
            checks["domain_round_trip"] = False
            return {"ok": False, "checks": checks, "detail": detail}

        kv = KVStore(db)
        probe_key = f"__cutover_smoke__:{uuid.uuid4().hex[:12]}"
        probe_value = {"probe": "cutover", "at": time.time()}
        try:
            kv.set(probe_key, probe_value)
            checks["kv_round_trip"] = kv.get(probe_key) == probe_value
        finally:
            kv.delete(probe_key)

        log = ConversationLog(db)
        cid = f"cutover-smoke-{uuid.uuid4().hex[:12]}"
        mid = f"cutover-smoke-msg-{uuid.uuid4().hex[:12]}"
        now = time.time()
        try:
            log.ensure_conversation(cid, title="cutover smoke",
                                    agent="cutover", channel="smoke",
                                    now=now)
            log.append_message(row_id=mid, conversation_id=cid,
                               role="user", content="ping",
                               created_at=now)
            conv = log.get_conversation(cid)
            history = log.history(cid, limit=5)
            checks["domain_round_trip"] = bool(
                conv is not None
                and any(m.get("content") == "ping" for m in history))
        finally:
            store.delete("conversations", cid)
            store.delete("messages", mid)
    finally:
        db.close()
    return {"ok": all(checks.values()), "checks": checks,
            "detail": detail}


# ── the flow ─────────────────────────────────────────────────────────────

def run_cutover(db: Any, settings: Any, *, target: Any,
                dry_run: bool = False,
                progress: Callable[[dict[str, Any]], None] | None = None,
                ) -> dict[str, Any]:
    """Run the cutover flow against ``target`` (a DocumentStore).

    Steps: preflight → carry → parity gate → flip → smoke. Any
    refusal stops the flow before the next step and is recorded in
    ``cutover_runs``. ``dry_run`` walks everything except the flip
    (carry dry, gate unrecorded and informational) and says so.
    """
    report: dict[str, Any] = {
        "dry_run": dry_run, "ok": False, "flipped": False,
        "mapping_version": MAPPING_VERSION, "steps": [],
    }

    def step(name: str, ok: bool, detail: Any = None,
             status: str | None = None) -> None:
        report["steps"].append({
            "step": name, "ok": ok,
            "status": status or ("ok" if ok else "failed"),
            "detail": detail})

    pre = preflight(db, settings, target)
    step("preflight", pre["ok"], {
        "checks": pre["checks"],
        "missing_dispositions": pre["missing_dispositions"],
        "unattributed_files": pre["residue"]["unattributed"],
    })
    if not pre["ok"]:
        report["refused_at"] = "preflight"
        _record_run(db, "cutover", dry_run=dry_run, ok=False,
                    detail=report)
        return report

    try:
        carried = carry(db, target, dry_run=dry_run, progress=progress)
    except Exception as exc:  # noqa: BLE001 - the flow reports, recorded
        step("carry", False, f"{type(exc).__name__}: {exc}")
        report["refused_at"] = "carry"
        _record_run(db, "cutover", dry_run=dry_run, ok=False,
                    detail=report)
        return report
    step("carry", bool(carried.ok), carried.to_dict())
    if not carried.ok:
        report["refused_at"] = "carry"
        _record_run(db, "cutover", dry_run=dry_run, ok=False,
                    detail=report)
        return report

    gate = run_parity_gate(db, target, record=not dry_run)
    step("parity_gate", bool(gate["passed"]), {
        "passed": bool(gate["passed"]),
        "mapping_version": gate["mapping_version"],
        "recorded": not dry_run,
        "informational": dry_run,
        "collections": len(gate["collections"]),
        "divergent": [c["collection"] for c in gate["collections"]
                      if not c["ok"]],
    })
    if dry_run:
        # The dry-run gate is informational only: it compares the
        # target as it stands, and an as-yet-uncarried target diverges
        # by definition — that is not a refusal. The walk completes;
        # the verdict rides along as dry_gate_passed.
        step("flip", True, "skipped — dry run, configuration untouched",
             status="skipped")
        step("smoke", True, "skipped — dry run", status="skipped")
        report["ok"] = True
        report["dry_gate_passed"] = bool(gate["passed"])
        _record_run(db, "cutover", dry_run=True, ok=True, detail=report)
        return report
    if not gate["passed"]:
        report["refused_at"] = "parity_gate"
        _record_run(db, "cutover", dry_run=False, ok=False,
                    detail=report)
        return report

    storage = settings.storage
    flip = flip_config(
        settings, backend="mongo",
        mongo_uri=storage.mongo_uri or DEFAULT_MONGO_URI,
        mongo_db=storage.mongo_db)
    step("flip", bool(flip["effective"]), flip)
    if not flip["effective"]:
        report["refused_at"] = "flip"
        _record_run(db, "cutover", dry_run=False, ok=False,
                    detail=report)
        return report
    report["flipped"] = True

    smoke = smoke_test(settings)
    step("smoke", bool(smoke["ok"]), smoke)
    report["ok"] = bool(smoke["ok"])
    if not smoke["ok"]:
        report["refused_at"] = "smoke"
        report["rollback_hint"] = (
            "the flip is written but smoke failed; run "
            "`nm db cutover rollback` to return this install to "
            "sqlite (the source database was never modified)")
    _record_run(db, "cutover", dry_run=False, ok=report["ok"],
                detail=report)
    return report


def rollback_cutover(db: Any, settings: Any) -> dict[str, Any]:
    """Flip the install back to the SQLite rung.

    The SQLite source was never written by the carry, so rollback is
    only the configuration flip back; the Mongo copy is left in place,
    untouched, for forensics or a later re-attempt. Recorded in
    ``cutover_runs`` like every other run.
    """
    flip = flip_config(settings, backend="sqlite")
    report = {"ok": bool(flip["effective"]), "flip": flip,
              "note": "mongo data left in place; the sqlite source "
                      "was never modified by the carry"}
    _record_run(db, "rollback", dry_run=False, ok=report["ok"],
                detail=report)
    return report


def cutover_status(db: Any, settings: Any) -> dict[str, Any]:
    """One-call cutover state: residue probe + dispositions on
    record, the parity gate's latest record, and recent runs."""
    recorded = dispositions(db)
    residue = residue_report(settings.home_path, settings.db_path)
    missing = sorted(
        name for name, info in residue["stores"].items()
        if info["has_data"] and name not in recorded)
    stores = {}
    for name in sorted(residue["stores"]):
        info = residue["stores"][name]
        stores[name] = {
            "has_data": info["has_data"],
            "files": info["files"],
            "disposition": recorded.get(name),
        }
    return {
        "residue_stores": stores,
        "unattributed_files": residue["unattributed"],
        "missing_dispositions": missing,
        "parity_gate": parity_gate_status(db),
        "recent_runs": _recent_runs(db),
    }
