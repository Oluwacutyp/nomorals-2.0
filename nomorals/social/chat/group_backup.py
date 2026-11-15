"""Per-group backup/export (DESIGN §9.8).

One group, one file: the group's full sovereign state — stored
settings, roster (members + identity-change history), audit log,
join requests, acceptance state, extras (proposals, gangs, estate,
channel queue, boards) — plus the group economy's whole per-group
file set (ledger first of all). A backup is a single self-contained
JSON file under ``<store>/group_backups/<slug>/<id>.json`` with a
manifest carrying a sha256 over the payload, so a tampered or
half-written file is refused rather than restored.

The export IS the stored file: ``/gbackup export`` sends it as a
document (the owner's standing delivery rule — a pull lands as a
file, the chat reply carries the summary). Restore is two steps by
design: ``restore <id>`` shows the preview — members added /
removed / changed, balances moved, settings keys changed,
economy files replaced / added / removed — and only
``restore <id> yes`` applies it. Applying replaces the store
entry and the economy files exactly with the snapshot (files the
backup never knew about are removed, and named in the preview),
then journals the restore itself into the audit log. A failed
apply rolls the economy files and the store entry back.

Scheduled backups ride the same builder: a per-group ``backup``
settings block (cadence off/daily/weekly, hour, keep) and an
hourly tick job that backs up every group whose cadence is due.
The tick is routine and silent — it journals to each group's
audit log and never posts anywhere.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import re
import secrets
import time
from pathlib import Path
from typing import Any

from .failures import record_failure

#: The backup file format. A file whose manifest carries another
#: version is refused, never guessed at.
BACKUP_FORMAT_VERSION = 1
BACKUP_KIND = "devon-group-backup"

#: Durable backup store, beside the group store itself.
BACKUP_DIRNAME = "group_backups"

#: Scheduler job (mirrors the season tick's ensure_*_job shape).
BACKUP_JOB = "group backup tick"
BACKUP_TOOL = "__group_backup_tick__"
BACKUP_SPEC = "every 1h"

#: Per-group backup settings block (see BACKUP_DEFAULTS). Cadence
#: defaults to off: backups are housekeeping the admin turns on,
#: never a surprise write.
BACKUP_DEFAULTS: dict[str, Any] = {
    "cadence": "off",      # off | daily | weekly
    "hour": 2,             # wall-clock hour the run becomes due
    "keep": 7,             # newest backups kept per group
    "last_run_ts": 0.0,
    "last_backup_id": "",
}

CADENCES = ("off", "daily", "weekly")
DEFAULT_KEEP = 7
MAX_KEEP = 30
_STORE_KEYS = ("settings", "members", "changes", "audit",
               "join_requests", "acceptance_state", "extras")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


class GroupBackupError(RuntimeError):
    """A backup/export/restore operation failed."""


class GroupBackupCorrupt(GroupBackupError):
    """A backup file is unreadable, the wrong version, for another
    group, or fails its checksum — refused, never restored."""


class GroupBackupRefused(GroupBackupError):
    """A backup/restore was refused for a stated reason (unknown
    group, channel, no admin proof, no such backup)."""


def _slug(group_key: str) -> str:
    from .group_surface import _slug as _surface_slug

    return _surface_slug(group_key)


def _economy_slug(group_id: str) -> str:
    """The economy directory name for a group — mirrors
    ``GroupEconomy._group_dir`` exactly (same sanitisation, same
    strip); the round-trip tests prove the mirror holds."""
    safe = "".join(c if c.isalnum() or c in ("_", "-", ".") else "_"
                   for c in str(group_id)).strip("_") or "group"
    return safe


def backup_root(surface: Any) -> Path:
    """Where this surface's backups live (beside its store)."""
    return Path(surface.store.path).parent / BACKUP_DIRNAME


def group_backup_dir(root: Path | str, group_key: str) -> Path:
    return Path(root) / _slug(group_key)


def _canonical(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _new_backup_id(now: float) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now))
    return f"{stamp}-{secrets.token_hex(3)}"


def _safe_name(name: str) -> bool:
    return bool(_NAME_RE.match(str(name))) and ".." not in str(name)


def backup_settings(surface: Any, group_key: str) -> dict[str, Any]:
    """The merged backup block for a group (defaults filled in)."""
    merged = surface.store.settings(group_key)
    block = dict(merged.get("backup") or {})
    out = dict(BACKUP_DEFAULTS)
    out.update({k: v for k, v in block.items() if k in out})
    try:
        out["hour"] = int(out["hour"])
    except (TypeError, ValueError):
        out["hour"] = BACKUP_DEFAULTS["hour"]
    out["hour"] = max(0, min(23, out["hour"]))
    try:
        out["keep"] = int(out["keep"])
    except (TypeError, ValueError):
        out["keep"] = DEFAULT_KEEP
    out["keep"] = max(1, min(MAX_KEEP, out["keep"]))
    if str(out["cadence"]) not in CADENCES:
        out["cadence"] = "off"
    try:
        out["last_run_ts"] = float(out["last_run_ts"] or 0.0)
    except (TypeError, ValueError):
        out["last_run_ts"] = 0.0
    out["last_backup_id"] = str(out["last_backup_id"] or "")
    return out


def backup_due(cfg: dict[str, Any], now: float) -> bool:
    """Is a scheduled run due? One run per interval, never before
    the configured wall-clock hour."""
    cadence = str(cfg.get("cadence") or "off")
    if cadence not in ("daily", "weekly"):
        return False
    interval = 86400.0 if cadence == "daily" else 604800.0
    last = float(cfg.get("last_run_ts") or 0.0)
    if now - last < interval and last:
        return False
    return time.localtime(now).tm_hour >= int(cfg.get("hour") or 0)


def _next_run_desc(cfg: dict[str, Any], now: float) -> str:
    cadence = str(cfg.get("cadence") or "off")
    if cadence not in ("daily", "weekly"):
        return "off — backups run only when you ask"
    interval = 86400.0 if cadence == "daily" else 604800.0
    last = float(cfg.get("last_run_ts") or 0.0)
    hour = int(cfg.get("hour") or 0)
    lt = time.localtime(now)
    candidate = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday,
                             hour, 0, 0, lt.tm_wday, lt.tm_yday,
                             lt.tm_isdst))
    if candidate <= now:
        candidate += interval
    if last and candidate - last < interval:
        candidate = last + interval
        clt = time.localtime(candidate)
        if clt.tm_hour != hour:
            candidate = time.mktime(
                (clt.tm_year, clt.tm_mon, clt.tm_mday, hour, 0, 0,
                 clt.tm_wday, clt.tm_yday, clt.tm_isdst))
            if candidate <= now:
                candidate += interval
    when = time.strftime("%d %b %Y %H:%M", time.localtime(candidate))
    return f"next run {when} ({cadence} at {hour:02d}:00)"


def _economy_files(surface: Any, group_key: str
                   ) -> tuple[dict[str, Any], dict[str, str], str]:
    """Read the group's economy file set: ({name: data},
    {name: reason} for files that would not parse, status).
    Status is "ok", or "degraded" with the reason when the ledger
    itself is unreadable (the economy's own integrity rule: an
    unreadable ledger is never read as empty)."""
    econ = surface.economy
    files, unreadable = econ.read_state_files(group_key)
    status = "ok"
    reason = ""
    try:
        econ.verify(group_key)
    except Exception as exc:  # noqa: BLE001 - the reason is the report
        status = "degraded"
        reason = str(exc)
    if "ledger" in unreadable:
        status = "degraded"
        reason = reason or f"ledger unreadable: {unreadable['ledger']}"
    return files, unreadable, status, reason


def create_backup(surface: Any, group_key: str, *, by: str,
                  reason: str, now: float | None = None) -> dict[str, Any]:
    """Snapshot one group into a new backup file. Returns the
    backup info (manifest + counts + path). Journals the run to
    the group's audit log. Raises GroupBackupRefused for an
    unknown group; a degraded economy still backs up (settings +
    roster + audit intact) and says so in the info and the reply."""
    now = now if now is not None else time.time()
    entry = surface.store.data.get(group_key)
    if entry is None:
        raise GroupBackupRefused(
            f"no group here answers to {group_key} — the backup "
            "covers groups Devon already serves.")
    if surface.is_channel(group_key):
        raise GroupBackupRefused(
            "channels have no roster or economy to back up — their "
            "queue and settings live in the same store and survive "
            "a kick.")
    store_payload: dict[str, Any] = {}
    store_payload["settings"] = copy.deepcopy(
        entry.get("settings") or {})
    store_payload["members"] = copy.deepcopy(entry.get("members") or {})
    store_payload["changes"] = copy.deepcopy(entry.get("changes") or [])
    store_payload["audit"] = copy.deepcopy(entry.get("audit") or [])
    store_payload["join_requests"] = copy.deepcopy(
        entry.get("join_requests") or {})
    store_payload["acceptance_state"] = copy.deepcopy(
        entry.get("acceptance_state") or {})
    store_payload["extras"] = copy.deepcopy(entry.get("extras") or {})
    econ_files, econ_unreadable, econ_status, econ_reason = \
        _economy_files(surface, group_key)
    payload = {"store": store_payload, "economy": econ_files}
    backup_id = _new_backup_id(now)
    root = backup_root(surface)
    gdir = group_backup_dir(root, group_key)
    gdir.mkdir(parents=True, exist_ok=True)
    path = gdir / f"{backup_id}.json"
    for _ in range(3):
        if not path.exists():
            break
        backup_id = _new_backup_id(now)
        path = gdir / f"{backup_id}.json"
    ledger_entries = econ_files.get("ledger")
    counts = {
        "members": len(store_payload["members"]),
        "ledger_entries": len(ledger_entries)
        if isinstance(ledger_entries, list) else 0,
        "audit_entries": len(store_payload["audit"]),
        "economy_files": len(econ_files),
        "settings_keys": len(store_payload["settings"]),
    }
    manifest = {
        "kind": BACKUP_KIND,
        "format_version": BACKUP_FORMAT_VERSION,
        "group_key": group_key,
        "backup_id": backup_id,
        "created_ts": now,
        "created_by": str(by or ""),
        "reason": str(reason or ""),
        "counts": counts,
        "economy_status": econ_status,
        "economy_reason": econ_reason,
        "economy_unreadable": sorted(econ_unreadable),
        "sha256": _sha256(_canonical(payload)),
    }
    body = {"manifest": manifest, "store": store_payload,
            "economy": econ_files}
    raw = json.dumps(body, ensure_ascii=False, indent=1).encode("utf-8")
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes(raw)
        tmp.replace(path)
    except OSError as exc:
        raise GroupBackupError(
            f"writing the backup for {group_key} failed: {exc} — "
            "nothing was backed up") from exc
    info: dict[str, Any] = {
        "backup_id": backup_id, "filename": path.name,
        "path": str(path), "group_key": group_key,
        "created_ts": now, "created_by": str(by or ""),
        "reason": str(reason or ""), "counts": counts,
        "size_bytes": len(raw), "economy_status": econ_status,
        "economy_reason": econ_reason, "pruned": [],
    }
    keep = backup_settings(surface, group_key)["keep"]
    info["pruned"] = _prune(gdir, keep, exclude=backup_id)
    surface._audit(group_key, "backup_created", by=str(by or ""),
                   backup_id=backup_id, reason=str(reason or ""),
                   economy_status=econ_status,
                   pruned=info["pruned"])
    try:
        surface.store.save()
    except OSError as exc:
        record_failure("group_backup.audit_save", exc)
    cfg = backup_settings(surface, group_key)
    cfg["last_run_ts"] = now
    cfg["last_backup_id"] = backup_id
    _write_backup_settings(surface, group_key, cfg)
    return info


def _prune(gdir: Path, keep: int, *, exclude: str) -> list[str]:
    """Keep the newest `keep` backups in a group dir. The new
    backup (excluded from the file list) counts toward `keep`, so
    one older file goes per new one past the limit. Order is
    disk mtime — backup ids share a timestamp prefix within a
    second, so the name alone does not order them. Returns the
    pruned ids."""
    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    files = sorted((p for p in gdir.glob("*.json")
                    if p.name != f"{exclude}.json"),
                   key=lambda p: (_mtime(p), p.name))
    excess = files[:max(0, len(files) + 1 - max(1, keep))]
    pruned: list[str] = []
    for path in excess:
        try:
            path.unlink()
            pruned.append(path.stem)
        except OSError as exc:
            record_failure("group_backup.prune", exc)
    return pruned


def _write_backup_settings(surface: Any, group_key: str,
                           cfg: dict[str, Any]) -> None:
    entry = surface.store.group(group_key)
    stored = entry.setdefault("settings", {})
    stored["backup"] = {
        "cadence": str(cfg.get("cadence") or "off"),
        "hour": int(cfg.get("hour") or 0),
        "keep": int(cfg.get("keep") or DEFAULT_KEEP),
        "last_run_ts": float(cfg.get("last_run_ts") or 0.0),
        "last_backup_id": str(cfg.get("last_backup_id") or ""),
    }
    surface.store.save()


def list_backups(surface: Any, group_key: str
                 ) -> tuple[list[dict[str, Any]], int]:
    """(readable backup infos newest-first, corrupt-file count).
    A file that will not parse is counted, never listed as a
    backup it is not."""
    gdir = group_backup_dir(backup_root(surface), group_key)
    infos: list[dict[str, Any]] = []
    corrupt = 0
    if not gdir.exists():
        return infos, corrupt
    for path in sorted(gdir.glob("*.json"), reverse=True):
        try:
            # The same validation a restore runs — kind, version,
            # group, checksum — so a file changed since it was
            # written counts as corrupt here instead of listing
            # as a good backup and refusing only at restore time.
            body = load_backup(surface, group_key, path.stem)
            manifest = body["manifest"]
            infos.append({
                "backup_id": str(manifest.get("backup_id")
                                 or path.stem),
                "filename": path.name, "path": str(path),
                "group_key": str(manifest.get("group_key") or ""),
                "created_ts": float(manifest.get("created_ts") or 0.0),
                "created_by": str(manifest.get("created_by") or ""),
                "reason": str(manifest.get("reason") or ""),
                "counts": dict(manifest.get("counts") or {}),
                "size_bytes": path.stat().st_size,
                "economy_status": str(manifest.get("economy_status")
                                      or "ok"),
                "economy_reason": str(manifest.get("economy_reason")
                                      or ""),
            })
        except (GroupBackupCorrupt, GroupBackupRefused, OSError,
                ValueError, KeyError, TypeError):
            corrupt += 1
    infos.sort(key=lambda i: i["created_ts"], reverse=True)
    return infos, corrupt


def load_backup(surface: Any, group_key: str, backup_id: str
                ) -> dict[str, Any]:
    """Read + validate one stored backup: right kind, right
    version, right group, checksum intact. Anything else is
    GroupBackupCorrupt with the plain reason."""
    gdir = group_backup_dir(backup_root(surface), group_key)
    path = gdir / f"{backup_id}.json"
    if not path.exists():
        raise GroupBackupRefused(
            f"no backup {backup_id} for {group_key} — "
            "/gbackup lists the ones on file.")
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GroupBackupCorrupt(
            f"backup {backup_id} will not parse ({exc}) — refused, "
            "never restored") from exc
    if not isinstance(body, dict):
        raise GroupBackupCorrupt(
            f"backup {backup_id} is not an object — refused")
    manifest = body.get("manifest")
    if not isinstance(manifest, dict):
        raise GroupBackupCorrupt(
            f"backup {backup_id} carries no manifest — refused")
    if manifest.get("kind") != BACKUP_KIND:
        raise GroupBackupCorrupt(
            f"backup {backup_id} is not a group backup — refused")
    if manifest.get("format_version") != BACKUP_FORMAT_VERSION:
        raise GroupBackupCorrupt(
            f"backup {backup_id} is format "
            f"{manifest.get('format_version')} — this build reads "
            f"format {BACKUP_FORMAT_VERSION} only")
    if manifest.get("group_key") != group_key:
        raise GroupBackupCorrupt(
            f"backup {backup_id} belongs to "
            f"{manifest.get('group_key')} — a backup restores into "
            "its own group only")
    store = body.get("store")
    economy = body.get("economy")
    if not isinstance(store, dict) or not isinstance(economy, dict):
        raise GroupBackupCorrupt(
            f"backup {backup_id} is missing its store or economy "
            "half — refused")
    for name in economy:
        if not _safe_name(name):
            raise GroupBackupCorrupt(
                f"backup {backup_id} names an unsafe economy file "
                f"{name!r} — refused")
    expect = _sha256(_canonical({"store": store, "economy": economy}))
    if manifest.get("sha256") != expect:
        raise GroupBackupCorrupt(
            f"backup {backup_id} fails its checksum — the file was "
            "changed since it was written, refused")
    return body


def _balances_from_ledger(entries: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        member = str(entry.get("member") or "")
        try:
            amount = int(entry.get("amount") or 0)
        except (TypeError, ValueError):
            continue
        out[member] = out.get(member, 0) + amount
    return {m: b for m, b in out.items() if b != 0}


def _name_of(member: dict[str, Any]) -> str:
    return (str(member.get("username") or "")
            or str(member.get("name") or "")
            or str(member.get("user_id") or ""))


def _named(ids: list[str], members: dict[str, Any],
           limit: int = 5) -> str:
    shown = [_name_of(members[i]) for i in ids[:limit]
             if isinstance(members.get(i), dict)]
    extra = len(ids) - len(shown)
    text = ", ".join(shown)
    if extra > 0:
        text += f" (+{extra} more)"
    return text or "—"


def preview_restore(surface: Any, group_key: str,
                    backup: dict[str, Any]) -> dict[str, Any]:
    """The diff a restore would apply, as data + plain lines.
    Nothing is changed by this call."""
    manifest = backup["manifest"]
    bstore = backup["store"]
    becon = backup["economy"]
    entry = surface.store.data.get(group_key) or {}
    now_members = entry.get("members") or {}
    bak_members = bstore.get("members") or {}
    now_ids = set(now_members)
    bak_ids = set(bak_members)
    added = sorted(bak_ids - now_ids)
    removed = sorted(now_ids - bak_ids)
    changed = sorted(
        i for i in now_ids & bak_ids
        if (str((now_members[i] or {}).get("username") or "")
            != str((bak_members[i] or {}).get("username") or ""))
        or (str((now_members[i] or {}).get("name") or "")
            != str((bak_members[i] or {}).get("name") or "")))
    try:
        now_balances = surface.economy.balances(group_key)
        balances_note = ""
    except Exception as exc:  # noqa: BLE001 - the ledger state is the report
        now_balances = {}
        balances_note = (f"live balances unreadable right now "
                         f"({exc}) — the restore replaces the "
                         f"ledger, so they come back from the backup")
    bak_balances = _balances_from_ledger(becon.get("ledger"))
    moved = sorted(
        ((m, now_balances.get(m, 0), bak_balances.get(m, 0))
         for m in set(now_balances) | set(bak_balances)
         if now_balances.get(m, 0) != bak_balances.get(m, 0)),
        key=lambda t: -abs(t[1] - t[2]))[:5]
    moved_total = sum(1 for m in set(now_balances) | set(bak_balances)
                      if now_balances.get(m, 0) != bak_balances.get(m, 0))
    now_settings = entry.get("settings") or {}
    bak_settings = bstore.get("settings") or {}
    # The "backup" block is bookkeeping — creating the backup
    # itself stamps last_run_ts/last_backup_id into it, so it
    # would show as changed on every preview. It is not group
    # state; the restore's own bookkeeping replaces it anyway.
    changed_keys = sorted(
        k for k in (set(now_settings) | set(bak_settings)) - {"backup"}
        if _canonical(now_settings.get(k)) != _canonical(bak_settings.get(k)))
    try:
        now_econ, _unreadable = surface.economy.read_state_files(group_key)
        now_econ_names = set(now_econ)
    except Exception:  # noqa: BLE001 - the file diff degrades, the restore still previews
        now_econ_names = set()
    bak_econ_names = set(becon)
    econ_replace = sorted(now_econ_names & bak_econ_names)
    econ_add = sorted(bak_econ_names - now_econ_names)
    econ_remove = sorted(now_econ_names - bak_econ_names)
    lines = [
        f"members: {len(now_members)} now → {len(bak_members)} in "
        f"the backup",
    ]
    if added:
        lines.append(f"  back in: {_named(added, bak_members)}")
    if removed:
        lines.append(f"  gone: {_named(removed, now_members)}")
    if changed:
        lines.append(f"  renamed: {_named(changed, bak_members)}")
    lines.append(
        f"balances: treasury {now_balances.get('', 0)} → "
        f"{bak_balances.get('', 0)} sparks · {moved_total} holder(s) "
        f"move")
    for member, now_bal, bak_bal in moved:
        lines.append(f"  {_name_of(now_members.get(member) or bak_members.get(member) or {'user_id': member})}: "
                     f"{now_bal} → {bak_bal}")
    if balances_note:
        lines.append(f"  {balances_note}")
    lines.append(
        f"settings: {len(changed_keys)} key(s) change"
        + (f" ({', '.join(changed_keys[:8])})" if changed_keys
           else " — none differ"))
    lines.append(
        f"economy files: {len(econ_replace)} replaced"
        + (f", {len(econ_add)} added" if econ_add else "")
        + (f", {len(econ_remove)} removed "
           f"({', '.join(econ_remove[:5])})" if econ_remove else ""))
    lines.append(
        f"audit log: {len(entry.get('audit') or [])} entries now → "
        f"{len(bstore.get('audit') or [])} in the backup (the "
        f"restore itself is journaled after)")
    return {
        "backup_id": str(manifest.get("backup_id") or ""),
        "created_ts": float(manifest.get("created_ts") or 0.0),
        "created_by": str(manifest.get("created_by") or ""),
        "reason": str(manifest.get("reason") or ""),
        "members": {"now": len(now_members), "backup": len(bak_members),
                    "added": added, "removed": removed,
                    "changed": changed},
        "balances": {"now": now_balances, "backup": bak_balances,
                     "moved": moved, "moved_total": moved_total},
        "settings_changed_keys": changed_keys,
        "economy": {"replace": econ_replace, "add": econ_add,
                    "remove": econ_remove},
        "audit": {"now": len(entry.get("audit") or []),
                  "backup": len(bstore.get("audit") or [])},
        "lines": lines,
    }


def apply_restore(surface: Any, group_key: str,
                  backup: dict[str, Any], *, by: str) -> dict[str, Any]:
    """Apply a validated backup to its group. Validates again
    (checksum + group key), then replaces the store entry and the
    economy files exactly with the snapshot; a failure mid-apply
    rolls both back and raises. Journals the restore itself."""
    manifest = backup["manifest"]
    backup_id = str(manifest.get("backup_id") or "")
    if manifest.get("group_key") != group_key:
        raise GroupBackupRefused(
            f"backup {backup_id} belongs to "
            f"{manifest.get('group_key')} — a backup restores into "
            "its own group only")
    expect = _sha256(_canonical({"store": backup["store"],
                                 "economy": backup["economy"]}))
    if manifest.get("sha256") != expect:
        raise GroupBackupCorrupt(
            f"backup {backup_id} fails its checksum — refused")
    entry = surface.store.data.get(group_key)
    if entry is None:
        raise GroupBackupRefused(
            f"no group here answers to {group_key}.")
    old_entry = copy.deepcopy(entry)
    try:
        old_econ, _unreadable = surface.economy.read_state_files(group_key)
    except Exception:  # noqa: BLE001 - no readable economy to roll back to
        old_econ = {}
    bstore = backup["store"]
    new_entry = {
        "settings": copy.deepcopy(bstore.get("settings") or {}),
        "members": copy.deepcopy(bstore.get("members") or {}),
        "changes": copy.deepcopy(bstore.get("changes") or []),
        "audit": copy.deepcopy(bstore.get("audit") or []),
        "join_requests": copy.deepcopy(bstore.get("join_requests") or {}),
        "acceptance_state": copy.deepcopy(
            bstore.get("acceptance_state") or {}),
        "extras": copy.deepcopy(bstore.get("extras") or {}),
    }
    for key in entry:
        if key not in new_entry:
            new_entry[key] = copy.deepcopy(entry[key])
    applied_econ = False
    try:
        surface.economy.write_state_files(group_key, backup["economy"])
        applied_econ = True
        surface.store.data[group_key] = new_entry
        surface.store.save()
    except Exception as exc:
        if applied_econ:
            try:
                surface.economy.write_state_files(group_key, old_econ)
            except Exception as rollback_exc:  # noqa: BLE001 - the rollback failure is the report
                record_failure("group_backup.restore_rollback_econ",
                               rollback_exc)
        try:
            surface.store.data[group_key] = old_entry
            surface.store.save()
        except Exception as rollback_exc:  # noqa: BLE001 - same
            record_failure("group_backup.restore_rollback_store",
                           rollback_exc)
        raise GroupBackupError(
            f"restoring backup {backup_id} failed halfway ({exc}) "
            "— the group's live state was rolled back") from exc
    surface._audit(group_key, "backup_restored", by=str(by or ""),
                   backup_id=backup_id,
                   backup_created_ts=float(
                       manifest.get("created_ts") or 0.0),
                   members=len(new_entry["members"]))
    try:
        surface.store.save()
    except OSError as exc:
        record_failure("group_backup.restore_audit_save", exc)
    try:
        surface.refresh_moderator(group_key)
    except Exception as exc:  # noqa: BLE001 - the restore landed; the moderator refresh is best-effort
        record_failure("group_backup.restore_moderator", exc)
    return {
        "backup_id": backup_id, "group_key": group_key,
        "members": len(new_entry["members"]),
        "ledger_entries": len(backup["economy"].get("ledger") or [])
        if isinstance(backup["economy"].get("ledger"), list) else 0,
        "economy_files": len(backup["economy"]),
    }


def delete_backup(surface: Any, group_key: str, backup_id: str, *,
                  by: str) -> dict[str, Any]:
    """Delete one stored backup (the live state is untouched).
    Journals the deletion."""
    gdir = group_backup_dir(backup_root(surface), group_key)
    path = gdir / f"{backup_id}.json"
    if not path.exists():
        raise GroupBackupRefused(
            f"no backup {backup_id} for {group_key}.")
    try:
        path.unlink()
    except OSError as exc:
        raise GroupBackupError(
            f"deleting backup {backup_id} failed: {exc}") from exc
    surface._audit(group_key, "backup_deleted", by=str(by or ""),
                   backup_id=backup_id)
    try:
        surface.store.save()
    except OSError as exc:
        record_failure("group_backup.delete_audit_save", exc)
    return {"backup_id": backup_id, "group_key": group_key}


def set_backup_schedule(surface: Any, group_key: str, *,
                        cadence: str, hour: int | None = None,
                        keep: int | None = None,
                        by: str = "") -> dict[str, Any]:
    """Set one group's backup cadence. Returns the merged block
    plus the next-run line for the reply."""
    cadence = str(cadence or "").lower()
    if cadence not in CADENCES:
        raise GroupBackupRefused(
            f"cadence is off, daily or weekly — not {cadence!r}.")
    cfg = backup_settings(surface, group_key)
    cfg["cadence"] = cadence
    if hour is not None:
        if not 0 <= int(hour) <= 23:
            raise GroupBackupRefused(
                f"hour is 0–23 — not {hour}.")
        cfg["hour"] = int(hour)
    if keep is not None:
        if not 1 <= int(keep) <= MAX_KEEP:
            raise GroupBackupRefused(
                f"keep is 1–{MAX_KEEP} — not {keep}.")
        cfg["keep"] = int(keep)
    _write_backup_settings(surface, group_key, cfg)
    surface._audit(group_key, "backup_schedule", by=str(by or ""),
                   cadence=cadence, hour=cfg["hour"], keep=cfg["keep"])
    try:
        surface.store.save()
    except OSError as exc:
        record_failure("group_backup.schedule_audit_save", exc)
    return {"settings": cfg,
            "next": _next_run_desc(cfg, time.time())}


def run_group_backup_tick_once(*, context: Any = None,
                               surface: Any = None,
                               workspace_dir: str | Path | None = None,
                               now: float | None = None) -> dict[str, Any]:
    """One scheduled backup pass: every group whose cadence is due
    gets a fresh backup (reason "scheduled"). Routine and silent —
    nothing is posted anywhere; each run journals to its group's
    audit log. One bad group never kills the pass."""
    now = now if now is not None else time.time()
    if surface is None:
        ws = workspace_dir
        if ws is None and context is not None:
            ws = getattr(getattr(context, "settings", None),
                         "workspace_dir", ".") or "."
        from .group_surface import default_surface

        surface = default_surface(ws or ".")
    summary: dict[str, Any] = {"ok": True, "backed_up": [],
                               "failed": [], "skipped": 0}
    for key in sorted(surface.store.data.keys()):
        try:
            if surface.is_channel(key):
                continue
        except Exception:  # noqa: BLE001 - an unreadable key is skipped, not fatal
            continue
        try:
            cfg = backup_settings(surface, key)
        except Exception as exc:  # noqa: BLE001 - one bad group never kills the pass
            record_failure("group_backup.tick_settings", exc)
            summary["failed"].append({"group": key,
                                     "error": str(exc)})
            continue
        if not backup_due(cfg, now):
            summary["skipped"] += 1
            continue
        try:
            info = create_backup(surface, key, by="scheduler",
                                 reason="scheduled", now=now)
            summary["backed_up"].append(
                {"group": key, "backup_id": info["backup_id"]})
        except Exception as exc:  # noqa: BLE001 - one bad group never kills the pass
            record_failure("group_backup.tick", exc)
            summary["failed"].append({"group": key,
                                     "error": str(exc)})
    return summary


def ensure_group_backup_job(scheduler: Any, *,
                            spec: str = BACKUP_SPEC) -> Any:
    """Register the durable group-backup tick job. Idempotent —
    mirrors ``ensure_season_tick_job``: an existing job at the
    requested cadence is returned as-is, a stale one is replaced.
    Best-effort: never raises."""
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name == BACKUP_JOB:
                job_spec = (job.get("spec") if isinstance(job, dict)
                            else getattr(job, "spec", None))
                if job_spec == spec:
                    return job
                with contextlib.suppress(Exception):
                    job_id = (job.get("id") if isinstance(job, dict)
                              else getattr(job, "id", None))
                    scheduler.remove(job_id)
                break
        return scheduler.add(BACKUP_JOB, spec, "tool",
                             {"tool": BACKUP_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
