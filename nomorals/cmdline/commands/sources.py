"""``nm sources`` — the active source-candidate intake (mp3.pm lesson).

The ONE registry (``media/source_registry.py``) only grew
passively: ``note_candidate`` records hosts already encountered
on the real path, so a source nobody ever pastes is never
discovered.  This is the operator door for the active intake —
it adds no second implementation of anything:

* ``nm sources add <host|url> [--class <class>]`` drops a named
  host into the candidate queue through
  :meth:`SourceRegistry.note_named_candidate` — same queue, same
  store, same health loop as the passive intake.  A named
  candidate is untrusted until :meth:`verify_health` promotes
  it (origin ``grown``); nothing here promotes anything.
* ``nm sources candidates`` lists the pending queue with status
  and provenance (who named it, when, what class, via which
  path, last verification error) from
  :meth:`SourceRegistry.candidate_list` — the same rows the
  registry status surface renders.

Operator text carries hosts, counts, reasons and times only.
The registry is the shared ``DEFAULT_REGISTRY`` bound to the
context's KV store — the same binding every other consumer
uses — so a candidate added here persists in the registry's
own store and the health loop sees it on its next run.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.parse
from typing import Any

from ..emit import _emit

#: The classes ``nm sources add`` accepts.  The label is the
#: candidate's class AND (unless it maps to the download pair)
#: its registry kind — see ``_NAMED_CLASS_KINDS`` in the
#: registry module.
CLASSES = ("downloads", "music", "film", "news", "books", "maps",
           "general")


def _registry_for_context(context: Any) -> Any:
    """The ONE registry for this context: an injected registry
    wins (tests); otherwise the shared ``DEFAULT_REGISTRY``
    bound to the context's KV store — the same binding every
    other consumer uses — and only when the context carries a
    database, so db-less callers keep a memory-only registry
    for the session.  Never raises (None on failure)."""
    injected = getattr(context, "source_registry", None)
    if injected is not None:
        return injected
    try:
        from ...media.source_registry import DEFAULT_REGISTRY

        reg = DEFAULT_REGISTRY
        db = getattr(context, "db", None)
        if db is not None:
            try:
                from ...storage.kv import KVStore

                reg.bind_store(KVStore(db))
            except Exception:  # noqa: BLE001 - persistence is an
                # enhancement; its absence never sinks the command
                pass
        return reg
    except Exception:  # noqa: BLE001
        return None


def _when(ts: Any) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S",
                             time.localtime(float(ts)))
    except (TypeError, ValueError, OverflowError, OSError):
        return "never"


def _host_of(text: str) -> str:
    """The bare host of a host-or-URL argument, for the
    already-known / shed disambiguation below."""
    try:
        raw = str(text or "").strip()
        host = ""
        if "://" in raw:
            host = urllib.parse.urlparse(raw).hostname or ""
        elif "/" in raw:
            host = raw.split("/", 1)[0]
        else:
            host = raw
        host = host.lower().split(":")[0]
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:  # noqa: BLE001
        return ""


def _plausible_host(text: str) -> bool:
    """A host with a dot that is not a bare IP — the same bar
    ``note_candidate`` applies (a shed-window refusal and an
    implausible host are both False, and the caller tells
    them apart with this)."""
    host = _host_of(text)
    if not host or "." not in host:
        return False
    return not all(part.isdigit() for part in host.split("."))


def _render_candidates(rows: list[dict[str, Any]], *,
                       persisted: bool) -> str:
    if not rows:
        return ("no candidates pending — drop one in with:\n"
                "  nm sources add <host> [--class <class>]\n"
                f"registry: {'persisted' if persisted else 'memory-only'}")
    lines = [f"source candidates — {len(rows)} pending, untrusted "
             "until the health loop verifies them"]
    for row in rows:
        domain = str(row.get("domain") or "")
        health = str(row.get("health") or "unknown")
        named = str(row.get("named_by") or "")
        cls = str(row.get("class") or "")
        via = str(row.get("via") or "")
        line = (f"  ? {domain} [candidate/{health}]")
        if cls:
            line += f" class={cls}"
        if named:
            line += f" named by {named}"
        if via:
            line += f" via {via}"
        line += (f" seen {int(row.get('seen') or 0)}x"
                 f" first {_when(row.get('first_seen'))}")
        lines.append(line)
        last_error = str(row.get("last_error") or "")
        if last_error:
            lines.append(f"    last verification error: {last_error}")
    lines.append(f"registry: {'persisted' if persisted else 'memory-only'}")
    return "\n".join(lines)


def _candidates(args: argparse.Namespace, context: Any) -> int:
    reg = _registry_for_context(context)
    if reg is None:
        print("sources: the source registry is unavailable",
              file=sys.stderr)
        return 1
    rows = reg.candidate_list()
    persisted = bool(getattr(reg, "_store", None) is not None)
    _emit(args, {"candidates": rows, "count": len(rows),
                 "persisted": persisted},
          _render_candidates(rows, persisted=persisted))
    return 0


def _add(args: argparse.Namespace, context: Any) -> int:
    reg = _registry_for_context(context)
    if reg is None:
        print("sources: the source registry is unavailable",
              file=sys.stderr)
        return 1
    host = str(getattr(args, "host", "") or "")
    source_class = str(getattr(args, "source_class", "") or "downloads")
    if reg.note_named_candidate(host, named_by="owner",
                                source_class=source_class):
        row = next((r for r in reg.candidate_list()
                    if r.get("domain") == _host_of(host)), None)
        domain = str((row or {}).get("domain") or _host_of(host) or host)
        _emit(args, {"added": True, "host": domain,
                     "class": source_class, "candidate": row},
              f"candidate recorded: {domain} [class {source_class}, "
              "named by owner] — untrusted until the health loop "
              "verifies it")
        return 0
    # Not a new candidate — say which of the four refusals it is.
    desc = reg.for_domain(host)
    if desc is not None:
        origin = reg.origin_for(desc)
        health = reg.health_for(desc)
        _emit(args, {"added": False, "host": desc.domain,
                     "reason": "already_registered",
                     "origin": origin, "health": health},
              f"already a registered source: {desc.domain} "
              f"[{origin}/{health}] — nothing to add")
        return 0
    row = next((r for r in reg.candidate_list()
                if r.get("domain") == _host_of(host)), None)
    if row is not None:
        _emit(args, {"added": False, "host": row.get("domain"),
                     "reason": "already_candidate", "candidate": row},
              f"already a candidate: {row.get('domain')} "
              f"[candidate/{row.get('health')}] "
              f"class={row.get('class') or '?'} "
              f"named by {row.get('named_by') or '?'} "
              f"via {row.get('via') or '?'} "
              f"seen {int(row.get('seen') or 0)}x")
        return 0
    if _plausible_host(host):
        shed = [e for e in (reg.status().get("shed_list") or [])
                if isinstance(e, dict)
                and e.get("domain") == _host_of(host)]
        if shed:
            entry = shed[-1]
            _emit(args, {"added": False, "host": _host_of(host),
                         "reason": "shed_window",
                         "shed_reason": entry.get("reason"),
                         "shed_at": entry.get("at")},
                  f"shed {_when(entry.get('at'))} "
                  f"({entry.get('reason') or 'failed verification'}) "
                  "— re-intake opens 7 days after the shed")
            return 0
    print(f"sources add: not a plausible host: {host!r}",
          file=sys.stderr)
    return 2


def _cmd_sources(args: argparse.Namespace, context: Any) -> int:
    """Route ``nm sources`` — candidate intake + listing."""
    action = getattr(args, "sources_action", "")
    try:
        if action == "candidates":
            return _candidates(args, context)
        if action == "add":
            return _add(args, context)
    except Exception as exc:  # noqa: BLE001 - CLI last line of defence
        print(f"sources: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"unknown sources action: {action}", file=sys.stderr)
    return 2
