"""Watchtower brain tool (Downdetector native build, 52-websites #45).

``service_watch`` — watch an external service the owner depends on
and ask whether it is down, from this host's own probes plus public
status-page feeds. Ask it as "is Binance down?" or "watch X and tell
me if it drops". Every verdict carries the narrow claim: one host's
probes + public feeds, not Downdetector's crowd of millions.
"""

from __future__ import annotations

from typing import Any

from ..core.policy import Capability
from ..sources.watchtower import (
    KINDS,
    VANTAGE,
    WatchTarget,
    WatchtowerStore,
    _resolve_home,
    check_target,
    evaluate_verdict,
    format_watchtower_reply,
)

__all__ = ["register", "service_watch"]


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _store(context: Any) -> WatchtowerStore:
    home = _home_from_context(context)
    return WatchtowerStore(_resolve_home(home, context))


def _status_for(store: WatchtowerStore, target: WatchTarget,
                *, now: float | None = None) -> dict[str, Any]:
    verdict = evaluate_verdict(target.name,
                               store.samples_for(target.name), now=now)
    result = {"ok": True, "target": target.name, "state": verdict.state,
              "reason": verdict.reason, "evidence": verdict.evidence,
              "vantage": VANTAGE, "error": ""}
    result["reply"] = format_watchtower_reply(result)
    return result


def service_watch(action: str = "status", name: str = "", url: str = "",
                  kind: str = "http", critical: bool = False,
                  signals: list[str] | None = None,
                  context: Any = None) -> dict[str, Any]:
    """The ``service_watch`` implementation (also called by the tool
    wrapper below, so tests can drive it without a registry)."""
    import time

    action = (action or "status").strip().lower()
    name = (name or "").strip()
    try:
        store = _store(context)
        if action == "watch":
            if not name or not url:
                return {"ok": False, "error": "watch needs a name and a url",
                        "reply": "To watch a service I need a name and a url."}
            kind = (kind or "http").strip().lower()
            if kind not in KINDS:
                return {"ok": False,
                        "error": f"unknown probe kind: {kind!r}",
                        "reply": (f"I can probe http, tcp or feed — "
                                  f"{kind!r} is not one of those.")}
            target = WatchTarget(name=name, url=url.strip(), kind=kind,
                                 critical=bool(critical),
                                 added_at=time.time())
            if not store.add_target(target):
                return {"ok": False,
                        "error": f"{name} is already watched",
                        "reply": f"{name} is already on the watchlist."}
            return {"ok": True, "target": name, "state": "unknown",
                    "reason": "insufficient baseline (0/30 samples) — "
                              "report-only, never a confident verdict",
                    "evidence": {}, "vantage": VANTAGE, "error": "",
                    "reply": (f"Watching {name} ({url.strip()}). "
                              f"No verdict yet — the baseline needs 30 "
                              f"probes first, so early answers stay honest.")}
        if action == "unwatch":
            if not name:
                return {"ok": False, "error": "unwatch needs a name",
                        "reply": "Which service should I stop watching?"}
            if not store.remove_target(name):
                return {"ok": False, "error": f"{name} is not watched",
                        "reply": f"{name} is not on the watchlist."}
            return {"ok": True, "target": name,
                    "reply": f"Stopped watching {name}."}
        if action == "list":
            targets = store.list_targets()
            rows = []
            for target in targets:
                verdict = evaluate_verdict(
                    target.name, store.samples_for(target.name))
                rows.append({"name": target.name, "url": target.url,
                             "kind": target.kind,
                             "critical": target.critical,
                             "state": verdict.state,
                             "samples": verdict.samples})
            if not rows:
                return {"ok": True, "targets": [],
                        "reply": "Nothing is on the watchlist yet."}
            lines = ["Watchlist:"]
            for row in rows:
                flag = " (critical)" if row["critical"] else ""
                lines.append(f"- {row['name']}: {row['state']}"
                             f"{flag} — {row['samples']} probes")
            return {"ok": True, "targets": rows,
                    "reply": "\n".join(lines)}
        if action == "check":
            if not name:
                return {"ok": False, "error": "check needs a name",
                        "reply": "Which service should I check?"}
            target = store.get_target(name)
            if target is None:
                return {"ok": False, "error": f"{name} is not watched",
                        "reply": (f"{name} is not on the watchlist — "
                                  f"watch it first.")}
            result = check_target(store, target, signals=signals)
            result["reply"] = format_watchtower_reply(result)
            return result
        if action == "status":
            if name:
                target = store.get_target(name)
                if target is None:
                    return {"ok": False,
                            "error": f"{name} is not watched",
                            "reply": (f"{name} is not on the watchlist — "
                                      f"watch it first.")}
                return _status_for(store, target)
            targets = store.list_targets()
            if not targets:
                return {"ok": True, "targets": [],
                        "reply": "Nothing is on the watchlist yet."}
            rows = [_status_for(store, t) for t in targets]
            lines = ["Service status:"]
            for row in rows:
                lines.append(f"- {row['target']}: {row['reply']}")
            return {"ok": True, "targets": rows,
                    "reply": "\n".join(lines)}
        return {"ok": False, "error": f"unknown action: {action!r}",
                "reply": ("I can watch, unwatch, list, status or check "
                          "a service.")}
    except Exception as exc:  # noqa: BLE001 — the tool never raises
        return {"ok": False, "error": f"service_watch failed: {exc}",
                "reply": f"Could not do that: {exc}"}


def register(registry: Any) -> None:
    """Attach the ``service_watch`` tool to a registry."""
    context = registry.context

    @registry.register(
        "service_watch",
        description=(
            "Watch an external service the owner depends on and say "
            "whether it is down, from this host's own probes plus "
            "public status-page feeds — Downdetector's baseline-"
            "deviation idea without its crowd. Ask it as 'is Binance "
            "down?' or 'watch X and tell me if it drops'. "
            "action=<watch|unwatch|list|status|check>; "
            "watch needs name=<service name> url=<probe url> "
            "kind=<http|tcp|feed, default http> critical=<true when "
            "the owner trades through it — a critical target going "
            "down pages him once>; status/check take name=<service>. "
            "Verdicts are ok / possible / down / unknown; below 30 "
            "probes the verdict is honestly 'unknown', never a "
            "confident call. Every verdict is labelled single-host "
            "probe + public feed, not crowd reports."
        ),
        capability=Capability.NET_OUT,
    )
    def service_watch_tool(action: str = "status", name: str = "",
                           url: str = "", kind: str = "http",
                           critical: bool = False,
                           signals: list[str] | None = None) -> dict[str, Any]:
        return service_watch(action=action, name=name, url=url,
                             kind=kind, critical=critical,
                             signals=signals, context=context)
