"""Watch-offers brain tool (JustWatch native build, 52-websites #5).

``watch_offers`` — where can a film or show be watched, legally, in
the owner's country: which services stream it, rent it or sell it,
at what price and quality, with the date each offer was last
verified. Ask it as "where can I watch Dune in Nigeria?" Every
answer carries checked_at per offer plus a staleness label — an
availability claim without a date is exactly the ungrounded shape
Devon avoids elsewhere. Watchlisted titles are re-checked daily;
a new or leaving offer pages the owner once.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.policy import Capability
from ..media.film_sources import (
    DEFAULT_TERRITORY,
    OfferStore,
    _resolve_home,
    default_offer_adapters,
    format_offers_reply,
    reconcile_offers,
    where_to_watch,
)

__all__ = ["register", "watch_offers"]


def _home_from_context(context: Any) -> str | None:
    settings = getattr(context, "settings", None) if context else None
    if settings is None:
        return None
    home = getattr(settings, "home_path", None) or getattr(
        settings, "home", None)
    return str(home) if home else None


def _store(context: Any) -> OfferStore:
    return OfferStore(_resolve_home(_home_from_context(context), context))


def watch_offers(action: str = "where", title: str = "",
                 territory: str = DEFAULT_TERRITORY,
                 refresh: bool = False,
                 context: Any = None) -> dict[str, Any]:
    """The ``watch_offers`` implementation (also called by the tool
    wrapper below, so tests can drive it without a registry)."""
    action = (action or "where").strip().lower()
    title = (title or "").strip()
    territory = (territory or DEFAULT_TERRITORY).strip().upper()
    try:
        store = _store(context)
        if action == "where":
            if not title:
                return {"ok": False, "error": "where needs a title",
                        "reply": "Which film or show should I look up?"}
            result = where_to_watch(
                store, title, territory,
                adapters=default_offer_adapters(context),
                context=context, refresh=bool(refresh))
            result["reply"] = format_offers_reply(result)
            return result
        if action == "watch":
            if not title:
                return {"ok": False, "error": "watch needs a title",
                        "reply": "Which film or show should I watch for?"}
            if not store.watch(title, territory, now=time.time()):
                return {"ok": False,
                        "error": f"{title} is already watched in {territory}",
                        "reply": (f"{title} is already on the watchlist "
                                  f"for {territory}.")}
            return {"ok": True, "title": title, "territory": territory,
                    "reply": (f"Watching {title} in {territory} — I'll "
                              f"tell you when a new service picks it up "
                              f"or one drops it.")}
        if action == "unwatch":
            if not title:
                return {"ok": False, "error": "unwatch needs a title",
                        "reply": "Which film or show should I stop watching?"}
            if not store.unwatch(title, territory):
                return {"ok": False,
                        "error": f"{title} is not watched in {territory}",
                        "reply": (f"{title} is not on the watchlist "
                                  f"for {territory}.")}
            return {"ok": True, "title": title, "territory": territory,
                    "reply": f"Stopped watching {title} in {territory}."}
        if action == "list":
            watchlist = store.list_watchlist()
            if not watchlist:
                return {"ok": True, "watchlist": [],
                        "reply": "Nothing is on the watchlist yet."}
            rows = []
            lines = ["Watchlist:"]
            for watched_title, code in watchlist:
                offers = store.offers_for(watched_title, code)
                stale = any(
                    OfferStore.staleness(o) == "stale" for o in offers)
                rows.append({"title": watched_title, "territory": code,
                             "offers": len(offers), "stale": stale})
                lines.append(f"- {watched_title} ({code}): "
                             f"{len(offers)} offer(s)"
                             + (" — stale, needs a refresh" if stale else ""))
            return {"ok": True, "watchlist": rows,
                    "reply": "\n".join(lines)}
        if action == "reconcile":
            if not title:
                return {"ok": False, "error": "reconcile needs a title",
                        "reply": "Which film or show should I re-check?"}
            result = reconcile_offers(
                store, title, territory,
                adapters=default_offer_adapters(context),
                context=context)
            result["reply"] = format_offers_reply(result)
            if result.get("ok"):
                new = result.get("new") or []
                leaving = result.get("leaving") or []
                if new or leaving:
                    result["reply"] += "\n" + "; ".join(
                        ([f"new: {', '.join(new)}"] if new else [])
                        + ([f"leaving: {', '.join(leaving)}"]
                           if leaving else []))
            return result
        if action == "changes":
            unseen = store.unseen_changes()
            if not unseen:
                return {"ok": True, "changes": [],
                        "reply": "No new availability changes."}
            rows = [{"title": c.title, "territory": c.territory,
                     "service": c.service, "kind": c.kind,
                     "at": c.at} for c in unseen]
            lines = ["Availability changes:"]
            for row in rows:
                when = time.strftime(
                    "%Y-%m-%d", time.gmtime(row["at"]))
                lines.append(f"- {row['title']}: {row['kind']} — "
                             f"{row['service']} ({when})")
            store.mark_changes_seen(unseen)
            return {"ok": True, "changes": rows,
                    "reply": "\n".join(lines)}
        return {"ok": False, "error": f"unknown action: {action!r}",
                "reply": ("I can tell you where to watch something, "
                          "watch or unwatch a title, list the watchlist, "
                          "reconcile, or show changes.")}
    except Exception as exc:  # noqa: BLE001 — the tool never raises
        return {"ok": False, "error": f"watch_offers failed: {exc}",
                "reply": f"Could not do that: {exc}"}


def register(registry: Any) -> None:
    """Attach the ``watch_offers`` tool to a registry."""
    context = registry.context

    @registry.register(
        "watch_offers",
        description=(
            "Say where a film or show can be watched, legally, in a "
            "country — which services stream it, rent it or sell it, "
            "at what price and quality, with the date each offer was "
            "last verified. Ask it as 'where can I watch Dune in "
            "Nigeria?' or 'where can I watch X?'. "
            "action=<where|watch|unwatch|list|reconcile|changes>; "
            "where/reconcile take title=<film or show> "
            "territory=<2-letter country code, default NG> "
            "refresh=<true to re-check the catalogues now>; watch "
            "adds a title to the daily watchlist (new or leaving "
            "offers alert the owner once). Every offer carries its "
            "checked_at and a staleness label (fresh/aging/stale); "
            "an offer with no fresh check says so and names its "
            "last check — never an availability claim without a date."
        ),
        capability=Capability.NET_OUT,
    )
    def watch_offers_tool(action: str = "where", title: str = "",
                          territory: str = DEFAULT_TERRITORY,
                          refresh: bool = False) -> dict[str, Any]:
        return watch_offers(action=action, title=title,
                            territory=territory, refresh=refresh,
                            context=context)
