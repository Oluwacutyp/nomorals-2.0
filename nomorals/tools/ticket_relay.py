"""Ticket relay tool: ``ticket_relay`` — the brain's handle on the
Discord↔GitHub ticket relay, plus the ``__ticket_relay_sync__``
scheduler tick.

One tool, action-dispatched, over :mod:`nomorals.relay`; the organ
owns the tickets, the issue contract and the loop — this module is
only the surface.

``action=open`` files a ticket (``repo=owner/name``,
``title=``, ``body=``, ``reporter=``, ``reporter_id=``,
``origin_chat=discord:<channel id>``, ``ticket_type=``,
``labels=`` comma list); ``action=list`` lists tickets;
``action=status`` shows one ticket whole (``ref=``);
``action=comment`` posts a Discord reply to the issue
(``ref=``, ``text=``); ``action=close`` closes the issue
(``ref=``); ``action=sync`` pulls issue state and comments back
to Discord now (``ref=`` for one ticket, empty for all open ones).

Without a GitHub credential every mutating action answers
honestly as not connected — run ``nm connectors connect --name
github`` first — never a fabricated issue link. Without a live
gateway, sync events queue on the ticket as pending posts for the
next sync that has one.
"""
from __future__ import annotations

from typing import Any

from ..relay import github as _github_mod
from ..relay import surface as _surface_mod
from ..relay.jobs import TICKET_RELAY_TOOL
from ..relay.relay import RelayEngine
from ..relay.store import TicketRelayStore, ticket_relay_path

__all__ = ["register"]


def _store(context: Any) -> TicketRelayStore:
    settings = getattr(context, "settings", None)
    return TicketRelayStore(ticket_relay_path(settings))


def _send_fn(context: Any) -> Any | None:
    """A ``send(chat_key, text)`` callable, or None when offline.

    A send passed in on the context (``context._ticket_send``)
    wins — the test seam. Otherwise the live chat gateway,
    resolved the way the delegation manager resolves it.
    """
    injected = getattr(context, "_ticket_send", None)
    if injected is not None:
        return injected
    try:
        from ..agents.notifier import resolve_gateway
    except Exception:  # noqa: BLE001 - no gateway, no send
        return None
    gateway = resolve_gateway(context)
    if gateway is None:
        return None

    def _send(chat_key: str, text: str) -> Any:
        platform = str(chat_key or "").split(":", 1)[0] or "discord"
        return gateway.send(platform, chat_key, text)

    return _send


def _engine(context: Any) -> RelayEngine | None:
    """The relay engine, or None when GitHub is not connected."""
    client = _github_mod.github_client(context)
    if client is None:
        return None
    return RelayEngine(client, send=_send_fn(context))


def _labels(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [p.strip() for p in str(value or "").split(",") if p.strip()]


def register(registry: Any) -> None:
    """Attach the ``ticket_relay`` tool and the sync tick tool."""
    context = registry.context

    @registry.register(
        "ticket_relay",
        description=(
            "Discord tickets filed as GitHub issues, both ways. "
            "action=open files a ticket (repo=owner/name, title=, "
            "body=, reporter=, reporter_id=, origin_chat=discord:<id>, "
            "ticket_type=, labels= comma list). action=list lists "
            "tickets. action=status shows one ticket whole (ref=). "
            "action=comment posts a Discord reply to the issue "
            "(ref=, text=). action=close closes the issue (ref=). "
            "action=sync pulls issue state and comments back to "
            "Discord now (ref= for one ticket, empty for all open). "
            "Without a GitHub credential every mutating action "
            "answers honestly as not connected, never a fabricated "
            "issue link."
        ),
        capability="owner",
    )
    def ticket_relay_tool(
        action: str = "list",
        ref: str = "",
        repo: str = "",
        title: str = "",
        body: str = "",
        reporter: str = "",
        reporter_id: str = "",
        origin_chat: str = "",
        ticket_type: str = "",
        labels: Any = None,
        text: str = "",
    ) -> dict[str, Any]:
        store = _store(context)
        action = (action or "list").strip().lower()
        if action == "list":
            links = store.list_all()
            return {"ok": True, "tickets": [
                {"ref": t.ref, "repo": t.repo,
                 "issue_number": t.issue_number,
                 "issue_url": t.issue_url, "title": t.title,
                 "state": t.state,
                 "reporter": t.reporter}
                for t in links],
                "board": _surface_mod.render_ticket_board(links)}
        if action == "status":
            link = store.get(ref)
            if link is None:
                return {"ok": False,
                        "error": f"no ticket {ref} on record"}
            return {"ok": True,
                    "ticket": link.to_dict(),
                    "render": _surface_mod.render_ticket(link)}
        engine = _engine(context)
        if engine is None:
            return {"ok": False,
                    "error": _github_mod.github_not_connected_reason()}
        if action == "open":
            out = engine.open_ticket(
                store, repo=repo, title=title, body=body,
                reporter=reporter, reporter_id=reporter_id,
                origin_chat=origin_chat,
                ticket_type=ticket_type or None,
                labels=_labels(labels))
            if not out.get("ok"):
                return {"ok": False, "error": out.get("reason", "")}
            link = store.get(out["ref"])
            render = ""
            if link is not None:
                render = _surface_mod.render_open_result(
                    link, {"url": out.get("issue_url", "")})
            return {**out, "render": render}
        if action == "comment":
            out = engine.post_comment(store, ref, author=reporter,
                                      text=text)
            if not out.get("ok"):
                return {"ok": False, "error": out.get("reason", "")}
            return out
        if action == "close":
            out = engine.close_ticket(store, ref, by=reporter)
            if not out.get("ok"):
                return {"ok": False, "error": out.get("reason", "")}
            return out
        if action == "sync":
            out = (engine.sync_ticket(store, ref)
                   if ref else engine.sync_all(store))
            if not out.get("ok"):
                return {"ok": False, "error": out.get("reason", "")}
            return out
        return {"ok": False, "error": f"unknown ticket_relay action {action!r}"}

    @registry.register(
        TICKET_RELAY_TOOL,
        description="Run one ticket-relay sync tick: poll every open "
                    "ticket's GitHub issue and post new comments and "
                    "state changes back to Discord. Scheduler-internal; "
                    "runs hourly. Never raises — a bad tick is recorded, "
                    "never fatal to the job.",
        capability="owner",
        result_protocol="dict_ok",
    )
    def _ticket_relay_sync_tool() -> dict[str, Any]:
        store = _store(context)
        engine = _engine(context)
        if engine is None:
            return {"ok": False,
                    "error": _github_mod.github_not_connected_reason()}
        return engine.sync_all(store)
