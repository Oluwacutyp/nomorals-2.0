"""Expose integrations — thin wrappers, no rebuilds.

Every function here delegates to an existing module under
:mod:`nomorals.integrations` where one exists. Where the underlying
module only *advertises* a backend (docstring) but has no code path
for it, the descriptor says so honestly instead of pretending the
backend works. Nothing here executes a device command, sends mail,
moves money, or activates an automation without an explicit,
separately-confirmed call.

Verified against the real modules (HEAD 7ba7b40):

* Email — ``nomorals/integrations/email_integration.py``:
  ``_detect_backend`` returns only ``"gmail_api"`` or ``"imap"``
  (``email_integration.py:326``); its docstring also names
  ``"browser"`` but no code path ever returns it.
* Calendar — ``nomorals/integrations/calendar_integration.py``:
  module docstring advertises CalDAV, but ``_detect_backend``
  (``calendar_integration.py:451``) returns only ``"google_api"``
  or ``"none"``; every public method raises unless ``"google_api"``.
* Payments — ``nomorals/integrations/payment_integration.py:468``:
  ``create_virtual_card`` is real (vault-stored).
* Smart home — ``nomorals/integrations/smarthome_integration.py``:
  Home Assistant backend only; ``list_devices`` returns ``[]``
  with no backend.
* Digital twin — ``nomorals/integrations/digital_twin.py:784``:
  ``what_changed`` is the real "what changed since I left?".
* Naija shopping — ``nomorals/integrations/naija_shopping.py:749``:
  ``compute_steal_score`` = savings vs 30-day median (50%) +
  savings vs cross-site best (30%) + trust (20%).
* Routines — ``nomorals/integrations/routines.py``:
  ``build_routine`` → ``validate`` → ``confirm`` → ``to_ha_automation``;
  ``activate`` only runs through an explicit call with a confirmed
  routine (see :func:`activate_automation`).
"""

from __future__ import annotations

import statistics
from typing import Any, Sequence

__all__ = [
    "describe_email_backends",
    "describe_calendar_backends",
    "describe_smart_home_backends",
    "make_email",
    "make_calendar",
    "make_payments",
    "make_smart_home",
    "make_home_twin",
    "make_naija_shopping",
    "send_email",
    "read_inbox",
    "add_calendar_event",
    "list_calendar_events",
    "create_virtual_card",
    "list_virtual_cards",
    "deactivate_virtual_card",
    "list_smart_home_devices",
    "smart_home_device_status",
    "set_smart_home_light",
    "what_changed_since",
    "diff_snapshots",
    "steal_score_from_history",
    "compute_steal_score",
    "parse_automation",
    "confirm_automation",
    "activate_automation",
]


# ── backend descriptors (honest availability) ─────────────────────────────


def describe_email_backends() -> list[dict[str, Any]]:
    """The three email backends, with honest availability.

    ``gmail_api`` and ``imap`` (SMTP send + IMAP read) are the two
    backends ``_detect_backend`` can actually return. ``browser`` is
    named in that method's docstring but no code path returns it, so
    it is reported as advertised-but-unimplemented, not as working.
    """
    return [
        {
            "name": "gmail_api",
            "transport": "Gmail REST API (OAuth)",
            "available": True,
            "evidence": "nomorals/integrations/email_integration.py:326",
            "note": "Returned for @gmail.com accounts with an oauth_token credential.",
        },
        {
            "name": "imap",
            "transport": "SMTP send + IMAP read",
            "available": True,
            "evidence": "nomorals/integrations/email_integration.py:326",
            "note": "Default backend for every non-Gmail-API account.",
        },
        {
            "name": "browser",
            "transport": "webmail browser automation",
            "available": False,
            "evidence": "nomorals/integrations/email_integration.py:326",
            "note": "Named in _detect_backend's docstring; no code path returns it. "
                    "Do not claim webmail email works until a path exists.",
        },
    ]


def describe_calendar_backends() -> list[dict[str, Any]]:
    """Calendar backends, with honest availability.

    The module docstring advertises CalDAV (iCloud, Nextcloud), but
    the code has no CalDAV path: ``_detect_backend`` returns only
    ``"google_api"`` or ``"none"`` and every public method raises
    unless the backend is ``"google_api"``.
    """
    return [
        {
            "name": "google_api",
            "transport": "Google Calendar API (OAuth)",
            "available": True,
            "evidence": "nomorals/integrations/calendar_integration.py:451",
            "note": "Returned for @gmail.com accounts with a google_calendar_oauth credential.",
        },
        {
            "name": "caldav",
            "transport": "CalDAV (iCloud, Nextcloud, …)",
            "available": False,
            "evidence": "nomorals/integrations/calendar_integration.py:1",
            "note": "Advertised in the module docstring only. No CalDAV code "
                    "path exists; _detect_backend never returns it.",
        },
        {
            "name": "browser",
            "transport": "browser automation fallback",
            "available": False,
            "evidence": "nomorals/integrations/calendar_integration.py:1",
            "note": "Advertised in the module docstring only. No browser code path exists.",
        },
    ]


def describe_smart_home_backends() -> list[dict[str, Any]]:
    """Smart-home backends: Home Assistant only."""
    return [
        {
            "name": "homeassistant",
            "transport": "Home Assistant REST + WebSocket",
            "available": True,
            "evidence": "nomorals/integrations/smarthome_integration.py",
            "note": "list_devices returns [] with no backend; all control calls "
                    "require the Home Assistant backend.",
        },
        {
            "name": "mqtt",
            "transport": "MQTT / Zigbee2MQTT bridge",
            "available": True,
            "evidence": "nomorals/integrations/mqtt_client.py",
            "note": "Separate MQTTBridge module; not a SmartHomeIntegration backend. "
                    "Use it directly for MQTT state publish/subscribe.",
        },
    ]


# ── constructors (pass-through, no new behaviour) ──────────────────────────


def make_email(account_manager: Any, session_manager: Any) -> Any:
    """Build :class:`EmailIntegration` (needs account + session managers)."""
    from .email_integration import EmailIntegration

    return EmailIntegration(account_manager, session_manager)


def make_calendar(account_manager: Any, session_manager: Any) -> Any:
    """Build :class:`CalendarIntegration`."""
    from .calendar_integration import CalendarIntegration

    return CalendarIntegration(account_manager, session_manager)


def make_payments(account_manager: Any, session_manager: Any) -> Any:
    """Build :class:`PaymentIntegration` (all spends need approval)."""
    from .payment_integration import PaymentIntegration

    return PaymentIntegration(account_manager, session_manager)


def make_smart_home(account_manager: Any, session_manager: Any) -> Any:
    """Build :class:`SmartHomeIntegration`."""
    from .smarthome_integration import SmartHomeIntegration

    return SmartHomeIntegration(account_manager, session_manager)


def make_home_twin(db_path: str = ":memory:") -> Any:
    """Build :class:`HomeTwin` on ``db_path`` (``:memory:`` = throwaway)."""
    from .digital_twin import HomeTwin

    return HomeTwin(db_path)


def make_naija_shopping(db: Any, browser: Any, proxy_manager: Any = None) -> Any:
    """Build :class:`NaijaShoppingEngine` (needs DB + browser session)."""
    from .naija_shopping import NaijaShoppingEngine

    return NaijaShoppingEngine(db, browser, proxy_manager)


# ── email / calendar wrappers (async, delegate) ────────────────────────────


async def send_email(email: Any, *, to: str | list[str], subject: str,
                     body: str, **kw: Any) -> Any:
    """Delegate to ``EmailIntegration.send``."""
    return await email.send(to, subject, body, **kw)


async def read_inbox(email: Any, account: str, *, limit: int = 20,
                     folder: str = "INBOX", unread_only: bool = False) -> Any:
    """Delegate to ``EmailIntegration.read_inbox``."""
    return await email.read_inbox(account, limit=limit, folder=folder,
                                  unread_only=unread_only)


async def add_calendar_event(calendar: Any, *, title: str, start: str,
                             end: str, account: str = "", **kw: Any) -> Any:
    """Delegate to ``CalendarIntegration.add_event``."""
    return await calendar.add_event(title=title, start=start, end=end,
                                    account=account, **kw)


async def list_calendar_events(calendar: Any, account: str, *,
                              days_ahead: int = 7, **kw: Any) -> Any:
    """Delegate to ``CalendarIntegration.list_events``."""
    return await calendar.list_events(account, days_ahead, **kw)


# ── payments: virtual cards ────────────────────────────────────────────────


async def create_virtual_card(payments: Any, *, merchant: str = "",
                              limit: float = 100.0, currency: str = "USD",
                              single_use: bool = False) -> Any:
    """Delegate to ``PaymentIntegration.create_virtual_card``.

    The card is generated locally and stored in the vault
    (``payment_integration.py:468``). It is a Devon-side virtual card
    record, not a bank-issued card — never present it as a bank card.
    """
    return await payments.create_virtual_card(
        merchant=merchant, limit=limit, currency=currency,
        single_use=single_use)


async def list_virtual_cards(payments: Any) -> Any:
    """Delegate to ``PaymentIntegration.get_virtual_cards``."""
    return await payments.get_virtual_cards()


async def deactivate_virtual_card(payments: Any, card_id: str) -> Any:
    """Delegate to ``PaymentIntegration.deactivate_virtual_card``."""
    return await payments.deactivate_virtual_card(card_id)


# ── smart home wrappers (async, delegate) ──────────────────────────────────


async def list_smart_home_devices(home: Any, room: str = "",
                                  device_type: str = "") -> list[Any]:
    """Delegate to ``SmartHomeIntegration.list_devices`` ([] with no backend)."""
    return await home.list_devices(room, device_type)


async def smart_home_device_status(home: Any, device_id_or_name: str) -> Any:
    """Delegate to ``SmartHomeIntegration.get_device_status``."""
    return await home.get_device_status(device_id_or_name)


async def set_smart_home_light(home: Any, device_id_or_name: str, *,
                              on: bool | None = None,
                              brightness: int | None = None,
                              color: str | None = None,
                              color_temp: int | None = None) -> Any:
    """Delegate to ``SmartHomeIntegration.set_light``."""
    return await home.set_light(device_id_or_name, on=on,
                                brightness=brightness, color=color,
                                color_temp=color_temp)


# ── digital twin: "what changed since I left?" ─────────────────────────────


def what_changed_since(twin: Any, since: float, *, limit: int = 30) -> str:
    """Delegate to ``HomeTwin.what_changed`` — plain-language diff.

    ``since`` is a unix timestamp (e.g. when the owner left home).
    Returns ``"nothing changed since then."`` when there is nothing.
    """
    return twin.what_changed(since, limit=limit)


def diff_snapshots(before: dict[str, Any],
                   after: dict[str, Any]) -> dict[str, Any]:
    """Diff two supplied home snapshots (``{entity_id: state}``).

    Pure function — no twin, no DB, no devices touched. Returns
    ``{"added", "removed", "changed", "unchanged"}`` where ``changed``
    maps entity → ``{"before", "after"}``. This is the snapshot-diff
    behind "what changed since I left?" when the caller supplies both
    snapshots instead of querying the twin.
    """
    before = dict(before or {})
    after = dict(after or {})
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = {
        eid: {"before": before[eid], "after": after[eid]}
        for eid in sorted(set(before) & set(after))
        if before[eid] != after[eid]
    }
    unchanged = sorted(
        eid for eid in (set(before) & set(after)) if before[eid] == after[eid])
    return {"added": added, "removed": removed, "changed": changed,
            "unchanged": unchanged}


# ── Naija shopping: steal scores ──────────────────────────────────────────


def steal_score_from_history(
    price_history: Sequence[float],
    current_price: float,
    *,
    trust: float = 0.5,
    cross_site_best: float | None = None,
) -> dict[str, Any] | None:
    """Steal score vs the 30-day median, from *supplied* price history.

    Mirrors ``NaijaShoppingEngine.compute_steal_score``
    (``naija_shopping.py:749``) exactly: the caller supplies the
    30-day price history (the engine reads it from ``price_snapshots``;
    this function takes it as data so the score is computable without
    a DB or a scrape). Formula::

        savings_vs_median = (median - current) / median * 100
        savings_vs_cross   = (cross_best - current) / cross_best * 100
        score = savings_vs_median * 0.5 + savings_vs_cross * 0.3 + trust * 20

    Returns ``None`` when the history is empty or has no usable price
    (the engine returns ``None`` in the same case) — never a fake
    score. ``cross_site_best`` defaults to ``current_price`` (no
    cross-site saving) when not supplied.
    """
    prices = sorted(p for p in (price_history or []) if p and p > 0)
    if not prices or not current_price or current_price <= 0:
        return None
    median_30d = statistics.median(prices)
    best = cross_site_best if cross_site_best else current_price
    if best <= 0:
        best = current_price
    savings_vs_median = ((median_30d - current_price) / median_30d * 100
                         if median_30d > 0 else 0.0)
    savings_vs_cross = ((best - current_price) / best * 100
                        if best > 0 else 0.0)
    score = (savings_vs_median * 0.5 + savings_vs_cross * 0.3
             + float(trust) * 20)
    return {
        "current_price": current_price,
        "median_30d": median_30d,
        "cross_site_best": best,
        "savings_vs_median": savings_vs_median,
        "savings_vs_cross_site": savings_vs_cross,
        "trust": float(trust),
        "steal_score": score,
    }


async def compute_steal_score(engine: Any, product_url: str) -> Any:
    """Delegate to ``NaijaShoppingEngine.compute_steal_score`` (DB-backed)."""
    return await engine.compute_steal_score(product_url)


# ── NL → automation: parse only, never execute without confirmation ────────


def parse_automation(nl_text: str, *, devices: list[dict[str, Any]] | None = None,
                     aliases: dict[str, str] | None = None,
                     memory: Any = None, llm_parse: Any = None) -> dict[str, Any]:
    """Parse ``"every morning at 7, kitchen lights + coffee"`` to a descriptor.

    Delegates to ``routines.build_routine`` + ``validate`` + ``describe``.
    The result is a *descriptor* — it is never executed. It carries
    ``requires_confirmation=True`` and ``confirmed=False``; activation
    needs :func:`confirm_automation` followed by an explicit
    :func:`activate_automation` call with ``confirmed=True``.
    """
    from . import routines

    draft = routines.build_routine(nl_text, devices=devices, aliases=aliases,
                                   memory=memory, llm_parse=llm_parse)
    errors = routines.validate(draft)
    return {
        "draft_id": draft.id,
        "draft": draft,
        "raw": draft.raw,
        "name": draft.name,
        "starter": draft.starter,
        "conditions": draft.conditions,
        "actions": draft.actions,
        "unresolved": list(draft.unresolved),
        "needs_clarification": draft.needs_clarification,
        "clarification": draft.clarification,
        "errors": [{"code": e.code, "message": e.message} for e in errors],
        "description": routines.describe(draft),
        "requires_confirmation": True,
        "confirmed": False,
        "ha_automation": None,
    }


def confirm_automation(descriptor_or_draft: Any) -> dict[str, Any]:
    """Validate + freeze a parsed draft into a routine (still not activated).

    Delegates to ``routines.confirm`` (raises ``RoutineBuildError`` on
    validation errors) and ``routines.to_ha_automation`` to pre-build the
    Home Assistant trigger/actions/conditions. The returned descriptor
    has ``confirmed=True`` but ``ha_automation_id`` empty — nothing has
    been sent to Home Assistant. Call :func:`activate_automation` with
    ``confirmed=True`` to actually activate.
    """
    from . import routines

    draft = (descriptor_or_draft.get("draft")
             if isinstance(descriptor_or_draft, dict) else descriptor_or_draft)
    routine = routines.confirm(draft)
    trigger, actions, conditions = routines.to_ha_automation(routine)
    return {
        "routine": routine,
        "name": routine.name,
        "starter": routine.starter,
        "conditions": routine.conditions,
        "actions": routine.actions,
        "ha_automation": {"trigger": trigger, "actions": actions,
                          "conditions": conditions},
        "ha_automation_id": routine.ha_automation_id,
        "requires_confirmation": True,
        "confirmed": True,
        "activated": False,
    }


async def activate_automation(confirmed: dict[str, Any], integration: Any) -> Any:
    """Activate a *confirmed* routine via Home Assistant.

    Refuses unless ``confirmed["confirmed"]`` is truthy — an
    unconfirmed descriptor can never reach ``routines.activate``.
    Delegates to ``routines.activate``.
    """
    from . import routines

    if not isinstance(confirmed, dict) or not confirmed.get("confirmed"):
        raise RuntimeError(
            "refusing to activate an unconfirmed automation — call "
            "confirm_automation() first and pass its result with "
            "confirmed=True")
    routine = confirmed["routine"]
    return await routines.activate(routine, integration)
