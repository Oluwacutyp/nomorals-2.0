"""Telegram menu coverage derived from the command registry.

The old failure mode was ``test_help_sync`` drift: a hand-maintained menu
listed commands the system no longer had (or missed new ones), so
"reachable capabilities must be in catalog" was aspirational. This module
derives both menus' COVERAGE — public and owner — straight from
:mod:`nomorals.social.chat.control` at call time:

* public menu: exactly :data:`PUBLIC_COMMANDS` (the curated BotFather
  list), grouped by the registry's ``LIST_GROUPS`` order.
* owner menu: every key of :data:`CONTROL_COMMANDS`.

Invariant: ``validate_menu()`` returns ``(True, [])`` iff the owner menu
covers every registered command and the public menu covers every public
command while containing nothing a non-owner may not run
(:func:`caller_may_run`). Any registry change that breaks that fails
loudly instead of drifting silently.

The RENDERED menu lives in control now (presentation Phase 2,
2026-11-11): the live session render and its text form
(``menu_session_render`` / ``menu_home_text``) compose the one
designed menu from the same registry. This module's own text
renderer (``menu_text``) was a second, dead render of the same
data — no production caller — and is removed; what remains is
the coverage structure and the drift invariant the help-sync
tests pin.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "build_telegram_menu",
    "menu_covers",
    "validate_menu",
    "public_menu_kinds",
    "owner_menu_kinds",
]


def public_menu_kinds() -> list[str]:
    """Public kinds, in registry order (LIST_GROUPS, then leftovers)."""
    from .control import CONTROL_COMMANDS, LIST_GROUPS, PUBLIC_COMMANDS

    ordered: list[str] = []
    for _, kinds in LIST_GROUPS:
        for kind in kinds:
            if kind in PUBLIC_COMMANDS and kind not in ordered:
                ordered.append(kind)
    for kind in sorted(PUBLIC_COMMANDS):
        if kind not in ordered:
            ordered.append(kind)
    # Guard against a PUBLIC_COMMANDS entry with no registry row: the
    # menu must still list it (caller_may_run is the grant, not the group).
    for kind in sorted(PUBLIC_COMMANDS):
        if kind not in CONTROL_COMMANDS and kind not in ordered:
            ordered.append(kind)
    return ordered


def owner_menu_kinds() -> list[str]:
    """Every registered command, in registry order (LIST_GROUPS first)."""
    from .control import CONTROL_COMMANDS, LIST_GROUPS

    ordered: list[str] = []
    for _, kinds in LIST_GROUPS:
        for kind in kinds:
            if kind in CONTROL_COMMANDS and kind not in ordered:
                ordered.append(kind)
    for kind in sorted(CONTROL_COMMANDS):
        if kind not in ordered:
            ordered.append(kind)
    return ordered


def build_telegram_menu(is_owner: bool) -> dict[str, list[str]]:
    """``{group: [kinds...]}`` for the caller — registry-derived.

    Public callers get only their grant; owner-only commands are never
    listed for them, and the grant itself (not menu hiding) is what
    denies them — see :func:`caller_may_run` in control.
    """
    from .control import LIST_GROUPS, PUBLIC_COMMANDS

    kinds = set(owner_menu_kinds() if is_owner else public_menu_kinds())
    menu: dict[str, list[str]] = {}
    for group, group_kinds in LIST_GROUPS:
        shown = [k for k in group_kinds if k in kinds]
        if shown:
            menu[group] = shown
    # Anything the registry knows but no group files (e.g. a newly added
    # command not yet grouped) still appears, under "more".
    shown_all = {k for v in menu.values() for k in v}
    rest = sorted(kinds - shown_all)
    if rest:
        menu.setdefault("more", []).extend(rest)
    if not is_owner:
        # PUBLIC_COMMANDS entries missing from every LIST_GROUPS group
        # (e.g. "game") still surface — grouped, not dropped.
        for kind in sorted(PUBLIC_COMMANDS):
            if kind in kinds and kind not in shown_all and kind not in menu.get("more", []):
                menu.setdefault("more", []).append(kind)
    return menu


def menu_covers(kind: str, is_owner: bool) -> bool:
    """True iff *kind* appears in the menu this caller sees."""
    menu = build_telegram_menu(is_owner)
    return any(kind in kinds for kinds in menu.values())


def validate_menu() -> tuple[bool, list[str]]:
    """Check both menus against the registry. Returns ``(ok, problems)``.

    Problems name the exact drift: a registered command missing from the
    owner menu, a public command missing from the public menu, or a
    public-menu entry a non-owner may not run.
    """
    from .control import CONTROL_COMMANDS, PUBLIC_COMMANDS, caller_may_run

    problems: list[str] = []
    owner = build_telegram_menu(True)
    owner_kinds = {k for v in owner.values() for k in v}
    for kind in sorted(CONTROL_COMMANDS):
        if kind not in owner_kinds:
            problems.append(f"owner menu missing registered command /{kind}")
    public = build_telegram_menu(False)
    public_kinds = {k for v in public.values() for k in v}
    for kind in sorted(PUBLIC_COMMANDS):
        if kind not in public_kinds:
            problems.append(f"public menu missing public command /{kind}")
    for kind in sorted(public_kinds):
        if not caller_may_run(kind, is_owner=False):
            problems.append(f"public menu lists /{kind}, which non-owners may not run")
    return (not problems, problems)
