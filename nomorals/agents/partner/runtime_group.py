"""RuntimeGroupMixin: the group surface in the live event path
(DESIGN §1, P1 — Devon Telegram group & community system).

Group chats are claimed by the surface BEFORE conversation:
service events (join/leave) run the welcome/ledger organs and are
consumed; ordinary messages are observed into the member ledger
(identity snapshots + change detection) and run through the
moderation pipeline, whose actions this mixin executes through
the platform adapter's admin surface where it has one:

- ``delete`` → ``admin_delete``; ``ban`` → ``admin_ban``;
  ``mute`` → ``admin_restrict`` (timed, 1h by default posture);
- ``warn`` → one short public notice (moderation is an event
  Devon owns — the quiet-by-default law names it);
- an action the platform path cannot execute is recorded in the
  group's held audit log as unexecuted, with the reason — never
  faked, never narrated into the chat.

A message deleted or banned by moderation is consumed: Devon
does not also converse with it.  Everything else flows on to the
normal pipeline (claimed-once, games, conversation).
"""

from __future__ import annotations

import logging
import time
from typing import Any

_log = logging.getLogger(__name__)


class _GroupWagerPurse:
    """Wager-purse backend binding one group room to the group
    economy estate (see games/purse.py for the backend contract).
    Player keys are ``platform:sender_id`` — the sender id IS the
    member id, exactly as the coin sink resolves it.  Every movement
    journals in the group ledger via the surface's wager methods,
    tagged by the bound purse with game + room."""

    name = "group purse"

    def __init__(self, surface, group_key: str) -> None:  # type: ignore[no-untyped-def]
        self._surface = surface
        self._group_key = group_key

    @staticmethod
    def _member(player) -> str:  # type: ignore[no-untyped-def]
        key = str(getattr(player, "key", "") or "")
        return key.split(":", 1)[1] if ":" in key else ""

    def balance(self, player) -> int:  # type: ignore[no-untyped-def]
        member = self._member(player)
        if not member:
            return 0
        return self._surface.game_wager_balance(self._group_key,
                                                member)

    def debit(self, player, amount: int, reason: str,  # type: ignore[no-untyped-def]
              ref: str) -> bool:
        member = self._member(player)
        if not member:
            return False
        return self._surface.game_wager_debit(
            self._group_key, member, int(amount), reason, ref=ref)

    def credit(self, player, amount: int, reason: str,  # type: ignore[no-untyped-def]
               ref: str) -> int:
        member = self._member(player)
        if not member:
            return 0
        return self._surface.game_wager_credit(
            self._group_key, member, int(amount), reason, ref=ref)


class _GamesCoinWallet:
    """The games-wallet side of a §25 conversion, bound to one
    member.  Coins move through the PlayerStore (the same
    atomic spend / credit the wager purse uses), and every
    movement is also written to the games economy ledger, so
    that side's health picture sees conversions as what they
    are: coins sunk out of play going forward, coins issued
    back into play coming back."""

    def __init__(self, engine, player) -> None:  # type: ignore[no-untyped-def]
        self._store = engine.store
        self._db = engine.db
        self._player = player

    def balance(self) -> int:
        return int(self._store.get_for(self._player).coins)

    def spend(self, amount: int, reason: str) -> bool:
        amount = int(amount)
        if amount <= 0:
            return False
        if self._store.spend_coins(
                self._player, amount, reason=reason) is None:
            return False
        from ...games.economy import record_ledger

        record_ledger(self._db, self._player.key, "sink", amount,
                      "convert")
        return True

    def credit(self, amount: int, reason: str) -> int:
        amount = int(amount)
        if amount <= 0:
            return 0
        self._store.add_coins(self._player, amount, reason=reason)
        from ...games.economy import record_ledger

        record_ledger(self._db, self._player.key, "source", amount,
                      "convert")
        return amount


class RuntimeGroupMixin:
    """Group-surface turn for PartnerRuntime._process_inner."""

    def _group_surface(self):  # type: ignore[no-untyped-def]
        surface = getattr(self, "_group_surface_obj", None)
        if surface is None:
            from ...social.chat.group_surface import default_surface

            ws_dir = getattr(self.context.settings, "workspace_dir", ".")
            surface = default_surface(ws_dir)
            self._group_surface_obj = surface
        # §19: ONE owner record per process — the surface's
        # pipeline reads the same registry object the owner's
        # DM surface edits (seeded once, below).
        try:
            registry = self._owner_registry()
            if surface.owner_registry is not registry:
                surface.owner_registry = registry
        except Exception:  # noqa: BLE001 - a rig without a
            pass               # registry keeps the surface's own
        return surface

    def _owner_registry(self):  # type: ignore[no-untyped-def]
        """The §19 owner record for this process, formed once
        from the bindings that already exist (config sender
        ids, the §15 account pool, the core seal's handles) —
        seeding fills empty slots only and never overwrites an
        edit the owner made through /owner."""
        registry = getattr(self, "_owner_registry_obj", None)
        if registry is None:
            from ...social.chat.owner_system import \
                default_registry

            context = getattr(self, "context", None)
            settings = getattr(context, "settings", None)
            ws_dir = getattr(settings, "workspace_dir", ".")
            registry = default_registry(ws_dir)
            try:
                from ...core.owner import OWNER_IDENTITIES

                accounts = []
                supervisor = getattr(self, "_pairing_supervisor",
                                     None)
                if supervisor is not None:
                    accounts = supervisor().registry.all()
                registry.seed(
                    telegram_ids=getattr(
                        self, "_owner_sender_ids", ()) or (),
                    accounts=accounts,
                    handles=OWNER_IDENTITIES)
            except Exception:  # noqa: BLE001 - seeding is
                _log.debug("owner registry seed failed",
                           exc_info=True)
            self._owner_registry_obj = registry
        return registry

    def _webshot_book(self):  # type: ignore[no-untyped-def]
        book = getattr(self, "_webshot_book_obj", None)
        if book is None:
            from ...social.chat.webshot import default_book

            context = getattr(self, "context", None)
            settings = getattr(context, "settings", None)
            ws_dir = getattr(settings, "workspace_dir", ".")
            book = default_book(ws_dir)
            self._webshot_book_obj = book
        return book

    def _secaudit_book(self):  # type: ignore[no-untyped-def]
        book = getattr(self, "_secaudit_book_obj", None)
        if book is None:
            from ...security.audit_report import default_book

            context = getattr(self, "context", None)
            settings = getattr(context, "settings", None)
            ws_dir = getattr(settings, "workspace_dir", ".")
            book = default_book(ws_dir)
            self._secaudit_book_obj = book
        return book

    # ── security audit (owner, 2026-11-03) ───────────────────
    _SECAUDIT_DRIVE_THROTTLE_S = 15.0

    def _control_audit_scan(self, tail, message):  # type: ignore[no-untyped-def]
        """/scan <url>: queue an exposure scan.  DM-first by
        law — findings are disclosure material, they never post
        to a group; in a group the command points at DM and
        queues nothing."""
        if self._group_key_or_none(message) is not None:
            return ("🛡️ audits run in DM — findings are "
                    "disclosure material and never post to a "
                    "group. Run /scan <url> in my DM.")
        url = str(tail or "").strip().split()[0] \
            if str(tail or "").strip() else ""
        if not url:
            return "usage: /scan <url>"
        book = self._secaudit_book()
        job = book.enqueue(
            kind="web", target=url,
            chat_platform=str(message.chat.platform or ""),
            chat_key=str(message.chat.key or ""),
            requester=str(message.sender_id or message.sender
                          or ""))
        own = book.is_own_asset(url)
        return (f"🛡️ exposure scan queued ({job['id']}) for {url}"
                + (" — your own asset: full-depth capture"
                   if own else "")
                + ". The graded report lands here when it's done.")

    def _control_audit(self, tail, message):  # type: ignore[no-untyped-def]
        """/audit <repo url|path> — queue a repo audit (secrets
        incl. history, dependencies, insecure patterns) with a
        disclosure draft when a public contact exists.
        /audit own add|remove|list <asset> — the own-asset
        registry the owner controls: his declared assets are
        captured unmasked, at full sample depth."""
        if self._group_key_or_none(message) is not None:
            return ("🛡️ audits run in DM — findings are "
                    "disclosure material and never post to a "
                    "group. Run /audit in my DM.")
        text = str(tail or "").strip()
        words = text.split()
        book = self._secaudit_book()
        if words and words[0].lower() == "own":
            return self._control_audit_own(words[1:], book)
        if not words:
            return ("usage: /audit <repo url|path> · "
                    "/audit own add|remove|list <asset>")
        target = words[0]
        job = book.enqueue(
            kind="repo", target=target,
            chat_platform=str(message.chat.platform or ""),
            chat_key=str(message.chat.key or ""),
            requester=str(message.sender_id or message.sender
                          or ""))
        own = book.is_own_asset(target)
        return (f"🛡️ repo audit queued ({job['id']}) for {target}"
                + (" — your own asset: full-depth capture"
                   if own else "")
                + ". The graded report lands here when it's done.")

    def _control_audit_own(self, words, book):  # type: ignore[no-untyped-def]
        sub = words[0].lower() if words else "list"
        if sub in ("list", ""):
            assets = book.own_assets()
            if not assets:
                return ("own-asset registry is empty — "
                        "/audit own add <host|url|path> declares "
                        "one of your systems (full-depth, "
                        "unmasked capture for those targets).")
            return ("🛡️ your declared assets:\n"
                    + "\n".join(f"· {a}" for a in assets))
        if sub in ("add", "remove"):
            if len(words) < 2:
                return f"usage: /audit own {sub} <host|url|path>"
            value = " ".join(words[1:])
            if sub == "add":
                added = book.add_own_asset(value)
                return (f"🛡️ {value} declared yours — audits of "
                        "it capture at full depth, unmasked."
                        if added else
                        f"{value} is already in the registry.")
            removed = book.remove_own_asset(value)
            return (f"🛡️ {value} removed from the registry."
                    if removed else
                    f"{value} was not in the registry.")
        return ("usage: /audit own add|remove|list <asset>")

    def _secaudit_report_card(self, outcome, *, platform=""):  # type: ignore[no-untyped-def]
        """The chat card for a finished audit: severity counts
        + the top findings, via the typography layer (§27)."""
        from ...social.chat.typography import Card, render_card

        summary = outcome.get("summary") or {}
        counts = summary.get("counts") or {}
        card = Card("announcement", title="Security audit",
                    icon="🛡️", tone="info")
        kind = "web app" if outcome.get("kind") == "web" \
            else "repository"
        card.line(f"{kind}: {outcome.get('target')}")
        if outcome.get("own_asset"):
            card.line("your own asset — full-depth capture")
        card.stats([
            ("critical", counts.get("critical", 0)),
            ("high", counts.get("high", 0)),
            ("medium", counts.get("medium", 0)),
            ("low", counts.get("low", 0)),
        ], title="findings by severity")
        top = (outcome.get("findings") or [])[:5]
        if top:
            card.block_lines(
                [f"[{str(f.get('severity') or '').upper()}] "
                 f"{f.get('title')}" for f in top],
                title="top findings")
        else:
            card.line("no findings — every check came back clean")
        for note in (outcome.get("notes") or [])[:3]:
            card.line(f"note: {note}")
        if outcome.get("draft_path"):
            card.line("a disclosure draft is ready — sending "
                      "it is your call")
        card.note(f"audit {outcome.get('audit_id')} · full "
                  "report in the attached file")
        rendered = render_card(card, platform=platform or
                               "telegram-bot", settings=None)
        return rendered.text

    def _drive_security_audits(self):  # type: ignore[no-untyped-def]
        """The audit beat: one queued job per cycle, delivered
        to the DM it was requested from (card + report file +
        disclosure draft when one was written).  No daemon —
        the page-watch discipline."""
        import time as _time
        from types import SimpleNamespace

        now = _time.time()
        last = getattr(self, "_secaudit_last", 0.0)
        if now - last < self._SECAUDIT_DRIVE_THROTTLE_S:
            return
        self._secaudit_last = now
        book = self._secaudit_book()
        job = book.next_job()
        if job is None:
            return
        from ...security.audit_service import AuditService, run_job

        # Jobs exist only because an operator queued one: the
        # kinds are owner-only (absent from PUBLIC_COMMANDS)
        # and the handlers refuse groups — so the operator's
        # private-target privilege (his own LAN) applies here
        # exactly as it does for /webshot in his DM.
        service = AuditService(book, allow_private=True)
        try:
            outcome = run_job(service, job)
        except Exception as exc:  # noqa: BLE001 - a failed job
            book.finish_job(job["id"], {   # is reported, not lost
                "error": f"{type(exc).__name__}: {exc}"})
            self.gateway.send(
                job.get("chat_platform") or "",
                SimpleNamespace(
                    platform=job.get("chat_platform") or "",
                    key=job.get("chat_key") or ""),
                f"🛡️ audit of {job.get('target')} failed: "
                f"{type(exc).__name__}: {exc}")
            return
        book.finish_job(job["id"], {
            "audit_id": outcome.get("audit_id"),
            "summary": outcome.get("summary"),
            "report_path": outcome.get("report_path"),
            "draft_path": outcome.get("draft_path")})
        chat = SimpleNamespace(
            platform=job.get("chat_platform") or "",
            key=job.get("chat_key") or "")
        card = self._secaudit_report_card(
            outcome, platform=job.get("chat_platform") or "")
        self.gateway.send(chat.platform, chat, card)
        caption = (f"🛡️ security audit — {outcome.get('target')}")
        self.gateway.send_file(
            chat.platform, str(chat.key),
            str(outcome.get("report_path") or ""), caption=caption)
        if outcome.get("draft_path"):
            self.gateway.send_file(
                chat.platform, str(chat.key),
                str(outcome["draft_path"]),
                caption="✉️ disclosure draft — sending is your call")

    def _send(self, chat, text: str, *, buttons=None,  # type: ignore[no-untyped-def]
              parse_mode: str = ""):
        """The mixin's own outbound channel (welcomes, goodbyes,
        notices).  This method was CALLED by every welcome path
        since P1 but defined NOWHERE on PartnerRuntime — the live
        bug found 2026-11-03: every join's welcome died on
        AttributeError, caught upstream and logged, never sent.
        Defined here, on the mixin that calls it, riding the same
        gateway contract (buttons + parse_mode) as every surface.
        Returns the gateway's SendResult (callers that need the
        posted message's id — the notice layer — read it there;
        everyone else ignores it, as before)."""
        return self.gateway.send(chat.platform, chat, str(text),
                                 buttons=buttons,
                                 parse_mode=parse_mode)

    def _send_best_effort(self, chat, text: str, **kw):  # type: ignore[no-untyped-def]
        """A side-send from a control handler (a notice for an
        action whose RECORD is already durable on the surface):
        the send is the announcement, never the state — a
        transport that can't take it here degrades to the debug
        log instead of failing the command that caused it."""
        try:
            return self._send(chat, text, **kw)
        except Exception:  # noqa: BLE001 - announcement, not state
            _log.debug("group side-send failed", exc_info=True)
            return None

    # ---- the system-output gate (DESIGN §33) --------------------
    def _system_send_ok(self, surface, key, message,  # type: ignore[no-untyped-def]
                        what: str) -> bool:
        """§33, evaluated at send time: may Devon's BOT SYSTEMS
        post this into this room, through THIS session?  The
        surface's one helper decides (owner designation, the
        bot's own standing, the sending session's platform).
        A refusal is a journaled non-event in the group's held
        audit — never a post, never a fake success.  The
        as-him conversational path never comes through here."""
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if surface.system_output_allowed(key,
                                         via_platform=platform):
            return True
        surface._audit(key, "system_output_suppressed",
                       what=what, via=platform)
        surface.store.save()
        return False

    def _rhythm_hold(self, surface, key, what, text) -> bool:  # type: ignore[no-untyped-def]
        """Silent hours (P3): when the group's rhythm says the
        room is quiet, a non-urgent system post is HELD for the
        morning digest instead of sent — returns True when held
        (the caller then skips the send).  Moderation output
        never comes through here: enforcement is not scheduled.
        As-him conversation never comes through here either
        (§33 hat-1)."""
        if not text:
            return False
        if surface.rhythm_silent(key):
            surface.rhythm_hold(key, what, str(text))
            return True
        return False

    _BOT_CLAIM_TTL_S = 600

    def _probe_bot_claim(self, surface, key, message) -> None:  # type: ignore[no-untyped-def]
        """Keep the room's bot standing (§33a) honest: when a
        BOT-platform session serves this group, verify the
        bot's own member status through the adapter and stamp
        the result on the surface — a demotion observed here
        lapses the claim for every session at once.  Probed at
        most once per room per TTL (the fact changes rarely);
        an adapter that cannot say leaves the stamp alone."""
        platform = str(getattr(message.chat, "platform", "")
                       or "")
        if platform not in ("telegram-bot", "telegram-personal"):
            return
        probes = getattr(self, "_bot_claim_probes", None)
        if probes is None:
            probes = self._bot_claim_probes = {}
        now = time.time()
        if now - float(probes.get(key) or 0) \
                < self._BOT_CLAIM_TTL_S:
            return
        adapter = self._group_adapter(platform)
        bot_id = getattr(adapter, "_bot_id", None) \
            if adapter is not None else None
        fn = getattr(adapter, "admin_member", None) \
            if adapter is not None else None
        if bot_id is None or fn is None:
            return
        probes[key] = now
        try:
            info = fn(message.chat, str(bot_id))
        except Exception:  # noqa: BLE001 - unproven, unstamped
            return
        if not isinstance(info, dict):
            return
        status = info.get("status") or info.get("role") or ""
        if not status and isinstance(info.get("result"), dict):
            status = info["result"].get("status") or ""
        if not status:
            return
        surface.note_bot_admin(
            key, str(status).lower() in (
                "creator", "administrator", "admin", "owner"))

    def _send_action_notice(self, surface, key, message,  # type: ignore[no-untyped-def]
                            action) -> None:
        """Post one §31 notice for an executed action.  The text
        and the Undo button are authored from the SAME record
        the surface stored (never re-derived at send time), and
        the posted message's id is written back to the record so
        an undo can edit the notice in place where the platform
        supports it."""
        from ...social.chat.group_notices import (
            notice_keyboard, notice_text)

        if not self._system_send_ok(surface, key, message,
                                    "notice"):
            return
        nid = str(action.get("notice_id") or "")
        record = surface.get_notice(key, nid) if nid else None
        text = str(action.get("warning") or "") or str(
            (record or {}).get("text") or "")
        if not text:
            return
        platform = str(message.chat.platform or "")
        if record is not None:
            body = notice_text({**record, "text": text}, platform)
            buttons = notice_keyboard(
                record, settings=surface.store.settings(key))
        else:
            body, buttons = text, None
        result = self._send(
            message.chat, body, buttons=buttons,
            parse_mode=self._rendered_mode(message, body))
        if record is not None:
            mid = getattr(result, "message_id", "") or ""
            if mid:
                surface.set_notice_message(key, nid, str(mid))
        if str(action.get("kind") or "") == "ban":
            # A ban lands with theatre (§21); a warning never
            # does. The expression mixin owns the firing — the
            # getattr guard keeps bare group rigs (tests) whole.
            fire = getattr(self, "_fire_expression", None)
            if fire is not None:
                fire(message, "ban")

    def _rendered_mode(self, message, text: str) -> str:  # type: ignore[no-untyped-def]
        """Parse mode for a surface-rendered text (welcome /
        goodbye): the typography organ renders Telegram-family
        surfaces as HTML, so the send must SAY so — otherwise
        the gateway's F1 presentation would convert the already-
        rendered HTML a second time.  Other platforms' renders
        carry their own native marks and send bare."""
        platform = str(getattr(message.chat, "platform", "") or "")
        if platform.startswith("telegram") and "<" in str(text):
            return "HTML"
        return ""

    def _card(self, message, card, buttons=None, *, xmd_frame: bool = False, session: str = "", session_edit: bool = False):  # type: ignore[no-untyped-def]
        """Render a typography Card (§27) for this message's
        platform under the group's presentation settings, and
        bundle the buttons the handler authored from the same
        state.  This is the ONE way group control replies are
        composed now: handlers name what renders, the group's
        theme decides how it looks.

        ``xmd_frame`` (the group help's owner-ordered WhatsApp
        frame, 2026-11-05): on WhatsApp the same Card renders in
        the XMD box grammar instead of the flat dialect render —
        same content, same buttons, same stamp resolution; every
        other platform ignores the flag.

        ``session`` / ``session_edit`` declare the card as one
        render of a live session (presentation architecture —
        see social/chat/live_session): the DM console's screens
        ride it, so navigating the console edits ONE message
        instead of stacking a new card per tap."""
        from ...social.chat.tgbot_buttons import actionable
        from ...social.chat.typography import render_card, render_card_framed

        surface = self._group_surface()
        key = message.chat.key
        platform = str(message.chat.platform or "")
        if xmd_frame and platform.strip().lower() in ("whatsapp", "wa"):
            rendered = render_card_framed(
                card, settings=surface.store.settings(key))
        else:
            rendered = render_card(
                card, platform=platform,
                settings=surface.store.settings(key))
        if buttons is not None:
            return actionable(rendered.text, buttons,
                              session=session,
                              session_edit=session_edit)
        return rendered.text

    def _welcome_buttons(self, surface: Any, key: str) -> list:
        """Buttons authored from the group's LIVE state: rules are
        always reachable; the sparks row appears only when the
        economy actually runs here. Labels come from the one
        button vocabulary (§21), under the group's own
        presentation settings."""
        from ...social.chat.expression import buttons_from_spec

        settings = surface.store.settings(key)
        spec: list = [[("rules", "Rules", "rules")]]
        try:
            if surface.economy_on(key):
                spec.append([("balance", "My sparks", "balance"),
                             ("quests", "Quests", "quests")])
        except Exception:  # noqa: BLE001 - buttons never block a welcome
            pass
        return buttons_from_spec(spec, settings=settings)

    # ---- join acceptance execution (DESIGN §28) -------------------
    # The surface judges; these helpers perform.  Every execution
    # speaks the adapter's own grammar and reports honestly —
    # a missing surface is an audited non-execution, never a
    # claimed success.

    def _acceptance_restrict(self, adapter, chat, user_id: str, *,
                             on: bool) -> bool:  # type: ignore[no-untyped-def]
        """Restrict (read-only) or release one member.  The two
        Telegram adapters spell restriction differently (MTProto
        minutes=0 = indefinite; the Bot API's until defaults to
        forever) — try each one's own grammar."""
        if adapter is None:
            return False
        if on:
            fn = getattr(adapter, "admin_restrict", None)
            if fn is None:
                return False
            try:
                out = fn(chat, user_id, minutes=0)
            except TypeError:
                try:
                    out = fn(chat, user_id)
                except Exception:  # noqa: BLE001 - reported below
                    return False
            except Exception:  # noqa: BLE001 - reported below
                return False
        else:
            fn = getattr(adapter, "admin_unrestrict", None)
            if fn is None:
                return False
            try:
                out = fn(chat, user_id)
            except Exception:  # noqa: BLE001 - reported below
                return False
        return not (isinstance(out, dict)
                    and out.get("ok") is False)

    def _acceptance_photo(self, adapter, chat, user_id: str):  # type: ignore[no-untyped-def]
        """Fresh profile-photo fact for the §28 photo criterion:
        True/False, or None when the platform can't say (an
        adapter without the surface, or a failed lookup) —
        None is UNKNOWN to the evaluator, never a failure."""
        fn = getattr(adapter, "admin_member_photo", None) \
            if adapter else None
        if fn is None:
            return None
        try:
            out = fn(chat, user_id)
        except Exception:  # noqa: BLE001 - unknown, not failed
            return None
        if isinstance(out, dict) and out.get("ok"):
            return bool(out.get("has_photo"))
        return None

    def _acceptance_join_request(self, surface, key, adapter,  # type: ignore[no-untyped-def]
                                 chat, user_id: str, *,
                                 approve: bool) -> bool:
        """Perform one join-request decision through the
        adapter; audit the outcome either way."""
        fn = getattr(adapter, "admin_join_request", None) \
            if adapter else None
        action = ("join_request_approve" if approve
                  else "join_request_decline")
        if fn is None:
            surface._audit(
                key, "moderation_unexecuted", action=action,
                rule="acceptance", user_id=user_id,
                reason="adapter has no admin_join_request")
            surface.store.save()
            return False
        try:
            out = fn(chat, user_id, approve)
            if isinstance(out, dict) and out.get("ok") is False:
                raise RuntimeError(str(out.get("error")
                                       or "platform refused"))
        except Exception as exc:  # noqa: BLE001 - audited
            surface._audit(
                key, "moderation_unexecuted", action=action,
                rule="acceptance", user_id=user_id,
                reason=f"{type(exc).__name__}: {exc}")
            surface.store.save()
            return False
        surface._audit(key, "moderation_executed", action=action,
                       rule="acceptance", user_id=user_id)
        surface.store.save()
        return True

    def _acceptance_facts(self, member: dict) -> dict:  # type: ignore[no-untyped-def]
        """Identity facts for one arrival, in the evaluator's
        shape, from a converter-supplied member/requester dict."""
        name = str(member.get("name") or "")
        first = str(member.get("first_name") or name)
        last = str(member.get("last_name") or "")
        if not last and name and first and name.startswith(first):
            rest = name[len(first):].strip()
            if rest:
                last = rest
        return {"user_id": str(member.get("id") or ""),
                "username": str(member.get("username") or ""),
                "name": name or first, "first_name": first,
                "last_name": last,
                "is_bot": bool(member.get("is_bot"))}

    def _send_acceptance_note(self, surface, key, message,  # type: ignore[no-untyped-def]
                              label: str, reasons: list, *,
                              grace_minutes: int | None = None,
                              pending_kick: bool = False) -> None:
        """The restricted member's one message (§28): what to
        fix + the “check me again” button, through the
        typography organ like every presented surface."""
        from ...social.chat.group_acceptance import fix_it_text
        from ...social.chat.typography import Card

        if not self._system_send_ok(surface, key, message,
                                    "acceptance_note"):
            return
        card = Card("moderation", title=f"{label} — one moment",
                    icon="🚪")
        body = fix_it_text(
            list(reasons or []),
            grace_minutes=grace_minutes if pending_kick else None)
        card.lines(body.splitlines())
        from ...social.chat.expression import buttons_from_spec

        text = surface._card_render(key, card)
        self._send(message.chat, text,
                   buttons=buttons_from_spec(
                       [[("check", "Check me again", "recheck")]],
                       settings=surface.store.settings(key)),
                   parse_mode=self._rendered_mode(message, text))

    def _send_acceptance_queue_note(self, surface, key, message,  # type: ignore[no-untyped-def]
                                    record: dict,
                                    reasons: list) -> None:
        """The admin queue notice for one held request (§28):
        who, what their evaluation said, and Approve / Decline
        buttons authored from the request itself.  The request
        id rides the buttons only — never the visible text
        (internal ids don't print)."""
        from ...social.chat.typography import Card

        if not self._system_send_ok(surface, key, message,
                                    "acceptance_queue"):
            return
        label = (record.get("name")
                 or (f"@{record['username']}"
                     if record.get("username") else "")
                 or record.get("user_id") or "someone")
        card = Card("moderation", title="Join request waiting",
                    icon="🚪")
        lines = [f"{label} wants in."]
        if reasons:
            lines.append("their check: " + "; ".join(reasons))
        elif record.get("unknown"):
            lines.append("one check couldn't be verified — "
                         "a human decides.")
        else:
            lines.append("they meet every rule — auto-approve "
                         "is off, so this one's yours.")
        lines.append("requests stay open until an admin decides "
                     "or the invite link is revoked.")
        from ...social.chat.expression import buttons_from_spec

        card.lines(lines)
        text = surface._card_render(key, card)
        rid = str(record.get("id") or "")
        self._send(message.chat, text, buttons=buttons_from_spec([
            [("approve", "Approve", f"joinq approve {rid}"),
             ("decline", "Decline", f"joinq decline {rid}")],
            [("queue", "The queue", "joinq")]],
            settings=surface.store.settings(key)),
            parse_mode=self._rendered_mode(message, text))

    def _group_adapter(self, platform: str):  # type: ignore[no-untyped-def]
        gateway = getattr(self, "gateway", None)
        adapters = getattr(gateway, "adapters", {}) or {}
        return adapters.get(platform)

    def _reputation_join_hook(self, surface, key, message, *,  # type: ignore[no-untyped-def]
                              user_id, username="", name="",
                              held=False):
        """The §32 join hook: fires AFTER §28 admitted the person
        (admission was decided on the group's own criteria — a
        record never changes it).  A live record earns the group
        its configured warning through the §31 notice layer:
        public posts the factual ⚠️ notice, admins writes the
        /glog trail only, restricted admits them restricted until
        an admin taps Clear, off stays silent.  Public/restricted
        posts are system output: the §33 gate and the group's
        quiet mode apply, and every outcome — landed or
        suppressed, with its reason — is journaled."""
        from ...social.chat.group_notices import (
            reputation_warning_lines,
        )
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card

        check = surface.reputation_join_check(
            key, user_id, username=username, name=name)
        if not check.get("found"):
            return
        record = check["record"]
        mode = check["mode"]
        rid = record["id"]
        if check.get("dismissed"):
            surface.reputation_note_suppressed(
                key, rid, user_id=user_id, reason="dismissed",
                mode=mode)
            return
        if mode == "off":
            surface.reputation_note_suppressed(
                key, rid, user_id=user_id, reason="mode_off",
                mode=mode)
            return
        if mode == "admins":
            # The /glog trail IS the delivery in admins mode.
            surface.reputation_note_warned(
                key, rid, user_id=user_id, mode="admins")
            return
        if not self._system_send_ok(
                surface, key, message, "reputation warning"):
            surface.reputation_note_suppressed(
                key, rid, user_id=user_id, reason="gate",
                mode=mode)
            return
        settings = surface.store.settings(key)
        if not (settings.get("defense") or {}).get(
                "autonomy", True):
            surface.reputation_note_suppressed(
                key, rid, user_id=user_id,
                reason="autonomy_off", mode=mode)
            return
        member = surface.member(key, user_id) or {}
        # The §31 mention convention (name, else @handle, else
        # the id) — the same shape every notice carries.
        mname = str(member.get("name") or name or "")
        muser = str(member.get("username") or username or "")
        mention = mname or (f"@{muser}" if muser else str(user_id))
        source_label = str(record.get("source_group") or "")
        date_text = time.strftime(
            "%Y-%m-%d", time.gmtime(
                float(record.get("created_ts") or 0)))
        restricted = mode == "restricted" and not held
        lines = reputation_warning_lines(
            record, mention=mention, user_id=str(user_id),
            source_label=source_label, date_text=date_text,
            more=int(check.get("more") or 0),
            restricted=restricted, held=held)
        card = Card("moderation", title="Known record",
                    icon="⚠️")
        card.block_lines(lines)
        spec: list = []
        if restricted:
            surface.reputation_apply_restriction(
                key, user_id, record)
            adapter = self._group_adapter(message.chat.platform)
            self._execute_group_action(
                surface, key, adapter, message,
                {"kind": "acceptance_restrict",
                 "user_id": str(user_id), "rule": "reputation"})
            spec.append(("check", "Clear",
                         f"repclear {user_id}"))
        spec.append(("dismiss", "Dismiss",
                     f"repdismiss {rid}"))
        settings_now = surface.store.settings(key)
        buttons = buttons_from_spec([spec],
                                    settings=settings_now)
        if not buttons and str(message.chat.platform or "") \
                not in ("telegram-bot", "telegram"):
            # No-button platforms get the text form of the same
            # affordances (§31 platform honesty).
            card.note(f"admins: /repclear {user_id} clears a "
                      f"hold · /repdismiss {rid} dismisses this "
                      f"warning here")
        self._send(message.chat, surface._card_render(
            key, card), buttons=buttons,
            parse_mode="Markdown")
        surface.reputation_note_warned(
            key, rid, user_id=user_id, mode=mode)

    def _group_surface_turn(self, message) -> bool:  # type: ignore[no-untyped-def]
        """Run one group message through the surface.  Returns
        True when the message is fully consumed (service event,
        or moderated away)."""
        surface = self._group_surface()
        key = message.chat.key
        self._probe_bot_claim(surface, key, message)
        meta = message.meta or {}
        service = str(meta.get("service") or "")

        if service == "reaction":
            # Reactions IN (§22): a member's reaction is room
            # signal — journaled to the held audit (the surface
            # engagement counts read from), never answered, never
            # narrated. Not system output: the §33 gate does not
            # apply to READING the room.
            surface._audit(
                key, "reaction",
                user_id=str(message.sender_id or ""),
                emojis=",".join(meta.get("reaction_new") or []),
                removed=",".join(meta.get("reaction_old") or []),
                message_id=str(meta.get("reacted_message_id")
                               or ""))
            surface.store.save()
            return True
        if service in ("chat_boost", "chat_boost_removed"):
            # Boost intel (§16): who is boosting the room,
            # journaled into the surface's boost ledger —
            # counted, attributed exactly as Telegram's source
            # named it, never answered and never narrated
            # (reading the room, not system output).
            boost = meta.get("boost") or {}
            if isinstance(boost, dict) and boost.get("boost_id"):
                if service == "chat_boost":
                    surface.note_boost(
                        key, boost_id=str(boost["boost_id"]),
                        user_id=str(boost.get("user_id") or ""),
                        source_kind=str(boost.get("source_kind")
                                        or "unknown"),
                        add_ts=boost.get("add_date"),
                        expiration_ts=boost.get("expiration_date"))
                else:
                    surface.note_boost_removed(
                        key, boost_id=str(boost["boost_id"]),
                        user_id=str(boost.get("user_id") or ""))
            return True
        if service == "forum_topic_created":
            # A forum topic's birth (§16): recorded in the
            # surface's topic ledger.  An auto-named topic
            # (the bot surface's is_name_implicit flag) becomes
            # a rename candidate — the duty itself rides the
            # topic's first real message, below.
            topic = meta.get("topic") or {}
            thread = str(getattr(message.chat, "thread_id", "")
                         or meta.get("thread_id") or "")
            if thread and isinstance(topic, dict):
                surface.topic_note_created(
                    key, thread_id=thread,
                    title=str(topic.get("title") or ""),
                    implicit=bool(topic.get("implicit")))
            return True
        if service == "join":
            adapter = self._group_adapter(message.chat.platform)
            acc_cfg = (surface.store.settings(key)
                       .get("acceptance") or {})
            criteria = acc_cfg.get("criteria") or {}
            # Acceptance is INERT when no criterion is on and
            # the mode is open: the standing P1 path, byte for
            # byte.  The moment a group configures a rule, every
            # arrival is judged (§28).
            inert = (str(acc_cfg.get("mode") or "open") == "open"
                     and not any(criteria.values()))
            welcomed = []
            for member in meta.get("new_members") or []:
                if not isinstance(member, dict):
                    continue
                user_id = str(member.get("id") or "")
                if not user_id:
                    continue
                if inert:
                    text = surface.record_join(
                        key, user_id,
                        username=str(member.get("username") or ""),
                        name=str(member.get("name") or ""),
                        group_name=str(message.chat.title or ""))
                    if text:
                        welcomed.append(text)
                    self._reputation_join_hook(
                        surface, key, message, user_id=user_id,
                        username=str(member.get("username") or ""),
                        name=str(member.get("name") or ""))
                    continue
                facts = self._acceptance_facts(member)
                has_photo = member.get("has_photo")
                if criteria.get("photo") and has_photo is None:
                    has_photo = self._acceptance_photo(
                        adapter, message.chat, user_id)
                out = surface.acceptance_on_join(
                    key, facts, has_photo=has_photo)
                for action in out.get("actions") or []:
                    self._execute_group_action(
                        surface, key, adapter, message, action)
                if out.get("welcome"):
                    text = surface.welcome_text_for(
                        key, name=facts["name"],
                        username=facts["username"],
                        group_name=str(message.chat.title or ""))
                    if text:
                        welcomed.append(text)
                elif out.get("verdict") == "restricted":
                    label = (facts["name"]
                             or (f"@{facts['username']}"
                                 if facts["username"] else "")
                             or user_id)
                    self._send_acceptance_note(
                        surface, key, message, label,
                        out.get("reasons") or [],
                        grace_minutes=out.get("grace_minutes"),
                        pending_kick=bool(out.get("pending_kick")))
                # §32: admitted (welcomed or restricted-admitted)
                # — the reputation hook hears about the record;
                # declined/kicked arrivals never reach it.
                if out.get("welcome") \
                        or out.get("verdict") == "restricted":
                    held = bool((surface.member(key, user_id)
                                 or {}).get("acceptance"))
                    self._reputation_join_hook(
                        surface, key, message, user_id=user_id,
                        username=facts["username"],
                        name=facts["name"], held=held)
            if welcomed and self._system_send_ok(
                    surface, key, message, "welcome"):
                if surface.rhythm_silent(key):
                    # Welcomes are system output: during silent
                    # hours they wait for the morning digest.
                    for text in welcomed:
                        surface.rhythm_hold(key, "welcome",
                                            text)
                else:
                    for text in welcomed:
                        self._send(
                            message.chat, text,
                            buttons=self._welcome_buttons(
                                surface, key),
                            parse_mode=self._rendered_mode(
                                message, text))
                    # The welcome moment (§21): a GIF/sticker
                    # rides the welcome where the group's
                    # expression settings allow — same gate,
                    # same room.
                    fire = getattr(self, "_fire_expression",
                                   None)
                    if fire is not None:
                        fire(message, "welcome")
            self._clean_service_message(surface, message)
            return True
        if service == "join_request":
            # Telegram's approval-mode invite flow (DESIGN §28):
            # the candidate is not in the group — the surface
            # judges, this branch performs the decision through
            # the adapter, and held requests surface ONE queue
            # notice with the decision on its buttons.
            adapter = self._group_adapter(message.chat.platform)
            requester = meta.get("requester") or {}
            if isinstance(requester, dict) and requester.get("id"):
                facts = self._acceptance_facts(requester)
                # §16 P5: the guard-bot query id and the
                # applicant DM window ride the queue record —
                # the "queue" verdict is answered through the
                # query, and /joinq ask writes through the
                # window while it stands.
                if meta.get("join_query_id"):
                    facts["query_id"] = str(meta["join_query_id"])
                if meta.get("user_chat_id"):
                    facts["user_chat_id"] = str(
                        meta["user_chat_id"])
                user_id = facts["user_id"]
                acc_cfg = (surface.store.settings(key)
                           .get("acceptance") or {})
                criteria = acc_cfg.get("criteria") or {}
                has_photo = requester.get("has_photo")
                if criteria.get("photo") and has_photo is None:
                    has_photo = self._acceptance_photo(
                        adapter, message.chat, user_id)
                out = surface.acceptance_on_request(
                    key, facts, has_photo=has_photo)
                verdict = out.get("verdict")
                if verdict in ("approve", "approve_restricted"):
                    done = self._acceptance_join_request(
                        surface, key, adapter, message.chat,
                        user_id, approve=True)
                    if done:
                        admitted = \
                            surface.acceptance_request_admitted(
                                key, str(out.get("request_id")
                                         or ""))
                        if admitted.get("restricted"):
                            self._execute_group_action(
                                surface, key, adapter, message,
                                {"kind": "acceptance_restrict",
                                 "user_id": user_id,
                                 "rule": "acceptance"})
                            label = (facts["name"]
                                     or (f"@{facts['username']}"
                                         if facts["username"]
                                         else "") or user_id)
                            self._send_acceptance_note(
                                surface, key, message, label,
                                admitted.get("reasons") or [],
                                grace_minutes=admitted.get(
                                    "grace_minutes"))
                        elif admitted.get("welcome"):
                            text = surface.welcome_text_for(
                                key, name=facts["name"],
                                username=facts["username"],
                                group_name=str(
                                    message.chat.title or ""))
                            if text and self._system_send_ok(
                                    surface, key, message,
                                    "welcome"):
                                if not self._rhythm_hold(
                                        surface, key, "welcome",
                                        text):
                                    self._send(
                                        message.chat, text,
                                        buttons=(
                                            self._welcome_buttons(
                                                surface, key)),
                                        parse_mode=(
                                            self._rendered_mode(
                                                message, text)))
                                    fire = getattr(
                                        self, "_fire_expression",
                                        None)
                                    if fire is not None:
                                        fire(message, "welcome")
                        # §32: the request was admitted — the
                        # reputation hook hears about any record.
                        if admitted.get("welcome") \
                                or admitted.get("restricted"):
                            self._reputation_join_hook(
                                surface, key, message,
                                user_id=user_id,
                                username=facts["username"],
                                name=facts["name"],
                                held=bool(admitted.get(
                                    "restricted")))
                elif verdict == "decline":
                    self._acceptance_join_request(
                        surface, key, adapter, message.chat,
                        user_id, approve=False)
                else:  # queued — a human decides
                    record = out.get("record") or {}
                    if record:
                        # §16 queue verdict: when Telegram
                        # handed the bot a join QUERY, holding
                        # the request is itself an answer —
                        # "queue" defers the decision to the
                        # chat's other admins (the classic
                        # approve/decline calls remain for the
                        # human who later decides from /joinq).
                        # Best-effort by design — the query
                        # expires in ~10 seconds — and the queue
                        # record stands either way.
                        _qid = str(record.get("query_id") or "")
                        _answer = getattr(
                            adapter, "admin_answer_join_query",
                            None)
                        if _qid and callable(_answer):
                            try:
                                _res = _answer(_qid, "queue")
                            except Exception:  # noqa: BLE001
                                _res = None
                            surface._audit(
                                key, "join_query_deferred",
                                request=str(record.get("id")
                                            or ""),
                                ok=bool(isinstance(_res, dict)
                                        and _res.get("ok")))
                            surface.store.save()
                        self._send_acceptance_queue_note(
                            surface, key, message, record,
                            out.get("reasons") or [])
            return True
        if service == "leave":
            left = meta.get("left_member") or {}
            if isinstance(left, dict) and left.get("id"):
                text = surface.record_leave(
                    key, str(left.get("id") or ""),
                    group_name=str(message.chat.title or ""))
                if text and self._system_send_ok(
                        surface, key, message, "goodbye"):
                    if not self._rhythm_hold(
                            surface, key, "goodbye", text):
                        self._send(
                            message.chat, text,
                            parse_mode=self._rendered_mode(
                                message, text))
            self._clean_service_message(surface, message)
            return True
        if service == "delete":
            # Evidentiary antidelete (defense brief §2.1): fed
            # only by surfaces that can see deletions (MTProto /
            # Baileys / Discord) — the Bot API never sends these.
            ids = meta.get("deleted_ids") or []
            if not ids and meta.get("deleted_id"):
                ids = [meta["deleted_id"]]
            for mid in ids:
                out = surface.handle_delete(
                    key, message_id=str(mid),
                    by_user_id=str(meta.get("by_user_id") or ""))
                for callout in out.get("callouts") or []:
                    if callout.get("to") == "group" \
                            and self._system_send_ok(
                                surface, key, message,
                                "antidelete_callout"):
                        self._send(message.chat,
                                   str(callout.get("text") or ""))
            return True

        user_id = str(message.sender_id or message.sender or "")
        if not user_id:
            return False
        if meta.get("edited"):
            # Evidentiary antiedit: record both versions, then
            # let the edited text flow through the normal path.
            surface.handle_edit(
                key, message_id=str(message.message_id or ""),
                new_text=str(message.text or ""))
        # Solicited game input is play, not spam: the games seam
        # judges from real room state (live room + seated sender
        # + a move the game recognises) whether this message is
        # an input to the game running in this room, and the
        # surface forwards the verdict to the Moderator's
        # counting stages.  Without the games mixin (partial
        # rigs) or on any probe failure the verdict is False —
        # moderation behaves exactly as it always has.
        game_input = False
        probe = getattr(self, "_is_game_input", None)
        player_of = getattr(self, "_game_player", None)
        if probe is not None and player_of is not None:
            try:
                game_input = bool(probe(
                    key, str(message.text or ""),
                    player=player_of(message),
                    kind=str(message.chat.kind or "")))
            except Exception:  # noqa: BLE001 - no verdict, no exemption
                game_input = False
        result = surface.handle_message(
            key, user_id,
            username=str(message.sender_username or ""),
            name=str(message.sender or ""),
            text=str(message.text or ""),
            message_id=str(message.message_id or ""),
            reply_to=str(getattr(message, "reply_to", "")
                         or ""),
            envelope=meta.get("envelope"),
            sender_is_admin=self._group_admin_ok(message),
            sender_is_operator=self._operator_or_false(message),
            game_input=game_input)
        notices = result.get("notices") or []
        if notices and self._system_send_ok(
                surface, key, message, "notice"):
            from ...social.chat.expression import (
                classify_notice_moment)

            for notice in notices:
                self._send(message.chat, str(notice))
                # Celebration notices carry a moment (§21): a
                # finished quest / a season podium gets her
                # media voice; every other notice stays text.
                moment = classify_notice_moment(str(notice))
                if moment:
                    fire = getattr(self, "_fire_expression", None)
                    if fire is not None:
                        fire(message, moment)
        # §34 closes ride the same cycle: a timed math round
        # that expired in this room posts its answer + working
        # here, on the cycle's pass — not on the member's next
        # /math. (RuntimeGamesMixin owns the drain; §33/§31 are
        # evaluated inside, at send time.)
        post_closes = getattr(self, "_post_math_round_closes",
                              None)
        if post_closes is not None:
            post_closes(surface, key, message)
        # §35 drop announcements ride the same cycle: the
        # surface queued them (with their claim buttons) when
        # the conflict cycle opened them; this drains the
        # queue under the same §33/§31 rules.
        self._post_conflict_feed(surface, key, message)
        self._execute_pending_pin(surface, message)
        # §16/P4a: group turns also drive the channel posting
        # cycle — due queue posts in claimed channels ride the
        # same beat as every other system (no daemon); the
        # surface checks each channel's claim before posting.
        self._drive_all_channels(surface)
        # Member cleanup upkeep rides the same beat: in a
        # claimed Telegram group where she holds admin power,
        # deleted accounts are removed automatically (throttled
        # per group inside the drive, journal-only — nothing is
        # posted to the room).  Failure-isolated inside.
        self._cleanup_auto_drive(surface, key, message)
        # §28's WhatsApp limb rides the same beat: Telegram
        # join requests arrive as events, WhatsApp holds them
        # server-side, so a claimed request-mode group polls
        # its door (throttled per group inside the drive) and
        # feeds new requesters into the same queue.
        self._wa_join_request_drive(surface, key, message)
        # §16's topic duty rides the same beat: a pending
        # auto-named forum topic gets a proper name derived
        # from its traffic (claimed rooms, duty switched on).
        self._p5_topic_duty(surface, key, message)
        actions = result.get("actions") or []
        # §21 auto-collect rides the same cycle: a sticker
        # that landed in a claimed room whose admin enabled
        # it joins her library, passively.  A message the
        # pipeline deleted is never collected from — the
        # room removed it; she doesn't keep its media.
        # (RuntimeExpressionMixin owns the beat; scope,
        # toggle and capture discipline live there.)
        collect = getattr(self, "_auto_collect_media", None)
        if collect is not None and not any(
                str(a.get("kind") or "") == "delete"
                for a in actions):
            collect(surface, key, message)
        if not actions:
            return False
        adapter = self._group_adapter(message.chat.platform)
        consumed = False
        for action in actions:
            kind = str(action.get("kind") or "")
            executed = self._execute_group_action(
                surface, key, adapter, message, action)
            warning = str(action.get("warning") or "")
            if executed and (warning or action.get("notice_id")):
                # The organ's one public warning IS the record
                # (no meta-narration): name + rule + consequence
                # — posted through the §31 notice layer, with
                # its Undo authored from the same record.
                self._send_action_notice(surface, key, message,
                                         action)
            elif kind == "warn" and executed \
                    and self._system_send_ok(
                        surface, key, message, "warning"):
                who = (message.sender_username
                       and f"@{message.sender_username}") \
                    or message.sender or "there"
                self._send(
                    message.chat,
                    f"⚠️ {who} — {action.get('reason') or 'warning'}")
            if kind == "ban" and executed:
                # §32: a defense sensor's ban PROPOSES a flag
                # into this group's queue — an admin confirms or
                # discards it; the sensor itself brands nobody.
                self._sensor_flag_proposal(surface, key, action)
            if kind in ("delete", "ban", "kick") and executed:
                consumed = True
        return consumed

    def _sensor_flag_proposal(self, surface, key, action):  # type: ignore[no-untyped-def]
        """Map an executed sensor ban to a §32 flag proposal
        (spam-ring for the spam-family sensors, raid for the
        crash sensor).  Acceptance/manual actions never propose —
        only defense rules do — and a person already carrying a
        live same-category record from THIS group is not
        re-proposed."""
        # Sensor bans land through the escalation ladder, whose
        # action names the ladder as its rule and carries the
        # ORIGINATING sensor as _strike_rule — read that first.
        rule = str(action.get("_strike_rule")
                   or action.get("rule") or "")
        category = {"anticrash": "raid",
                    "antichannel": "spam-ring",
                    "antitag": "spam-ring",
                    "flood": "spam-ring",
                    "repeat": "spam-ring"}.get(rule)
        if not category:
            return
        user_id = str(action.get("user_id") or "")
        if not user_id:
            return
        member = surface.member(key, user_id) or {}
        person = surface.reputation_person(
            key, user_id,
            username=str(member.get("username") or ""),
            name=str(member.get("name") or ""))
        for record in surface.reputation.live_for(person):
            if record.get("source_group") == key \
                    and record.get("category") == category:
                return
        surface.stage_flag(
            key, user_id=user_id, person=person,
            category=category,
            note=f"auto-proposed: the {rule} sensor banned them "
                 f"here — an admin must confirm before this "
                 f"brands anyone",
            by=f"sensor:{rule}", by_label=f"the {rule} sensor",
            direct=False, source="sensor")

    def _execute_pending_pin(self, surface, message) -> None:  # type: ignore[no-untyped-def]
        """A bought Pinned Message grant acts on the buyer's next
        message: pin it (best-effort, audited)."""
        try:
            user_id = str(message.sender_id or message.sender or "")
            if not user_id or not message.message_id:
                return
            if not surface.take_pending_pin(message.chat.key, user_id):
                return
            adapter = self._group_adapter(message.chat.platform)
            fn = getattr(adapter, "admin_pin", None) if adapter \
                else None
            if fn is None:
                # No pin surface on this adapter: the pin did
                # not happen, so the bought grant is NOT spent —
                # hand it back for the buyer's next message.
                surface.grant_pin(message.chat.key, user_id)
                return
            try:
                out = fn(message.chat, str(message.message_id))
                if isinstance(out, dict) \
                        and out.get("ok") is False:
                    raise RuntimeError(str(out.get("error")
                                           or "platform refused"))
            except Exception as exc:  # noqa: BLE001 - audited below
                # A refused (or failed) pin is not a used grant:
                # restore it and audit the refusal, never a use.
                surface.grant_pin(message.chat.key, user_id)
                surface._audit(
                    message.chat.key, "pin_grant_refused",
                    user_id=user_id,
                    message_id=str(message.message_id),
                    reason=f"{type(exc).__name__}: {exc}")
                surface.store.save()
                return
            surface._audit(message.chat.key, "pin_grant_used",
                           user_id=user_id,
                           message_id=str(message.message_id))
            surface.store.save()
        except Exception:  # noqa: BLE001 - a grant never breaks chat
            _log.debug("group pin grant failed", exc_info=True)

    def _group_coin_sink(self, chat_key, player, coins, game) -> None:  # type: ignore[no-untyped-def]
        """GameEngine bridge (P2): a human's in-group game coins
        land in the group purse.  Player keys are
        ``platform:sender_id`` — the sender id IS the member id."""
        try:
            key = str(getattr(player, "key", "") or "")
            if ":" not in key:
                return
            member = key.split(":", 1)[1]
            self._group_surface().game_award(
                chat_key, member, int(coins), str(game))
        except Exception:  # noqa: BLE001 - bridge is best-effort
            _log.debug("group coin sink failed", exc_info=True)

    def _group_wager_purse(self, room):  # type: ignore[no-untyped-def]
        """GameEngine wager-purse source: a GROUP room's table
        stakes the group economy estate — the same purse the coin
        sink credits and members see on /balance.  Any other room
        (or a group whose economy is off / unclaimed) returns None
        and the engine falls back to the games wallet."""
        try:
            if getattr(room, "kind", "") != "group":
                return None
            surface = self._group_surface()
            if not surface.game_wager_available(room.chat_key):
                return None
            return _GroupWagerPurse(surface, room.chat_key)
        except Exception:  # noqa: BLE001 - binding never breaks a table
            _log.debug("group wager purse bind failed", exc_info=True)
            return None

    def _clean_service_message(self, surface, message) -> None:  # type: ignore[no-untyped-def]
        """Service-message hygiene (P1): the raw "X joined the
        group" service message is deleted after the surface has
        handled it — the welcome is the visible artifact, the
        platform's own notice is clutter.  Setting-gated,
        best-effort, audited either way: a delete the platform
        refused (ok:False dict) is audited as a refusal, never
        as a cleaning that happened."""
        try:
            settings = surface.store.settings(message.chat.key)
            if not settings.get("clean_service"):
                return
            if not message.message_id:
                return
            adapter = self._group_adapter(message.chat.platform)
            fn = getattr(adapter, "admin_delete", None) \
                if adapter else None
            if fn is None:
                return
            out = fn(message.chat, str(message.message_id))
            if isinstance(out, dict) and out.get("ok") is False:
                surface._audit(
                    message.chat.key, "service_clean_refused",
                    message_id=str(message.message_id),
                    reason=str(out.get("error")
                               or "platform refused"))
                surface.store.save()
                return
            surface._audit(message.chat.key, "service_cleaned",
                           message_id=str(message.message_id))
            surface.store.save()
        except Exception:  # noqa: BLE001 - hygiene never breaks joins
            _log.debug("group service cleanup failed", exc_info=True)

    # ---- admin gate ------------------------------------------------
    def _operator_or_false(self, message) -> bool:  # type: ignore[no-untyped-def]
        """Operator status, quietly: surfaces without operator
        power (and bare test rigs) read as non-operator."""
        try:
            return bool(self._is_operator(message))
        except Exception:  # noqa: BLE001 - no operator surface
            return False

    def _group_admin_ok(self, message) -> bool:  # type: ignore[no-untyped-def]
        """Operator (on surfaces where operator power exists) OR
        the platform's own word that this sender admins the group.
        On the gated public bot nobody is an operator — surface law
        — so platform admin status is the authority there, exactly
        how group-helper bots work."""
        try:
            if self._is_operator(message):
                return True
        except Exception:  # noqa: BLE001 - fall to platform check
            pass
        adapter = self._group_adapter(message.chat.platform)
        if adapter is None:
            return False
        fn = getattr(adapter, "admin_member", None) or getattr(
            adapter, "admin_member_mtproto", None)
        if fn is None:
            return False
        try:
            info = fn(message.chat, str(message.sender_id or ""))
        except Exception:  # noqa: BLE001 - no proof, no admin
            return False
        if not isinstance(info, dict):
            return False
        status = info.get("status") or info.get("role") or ""
        if not status and isinstance(info.get("result"), dict):
            status = info["result"].get("status") or ""
        return str(status).lower() in (
            "creator", "administrator", "admin", "owner")

    def _member_status(self, message):  # type: ignore[no-untyped-def]
        """The platform's word on this sender's standing in the
        group: (status string, raw info dict) — ("", {}) when the
        surface cannot say."""
        adapter = self._group_adapter(message.chat.platform)
        if adapter is None:
            return "", {}
        fn = getattr(adapter, "admin_member", None) or getattr(
            adapter, "admin_member_mtproto", None)
        if fn is None:
            return "", {}
        try:
            info = fn(message.chat, str(message.sender_id or ""))
        except Exception:  # noqa: BLE001 - no proof, no standing
            return "", {}
        if not isinstance(info, dict):
            return "", {}
        status = info.get("status") or info.get("role") or ""
        if not status and isinstance(info.get("result"), dict):
            status = info["result"].get("status") or ""
        return str(status).lower(), info

    def _group_owner_ok(self, message) -> bool:  # type: ignore[no-untyped-def]
        """The OWNER gate (defense brief §2.3): who may flip the
        autonomy master switch.  The operator always may.  On
        WhatsApp any group admin may (the platform hands group
        control to admins collectively).  On Telegram only the
        group's creator/owner — admin is a delegated role there,
        and the kill-switch outranks delegation.  Discord: the
        server owner."""
        if self._operator_or_false(message):
            return True
        platform = str(message.chat.platform or "")
        if platform.startswith("whatsapp"):
            return self._group_admin_ok(message)
        status, info = self._member_status(message)
        if platform.startswith("telegram"):
            return status in ("creator", "owner")
        if platform.startswith("discord"):
            return status in ("owner", "creator") \
                or info.get("is_owner") is True
        return False

    def _defense_surface_kind(self, message) -> str:  # type: ignore[no-untyped-def]
        """Which adapter family serves this chat — the key the
        platform-honesty matrix reads (Bot API vs MTProto is a
        capability cliff for delete events)."""
        platform = str(message.chat.platform or "")
        if platform.startswith("whatsapp"):
            return "baileys"
        if platform.startswith("discord"):
            return "user"
        if platform == "telegram":
            return "mtproto"
        adapter = self._group_adapter(platform)
        cls = type(adapter).__name__.lower() if adapter else ""
        return "mtproto" if "mtproto" in cls or "user" in cls \
            else "botapi"

    def _defense_update(self, surface, key, sensor, **fields):  # type: ignore[no-untyped-def]
        """Mutate one defense settings block through the store's
        own settings machinery (defaults-merged, persisted)."""
        import copy as _copy

        current = surface.store.settings(key)
        defense = _copy.deepcopy(current.get("defense") or {})
        if sensor == "_root":
            defense.update(fields)
        else:
            block = dict(defense.get(sensor) or {})
            block.update(fields)
            defense[sensor] = block
        surface.update_settings(key, defense=defense)
        return defense

    def _presentation_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the presentation settings block (DESIGN §27:
        theme / per-class presets / stamp) through the store's
        own settings machinery (defaults-merged, persisted)."""
        import copy as _copy

        current = surface.store.settings(key)
        pres = _copy.deepcopy(current.get("presentation") or {})
        pres.update(fields)
        surface.update_settings(key, presentation=pres)
        return pres

    def _expression_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the expression settings block (DESIGN §21)
        through the store's own settings machinery — the one
        write path behind both /gsettings and the console."""
        import copy as _copy

        current = surface.store.settings(key)
        expr = _copy.deepcopy(current.get("expression") or {})
        expr.update(fields)
        surface.update_settings(key, expression=expr)
        return expr

    def _apply_expression_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The expression grammar, shared by /gsettings (words
        after the `expression` head) and the console's
        expression section (bare words). Returns the reply line,
        or None when the words aren't expression grammar.
        Forms: `on|off` · `gifs off|low|normal` ·
        `welcome on|off` · `ban on|off` · `collect on|off`."""
        w = [str(x).lower() for x in words or []]
        if w == ["on"] or w == ["off"]:
            on = w[0] == "on"
            self._expression_update(surface, key, enabled=on)
            return (f"expression {'on' if on else 'off'} — "
                    + ("moments get their media voice here."
                       if on else
                       "moments ship text-only here."))
        if len(w) == 2 and w[0] == "gifs" \
                and w[1] in ("off", "low", "normal"):
            self._expression_update(surface, key, gifs=w[1])
            return {
                "off": "gifs off — every moment ships text-only.",
                "low": "gifs low — welcomes and celebrations "
                       "only; bans land without theatre.",
                "normal": "gifs normal — welcomes, celebrations "
                          "and ban theatre.",
            }[w[1]]
        if len(w) == 2 and w[0] == "welcome" \
                and w[1] in ("on", "off"):
            on = w[1] == "on"
            self._expression_update(surface, key,
                                    welcome_media=on)
            return f"welcome media {'on' if on else 'off'}."
        if len(w) == 2 and w[0] == "ban" \
                and w[1] in ("on", "off"):
            on = w[1] == "on"
            self._expression_update(surface, key, ban_theatre=on)
            return (f"ban theatre {'on' if on else 'off'} — "
                    "warnings stay dry either way.")
        if len(w) == 2 and w[0] == "collect" \
                and w[1] in ("on", "off"):
            on = w[1] == "on"
            self._expression_update(surface, key, auto_collect=on)
            return (f"auto-collect {'on' if on else 'off'} — "
                    + ("stickers landing here join my library "
                       "as they pass." if on else
                       "I only take stickers I'm handed here "
                       "(/steal)."))
        return None

    def _reputation_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the reputation settings block (DESIGN §32)
        through the store's own settings machinery — the one
        write path behind both /gsettings and the console."""
        import copy as _copy

        current = surface.store.settings(key)
        rep = _copy.deepcopy(current.get("reputation") or {})
        rep.update(fields)
        surface.update_settings(key, reputation=rep)
        return rep

    def _apply_reputation_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The reputation grammar, shared by /gsettings (words
        after the `reputation` head) and the console's reputation
        section.  Returns the reply line, or None when the words
        aren't reputation grammar.  Forms: `public|admins|
        restricted|off` (optionally after `mode`)."""
        from ...social.chat.group_reputation import WARNING_MODES

        w = [str(x).lower() for x in words or []]
        if w and w[0] == "mode":
            w = w[1:]
        if len(w) == 1 and w[0] in WARNING_MODES:
            self._reputation_update(surface, key, warning=w[0])
            return {
                "public": "reputation warnings: public — a "
                          "flagged joiner gets a factual ⚠️ "
                          "notice in the group.",
                "admins": "reputation warnings: admins only — "
                          "flagged joiners land in the console "
                          "and /glog, nothing posts in-group.",
                "restricted": "reputation warnings: restricted — "
                              "a flagged joiner is admitted "
                              "restricted until an admin taps "
                              "Clear on the notice.",
                "off": "reputation warnings: off — records still "
                       "exist in the book, this group just never "
                       "hears about them.",
            }[w[0]]
        return None

    def _conflict_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the conflict settings block (§23) through
        the store's own settings machinery — the one write
        path behind both /gsettings and the console."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("conflict") or {})
        block.update(fields)
        surface.update_settings(key, conflict=block)
        return block

    def _apply_conflict_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The conflict grammar, shared by /gsettings (words
        after the `conflict` head) and the console's conflict
        section (bare words).  Returns the reply line, or None
        when the words aren't conflict grammar.  Forms:
        `off|crime|warfare` · `drops N` · `police on|off` ·
        `escrow N`."""
        w = [str(x).lower() for x in words or []]
        if len(w) == 1 and w[0] in ("off", "crime", "warfare"):
            self._conflict_update(surface, key, mode=w[0])
            return {
                "off": "conflict: off — the street sleeps; "
                       "crime, gangs, wars and system drops "
                       "don't run here.",
                "crime": "conflict: crime — geared crime, "
                         "heists, gangs, the police and "
                         "system drops run here.",
                "warfare": "conflict: warfare — everything in "
                           "crime, plus gang wars, territory "
                           "and governance capture.",
            }[w[0]]
        if len(w) == 2 and w[0] == "drops" and w[1].isdigit():
            n = max(0, min(20, int(w[1])))
            self._conflict_update(surface, key, drops_per_day=n)
            return (f"system drops: {n} per day budget"
                    + (" — drops are off." if n == 0 else
                       " — contents roll fresh every time "
                       "(sparks, gems, gear)."))
        if len(w) == 2 and w[0] == "police" \
                and w[1] in ("on", "off"):
            on = w[1] == "on"
            self._conflict_update(surface, key, police=on)
            return (f"police {'on' if on else 'off'}"
                    + (" — the precinct watches the bully "
                       "index." if on else
                       " — nobody hunts the bullies here."))
        if len(w) == 2 and w[0] == "escrow" and w[1].isdigit():
            n = max(0, int(w[1]))
            self._conflict_update(surface, key, war_escrow=n)
            return (f"war escrow: {n} sparks a side, from the "
                    "gang treasuries.")
        return None

    # ---- rhythm: settings grammar (P3) ---------------------------
    def _rhythm_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the rhythm settings block through the store's
        own settings machinery — the one write path behind both
        /gsettings and the console's rhythm section."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("rhythm") or {})
        block.update(fields)
        surface.update_settings(key, rhythm=block)
        return block

    def _apply_rhythm_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The rhythm grammar, shared by /gsettings (words after
        the `rhythm` head) and the console's rhythm section.
        Returns the reply line, or None when the words aren't
        rhythm grammar.  Forms: `on|off` · `silent HH:MM HH:MM` ·
        `tz <offset>` · `nightshield on|off` · `briefing on|off` ·
        `briefing hour N` · `reminders on|off` ·
        `reminder add HH:MM <text>` · `reminder rules HH:MM` ·
        `reminder remove N` · `reminder list` · `quorum N`."""
        from ...social.chat.group_rhythm import (
            format_offset, parse_hhmm, parse_offset)

        w = [str(x) for x in words or []]
        low = [x.lower() for x in w]
        if not w:
            return None
        if len(w) == 1 and low[0] in ("on", "off"):
            on = low[0] == "on"
            cfg = self._rhythm_update(
                surface, key, silent_enabled=on)
            if on:
                return (f"silent hours on — "
                        f"{cfg.get('silent_start')}–"
                        f"{cfg.get('silent_end')} "
                        f"({format_offset(int(cfg.get('tz_offset_min') or 0))}); "
                        "system posts wait for the morning "
                        "digest. Moderation never waits.")
            return ("silent hours off — system posts land as "
                    "they happen again.")
        if len(w) == 3 and low[0] == "silent" \
                and parse_hhmm(w[1]) is not None \
                and parse_hhmm(w[2]) is not None:
            cfg = self._rhythm_update(
                surface, key, silent_start=w[1], silent_end=w[2],
                silent_enabled=True)
            return (f"silent hours: {cfg['silent_start']}–"
                    f"{cfg['silent_end']} "
                    f"({format_offset(int(cfg.get('tz_offset_min') or 0))}), "
                    "on.")
        if len(w) == 2 and low[0] == "tz":
            off = parse_offset(w[1])
            if off is None:
                return ("that offset doesn't parse — try +1, "
                        "-5 or +05:30.")
            off = max(-12 * 60, min(14 * 60, off))
            self._rhythm_update(surface, key, tz_offset_min=off)
            return (f"group time: {format_offset(off)} — silent "
                    "hours, briefings and reminders read this "
                    "clock.")
        if len(w) == 2 and low[0] == "nightshield" \
                and low[1] in ("on", "off"):
            on = low[1] == "on"
            self._rhythm_update(surface, key, nightshield=on)
            return ("nightshield on — during the quiet window "
                    "joins take the request posture and drop/"
                    "war events pause."
                    if on else
                    "nightshield off — the night door works "
                    "like the day door.")
        if len(w) == 2 and low[0] == "briefing" \
                and low[1] in ("on", "off"):
            on = low[1] == "on"
            cfg = self._rhythm_update(
                surface, key, briefing_enabled=on)
            return (f"group briefing on — yesterday's shape "
                    f"posts at {int(cfg.get('briefing_hour') or 8)}"
                    ":00 group time."
                    if on else
                    "group briefing off.")
        if len(w) == 3 and low[0] == "briefing" \
                and low[1] == "hour" and w[2].isdigit():
            hour = max(0, min(23, int(w[2])))
            self._rhythm_update(surface, key, briefing_hour=hour)
            return f"briefing hour: {hour}:00 group time."
        if len(w) == 2 and low[0] == "reminders" \
                and low[1] in ("on", "off"):
            on = low[1] == "on"
            self._rhythm_update(surface, key,
                                reminders_enabled=on)
            return f"reminders {'on' if on else 'off'}."
        if low[0] == "reminder" and len(w) >= 2:
            cfg = surface.rhythm_cfg(key)
            reminders = list(cfg.get("reminders") or [])
            if low[1] == "list":
                if not reminders:
                    return "no reminders set."
                return "reminders:\n" + "\n".join(
                    f"{i}. {r.get('time')} — "
                    + ("the group rules card"
                       if r.get("kind") == "rules"
                       else str(r.get("text") or ""))
                    for i, r in enumerate(reminders, 1))
            if low[1] == "add" and len(w) >= 4 \
                    and parse_hhmm(w[2]) is not None:
                seq = max([int(str(r.get("id") or "r0")[1:])
                           for r in reminders
                           if str(r.get("id") or "").startswith("r")
                           and str(r.get("id"))[1:].isdigit()]
                          or [0]) + 1
                reminders.append({
                    "id": f"r{seq}", "text": " ".join(w[3:]),
                    "time": w[2], "days": [], "kind": "custom"})
                self._rhythm_update(surface, key,
                                    reminders=reminders)
                return (f"reminder set: daily at {w[2]} — "
                        f"{' '.join(w[3:])}")
            if low[1] == "rules" and len(w) == 3 \
                    and parse_hhmm(w[2]) is not None:
                seq = max([int(str(r.get("id") or "r0")[1:])
                           for r in reminders
                           if str(r.get("id") or "").startswith("r")
                           and str(r.get("id"))[1:].isdigit()]
                          or [0]) + 1
                reminders.append({
                    "id": f"r{seq}", "text": "", "time": w[2],
                    "days": [], "kind": "rules"})
                self._rhythm_update(surface, key,
                                    reminders=reminders)
                return (f"rules reminder set: the rules card "
                        f"posts daily at {w[2]}.")
            if low[1] == "remove" and len(w) == 3 \
                    and w[2].isdigit():
                idx = int(w[2]) - 1
                if 0 <= idx < len(reminders):
                    gone = reminders.pop(idx)
                    self._rhythm_update(surface, key,
                                        reminders=reminders)
                    return (f"reminder removed: "
                            f"{gone.get('time')} — "
                            f"{gone.get('text') or 'rules card'}")
                return "no reminder with that number — " \
                    "`reminder list` shows them."
            return None
        if len(w) == 2 and low[0] == "quorum" and w[1].isdigit():
            n = max(1, min(50, int(w[1])))
            self._rhythm_update(surface, key, poll_quorum=n)
            return (f"poll quorum: {n} — a treasury poll needs "
                    "at least that many votes cast before its "
                    "result can move money.")
        return None

    def _apply_review_grammar(self, surface, key, words,  # type: ignore[no-untyped-def]
                              by: str = "admin"):
        """The §26 review grammar, shared by /gsettings (words
        after the `review` head) and the console's rhythm
        section.  Forms: `off|auto|assisted|manual` · `queue` ·
        `approve N` · `reject N` (N = the number in the queue
        listing)."""
        w = [str(x) for x in words or []]
        low = [x.lower() for x in w]
        if not w:
            return None
        if len(w) == 1 and low[0] in ("off", "auto", "assisted",
                                      "manual"):
            mode = low[0]
            self._rhythm_update(surface, key,
                                review={"mode": mode})
            return {
                "off": "contribution review: off — contributions "
                       "count toward the rankers as they land, "
                       "as they always have.",
                "auto": "contribution review: auto — the "
                        "ensemble decides (length, structure, "
                        "duplication); members see why and can "
                        "appeal once.",
                "assisted": "contribution review: assisted — "
                            "clear cases decide themselves; "
                            "borderline ones wait in the queue "
                            "for an admin.",
                "manual": "contribution review: manual — every "
                          "contribution waits in the queue for "
                          "an admin's ruling.",
            }[mode]
        if len(w) == 1 and low[0] in ("queue", "list"):
            queue = surface.contribution_queue(key)
            if not queue:
                return "the review queue is empty."
            lines = ["📋 *Contribution review queue*"]
            for i, event in enumerate(queue, 1):
                label = surface._member_label(
                    key, str(event.get("user_id") or ""))
                flag = " (appeal)" if event.get("appealed") \
                    else ""
                lines.append(
                    f"{i}. {label}{flag} — score "
                    f"{event.get('score')}: "
                    f"{str(event.get('text') or '')[:90]}")
            lines.append("rule: /gsettings review approve N · "
                         "reject N")
            return "\n".join(lines)
        if len(w) == 2 and low[0] in ("approve", "reject") \
                and w[1].isdigit():
            queue = surface.contribution_queue(key)
            idx = int(w[1]) - 1
            if not (0 <= idx < len(queue)):
                return ("no queue item with that number — "
                        "`review queue` lists them.")
            event = queue[idx]
            out = surface.review_contribution(
                key, str(event.get("id") or ""),
                approve=low[0] == "approve", by=by)
            if not out.get("ok"):
                return "that item was already decided."
            label = surface._member_label(
                key, str(event.get("user_id") or ""))
            return (f"{'✅ approved' if low[0] == 'approve' else '🚫 rejected'}"
                    f" — {label}'s contribution now "
                    f"{'counts' if low[0] == 'approve' else 'does not count'}"
                    " toward the rankers.")
        return None

    # ---- polls / milestones / contributions (P3) -----------------
    def _poll_card(self, message, surface, key, poll):  # type: ignore[no-untyped-def]
        """The house poll card: one counting core (the surface),
        rendered per platform — vote buttons where the platform
        carries them, /vote numbers in the text everywhere."""
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card

        settings = surface.store.settings(key)
        card = Card("announcement", title="Poll", icon="🗳️")
        card.line(f"**{poll['question']}**")
        for i, option in enumerate(poll.get("options") or [], 1):
            card.line(f"{i}. {option}")
        left = int(float(poll.get("closes_at") or 0)
                   - float(getattr(message, "ts", 0) or 0))
        bits = []
        if left > 0:
            bits.append(f"closes in {_duration_text(left)}")
        if poll.get("kind") == "treasury":
            bits.append(f"treasury vote — quorum "
                        f"{poll.get('quorum')}")
        if poll.get("changeable"):
            bits.append("votes can change until close")
        else:
            bits.append("one vote, final")
        bits.append("vote with /vote <option number>")
        card.note(" · ".join(bits))
        spec = [[("poll", f"{i}. {option}",
                  f"vote {poll['id']} {i}")]
                for i, option in enumerate(
                    poll.get("options") or [], 1)]
        buttons = buttons_from_spec(spec, settings=settings)
        return self._card(message, card, buttons=buttons)

    def _control_poll(self, tail, message):  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        if not self._system_send_ok(surface, key, message,
                                    "poll"):
            return ("polls run where Devon's bot systems are "
                    "claimed — this room isn't one of those.")
        raw = str(tail or "").strip()
        sender = str(message.sender_id or message.sender or "")
        if not raw:
            open_polls = surface.polls(key, open_only=True)
            if not open_polls:
                return ("no open polls. Start one: /poll "
                        "<question> | <option> | <option>")
            now = float(getattr(message, "ts", 0) or time.time())
            lines = ["🗳️ Open polls:"]
            for i, poll in enumerate(open_polls, 1):
                left = int(float(poll.get("closes_at") or 0)
                           - now)
                lines.append(
                    f"{i}. {poll['question']} — "
                    f"{len(poll.get('votes') or {})} vote(s)"
                    + (f", closes in {_duration_text(left)}"
                       if left > 0 else ""))
            lines.append("vote: /vote <poll number> <option "
                         "number> — or just /vote <option> on "
                         "the newest.")
            return "\n".join(lines)
        if raw.lower() == "close" or raw.lower().startswith(
                "close "):
            rest = raw[5:].strip()
            open_polls = surface.polls(key, open_only=True)
            if not open_polls:
                return "no open poll to close."
            poll = None
            if rest.isdigit():
                idx = int(rest) - 1
                if 0 <= idx < len(open_polls):
                    poll = open_polls[idx]
            else:
                mine = [p for p in open_polls
                        if str(p.get("by") or "") == sender]
                poll = mine[-1] if mine else None
                if poll is None and self._group_admin_ok(message):
                    poll = open_polls[-1]
            if poll is None:
                return ("name the poll's number from /poll — "
                        "or the poll's creator / an admin "
                        "closes it.")
            if str(poll.get("by") or "") != sender \
                    and not self._group_admin_ok(message):
                return ("only the poll's creator or an admin "
                        "can close it early.")
            out = surface.close_poll(key, str(poll["id"]),
                                     by=sender,
                                     now=float(getattr(
                                         message, "ts", 0)
                                         or time.time()))
            return out.get("text") or "poll closed."
        if raw.lower().startswith("treasury"):
            # /poll treasury grant @member 500 [reason…] — a
            # Yes/No vote whose passing EXECUTES the grant
            # through governance on close.
            if not self._group_admin_ok(message) \
                    and not surface.gang_power_actor(key, sender):
                return "treasury polls are for admins."
            key2, err = self._economy_gate(message)
            if err:
                return err
            parts = raw.split()
            if len(parts) < 4 or parts[1].lower() != "grant":
                return ("usage: /poll treasury grant @member "
                        "500 [reason]")
            amount = None
            target_token = ""
            for part in parts[2:]:
                if part.isdigit() and amount is None:
                    amount = int(part)
                elif not target_token:
                    target_token = part
            if amount is None or not target_token:
                return ("usage: /poll treasury grant @member "
                        "500 [reason]")
            recipient, miss = self._named_ref(
                surface, key, target_token, message)
            if not recipient:
                return miss
            reason = raw.split(str(amount), 1)[-1].strip() \
                if str(amount) in raw else ""
            label = surface._member_label(key, recipient)
            out = surface.create_poll(
                key, question=f"Grant {amount:,} sparks to "
                              f"{label} from the treasury?",
                options=["Yes", "No"], by=sender,
                kind="treasury",
                action={"kind": "grant",
                        "payload": {"recipient": recipient,
                                    "amount": amount,
                                    "reason": reason
                                    or "treasury poll"}},
                now=float(getattr(message, "ts", 0)
                          or time.time()))
            if not out.get("ok"):
                return "that poll couldn't be opened."
            return self._poll_card(message, surface, key,
                                   out["poll"])
        segments = [s.strip() for s in raw.split("|")]
        question = segments[0] if segments else ""
        options: list[str] = []
        duration = None
        changeable = False
        for seg in segments[1:]:
            low = seg.lower()
            if low == "flexible":
                changeable = True
            elif low.startswith("for "):
                secs, _rest = _parse_duration(seg[4:].strip())
                if secs:
                    duration = float(secs)
            elif seg:
                options.append(seg)
        if not question or len(options) < 2:
            return ("usage: /poll <question> | <option> | "
                    "<option> — add `| for 2h` to shorten it, "
                    "`| flexible` to let votes change.")
        out = surface.create_poll(
            key, question=question, options=options, by=sender,
            duration_s=duration, changeable=changeable,
            now=float(getattr(message, "ts", 0) or time.time()))
        if not out.get("ok"):
            return "that poll couldn't be opened."
        return self._poll_card(message, surface, key,
                               out["poll"])

    def _control_vote(self, tail, message):  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        sender = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        now = float(getattr(message, "ts", 0) or time.time())
        poll = None
        option = None
        if len(parts) == 1 and parts[0].isdigit():
            option = int(parts[0]) - 1
        elif len(parts) == 2 and parts[0].isdigit() \
                and parts[1].isdigit():
            open_polls = surface.polls(key, open_only=True)
            idx = int(parts[0]) - 1
            if not (0 <= idx < len(open_polls)):
                return "no open poll with that number — /poll lists them."
            poll = open_polls[idx]
            option = int(parts[1]) - 1
        elif len(parts) == 2 and parts[1].isdigit():
            # the button form: vote <poll ref> <option number>
            # (the ref rides callback data only, never text)
            poll = surface.get_poll(key, parts[0])
            if poll is None:
                return "that poll is gone."
            option = int(parts[1]) - 1
        if option is None:
            return ("usage: /vote <option number> (newest "
                    "poll), or /vote <poll number> <option "
                    "number>.")
        out = surface.vote_poll(
            key, str(poll["id"]) if poll else "", sender,
            option, now=now)
        if not out.get("ok"):
            return {"no_poll": "no open poll to vote on.",
                    "closed": "that poll has closed.",
                    "option": "no option with that number.",
                    "locked": "your vote is in and this poll "
                              "doesn't allow changing it.",
                    }.get(str(out.get("reason") or ""),
                          "the vote didn't land.")
        poll = out["poll"]
        counts = surface._poll_tally(poll)
        choice = (poll.get("options") or [])[option]
        return (f"🗳️ vote counted — **{choice}** "
                f"({counts[option]} so far"
                + (", changed from your earlier vote"
                   if out.get("changed") else "")
                + ").")

    def _control_milestones(self, tail, message):  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        entries = surface.milestones(key, limit=10)
        if not entries:
            return ("the group's memory book is empty — "
                    "milestones (first gang, message landmarks, "
                    "join anniversaries, economy records) are "
                    "written here as they happen.")
        lines = ["📖 *The group's memory book*"]
        for entry in entries:
            when = time.strftime(
                "%Y-%m-%d",
                time.gmtime(float(entry.get("ts") or 0)))
            lines.append(f"{when} — {entry.get('text') or ''}")
        return "\n".join(lines)

    def _control_contributions(self, tail, message):  # type: ignore[no-untyped-def]
        """§26 transparency: a member sees their own counted /
        pending / not-counted contributions and why, and can
        appeal a rejection once.  `top` shows the reviewed
        ranker — the board the review feeds."""
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        sender = str(message.sender_id or message.sender or "")
        raw = str(tail or "").strip()
        low = raw.lower()
        if low == "top":
            from ...social.chat.typography import Card

            rows = surface.contribution_ranking(key, limit=10)
            card = Card("leaderboard",
                        title="Contribution board (reviewed)",
                        icon="🏆")
            if not rows:
                card.line("No counted contributions yet — "
                          "substantive posts are what this "
                          "board counts.")
            else:
                card.stats([(r["label"], str(r["counted"]))
                            for r in rows])
            card.note("raw activity still counts every message "
                      "(/member); this board counts reviewed "
                      "contributions only." if
                      surface.review_mode(key) != "off" else
                      "contribution review is off here — every "
                      "contribution counts as it lands.")
            return self._card(message, card)
        if low.startswith("appeal"):
            parts = raw.split()
            if len(parts) != 2 or not parts[1].isdigit():
                return "usage: /contributions appeal <number from your list below>"
            mine = surface.contributions_of(key, sender)
            rejected = [e for e in mine
                        if e.get("state") == "rejected"]
            idx = int(parts[1]) - 1
            if not (0 <= idx < len(rejected)):
                return "no rejected contribution with that number in your list."
            out = surface.appeal_contribution(
                key, str(rejected[idx].get("id") or ""), sender)
            if not out.get("ok"):
                return {"not_yours": "that one isn't yours.",
                        "not_rejected": "that one wasn't rejected.",
                        "appealed": "you already appealed that one — an admin's ruling on an appeal is final.",
                        }.get(str(out.get("reason") or ""),
                              "the appeal didn't land.")
            return ("appeal filed — the contribution is back "
                    "in the admin review queue; their ruling "
                    "is final.")
        mode = surface.review_mode(key)
        mine = surface.contributions_of(key, sender)
        member = surface.member(key, sender) or {}
        counted = int((member.get("contributions") or {}).get(
            "counted") or 0)
        lines = [f"📚 *Your contributions* — review mode: "
                 f"{mode}"]
        lines.append(f"counted toward the rankers: {counted}")
        if not mine:
            lines.append("nothing filed yet — substantive "
                         "posts (a real point, a source, an "
                         "answer) are what gets filed.")
            return "\n".join(lines)
        pending = [e for e in mine
                   if e.get("state") == "pending"]
        rejected = [e for e in mine
                    if e.get("state") == "rejected"]
        if pending:
            lines.append(f"⏳ waiting on review: {len(pending)}")
        for i, event in enumerate(rejected[-5:], 1):
            why = "; ".join(event.get("reasons") or []) \
                or "scored below the bar"
            appealed = " (appealed)" if event.get("appealed") \
                else ""
            lines.append(
                f"✗ not counted{appealed}: "
                f"{str(event.get('text') or '')[:60]}… — {why}")
        if rejected:
            lines.append("appeal once per item: "
                         "/contributions appeal <number> — "
                         "the numbers count your rejected "
                         "items, oldest first.")
        return "\n".join(lines)

    # ---- the conflict feed (§35 announcements) -------------------
    def _post_conflict_feed(self, surface, key, message) -> None:  # type: ignore[no-untyped-def]
        """Drain the surface's queued conflict announcements
        (system-drop openings carry Claim buttons; inter-group
        war declarations arrive as lines).  §33 at send time
        and §31 quiet, like every system output: a refused
        room keeps its queue — and a drop whose announcement
        never posted stays claimable by /claimdrop, because
        the drop is state, not the post."""
        extras = surface._extras(key)
        if not (extras.get("conflict_feed") or []):
            return
        settings = surface.store.settings(key)
        quiet = (settings.get("defense") or {}).get(
            "autonomy", True) is False
        # Silent hours hold the feed too (P3): the queue stays
        # put and drains when the room wakes — drop announcements
        # are games/economy output, never urgent.
        if quiet or surface.rhythm_silent(key) \
                or not self._system_send_ok(
                    surface, key, message, "conflict feed"):
            return
        items = surface.conflict_feed_take(key)
        for item in items:
            try:
                if item.get("kind") == "line":
                    self._send(message.chat, str(item.get("text")
                                                 or ""))
                elif item.get("kind") == "sysdrop":
                    drop = item.get("drop") or {}
                    text, buttons = self._sysdrop_render(
                        surface, key, message, drop)
                    result = self._send(
                        message.chat, text, buttons=buttons,
                        parse_mode=self._rendered_mode(
                            message, text))
                    mid = getattr(result, "message_id", "") \
                        or ""
                    if mid:
                        msgs = dict(extras.get("sysdrop_msgs")
                                    or {})
                        msgs[str(drop.get("id"))] = str(mid)
                        extras["sysdrop_msgs"] = msgs
                        surface.store.save()
            except Exception:  # noqa: BLE001 - one bad item
                _log.warning("conflict feed item failed",  # never
                             exc_info=True)  # breaks the turn

    def _sysdrop_render(self, surface, key, message, drop):  # type: ignore[no-untyped-def]
        """The system-drop announcement: contents named (never
        an id), claim slots counted, the Claim button carrying
        this drop's own reference — a settlement affordance,
        not a notice, so no Undo rides it (§35 + the brief's
        settlements rule)."""
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card, sparks

        settings = surface.store.settings(key)
        card = Card("announcement", title="Supply drop",
                    icon="🪂")
        kind = str(drop.get("kind") or "")
        if kind in ("sparks", "sparks_big"):
            card.line(f"**{sparks(int(drop.get('amount') or 0))}**"
                      f" scattered in the open.")
            card.stats([("Per claim",
                         sparks(int(drop.get("per_claim") or 0)),
                         True),
                        ("Claims", str(drop.get("slots") or 1))])
        elif kind == "gems":
            card.line(f"**{int(drop.get('amount') or 0)} 💎** "
                      f"glinting in the open — one claim takes "
                      f"them.")
        elif kind in ("gear", "gear_rare"):
            from ...community.group_conflict import ARSENAL

            spec = ARSENAL.get(str(drop.get("item") or "")) or {}
            card.line(f"a **{spec.get('icon', '🎁')} "
                      f"{spec.get('name', 'piece of gear')}** "
                      f"lies in the open — one claim takes it.")
        card.note("first come, first served — claims are final")
        buttons = buttons_from_spec(
            [[("claim", "Claim", f"claimdrop {drop.get('id')}")]],
            settings=settings)
        rendered = self._card(message, card, buttons)
        return rendered.text, rendered.buttons

    # ---- group commands (P1 slice 2) --------------------------------
    def _control_group_help(self, message):  # type: ignore[no-untyped-def]
        """Bare /help INSIDE a group (owner order 2026-11-03):
        list what is actually available in this group — the member
        surface and the admin surface side by side, rendered from
        this group's live state, not the global owner catalog.
        Everyone sees both sections; the admin commands themselves
        stay gated at execution.

        Presentation build: the reply is an ActionableMessage —
        text and buttons authored from the SAME state (the
        sparks row exists exactly when the economy runs here; the
        buttons mirror the sections on screen).  Typography
        (§27): composed as a Card, settings class — the group's
        theme decides the form; on WhatsApp the card renders in
        the XMD box grammar (owner order 2026-11-05), the same
        frame as the WhatsApp menu — content unchanged, cut down
        to this room, claim block included."""
        from ...social.chat.typography import Card

        key = message.chat.key
        surface = self._group_surface()
        settings = surface.store.settings(key)
        members = surface.members(key)
        eco = surface.economy_on(key)
        # WhatsApp claim state (command census, gaps #2/#5):
        # WhatsApp has no Devon bot, so a room here runs her
        # group systems only when the owner designated it (§33)
        # — and until this block existed, nothing in the room
        # said so: the economy sections below simply vanished
        # and every refused command had to explain the wall
        # after the fact.  Stated here, in the room, before
        # anyone hits a refusal — read from the same
        # claim_state the §33 gate itself reads.  Telegram
        # rooms are claimed by her bot's presence; their help
        # is unchanged.
        wa_claim_lines: list[str] | None = None
        on_whatsapp = key.partition(":")[0].strip().lower() \
            in ("whatsapp", "wa")
        if on_whatsapp:
            claim = surface.claim_state(key)
            if claim["claimed"]:
                wa_claim_lines = [
                    "✅ switched on — the owner designated this "
                    "group, so my group systems run here: the "
                    "purse, games, polls, quests and the rest "
                    "below.",
                    "the switch is the owner's: "
                    "`/gsettings system output off` lifts the "
                    "designation.",
                ]
            else:
                wa_claim_lines = [
                    "⏸️ not switched on yet — WhatsApp has no "
                    "Devon bot, so I'm in this group on the "
                    "owner's own account, and my group systems "
                    "stay off in a room until the owner "
                    "designates it.",
                    "what works here today: /math, the group "
                    "pages — /rules · /member · /glog · "
                    "/boards · /milestones · /contributions — "
                    "and /gsettings for admins.",
                    "designation switches on the rest: the "
                    "purse (sparks — /balance · /work · /pay "
                    "· /gshop), the engine games, /poll, "
                    "property and gangs.  Switching a room on "
                    "is the owner's call: "
                    "`/gsettings system output on`.",
                ]
        card = Card("settings",
                    title=f"{message.chat.title or 'this group'} "
                          f"— what I do here",
                    icon="🛠️")
        everyone = [
            "/rules — the group rules",
            "/report — flag a message or member for the admins "
            "(reply to it, or /report @member why)",
            "/member [@name] — a member's card: tenure, activity, "
            "username/name changes",
            "/recheck — if the door rules let you in restricted "
            "(muted until you fix what's missing), tap this after "
            "fixing it and the checks re-run live",
            "/ping — check I'm alive, and how fast I answered",
            "🎮 games — /ttt /connect4 /reversi /checkers /mines "
            "and the rest of the board; /game lists them, moves are "
            "plain text (a1, b2 …). Some games queue you against "
            "players on other platforms.",
        ]
        if eco:
            everyone.append(
                "🪙 sparks (the group purse) — /balance · /daily · "
                "/mine · /work · /pay @member 50 · /deposit · "
                "/withdraw · /gshop · /quests · /treasury · /top · "
                "/tax · /claim when a drop is live. Chatting, "
                "streaks, quests and in-group game wins all earn.")
            everyone.append(
                "🥷 the risky side — /rob @member (fail and it's "
                "jail; /bail gets you out) · 🎰 /slots 10 · 🪙 "
                "/coinflip 10 — losses feed the treasury and the "
                "jackpot pot; triple 7️⃣ takes the pot.")
            everyone.append(
                "🏘️ property — /property (districts, the market "
                "index, your portfolio) · /buyproperty "
                "<district> <tier> · /rent collects daily rent "
                "as gems 💎 · /sellproperty · /listproperty "
                "(the member market) · /auctions (treasury "
                "escrow) · /mortgage (borrow against a "
                "building) · /insure (cover against repair "
                "waves). Gems buy the premium shelf in /gshop "
                "and the penthouse/tower tiers — they never "
                "come from sparks.")
        conflict_mode = surface._conflict_mode(key)
        if eco and conflict_mode != "off":
            everyone.append(
                "🧰 the arsenal — /gshop carries the conflict "
                "shelf (breach gear, defenses, armor, support); "
                "/gear is your loadout, /gear install <item> "
                "treasury|bank <district>|vault puts a defense "
                "on guard. Gear also falls from 🪂 supply drops "
                "— tap Claim when one lands (claims are final).")
            everyone.append(
                "💰 crime, geared — /shoot · /kill (a real "
                "weapon) · /assassinate (a heavy one) · "
                "/sabotage @member (wrecks their best property) "
                "· /heist start treasury|bank <district>|vault "
                "@member, /heist join, /heist go — crews of 2–5 "
                "against installed defenses, odds printed up "
                "front. Failure is jail; a kill puts them in "
                "hospital (/bail pays the bill, a medkit "
                "discharges free).")
            everyone.append(
                "🏴 gangs — /gang create <lifestyle> <name> "
                "(syndicate · cartel · bikers · street · mafia "
                "· vigilantes) · /gang join · /gang job · /gang "
                "deposit — crews grow into gangs (5+) and "
                "syndicates (15+); strength is members + armory "
                "+ territory. 🚔 /police is the precinct board: "
                "the wanted list, open bounties, your heat — "
                "the police hunt members who farm the weak and "
                "the new, never ordinary play. /bounty post "
                "@member 200 · /bounty hunt @member.")
            if conflict_mode == "warfare":
                everyone.append(
                    "⚔️ warfare — /war declare <gang> (in this "
                    "group, or across claimed groups) · /war "
                    "accept · /war join — escrowed pots, rounds "
                    "on the cycle, the winner takes the pot and "
                    "can claim a district: /gang claim "
                    "<district> holds ground that pays daily. "
                    "The gang holding 2+ districts and the "
                    "strength crown RULES: its leaders drive "
                    "/grant and /approve like admins, and set "
                    "the transfer tax with /gang tax N.")
        everyone.append(
            "🗳️ polls — /poll <question> | <option> | <option> "
            "opens one (tap a button or /vote <option number>); "
            "a treasury poll that passes actually spends — the "
            "group votes, the treasury moves.")
        everyone.append(
            "📚 contributions — substantive posts are what the "
            "group remembers: /contributions shows yours "
            "(counted, waiting, not counted — and why), "
            "/contributions top is the reviewed board, "
            "/milestones is the group's memory book.")
        everyone.append(
            "/help /balance — the full page for any command")
        if wa_claim_lines is not None:
            card.block_lines(wa_claim_lines,
                             title="this room, on WhatsApp:")
        card.block_lines(everyone, title="for everyone:")
        card.block_lines([
            "/gsettings — welcome & goodbye, the rules list, "
            "flood/link moderation, service-message cleanup, the "
            "economy switch and transfer tax",
            "/drop 100 10 — fund a giveaway pool from your own "
            "sparks (or /drop treasury 100 10 from the treasury); "
            "members claim with /claim",
            "/grant @member 100 — pay a member from the treasury; "
            "above the governance threshold a second admin must "
            "/approve it first",
            "/grant gems @member 25 — gift gems 💎 (the premium "
            "purse moves only through governance)",
            "/buyproperty public <district> <tier> — the treasury "
            "buys a public building; its rent funds drops and "
            "prizes (large buys go through /approve like any "
            "treasury spend)",
            # The mute family is not listed on WhatsApp: the
            # platform has no per-person mute to execute it
            # (owner ruling 2026-11-04 — a command the platform
            # cannot run does not sit on its surface).
            ("moderation, by hand: reply to a member's message "
             "with /warn, /ban or /kick — or name @member "
             "instead. /unban undoes a ban; /pin and /purge "
             "work on the message you reply to; warns pile up "
             "the ladder toward its final step (limit in "
             "/gsettings). WhatsApp has no per-person mute — "
             "removal is the strongest step there is."
             if on_whatsapp else
             "moderation, by hand: reply to a member's message "
             "with /warn, /mute 10m|2h|1d, /ban or /kick — or name "
             "@member instead. /unmute /unban undo them; /pin and "
             "/purge work on the message you reply to; warns pile "
             "up to an automatic mute (limit in /gsettings)."),
            # Member cleanup is Telegram machinery (member list +
            # deleted-account markers); on WhatsApp the line
            # states that truth instead of listing commands the
            # platform cannot run (owner ruling 2026-11-04).
            ("🧹 member cleanup: deleted accounts are removed "
             "automatically on my upkeep cycle (claimed groups "
             "where I hold admin power; journaled, never "
             "announced) — /cleanup scans and reports as a dry "
             "run, /rmdeleted runs the removal now, and "
             "/rminactive removes members inactive past the "
             "window by command (kicked, not banned; every "
             "outcome in /glog). The switches live in "
             "/gsettings (cleanup)."
             if not on_whatsapp else
             "🧹 member cleanup is a Telegram tool — WhatsApp "
             "gives me no member list to scan and no "
             "deleted-account markers to find."),
            # The §16 P5 kinds, same treatment as cleanup:
            # listed where the platform can run them; on
            # WhatsApp the line states the truth instead.
            ("🏷️ the §16 kit: /tag @member <words> — a "
             "member's tag (16 characters, no emoji; shows on "
             "/member) · /bansender — reply to a channel post "
             "to ban that CHANNEL identity (never its owner); "
             "/unbansender lifts it · /reactwipe @member — "
             "wipe their recent reactions · /boosters — who's "
             "boosting this room · auto-named forum topics "
             "rename themselves when the duty is on "
             "(/gsettings p5)"
             if not on_whatsapp else
             "🏷️ member tags, channel bans, reaction wipes "
             "and boosts are Telegram systems — WhatsApp has "
             "none of them."),
            "/glog — the group audit log (I keep it; I only show "
            "it when an admin asks)",
            "/strikes — the defense strike ledger: who is on "
            "the ladder, and how close the rungs are",
            "/joinq — the join queue: requests the acceptance "
            "rules held for a human, decided here or on the "
            "notice's buttons; the door rules themselves live "
            "in /gsettings (acceptance block)",
            "/flag @member <kind> [note] — reputation (§32): "
            "stage a cross-group record (scam · fraud · raid · "
            "harassment · spam-ring · other); an admin confirms "
            "it, then it warns every group they join — without "
            "ever deciding admission itself. Bare /flag reviews "
            "the proposals reports and sensor bans filed; "
            "/flag retract @member clears this group's records",
            "/rep @member — their reputation file: live records, "
            "sources, and what became of them",
            "/gsettings conflict off|crime|warfare — the "
            "conflict layer (§23): crime needs crime mode; "
            "wars, territory and governance capture need "
            "warfare. `/gsettings conflict drops N` sets the "
            "daily system-drop budget (§35, 0 = none); "
            "`conflict police on|off` stands the precinct up "
            "or down. Police actions land in /glog with their "
            "evidence; a police jail's Undo lifts the jail "
            "only — fines, raids and bounties are settlements.",
            "I moderate on my own: floods, repeat spam and "
            "off-allowlist links get deleted; crash payloads are "
            "quarantined, channel injections and mass tags are "
            "struck; repeat offenders are muted, kicked, then "
            "banned — all tuned in /gsettings.",
            "/gsettings rhythm — the room's clock: silent "
            "hours (system posts hold for a morning digest; "
            "moderation never holds), the group's own clock "
            "offset, the nightshield (the door tightens at "
            "night, drop/war events pause), the daily group "
            "briefing, reminders, and poll quorum. "
            "`/gsettings review …` runs §26 contribution "
            "review (off|auto|assisted|manual + the queue).",
            "/poll treasury grant @member 500 — a treasury "
            "poll: it passes with quorum and the grant executes "
            "through governance on close; it fails, nothing "
            "moves.",
        ], title="for admins:")
        card.note(
            f"{len(members)} members in my ledger · "
            f"{len(settings.get('rules') or [])} rules · welcome "
            f"{'on' if settings['welcome_enabled'] else 'off'} · "
            f"link moderation "
            f"{'allowlist' if (settings['moderation'].get('link_allowlist')) else 'open'}")
        from ...social.chat.expression import buttons_from_spec

        spec = [[("rules", "Rules", "rules"),
                 ("games", "Games", "game")]]
        if eco:
            spec.append([("balance", "My sparks", "balance"),
                         ("quests", "Quests", "quests"),
                         ("treasury", "Treasury", "treasury")])
        spec.append([("settings", "Group settings", "gsettings"),
                     ("glog", "Audit log", "glog")])
        buttons = buttons_from_spec(spec, settings=settings)
        return self._card(message, card, buttons, xmd_frame=True)

    def _group_key_or_none(self, message):  # type: ignore[no-untyped-def]
        if message is None:
            return None
        from ...social.chat.base import ChatKind

        if message.chat.kind != ChatKind.GROUP:
            return None
        return message.chat.key

    def _rules_page_url(self, key: str) -> str:
        """The group's public rules-page URL, or "" when no
        public base URL is configured or the room isn't claimed
        (an unclaimed room publishes no page to link to)."""
        settings = getattr(getattr(self, "context", None),
                           "settings", None)
        web = getattr(settings, "web", None)
        base = str(getattr(web, "base_url", "") or "").strip()
        if not base:
            return ""
        from ...social.site import group_slug, rules_url

        surface = self._group_surface()
        if key not in {k for k in surface.store.data
                       if surface.system_output_allowed(k)}:
            return ""
        title = ""
        try:
            db = getattr(getattr(self, "gateway", None), "db", None)
            if db is not None:
                row = db.query_one(
                    "SELECT title FROM chats WHERE key = ?", [key])
                if row:
                    title = str(row.get("title") or "")
        except Exception:  # noqa: BLE001 - the title only
            title = ""                    # pretties the slug
        return rules_url(base, group_slug(key, title))

    def _control_rules(self, message):  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        text = self._group_surface().rules_text(key)
        url = self._rules_page_url(key)
        if not url:
            return text
        platform = str(getattr(message.chat, "platform", "") or "")
        if platform == "telegram-bot":
            # The bot surface carries buttons: the page rides
            # as a native URL button (the closest Telegram
            # equivalent of an HTML link).  Everywhere else —
            # the userbot (its sends drop inline keyboards),
            # WhatsApp, Discord — the bare link goes in the
            # text, where it always works.
            from ...social.chat.tgbot_buttons import actionable

            return actionable(text, [[("📜 Rules page", url)]])
        return f"{text}\n📜 full rules page: {url}"

    def _control_member(self, arg, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        wanted = str(arg or "").strip()
        if wanted:
            # §29: the card names a person by every path, not
            # only the ledger's memory.
            res = self._resolver(surface).resolve_token(
                key, message, wanted)
            if not res.found:
                if res.channel_post:
                    return (f"that was posted as "
                            f"{res.channel_label} — a channel, "
                            f"not a member with a card.")
                return res.exhaustion_text()
            target = surface.member(key, res.member.user_id)
            if target is None:
                # Resolved on a platform surface but never
                # recorded: the card is built from what the
                # resolution itself proved.
                target = {"user_id": res.member.user_id,
                          "username": res.member.username,
                          "name": res.member.name,
                          "first_seen": None, "joined_ts": None,
                          "message_count": 0, "last_seen": None}
            return _member_card(surface, key, target)
        target = surface.members(key).get(
            str(message.sender_id or ""))
        if target is None:
            return "I don't have a card for you in this group yet."
        return _member_card(surface, key, target)

    def _control_glog(self, arg, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "the group log is for admins."
        try:
            n = int(str(arg or "10").strip() or 10)
        except ValueError:
            n = 10
        entries = self._group_surface().audit(
            key, limit=max(1, min(n, 50)))
        if not entries:
            return "nothing in the group log yet."
        from ...social.chat.typography import Card

        card = Card("settings", title="Group log", icon="📋")
        card.board("\n".join(_audit_line(entry)
                             for entry in entries))
        card.note("held by me, shown on demand")
        return self._card(message, card)

    def _control_gsettings(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "group settings are for admins."
        surface = self._group_surface()
        text = str(tail or "").strip()
        if not text:
            s = surface.store.settings(key)
            mod = s["moderation"]

            from ...social.chat.expression import button_label

            def _toggle(action: str, label: str, kind: str,
                        on_now: bool) -> tuple:
                # The button SHOWS the current state and carries
                # the flip — authored from the live settings, so
                # it can never disagree with the summary above it.
                # Its emoji comes from the one button vocabulary
                # (§21), under this group's presentation settings.
                return (button_label(
                            action,
                            f"{label}: {'ON' if on_now else 'OFF'}",
                            settings=s),
                        f"gsettings {kind} "
                        f"{'off' if on_now else 'on'}")

            pres = s.get("presentation") or {}
            theme_cycle = {"house": "clean", "clean": "board",
                           "board": "house"}
            theme_now = str(pres.get("theme") or "house")
            buttons = [
                [_toggle("welcome", "Welcome", "welcome",
                         bool(s["welcome_enabled"])),
                 _toggle("goodbye", "Goodbye", "goodbye",
                         bool(s["goodbye_enabled"]))],
                [_toggle("cleanup", "Cleanup", "clean",
                         bool(s["clean_service"])),
                 _toggle("economy", "Economy", "economy",
                         bool(s["economy_enabled"]))],
                [(button_label("theme", f"Theme: {theme_now}",
                               settings=s),
                  f"gsettings theme "
                  f"{theme_cycle.get(theme_now, 'house')}"),
                 _toggle("stamp", "Stamp", "stamp",
                         bool(pres.get("stamp", True)))],
            ]
            # Defense block (DESIGN §17): sensor states from the
            # same settings the pipeline reads + the platform-
            # honesty notes + what has degraded on THIS surface.
            from ...social.chat.group_defense import (
                defense_availability)
            d = s.get("defense") or {}
            d_ac = d.get("anticrash") or {}
            d_ch = d.get("antichannel") or {}
            d_tg = d.get("antitag") or {}
            d_dl = d.get("antidelete") or {}
            d_ed = d.get("antiedit") or {}
            d_esc = d.get("escalation") or {}
            avail = defense_availability(
                str(message.chat.platform or ""),
                self._defense_surface_kind(message))
            honesty = []
            if not avail["antidelete"]:
                honesty.append(avail["antidelete_note"])
            if not avail["antiedit"]:
                honesty.append(avail["antiedit_note"])
            if key.partition(":")[0].strip().lower() \
                    in ("whatsapp", "wa") \
                    and ((mod.get("filter_action") or "")
                         == "mute"
                         or int(s.get("mute_default_s") or 0) > 0):
                # Stored on another surface, or before the
                # platform rule: the values stay as data, but
                # the panel says plainly they cannot fire here.
                honesty.append(
                    "mute settings are stored for this group, "
                    "but WhatsApp has no per-person mute for "
                    "them to fire — at execution they refuse "
                    "honestly (see /glog) instead of pretending.")
            degraded = []
            for entry in surface.audit(key, limit=50):
                if entry.get("kind") == "moderation_unexecuted":
                    degraded.append(
                        f"{entry.get('action')} — "
                        f"{entry.get('reason')}")
            seen: set = set()
            degraded = [x for x in degraded
                        if not (x in seen or seen.add(x))][:3]
            # Typography (§27): the settings panel is a Card —
            # state blocks, then the change grammar as code
            # lines, each rendered per platform by the organ.
            from ...social.chat.typography import (
                DEFAULT_STAMP_TEXT, Card, resolve_style)

            card = Card("settings", title="Group settings",
                        icon="⚙️")
            card.block_lines([
                f"welcome: {'on' if s['welcome_enabled'] else 'off'} · "
                f"goodbye: {'on' if s['goodbye_enabled'] else 'off'} · "
                f"service cleanup: {'on' if s['clean_service'] else 'off'}",
                f"rules: {len(s.get('rules') or [])} · "
                f"flood: {mod['flood_max']} per "
                f"{int(mod['flood_window_s'])}s · repeats ×"
                f"{mod['repeat_threshold']} · links: "
                f"{', '.join(mod.get('link_allowlist') or []) or 'open'}"
                f" · filters: {len(mod.get('word_filters') or [])} "
                f"({mod.get('filter_action') or 'delete'})",
                f"economy: {'on' if s['economy_enabled'] else 'off'}"
                f" · transfer tax {s.get('transfer_tax_pct', 2)}%"
                f" · fine {s.get('fine_amount', 0)}"
                f" · governance ≥{s.get('governance_threshold', 500)}",
            ])
            defense_lines = [
                f"autonomy "
                f"{'ON' if d.get('autonomy', True) else 'OFF'}"
                f" (owner's switch — WhatsApp: any admin; "
                f"Telegram: group owner only)",
                f"anticrash: "
                f"{'on' if d_ac.get('enabled', True) else 'off'}"
                f" ({d_ac.get('action', 'delete')}) · antichannel: "
                f"{'on' if d_ch.get('enabled', True) else 'off'}"
                f" · antitag: "
                f"{'on' if d_tg.get('enabled', True) else 'off'}"
                f" (≤{d_tg.get('max_mentions', 10)} mentions or "
                f"{d_tg.get('max_roster_pct', 20)}% of the group)",
                f"antidelete: "
                f"{'on' if d_dl.get('enabled', True) else 'off'}"
                f" ({d_dl.get('mode', 'log')}) · antiedit: "
                f"{'on' if d_ed.get('enabled', True) else 'off'}"
                f" ({d_ed.get('mode', 'log')}) · "
                f"mute default: "
                f"{_duration_text(int(s.get('mute_default_s') or 0)) if s.get('mute_default_s') else 'ask'}",
            ]
            defense_lines += [f"⚠ {note}" for note in honesty]
            if degraded:
                defense_lines.append(
                    "degraded here: " + "; ".join(degraded))
            card.block_lines(defense_lines, title="Defense")
            game_style = resolve_style(s, "game")
            economy_style = resolve_style(s, "economy")
            card.block_lines([
                f"theme: {game_style.theme} · games render "
                f"{game_style.preset} · panels render "
                f"{economy_style.preset}",
                f"stamp: "
                f"{'ON' if pres.get('stamp', True) else 'OFF'}"
                f" — \"{pres.get('stamp_text') or DEFAULT_STAMP_TEXT}\""
                f" on presented panels"
                + (f" (classes: "
                   f"{', '.join(pres.get('stamp_classes') or [])})"
                   if pres.get("stamp_classes") else ""),
            ], title="Presentation")
            # Acceptance block (DESIGN §28): the door rules as
            # the evaluator reads them + the live queue depth +
            # the raid posture right now.
            acc = s.get("acceptance") or {}
            acc_crit = acc.get("criteria") or {}
            crit_on = [name for name in (
                "username", "photo", "firstname", "lastname",
                "hygiene", "banned", "bots") if acc_crit.get(name)]
            _acc_cfg, raid_now = surface.acceptance_cfg(key)
            queue_n = len(surface.acceptance_queue(key))
            card.block_lines([
                f"mode: {acc.get('mode', 'open')} · action on a "
                f"failed check: {acc.get('action', 'restricted')}"
                f" · grace {int(acc.get('grace_minutes') or 60)}m"
                f" · auto-approve "
                f"{'on' if acc.get('auto_approve', True) else 'off'}",
                "criteria on: "
                + (", ".join(crit_on) if crit_on
                   else "none — the door is open"),
                f"queue: {queue_n} waiting · raid posture: "
                f"{'ACTIVE (tightened)' if raid_now else 'calm'}",
            ], title="Acceptance")
            # Escalation block (DESIGN §30): the ladder EVERY
            # strike consumer reads — sensors, manual warns,
            # acceptance enforcement — as the group designed it.
            esc_now = surface.escalation_config(key)
            card.block_lines([
                esc_now.describe(),
                "every warning names the count of the limit, "
                "the next step, and when the oldest warn clears",
            ], title="Escalation")
            # Estate block (§17/§20 levers — their grammar lives
            # here and in the DM console).
            est = s.get("estate") or {}
            card.block_lines([
                f"market cut {est.get('market_cut_pct', 5)}% · "
                f"sell-back {est.get('sellback_pct', 70)}% · "
                f"ownership cap {est.get('ownership_cap', 12)} · "
                f"mortgage LTV {est.get('ltv_pct', 60)}% at "
                f"{est.get('mortgage_rate_pct', 5)}%",
            ], title="Estate")
            # Conflict block (§23/§24/§35): the mode IS the
            # layer's existence; the drop budget and the
            # precinct read from the same settings the organs
            # read.
            con = s.get("conflict") or {}
            con_health = surface.conflict.health(key)
            card.block_lines([
                f"mode: {con.get('mode', 'off')} — crime verbs, "
                f"heists, gangs and drops need crime; wars, "
                f"territory and capture need warfare",
                f"system drops: {con.get('drops_per_day', 2)}"
                f"/day budget · police: "
                f"{'on' if con.get('police', True) else 'off'}"
                f" · war escrow {con.get('war_escrow', 200)}",
                (f"street now: {con_health.get('gangs', 0)} "
                 f"gangs · {con_health.get('wars', 0)} wars · "
                 f"{con_health.get('wanted', 0)} wanted · "
                 f"{con_health.get('open_bounties', 0)} bounties"
                 if con_health.get("ok") else
                 "street now: the conflict state is unreadable "
                 "— the layer is frozen (see /glog)"),
            ], title="Conflict")
            # System-output claim (§33): where Devon's bot
            # systems may speak and accrue in this room — the
            # two-hats law, read from the same state the gate
            # itself reads.
            claim_now = surface.claim_state(key)
            if claim_now["claimed"]:
                claim_line = (
                    f"system output: ON — claimed "
                    f"({claim_now['source']})")
            else:
                claim_line = (
                    "system output: OFF — my bot doesn't run "
                    "here; my sessions only watch and converse")
            card.block_lines([
                claim_line,
                "the owner designates a room with "
                "`/gsettings system output on`",
            ], title="System output")
            # Expression block (§21): her GIF/sticker voice in
            # this room — state read from the same settings the
            # moment service reads, library size included.
            expr = s.get("expression") or {}
            lib_n = ""
            try:
                svc = getattr(self, "_expression_service", None)
                if svc is not None:
                    lib_n = (f" · library "
                             f"{len(svc().library)} items")
            except Exception:  # noqa: BLE001 - display only
                lib_n = ""
            card.block_lines([
                f"expression: "
                f"{'on' if expr.get('enabled', True) else 'off'}"
                f" · gifs {expr.get('gifs', 'normal')}"
                f" · welcome media "
                f"{'on' if expr.get('welcome_media', True) else 'off'}"
                f" · ban theatre "
                f"{'on' if expr.get('ban_theatre', True) else 'off'}"
                f" · auto-collect "
                f"{'on' if expr.get('auto_collect', False) else 'off'}"
                f"{lib_n}",
                "warnings stay dry at every setting — a warning "
                "is a record, not a show",
            ], title="Expression")
            # Reputation block (§32): this group's warning mode +
            # the live shape of the book and its queue here.
            rep = s.get("reputation") or {}
            rep_stats = surface.reputation.stats()
            pending_n = len(surface.flag_proposals(key))
            card.block_lines([
                f"join warnings: {rep.get('warning', 'public')}"
                f" — what happens when someone carrying a "
                f"cross-group record is admitted here",
                f"the book: {rep_stats['live']} live records "
                f"across every group I serve"
                + (f" · {pending_n} flag proposal"
                   f"{'s' if pending_n != 1 else ''} waiting "
                   f"here (/flag reviews them)"
                   if pending_n else ""),
                "a record never decides admission — the door "
                "rules above do; it only warns, and only about "
                "what an admin somewhere confirmed",
            ], title="Reputation")
            # Rhythm block (P3): the room's clock — silent
            # hours, the nightshield, the briefing, reminders,
            # poll quorum and the §26 review mode, read from
            # the same settings the cycle reads.
            from ...social.chat.group_rhythm import (
                format_offset as _fmt_offset)

            rhy = s.get("rhythm") or {}
            rhy_review = (rhy.get("review") or {}).get(
                "mode", "off")
            rhy_queue = len(surface.contribution_queue(key))
            card.block_lines([
                f"silent hours: "
                f"{'on' if rhy.get('silent_enabled') else 'off'}"
                f" · {rhy.get('silent_start', '23:00')}–"
                f"{rhy.get('silent_end', '07:00')} "
                f"{_fmt_offset(int(rhy.get('tz_offset_min') or 0))}"
                f" · nightshield "
                f"{'on' if rhy.get('nightshield') else 'off'}",
                f"briefing: "
                f"{'on' if rhy.get('briefing_enabled') else 'off'}"
                f" at {rhy.get('briefing_hour', 8)}:00 group "
                f"time · reminders: "
                f"{len(rhy.get('reminders') or [])} set, "
                f"{'on' if rhy.get('reminders_enabled', True) else 'off'}",
                f"polls: quorum {rhy.get('poll_quorum', 3)} · "
                f"contribution review (§26): {rhy_review}"
                + (f" · {rhy_queue} waiting in the queue"
                   if rhy_queue else ""),
            ], title="Rhythm")
            # Boards (§14): what this group publishes, and
            # when — the same block the console's boards
            # section and the cycle read.
            brd = s.get("boards") or {}
            card.block_lines([
                f"daily: {', '.join(brd.get('daily') or []) or '—'}"
                f" · weekly: "
                f"{', '.join(brd.get('weekly') or []) or '—'}"
                f" · publish hour "
                f"{int(brd.get('hour', 20))}:00 group time",
                "boards post on the cycle and hold through "
                "silent hours; movement reads the last "
                "published snapshot — /boards peeks never move "
                "the baseline",
            ], title="Boards")
            # Member cleanup: the deleted-account scan + the
            # inactivity sweep — state read from the same block
            # the commands read.
            cln = s.get("cleanup") or {}
            card.block_lines([
                f"removals: "
                f"{'on' if cln.get('enabled', True) else 'OFF'}"
                f" · auto-remove deleted: "
                f"{'on' if cln.get('auto_deleted', True) else 'OFF'}"
                f" · inactivity window: "
                f"{int(cln.get('inactive_days') or 90)} days",
                "deleted accounts are removed automatically "
                "on the upkeep cycle (claimed groups where I "
                "hold admin power, under the defense autonomy "
                "switch; journaled in /glog, never announced) · "
                "/cleanup scans and reports · /rmdeleted runs "
                "the removal now · /rminactive removes members "
                "inactive past the window, by command only — "
                "Telegram groups only; admins, bots and my own "
                "accounts are never removed",
            ], title="Member cleanup")
            # The §16 P5 layer's switches — Telegram rooms
            # only: the block doesn't render on WhatsApp, whose
            # surface carries none of these systems.
            _on_wa = str(key).partition(":")[0].strip().lower() \
                in ("whatsapp", "wa")
            if not _on_wa:
                p5 = s.get("p5") or {}
                card.block_lines([
                    f"topic auto-rename: "
                    f"{'on' if p5.get('topics_rename') else 'OFF'}"
                    f" · admin command scopes: "
                    f"{'on' if p5.get('member_scopes') else 'OFF'}"
                    f" · ephemeral answers: "
                    f"{'on' if p5.get('ephemeral') else 'OFF'}",
                    "auto-named forum topics get a proper name "
                    "from their first message · scopes sync "
                    "gives this room's admins their full "
                    "command menu (bot surface) · ephemeral "
                    "sends sensitive replies (/balance) to the "
                    "asker only · /tag, /bansender, "
                    "/reactwipe and /boosters ride along",
                ], title="P5 layer")
            # The ladder's step menu (§30): on WhatsApp the
            # mute/restrict steps are not offered — the
            # platform cannot fire them (see
            # _apply_escalation_grammar).
            escalation_steps_offer = (
                "`/gsettings escalation at N fine AMOUNT` · "
                "`… at N kick` · `… clear N` · `… reset` — no "
                "mute or restrict step on WhatsApp: there is "
                "no per-person mute for the ladder to fire"
                if key.partition(":")[0].strip().lower()
                in ("whatsapp", "wa") else
                "`/gsettings escalation at N mute [10m|2h]` · "
                "`… at N fine AMOUNT` · `… at N restrict|kick` · "
                "`… clear N` · `… reset`")
            # The mute-bottomed settings get the same treatment
            # (owner ruling 2026-11-04): on WhatsApp the offers
            # state the platform truth instead of listing a
            # mute the platform cannot fire.
            _on_wa = key.partition(":")[0].strip().lower() \
                in ("whatsapp", "wa")
            mute_default_offer = (
                "`/gsettings mute default off` — no standing "
                "mute length on WhatsApp: there is no "
                "per-person mute for one to fire"
                if _on_wa else
                "`/gsettings mute default 2h|off`")
            filter_action_offer = (
                "`/gsettings filter action delete|warn` — no "
                "mute action on WhatsApp: there is no "
                "per-person mute for it to fire"
                if _on_wa else
                "`/gsettings filter action delete|warn|mute`")
            card.block_lines([
                "`/gsettings welcome on|off` · "
                "`/gsettings goodbye on|off` · "
                "`/gsettings clean on|off`",
                "`/gsettings rules add TEXT` · "
                "`/gsettings rules clear`",
                "`/gsettings flood N` · "
                "`/gsettings links DOMAINS|off`",
                "`/gsettings economy on|off` · "
                "`/gsettings tax PCT` · "
                "`/gsettings warns N` · "
                + mute_default_offer,
                "`/gsettings fine N` · "
                "`/gsettings governance N`",
                "`/gsettings filter add WORD` · "
                + filter_action_offer,
                "`/gsettings autonomy on|off` · "
                "`/gsettings anticrash on|off|delete|flag`",
                "`/gsettings antichannel on|off` · "
                "`/gsettings antitag on|off` · "
                "`/gsettings antitag N`",
                "`/gsettings antidelete on|off` · "
                "`/gsettings antidelete mode log|callout`",
                "`/gsettings antiedit on|off` · "
                "`/gsettings antiedit mode log|callout`",
                "`/gsettings theme house|clean|board` · "
                "`/gsettings class game board|clean|default`",
                "`/gsettings stamp on|off` · "
                "`/gsettings stamp text TEXT|default` · "
                "`/gsettings stamp classes a,b|default`",
                "`/gsettings acceptance mode open|request|invite`",
                "`/gsettings acceptance username|photo|firstname"
                "|lastname|hygiene|banned|bots on|off`",
                "`/gsettings acceptance action decline|restricted"
                "|kick` · `/gsettings acceptance grace MIN`",
                "`/gsettings acceptance auto on|off` · "
                "`/gsettings acceptance override CRITERION "
                "decline|restricted|kick|default`",
                "`/gsettings acceptance whitelist add|remove "
                "@name|id`",
                "`/gsettings escalation limit N` · "
                "`/gsettings escalation final kick|ban` · "
                "`/gsettings escalation window DAYS`",
                escalation_steps_offer,
                "`/gsettings estate cut PCT` · "
                "`/gsettings estate sellback PCT` · "
                "`/gsettings estate cap N` · "
                "`/gsettings estate ltv PCT` · "
                "`/gsettings estate rate PCT`",
                "`/gsettings system output on|off` "
                "(owner — designates this room for my "
                "bot-system output, §33)",
                "`/gsettings expression on|off` · "
                "`/gsettings expression gifs off|low|normal`",
                "`/gsettings expression welcome on|off` · "
                "`/gsettings expression ban on|off` · "
                "`/gsettings expression collect on|off`",
                "`/gsettings reputation public|admins|restricted"
                "|off` — join warnings for flagged members (§32)",
                "`/gsettings conflict off|crime|warfare` · "
                "`/gsettings conflict drops N` (system drops "
                "per day, §35) · `/gsettings conflict police "
                "on|off` · `/gsettings conflict escrow N` (war "
                "escrow from gang treasuries)",
                "`/gsettings rhythm on|off` · "
                "`/gsettings rhythm silent HH:MM HH:MM` · "
                "`/gsettings rhythm tz +1|+05:30` (the group's "
                "clock) · `/gsettings rhythm nightshield "
                "on|off`",
                "`/gsettings rhythm briefing on|off` · "
                "`… briefing hour N` · `/gsettings rhythm "
                "reminders on|off` · `… reminder add HH:MM "
                "TEXT` · `… reminder rules HH:MM` · "
                "`… reminder remove N` · "
                "`/gsettings rhythm quorum N`",
                "`/gsettings review off|auto|assisted|manual` "
                "(§26) · `/gsettings review queue` · "
                "`/gsettings review approve N|reject N`",
                "`/gsettings boards daily sparks messages` · "
                "`/gsettings boards weekly contributions` · "
                "`/gsettings boards hour N` · "
                "`/gsettings boards off`",
                "`/gsettings cleanup on|off` · "
                "`/gsettings cleanup auto on|off` · "
                "`/gsettings cleanup days N` — member cleanup "
                "(deleted + inactive members), not the "
                "service-message cleanup above",
            ], title="Change it")
            card.note("tap a button to flip a switch, or type a "
                      "command")
            return self._card(message, card, buttons)
        parts = text.split(None, 2)
        head = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        rest2 = parts[2] if len(parts) > 2 else ""
        on = rest.lower() == "on"
        if head == "system" and rest.lower() == "output" \
                and rest2.lower() in ("on", "off"):
            # §33 designation is the OWNER's call: it decides
            # where his personal sessions may speak as the bot
            # system, so a group admin alone cannot flip it.
            if not self._operator_or_false(message):
                return ("designating a room for my system "
                        "output is the owner's call — it "
                        "decides where my bot systems may "
                        "speak through his accounts.")
            state = surface.set_designated(
                key, rest2.lower() == "on",
                by=str(message.sender_id or ""))
            if state["designated"]:
                return ("system output: this room is "
                        "designated — my bot systems (economy, "
                        "quests, games, notices) run here.")
            return ("system output: designation lifted — "
                    "this room is claimed only where my bot "
                    "itself operates.")
        if head == "expression":
            # Media expression (§21): on/off, GIF frequency,
            # welcome media, ban theatre — the same helper the
            # DM console's expression section applies.
            out = self._apply_expression_grammar(
                surface, key, text.split()[1:])
            if out is not None:
                return out
            return ("usage: /gsettings expression on|off · "
                    "expression gifs off|low|normal · "
                    "expression welcome on|off · "
                    "expression ban on|off · "
                    "expression collect on|off")
        if head == "reputation":
            # Cross-group reputation (§32): the join-warning
            # mode — the same helper the DM console's reputation
            # section applies.
            out = self._apply_reputation_grammar(
                surface, key, text.split()[1:])
            if out is not None:
                return out
            return ("usage: /gsettings reputation "
                    "public|admins|restricted|off")
        if head == "conflict":
            # The conflict layer (§23/§24/§35): mode, the drop
            # budget, the precinct, the war escrow — the same
            # helper the DM console's conflict section applies.
            out = self._apply_conflict_grammar(
                surface, key, text.split()[1:])
            if out is not None:
                return out
            return ("usage: /gsettings conflict "
                    "off|crime|warfare · conflict drops N · "
                    "conflict police on|off · conflict escrow N")
        if head == "rhythm":
            # The room's clock (P3): silent hours, timezone,
            # nightshield, briefing, reminders, poll quorum —
            # the same helper the DM console's rhythm section
            # applies.
            out = self._apply_rhythm_grammar(
                surface, key, text.split()[1:])
            if out is not None:
                return out
            return ("usage: /gsettings rhythm on|off · rhythm "
                    "silent HH:MM HH:MM · rhythm tz +1 · rhythm "
                    "nightshield on|off · rhythm briefing "
                    "on|off|hour N · rhythm reminder add HH:MM "
                    "TEXT · rhythm quorum N")
        if head == "boards":
            # §14 board publishing: the specs this group posts
            # on the daily/weekly beat — the same helper the
            # console's boards section applies.
            out = self._apply_boards_grammar(
                surface, key, text.split()[1:])
            if out is not None:
                return out
            return ("usage: /gsettings boards daily "
                    "sparks|gems|messages|contributions|gangs "
                    "(or full board ids) · boards weekly "
                    "<spec...> · boards hour N · boards off")
        if head == "cleanup":
            # Member cleanup (deleted + inactive): the removals
            # switch and the inactivity window — the same
            # helper the console's cleanup section applies.
            # NOT `clean` above: that's the service-message
            # cleanup, a different switch.
            out = self._apply_cleanup_grammar(
                surface, key, text.split()[1:])
            if out is not None:
                return out
            return ("usage: /gsettings cleanup on|off · "
                    "cleanup auto on|off · cleanup days N")
        if head in ("p5", "topics", "scopes", "ephemeral"):
            # The §16 P5 layer's switches: the topic
            # auto-rename duty, per-member command scopes, and
            # ephemeral answers — the same helper the console's
            # p5 section applies.
            out = self._apply_p5_grammar(
                surface, key, text.split(),
                by=str(message.sender_id or ""), message=message)
            if out is not None:
                return out
            return ("usage: /gsettings topics on|off · scopes "
                    "on|off|sync|clear · ephemeral on|off")
        if head == "review":
            # §26 contribution review: the mode and the queue —
            # the same helper the console's rhythm section
            # applies.
            out = self._apply_review_grammar(
                surface, key, text.split()[1:],
                by=str(message.sender_id or ""))
            if out is not None:
                return out
            return ("usage: /gsettings review "
                    "off|auto|assisted|manual · review queue · "
                    "review approve N · review reject N")
        if head == "warns" and rest.isdigit():
            surface.update_settings(
                key, warn_threshold=max(1, int(rest)))
            return f"warn limit {max(1, int(rest))} before an auto-mute."
        if head == "mute" and rest.lower() == "default":
            if rest2.lower() in ("off", "none", "0", ""):
                surface.update_settings(key, mute_default_s=0)
                return ("no standing mute length — when a member "
                        "is named without one, I'll ask.")
            if key.partition(":")[0].strip().lower() \
                    in ("whatsapp", "wa"):
                # Platform truth (owner ruling 2026-11-04, the
                # ladder rule extended): a mute length cannot be
                # newly selected where no per-person mute exists
                # to fire it. Clearing (off) above stays allowed;
                # a value stored earlier keeps its data and its
                # execution refuses honestly.
                return ("that can't be set here: WhatsApp has "
                        "no per-person mute, so a standing mute "
                        "length would never fire — removal is "
                        "the strongest step this platform has. "
                        "Nothing changed.")
            secs, _why = _parse_duration(rest2)
            if secs:
                surface.update_settings(key, mute_default_s=secs)
                return (f"standing mute length "
                        f"{_duration_text(secs)} when a named "
                        f"member is given none.")
            return "usage: /gsettings mute default 10m|2h|1d|off"
        if head == "fine" and rest.lstrip("-").isdigit():
            amount = max(0, int(rest))
            surface.update_settings(key, fine_amount=amount)
            return (f"moderation fine {amount} sparks per auto-mute "
                    f"(paid to the treasury)."
                    if amount else "moderation fines off.")
        if head == "governance" and rest.isdigit():
            threshold = max(0, int(rest))
            surface.update_settings(
                key, governance_threshold=threshold)
            return (f"governance threshold {threshold} sparks — "
                    "bigger treasury spends need a second admin.")
        if head == "economy" and rest.lower() in ("on", "off"):
            surface.update_settings(key, economy_enabled=on)
            return f"economy {'on' if on else 'off'}."
        if head == "tax" and rest.rstrip("%").isdigit():
            pct = max(0, min(25, int(rest.rstrip("%"))))
            surface.update_settings(key, transfer_tax_pct=pct)
            return f"transfer tax {pct}% to the treasury."
        if head == "exchange":
            # The §25 exchange: pool, rate, spread, skim,
            # allowance — the same settings home as tax/fine.
            cfg = dict(surface.store.settings(key).get(
                "exchange") or {})
            verb = rest.lower()
            if verb in ("on", "off") and not rest2:
                cfg["enabled"] = verb == "on"
                surface.update_settings(key, exchange=cfg)
                return (f"currency exchange "
                        f"{'on' if cfg['enabled'] else 'off'} — "
                        f"members convert games coins ⇄ sparks "
                        f"with /convert.")
            if verb == "pool" and rest2.isdigit():
                cfg["pool_sparks"] = max(0, int(rest2))
                surface.update_settings(key, exchange=cfg)
                if not cfg["pool_sparks"]:
                    return ("exchange pool set to 0 — the "
                            "exchange is closed until an admin "
                            "allocates a weekly pool.")
                return (f"exchange pool "
                        f"{cfg['pool_sparks']} sparks a week.")
            if verb == "rate" and rest2.isdigit():
                cfg["base_rate"] = max(1, int(rest2))
                surface.update_settings(key, exchange=cfg)
                return (f"exchange base rate "
                        f"{cfg['base_rate']} sparks per 100 coins "
                        f"(the live rate floats 60–140% of it "
                        f"with the pool).")
            if verb == "spread" and rest2.rstrip("%").isdigit():
                pct = max(0, min(90, int(rest2.rstrip("%"))))
                cfg["spread_bps"] = pct * 100
                surface.update_settings(key, exchange=cfg)
                return (f"reverse-door spread {pct}% — a round "
                        f"trip always loses it.")
            if verb == "skim" and rest2.rstrip("%").isdigit():
                pct = max(0, min(10, int(rest2.rstrip("%"))))
                cfg["skim_pct"] = pct
                surface.update_settings(key, exchange=cfg)
                return (f"territory skim {pct}% on conversions "
                        f"while a gang holds power "
                        f"(0 = no skim).") if pct else \
                    "territory skim off."
            if verb == "allowance" and rest2.isdigit():
                cfg["allowance_base"] = max(0, int(rest2))
                surface.update_settings(key, exchange=cfg)
                return (f"base conversion allowance "
                        f"{cfg['allowance_base']} sparks a week "
                        f"(level, mastery and gang rank grow it).")
            return ("usage: /gsettings exchange on|off · "
                    "exchange pool <sparks a week> · exchange "
                    "rate <sparks per 100 coins> · exchange "
                    "spread <pct> · exchange skim <pct> · "
                    "exchange allowance <sparks>")
        if head == "welcome" and rest.lower() in ("on", "off"):
            surface.update_settings(key, welcome_enabled=on)
            return f"welcome {'on' if on else 'off'}."
        if head == "goodbye" and rest.lower() in ("on", "off"):
            surface.update_settings(key, goodbye_enabled=on)
            return f"goodbye {'on' if on else 'off'}."
        if head == "clean" and rest.lower() in ("on", "off"):
            surface.update_settings(key, clean_service=on)
            return f"service cleanup {'on' if on else 'off'}."
        if head == "rules" and rest.lower() == "add" and rest2:
            current = surface.store.settings(key)
            rules = list(current.get("rules") or []) + [rest2]
            surface.update_settings(key, rules=rules)
            return f"rule added ({len(rules)} total)."
        if head == "rules" and rest.lower() == "clear":
            surface.update_settings(key, rules=[])
            return "rules cleared."
        if head == "flood" and rest.isdigit():
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            mod["flood_max"] = max(2, int(rest))
            surface.update_settings(key, moderation=mod)
            return f"flood limit {mod['flood_max']} per window."
        if head == "filter" and rest.lower() == "add" and rest2:
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            filters = list(mod.get("word_filters") or []) + [rest2]
            mod["word_filters"] = filters
            surface.update_settings(key, moderation=mod)
            return f"filter added ({len(filters)} active)."
        if head == "filter" and rest.lower() == "remove" and rest2:
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            filters = [f for f in (mod.get("word_filters") or [])
                       if f != rest2]
            mod["word_filters"] = filters
            surface.update_settings(key, moderation=mod)
            return f"filter list now has {len(filters)} entries."
        if head == "filter" and rest.lower() == "clear":
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            mod["word_filters"] = []
            surface.update_settings(key, moderation=mod)
            return "filters cleared."
        if head == "filter" and rest.lower() == "action" \
                and rest2.lower() in ("delete", "warn", "mute"):
            if rest2.lower() == "mute" \
                    and key.partition(":")[0].strip().lower() \
                    in ("whatsapp", "wa"):
                # Platform truth (owner ruling 2026-11-04, the
                # ladder rule extended): the mute action cannot
                # be newly selected where no per-person mute
                # exists to fire it. A value stored earlier
                # keeps its data; its execution refuses honestly.
                return ("that can't be set here: WhatsApp has "
                        "no per-person mute for the filter to "
                        "fire — delete and warn are the filter "
                        "actions this platform carries out. "
                        "Nothing changed.")
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            mod["filter_action"] = rest2.lower()
            surface.update_settings(key, moderation=mod)
            return f"filter action: {rest2.lower()}."
        if head == "links":
            current = surface.store.settings(key)
            mod = dict(current["moderation"])
            if rest.lower() == "off" or not rest:
                mod["link_allowlist"] = []
                surface.update_settings(key, moderation=mod)
                return "links open (no allowlist)."
            domains = [d.strip().lower() for d in
                       (rest + " " + rest2).replace(",", " ").split()
                       if d.strip()]
            mod["link_allowlist"] = domains
            surface.update_settings(key, moderation=mod)
            return f"link allowlist: {', '.join(domains)}."
        # ── defense grammar (DESIGN §17) ────────────────────────
        if head == "autonomy" and rest.lower() in ("on", "off"):
            # The master switch outranks ordinary admin: WhatsApp
            # any group admin; Telegram the group OWNER only.
            if not self._group_owner_ok(message):
                return ("autonomy is the owner's switch — on "
                        "Telegram only the group owner can flip "
                        "it (WhatsApp: any group admin). Admins "
                        "keep every manual command either way.")
            self._defense_update(surface, key, "_root",
                                 autonomy=on)
            return (f"defense autonomy {'on' if on else 'off'}."
                    + ("" if on else
                       " Sensors still watch and log to /glog — "
                       "nothing is enforced or announced."))
        if head in ("anticrash", "antichannel", "antitag",
                    "antidelete", "antiedit"):
            if rest.lower() in ("on", "off"):
                self._defense_update(surface, key, head,
                                     enabled=on)
                return f"{head} {'on' if on else 'off'}."
            if head == "anticrash" and rest.lower() in (
                    "delete", "flag"):
                self._defense_update(surface, key, head,
                                     action=rest.lower())
                return (f"anticrash action: {rest.lower()} "
                        f"quarantined payloads.")
            if head == "antitag" and rest.isdigit():
                ceiling = max(1, int(rest))
                self._defense_update(surface, key, head,
                                     max_mentions=ceiling)
                return f"antitag ceiling: {ceiling} mentions."
            if head in ("antidelete", "antiedit") \
                    and rest.lower() == "mode" \
                    and rest2.lower() in ("log", "callout"):
                self._defense_update(surface, key, head,
                                     mode=rest2.lower())
                return f"{head} mode: {rest2.lower()}."
            if head in ("antidelete", "antiedit") \
                    and rest.lower() == "to" \
                    and rest2.lower() in ("group", "admins"):
                self._defense_update(surface, key, head,
                                     callout_to=rest2.lower())
                return f"{head} callouts go to: {rest2.lower()}."
        # ── presentation grammar (DESIGN §27) ───────────────────
        if head == "theme" and rest.lower() in (
                "house", "clean", "board"):
            self._presentation_update(surface, key,
                                      theme=rest.lower())
            return (f"presentation theme: {rest.lower()} — the "
                    "whole group's surfaces follow it.")
        if head == "class" and rest.lower() in (
                "economy", "moderation", "game", "briefing",
                "announcement", "leaderboard", "settings",
                "chat") and rest2.lower() in (
                "clean", "board", "default"):
            current = surface.store.settings(key)
            classes = dict(
                (current.get("presentation") or {}).get(
                    "classes") or {})
            if rest2.lower() == "default":
                classes.pop(rest.lower(), None)
                self._presentation_update(surface, key,
                                          classes=classes)
                return (f"{rest.lower()} surfaces follow the "
                        "theme again.")
            classes[rest.lower()] = rest2.lower()
            self._presentation_update(surface, key,
                                      classes=classes)
            return (f"{rest.lower()} surfaces render "
                    f"{rest2.lower()} in this group.")
        if head == "stamp" and rest.lower() in ("on", "off"):
            self._presentation_update(
                surface, key, stamp=(rest.lower() == "on"))
            return (f"footer stamp {'on' if rest.lower() == 'on' else 'off'}"
                    " for presented panels.")
        if head == "stamp" and rest.lower() == "text":
            if rest2.lower() in ("", "default", "clear"):
                self._presentation_update(surface, key,
                                          stamp_text="")
                return "stamp text back to the house default."
            self._presentation_update(surface, key,
                                      stamp_text=rest2)
            return f"stamp text: “{rest2}”."
        if head == "stamp" and rest.lower() == "classes":
            if rest2.lower() in ("", "default", "clear"):
                self._presentation_update(surface, key,
                                          stamp_classes=[])
                return ("stamp classes back to the presented "
                        "defaults.")
            wanted = [c.strip().lower() for c in
                      rest2.replace(" ", ",").split(",")
                      if c.strip()]
            from ...social.chat.typography import SURFACE_CLASSES

            wanted = [c for c in wanted if c in SURFACE_CLASSES]
            if not wanted:
                return ("name at least one surface class: "
                        "economy, game, briefing, announcement, "
                        "leaderboard, moderation, settings, chat.")
            self._presentation_update(surface, key,
                                      stamp_classes=wanted)
            return f"stamp shows on: {', '.join(wanted)}."
        if head == "watermark" and rest.lower() in (
                "off", "text", "media", "both"):
            self._presentation_update(surface, key,
                                      watermark=rest.lower())
            return (f"media watermark: {rest.lower()} — §19's "
                    "modes on the one stamp: off = no media "
                    "mark · text = the signature rides captions "
                    "· media = generated media carries the "
                    "image stamp · both = both.")
        # ── escalation grammar (DESIGN §30) ────────────────────
        if head == "escalation":
            reply = self._apply_escalation_grammar(
                surface, key, text.split()[1:])
            if reply is not None:
                return reply
            if key.partition(":")[0].strip().lower() \
                    in ("whatsapp", "wa"):
                # The step menu offered here matches what the
                # grammar will accept for this group: no mute
                # or restrict step on WhatsApp.
                return ("escalation grammar: limit N · final "
                        "kick|ban · window DAYS · at N fine "
                        "AMOUNT · at N kick · clear N · reset "
                        "— no mute or restrict step on "
                        "WhatsApp: there's no per-person mute "
                        "for the ladder to fire. /gsettings "
                        "shows the ladder.")
            return ("escalation grammar: limit N · final "
                    "kick|ban · window DAYS · at N mute [10m] · "
                    "at N fine AMOUNT · at N restrict|kick · "
                    "clear N · reset — /gsettings shows the "
                    "ladder.")
        # ── estate grammar (§17/§20 levers) ─────────────────────
        if head == "estate":
            reply = self._apply_estate_grammar(
                surface, key, text.split()[1:])
            if reply is not None:
                return reply
            return ("estate grammar: cut PCT · sellback PCT · "
                    "cap N · ltv PCT · rate PCT — /gsettings "
                    "shows the levers.")
        # ── acceptance grammar (DESIGN §28) ─────────────────────
        if head == "acceptance":
            sub = rest.lower()
            if sub == "mode" and rest2.lower() in (
                    "open", "request", "invite"):
                self._acceptance_update(surface, key,
                                         mode=rest2.lower())
                return (f"join mode: {rest2.lower()} — "
                        + {"open": "anyone can walk in; the "
                                   "criteria judge them at the "
                                   "door.",
                           "request": "approval-mode invite "
                                      "links produce requests "
                                      "the rules judge first.",
                           "invite": "joins arrive by invite; "
                                     "the criteria still judge "
                                     "them on arrival."}
                        [rest2.lower()])
            if sub in ("username", "photo", "firstname",
                       "lastname", "hygiene", "banned", "bots") \
                    and rest2.lower() in ("on", "off"):
                current = surface.store.settings(key)
                crit = dict((current.get("acceptance") or {})
                            .get("criteria") or {})
                crit[sub] = (rest2.lower() == "on")
                self._acceptance_update(surface, key,
                                        criteria=crit)
                return (f"acceptance criterion {sub}: "
                        f"{rest2.lower()}.")
            if sub == "action" and rest2.lower() in (
                    "decline", "restricted", "kick"):
                self._acceptance_update(surface, key,
                                         action=rest2.lower())
                explain = {
                    "decline": "the request is refused / the "
                               "join is removed, reason on the "
                               "record",
                    "restricted": "they come in muted with a "
                                  "fix-it note and a "
                                  "check-me-again button",
                    "kick": "they come in muted for the grace "
                            "window, then are removed if still "
                            "unfixed"}[rest2.lower()]
                if key.partition(":")[0].strip().lower() in (
                        "whatsapp", "wa"):
                    # Platform truth (join-queue build):
                    # WhatsApp has no restricted state and no
                    # per-person mute — a restricted/kick
                    # verdict on a request-mode group QUEUES
                    # for a human instead, and the decision
                    # set there is approve or decline.  The
                    # stored action is unchanged; what it
                    # DOES on this platform is what the
                    # explainer must say.
                    explain = {
                        "decline": "their join request is "
                                   "refused, reason on the "
                                   "record",
                        "restricted": "WhatsApp admits fully "
                                      "or not at all — the "
                                      "request queues for a "
                                      "human to approve or "
                                      "decline; there is no "
                                      "restricted state to "
                                      "admit them into",
                        "kick": "WhatsApp admits fully or "
                                "not at all — the request "
                                "queues for a human to approve "
                                "or decline; there is no "
                                "muted grace window here"
                        }[rest2.lower()]
                return (f"a failed check now means: "
                        f"{rest2.lower()} ({explain}).")
            if sub == "grace" and rest2.rstrip("m").isdigit():
                minutes = max(1, min(10080,
                                     int(rest2.rstrip("m"))))
                self._acceptance_update(surface, key,
                                         grace_minutes=minutes)
                return (f"acceptance grace: {minutes} minutes "
                        f"before a kick-action escalates.")
            if sub == "auto" and rest2.lower() in ("on", "off"):
                self._acceptance_update(
                    surface, key,
                    auto_approve=(rest2.lower() == "on"))
                return (f"auto-approve of clean requests: "
                        f"{rest2.lower()}.")
            if sub == "override" and rest2:
                bits = rest2.split(None, 1)
                crit_name = bits[0].lower()
                act = bits[1].lower() if len(bits) > 1 else ""
                if crit_name in ("username", "photo", "firstname",
                                 "lastname", "hygiene", "banned",
                                 "bots") and act in (
                        "decline", "restricted", "kick",
                        "default"):
                    current = surface.store.settings(key)
                    overrides = dict(
                        (current.get("acceptance") or {})
                        .get("overrides") or {})
                    if act == "default":
                        overrides.pop(crit_name, None)
                    else:
                        overrides[crit_name] = act
                    self._acceptance_update(surface, key,
                                            overrides=overrides)
                    return (f"{crit_name} failures now take: "
                            f"{act if act != 'default' else 'the group action'}.")
                return ("usage: /gsettings acceptance override "
                        "username|photo|firstname|lastname|hygiene"
                        "|banned|bots decline|restricted|kick"
                        "|default")
            if sub == "whitelist" and rest2:
                bits = rest2.split(None, 1)
                verb = bits[0].lower()
                token = bits[1].strip() if len(bits) > 1 else ""
                if verb in ("add", "remove") and token:
                    current = surface.store.settings(key)
                    wl = [str(w) for w in (
                        (current.get("acceptance") or {})
                        .get("bot_whitelist") or [])]
                    if verb == "add" and token not in wl:
                        wl.append(token)
                    if verb == "remove":
                        wl = [w for w in wl if w != token]
                    self._acceptance_update(surface, key,
                                            bot_whitelist=wl)
                    return (f"bot whitelist: "
                            f"{', '.join(wl) or 'empty'}.")
                return ("usage: /gsettings acceptance whitelist "
                        "add|remove @name|id")
            return ("acceptance grammar: mode, the criterion "
                    "names, action, grace, auto, override, "
                    "whitelist — /gsettings shows the block.")
        return ("I didn't follow that — /gsettings alone shows the "
                "current settings and the change grammar.")

    # ---- escalation + estate settings (DESIGN §30 / §20) -------
    def _escalation_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the escalation settings block (the §30 ladder)
        through the store's own settings machinery."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("escalation") or {})
        block.update(fields)
        surface.update_settings(key, escalation=block)
        return block

    def _estate_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the estate settings block (the §17/§20 levers)
        through the store's own settings machinery — the grammar
        the estate build left to this console slice."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("estate") or {})
        block.update(fields)
        surface.update_settings(key, estate=block)
        return block

    def _apply_escalation_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The escalation change grammar, shared verbatim by
        /gsettings and the DM console: ``limit N`` · ``final
        kick|ban`` · ``window DAYS`` · ``at N mute [10m|2h]`` ·
        ``at N fine AMOUNT`` · ``at N restrict|kick`` ·
        ``clear N`` · ``reset``.  Returns the reply line, or
        None when the words don't parse."""
        from ...social.chat.group_defense import (
            ESCALATION_DEFAULTS, EscalationConfig)

        words = [str(w) for w in words]
        if not words:
            return None
        esc = surface.escalation_config(key)
        # WhatsApp groups cannot take a mute or restrict step:
        # both fire admin_restrict at warn time, and WhatsApp
        # has no per-person mute to fire (owner ruling
        # 2026-11-04).  This gates NEW selection only — a
        # ladder already stored with such a step keeps its
        # data; its execution refuses honestly, as before.
        on_whatsapp = str(key).partition(":")[0].strip().lower() \
            in ("whatsapp", "wa")

        def ladder() -> str:
            return ("ladder now: "
                    f"{surface.escalation_config(key).describe()}.")

        head = words[0].lower()
        if head == "limit" and len(words) == 2 \
                and words[1].isdigit():
            self._escalation_update(
                surface, key,
                warn_limit=max(1, int(words[1])))
            return (f"warn limit {max(1, int(words[1]))} — the "
                    "final action lands there. " + ladder())
        if head == "final" and len(words) == 2 \
                and words[1].lower() in ("kick", "ban"):
            self._escalation_update(surface, key,
                                    final_action=words[1].lower())
            return (f"final action: {words[1].lower()} at the "
                    f"limit. {ladder()}")
        if head == "window" and len(words) == 2:
            digits = words[1].lower().rstrip("d")
            if digits.isdigit():
                self._escalation_update(
                    surface, key,
                    window_days=max(1, int(digits)))
                return (f"warns expire after "
                        f"{max(1, int(digits))} days. {ladder()}")
            return None
        if head == "reset" and len(words) == 1:
            import copy as _copy

            fields = _copy.deepcopy(ESCALATION_DEFAULTS)
            if on_whatsapp:
                # The house ladder's mute step cannot be newly
                # installed on WhatsApp (see above): the reset
                # lands the same ladder minus its
                # restrict-bottomed steps, and ladder() below
                # shows exactly what was installed.
                fields["intervals"] = [
                    s for s in fields["intervals"]
                    if s.get("action") not in ("mute", "restrict")]
            self._escalation_update(surface, key, **fields)
            return ("escalation reset to the house ladder. "
                    f"{ladder()}")
        if head == "clear" and len(words) == 2 \
                and words[1].isdigit():
            at = int(words[1])
            steps = [s for s in esc.intervals if s["at"] != at]
            if len(steps) == len(esc.intervals):
                return f"no designed step sits at {at}."
            self._escalation_update(surface, key, intervals=steps)
            return f"step at {at} removed. {ladder()}"
        if head == "at" and len(words) >= 3 and words[1].isdigit():
            at = int(words[1])
            action = words[2].lower()
            if on_whatsapp and action in ("mute", "restrict"):
                return ("that step can't be designed here: "
                        "WhatsApp has no per-person mute for "
                        "the ladder to fire — fine and kick "
                        "are the steps this platform carries "
                        "out.")
            steps = [s for s in esc.intervals if s["at"] != at]
            if action == "mute":
                minutes = 60
                if len(words) >= 4:
                    secs, _rest = _parse_duration(words[3])
                    if not secs:
                        return None
                    minutes = max(1, secs // 60)
                steps.append({"at": at, "action": "mute",
                              "minutes": minutes})
            elif action == "fine":
                if len(words) < 4 or not words[3].isdigit():
                    return None
                steps.append({"at": at, "action": "fine",
                              "amount": max(1, int(words[3]))})
            elif action in ("restrict", "kick"):
                steps.append({"at": at, "action": action})
            else:
                return None
            steps.sort(key=lambda s: s["at"])
            self._escalation_update(surface, key, intervals=steps)
            return (f"step designed: {action} at {at}. "
                    f"{ladder()}")
        return None

    def _apply_estate_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The estate lever grammar, shared by /gsettings and
        the DM console: ``cut PCT`` · ``sellback PCT`` · ``cap
        N`` · ``ltv PCT`` · ``rate PCT``."""
        words = [str(w) for w in words]
        if len(words) != 2:
            return None
        field = {"cut": "market_cut_pct",
                 "sellback": "sellback_pct",
                 "cap": "ownership_cap",
                 "ltv": "ltv_pct",
                 "rate": "mortgage_rate_pct"}.get(words[0].lower())
        digits = words[1].rstrip("%")
        if field is None or not digits.isdigit():
            return None
        value = int(digits)
        if field == "ownership_cap":
            value = max(1, value)
        else:
            value = max(0, min(100, value))
        self._estate_update(surface, key, **{field: value})
        return f"estate {words[0].lower()}: {value}."

    # ---- boards: grammar, provider, the /boards command (§14) --
    def _boards_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the boards settings block through the store's
        own settings machinery — the one write path behind both
        /gsettings and the console's boards section."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("boards") or {})
        block.update(fields)
        surface.update_settings(key, boards=block)
        return block

    def _board_spec_ok(self, spec) -> bool:  # type: ignore[no-untyped-def]
        """A publish spec is a bare group metric (sparks, gems,
        messages, contributions, gangs) or a full board id the
        §14 organ can parse."""
        from ...social.chat.group_boards import (
            GROUP_METRICS, parse_board_id)

        spec = str(spec or "").strip().lower()
        if not spec:
            return False
        if ":" not in spec:
            return spec in GROUP_METRICS
        return parse_board_id(spec) is not None

    def _apply_boards_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The boards grammar, shared by /gsettings (words
        after the `boards` head) and the console's boards
        section.  Forms: `daily <spec...>` · `weekly <spec...>`
        · `daily off` · `weekly off` · `hour N` · `off`."""
        w = [str(x) for x in words or []]
        low = [x.lower() for x in w]
        if not w:
            return None
        if low[0] in ("daily", "weekly"):
            specs = low[1:]
            if specs == ["off"]:
                self._boards_update(surface, key, **{low[0]: []})
                return f"{low[0]} board publishing off."
            if not specs:
                return None
            bad = [s for s in specs if not self._board_spec_ok(s)]
            if bad:
                return (f"no such board: {', '.join(bad)} — "
                        "group boards are sparks, gems, "
                        "messages, contributions, gangs; wider "
                        "boards take full ids like "
                        "global:sparks or game:chess:elo.")
            self._boards_update(surface, key, **{low[0]: specs})
            return (f"{low[0]} boards: {', '.join(specs)} — "
                    "posted on the cycle at the publish hour, "
                    "with movement against the last publish.")
        if low[0] == "hour" and len(w) == 2 and w[1].isdigit():
            hour = max(0, min(23, int(w[1])))
            self._boards_update(surface, key, hour=hour)
            return f"board publish hour: {hour}:00 group time."
        if low == ["off"]:
            self._boards_update(surface, key, daily=[],
                                weekly=[])
            return "board publishing off — /boards still reads."
        return None

    def _board_games_rows(self):  # type: ignore[no-untyped-def]
        """The games provider for §14 boards: profile rows from
        the games spine (the same PlayerStore the /game boards
        read), enriched per game with the Glicko ordinal — the
        rating the matchmaker itself pairs by — and mastery
        points from the achievements stats.  None when this
        process carries no games engine: game boards then say
        so instead of inventing standings."""
        engine_fn = getattr(self, "_game_engine", None)
        if engine_fn is None:
            return None
        try:
            engine = engine_fn()
        except Exception:  # noqa: BLE001 - no spine, no rows
            return None
        store = getattr(engine, "store", None)
        if store is None:
            return None
        db = getattr(engine, "db", None)
        if db is not None:
            from ...games.mastery import (
                get_one_game_stats, mastery_points)
            from ...games.matchmaking import get_glicko
        from ...games.progression import level_for_xp

        rows = []
        for prof in store.all(limit=500):
            per_game: dict = {}
            mastery_total = 0
            for game, stats in (prof.per_game or {}).items():
                entry = {
                    "wins": int(stats.get("wins") or 0),
                    "points": int(stats.get("points") or 0),
                    "played": int(stats.get("played") or 0),
                    "elo": 0, "elo_games": 0}
                if db is not None:
                    glicko = get_glicko(db, prof.key, game)
                    entry["elo"] = int(round(float(
                        glicko.get("ordinal") or 0)))
                    entry["elo_games"] = int(
                        glicko.get("games") or 0)
                    mastery_total += mastery_points(
                        game, get_one_game_stats(
                            db, prof.key, game))
                per_game[game] = entry
            rows.append({
                "player_key": prof.key,
                "name": prof.name or prof.key,
                "platform": prof.platform or "",
                "coins": int(prof.coins or 0),
                "xp": int(prof.xp or 0),
                "level": level_for_xp(int(prof.xp or 0)),
                "wins": int(prof.wins or 0),
                "mastery": mastery_total,
                "per_game": per_game})
        return rows

    def _control_boards(self, tail, message):  # type: ignore[no-untyped-def]
        """/boards (§14): the board index, or one board by
        name.  Bare metric names read THIS group's board; full
        ids (global:sparks, platform:<platform>:<metric>,
        community:<id>:<metric>, game:<game>:elo, games:xp)
        read those boards.  Rendering is a PEEK — movement
        reads the last published snapshot and a peek never
        advances it; the group's scheduled publishes are what
        move the baseline.  /top is untouched: it IS the group
        sparks board."""
        surface = self._group_surface()
        key = self._group_key_or_none(message)
        name = str(tail or "").strip().lower()
        if not name:
            lines = ["🏆 <b>boards</b> — /boards &lt;name&gt; "
                     "reads one:"]
            if key:
                lines.append(
                    "this group: sparks · gems · messages · "
                    "contributions · gangs")
            lines.append(
                "wider: global:sparks · global:messages · "
                "global:contributions · "
                "platform:&lt;platform&gt;:&lt;metric&gt; · "
                "community:&lt;id&gt;:&lt;metric&gt;")
            lines.append(
                "games: games:xp · games:coins · games:wins · "
                "games:mastery · game:&lt;game&gt;:elo · "
                "game:&lt;game&gt;:wins · "
                "game:&lt;game&gt;:points")
            if key:
                cfg = surface.store.settings(key).get(
                    "boards") or {}
                lines.append(
                    f"this group publishes daily "
                    f"[{', '.join(cfg.get('daily') or []) or '—'}]"
                    f" · weekly "
                    f"[{', '.join(cfg.get('weekly') or []) or '—'}]"
                    f" at {int(cfg.get('hour', 20))}:00 — "
                    "/gsettings boards changes it.")
            return "\n".join(lines)
        board_id = name if ":" in name else f"group:{name}"
        text = surface.render_board(
            board_id, group_key=key or "",
            games_rows=self._board_games_rows())
        if text is None:
            return (f"no board called {name!r} — /boards lists "
                    "what exists (a games board also needs the "
                    "games spine in this process).")
        return text

    # ---- webshot (§18/§20): the capture surface --------------
    def _webshot_vision(self, data: bytes) -> str:
        """The §20c annotated shot: the vision organ reads the
        image.  Injectable in tests via ``_webshot_vision_fn``;
        an unavailable seer is an honest line, not a fake
        description."""
        fn = getattr(self, "_webshot_vision_fn", None)
        if callable(fn):
            return str(fn(data) or "")
        from ...vision.seer import see_bytes

        return str(see_bytes(
            data,
            "Describe this web page screenshot: what is here, "
            "and what looks broken, changed or notable?") or "")

    def _webshot_deliver(self, message, data: bytes, *,
                         caption: str, pdf: bool) -> str:
        """Write the shot beside the social data and send it
        as a file (photo for images — the gateway renders by
        kind) with its caption.  Returns the path sent."""
        import time as _time
        from pathlib import Path

        context = getattr(self, "context", None)
        settings = getattr(context, "settings", None)
        ws_dir = getattr(settings, "workspace_dir", ".")
        out_dir = Path(ws_dir) / "data" / "social" / "webshot_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".pdf" if pdf else ".png"
        path = out_dir / f"shot-{int(_time.time() * 1000)}{suffix}"
        path.write_bytes(data)
        self.gateway.send_file(
            message.chat.platform, str(message.chat.key),
            str(path), caption=caption)
        return str(path)

    def _control_webshot(self, tail, message):  # type: ignore[no-untyped-def]
        """/webshot (§18 + §20): render a page and return the
        shot as a photo with the URL + page title as its
        caption (Devon's own style).  One command, the whole
        variant space: full/viewport, phone|tablet|desktop,
        dark, idle-wait, element clip, PDF.  In a group this is
        system output — §33's claim gates it, at send time,
        like every other system post; in DM it is free.  A
        replied-to message's first link is the target when the
        command carries none (the resolver's reply discipline).
        ``watch`` rides the same command: page-watch re-shots
        on the cycle and alerts only on a real change."""
        from ...social.chat import webshot as shot_organ
        from ...social.chat.typography import resolve_media_mark

        text = str(tail or "").strip()
        words = text.split()
        key = self._group_key_or_none(message)
        surface = self._group_surface() if key else None
        requester = str(message.sender_id or message.sender
                        or "")
        book = self._webshot_book()

        if words and words[0].lower() == "watch":
            return self._webshot_watch(words[1:], message,
                                       book=book, key=key,
                                       surface=surface,
                                       requester=requester)

        reply_url = ""
        meta = message.meta or {}
        if meta.get("reply_to_text"):
            reply_url = shot_organ.extract_url(
                str(meta.get("reply_to_text") or ""))
        try:
            url, opts = shot_organ.parse_shot_tail(
                text, reply_url=reply_url)
        except shot_organ.ShotUsageError as exc:
            return str(exc)
        if key is not None and not self._system_send_ok(
                surface, key, message, "webshot"):
            # §33 before validation on purpose: an unclaimed
            # room gets the gate answer whatever the URL was —
            # no capture work, no DNS behaviour leaking into
            # a room that isn't hers.
            return ("webshot posts are system output — they "
                    "land only in claimed rooms, and this room "
                    "isn't claimed (my bot must be an admin "
                    "here, or the owner designates it).")
        try:
            url = shot_organ.validate_url(
                url,
                allow_private=self._operator_or_false(message))
        except shot_organ.ShotUsageError as exc:
            return str(exc)
        scope = key or str(message.chat.key or "")
        options_line = (
            f"{'full' if opts.full_page else 'viewport'} · "
            f"{opts.device} · {opts.scheme}"
            + (f" · clip {opts.clip}" if opts.clip else "")
            + (" · pdf" if opts.pdf else ""))
        db = getattr(getattr(self, "context", None), "db", None)
        try:
            result = shot_organ.capture(url, opts, db=db)
        except shot_organ.WebshotUnavailable as exc:
            book.record(url=url, ok=False, error=str(exc),
                        scope=scope, requester=requester,
                        options=options_line)
            return (f"📸 couldn't shoot {url} — every rung "
                    f"failed: {exc}")
        book.record(url=url, ok=True, rung=result.rung,
                    size=len(result.data), width=result.width,
                    height=result.height, pdf=result.pdf,
                    scope=scope, requester=requester,
                    options=options_line)
        # §19's media mark, from the one presentation home:
        # the destination group's settings in a group, the
        # house defaults + the owner's mark in DM.
        settings = surface.store.settings(key) if key else None
        registry = self._owner_registry()
        mark = resolve_media_mark(settings, "announcement",
                                  mark_text=registry.mark_text())
        data = result.data
        if mark.image and not result.pdf:
            from ...social.chat.owner_system import \
                apply_media_watermark

            data, applied = apply_media_watermark(
                data, mark.image_text)
            if applied:
                book.record(url=url, ok=True, rung="watermark",
                            size=len(data), scope=scope,
                            requester=requester,
                            options=f"image stamp · {mark.mode}")
        title = result.title or url
        caption = f"🌐 {title}\n{url}"
        if mark.text:
            caption += f"\n{mark.text}"
        try:
            self._webshot_deliver(message, data,
                                  caption=caption,
                                  pdf=result.pdf)
        except Exception as exc:  # noqa: BLE001 - the shot is
            return (f"📸 shot {url} ({len(data)} bytes via "
                    f"{result.rung}) but the send failed: "
                    f"{exc}")
        reply = (f"📸 {title} — {result.width}×{result.height}"
                 if not result.pdf else
                 f"📄 {title} — print PDF of the page")
        reply += f" · {options_line} · via {result.rung}"
        if opts.annotate and not result.pdf:
            try:
                note = self._webshot_vision(data)
            except Exception as exc:  # noqa: BLE001 - vision
                note = ""              # honesty below
                reply += (f"\n👁 vision read unavailable on "
                          f"this host ({type(exc).__name__})")
            if note:
                reply += f"\n👁 {note}"
        return reply

    def _webshot_watch(self, words, message, *, book, key,  # type: ignore[no-untyped-def]
                       surface, requester):
        """/webshot watch <url> [every 2h] · watch list ·
        watch remove <id>.  A watch is a standing system
        output: in a group it is created only where §33
        already lets Devon post, and the driver re-checks the
        claim before every alert."""
        from ...social.chat import webshot as shot_organ

        sub = words[0].lower() if words else ""
        scope = key or str(message.chat.key or "")
        if sub in ("list", ""):
            rows = [w for w in book.watches()
                    if w.get("scope") == scope]
            if not rows:
                return ("no page-watches here — /webshot watch "
                        "<url> [every 2h] starts one.")
            lines = ["👀 <b>page-watches</b>"]
            for w in rows:
                lines.append(
                    f"{w['id']} · {w['url']} · every "
                    f"{int(float(w.get('interval_s') or 0)) // 60}m"
                    f" · last change "
                    f"{_fmt_ts(w.get('last_change')) if w.get('last_change') else '—'}")
            return "\n".join(lines)
        if sub in ("remove", "stop", "drop"):
            if len(words) < 2:
                return "watch remove <id> — ids are on watch list."
            if book.remove_watch(words[1]):
                return f"👀 watch {words[1]} removed."
            return f"no watch called {words[1]!r} here."
        interval = shot_organ.WATCH_DEFAULT_INTERVAL_S
        tail_words = list(words)
        if len(tail_words) >= 2 and tail_words[-2].lower() \
                == "every":
            spec = tail_words[-1].lower()
            tail_words = tail_words[:-2]
            try:
                if spec.endswith("h"):
                    interval = float(spec[:-1]) * 3600
                elif spec.endswith("m"):
                    interval = float(spec[:-1]) * 60
                else:
                    interval = float(spec)
            except ValueError:
                return (f"I can't read {spec!r} as an interval "
                        "— use 30m, 2h, or plain seconds.")
        try:
            url, opts = shot_organ.parse_shot_tail(
                " ".join(tail_words))
            url = shot_organ.validate_url(
                url,
                allow_private=self._operator_or_false(message))
        except shot_organ.ShotUsageError as exc:
            return str(exc)
        if opts.pdf:
            return ("a watch re-shots images — drop the pdf "
                    "option and I'll watch the page as a shot.")
        if key is not None and not self._system_send_ok(
                surface, key, message, "webshot watch"):
            return ("page-watch alerts are system output — "
                    "they post only in claimed rooms, and this "
                    "room isn't claimed.")
        watch = book.add_watch(
            url=url, options=opts, scope=scope,
            requester=requester, interval_s=interval)
        return (f"👀 watching {url} ({watch['id']}) — I "
                "re-shoot it on the cycle and post here only "
                "when the page actually changes.")

    _WATCH_DRIVE_THROTTLE_S = 120.0

    def _drive_webshot_watches(self):  # type: ignore[no-untyped-def]
        """The page-watch beat (§20b): due watches re-shoot on
        the message cycle — no daemon, the channel drive's
        discipline.  Alerts are system output in groups
        (§33 re-checked at send time) and free in the owner's
        DM.  A held (silent-hours) group alert joins the
        morning digest like every other held post."""
        now = time.time()
        last = getattr(self, "_webshot_watch_last", 0.0)
        if now - last < self._WATCH_DRIVE_THROTTLE_S:
            return
        self._webshot_watch_last = now
        book = self._webshot_book()
        db = getattr(getattr(self, "context", None), "db", None)
        try:
            alerts = book.poll_watches(now=now, db=db)
        except Exception:  # noqa: BLE001 - the beat never
            _log.debug("webshot watch poll failed",
                       exc_info=True)
            return
        if not alerts:
            return
        surface = self._group_surface()
        from types import SimpleNamespace

        for alert in alerts:
            watch = alert["watch"]
            result = alert["result"]
            scope = str(watch.get("scope") or "")
            if not scope:
                continue
            caption = (
                f"👀 page changed ({alert['ratio']:.0%} of the "
                f"frame differs)\n{result.title or watch['url']}"
                f"\n{watch['url']}")
            probe = SimpleNamespace(
                chat=SimpleNamespace(
                    platform=scope.split(":", 1)[0], key=scope))
            if scope in surface.store.data:
                if not self._system_send_ok(
                        surface, scope, probe, "webshot watch"):
                    continue
                if self._rhythm_hold(surface, scope,
                                     "webshot watch", caption):
                    continue
            try:
                self._webshot_deliver(probe, result.data,
                                      caption=caption, pdf=False)
            except Exception:  # noqa: BLE001 - one bad alert
                _log.debug("webshot watch alert send failed",
                           exc_info=True)

    # ---- the owner system (§19/§20): DM surface --------------
    def _owner_status_card(self, registry):  # type: ignore[no-untyped-def]
        """Compose the owner's status card from real state
        only: runtime uptime (the console snapshot), the
        package version, the §15 pool counts, live queue
        depths read off the stores, the last briefing stamp."""
        from ...social.chat.owner_system import status_card
        from ...version import __version__

        uptime = None
        snapshot = getattr(self, "console_snapshot", None)
        if callable(snapshot):
            try:
                uptime = snapshot().get("uptime_s")
            except Exception:  # noqa: BLE001
                uptime = None
        accounts = None
        supervisor = getattr(self, "_pairing_supervisor", None)
        if supervisor is not None:
            try:
                accounts = supervisor().registry.counts(
                    probe=False)
            except Exception:  # noqa: BLE001
                accounts = None
        queues: dict[str, int] = {}
        try:
            surface = self._group_surface()
            queued = held = 0
            for entry in surface.store.data.values():
                extras = entry.get("extras") or {}
                queued += len(extras.get("channel_queue")
                              or [])
                held += len(extras.get("rhythm_held") or [])
            queues["channel posts queued"] = queued
            queues["held digest posts"] = held
        except Exception:  # noqa: BLE001 - queues are a
            queues = {}              # sourced field or absent
        try:
            queues["page-watches"] = len(
                self._webshot_book().watches())
        except Exception:  # noqa: BLE001
            pass
        last_briefing = None
        try:
            from ...agents.morning_briefing import \
                _last_briefing_at

            last_briefing = _last_briefing_at(self.context)
        except Exception:  # noqa: BLE001
            last_briefing = None
        return status_card(
            uptime_s=uptime, version=__version__,
            accounts=accounts, queues=queues,
            last_briefing_ts=last_briefing)

    def _control_owner(self, tail, message):  # type: ignore[no-untyped-def]
        """/owner (§19/§20): the owner system in DM — the
        verification card, the status surface (uptime lives
        here, never on /ping), and the record's edits.  The
        organ enforces the two hard rules on every edit: the
        actor must be the owner, and the record is NEVER
        editable from a group.  Sudo is recognition, never
        ownership — a sudo member's /owner meets the same
        refusal as anyone else's."""
        from ...social.chat.owner_system import (
            OwnerAuthorityError, verification_card)

        if not self._operator_or_false(message):
            return ("that's the owner's surface — it answers "
                    "only to the owner, on his own surfaces.")
        registry = self._owner_registry()
        in_group = self._group_key_or_none(message) is not None
        text = str(tail or "").strip()
        words = text.split()
        head = words[0].lower() if words else ""
        actor = str(message.sender_id or message.sender or "")

        if head in ("", "card"):
            accounts = []
            supervisor = getattr(self, "_pairing_supervisor",
                                 None)
            if supervisor is not None:
                try:
                    for rec in supervisor().registry.all():
                        accounts.append(
                            f"{rec.platform} {rec.identity} — "
                            f"{rec.status}")
                except Exception:  # noqa: BLE001 - the card
                    accounts = []      # degrades, never invents
            sources = []
            if getattr(self, "_owner_sender_ids", None):
                sources.append(
                    "telegram sender ids seeded from config "
                    "(NM_PARTNER_OWNER_SENDER_IDS)")
            if accounts:
                sources.append(
                    "whatsapp/telegram identities seeded from "
                    "the §15 pairing registry")
            from ...core.owner import seal_configured

            return verification_card(
                registry, seal_configured=seal_configured(),
                account_lines=accounts, sources=sources)
        if head == "status":
            return self._owner_status_card(registry)
        if head == "audit":
            rows = registry.audit(limit=10)
            if not rows:
                return "the owner record's audit is empty."
            return "\n".join(
                ["👑 <b>owner record — recent edits</b>"]
                + [f"{_fmt_ts(r.get('ts'))} · "
                   f"{r.get('action')} · "
                   f"{r.get('platform') or '—'} · "
                   f"{r.get('target') or '—'}"
                   for r in rows])

        # ── edits: the organ judges (owner + never-in-group) ──
        try:
            if head == "bind" and len(words) >= 3:
                changed = registry.bind(
                    "owner", via_group=in_group,
                    platform=words[1], user_id=words[2],
                    actor=actor)
                return (f"bound {words[1]}:{words[2]} to the "
                        "owner record." if changed else
                        "already bound.")
            if head == "unbind" and len(words) >= 3:
                changed = registry.unbind(
                    "owner", via_group=in_group,
                    platform=words[1], user_id=words[2],
                    actor=actor)
                return (f"unbound {words[1]}:{words[2]}."
                        if changed else "that id wasn't bound.")
            if head == "sudo" and len(words) >= 4:
                on = words[1].lower() == "add"
                if words[1].lower() not in ("add", "remove"):
                    return ("sudo grammar: /owner sudo add|"
                            "remove <platform> <id>")
                changed = registry.set_sudo(
                    "owner", via_group=in_group,
                    platform=words[2], user_id=words[3], on=on,
                    actor=actor)
                return (f"sudo {'added' if on else 'removed'}: "
                        f"{words[2]}:{words[3]}." if changed
                        else "no change — check the id.")
            if head == "deputy" and len(words) >= 4:
                on = words[1].lower() == "add"
                if words[1].lower() not in ("add", "remove"):
                    return ("deputy grammar: /owner deputy "
                            "add|remove <group key> <id>")
                changed = registry.set_deputy(
                    "owner", via_group=in_group,
                    group_key=words[2], user_id=words[3],
                    on=on, actor=actor)
                return (f"deputy {'added' if on else 'removed'}"
                        f": {words[3]} in {words[2]}."
                        if changed else
                        "no change — check the id.")
            if head == "mark":
                mark = text[len(words[0]):].strip()
                if mark.lower() in ("", "clear", "default"):
                    registry.set_mark_text(
                        "owner", via_group=in_group, text="",
                        actor=actor)
                    return ("mark cleared — the house default "
                            "signature applies.")
                registry.set_mark_text(
                    "owner", via_group=in_group, text=mark,
                    actor=actor)
                return f"owner mark: “{mark}” — it rides every platform."
            if head == "rebind" and len(words) >= 3:
                passphrase = text.split(None, 3)[3] \
                    if len(text.split(None, 3)) > 3 else ""
                if not passphrase:
                    return ("rebind grammar: /owner rebind "
                            "<platform> <new id> <seal "
                            "passphrase> — the passphrase is "
                            "verified and never stored.")
                registry.rebind_with_seal(
                    platform=words[1], new_id=words[2],
                    identity_claim=str(
                        message.sender_username or ""),
                    passphrase=passphrase)
                return (f"rebound: {words[1]} now points at "
                        f"{words[2]} — sealed recovery done.")
        except OwnerAuthorityError as exc:
            return str(exc)
        return ("owner grammar: /owner (verification card) · "
                "/owner status · /owner audit · bind|unbind "
                "<platform> <id> · sudo add|remove <platform> "
                "<id> · deputy add|remove <group key> <id> · "
                "mark <text> · rebind <platform> <new id> "
                "<passphrase>")

    def _control_sudo(self, tail, message):  # type: ignore[no-untyped-def]
        """Top-level /sudo (owner order 2026-11-05, register #32).

        ``/sudo`` met total silence in the owner's WhatsApp
        self-chat: the rung existed only as a subcommand of
        ``/owner``, so the bare word never parsed as a command at
        all and the message fell into conversation. This is the
        direct surface: ``add``/``remove`` delegate to the owner
        grammar (one edit path, one held audit); anything else
        answers the rung's state in plain words from the owner
        registry — who holds sudo, or that nobody does.
        """
        words = (tail or "").strip().split()
        head = words[0].lower() if words else ""
        if head in ("add", "remove"):
            if message is None:
                # Console-seam caller: handle_control's message=None
                # contract IS the owner (the same contract /owner's
                # own dispatch relies on — its handler reads the
                # caller as operator exactly then). The delegation
                # runs against a minimal owner stand-in carrying the
                # stamp _is_operator honors, rather than dying on
                # the missing message.
                from types import SimpleNamespace

                from ...social.chat.base import ChatKind, ChatRef
                message = SimpleNamespace(
                    chat=ChatRef(platform="whatsapp", chat_id="owner",
                                 kind=ChatKind.DM),
                    sender_id="", sender="owner",
                    sender_username="",
                    meta={"is_owner": True, "from_owner": True})
            return self._control_owner("sudo " + (tail or "").strip(),
                                       message)
        try:
            from .failures import record_failure
            registry = self._owner_registry()
            by_platform = dict((registry.data or {}).get("sudo") or {})
            grants = [
                f"{platform}:{user_id}"
                for platform, ids in by_platform.items()
                for user_id in (ids or [])]
        except Exception as exc:  # noqa: BLE001 - say it, never silence
            try:
                record_failure(
                    "runtime.PartnerRuntime._control_sudo", exc)
            except Exception as journal_exc:  # noqa: BLE001 - journaling never breaks a reply
                _log.debug("sudo journal failed: %s", journal_exc)
            return ("sudo: I couldn't read the owner record just "
                    "now — /owner shows the same rung on its card.")
        if grants:
            return (f"sudo: {', '.join(grants)} — recognized on "
                    "every surface, exempt from automatic actions, "
                    "never the owner. /sudo remove <platform> <id> "
                    "drops a grant.")
        return ("sudo: none — nobody holds sudo. Sudo holders are "
                "recognized on every surface and exempt from "
                "automatic actions, but never the owner. Grant it "
                "with /sudo add <platform> <id> "
                "(or /owner sudo add <platform> <id>).")

    # ---- channels: settings write path + the drive (§16/P4a) ---
    def _channel_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate a channel's settings block through the
        store's own settings machinery (the console's channel
        sections write through here)."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("channel") or {})
        block.update(fields)
        surface.update_settings(key, channel=block)
        return block

    def _channel_drive(self, key, *, now=None, force=False):  # type: ignore[no-untyped-def]
        """Drive one channel's posting cycle: sample the
        analytics the platform's adapter actually exposes
        (subscriber count, published-post views re-read,
        broadcast stats), then post due queue entries through
        the gateway.  The surface checks the claim before
        anything posts; a platform that exposes a metric is
        sampled on the sample cadence, a platform that exposes
        none is sampled never — absence stays absence, never
        a fabricated 0."""
        from types import SimpleNamespace

        from ...social.chat.base import ChatKind, ChatRef

        surface = self._group_surface()
        if not surface.is_channel(key):
            return []
        now = now if now is not None else time.time()
        platform = str(key).split(":", 1)[0]
        adapter = self._group_adapter(platform)
        cfg = surface.channel_cfg(key)
        # Sampling accrues state, so it follows the claim: a
        # foreign channel is noted in the registry, never
        # series-built (§33).
        if adapter is not None and cfg.get("analytics", True) \
                and surface.claim_state(key)["claimed"]:
            ref = ChatRef.parse(str(key), kind=ChatKind.CHANNEL)
            counter = getattr(adapter, "admin_member_count",
                              None)
            if callable(counter):
                try:
                    res = counter(ref)
                except Exception:  # noqa: BLE001 - a failed
                    res = None              # read is absence
                if isinstance(res, dict) and res.get("ok") \
                        and res.get("count") is not None:
                    surface.note_channel_count(
                        key, int(res["count"]), now=now)
            # Views re-read: the counters live on the published
            # posts and grow after posting, so the beat fetches
            # their CURRENT values for every post in the ring —
            # the send receipt's number was only ever a seed.
            viewer = getattr(adapter, "channel_post_views", None)
            if callable(viewer) \
                    and surface.channel_views_due(key, now=now):
                published = surface.channel_published_ids(key)
                if published:
                    try:
                        res = viewer(ref, published)
                    except Exception:  # noqa: BLE001 - a failed
                        res = None              # read is absence
                    if isinstance(res, dict) and res.get("ok"):
                        surface.note_channel_views_read(
                            key, res.get("views") or {}, now=now)
            # Broadcast stats (Telegram via the userbot): a
            # grant lands as a sample; a refusal lands as the
            # absence-with-reason state — and stamps the beat
            # either way, so a too-small channel is re-asked on
            # the sample cadence, never per turn.
            statser = getattr(adapter, "channel_broadcast_stats",
                              None)
            if callable(statser) \
                    and surface.channel_stats_due(key, now=now):
                try:
                    res = statser(ref)
                except Exception as exc:  # noqa: BLE001
                    res = {"ok": False, "error": str(exc)}
                if isinstance(res, dict) and res.get("ok"):
                    surface.note_channel_stats(key, res, now=now)
                elif isinstance(res, dict):
                    surface.note_channel_stats_unavailable(
                        key, str(res.get("error")
                                 or "broadcast stats refused"),
                        now=now)
            # Discussion linkage (§7): the channel's linked
            # discussion group, read from the platform on the
            # same sample cadence.  A landed answer records
            # (or clears) the linkage; a refusal/miss stamps
            # the beat and changes nothing — unknown is not
            # unlinked.
            linker = getattr(adapter, "channel_discussion_link",
                             None)
            if callable(linker) \
                    and surface.channel_link_due(key, now=now):
                try:
                    res = linker(ref)
                except Exception:  # noqa: BLE001 - a failed
                    res = None              # read is absence
                if isinstance(res, dict) and res.get("ok"):
                    surface.note_channel_discussion(
                        key, str(res.get("linked_chat_id")
                                 or ""), now=now)
                else:
                    surface.note_channel_link_miss(key, now=now)
        gateway = getattr(self, "gateway", None)
        if gateway is None \
                or not callable(getattr(gateway, "send", None)):
            return []

        def _send(text, entry):
            chat = ChatRef.parse(str(key), kind=ChatKind.CHANNEL)
            # Non-text transports ride the ADAPTER (the same
            # split as the analytics reads): a poll entry posts
            # through the adapter's send_poll, a cross-post
            # entry through its native forward.  A surface
            # without the verb fails the entry honestly — it
            # stays due and the failure is journaled, like any
            # other transport failure.
            poll = entry.get("poll") or {}
            if poll:
                fn = getattr(adapter, "send_poll", None) \
                    if adapter is not None else None
                if fn is None:
                    return False, ("this surface cannot post "
                                   "polls")
                try:
                    result = fn(
                        chat,
                        str(poll.get("question") or text),
                        [str(o) for o in
                         (poll.get("options") or [])],
                        anonymous=bool(poll.get(
                            "anonymous", True)),
                        multiple=bool(poll.get("multiple")),
                        quiz=bool(poll.get("quiz")),
                        correct_option=poll.get("correct"),
                        explanation=str(
                            poll.get("explanation") or ""))
                except Exception as exc:  # noqa: BLE001
                    return False, (f"{type(exc).__name__}: "
                                   f"{exc}")
                ok = bool(getattr(result, "ok", False))
                info = str(getattr(result, "message_id", "")
                           or getattr(result, "error", "") or "")
                if ok:
                    views = getattr(result, "views", None)
                    if views is not None:
                        surface.note_channel_views(
                            key, str(getattr(
                                result, "message_id", "") or ""),
                            int(views), now=now)
                return ok, info
            forward = entry.get("forward") or {}
            if forward:
                fn = getattr(adapter, "forward_post", None) \
                    if adapter is not None else None
                if fn is None:
                    return False, ("this surface cannot "
                                   "forward posts")
                src = ChatRef.parse(
                    str(forward.get("from_channel") or ""),
                    kind=ChatKind.CHANNEL)
                try:
                    result = fn(src, str(
                        forward.get("message_id") or ""), chat)
                except Exception as exc:  # noqa: BLE001
                    return False, (f"{type(exc).__name__}: "
                                   f"{exc}")
                ok = bool(getattr(result, "ok", False))
                info = str(getattr(result, "message_id", "")
                           or getattr(result, "error", "") or "")
                return ok, info
            media = entry.get("media") or {}
            ref = str(media.get("ref") or "")
            # _rendered_mode reads a message's chat platform;
            # the channel chat itself is the honest stand-in.
            probe = SimpleNamespace(chat=chat)
            mode = self._rendered_mode(probe, text)
            try:
                if ref and media.get("kind") == "file":
                    send_file = getattr(gateway, "send_file",
                                        None)
                    if send_file is not None:
                        result = send_file(platform, chat, ref,
                                           caption=text)
                    else:
                        result = gateway.send(
                            platform, chat,
                            f"{text}\n\n📎 {ref}",
                            parse_mode=mode)
                else:
                    result = gateway.send(
                        platform, chat, text, parse_mode=mode)
            except Exception as exc:  # noqa: BLE001 - the
                # surface journals the failure and keeps the
                # entry due; transport errors never vanish a post
                return False, f"{type(exc).__name__}: {exc}"
            views = getattr(result, "views", None)
            if views is not None:
                surface.note_channel_views(
                    key, str(getattr(result, "message_id", "")
                             or ""), int(views), now=now)
            ok = bool(getattr(result, "ok", False))
            info = str(getattr(result, "message_id", "")
                       or getattr(result, "error", "") or "")
            return ok, info

        return surface.channel_cycle(key, now=now, send=_send,
                                     force=force)

    def _drive_all_channels(self, surface):  # type: ignore[no-untyped-def]
        """Group turns power the channel cadence: due posts in
        claimed channels ride the same message cycle as every
        other Devon system — no daemon was added for them."""
        for ckey in surface.channel_keys():
            self._channel_drive(ckey)

    def _channel_note_and_drive(self, message):  # type: ignore[no-untyped-def]
        """A message arrived IN a channel: the channel itself
        still gets no ingest and no reply (§16) — but the
        registry notes the sighting, the bot-claim probe runs,
        and the posting cycle takes its beat.  Owner-side
        control plane only; nothing about the message is fed
        to memory or the brain."""
        surface = self._group_surface()
        key = message.chat.key
        surface.note_channel_seen(
            key, title=str(getattr(message.chat, "title", "")
                           or ""))
        self._probe_bot_claim(surface, key, message)
        self._channel_drive(key)

    # ---- the DM admin console (DESIGN §30) ----------------------
    _CONSOLE_TTL_S = 1800
    _CONSOLE_SECTIONS = ("defense", "acceptance", "economy",
                         "estate", "presentation", "welcome",
                         "expression", "reputation", "conflict",
                         "rhythm", "boards", "cleanup", "police",
                         "p5")
    _CONSOLE_CHANNEL_SECTIONS = ("queue", "post", "posts",
                                 "series", "analytics",
                                 "settings")

    def _console_session_map(self) -> dict:  # type: ignore[no-untyped-def]
        sessions = getattr(self, "_console_sessions", None)
        if sessions is None:
            sessions = self._console_sessions = {}
        return sessions

    def _console_member_status(self, key, user_id):  # type: ignore[no-untyped-def]
        """(status, info) for a user in a TARGET group — the
        _member_status logic aimed at the group being configured
        (the console itself runs in a DM; authority must be read
        where the authority lives)."""
        platform = str(key).split(":", 1)[0]
        adapter = self._group_adapter(platform)
        if adapter is None:
            return "", {}
        fn = getattr(adapter, "admin_member", None) or getattr(
            adapter, "admin_member_mtproto", None)
        if fn is None:
            return "", {}
        from ...social.chat.base import ChatRef

        try:
            info = fn(ChatRef.parse(str(key)), str(user_id))
        except Exception:  # noqa: BLE001 - no proof, no standing
            return "", {}
        if not isinstance(info, dict):
            return "", {}
        status = info.get("status") or info.get("role") or ""
        if not status and isinstance(info.get("result"), dict):
            status = info["result"].get("status") or ""
        return str(status).lower(), info

    def _console_is_admin(self, key, user_id) -> bool:  # type: ignore[no-untyped-def]
        status, _info = self._console_member_status(key, user_id)
        return status in ("creator", "administrator", "admin",
                          "owner")

    def _console_owner_ok(self, key, user_id, message) -> bool:  # type: ignore[no-untyped-def]
        """The owner gate against the TARGET group — the
        _group_owner_ok split, relocated: the operator always;
        WhatsApp any admin; Telegram the group's creator/owner
        only; Discord the server owner."""
        if self._operator_or_false(message):
            return True
        platform = str(key).split(":", 1)[0]
        if platform.startswith("whatsapp"):
            return self._console_is_admin(key, user_id)
        status, info = self._console_member_status(key, user_id)
        if platform.startswith("telegram"):
            return status in ("creator", "owner")
        if platform.startswith("discord"):
            return status in ("owner", "creator") \
                or info.get("is_owner") is True
        return False

    def _console_status_lines(self) -> list:  # type: ignore[no-untyped-def]
        """Runtime status for the console home: process uptime
        from the runtime's own boot marker (console_snapshot),
        per-platform connected duration from each adapter's
        run-loop start stamp, and the userbot session state.
        Runtime state only — a value the runtime doesn't track
        is omitted, never invented."""
        lines: list = []
        snap: dict = {}
        fn = getattr(self, "console_snapshot", None)
        if callable(fn):
            try:
                snap = fn() or {}
            except Exception:  # noqa: BLE001 - status is garnish
                snap = {}
        uptime = snap.get("uptime_s")
        if uptime is not None:
            lines.append(f"uptime {_duration_text(int(uptime))} "
                         "(this process)")
        gateway = getattr(self, "gateway", None)
        adapters = getattr(gateway, "adapters", {}) or {}
        now = time.time()
        for name in sorted(adapters):
            stats = getattr(adapters[name], "stats", {}) or {}
            started = float(stats.get("started_at") or 0)
            if started:
                lines.append(
                    f"{name}: connected "
                    f"{_duration_text(max(0, int(now - started)))}")
            else:
                lines.append(f"{name}: not running")
        if "telegram" in adapters:
            stats = getattr(adapters["telegram"], "stats", {}) \
                or {}
            live = bool(float(stats.get("started_at") or 0))
            lines.append("userbot session: "
                         + ("live" if live else "down"))
        return lines

    def _control_console(self, tail, message):  # type: ignore[no-untyped-def]
        """The DM admin console (§30): an admin opens Devon's DM,
        picks a group THEY admin (proved per group through the
        platform's own admin surface — never assumed from the
        DM), and configures its systems in tappable sections.
        Changes apply immediately and journal to the group's
        /glog naming the admin.  Authority is identical to
        in-group: owner-only items stay owner-only, evaluated
        against the target group."""
        from ...social.chat.base import ChatKind

        if message.chat.kind != ChatKind.DM:
            return ("the console lives in my DM — open a chat "
                    "with me there and send /console.")
        surface = self._group_surface()
        sender = str(message.sender_id or message.sender or "")
        words = str(tail or "").split()
        skey = f"{message.chat.platform}:{sender}"
        session = self._console_session_map().get(skey) or {}
        if time.time() - float(session.get("at") or 0) \
                > self._CONSOLE_TTL_S:
            session = {}
        if not words:
            return self._console_home(surface, message, sender,
                                      skey)
        token = words[0].lower()
        key = (session.get("groups") or {}).get(token)
        if key is None and (session.get("channels") or {}).get(
                token):
            return self._console_channel(
                surface, message, sender, session, token, words)
        if key is None:
            return self._console_home(
                surface, message, sender, skey,
                note="that group handle expired — pick again:")
        if len(words) == 1:
            return self._console_group_menu(surface, key, message,
                                            token)
        section = words[1].lower()
        if section not in self._CONSOLE_SECTIONS:
            return self._console_group_menu(surface, key, message,
                                            token)
        if len(words) == 2:
            return self._console_section(surface, key, message,
                                         token, section)
        # The autonomy pre-gate: a refused flip must not journal
        # as a console change, so the owner check (against the
        # TARGET group) runs before anything is applied.
        if section == "defense" and len(words) > 2 \
                and words[2].lower() == "autonomy" \
                and not self._console_owner_ok(key, sender,
                                               message):
            return ("autonomy is the owner's switch — on "
                    "Telegram only the group owner can flip it "
                    "(WhatsApp: any group admin). Nothing was "
                    "changed.")
        reply = self._console_apply(surface, key, message, sender,
                                    section, words[2:])
        if reply is None:
            return (f"I didn't follow that for the {section} "
                    "section — its card shows the grammar.")
        surface._audit(key, "console_change", by=sender,
                       user_id=sender, section=section,
                       change=" ".join(words[2:]))
        surface.store.save()
        return self._console_section(surface, key, message, token,
                                     section, lead=f"✓ {reply}")

    def _console_home(self, surface, message, sender, skey,  # type: ignore[no-untyped-def]
                      note: str = ""):
        from ...social.chat.typography import Card

        all_keys = [k for k in sorted(surface.store.data.keys())
                    if self._operator_or_false(message)
                    or self._console_is_admin(k, sender)]
        keys = [k for k in all_keys if not surface.is_channel(k)]
        channel_keys = [k for k in all_keys
                        if surface.is_channel(k)]
        if not keys and not channel_keys:
            return ("you don't admin any group I serve — the "
                    "console opens for group admins.")
        groups: dict = {}
        rows: list = []
        buttons: list = []
        for i, key in enumerate(keys, 1):
            token = f"g{i}"
            groups[token] = key
            members = surface.store.group(key).get("members") \
                or {}
            claim = surface.claim_state(key)
            claim_txt = (f"system output on ({claim['source']})"
                         if claim["claimed"]
                         else "system output off")
            rows.append(f"{token} — {key} "
                        f"({len(members)} in the ledger) · "
                        f"{claim_txt}")
            buttons.append([(f"⚙️ {token} · {key}",
                             f"console {token}")])
        channels: dict = {}
        channel_rows: list = []
        for i, key in enumerate(channel_keys, 1):
            token = f"c{i}"
            channels[token] = key
            claim = surface.claim_state(key)
            title = surface.channel_title(key) or key
            channel_rows.append(
                f"{token} — {title} · "
                f"{'claimed' if claim['claimed'] else 'unclaimed'}"
                f" · {len(surface.channel_queue(key))} queued"
                f" · {surface.channel_analytics(key)['posts']} "
                "posted")
            buttons.append([(f"📣 {token} · {title}",
                             f"console {token}")])
        self._console_session_map()[skey] = {
            "groups": groups, "channels": channels,
            "at": time.time()}
        card = Card("settings", title="Devon console", icon="🛠️")
        if note:
            card.line(note)
        status = self._console_status_lines()
        if status:
            card.block_lines(status, title="Status")
        if rows:
            card.block_lines(rows, title="Your groups")
        if channel_rows:
            card.block_lines(channel_rows, title="Your channels")
        card.note("tap a group, or type /console g1 — then a "
                  "section: " + ", ".join(self._CONSOLE_SECTIONS)
                  + (" · channels open with their c-handle: "
                     "queue, post, analytics, settings"
                     if channel_rows else ""))
        return self._card(message, card, buttons,
                              session="console", session_edit=True)

    def _console_group_menu(self, surface, key, message, token):  # type: ignore[no-untyped-def]
        from ...social.chat.typography import Card

        esc = surface.escalation_config(key)
        s = surface.store.settings(key)
        card = Card("settings", title=f"Console — {key}",
                    icon="🛠️")
        card.block_lines([
            f"defense autonomy "
            f"{'ON' if (s.get('defense') or {}).get('autonomy', True) else 'OFF'}"
            f" · ladder: {esc.describe()}",
            f"acceptance mode "
            f"{(s.get('acceptance') or {}).get('mode', 'open')}"
            f" · economy "
            f"{'on' if s.get('economy_enabled') else 'off'}"
            f" · governance "
            f"≥{s.get('governance_threshold', 500)}",
            f"welcome "
            f"{'on' if s.get('welcome_enabled') else 'off'}"
            f" · rules {len(s.get('rules') or [])} · presentation "
            f"theme {(s.get('presentation') or {}).get('theme', 'house')}",
            f"expression "
            f"{'on' if (s.get('expression') or {}).get('enabled', True) else 'off'}"
            f" · gifs {(s.get('expression') or {}).get('gifs', 'normal')}",
            f"reputation warnings "
            f"{(s.get('reputation') or {}).get('warning', 'public')}"
            f" · flag proposals waiting "
            f"{len(surface.flag_proposals(key))}",
            f"conflict mode "
            f"{(s.get('conflict') or {}).get('mode', 'off')}"
            f" · system drops "
            f"{(s.get('conflict') or {}).get('drops_per_day', 2)}/day"
            f" · police "
            f"{'on' if (s.get('conflict') or {}).get('police', True) else 'off'}",
            f"rhythm: silent hours "
            f"{'on' if (s.get('rhythm') or {}).get('silent_enabled') else 'off'}"
            f" {(s.get('rhythm') or {}).get('silent_start', '23:00')}"
            f"–{(s.get('rhythm') or {}).get('silent_end', '07:00')}"
            f" · briefing "
            f"{'on' if (s.get('rhythm') or {}).get('briefing_enabled') else 'off'}"
            f" · review "
            f"{((s.get('rhythm') or {}).get('review') or {}).get('mode', 'off')}",
            f"member cleanup: removals "
            f"{'on' if (s.get('cleanup') or {}).get('enabled', True) else 'OFF'}"
            f" · auto-deleted "
            f"{'on' if (s.get('cleanup') or {}).get('auto_deleted', True) else 'OFF'}"
            f" · inactive window "
            f"{int((s.get('cleanup') or {}).get('inactive_days') or 90)}d"
            f" (Telegram groups)",
        ])
        from ...social.chat.expression import buttons_from_spec

        buttons = buttons_from_spec([
            [("defense", "Defense + ladder",
              f"console {token} defense"),
             ("acceptance", "Acceptance",
              f"console {token} acceptance")],
            [("economy", "Economy", f"console {token} economy"),
             ("estate", "Estate", f"console {token} estate")],
            [("presentation", "Presentation",
              f"console {token} presentation"),
             ("expression", "Expression",
              f"console {token} expression")],
            [("welcome", "Welcome + rules",
              f"console {token} welcome"),
             ("rep", "Reputation",
              f"console {token} reputation")],
            [("conflict", "Conflict",
              f"console {token} conflict"),
             ("rhythm", "Rhythm",
              f"console {token} rhythm")],
            [("cleanup", "Cleanup",
              f"console {token} cleanup"),
             ("p5", "P5 layer",
              f"console {token} p5")],
            [("back", "All groups", "console")],
        ], settings=s)
        card.note("tap a section, or type its grammar — every "
                  "change lands in the group's /glog with your "
                  "name on it")
        return self._card(message, card, buttons,
                              session="console", session_edit=True)

    def _console_section(self, surface, key, message, token,  # type: ignore[no-untyped-def]
                         section, lead: str = ""):
        from ...social.chat.typography import (
            DEFAULT_STAMP_TEXT, Card)

        s = surface.store.settings(key)
        card = Card("settings",
                    title=f"{section.title()} — {key}", icon="🛠️")
        if lead:
            card.line(lead)
        buttons: list = []

        from ...social.chat.expression import (
            button_label, buttons_from_spec)

        def _toggle(action: str, label: str, grammar: str,
                    on_now: bool):
            return (button_label(
                        action,
                        f"{label}: {'ON' if on_now else 'OFF'}",
                        settings=s),
                    f"console {token} {section} {grammar} "
                    f"{'off' if on_now else 'on'}")

        if section == "defense":
            d = s.get("defense") or {}
            esc = surface.escalation_config(key)
            # Stored mute settings on WhatsApp are data that
            # cannot fire — the state line says so (owner
            # ruling 2026-11-04, the ladder rule extended).
            mute_truth = ""
            if key.partition(":")[0].strip().lower() \
                    in ("whatsapp", "wa") \
                    and (((s.get("moderation") or {})
                          .get("filter_action") or "") == "mute"
                         or int(s.get("mute_default_s") or 0) > 0):
                mute_truth = (" · stored mute settings can't "
                              "fire here (no per-person mute "
                              "on WhatsApp)")
            card.block_lines([
                f"autonomy "
                f"{'ON' if d.get('autonomy', True) else 'OFF'}"
                " (owner's switch — Telegram: group owner; "
                "WhatsApp: any admin)",
                f"sensors: anticrash "
                f"{'on' if (d.get('anticrash') or {}).get('enabled', True) else 'off'}"
                f" · antichannel "
                f"{'on' if (d.get('antichannel') or {}).get('enabled', True) else 'off'}"
                f" · antitag "
                f"{'on' if (d.get('antitag') or {}).get('enabled', True) else 'off'}"
                f" · antidelete "
                f"{'on' if (d.get('antidelete') or {}).get('enabled', True) else 'off'}"
                f" · antiedit "
                f"{'on' if (d.get('antiedit') or {}).get('enabled', True) else 'off'}",
                f"moderation: flood "
                f"{(s.get('moderation') or {}).get('flood_max', 6)}"
                f" per window · filters "
                f"{len((s.get('moderation') or {}).get('word_filters') or [])}"
                f" · standing mute "
                f"{_duration_text(int(s.get('mute_default_s') or 0)) if s.get('mute_default_s') else 'ask'}"
                f" · mute fine {s.get('fine_amount', 0)}"
                f"{mute_truth}",
                f"ladder: {esc.describe()}",
            ])
            buttons = [
                [_toggle("autonomy", "Autonomy", "autonomy",
                         bool(d.get("autonomy", True))),
                 _toggle("anticrash", "Anticrash", "anticrash",
                         bool((d.get('anticrash') or {})
                              .get('enabled', True)))],
                [_toggle("antichannel", "Antichannel", "antichannel",
                         bool((d.get('antichannel') or {})
                              .get('enabled', True))),
                 _toggle("antitag", "Antitag", "antitag",
                         bool((d.get('antitag') or {})
                              .get('enabled', True)))],
            ]
            # The ladder's step menu (§30): on WhatsApp the
            # mute/restrict steps are not offered — the
            # platform cannot fire them (see
            # _apply_escalation_grammar).
            escalation_steps_offer = (
                "`escalation at N fine AMOUNT` · `escalation "
                "at N kick` · `escalation clear N` · "
                "`escalation reset` — no mute/restrict step "
                "on WhatsApp (no per-person mute)"
                if key.partition(":")[0].strip().lower()
                in ("whatsapp", "wa") else
                "`escalation at N mute [10m]` · `escalation at N "
                "fine AMOUNT` · `escalation at N restrict|kick` "
                "· `escalation clear N` · `escalation reset`")
            # The mute-bottomed settings get the same treatment
            # (owner ruling 2026-11-04): on WhatsApp the offers
            # state the platform truth instead of listing a
            # mute the platform cannot fire.
            _on_wa = key.partition(":")[0].strip().lower() \
                in ("whatsapp", "wa")
            filter_action_offer = (
                "`filter action delete|warn` — no mute action "
                "on WhatsApp (no per-person mute)"
                if _on_wa else
                "`filter action delete|warn|mute`")
            mute_default_offer = (
                "`mute default off` — no standing mute length "
                "on WhatsApp (no per-person mute)"
                if _on_wa else
                "`mute default 2h|off`")
            card.block_lines([
                "`autonomy on|off` · `anticrash on|off|delete|flag`",
                "`antichannel|antitag|antidelete|antiedit on|off`"
                " · `antitag N` · `antidelete mode log|callout`",
                "`flood N` · `links DOMAINS|off` · "
                "`filter add|remove WORD` · `filter clear` · "
                + filter_action_offer,
                "`warns N` · " + mute_default_offer + " · `fine N`",
                "`escalation limit N` · `escalation final "
                "kick|ban` · `escalation window DAYS`",
                escalation_steps_offer,
            ], title="Grammar")
        elif section == "acceptance":
            acc = s.get("acceptance") or {}
            crit = acc.get("criteria") or {}
            on = [n for n in ("username", "photo", "firstname",
                              "lastname", "hygiene", "banned",
                              "bots") if crit.get(n)]
            card.block_lines([
                f"mode {acc.get('mode', 'open')} · action "
                f"{acc.get('action', 'restricted')} · grace "
                f"{int(acc.get('grace_minutes') or 60)}m · "
                f"auto-approve "
                f"{'on' if acc.get('auto_approve', True) else 'off'}",
                "criteria on: "
                + (", ".join(on) if on else "none — door open"),
            ])
            card.block_lines([
                "`mode open|request|invite` · `action "
                "decline|restricted|kick` · `grace MIN`",
                "`username|photo|firstname|lastname|hygiene|"
                "banned|bots on|off` · `auto on|off`",
                "`override CRITERION decline|restricted|kick|"
                "default` · `whitelist add|remove @name|id`",
            ], title="Grammar")
        elif section == "economy":
            card.block_lines([
                f"economy "
                f"{'on' if s.get('economy_enabled') else 'off'}"
                f" · transfer tax "
                f"{s.get('transfer_tax_pct', 2)}% · governance "
                f"threshold {s.get('governance_threshold', 500)}"
                f" · moderation fine {s.get('fine_amount', 0)}",
            ])
            buttons = [[_toggle("economy", "Economy", "economy",
                                bool(s.get("economy_enabled")))]]
            card.block_lines([
                "`economy on|off` · `tax PCT` · `governance N`"
                " · `fine N`",
            ], title="Grammar")
        elif section == "estate":
            est = s.get("estate") or {}
            card.block_lines([
                f"market cut {est.get('market_cut_pct', 5)}% · "
                f"sell-back {est.get('sellback_pct', 70)}% · "
                f"ownership cap {est.get('ownership_cap', 12)}",
                f"mortgage LTV {est.get('ltv_pct', 60)}% · rate "
                f"{est.get('mortgage_rate_pct', 5)}%",
            ])
            card.block_lines([
                "`cut PCT` · `sellback PCT` · `cap N` · "
                "`ltv PCT` · `rate PCT`",
            ], title="Grammar")
        elif section == "presentation":
            pres = s.get("presentation") or {}
            card.block_lines([
                f"theme {pres.get('theme', 'house')} · stamp "
                f"{'ON' if pres.get('stamp', True) else 'OFF'}"
                f" — “{pres.get('stamp_text') or DEFAULT_STAMP_TEXT}”",
            ])
            theme_cycle = {"house": "clean", "clean": "board",
                           "board": "house"}
            theme_now = str(pres.get("theme") or "house")
            buttons = [
                [(button_label("theme", f"Theme: {theme_now}",
                               settings=s),
                  f"console {token} presentation theme "
                  f"{theme_cycle.get(theme_now, 'house')}"),
                 _toggle("stamp", "Stamp", "stamp",
                         bool(pres.get("stamp", True)))],
            ]
            card.block_lines([
                "`theme house|clean|board` · `class game|"
                "economy|… board|clean|default`",
                "`stamp on|off` · `stamp text TEXT|default` · "
                "`stamp classes a,b|default`",
            ], title="Grammar")
        elif section == "welcome":
            card.block_lines([
                f"welcome "
                f"{'on' if s.get('welcome_enabled') else 'off'}"
                f" · goodbye "
                f"{'on' if s.get('goodbye_enabled') else 'off'}"
                f" · service cleanup "
                f"{'on' if s.get('clean_service') else 'off'}"
                f" · rules {len(s.get('rules') or [])}",
            ])
            buttons = [
                [_toggle("welcome", "Welcome", "welcome",
                         bool(s.get("welcome_enabled"))),
                 _toggle("goodbye", "Goodbye", "goodbye",
                         bool(s.get("goodbye_enabled")))],
                [_toggle("cleanup", "Cleanup", "clean",
                         bool(s.get("clean_service")))],
            ]
            card.block_lines([
                "`welcome on|off` · `goodbye on|off` · "
                "`clean on|off`",
                "`rules add TEXT` · `rules clear`",
            ], title="Grammar")
        elif section == "expression":
            expr = s.get("expression") or {}
            lib_line = "library size unknown"
            try:
                svc = getattr(self, "_expression_service", None)
                if svc is not None:
                    lib_line = (f"library {len(svc().library)} "
                                f"items (steals, makes and "
                                f"provider finds — it grows)")
            except Exception:  # noqa: BLE001 - display only
                pass
            card.block_lines([
                f"expression "
                f"{'ON' if expr.get('enabled', True) else 'OFF'}"
                f" · gifs {expr.get('gifs', 'normal')}",
                f"welcome media "
                f"{'on' if expr.get('welcome_media', True) else 'off'}"
                f" · ban theatre "
                f"{'on' if expr.get('ban_theatre', True) else 'off'}"
                f" · {lib_line}",
                f"auto-collect "
                f"{'on' if expr.get('auto_collect', False) else 'off'}"
                f" — stickers landing here join the library "
                f"passively, when on (claimed rooms only)",
                "moments: welcome · quest complete · jackpot · "
                "season podium · duel wins · estate milestones · "
                "bans (theatre) — warnings stay dry, always",
            ])
            gif_cycle = {"off": "low", "low": "normal",
                         "normal": "off"}
            gif_now = str(expr.get("gifs") or "normal")
            buttons = [
                [_toggle("expression", "Expression", "",
                         bool(expr.get("enabled", True))),
                 (button_label("gif", f"GIFs: {gif_now}",
                               settings=s),
                  f"console {token} expression gifs "
                  f"{gif_cycle.get(gif_now, 'normal')}")],
                [_toggle("welcome", "Welcome media", "welcome",
                         bool(expr.get("welcome_media", True))),
                 _toggle("ban", "Ban theatre", "ban",
                         bool(expr.get("ban_theatre", True)))],
                [_toggle("collect", "Auto-collect", "collect",
                         bool(expr.get("auto_collect", False)))],
            ]
            card.block_lines([
                "`on|off` · `gifs off|low|normal`",
                "`welcome on|off` · `ban on|off` · "
                "`collect on|off`",
            ], title="Grammar")
        elif section == "reputation":
            rep = s.get("reputation") or {}
            stats = surface.reputation.stats()
            pending = surface.flag_proposals(key)
            card.block_lines([
                f"join warnings: "
                f"{rep.get('warning', 'public')} — public posts "
                f"the factual ⚠️ notice in-group · admins keeps "
                f"it to the console + /glog · restricted holds "
                f"the joiner until an admin taps Clear · off "
                f"stays silent",
                f"the book: {stats['live']} live records across "
                f"every group I serve ({stats['total']} "
                f"all-time)",
                f"flag proposals waiting here: {len(pending)}"
                + (" — decide them in the group with /flag"
                   if pending else ""),
                "a record never decides admission — the door "
                "rules do; it only warns, and dismissal here "
                "never deletes another group's record",
            ])
            card.block_lines([
                "`public|admins|restricted|off`",
            ], title="Grammar")
        elif section == "conflict":
            con = s.get("conflict") or {}
            health = surface.conflict.health(key)
            card.block_lines([
                f"mode: {con.get('mode', 'off')} — off sleeps "
                f"the whole layer · crime runs geared crime, "
                f"heists, gangs, police and drops · warfare "
                f"adds wars, territory and governance capture",
                f"system drops: {con.get('drops_per_day', 2)}"
                f"/day · war escrow {con.get('war_escrow', 200)}"
                f" · police "
                f"{'on' if con.get('police', True) else 'off'}",
                (f"the street: {health.get('gangs', 0)} gangs "
                 f"· {health.get('wars', 0)} wars · "
                 f"{health.get('territory', 0)} districts held "
                 f"· {health.get('wanted', 0)} wanted · "
                 f"{health.get('open_bounties', 0)} bounties "
                 f"open" if health.get("ok") else
                 "the street: conflict state unreadable — the "
                 "layer is frozen until it's repaired"),
                "police actions journal their evidence (the "
                "bully index's inputs) — /glog shows exactly "
                "why the precinct moved",
            ])
            card.block_lines([
                "`off|crime|warfare` · `drops N` · "
                "`police on|off` · `escrow N`",
            ], title="Grammar")
        elif section == "rhythm":
            from ...social.chat.group_rhythm import (
                format_offset as _fmt_offset)

            rhy = s.get("rhythm") or {}
            queue = surface.contribution_queue(key)
            held = surface._extras(key).get(
                "rhythm_held") or []
            card.block_lines([
                f"silent hours: "
                f"{'on' if rhy.get('silent_enabled') else 'off'}"
                f" · {rhy.get('silent_start', '23:00')}–"
                f"{rhy.get('silent_end', '07:00')} "
                f"{_fmt_offset(int(rhy.get('tz_offset_min') or 0))}"
                f" — system posts hold for one morning digest; "
                f"moderation never holds"
                + (f" · {len(held)} post(s) held right now"
                   if held else ""),
                f"nightshield: "
                f"{'on' if rhy.get('nightshield') else 'off'}"
                f" — the door takes the raid posture at night; "
                f"drop/war events pause",
                f"briefing: "
                f"{'on' if rhy.get('briefing_enabled') else 'off'}"
                f" at {rhy.get('briefing_hour', 8)}:00 group "
                f"time · reminders: "
                f"{len(rhy.get('reminders') or [])} set",
                f"polls: quorum {rhy.get('poll_quorum', 3)}"
                f" · §26 review: "
                f"{(rhy.get('review') or {}).get('mode', 'off')}"
                + (f" · {len(queue)} contribution(s) waiting "
                   f"in the queue" if queue else
                   " · the review queue is empty"),
            ])
            card.block_lines([
                "`on|off` · `silent HH:MM HH:MM` · `tz +1` · "
                "`nightshield on|off`",
                "`briefing on|off` · `briefing hour N` · "
                "`reminders on|off`",
                "`reminder add HH:MM TEXT` · "
                "`reminder rules HH:MM` · `reminder remove N` "
                "· `reminder list`",
                "`quorum N` · `review "
                "off|auto|assisted|manual` · `review queue` · "
                "`review approve N` · `review reject N`",
            ], title="Grammar")
        elif section == "boards":
            brd = s.get("boards") or {}
            card.block_lines([
                f"daily publishes: "
                f"{', '.join(brd.get('daily') or []) or 'none set'}",
                f"weekly publishes: "
                f"{', '.join(brd.get('weekly') or []) or 'none set'}"
                " (Mondays, group time)",
                f"publish hour: {int(brd.get('hour', 20))}:00 "
                "group time — posts ride the message cycle and "
                "hold through silent hours like every other "
                "cycle post",
                "movement (▲▼) reads the last PUBLISHED "
                "snapshot; /boards peeks never move the "
                "baseline",
            ])
            card.block_lines([
                "`daily sparks messages` · "
                "`weekly contributions` · `hour N` · `off`",
                "specs: this group's sparks · gems · messages "
                "· contributions · gangs, or full board ids "
                "(global:sparks, platform:…, game:chess:elo)",
            ], title="Grammar")
        elif section == "cleanup":
            cln = s.get("cleanup") or {}
            card.block_lines([
                f"removals: "
                f"{'on' if cln.get('enabled', True) else 'OFF'}"
                f" — whether /rmdeleted and /rminactive may "
                f"act in this group (the /cleanup scan always "
                f"runs: it removes nothing)",
                f"auto-remove deleted: "
                f"{'on' if cln.get('auto_deleted', True) else 'OFF'}"
                f" — the upkeep cycle removes deleted accounts "
                f"on its own in claimed groups where she holds "
                f"admin power (and only while defense autonomy "
                f"is on); journaled in /glog, never announced",
                f"inactivity window: "
                f"{int(cln.get('inactive_days') or 90)} days — "
                f"members with no activity past it are what "
                f"/rminactive removes, by command only",
                "scope, honestly: activity truth is Devon's "
                "member ledger plus Telegram last-seen from "
                "her userbot session; the Bot API alone has "
                "neither a member list nor last-seen, so on "
                "that surface a scan covers the members she "
                "has records of. Admins, bots and her own "
                "accounts are never removed.",
            ])
            card.block_lines([
                "`on` · `off` · `auto on` · `auto off` · "
                "`days N`",
                "the sweep itself runs in the group: "
                "/cleanup (scan) · /rmdeleted · /rminactive "
                "[days]",
            ], title="Grammar")
        elif section == "p5":
            # The §16 layer's switches, read from the same
            # block the grammar writes.  Telegram rooms only —
            # on WhatsApp the section states the truth.
            p5 = s.get("p5") or {}
            if str(key).partition(":")[0].strip().lower() in (
                    "whatsapp", "wa"):
                card.block_lines([
                    "the §16 P5 layer is Telegram machinery — "
                    "member tags, sender-chat bans, reaction "
                    "wipes, forum topics, command scopes and "
                    "boosts have no WhatsApp underneath them, "
                    "so there is nothing to configure here.",
                ])
            else:
                card.block_lines([
                    f"topic auto-rename: "
                    f"{'on' if p5.get('topics_rename') else 'OFF'}"
                    f" — auto-named forum topics get a proper "
                    f"name from their first message (the bot "
                    f"surface flags them; journaled in /glog)",
                    f"admin command scopes: "
                    f"{'on' if p5.get('member_scopes') else 'OFF'}"
                    f" — the bot publishes this room's admin "
                    f"command menu per member + as a class "
                    f"(setMyCommands, bot surface only)",
                    f"ephemeral answers: "
                    f"{'on' if p5.get('ephemeral') else 'OFF'}"
                    f" — sensitive replies (/balance) go to "
                    f"the asker only: in-group ephemeral on the "
                    f"bot surface, else their DMs",
                    "riding along, no switches: /tag · "
                    "/bansender · /unbansender · /reactwipe · "
                    "/boosters — the acts live in the group; "
                    "their outcomes land in /glog.",
                ])
                card.block_lines([
                    "`topics on|off` · `scopes on|off|sync|clear`"
                    " · `ephemeral on|off`",
                ], title="Grammar")
        elif section == "police":
            # §36: the owner's override card.  OWNER-ONLY — a
            # group admin opening it sees the gate, not the
            # precinct.
            if not self._operator_or_false(message):
                card.block_lines([
                    "the police override is the owner's section "
                    "— the power to bend the precinct (set wanted "
                    "levels, pardon, call off enforcement, aim "
                    "its focus) belongs to the owner alone. "
                    "Admins meet the police as everyone does: "
                    "/police in the group.",
                ])
            else:
                view = surface.console_police_view(key)
                if not view.get("ok"):
                    card.block_lines([
                        "the street is frozen — the wanted "
                        "state is unreadable, so no override "
                        "can land until it's repaired."])
                else:
                    lines = [
                        f"enforcement focus: "
                        f"×{view['focus']:g} — the bully index "
                        "the ladder reads is scaled by it "
                        "(100% = the precinct's own judgment)",
                    ]
                    if view["wanted"]:
                        for row in view["wanted"]:
                            label = surface._member_label(
                                key, row["member"])
                            mark = (f" · owner's mark "
                                    f"{row['mark']}★"
                                    if row.get("mark")
                                    is not None else "")
                            lines.append(
                                f"{label}: {row['stars']}★ · "
                                f"index {row['index']:g} · "
                                f"rung served "
                                f"{row['episode_rung']}{mark}")
                    else:
                        lines.append(
                            "nobody is wanted here right now.")
                    card.block_lines(lines)
                card.block_lines([
                    "`wanted <member>` · `stars <member> "
                    "<0-5>` · `pardon <member>`",
                    "`calloff <member>` · `focus <pct>` · "
                    "`focus off`",
                    "members resolve by id, @username or "
                    "exact name; every override journals in "
                    "the police journal (held — a pardon "
                    "shows only as the release itself)",
                ], title="Grammar")
        buttons += buttons_from_spec(
            [[("back", "Sections", f"console {token}"),
              ("home", "Console home", "console")]], settings=s)
        return self._card(message, card, buttons,
                              session="console", session_edit=True)

    def _console_apply(self, surface, key, message, sender,  # type: ignore[no-untyped-def]
                       section, words):
        """Apply one console grammar line to the TARGET group.
        The words are the section's own grammar — the same words
        /gsettings takes in-group, minus the command head — and
        they land through the same update helpers.  Returns the
        reply line, or None when the words don't parse."""
        words = [str(w) for w in words]
        if not words:
            return None
        head = words[0].lower()
        rest = words[1].lower() if len(words) > 1 else ""
        on = rest == "on"
        if section == "defense":
            if head == "escalation":
                return self._apply_escalation_grammar(
                    surface, key, words[1:])
            if head == "autonomy" and rest in ("on", "off"):
                if not self._console_owner_ok(key, sender,
                                              message):
                    return ("autonomy is the owner's switch — "
                            "on Telegram only the group owner "
                            "can flip it (WhatsApp: any admin).")
                self._defense_update(surface, key, "_root",
                                     autonomy=on)
                return f"defense autonomy {'on' if on else 'off'}."
            if head in ("anticrash", "antichannel", "antitag",
                        "antidelete", "antiedit"):
                if rest in ("on", "off"):
                    self._defense_update(surface, key, head,
                                         enabled=on)
                    return f"{head} {'on' if on else 'off'}."
                if head == "anticrash" and rest in (
                        "delete", "flag"):
                    self._defense_update(surface, key, head,
                                         action=rest)
                    return f"anticrash action: {rest}."
                if head == "antitag" and rest.isdigit():
                    self._defense_update(
                        surface, key, head,
                        max_mentions=max(1, int(rest)))
                    return f"antitag ceiling: {rest} mentions."
                if head in ("antidelete", "antiedit") \
                        and rest == "mode" and len(words) > 2 \
                        and words[2].lower() in ("log", "callout"):
                    self._defense_update(surface, key, head,
                                         mode=words[2].lower())
                    return f"{head} mode: {words[2].lower()}."
                if head in ("antidelete", "antiedit") \
                        and rest == "to" and len(words) > 2 \
                        and words[2].lower() in ("group", "admins"):
                    self._defense_update(
                        surface, key, head,
                        callout_to=words[2].lower())
                    return f"{head} callouts: {words[2].lower()}."
                return None
            if head == "flood" and rest.isdigit():
                current = surface.store.settings(key)
                mod = dict(current["moderation"])
                mod["flood_max"] = max(2, int(rest))
                surface.update_settings(key, moderation=mod)
                return f"flood limit {mod['flood_max']} per window."
            if head == "links":
                current = surface.store.settings(key)
                mod = dict(current["moderation"])
                if rest in ("off", ""):
                    mod["link_allowlist"] = []
                    surface.update_settings(key, moderation=mod)
                    return "links open (no allowlist)."
                domains = [d.strip().lower() for d in
                           " ".join(words[1:]).replace(",", " ")
                           .split() if d.strip()]
                mod["link_allowlist"] = domains
                surface.update_settings(key, moderation=mod)
                return f"link allowlist: {', '.join(domains)}."
            if head == "filter" and len(words) >= 2:
                current = surface.store.settings(key)
                mod = dict(current["moderation"])
                if rest == "add" and len(words) > 2:
                    filters = list(
                        mod.get("word_filters") or []) + [words[2]]
                    mod["word_filters"] = filters
                    surface.update_settings(key, moderation=mod)
                    return f"filter added ({len(filters)} active)."
                if rest == "remove" and len(words) > 2:
                    filters = [f for f in
                               (mod.get("word_filters") or [])
                               if f != words[2]]
                    mod["word_filters"] = filters
                    surface.update_settings(key, moderation=mod)
                    return (f"filter list now has {len(filters)} "
                            "entries.")
                if rest == "clear":
                    mod["word_filters"] = []
                    surface.update_settings(key, moderation=mod)
                    return "filters cleared."
                if rest == "action" and len(words) > 2 \
                        and words[2].lower() in (
                        "delete", "warn", "mute"):
                    if words[2].lower() == "mute" \
                            and key.partition(":")[0].strip() \
                            .lower() in ("whatsapp", "wa"):
                        # Platform truth, same as /gsettings:
                        # no per-person mute exists on WhatsApp
                        # for the filter to fire.
                        return ("that can't be set here: "
                                "WhatsApp has no per-person mute "
                                "for the filter to fire — delete "
                                "and warn are the filter actions "
                                "this platform carries out. "
                                "Nothing changed.")
                    mod["filter_action"] = words[2].lower()
                    surface.update_settings(key, moderation=mod)
                    return f"filter action: {words[2].lower()}."
                return None
            if head == "warns" and rest.isdigit():
                surface.update_settings(
                    key, warn_threshold=max(1, int(rest)))
                return (f"warn limit {max(1, int(rest))} before "
                        "an auto-mute.")
            if head == "mute" and rest == "default":
                arg = words[2] if len(words) > 2 else ""
                if arg.lower() in ("off", "none", "0", ""):
                    surface.update_settings(key, mute_default_s=0)
                    return "no standing mute length."
                if key.partition(":")[0].strip().lower() \
                        in ("whatsapp", "wa"):
                    # Platform truth, same as /gsettings:
                    # no per-person mute exists on WhatsApp
                    # for a standing length to fire.
                    return ("that can't be set here: WhatsApp "
                            "has no per-person mute, so a "
                            "standing mute length would never "
                            "fire — removal is the strongest "
                            "step this platform has. Nothing "
                            "changed.")
                secs, _why = _parse_duration(arg)
                if secs:
                    surface.update_settings(key,
                                            mute_default_s=secs)
                    return (f"standing mute length "
                            f"{_duration_text(secs)}.")
                return None
            if head == "fine" and rest.lstrip("-").isdigit():
                amount = max(0, int(rest))
                surface.update_settings(key, fine_amount=amount)
                return f"moderation fine {amount} sparks."
            return None
        if section == "acceptance":
            sub = head
            arg = words[1] if len(words) > 1 else ""
            tail = " ".join(words[2:])
            if sub == "mode" and arg.lower() in (
                    "open", "request", "invite"):
                self._acceptance_update(surface, key,
                                        mode=arg.lower())
                return f"join mode: {arg.lower()}."
            if sub in ("username", "photo", "firstname",
                       "lastname", "hygiene", "banned", "bots") \
                    and arg.lower() in ("on", "off"):
                current = surface.store.settings(key)
                crit = dict((current.get("acceptance") or {})
                            .get("criteria") or {})
                crit[sub] = (arg.lower() == "on")
                self._acceptance_update(surface, key,
                                        criteria=crit)
                return f"acceptance criterion {sub}: {arg.lower()}."
            if sub == "action" and arg.lower() in (
                    "decline", "restricted", "kick"):
                self._acceptance_update(surface, key,
                                        action=arg.lower())
                return f"failed checks now mean: {arg.lower()}."
            if sub == "grace" and arg.rstrip("m").isdigit():
                minutes = max(1, min(10080, int(arg.rstrip("m"))))
                self._acceptance_update(surface, key,
                                        grace_minutes=minutes)
                return f"acceptance grace: {minutes} minutes."
            if sub == "auto" and arg.lower() in ("on", "off"):
                self._acceptance_update(
                    surface, key, auto_approve=(arg.lower() == "on"))
                return f"auto-approve: {arg.lower()}."
            if sub == "override" and tail:
                bits = tail.split(None, 1)
                crit_name = bits[0].lower()
                act = bits[1].lower() if len(bits) > 1 else ""
                if crit_name in ("username", "photo", "firstname",
                                 "lastname", "hygiene", "banned",
                                 "bots") and act in (
                        "decline", "restricted", "kick", "default"):
                    current = surface.store.settings(key)
                    overrides = dict(
                        (current.get("acceptance") or {})
                        .get("overrides") or {})
                    if act == "default":
                        overrides.pop(crit_name, None)
                    else:
                        overrides[crit_name] = act
                    self._acceptance_update(surface, key,
                                            overrides=overrides)
                    return f"{crit_name} failures now take: {act}."
                return None
            if sub == "whitelist" and tail:
                bits = tail.split(None, 1)
                verb = bits[0].lower()
                token = bits[1].strip() if len(bits) > 1 else ""
                if verb in ("add", "remove") and token:
                    current = surface.store.settings(key)
                    wl = [str(w) for w in (
                        (current.get("acceptance") or {})
                        .get("bot_whitelist") or [])]
                    if verb == "add" and token not in wl:
                        wl.append(token)
                    if verb == "remove":
                        wl = [w for w in wl if w != token]
                    self._acceptance_update(surface, key,
                                            bot_whitelist=wl)
                    return (f"bot whitelist: "
                            f"{', '.join(wl) or 'empty'}.")
                return None
            return None
        if section == "economy":
            if head == "economy" and rest in ("on", "off"):
                surface.update_settings(key, economy_enabled=on)
                return f"economy {'on' if on else 'off'}."
            if head == "tax" and rest.rstrip("%").isdigit():
                pct = max(0, min(25, int(rest.rstrip("%"))))
                surface.update_settings(key, transfer_tax_pct=pct)
                return f"transfer tax {pct}%."
            if head == "governance" and rest.isdigit():
                surface.update_settings(
                    key, governance_threshold=max(0, int(rest)))
                return f"governance threshold {int(rest)} sparks."
            if head == "fine" and rest.lstrip("-").isdigit():
                amount = max(0, int(rest))
                surface.update_settings(key, fine_amount=amount)
                return f"moderation fine {amount} sparks."
            return None
        if section == "estate":
            return self._apply_estate_grammar(surface, key, words)
        if section == "presentation":
            if head == "theme" and rest in (
                    "house", "clean", "board"):
                self._presentation_update(surface, key, theme=rest)
                return f"presentation theme: {rest}."
            if head == "class" and len(words) > 2:
                cls = rest
                preset = words[2].lower()
                if cls in ("economy", "moderation", "game",
                           "briefing", "announcement", "leaderboard",
                           "settings", "chat") and preset in (
                        "clean", "board", "default"):
                    current = surface.store.settings(key)
                    classes = dict(
                        (current.get("presentation") or {})
                        .get("classes") or {})
                    if preset == "default":
                        classes.pop(cls, None)
                    else:
                        classes[cls] = preset
                    self._presentation_update(surface, key,
                                              classes=classes)
                    return f"{cls} surfaces render {preset}."
                return None
            if head == "stamp" and rest in ("on", "off"):
                self._presentation_update(surface, key, stamp=on)
                return f"footer stamp {'on' if on else 'off'}."
            if head == "stamp" and rest == "text":
                text = " ".join(words[2:])
                if text.lower() in ("", "default", "clear"):
                    self._presentation_update(surface, key,
                                              stamp_text="")
                    return "stamp text back to the house default."
                self._presentation_update(surface, key,
                                          stamp_text=text)
                return f"stamp text: “{text}”."
            if head == "stamp" and rest == "classes":
                text = " ".join(words[2:])
                if text.lower() in ("", "default", "clear"):
                    self._presentation_update(surface, key,
                                              stamp_classes=[])
                    return "stamp classes back to the defaults."
                from ...social.chat.typography import \
                    SURFACE_CLASSES

                wanted = [c.strip().lower() for c in
                          text.replace(" ", ",").split(",")
                          if c.strip() in SURFACE_CLASSES]
                if not wanted:
                    return None
                self._presentation_update(surface, key,
                                          stamp_classes=wanted)
                return f"stamp shows on: {', '.join(wanted)}."
            if head == "watermark" and rest in (
                    "off", "text", "media", "both"):
                self._presentation_update(surface, key,
                                          watermark=rest)
                return (f"media watermark: {rest} — off · text "
                        "(caption signature) · media (image "
                        "stamp) · both.")
            return None
        if section == "welcome":
            if head == "welcome" and rest in ("on", "off"):
                surface.update_settings(key, welcome_enabled=on)
                return f"welcome {'on' if on else 'off'}."
            if head == "goodbye" and rest in ("on", "off"):
                surface.update_settings(key, goodbye_enabled=on)
                return f"goodbye {'on' if on else 'off'}."
            if head == "clean" and rest in ("on", "off"):
                surface.update_settings(key, clean_service=on)
                return f"service cleanup {'on' if on else 'off'}."
            if head == "rules" and rest == "add" \
                    and len(words) > 2:
                current = surface.store.settings(key)
                rules = list(current.get("rules") or []) + [
                    " ".join(words[2:])]
                surface.update_settings(key, rules=rules)
                return f"rule added ({len(rules)} total)."
            if head == "rules" and rest == "clear":
                surface.update_settings(key, rules=[])
                return "rules cleared."
            return None
        if section == "expression":
            # The console's words ARE the gsettings tail (minus
            # the head): one grammar, one write path (§21).
            return self._apply_expression_grammar(
                surface, key, list(words))
        if section == "reputation":
            # Same rule (§32): the console's words are the
            # gsettings grammar.
            return self._apply_reputation_grammar(
                surface, key, list(words))
        if section == "conflict":
            # Same rule (§23): the console's words are the
            # gsettings grammar.
            return self._apply_conflict_grammar(
                surface, key, list(words))
        if section == "rhythm":
            # Same rule (P3): the console's rhythm words are the
            # gsettings grammar — `review …` lines route to the
            # §26 grammar, everything else to rhythm's.
            if head == "review":
                return self._apply_review_grammar(
                    surface, key, list(words[1:]), by=sender)
            return self._apply_rhythm_grammar(
                surface, key, list(words))
        if section == "boards":
            # Same rule (§14): the console's boards words are
            # the gsettings grammar.
            return self._apply_boards_grammar(
                surface, key, list(words))
        if section == "cleanup":
            # Same rule: the console's cleanup words are the
            # gsettings grammar.
            return self._apply_cleanup_grammar(
                surface, key, list(words))
        if section == "p5":
            # Same rule (§16): the console's p5 words are the
            # gsettings grammar.
            return self._apply_p5_grammar(
                surface, key, list(words), by=sender,
                message=message)
        if section == "police":
            # §36: NOT a settings grammar — the owner's police
            # override.  Its own apply, owner-gated inside.
            return self._apply_police_grammar(
                surface, key, message, sender, list(words))
        return None

    def _apply_police_grammar(self, surface, key, message,  # type: ignore[no-untyped-def]
                              sender, words):
        """The owner's hand on the §24 police (§36): view a
        member's wanted state, set/clear their stars, pardon,
        call off an enforcement, set the group's enforcement
        focus.  OWNER-ONLY — the operator gate, not group
        adminship: this is the owner's manipulation power over
        the police, and every verb journals in the police
        journal.  Grammar: `wanted <member>` · `stars <member>
        <0-5>` · `pardon <member>` · `calloff <member>` ·
        `focus <pct>|off`."""
        if not self._operator_or_false(message):
            return ("the police override is the owner's hand "
                    "alone — group admins run the precinct "
                    "through /police, not over it. Nothing was "
                    "changed.")
        w = [str(x) for x in words or []]
        if not w:
            return None
        head = w[0].lower()
        by = surface._person_of(key, sender)
        if head == "focus":
            if len(w) == 2 and w[1].lower() == "off":
                out = surface.police_owner_focus(
                    key, 1.0, by_person=by)
            elif len(w) == 2 and w[1].rstrip("%").isdigit():
                pct = int(w[1].rstrip("%"))
                out = surface.police_owner_focus(
                    key, pct / 100.0, by_person=by)
            else:
                return None
            if not out.get("ok"):
                return "the registry is frozen — no override landed."
            return (f"enforcement focus: "
                    f"{out['aggression']:g}× — the bully index "
                    "thresholds read it from the next sweep.")
        if head not in ("wanted", "stars", "pardon", "calloff"):
            return None
        if len(w) < 2:
            return None
        member = surface.member_id_for_token(key, w[1])
        if member is None:
            return (f"no member matches {w[1]} in this group's "
                    "ledger — an id, @username or exact name "
                    "resolves; nothing was changed.")
        if head == "wanted":
            view = surface.police_wanted_view(key, member)
            if view.get("reason") == "frozen":
                return "the wanted state is unreadable right now."
            label = surface._member_label(key, member)
            bits = [f"{label}: {view['stars']}★ "
                    f"(index {view['index']:g}"]
            if view.get("mark") is not None:
                bits.append(f"owner's mark at {view['mark']}★")
            if view.get("episode_rung"):
                bits.append(
                    f"episode served to rung "
                    f"{view['episode_rung']}")
            if view.get("police_bounties"):
                bits.append(
                    f"{view['police_bounties']} police "
                    "bounty(ies) open")
            bits.append(f"focus ×{view['focus']:g}")
            return " · ".join(bits) + ")"
        if head == "stars":
            if len(w) < 3 or not w[2].isdigit():
                return None
            out = surface.police_owner_set_stars(
                key, member, int(w[2]), by_person=by)
            if not out.get("ok"):
                return ("stars live between 0 and 5 — nothing "
                        "was changed."
                        if out.get("reason") == "range" else
                        "the registry is frozen — no override "
                        "landed.")
            return (f"wanted level set: "
                    f"{surface._member_label(key, member)} now "
                    f"stands at {out['stars']}★ — the mark "
                    "holds for the index window and is journaled "
                    "in the police journal.")
        if head == "pardon":
            out = surface.police_owner_pardon(
                key, member, by_person=by)
            if not out.get("ok"):
                return "the registry is frozen — no pardon landed."
            line = (f"pardon: {surface._member_label(key, member)}"
                    " — slate forgiven, episode closed, "
                    f"{out['bounties_cancelled']} police "
                    "bounty(ies) cancelled.")
            if out.get("released"):
                line += " Their jail lifts — a release."
            return line
        out = surface.police_owner_call_off(
            key, member, by_person=by)
        if not out.get("ok"):
            return "the registry is frozen — no call-off landed."
        return (f"called off: the precinct stands down on "
                f"{surface._member_label(key, member)} at "
                f"{out['stars']}★ — the episode is marked "
                "handled and police bounties are cancelled, "
                "but the heat stays on the books.")

    # ---- the console's channel side (§16 / P4a) ------------------
    def _console_channel(self, surface, message, sender,  # type: ignore[no-untyped-def]
                         session, token, words):
        """Route one /console cN … line: menu, section card, or
        a section change.  Every change journals to the
        CHANNEL's /glog like a group console change, and queue
        changes take an immediate drive beat so a due post
        goes out on the owner's action, not the next stray
        group message."""
        key = (session.get("channels") or {})[token]
        if len(words) == 1:
            return self._console_channel_menu(surface, key,
                                              message, token)
        section = words[1].lower()
        if section not in self._CONSOLE_CHANNEL_SECTIONS:
            return self._console_channel_menu(surface, key,
                                              message, token)
        if len(words) == 2:
            return self._console_channel_section(
                surface, key, message, token, section)
        reply = self._console_channel_apply(
            surface, key, message, sender, session, section,
            words[2:])
        if reply is None:
            return (f"I didn't follow that for the channel "
                    f"{section} section — its card shows the "
                    "grammar.")
        surface._audit(key, "console_change", by=sender,
                       user_id=sender,
                       section=f"channel:{section}",
                       change=" ".join(words[2:]))
        surface.store.save()
        return self._console_channel_section(
            surface, key, message, token, section,
            lead=f"✓ {reply}")

    def _console_channel_menu(self, surface, key, message,  # type: ignore[no-untyped-def]
                              token):
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card

        claim = surface.claim_state(key)
        analytics = surface.channel_analytics(key)
        cfg = surface.channel_cfg(key)
        title = surface.channel_title(key) or key
        subs = analytics["subscribers"]
        card = Card("settings", title=f"Channel — {title}",
                    icon="📣")
        card.block_lines([
            f"claim: {'claimed (' + claim['source'] + ') — the queue posts' if claim['claimed'] else 'NOT claimed — the queue holds until my bot is an admin here, or you designate it in settings'}",
            f"subscribers: {f'{subs:,} (latest sample)' if subs is not None else 'not readable yet — see analytics'}",
            f"queue: {len(surface.channel_queue(key))} waiting"
            f" · {analytics['posts']} posted · cadence floor "
            f"{int(cfg.get('cadence_min') or 0)} min · trigger "
            f"links: {len(cfg.get('links') or {})} group(s)",
            f"series: {analytics['series']['active']} running"
            f" · digest: "
            f"{'on (' + str((cfg.get('digest') or {}).get('kind') or 'daily') + ')' if (cfg.get('digest') or {}).get('enabled') else 'off'}"
            f" · discussion: "
            f"{'linked' if analytics['discussion'] else 'none on record'}",
        ])
        card.note("the channel itself stays silent (§16) — "
                  "everything happens here in DM. Sections: "
                  "queue · post · posts · series · analytics "
                  "· settings")
        buttons = buttons_from_spec([
            [("queue", "Queue", f"console {token} queue"),
             ("post", "Post", f"console {token} post")],
            [("analytics", "Analytics",
              f"console {token} analytics"),
             ("settings", "Settings",
              f"console {token} settings")],
            [("home", "Console home", "console")],
        ], settings=surface.store.settings(key))
        return self._card(message, card, buttons,
                              session="console", session_edit=True)

    def _console_channel_section(self, surface, key, message,  # type: ignore[no-untyped-def]
                                 token, section, lead=""):
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card

        s = surface.store.settings(key)
        cfg = surface.channel_cfg(key)
        title = surface.channel_title(key) or key
        card = Card("settings",
                    title=f"Channel {section} — {title}",
                    icon="📣")
        if lead:
            card.line(lead)
        if section == "queue":
            queue = surface.channel_queue(key)
            if queue:
                tz = int((surface.rhythm_cfg(key) or {}).get(
                    "tz_offset_min") or 0)
                now = time.time()
                lines = []
                for i, entry in enumerate(queue, 1):
                    due = float(entry.get("due") or 0)
                    when = ("due now" if due <= now else
                            time.strftime("%a %d %b %H:%M",
                                          time.gmtime(due
                                                      + tz * 60))
                            + " channel time")
                    kind = str(entry.get("kind") or "once")
                    if kind in ("daily", "weekly"):
                        when = f"{kind} {entry.get('time')}" \
                               f" · next {when}"
                    src = str(entry.get("source") or "dm")
                    body = str(entry.get("text") or "")
                    if entry.get("poll"):
                        poll = entry.get("poll") or {}
                        body = (f"🗳 poll"
                                f"{' (quiz)' if poll.get('quiz') else ''}"
                                f" — {poll.get('question') or body}"
                                f" · {' · '.join(str(o) for o in (poll.get('options') or []))}")
                    elif entry.get("forward"):
                        fwd = entry.get("forward") or {}
                        body = (f"↪ cross-post from "
                                f"{fwd.get('from_title') or fwd.get('from_channel') or 'a channel'}")
                    elif entry.get("media"):
                        body = "📎 media" + (f" — {body}"
                                             if body else "")
                    lines.append(
                        f"{i}. [{kind} · {src}] {when} — "
                        f"{body[:80]}")
                card.block_lines(lines, title="Queued")
            else:
                card.line("the queue is empty — the post "
                          "section fills it.")
            card.block_lines([
                "`cancel N` · `clear` · `post now`",
            ], title="Grammar")
        elif section == "post":
            card.block_lines([
                "compose here; posts go out through the bot "
                "on their cadence, rendered as announcements "
                "(stamp on, per the channel's presentation "
                "settings)",
                "attach media to your console message and "
                "write `now` (plus an optional caption) to "
                "queue the file itself",
            ])
            card.block_lines([
                "`now TEXT` · just TEXT — post as soon as due",
                "`at HH:MM TEXT` — one-shot at that channel "
                "time",
                "`daily HH:MM TEXT` · `weekly mon HH:MM TEXT`",
                "`poll [quiz|multi|open] QUESTION | OPT | OPT"
                " …` — a poll post (anonymous unless `open`; "
                "in a quiz, mark the right option with `*`)",
                "`xpost <channel> <post N>` — native-forward "
                "post N from another claimed channel of yours "
                "into this one",
                "`fwd <source name> | TEXT` — a post forwarded "
                "from a source you name (journaled as such)",
                "`media <file path> | caption`",
            ], title="Grammar")
        elif section == "posts":
            ring = surface.channel_posts(key)
            views = surface.channel_post_views_map(key)
            if ring:
                lines = []
                for i, record in enumerate(ring, 1):
                    when = time.strftime(
                        "%d %b %H:%M", time.gmtime(
                            float(record.get("ts") or 0)))
                    kind = str(record.get("kind") or "once")
                    if record.get("poll"):
                        kind = "poll"
                    label = str(record.get("label") or "")
                    line = (f"{i}. [{kind} · "
                            f"{record.get('source') or 'dm'}] "
                            f"{when} UTC — {label[:60]}")
                    mid = str(record.get("message_id") or "")
                    if mid in views:
                        line += f" · {views[mid]:,} views"
                    if record.get("edited_at"):
                        line += " · edited"
                    lines.append(line)
                card.block_lines(lines, title="Published")
            else:
                card.line("nothing published yet — the post "
                          "section is where posts are born.")
            card.block_lines([
                "`edit N <new text>` — re-render + replace "
                "post N's text on the platform",
                "`delete N` — delete post N on the platform "
                "and from the record",
                "`results N` — a poll post's current results, "
                "read back from the platform",
            ], title="Grammar")
        elif section == "series":
            rows = surface.channel_series(key)
            if rows:
                lines = []
                days = ["mon", "tue", "wed", "thu", "fri",
                        "sat", "sun"]
                for i, row in enumerate(rows, 1):
                    cadence = str(row.get("kind") or "daily")
                    if cadence == "weekly":
                        cadence += " " + days[
                            int(row.get("weekday") or 0)]
                    cadence += f" {row.get('time') or ''}"
                    total = len(row.get("items") or [])
                    pos = int(row.get("position") or 0)
                    state = ("archived" if row.get("archived")
                             else "done" if row.get("done")
                             else "running")
                    lines.append(
                        f"{i}. {row.get('name')} — {cadence}"
                        f" · {pos}/{total} posted · {state}")
                card.block_lines(lines, title="Series")
            else:
                card.line("no series yet — a series is a "
                          "named format that posts its next "
                          "instalment on its cadence, in order.")
            card.block_lines([
                "`new <name> daily HH:MM` · "
                "`new <name> weekly mon HH:MM`",
                "`add N <text>` — append an instalment",
                "`archive N` · `resume N` · `drop N`",
            ], title="Grammar")
        elif section == "analytics":
            card.block_lines(
                self._channel_analytics_lines(surface, key),
                title="What the platform exposes")
        elif section == "settings":
            links = cfg.get("links") or {}
            link_lines = [
                f"{g} → {', '.join(kinds)}"
                for g, kinds in sorted(links.items())] \
                or ["no trigger links — group events don't "
                    "cross-post here"]
            pres = s.get("presentation") or {}
            dcfg = cfg.get("digest") or {}
            digest_line = "digest: off"
            if dcfg.get("enabled"):
                digest_line = (
                    f"digest: {dcfg.get('kind') or 'daily'}"
                    f"{' ' + ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'][int(dcfg['weekday'])] if dcfg.get('kind') == 'weekly' and dcfg.get('weekday') is not None else ''}"
                    f" at {dcfg.get('time') or '09:00'}"
                    f" · silence "
                    f"{'on' if dcfg.get('silence', True) else 'off'}")
            card.block_lines([
                f"cadence floor: "
                f"{int(cfg.get('cadence_min') or 0)} min "
                "between posts",
                f"analytics sampling: "
                f"{'on' if cfg.get('analytics', True) else 'off'}",
                f"stamp: "
                f"{'on' if pres.get('stamp', True) else 'off'}"
                " (presentation)",
                digest_line,
                "trigger links:",
                *link_lines,
            ])
            card.block_lines([
                "`cadence N` · `analytics on|off` · "
                "`stamp on|off`",
                "`digest on|off` · `digest daily HH:MM` · "
                "`digest weekly mon HH:MM` · "
                "`digest silence on|off`",
                "`link <group handle|key> "
                "[season_podium war_result milestone]`",
                "`unlink <group handle|key>` · "
                "`designate on|off` (owner)",
            ], title="Grammar")
        buttons = buttons_from_spec(
            [[("back", "Channel", f"console {token}"),
              ("home", "Console home", "console")]],
            settings=s)
        return self._card(message, card, buttons,
                              session="console", session_edit=True)

    def _channel_analytics_lines(self, surface, key):  # type: ignore[no-untyped-def]
        """The analytics truth for the console card: what was
        actually sampled, and — just as explicitly — what the
        platform does not expose.  "0 subscribers" (a real
        sample that read zero) and "cannot read" are different
        lines, always."""
        analytics = surface.channel_analytics(key)
        platform = str(key).split(":", 1)[0]
        adapter = self._group_adapter(platform)
        can_count = adapter is not None and callable(
            getattr(adapter, "admin_member_count", None))
        lines = []
        subs = analytics["subscribers"]
        if subs is not None:
            line = f"subscribers: {subs:,} " \
                   f"({analytics['samples']} sample(s) on the " \
                   "cycle)"
            if analytics["delta_24h"] is not None:
                line += f" · 24h {analytics['delta_24h']:+d}"
            if analytics["delta_7d"] is not None:
                line += f" · 7d {analytics['delta_7d']:+d}"
            lines.append(line)
        elif can_count:
            lines.append(
                "subscribers: this platform exposes the count "
                "and I read it on the cycle — no sample has "
                "landed yet, so there is no number to show (an "
                "unsampled channel is not a 0-subscriber one)")
        else:
            lines.append(
                "subscribers: this platform exposes no count "
                "to me — nothing is invented in its place")
        if analytics["posts"]:
            lines.append(
                f"posts: {analytics['posts']} posted from the "
                f"queue · last at "
                f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(analytics['last_post_at']))} UTC")
        else:
            lines.append("posts: none posted yet")
        if analytics["views_total"] is not None:
            line = (f"post views: "
                    f"{analytics['views_total']:,} total "
                    "where the platform returned them")
            if analytics["views_sampled_at"]:
                line += (f" · last read "
                         f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(analytics['views_sampled_at']))} UTC")
            lines.append(line)
        else:
            lines.append(
                "post views: not returned by any connected "
                "platform — not estimated, not guessed")
        stats = analytics["stats"]
        can_stats = adapter is not None and callable(
            getattr(adapter, "channel_broadcast_stats", None))
        if stats is not None:
            bits = []
            if stats["followers"] is not None:
                bit = f"followers {stats['followers']:,}"
                if stats["followers_delta"] is not None:
                    bit += (f" ({stats['followers_delta']:+,} "
                            "over the stats period)")
                bits.append(bit)
            if stats["views_per_post"] is not None:
                bits.append(f"{stats['views_per_post']:,.0f} "
                            "views/post")
            if stats["shares_per_post"] is not None:
                bits.append(f"{stats['shares_per_post']:,.0f} "
                            "shares/post")
            if stats["reactions_per_post"] is not None:
                bits.append(
                    f"{stats['reactions_per_post']:,.0f} "
                    "reactions/post")
            lines.append(
                "broadcast stats (Telegram's own, sampled "
                f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(stats['sampled_at']))} UTC): "
                + " · ".join(bits))
            if analytics["stats_unavailable"]:
                lines.append(
                    "the latest stats check was refused ("
                    f"{analytics['stats_unavailable']}) — the "
                    "sample above is the last one granted")
        elif analytics["stats_unavailable"]:
            lines.append(
                "broadcast stats: Telegram does not grant "
                f"them for this channel — "
                f"{analytics['stats_unavailable']} (checked "
                f"{time.strftime('%Y-%m-%d %H:%M', time.gmtime(analytics['stats_checked_at']))} UTC) "
                "— not a zero, and not retried off-cycle")
        elif can_stats:
            lines.append(
                "broadcast stats: not sampled yet — they ride "
                "the same cycle as the counts")
        else:
            lines.append(
                "broadcast stats: this platform exposes none "
                "to me — nothing is invented in their place")
        series = analytics["series"]
        if series["total"]:
            lines.append(
                f"series: {series['active']} running · "
                f"{series['done']} complete "
                f"({series['total']} total)")
        disc = analytics["discussion"]
        if disc:
            line = ("discussion: a linked group — "
                    f"{disc['mirrors']} mirrored post(s) on "
                    "record")
            if disc["comments_total"]:
                line += (f" · {disc['comments_total']} "
                         f"comment(s), {disc['comments_7d']} "
                         "in the last 7d")
            lines.append(line)
        return lines

    def _console_channel_apply(self, surface, key, message,  # type: ignore[no-untyped-def]
                               sender, session, section, words):
        """Apply one channel-console grammar line.  Returns the
        reply line, or None when the words don't parse."""
        w = [str(x) for x in words or []]
        low = [x.lower() for x in w]
        if not w:
            return None
        head = low[0]
        if section == "queue":
            if head == "cancel" and len(w) == 2 \
                    and w[1].isdigit():
                gone = surface.channel_cancel(key, int(w[1]))
                if gone is None:
                    return "no queue entry at that number."
                return (f"cancelled the {gone.get('kind')} "
                        "post — it will not go out.")
            if low == ["clear"]:
                n = surface.channel_clear(key)
                return f"queue cleared — {n} post(s) dropped."
            if low == ["post", "now"]:
                posts = self._channel_drive(key, force=True)
                if posts:
                    return f"{len(posts)} post(s) went out."
                if not surface.claim_state(key)["claimed"]:
                    return ("nothing posted — this channel is "
                            "not claimed yet (settings → "
                            "designate, or make my bot an "
                            "admin).")
                return "nothing was due."
            return None
        if section == "post":
            return self._console_channel_compose(
                surface, key, message, sender, w,
                session=session)
        if section == "posts":
            return self._console_channel_posts_apply(
                surface, key, message, w)
        if section == "series":
            return self._console_channel_series_apply(
                surface, key, sender, w)
        if section == "settings":
            if head == "cadence" and len(w) == 2 \
                    and w[1].isdigit():
                n = max(0, min(1440, int(w[1])))
                self._channel_update(surface, key,
                                     cadence_min=n)
                return (f"cadence floor: {n} min between "
                        "posts.")
            if head == "analytics" and len(w) == 2 \
                    and low[1] in ("on", "off"):
                on = low[1] == "on"
                self._channel_update(surface, key, analytics=on)
                return f"analytics sampling {'on' if on else 'off'}."
            if head == "stamp" and len(w) == 2 \
                    and low[1] in ("on", "off"):
                import copy as _copy

                on = low[1] == "on"
                current = surface.store.settings(key)
                pres = _copy.deepcopy(
                    current.get("presentation") or {})
                pres["stamp"] = on
                surface.update_settings(key, presentation=pres)
                return (f"stamp {'on' if on else 'off'} — "
                        "channel posts are presented surfaces"
                        if on else
                        "stamp off for this channel.")
            if head in ("link", "unlink") and len(w) >= 2:
                target = (session.get("groups") or {}).get(
                    low[1]) or w[1]
                if target not in surface.store.data \
                        or surface.is_channel(target):
                    return ("that group handle doesn't resolve "
                            "— use a g-handle from the console "
                            "home or the group's full key.")
                links = dict(surface.channel_cfg(key).get(
                    "links") or {})
                if head == "unlink":
                    links.pop(target, None)
                    self._channel_update(surface, key,
                                         links=links)
                    return f"unlinked {target}."
                kinds = [k for k in low[2:]
                         if k in surface.CHANNEL_TRIGGERS]
                if len(kinds) != len(low[2:]):
                    return ("trigger kinds are: "
                            "season_podium, war_result, "
                            "milestone.")
                links[target] = kinds or list(
                    surface.CHANNEL_TRIGGERS)
                self._channel_update(surface, key, links=links)
                return (f"linked {target}: "
                        f"{', '.join(links[target])} cross-post "
                        "here when they happen there.")
            if head == "designate" and len(w) == 2 \
                    and low[1] in ("on", "off"):
                if not self._operator_or_false(message):
                    return ("designation is the owner's call — "
                            "only the operator can claim a "
                            "channel by name.")
                on = low[1] == "on"
                surface.set_designated(key, on, by=sender)
                return (f"designation {'on' if on else 'off'} — "
                        "the claim follows it.")
            if head == "digest":
                from ...social.chat.group_rhythm import \
                    parse_hhmm

                dcfg = dict(surface.channel_cfg(key).get(
                    "digest") or {})
                if len(w) == 2 and low[1] in ("on", "off"):
                    dcfg["enabled"] = low[1] == "on"
                    self._channel_update(surface, key,
                                         digest=dcfg)
                    return (f"digest "
                            f"{'on' if dcfg['enabled'] else 'off'}"
                            " — it posts on its cadence from "
                            "recorded facts only.")
                if len(w) == 3 and low[1] == "silence" \
                        and low[2] in ("on", "off"):
                    dcfg["silence"] = low[2] == "on"
                    self._channel_update(surface, key,
                                         digest=dcfg)
                    return ("digest silence on — an empty "
                            "period posts nothing."
                            if dcfg["silence"] else
                            "digest silence off — an empty "
                            "period still posts its quiet line.")
                days = {"mon": 0, "tue": 1, "wed": 2, "thu": 3,
                        "fri": 4, "sat": 5, "sun": 6}
                if low[1:2] == ["daily"] and len(w) == 3 \
                        and parse_hhmm(w[2]) is not None:
                    dcfg.update({"enabled": True,
                                 "kind": "daily", "time": w[2],
                                 "weekday": None})
                    self._channel_update(surface, key,
                                         digest=dcfg)
                    surface.channel_digest_reset(key)
                    return (f"digest: daily at {w[2]} channel "
                            "time.")
                if low[1:2] == ["weekly"] and len(w) == 4 \
                        and low[2][:3] in days \
                        and parse_hhmm(w[3]) is not None:
                    dcfg.update({"enabled": True,
                                 "kind": "weekly", "time": w[3],
                                 "weekday": days[low[2][:3]]})
                    self._channel_update(surface, key,
                                         digest=dcfg)
                    surface.channel_digest_reset(key)
                    return (f"digest: weekly {low[2][:3]} at "
                            f"{w[3]} channel time.")
                return ("digest grammar: `digest on|off` · "
                        "`digest daily HH:MM` · "
                        "`digest weekly mon HH:MM` · "
                        "`digest silence on|off`.")
            return None
        return None

    def _console_channel_posts_apply(self, surface, key,  # type: ignore[no-untyped-def]
                                     message, w):
        """Published-post management grammar (§7): edit /
        delete / results by the number the posts card shows.
        Every action runs through the adapter holding THIS
        channel; a missing verb or a platform refusal is
        reported in its own words, and the record changes
        only when the platform confirms."""
        from ...social.chat.base import ChatKind, ChatRef

        low = [x.lower() for x in w]
        if len(w) < 2 or not w[1].isdigit():
            return None
        head = low[0]
        if head not in ("edit", "delete", "results"):
            return None
        record = surface.channel_post_record(key, int(w[1]))
        if record is None:
            return "no published post at that number."
        mid = str(record.get("message_id") or "")
        if not mid:
            return ("that post has no platform id on record, "
                    "so I can't reach it on the platform.")
        platform = str(key).split(":", 1)[0]
        adapter = self._group_adapter(platform)
        ref = ChatRef.parse(str(key), kind=ChatKind.CHANNEL)
        if head == "delete":
            fn = getattr(adapter, "admin_delete", None) \
                if adapter is not None else None
            if fn is None:
                return "this surface cannot delete posts."
            try:
                res = fn(ref, mid)
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "error": str(exc)}
            if not (isinstance(res, dict) and res.get("ok")):
                return ("the platform refused the delete ("
                        + str((res or {}).get("error")
                              or "no reason given")
                        + ") — the post stays, on the platform "
                        "and in the record.")
            surface.channel_forget_post(key, mid)
            return ("deleted — on the platform and in the "
                    "record.")
        if head == "edit":
            text = " ".join(w[2:]).strip()
            if not text:
                return ("edit needs the new text: "
                        "`edit N <new text>`.")
            fn = getattr(adapter, "edit_post", None) \
                if adapter is not None else None
            if fn is None:
                return "this surface cannot edit posts."
            rendered = surface._channel_post_text(
                key, {"kind": "once", "text": text})
            try:
                res = fn(ref, mid, rendered)
            except Exception as exc:  # noqa: BLE001
                res = {"ok": False, "error": str(exc)}
            if not (isinstance(res, dict) and res.get("ok")):
                return ("the platform refused the edit ("
                        + str((res or {}).get("error")
                              or "no reason given")
                        + ") — the post keeps its old text.")
            surface.note_channel_post_edited(key, mid,
                                             label=text)
            return "edited — the post now carries the new text."
        # results
        if not record.get("poll"):
            return ("that post is not a poll — results are a "
                    "poll read.")
        fn = getattr(adapter, "channel_poll_results", None) \
            if adapter is not None else None
        if fn is None:
            return ("this surface can't read poll results "
                    "back (the Bot API exposes none) — poll "
                    "results ride the userbot surface.")
        try:
            res = fn(ref, mid)
        except Exception as exc:  # noqa: BLE001
            res = {"ok": False, "error": str(exc)}
        if not (isinstance(res, dict) and res.get("ok")):
            return ("the platform returned no results ("
                    + str((res or {}).get("error")
                          or "no reason given") + ").")
        total = res.get("total_voters")
        bits = []
        for opt in res.get("options") or []:
            voters = opt.get("voters")
            bit = (f"{opt.get('text')}: {voters:,}"
                   if voters is not None else
                   f"{opt.get('text')}: not exposed")
            if opt.get("correct"):
                bit += " ✓"
            bits.append(bit)
        head_line = (f"{total:,} voter(s)" if total is not None
                     else "total not exposed")
        return f"poll results — {head_line} · " \
               + " · ".join(bits)

    def _console_channel_series_apply(self, surface, key,  # type: ignore[no-untyped-def]
                                      sender, w):
        """Series grammar (§7): open a named format, append
        instalments, archive/resume/drop — by the number the
        series card shows."""
        low = [x.lower() for x in w]
        head = low[0]
        rows = surface.channel_series(key)
        days = {"mon": 0, "tue": 1, "wed": 2, "thu": 3,
                "fri": 4, "sat": 5, "sun": 6}
        if head == "new":
            from ...social.chat.group_rhythm import \
                parse_hhmm

            # The cadence is the LAST daily/weekly token whose
            # tail parses — a series may itself be named
            # "Daily" ("TA Daily daily 09:00" is name +
            # cadence, not cadence + cadence).
            idx = None
            kind = ""
            for i in range(len(low) - 1, 0, -1):
                if low[i] == "daily" and i + 1 < len(w) \
                        and parse_hhmm(w[i + 1]) is not None:
                    idx, kind = i, "daily"
                    break
                if low[i] == "weekly" and i + 2 < len(w) \
                        and low[i + 1][:3] in days \
                        and parse_hhmm(w[i + 2]) is not None:
                    idx, kind = i, "weekly"
                    break
            if idx is None or idx == 0:
                return ("a series opens with a name and a "
                        "cadence: `new <name> daily HH:MM` or "
                        "`new <name> weekly mon HH:MM`.")
            name = " ".join(w[1:idx])
            rest = w[idx + 1:]
            weekday = None
            if kind == "weekly":
                if len(rest) < 2 \
                        or rest[0].lower()[:3] not in days:
                    return ("weekly needs a day: "
                            "`new <name> weekly mon HH:MM`.")
                weekday = days[rest[0].lower()[:3]]
                at = rest[1]
            else:
                if not rest:
                    return ("daily needs a time: "
                            "`new <name> daily HH:MM`.")
                at = rest[0]
            row = surface.channel_series_create(
                key, name, {"kind": kind, "time": at,
                            "weekday": weekday}, by=sender)
            if row is None:
                return ("that didn't open — the time must "
                        "parse as HH:MM, and a channel runs "
                        "at most 20 series.")
            return (f"series open: {name} ({kind} {at}) — "
                    f"add instalments with "
                    f"`add {len(rows) + 1} <text>`.")
        if head in ("add", "archive", "resume", "drop"):
            if len(w) < 2 or not w[1].isdigit():
                return None
            pos = int(w[1])
            if pos < 1 or pos > len(rows):
                return "no series at that number."
            row = rows[pos - 1]
            if head == "add":
                text = " ".join(w[2:]).strip()
                if not text:
                    return ("add needs the instalment's "
                            "words: `add N <text>`.")
                added = surface.channel_series_add(
                    key, str(row.get("id") or ""), text)
                if added is None:
                    return "that series is gone."
                return (f"instalment "
                        f"{len(added.get('items') or [])} "
                        f"added to {row.get('name')}.")
            if head == "archive":
                surface.channel_series_set_archived(
                    key, str(row.get("id") or ""), True)
                return (f"archived {row.get('name')} — it "
                        "posts nothing until resumed.")
            if head == "resume":
                surface.channel_series_set_archived(
                    key, str(row.get("id") or ""), False)
                return f"resumed {row.get('name')}."
            surface.channel_series_drop(
                key, str(row.get("id") or ""))
            return (f"dropped {row.get('name')} — its "
                    "unposted instalments go with it.")
        return None

    def _console_channel_enqueue_now(self, surface, key,  # type: ignore[no-untyped-def]
                                     sender, now, entry):
        """Enqueue a due-now entry and take the immediate
        drive beat — the compose tail for the poll and
        cross-post grammars (same replies as a text `now`)."""
        stored = surface.channel_enqueue(
            key, {**entry, "created_by": sender}, now=now)
        if stored is None:
            return ("the queue is full (100 posts) — cancel "
                    "something from the queue section first.")
        posts = self._channel_drive(key, now=now, force=True)
        if any(p.get("entry") == stored["id"] for p in posts):
            return "posted now."
        if not surface.claim_state(key)["claimed"]:
            return ("queued — it posts the moment this "
                    "channel is claimed (my bot made admin, "
                    "or your designation in settings).")
        return "queued — it rides the next cycle beat."

    def _channel_poll_parse(self, body: str):  # type: ignore[no-untyped-def]
        """The poll grammar: ``[quiz|multi|open] QUESTION |
        OPT | OPT …`` — a leading ``*`` marks a quiz's correct
        option.  Returns the poll block, or a refusal line in
        Devon's words (Telegram's caps checked here so a bad
        poll never reaches the platform to fail there)."""
        parts = [p.strip() for p in str(body or "").split("|")]
        head = parts[0] if parts else ""
        flags = {"quiz": False, "multi": False, "open": False}
        words = head.split()
        while words and words[0].lower() in flags:
            flags[words[0].lower()] = True
            words = words[1:]
        question = " ".join(words)
        options: list[str] = []
        correct = None
        for i, raw_opt in enumerate(parts[1:]):
            opt = raw_opt
            if opt.startswith("*"):
                correct = i
                opt = opt[1:].strip()
            options.append(opt)
        if not question or len(options) < 2:
            return ("a poll needs a question and at least two "
                    "options: `poll QUESTION | OPT | OPT`.")
        if len(options) > 10 or any(
                not o or len(o) > 100 for o in options):
            return ("Telegram caps polls at 2–10 options of "
                    "≤100 characters each.")
        if len(question) > 255:
            return ("Telegram caps a poll question at 255 "
                    "characters.")
        if flags["quiz"] and correct is None:
            return ("a quiz poll marks its correct option "
                    "with `*`: `poll quiz Q | *right | wrong`.")
        return {"question": question, "options": options,
                "anonymous": not flags["open"],
                "multiple": flags["multi"],
                "quiz": flags["quiz"], "correct": correct,
                "explanation": ""}

    def _channel_xpost_parse(self, surface, key, session,  # type: ignore[no-untyped-def]
                             body: str):
        """The cross-post grammar: ``xpost <channel> <post
        N>``.  Both ends must be the owner's claimed channels
        — the source so she may read/reach its posts, the
        target enforced again at post time by the cycle's
        claim gate.  Returns a forward queue entry, or a
        refusal line."""
        parts = str(body or "").split()
        if len(parts) != 2 or not parts[1].isdigit():
            return ("xpost needs a channel and a post "
                    "number: `xpost <channel> <post N>`.")
        token = parts[0]
        skey = None
        if session:
            skey = (session.get("channels") or {}).get(
                token.lower())
        if skey is None and token in surface.store.data \
                and surface.is_channel(token):
            skey = token
        if skey is None:
            for cand in surface.channel_keys():
                if surface.channel_title(cand).lower() \
                        == token.lower():
                    skey = cand
                    break
        if skey is None:
            return ("that channel doesn't resolve — use its "
                    "c-handle from the console home, its "
                    "title, or its full key.")
        if skey == key:
            return "a channel can't cross-post into itself."
        if not surface.claim_state(skey)["claimed"]:
            return ("the source channel isn't claimed — "
                    "cross-posts run only between your "
                    "claimed channels.")
        record = surface.channel_post_record(
            skey, int(parts[1]))
        if record is None:
            return ("no published post at that number in "
                    "the source channel.")
        mid = str(record.get("message_id") or "")
        if not mid:
            return ("that post has no platform id on record "
                    "— I can't forward what I can't name.")
        title = surface.channel_title(skey) or "a channel"
        return {"kind": "forward", "text": "",
                "forward": {"from_channel": skey,
                            "message_id": mid,
                            "from_title": title},
                "due": 0.0, "source": f"xpost:{title}",
                "label": (f"↪ from {title}: "
                          f"{record.get('label') or ''}")[:80]}

    def _console_channel_compose(self, surface, key, message,  # type: ignore[no-untyped-def]
                                 sender, words, session=None):
        """The post grammar: compose one queue entry from the
        owner's words (and any media attached to the console
        message itself).  A due entry takes an immediate drive
        beat, so `now` really means now when the channel is
        claimed."""
        from ...social.chat.group_rhythm import parse_hhmm

        raw = " ".join(words).strip()
        if not raw:
            return None
        low = raw.lower()
        now = time.time()
        tz = int((surface.rhythm_cfg(key) or {}).get(
            "tz_offset_min") or 0)
        kind = "once"
        due = now
        at_time = ""
        weekday = None
        source = "dm"
        text = raw
        media = None
        first_media = getattr(message, "first_media", None)
        attached = first_media() if callable(first_media) \
            else None
        if attached is not None and getattr(attached, "path",
                                           ""):
            media = {"kind": "file",
                     "ref": str(attached.path)}
        if low.startswith("poll "):
            parsed = self._channel_poll_parse(raw[5:])
            if isinstance(parsed, str):
                return parsed
            return self._console_channel_enqueue_now(
                surface, key, sender, now, {
                    "kind": "once",
                    "text": parsed["question"], "poll": parsed,
                    "due": now, "source": "dm",
                    "label": f"🗳 {parsed['question']}"[:80]})
        if low.startswith("xpost "):
            parsed = self._channel_xpost_parse(
                surface, key, session, raw[6:])
            if isinstance(parsed, str):
                return parsed
            skey = str((parsed.get("forward") or {}).get(
                "from_channel") or "")
            sources = list(surface.channel_cfg(key).get(
                "xpost_sources") or [])
            if skey and skey not in sources:
                # The pairing is owner-designated by this very
                # act (the console is owner-only): recorded, so
                # the relationship is state, not folklore.
                self._channel_update(
                    surface, key,
                    xpost_sources=sources + [skey])
            return self._console_channel_enqueue_now(
                surface, key, sender, now, parsed)
        if low.startswith("media ") and "|" in raw:
            ref, _, caption = raw[6:].partition("|")
            media = {"kind": "file", "ref": ref.strip()}
            text = caption.strip()
        elif low.startswith("fwd ") and "|" in raw:
            label, _, body = raw[4:].partition("|")
            source = f"forward:{label.strip()}"
            text = body.strip()
        elif low.startswith("daily "):
            parts = raw.split(None, 2)
            if len(parts) < 3 \
                    or parse_hhmm(parts[1]) is None:
                return ("daily needs a time and the words: "
                        "daily HH:MM <text>.")
            kind, at_time, text = "daily", parts[1], parts[2]
            due = surface._channel_next_due(
                {"kind": "daily", "time": at_time}, now, tz)
        elif low.startswith("weekly "):
            parts = raw.split(None, 3)
            days = {"mon": 0, "tue": 1, "wed": 2, "thu": 3,
                    "fri": 4, "sat": 5, "sun": 6}
            if len(parts) < 4 or parts[1].lower()[:3] not in days \
                    or parse_hhmm(parts[2]) is None:
                return ("weekly needs a day, a time and the "
                        "words: weekly mon HH:MM <text>.")
            kind = "weekly"
            weekday = days[parts[1].lower()[:3]]
            at_time, text = parts[2], parts[3]
            due = surface._channel_next_due(
                {"kind": "weekly", "time": at_time,
                 "weekday": weekday}, now, tz)
        elif low.startswith("at "):
            parts = raw.split(None, 2)
            if len(parts) < 3 \
                    or parse_hhmm(parts[1]) is None:
                return ("at needs a time and the words: "
                        "at HH:MM <text>.")
            at_time, text = parts[1], parts[2]
            due = surface._channel_next_due(
                {"kind": "daily", "time": at_time}, now, tz)
        elif low.startswith("now "):
            text = raw[4:]
        elif low == "now":
            text = ""
        if not text and not media:
            return ("nothing to post — give me the words, or "
                    "attach media to the message.")
        entry = surface.channel_enqueue(key, {
            "kind": kind, "text": text, "media": media,
            "due": due, "time": at_time, "weekday": weekday,
            "source": source, "created_by": sender,
        }, now=now)
        if entry is None:
            return ("the queue is full (100 posts) — cancel "
                    "something from the queue section first.")
        posts = self._channel_drive(key, now=now, force=True)
        if any(p.get("entry") == entry["id"] for p in posts):
            return "posted now."
        if kind in ("daily", "weekly"):
            return (f"queued — {kind} at {at_time} channel "
                    "time, first run on its beat.")
        if due > now:
            return (f"queued — it posts at {at_time} channel "
                    "time.")
        if not surface.claim_state(key)["claimed"]:
            return ("queued — it posts the moment this "
                    "channel is claimed (my bot made admin, "
                    "or your designation in settings).")
        return "queued — it rides the next cycle beat."

    # ---- the notice undo (DESIGN §31) ---------------------------
    def _control_undonotice(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """Undo ONE automated action from its notice: the strike
        it recorded is removed, any fine is refunded, and the
        platform action is reversed as far as the platform
        allows (unmute / unban / unrestrict; a kick needs
        nothing — the member can already rejoin; a deleted
        message is stated plainly as unrestorable).  Journaled
        like the action (notice_undo in /glog, naming the
        admin).  Where the platform edits, the notice itself is
        edited in place and this handler answers with silence;
        elsewhere the outcome posts as one follow-up line."""
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "undoing an action is for admins."
        surface = self._group_surface()
        words = str(tail or "").strip().split()
        nid = words[0] if words else ""
        record = surface.get_notice(key, nid) if nid else None
        if record is None:
            return ("I don't have that notice on record — "
                    "notices age out of the registry (the /glog "
                    "journal keeps the history).")
        if record.get("undone"):
            return "already undone — the reversal stands."
        actor = str(message.sender_id or message.sender or "")
        user_id = str(record.get("user_id") or "")
        bits: list[str] = []
        if record.get("strike_ts"):
            removed = surface.remove_strike(
                key, user_id,
                rule=str(record.get("strike_rule")
                         or record.get("rule") or ""),
                ts=float(record["strike_ts"]))
            if removed is not None:
                bits.append("the strike is removed")
        fine = int(record.get("fine") or 0)
        if fine:
            if surface.refund_fine(key, user_id, fine, by=actor):
                bits.append(f"the {fine}-spark fine is refunded")
            else:
                bits.append("the fine could not be refunded "
                            "(see /glog)")
        action = str(record.get("action") or "")
        extra = record.get("extra") or {}
        if action == "mute":
            ok = self._mod_run(
                surface, key, message, "unmute", user_id,
                self._mod_unrestrict_call(message, user_id))
            bits.append("the mute is lifted" if ok else
                        "the mute could not be lifted (see /glog)")
        elif action == "ban":
            ok = self._mod_run(
                surface, key, message, "unban", user_id,
                self._mod_unban_call(message, user_id))
            bits.append("the ban is lifted" if ok else
                        "the ban could not be lifted (see /glog)")
        elif action == "acceptance_restrict":
            ok = self._mod_run(
                surface, key, message, "unmute", user_id,
                self._mod_unrestrict_call(message, user_id))
            surface.clear_acceptance(key, user_id)
            bits.append("the restriction is lifted" if ok else
                        "the restriction could not be lifted "
                        "(see /glog)")
        elif action == "kick":
            bits.append("a removal needs nothing lifted — they "
                        "can rejoin")
        elif action == "delete":
            bits.append("the deleted message itself can't be "
                        "restored")
        elif action == "drop":
            out = surface.cancel_drop(
                key, str(extra.get("drop_id") or ""), by=actor)
            if out.get("ok"):
                refund = int(out.get("refunded") or 0)
                bits.append("the drop is closed"
                            + (f" — {refund} unclaimed sparks "
                               "refunded" if refund else ""))
            else:
                bits.append("the drop was already closed")
        elif action == "police_jail":
            # §24: an undo reverses the JAIL — the one
            # reversible piece of a police action.  The fine,
            # the vault raid and any bounty are settlements
            # and stand, exactly as the notice's own extra
            # recorded them.
            if surface.jail_clear(key, user_id):
                bits.append("the jail time is lifted")
            else:
                bits.append("the jail time had already run out")
            settled = []
            if extra.get("fine_paid"):
                settled.append(f"the {extra['fine_paid']}-spark "
                               "fine stands")
            if extra.get("raid_paid"):
                settled.append(f"the {extra['raid_paid']}-spark "
                               "vault raid stands")
            if settled:
                bits.append("; ".join(settled)
                            + " (settlements aren't reversible)")
        elif action == "police":
            bits.append("a police fine, raid or bounty is a "
                        "settlement — it stands; only jail time "
                        "can be undone, from its own notice")
        outcome = "; ".join(bits) if bits else "the record is reversed"
        surface.mark_notice_undone(key, nid, by=actor)
        if self._edit_notice_in_place(surface, key, message,
                                      record, outcome):
            return ""
        mention = str(record.get("mention") or user_id)
        return f"↩️ {mention} [{user_id}] — undone: {outcome}."

    def _mod_unrestrict_call(self, message, user_id):  # type: ignore[no-untyped-def]
        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_unrestrict", None)
            if fn is None:
                raise RuntimeError("this platform path cannot "
                                   "lift a restriction")
            return fn(message.chat, user_id)

        return call

    def _mod_unban_call(self, message, user_id):  # type: ignore[no-untyped-def]
        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_unban", None)
            if fn is None:
                raise RuntimeError("this platform path cannot "
                                   "unban")
            return fn(message.chat, user_id)

        return call

    def _edit_notice_in_place(self, surface, key, message,  # type: ignore[no-untyped-def]
                              record, outcome: str) -> bool:
        """Edit the posted notice to show its reversal, where
        the adapter supports editing and the posted message's id
        was captured.  Honest False everywhere else — the caller
        posts the follow-up line instead."""
        from ...social.chat.group_notices import undone_text

        message_id = str(record.get("message_id") or "")
        if not message_id:
            return False
        adapter = self._group_adapter(message.chat.platform)
        edit = getattr(adapter, "edit_message", None) \
            if adapter else None
        if edit is None:
            return False
        try:
            out = edit(message.chat, message_id,
                       undone_text(record, outcome), buttons=[])
        except Exception:  # noqa: BLE001 - fall back to a send
            return False
        return out is not None

    # ---- join acceptance commands (DESIGN §28) --------------------
    def _acceptance_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the acceptance settings block through the
        store's own settings machinery (defaults-merged,
        persisted) — the _defense_update shape."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("acceptance") or {})
        block.update(fields)
        surface.update_settings(key, acceptance=block)
        return block

    def _control_recheck(self, message) -> str:  # type: ignore[no-untyped-def]
        """“Check me again” (§28): the restricted member's own
        button/command.  Re-runs the criteria live; a pass
        releases them on the spot and the welcome finally
        fires."""
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group thing — tap it in the group."
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        if not user_id:
            return "I couldn't tell who tapped that."
        adapter = self._group_adapter(message.chat.platform)
        acc_cfg = (surface.store.settings(key)
                   .get("acceptance") or {})
        has_photo = None
        if (acc_cfg.get("criteria") or {}).get("photo"):
            has_photo = self._acceptance_photo(
                adapter, message.chat, user_id)
        res = surface.acceptance_recheck(
            key, user_id,
            username=str(message.sender_username or ""),
            name=str(message.sender or ""), has_photo=has_photo)
        if res.get("state") == "not_restricted":
            return "you're not on the acceptance list — nothing to check."
        if res.get("state") == "passed":
            if res.get("was_restricted"):
                self._execute_group_action(
                    surface, key, adapter, message,
                    {"kind": "acceptance_unrestrict",
                     "user_id": user_id, "rule": "acceptance"})
            if res.get("welcome"):
                member = surface.store.group(key)["members"].get(
                    user_id) or {}
                text = surface.welcome_text_for(
                    key, name=str(member.get("name") or ""),
                    username=str(member.get("username") or ""),
                    group_name=str(message.chat.title or ""))
                if text:
                    return text
            return "✅ all checks pass — you're in. Welcome!"
        from ...social.chat.group_acceptance import fix_it_text
        from ...social.chat.typography import Card

        card = Card("moderation", title="Not quite yet", icon="🚪")
        card.lines(fix_it_text(res.get("reasons") or []).splitlines())
        return self._card(message, card,
                          [[("✅ Check me again", "recheck")]])

    def _joinq_label(self, surface, key, record) -> str:  # type: ignore[no-untyped-def]
        return (record.get("name")
                or (f"@{record['username']}"
                    if record.get("username") else "")
                or surface._member_label(
                    key, str(record.get("user_id") or ""))
                or "someone")

    def _joinq_ask(self, surface, key, adapter, message,  # type: ignore[no-untyped-def]
                   req_id, text, *, by="") -> str:
        """The §16 DM window: write to a waiting applicant
        BEFORE deciding.  Telegram opens the window for 5
        minutes from the request, and only if no other admin
        contacted them first — outside it the platform refuses
        and the refusal is the answer.  WhatsApp requesters
        are reachable at their own chat while the request
        stands.  The ask is journaled either way; the request
        itself stays queued."""
        record = None
        for row in surface.acceptance_queue(key):
            if str(row.get("id") or "") == str(req_id):
                record = row
                break
        if record is None:
            return ("that request isn't waiting anymore — "
                    "/joinq shows what's live.")
        label = self._joinq_label(surface, key, record)
        platform = str(message.chat.platform or "")
        sent, why = False, ""
        if platform.startswith("telegram"):
            chat_id = str(record.get("user_chat_id") or "")
            verb = getattr(adapter, "admin_dm_applicant", None)
            if not chat_id or not callable(verb):
                why = ("this request never opened a DM window "
                       "(it arrived without one)")
            elif time.time() - float(record.get("ts") or 0) \
                    > 300:
                why = ("the 5-minute window has closed — "
                       "Telegram only lets the bot write first "
                       "within 5 minutes of the request")
            else:
                out = verb(chat_id, str(text))
                sent = isinstance(out, dict) \
                    and bool(out.get("ok"))
                if not sent:
                    why = str((out or {}).get("error")
                              or "the platform refused")
        elif platform == "whatsapp":
            from ...social.chat.base import ChatKind, ChatRef

            dm = ChatRef(platform=platform,
                         chat_id=str(record.get("user_id")
                                     or ""),
                         kind=ChatKind.DM)
            try:
                result = self._send(dm, str(text))
                sent = bool(getattr(result, "ok", False))
                if not sent:
                    why = str(getattr(result, "error", "")
                              or "the send didn't land")
            except Exception as exc:  # noqa: BLE001 - reported
                why = f"{type(exc).__name__}: {exc}"
        else:
            why = "this platform has no applicant window"
        surface._audit(key, "join_request_asked",
                       request=str(req_id), by=by, ok=sent,
                       reason=why)
        surface.store.save()
        if sent:
            return (f"✉️ sent to {label} — they're still in "
                    "the queue; decide when they answer.")
        return (f"couldn't reach {label}: {why}. The request "
                "stays queued either way.")

    def _joinq_execute(self, surface, key, adapter, message,  # type: ignore[no-untyped-def]
                       decided: dict, *, approve: bool) -> str:
        """Perform one decided queue record; returns the outcome
        line for the admin.  Execution failures are audited by
        the executors themselves and reported here plainly."""
        record = decided.get("record") or {}
        user_id = str(record.get("user_id") or "")
        label = self._joinq_label(surface, key, record)
        execute = decided.get("execute")
        if execute == "join_request":
            done = self._acceptance_join_request(
                surface, key, adapter, message.chat, user_id,
                approve=approve)
            if not done:
                door = ("WhatsApp" if str(
                    message.chat.platform or "") == "whatsapp"
                    else "Telegram")
                return (f"{label}: {door} wouldn't carry that "
                        f"out — the request may have expired or "
                        f"been decided already (see /glog).")
            if not approve:
                return f"🚫 {label}'s request declined."
            admitted = surface.acceptance_request_admitted(
                key, str(record.get("id") or ""))
            if admitted.get("restricted"):
                if str(message.chat.platform or "") == "whatsapp":
                    # WhatsApp has no restricted-admit state:
                    # the approval admitted them FULLY.  No
                    # restrict is performed and no fix-it note
                    # is sent — the unmet checks live in /glog,
                    # and the admin who approved sees them here.
                    return (
                        f"✅ {label} approved — they're in. "
                        f"WhatsApp has no restricted state, so "
                        f"they're a full member; the unmet "
                        f"checks stand in /glog: "
                        f"{'; '.join(admitted.get('reasons') or [])}")
                self._execute_group_action(
                    surface, key, adapter, message,
                    {"kind": "acceptance_restrict",
                     "user_id": user_id, "rule": "acceptance"})
                self._send_acceptance_note(
                    surface, key, message, label,
                    admitted.get("reasons") or [],
                    grace_minutes=admitted.get("grace_minutes"))
                return (f"✅ {label} approved — in, restricted "
                        f"until they fix: "
                        f"{'; '.join(admitted.get('reasons') or [])}")
            if admitted.get("welcome"):
                text = surface.welcome_text_for(
                    key, name=str(record.get("name") or ""),
                    username=str(record.get("username") or ""),
                    group_name=str(message.chat.title or ""))
                if text and self._system_send_ok(
                        surface, key, message, "welcome"):
                    self._send(
                        message.chat, text,
                        buttons=self._welcome_buttons(surface, key),
                        parse_mode=self._rendered_mode(message,
                                                       text))
            return f"✅ {label} approved — they're in."
        if execute == "release":
            done = self._execute_group_action(
                surface, key, adapter, message,
                {"kind": "acceptance_unrestrict",
                 "user_id": user_id, "rule": "acceptance"})
            if record.get("welcome"):
                text = surface.welcome_text_for(
                    key, name=str(record.get("name") or ""),
                    username=str(record.get("username") or ""),
                    group_name=str(message.chat.title or ""))
                if text and self._system_send_ok(
                        surface, key, message, "welcome"):
                    self._send(
                        message.chat, text,
                        buttons=self._welcome_buttons(surface, key),
                        parse_mode=self._rendered_mode(message,
                                                       text))
            return (f"✅ {label} released — full member."
                    if done else
                    f"{label} released on the books — the "
                    f"platform didn't lift the restriction "
                    f"(see /glog).")
        if execute == "kick":
            done = self._execute_group_action(
                surface, key, adapter, message,
                {"kind": "kick", "user_id": user_id,
                 "rule": "acceptance"})
            return (f"🚫 {label} removed from the queue."
                    if done else
                    f"{label}: the removal didn't go through "
                    f"(see /glog).")
        return f"{label}: nothing to do."

    def _control_joinq(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """The admin join queue (§28): who's waiting, what their
        evaluation said, and the decisions — typed or tapped."""
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        adapter = self._group_adapter(message.chat.platform)
        text = str(tail or "").strip()
        parts = text.split()
        by = str(message.sender_id or message.sender or "")
        if parts and parts[0].lower() in ("approve", "decline") \
                and len(parts) >= 2:
            approve = parts[0].lower() == "approve"
            if parts[1].lower() == "all":
                lines = []
                for record in surface.acceptance_queue(key):
                    decided = surface.acceptance_decide(
                        key, str(record.get("id") or ""),
                        approve=approve, by=by)
                    if decided.get("ok"):
                        lines.append(self._joinq_execute(
                            surface, key, adapter, message,
                            decided, approve=approve))
                if not lines:
                    return "the join queue is empty."
                return "\n".join(lines)
            decided = surface.acceptance_decide(
                key, parts[1], approve=approve, by=by)
            if not decided.get("ok"):
                return ("that request isn't waiting anymore — "
                        "/joinq shows what's live.")
            return self._joinq_execute(
                surface, key, adapter, message, decided,
                approve=approve)
        if parts and parts[0].lower() == "ask":
            if len(parts) < 3:
                return ("usage: /joinq ask <id> <message> — "
                        "write to the applicant while their "
                        "window is open; the id is on the "
                        "queue list.")
            return self._joinq_ask(
                surface, key, adapter, message, parts[1],
                " ".join(parts[2:]), by=by)
        pending = surface.acceptance_queue(key)
        if not pending:
            return ("the join queue is empty — nobody's waiting "
                    "on a decision.")
        from ...social.chat.typography import Card

        now = time.time()
        card = Card("moderation", title="Join queue", icon="🚪")
        lines = []
        buttons = []
        for record in pending:
            label = self._joinq_label(surface, key, record)
            waited = max(0, int((now - float(
                record.get("ts") or now)) // 60))
            detail = "; ".join(record.get("reasons") or [])
            if not detail and record.get("unknown"):
                detail = ("couldn't verify: "
                          + ", ".join(record["unknown"]))
            if not detail:
                detail = ("meets every rule"
                          if record.get("kind") == "request"
                          else "held by the raid posture")
            lines.append(f"{label} — {detail} "
                         f"· waiting {waited}m")
            rid = str(record.get("id") or "")
            buttons.append(
                [(f"✅ {label}"[:24], f"joinq approve {rid}"),
                 (f"🚫 {label}"[:24], f"joinq decline {rid}")])
        if str(message.chat.platform or "") == "whatsapp":
            lines.append(
                "WhatsApp admits fully or not at all — there "
                "is no restricted state here, so every decision "
                "is approve or decline.")
        card.lines(lines)
        card.note("requests stay open until decided or the "
                  "invite link is revoked")
        return self._card(message, card, buttons)

    # ---- cross-group reputation (DESIGN §32) -----------------------

    def _flag_who(self, surface, key, prop):  # type: ignore[no-untyped-def]
        """The admin-facing label for a proposal's person:
        name/@handle from the ledger + the numeric id."""
        user_id = str(prop.get("user_id") or "")
        member = surface.member(key, user_id) or {}
        name = str(member.get("name") or "")
        uname = str(member.get("username") or "")
        label = name or (f"@{uname}" if uname else user_id)
        return f"{label} [{user_id}]"

    def _flag_proposal_card(self, surface, key, message, prop,  # type: ignore[no-untyped-def]
                            label=""):
        """One staged flag, shown for its human decision — the
        confirm tap is what creates a record; until then the
        proposal brands nobody."""
        from ...social.chat.group_reputation import CATEGORIES
        from ...social.chat.typography import Card

        card = Card("moderation", title="Flag proposal",
                    icon="🚩")
        who = label or self._flag_who(surface, key, prop)
        if "[" not in who:
            who = f"{who} [{prop.get('user_id')}]"
        category = str(prop.get("category") or "other")
        cat_label = (CATEGORIES.get(category) or {}).get(
            "label", category)
        lines = [f"{who}", f"category: **{cat_label}**"]
        if prop.get("note"):
            lines.append(f"note: {prop['note']}")
        origin = ("an admin's /flag" if prop.get("direct")
                  else f"proposed by {prop.get('source') or 'a report'}")
        lines.append(f"{origin} · proposed by "
                     f"{prop.get('by_label') or prop.get('by')}")
        lines.append("nothing is recorded until an admin "
                     "confirms — unconfirmed proposals expire "
                     "in 3 days and brand nobody")
        card.lines(lines)
        fid = str(prop.get("id") or "")
        buttons = [[("✅ Confirm", f"flag ok {fid}"),
                    ("🚫 Discard", f"flag no {fid}")]]
        return self._card(message, card, buttons)

    def _control_flag(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """Cross-group reputation flags (§32): an admin flags a
        member HERE; the record is born only on the confirm tap
        and then follows the person — ids + username history —
        into every group Devon serves.  `retract` is the source
        group's undo: it clears records this group sourced."""
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        text = str(tail or "").strip()
        parts = text.split()
        low = [p.lower() for p in parts]
        by = str(message.sender_id or message.sender or "")
        by_label = str(message.sender or "")
        if len(parts) >= 2 and low[0] in (
                "ok", "confirm", "no", "discard"):
            approve = low[0] in ("ok", "confirm")
            out = surface.decide_flag(
                key, parts[1], approve=approve, by=by,
                by_label=by_label)
            if not out.get("ok"):
                return ("that proposal isn't waiting anymore — "
                        "/flag shows what's live.")
            prop = out.get("proposal") or {}
            who = self._flag_who(surface, key, prop)
            if not approve:
                return (f"discarded — no record for {who}. "
                        f"Proposals brand nobody until confirmed.")
            record = out.get("record") or {}
            from ...social.chat.group_reputation import \
                CATEGORIES

            cat_label = (CATEGORIES.get(
                record.get("category") or "") or {}).get(
                    "label", record.get("category") or "")
            return (f"🚩 recorded: {who} — **{cat_label}** "
                    f"[{record.get('id')}]. Every group I serve "
                    f"will see this record when they join; it "
                    f"never decides admission by itself. Retract "
                    f"it here any time with /flag retract.")
        if not parts or low[0] == "queue":
            pending = surface.flag_proposals(key)
            if not pending:
                return ("no flag proposals waiting here. "
                        "/flag @member <category> [note] stages "
                        "one; reports and sensor bans propose "
                        "their own.")
            from ...social.chat.typography import Card

            card = Card("moderation", title="Flag proposals",
                        icon="🚩")
            lines = []
            buttons = []
            for prop in pending:
                who = self._flag_who(surface, key, prop)
                note = str(prop.get("note") or "")[:80]
                lines.append(
                    f"{who} — {prop.get('category')} · {note} "
                    f"· by {prop.get('by_label') or prop.get('by')}")
                fid = str(prop.get("id") or "")
                buttons.append(
                    [(f"✅ {who}"[:24], f"flag ok {fid}"),
                     (f"🚫 {who}"[:24], f"flag no {fid}")])
            card.lines(lines)
            card.note("confirming creates the record — it then "
                      "follows the person across every group; "
                      "unconfirmed proposals expire in 3 days")
            return self._card(message, card, buttons)
        if low[0] == "retract":
            rest = text.split(None, 1)[1] if len(parts) > 1 \
                else ""
            res = self._mod_target(surface, key, rest, message)
            if not res.found:
                msg = self._mod_no_target(res, "retract")
                return msg or (
                    "usage: /flag retract @member — clears the "
                    "records THIS group sourced for them.")
            gone = surface.reputation_retract_for(
                key, res.member.user_id, by=by,
                by_label=by_label,
                any_source=self._operator_or_false(message))
            if not gone:
                return (f"no live record from this group for "
                        f"{res.member.label} — nothing to "
                        f"retract. (Another group's record is "
                        f"theirs to retract.)")
            cats = ", ".join(sorted(
                {str(r.get('category') or '') for r in gone}))
            return (f"retracted — {res.member.label}'s record "
                    f"({cats}) is cleared everywhere; it stops "
                    f"warning from now. Past warnings aren't "
                    f"un-sent.")
        res = self._mod_target(surface, key, text, message)
        if not res.found:
            msg = self._mod_no_target(res, "flag")
            return msg or (
                "usage: /flag @member <category> [note] — "
                "categories: scam · fraud · raid · harassment · "
                "spam-ring · other. Reply to their message, or "
                "name them.")
        from ...social.chat.group_reputation import CATEGORIES

        rest = str(res.rest or "").split(None, 1)
        if not rest or rest[0].lower() not in CATEGORIES:
            return (f"which kind? /flag {res.member.label} "
                    f"<category> [note] — categories: scam · "
                    f"fraud · raid · harassment · spam-ring · "
                    f"other.")
        category = rest[0].lower()
        note = rest[1] if len(rest) > 1 else ""
        person = surface.reputation_person(
            key, res.member.user_id,
            username=res.member.username, name=res.member.name)
        prop = surface.stage_flag(
            key, user_id=res.member.user_id, person=person,
            category=category, note=note, by=by,
            by_label=by_label, direct=True, source="command")
        if prop.get("duplicate"):
            return (f"already waiting: a {category} flag for "
                    f"{res.member.label} is in the queue — "
                    f"/flag reviews it.")
        return self._flag_proposal_card(
            surface, key, message, prop, res.member.label)

    def _control_rep(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """The admin read of the reputation book (§32): one
        person's whole file — live, retracted and aged records
        with their sources, confirmers and journals."""
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        text = str(tail or "").strip()
        if not text:
            stats = surface.reputation.stats()
            return (f"the reputation book: {stats['live']} live "
                    f"records ({stats['total']} all-time) across "
                    f"every group I serve. /rep @member shows one "
                    f"person's file; /flag starts one.")
        res = self._mod_target(surface, key, text, message)
        if res.found:
            view = surface.reputation_view(
                key, res.member.user_id,
                username=res.member.username,
                name=res.member.name)
            records = view["records"]
            who = f"{res.member.label} [{res.member.user_id}]"
        else:
            # One more path (§29's spirit): the BOOK itself.  A
            # person who has never joined THIS group can still
            # have a file — probe it by handle, or by id on this
            # platform, before answering "not found".
            token = text.split()[0].lstrip("@").lower()
            platform = key.split(":", 1)[0]
            probe = {"ids": {}, "usernames": [token], "names": []}
            if token.isdigit():
                probe = {"ids": {platform: token},
                         "usernames": [], "names": []}
            records = surface.reputation.all_for(probe)
            if not records:
                msg = self._mod_no_target(res, "look up")
                return msg or "usage: /rep @member"
            person = records[0].get("person") or {}
            names = person.get("names") or []
            handles = person.get("usernames") or []
            ids = person.get("ids") or {}
            label = (names[0] if names else
                     f"@{handles[0]}" if handles else
                     next(iter(ids.values()), token))
            who = f"{label} [{next(iter(ids.values()), token)}]"
        if not records:
            return f"{who} — no record on file, anywhere."
        from ...social.chat.group_reputation import CATEGORIES
        from ...social.chat.typography import Card

        card = Card("moderation",
                    title=f"Reputation — {who.split(' [')[0]}",
                    icon="🗂️")
        lines = []
        for record in records:
            status = str(record.get("status") or "live")
            mark = {"live": "⚠️", "retracted": "↩️",
                    "aged": "🕰️"}.get(status, "•")
            category = str(record.get("category") or "other")
            cat_label = (CATEGORIES.get(category) or {}).get(
                "label", category)
            lines.append(
                f"{mark} **{cat_label}** [{record['id']}] · "
                f"{status} · from "
                f"{record.get('source_group')} · "
                f"{_fmt_ts(record.get('created_ts'))} · "
                f"confirmed by "
                f"{record.get('confirmed_by_label') or record.get('confirmed_by')}")
            if record.get("note"):
                lines.append(f"   note: {record['note']}")
            events = record.get("events") or []
            warned = {e.get("group") for e in events
                      if e.get("kind") == "warned"}
            bits = []
            if warned:
                bits.append(f"warned in {len(warned)} group"
                            f"{'s' if len(warned) != 1 else ''}")
            if any(e.get("kind") == "dismissed" for e in events):
                bits.append("dismissed somewhere")
            if status == "live":
                bits.append("expires "
                            + _fmt_ts(record.get("expires_ts")))
            if bits:
                lines.append("   " + " · ".join(bits))
        card.lines(lines)
        card.note("retraction belongs to the source group: "
                  "/flag retract there")
        return self._card(message, card)

    def _control_repclear(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """Clear on a restricted-mode warning (§32): lift the
        reputation hold on a member in THIS group.  The record
        itself stands in the book."""
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        words = str(tail or "").strip().split()
        if not words:
            return ("usage: /repclear <member id> — or tap "
                    "Clear on the warning notice.")
        target = words[0]
        out = surface.reputation_clear(
            key, target,
            by=str(message.sender_id or message.sender or ""),
            by_label=str(message.sender or ""))
        if not out.get("cleared"):
            return ("that member isn't held by a reputation "
                    "record here.")
        adapter = self._group_adapter(message.chat.platform)
        self._execute_group_action(
            surface, key, adapter, message,
            {"kind": "acceptance_unrestrict", "user_id": target,
             "rule": "reputation"})
        member = surface.member(key, target) or {}
        label = str(member.get("name") or "") \
            or (f"@{member['username']}"
                if member.get("username") else target)
        return (f"cleared — {label} [{target}] is free of the "
                f"reputation hold here. The record itself stands "
                f"in the book.")

    def _control_repdismiss(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """Dismiss on a warning (§32): this group stops hearing
        about that record; the record stands globally.  A hold
        the record is causing here lifts with the dismissal —
        the group has made its call."""
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        words = str(tail or "").strip().split()
        if not words or not words[0].startswith("rep-"):
            return ("usage: /repdismiss <record id> — or tap "
                    "Dismiss on the warning notice.")
        record_id = words[0]
        by = str(message.sender_id or message.sender or "")
        out = surface.reputation_dismiss(
            key, record_id, by=by,
            by_label=str(message.sender or ""))
        if not out.get("ok"):
            return "no such record in the book."
        adapter = self._group_adapter(message.chat.platform)
        released = 0
        for uid, row in surface.members(key).items():
            state = (row or {}).get("reputation") or {}
            if state.get("state") == "restricted" \
                    and state.get("record_id") == record_id:
                surface.reputation_clear(
                    key, uid, by=by,
                    by_label=str(message.sender or ""))
                self._execute_group_action(
                    surface, key, adapter, message,
                    {"kind": "acceptance_unrestrict",
                     "user_id": uid, "rule": "reputation"})
                released += 1
        tail_note = (f" The hold it had here is lifted too."
                     if released else "")
        return (f"dismissed for this group — record "
                f"[{record_id}] stands in the book and still "
                f"warns elsewhere.{tail_note}")

    # ---- manual moderation (DESIGN §4: the admin command set) ------
    def _mod_gate(self, message):  # type: ignore[no-untyped-def]
        """(key, None) for an admin in a group, else (None, text)."""
        key = self._group_key_or_none(message)
        if key is None:
            return None, "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return None, "that's an admin call — I'm not taking it from a non-admin."
        return key, None

    def _resolver(self, surface):  # type: ignore[no-untyped-def]
        """The member-resolution organ (DESIGN §29): every
        path Devon has — reply, mention, ledger, bot admin
        surface, the userbot — behind one call."""
        from ...social.chat.member_resolution import MemberResolver

        return MemberResolver(surface, self._group_adapter)

    def _mod_target(self, surface, key, tail, message):  # type: ignore[no-untyped-def]
        """Resolve who a moderation command acts on (§29):
        the AUTHOR of the replied-to message first (platform
        report, ledger message map, then the userbot fetching
        the message itself — the path that reaches pre-
        deployment messages), else the named token by every
        path.  Returns the organ's Resolution."""
        return self._resolver(surface).resolve_command(
            key, message, str(tail or ""))

    def _mod_no_target(self, res, verb: str) -> str:  # type: ignore[no-untyped-def]
        """The honest failure text when a moderation command
        named someone the resolver could not land (§29): a
        channel post is its own fact with its own remedy;
        a named-but-missing member gets ONE line naming what
        was tried — never the generic who-prompt.  "" means
        nothing was named at all (caller shows usage)."""
        if res.channel_post:
            return (f"that message was posted as "
                    f"{res.channel_label} — there's no member "
                    f"to {verb}. Reply to it with /purge to "
                    f"delete it, and /gsettings antichannel "
                    f"keeps channel posts out.")
        if res.named_token:
            return res.exhaustion_text()
        return ""

    def _mod_run(self, surface, key, message, action, user_id,  # type: ignore[no-untyped-def]
                 call) -> bool:
        """Execute one adapter admin call; audit either way; a
        platform refusal (ok=False dict or exception) is a failed
        action, never a reported success."""
        adapter = self._group_adapter(message.chat.platform)
        actor = str(message.sender_id or message.sender or "")
        try:
            if adapter is None:
                raise RuntimeError("no adapter for this platform")
            result = call(adapter)
            if isinstance(result, dict) \
                    and result.get("ok") is False:
                raise RuntimeError(str(result.get("error")
                                       or "platform refused"))
        except Exception as exc:  # noqa: BLE001 - audited, not raised
            surface._audit(key, "admin_action_failed", action=action,
                           user_id=user_id, by=actor,
                           reason=f"{type(exc).__name__}: {exc}")
            surface.store.save()
            return False
        surface._audit(key, "admin_action", action=action,
                       user_id=user_id, by=actor)
        surface.store.save()
        return True

    def _mod_mute_call(self, message, user_id, seconds):  # type: ignore[no-untyped-def]
        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_restrict", None)
            if fn is None:
                raise RuntimeError("this platform path cannot mute")
            if seconds and seconds > 0:
                try:
                    return fn(message.chat, user_id,
                              minutes=max(1, seconds // 60))
                except TypeError:
                    return fn(message.chat, user_id,
                              until=int(time.time()) + seconds)
            try:
                return fn(message.chat, user_id, minutes=0)
            except TypeError:
                return fn(message.chat, user_id, until=0)
        return call

    # ---- member cleanup: deleted accounts + inactivity --------
    # The class the DESIGN gap list named on 2026-10-25: the
    # ledger always KNEW who was silent (the Roster's "silent"
    # bucket) and Telegram always marked its deleted accounts —
    # nothing acted on either.  /cleanup is the scan (a dry run,
    # always), /rmdeleted and /rminactive are the removals, and
    # the analysis itself lives in social/chat/group_cleanup.py.
    def _cleanup_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the cleanup settings block through the store's
        own settings machinery (defaults-merged, persisted)."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("cleanup") or {})
        block.update(fields)
        surface.update_settings(key, cleanup=block)
        return block

    # ---- §16 P5 layer: settings grammar + scope sync ----------

    def _p5_update(self, surface, key, **fields):  # type: ignore[no-untyped-def]
        """Mutate the p5 settings block through the store's own
        settings machinery (the _cleanup_update shape)."""
        import copy as _copy

        current = surface.store.settings(key)
        block = _copy.deepcopy(current.get("p5") or {})
        block.update(fields)
        surface.update_settings(key, p5=block)
        return block

    #: The command set a scope sync publishes: the room's
    #: admin surface (settings, moderation, the §16 kinds) —
    #: what BotCommandScopeChatAdministrators / ChatMember
    #: exist to show the people who run the room.
    _P5_SCOPE_KINDS = (
        "gsettings", "glog", "member", "rules", "strikes",
        "recheck", "joinq", "console", "undonotice", "flag",
        "rep", "mute", "unmute", "ban", "unban", "kick", "warn",
        "pin", "unpin", "purge", "report", "cleanup",
        "rmdeleted", "rminactive", "tag", "bansender",
        "unbansender", "reactwipe", "boosters", "grant", "tax",
        "treasury")

    def _p5_scope_commands(self):  # type: ignore[no-untyped-def]
        """(name, description) pairs for a scope publish, from
        the catalog's own words — the first clause of each
        kind's "what", capped to the API's description size."""
        from ...social.chat.control import COMMAND_DETAILS

        out = []
        for kind in self._P5_SCOPE_KINDS:
            detail = COMMAND_DETAILS.get(kind) or {}
            what = str(detail.get("what") or kind)
            desc = what.split(":")[0].split(".")[0].strip()
            out.append((kind, (desc or kind)[:100]))
        return out

    def _p5_scope_sync(self, surface, key, *, chat, platform,  # type: ignore[no-untyped-def]
                       clear=False):
        """Apply (or clear) this room's Telegram command
        scopes on the BOT surface: the administrators scope
        plus one chat-member scope per admin the platform
        lists.  Returns (ok_count, total, note) — partial
        application is reported as partial, never dressed as
        a full sync."""
        adapter = self._group_adapter(platform)
        if adapter is None:
            return 0, 0, "no adapter for this platform"
        set_admin = getattr(adapter, "admin_set_admin_commands",
                            None)
        if not callable(set_admin) and not clear:
            return 0, 0, ("command scopes belong to the bot "
                          "surface — this surface has no bot "
                          "menu to scope")
        admins_out = getattr(adapter, "admin_administrators",
                             None)
        admins = []
        if callable(admins_out):
            try:
                res = admins_out(chat)
            except Exception:  # noqa: BLE001 - reported below
                res = None
            if isinstance(res, dict) and res.get("ok"):
                admins = [str(a.get("id") or "")
                          for a in res.get("administrators") or []
                          if str(a.get("id") or "")]
        commands = self._p5_scope_commands()
        calls = []
        if clear:
            clear_member = getattr(
                adapter, "admin_clear_member_commands", None)
            if callable(clear_member):
                for uid in admins:
                    calls.append(clear_member(chat, uid))
            clear_admin = getattr(
                adapter, "admin_clear_admin_commands", None)
            if callable(clear_admin):
                calls.append(clear_admin(chat))
        else:
            if callable(set_admin):
                calls.append(set_admin(chat, commands))
            set_member = getattr(
                adapter, "admin_set_member_commands", None)
            if callable(set_member):
                for uid in admins:
                    calls.append(set_member(chat, uid, commands))
        ok = sum(1 for c in calls
                 if isinstance(c, dict) and c.get("ok"))
        return ok, len(calls), ""

    def _apply_p5_grammar(self, surface, key, words, *,  # type: ignore[no-untyped-def]
                          by="", message=None):
        """The §16 P5 grammar, shared by /gsettings (words
        from the `p5`/`topics`/`scopes`/`ephemeral` head) and
        the console's p5 section.  Forms: `topics on|off` ·
        `ephemeral on|off` · `scopes on|off|sync|clear`.
        Returns the reply line, or None when the words don't
        parse."""
        w = [str(x) for x in words or []]
        low = [x.lower() for x in w]
        if low and low[0] == "p5":
            low = low[1:]
            w = w[1:]
        if not low:
            return None
        head = low[0]
        platform = str(key).partition(":")[0].strip().lower()
        if head == "topics" and len(low) == 2 \
                and low[1] in ("on", "off"):
            on = low[1] == "on"
            self._p5_update(surface, key, topics_rename=on)
            surface._audit(key, "p5_setting", by=by,
                           field="topics_rename", value=on)
            surface.store.save()
            return (f"topic auto-rename {'on' if on else 'off'}"
                    " — auto-named forum topics get a proper "
                    "name from their first message (the bot "
                    "surface flags them; MTProto topics are "
                    "recorded, never auto-renamed).") \
                if on else "topic auto-rename off."
        if head == "ephemeral" and len(low) == 2 \
                and low[1] in ("on", "off"):
            on = low[1] == "on"
            self._p5_update(surface, key, ephemeral=on)
            surface._audit(key, "p5_setting", by=by,
                           field="ephemeral", value=on)
            surface.store.save()
            return (f"ephemeral answers {'on' if on else 'off'}"
                    " — sensitive replies (/balance) go to "
                    "the asker only: in-group ephemeral on the "
                    "bot surface, else their DMs.") if on else \
                "ephemeral answers off — replies post in the " \
                "room as usual."
        if head == "scopes" and len(low) == 2 \
                and low[1] in ("on", "off", "sync", "clear"):
            verb = low[1]
            if verb == "off":
                self._p5_update(surface, key, member_scopes=False)
                surface._audit(key, "p5_setting", by=by,
                               field="member_scopes", value=False)
                surface.store.save()
                return ("admin command scopes off — the "
                        "published scopes stay as they are "
                        "until a `scopes clear`; the setting "
                        "only stops future syncs.")
            if not platform.startswith("telegram-bot"):
                return ("command scopes are the bot surface's "
                        "menu (setMyCommands) — this room's "
                        "surface has no bot menu to scope.")
            chat = getattr(message, "chat", None)
            if chat is None:
                from ...social.chat.base import (
                    ChatKind, ChatRef)
                chat = ChatRef.parse(key, kind=ChatKind.GROUP)
            ok, total, note = self._p5_scope_sync(
                surface, key, chat=chat, platform=platform,
                clear=verb == "clear")
            if total == 0:
                return note or ("scope sync found nothing to "
                                "do — no administrators scope "
                                "verb on this surface.")
            if verb == "clear":
                surface._audit(key, "p5_scopes_cleared", by=by,
                               applied=ok, of=total)
                surface.store.save()
                return (f"member scopes cleared ({ok}/{total} "
                        "calls landed) — members fall back to "
                        "the chat and default scopes.")
            self._p5_update(surface, key, member_scopes=True)
            surface._audit(key, "p5_scopes_synced", by=by,
                           applied=ok, of=total)
            surface.store.save()
            return (f"admin command scopes synced ({ok}/{total} "
                    "calls landed) — this room's admins get "
                    "the full group command menu, per member "
                    "and as a class.")
        return None

    def _apply_cleanup_grammar(self, surface, key, words):  # type: ignore[no-untyped-def]
        """The member-cleanup grammar, shared by /gsettings
        (words after the `cleanup` head) and the console's
        cleanup section.  Forms: `on` · `off` · `auto on|off` ·
        `days N`.  (Distinct from `clean` — the service-message
        cleanup.)"""
        w = [str(x) for x in words or []]
        low = [x.lower() for x in w]
        if not w:
            return None
        if low == ["on"]:
            self._cleanup_update(surface, key, enabled=True)
            return ("member cleanup removals on — /rmdeleted "
                    "and /rminactive may act here.")
        if low == ["off"]:
            self._cleanup_update(surface, key, enabled=False)
            return ("member cleanup removals off — /cleanup "
                    "still scans; nothing is removed.")
        if low[0] == "auto":
            if low == ["auto", "on"]:
                self._cleanup_update(surface, key,
                                     auto_deleted=True)
                return ("auto-remove deleted accounts on — "
                        "the upkeep cycle clears them from "
                        "claimed groups where I hold admin "
                        "power, without waiting for a command.")
            if low == ["auto", "off"]:
                self._cleanup_update(surface, key,
                                     auto_deleted=False)
                return ("auto-remove deleted accounts off — "
                        "deleted accounts are removed only "
                        "when an admin runs /rmdeleted.")
            return None
        if low[0] == "days" and len(w) == 2 and w[1].isdigit():
            days = max(1, min(3650, int(w[1])))
            self._cleanup_update(surface, key,
                                 inactive_days=days)
            return (f"inactivity window: {days} days — the "
                    f"sweep judges members against it.")
        return None

    def _cleanup_gate(self, message):  # type: ignore[no-untyped-def]
        """(key, None) for an admin in a CLAIMED Telegram group,
        else (None, text).  The scan reads Telegram's member
        surfaces, so other platforms get the platform truth; the
        claim requirement is §33's — her group systems act in a
        room only where she is claimed (bot operating there, or
        the owner's designation)."""
        key = self._group_key_or_none(message)
        if key is None:
            return None, ("that's a group command — run it "
                          "inside the group.")
        platform = str(message.chat.platform or "")
        if not platform.startswith("telegram"):
            return None, ("group cleanup is a Telegram tool — "
                          "this platform gives me no member "
                          "list to scan and no deleted-account "
                          "markers to find.")
        if not self._group_admin_ok(message):
            return None, ("that's an admin call — I'm not "
                          "taking it from a non-admin.")
        surface = self._group_surface()
        if not surface.claim_state(key)["claimed"]:
            return None, ("this room isn't switched on for my "
                          "group systems yet — cleanup runs "
                          "where I'm claimed (my bot operating "
                          "here, or the owner's designation: "
                          "/gsettings system output on).")
        return key, None

    def _cleanup_roster(self, surface, key, message):  # type: ignore[no-untyped-def]
        """Assemble the member universe for a cleanup scan, from
        the best enumeration this deployment actually has:

        1. the userbot's FULL member list (MTProto) — every
           member, with Telegram's deleted flags and last-seen;
        2. any adapter's capped member list;
        3. the Bot API path: no member list exists there, so
           the universe is Devon's own ledger, each member
           probed once via getChatMember (which is also where
           a deleted account shows its marker on that surface).

        Returns {rows, source, truncated, member_count,
        protected_ids, self_ids, admins_known}."""
        from ...social.chat.base import ChatRef
        from ...social.chat.group_cleanup import is_deleted_row

        platform = str(message.chat.platform or "")
        chat_id = str(getattr(message.chat, "chat_id", "")
                      or "")
        if not chat_id and ":" in key:
            chat_id = key.split(":", 1)[1]
        title = str(getattr(message.chat, "title", "") or "")

        def ref(plat: str) -> ChatRef:
            return ChatRef(platform=plat, chat_id=chat_id,
                           kind=message.chat.kind, title=title)

        candidates: list = []
        seen_ids: set = set()
        for plat in (platform, "telegram", "telegram-bot",
                     "telegram-personal"):
            adapter = self._group_adapter(plat)
            if adapter is not None and id(adapter) not in seen_ids:
                seen_ids.add(id(adapter))
                candidates.append((plat, adapter))

        protected_ids: set = set()
        admins_known = False
        for plat, adapter in candidates:
            fn = getattr(adapter, "admin_administrators", None)
            if fn is None:
                continue
            try:
                result = fn(ref(plat))
            except Exception:  # noqa: BLE001 - a path, not a need
                result = None
            if isinstance(result, dict) and result.get("ok"):
                admins_known = True
                for row in result.get("administrators") or []:
                    if isinstance(row, dict) and row.get("id"):
                        protected_ids.add(str(row["id"]))
                break

        self_ids: set = set()
        for _plat, adapter in candidates:
            for attr in ("_me_id", "_bot_id"):
                value = getattr(adapter, attr, None)
                if value:
                    self_ids.add(str(value))
            companions = getattr(adapter, "companion_bot_ids",
                                 None)
            if callable(companions):
                try:
                    self_ids.update(
                        str(x) for x in companions())
                except Exception:  # noqa: BLE001 - garnish
                    pass

        def _normalise(row: dict) -> dict:
            return {
                "user_id": str(row.get("id")
                               or row.get("user_id") or ""),
                "name": str(row.get("name") or ""),
                "username": str(row.get("username") or ""),
                "bot": bool(row.get("bot")
                            or row.get("is_bot")),
                "deleted": bool(row.get("deleted")),
                "status_kind": str(row.get("status_kind")
                                   or ""),
                "was_online": row.get("was_online"),
                "last_name": str(row.get("last_name") or ""),
            }

        rows: list | None = None
        source = ""
        truncated = False
        for plat, adapter in candidates:
            fn = getattr(adapter, "admin_members_full", None)
            if fn is None:
                continue
            try:
                result = fn(ref(plat))
            except Exception:  # noqa: BLE001 - degrade a path
                result = None
            if isinstance(result, dict) and result.get("ok"):
                rows = [_normalise(r)
                        for r in result.get("members") or []
                        if isinstance(r, dict)]
                source = "the userbot's full member list"
                truncated = bool(result.get("truncated"))
                break
        if rows is None:
            for plat, adapter in candidates:
                fn = getattr(adapter, "admin_members", None)
                if fn is None:
                    continue
                try:
                    result = fn(ref(plat), limit=500)
                except Exception:  # noqa: BLE001 - degrade
                    result = None
                if isinstance(result, dict) \
                        and result.get("ok"):
                    rows = [_normalise(r)
                            for r in result.get("members") or []
                            if isinstance(r, dict)]
                    source = "the member list (first 500)"
                    truncated = len(rows) >= 500
                    break
        if rows is None:
            # Bot API path: the ledger is the universe; probe
            # each known member once for the deleted marker and
            # for proof they are still in the group at all.
            probe = None
            probe_ref = None
            for plat, adapter in candidates:
                fn = getattr(adapter,
                             "admin_member_profile", None)
                if fn is not None:
                    probe, probe_ref = fn, ref(plat)
                    break
            rows = []
            for uid, led in surface.members(key).items():
                if led.get("left_ts"):
                    continue
                row = {
                    "user_id": uid,
                    "name": str(led.get("name") or ""),
                    "username": str(led.get("username") or ""),
                    "bot": bool(led.get("is_bot")),
                    "deleted": False,
                    "status_kind": "",
                    "was_online": None,
                    "last_name": str(led.get("last_name") or ""),
                }
                if probe is not None:
                    try:
                        prof = probe(probe_ref, uid)
                    except Exception:  # noqa: BLE001 - unprobed
                        prof = None
                    if isinstance(prof, dict) \
                            and prof.get("ok"):
                        if str(prof.get("status") or "").lower() \
                                in ("left", "kicked"):
                            continue  # already out — no ghost
                        user = prof.get("user") or {}
                        row["name"] = " ".join(filter(None, [
                            str(user.get("first_name") or ""),
                            str(user.get("last_name") or ""),
                        ])).strip() or row["name"]
                        row["last_name"] = str(
                            user.get("last_name") or "")
                        row["username"] = str(
                            user.get("username") or "") \
                            or row["username"]
                        row["bot"] = bool(user.get("is_bot"))
                        row["deleted"] = is_deleted_row(row)
                rows.append(row)
            source = ("my member ledger — the Bot API has no "
                      "member list, so this scan covers the "
                      "members I have records of")
        member_count = None
        for plat, adapter in candidates:
            fn = getattr(adapter, "admin_member_count", None)
            if fn is None:
                continue
            try:
                result = fn(ref(plat))
            except Exception:  # noqa: BLE001 - garnish
                result = None
            if isinstance(result, dict) and result.get("ok"):
                member_count = int(result.get("count") or 0)
                break
        return {"rows": rows, "source": source,
                "truncated": truncated,
                "member_count": member_count,
                "protected_ids": protected_ids,
                "self_ids": self_ids,
                "admins_known": admins_known}

    def _cleanup_scan(self, surface, key, message, days):  # type: ignore[no-untyped-def]
        """Run the analysis: roster + ledger → findings."""
        from ...social.chat.group_cleanup import analyze_cleanup

        roster = self._cleanup_roster(surface, key, message)
        findings = analyze_cleanup(
            roster["rows"], surface.members(key),
            inactive_days=days,
            protected_ids=roster["protected_ids"],
            self_ids=roster["self_ids"])
        return findings, roster

    def _control_cleanup(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """The cleanup SCAN — a dry run by construction: it
        enumerates, classifies and reports.  It never calls a
        removal verb; the remove commands are separate kinds so
        a scan can never drift into acting."""
        key, err = self._cleanup_gate(message)
        if err:
            return err
        from ...social.chat.group_cleanup import (
            DEFAULT_INACTIVE_DAYS)
        from ...social.chat.typography import Card

        surface = self._group_surface()
        cfg = surface.cleanup_cfg(key)
        days = int(cfg.get("inactive_days")
                   or DEFAULT_INACTIVE_DAYS)
        findings, roster = self._cleanup_scan(
            surface, key, message, days)
        page = 1
        tail_txt = str(tail or "").strip()
        if tail_txt.isdigit():
            page = max(1, int(tail_txt))
        per_page = 10
        cov = findings["coverage"]

        def _page(rows: list) -> tuple:
            total_pages = max(
                1, -(-len(rows) // per_page))
            use = min(page, total_pages)
            return (rows[(use - 1) * per_page:use * per_page],
                    total_pages)

        card = Card("settings", title="Group cleanup — scan",
                    icon="🧹")
        card.block_lines([
            f"deleted accounts: {len(findings['deleted'])} · "
            f"inactive past {days}d: "
            f"{len(findings['inactive'])} · couldn't judge: "
            f"{len(findings['unknown'])} · left alone: "
            f"{len(findings['protected'])}",
            f"removals are "
            f"{'ON' if cfg.get('enabled', True) else 'OFF'} "
            f"here — /rmdeleted removes the deleted, "
            f"/rminactive the inactive (window "
            f"{days} days; /gsettings cleanup days N changes "
            f"it)",
            f"deleted accounts are also removed automatically "
            f"on the upkeep cycle: "
            f"{'on' if cfg.get('auto_deleted', True) else 'off'}"
            f" (/gsettings cleanup auto on|off)",
        ])
        coverage = [
            f"scanned {cov['scanned']} members from "
            f"{roster['source']}",
        ]
        if roster["truncated"]:
            coverage.append("⚠ the member list hit its cap — "
                            "members past it are not in this "
                            "scan")
        if roster["member_count"]:
            coverage.append(
                f"Telegram counts "
                f"{roster['member_count']} members in this "
                f"group")
        if cov["bots_skipped"]:
            coverage.append(
                f"{cov['bots_skipped']} bot(s) sit outside the "
                f"inactivity sweep — a utility bot is quiet by "
                f"design (a deleted bot still counts as "
                f"deleted)")
        if not roster["admins_known"]:
            coverage.append(
                "the admin list wasn't readable on any "
                "surface — removals re-check each member's "
                "standing before acting")
        card.block_lines(coverage, title="Coverage")
        if findings["deleted"]:
            rows, pages = _page(findings["deleted"])
            card.block_lines(
                [f"{f['label']} — Telegram marks this "
                 f"account deleted" for f in rows],
                title=f"Deleted accounts "
                      f"({len(findings['deleted'])})")
        if findings["inactive"]:
            rows, pages = _page(findings["inactive"])
            lines = []
            for f in rows:
                basis = ("my records" if f["basis"] == "ledger"
                         else "Telegram's last-seen")
                lines.append(
                    f"{f['label']} — last active "
                    f"{f['days_inactive']}d ago ({basis})")
            card.block_lines(
                lines,
                title=f"Inactive past {days} days "
                      f"({len(findings['inactive'])})")
        if findings["unknown"]:
            rows, pages = _page(findings["unknown"])
            card.block_lines(
                [f"{f['label']} — {f['reason']}"
                 for f in rows],
                title=f"Couldn't judge "
                      f"({len(findings['unknown'])}) — never "
                      f"removed on a guess")
        if findings["protected"]:
            card.block_lines(
                [f"{f['label']} — flagged, but an admin / "
                 f"the owner: cleanup never removes them"
                 for f in findings["protected"]],
                title="Left alone")
        total_pages = max(
            1,
            -(-max(len(findings["deleted"]),
                    len(findings["inactive"]),
                    len(findings["unknown"])) // per_page))
        note = ("scan only — nothing was removed. "
                "/rmdeleted removes the deleted · /rminactive "
                "removes the inactive.")
        if total_pages > 1:
            note += (f" Page {min(page, total_pages)} of "
                     f"{total_pages} — /cleanup <n> for more.")
        card.note(note)
        return self._card(message, card)

    def _cleanup_standing(self, adapter, message, user_id):  # type: ignore[no-untyped-def]
        """One removal target's standing, re-checked at removal
        time on the acting surface: "creator" / "administrator"
        / "member" / "left" / "unknown" (a check surface exists
        but could not answer — never remove on a guess), or ""
        when the surface has no member check at all (the
        platform's own power bounds what the call can do)."""
        fn = getattr(adapter, "admin_member", None) or getattr(
            adapter, "admin_member_mtproto", None)
        if fn is None:
            return ""
        try:
            info = fn(message.chat, user_id)
        except Exception:  # noqa: BLE001 - unverifiable
            return "unknown"
        if not isinstance(info, dict) \
                or info.get("ok") is False:
            return "unknown"
        ptype = str(info.get("participant_type") or "")
        if ptype:
            if "Creator" in ptype:
                return "creator"
            if "Admin" in ptype:
                return "administrator"
            return "member"
        status = str(info.get("status")
                     or info.get("role") or "").lower()
        if not status and isinstance(info.get("result"), dict):
            status = str(
                info["result"].get("status") or "").lower()
        return status or "unknown"

    def _cleanup_kick_call(self, message, user_id):  # type: ignore[no-untyped-def]
        """A cleanup removal is a KICK (ban immediately lifted —
        the same primitive as /kick): the member is out, never
        banned; an inactive member who returns is free to
        rejoin, and a deleted account has nothing to rejoin
        with."""
        def call(adapter):  # type: ignore[no-untyped-def]
            ban = getattr(adapter, "admin_ban", None)
            unban = getattr(adapter, "admin_unban", None)
            if ban is None or unban is None:
                raise RuntimeError(
                    "this platform path cannot remove members")
            out = ban(message.chat, user_id)
            if isinstance(out, dict) and out.get("ok") is False:
                return out
            return unban(message.chat, user_id)
        return call

    def _cleanup_act_on(self, surface, key, message, adapter,  # type: ignore[no-untyped-def]
                        finding, which: str) -> tuple:
        """Act on ONE cleanup finding through the admin seam.
        Returns (outcome, why): "removed" / "failed" /
        "skipped" (+ the skip's plain reason).  The standing
        re-check lives here so the commands and the automatic
        upkeep pass share it exactly: admins/owner and
        unverifiable standings are never removed; executions
        journal through _mod_run, refusals-to-act journal as
        cleanup_skip — with the acting identity read from the
        message (the commanding admin, or the upkeep marker)."""
        user_id = finding["user_id"]
        actor = str(message.sender_id or message.sender or "")
        standing = self._cleanup_standing(
            adapter, message, user_id) if adapter else ""
        if standing in ("creator", "administrator", "admin",
                        "owner"):
            surface._audit(
                key, "cleanup_skip", user_id=user_id, by=actor,
                target=which,
                reason="group admin/owner — cleanup never "
                       "removes them")
            surface.store.save()
            return "skipped", "an admin — never removed by cleanup"
        if standing == "unknown":
            surface._audit(
                key, "cleanup_skip", user_id=user_id, by=actor,
                target=which,
                reason="standing unverifiable — not removed "
                       "on a guess")
            surface.store.save()
            return "skipped", "standing couldn't be verified"
        if standing in ("left", "kicked"):
            surface.mark_removed(key, user_id)
            return "skipped", "already out"
        ok = self._mod_run(
            surface, key, message, f"cleanup_{which}", user_id,
            self._cleanup_kick_call(message, user_id))
        if ok:
            surface.mark_removed(key, user_id)
            return "removed", ""
        return "failed", ""

    def _cleanup_remove(self, message, *, which: str,  # type: ignore[no-untyped-def]
                        days_override: int | None = None) -> str:
        """Shared removal engine for /rmdeleted and /rminactive:
        fresh scan → per-target standing re-check → kick through
        the admin seam (_mod_run journals every outcome) → the
        ledger learns the member is out.  One member's failure
        never aborts the rest and never reports as success."""
        key, err = self._cleanup_gate(message)
        if err:
            return err
        from ...social.chat.group_cleanup import (
            DEFAULT_INACTIVE_DAYS)

        surface = self._group_surface()
        cfg = surface.cleanup_cfg(key)
        if not cfg.get("enabled", True):
            from ...social.chat.tgbot_buttons import actionable

            return actionable(
                "member cleanup removals are off in this group. "
                "The scan still runs.",
                [[("🧹 Turn removals back on",
                   "gsettings cleanup on")],
                 [("🔍 Run the scan", "cleanup")]])
        days = days_override or int(
            cfg.get("inactive_days") or DEFAULT_INACTIVE_DAYS)
        findings, _roster = self._cleanup_scan(
            surface, key, message, days)
        targets = findings["deleted" if which == "deleted"
                           else "inactive"]
        noun = "deleted account" if which == "deleted" \
            else "inactive member"
        if not targets:
            return (f"nothing to remove — the scan found no "
                    f"{noun}s"
                    + (f" past {days} days." if which != "deleted"
                       else "."))
        adapter = self._group_adapter(message.chat.platform)
        removed: list = []
        failed: list = []
        skipped: list = []
        for finding in targets:
            outcome, why = self._cleanup_act_on(
                surface, key, message, adapter, finding, which)
            if outcome == "removed":
                removed.append(finding)
            elif outcome == "failed":
                failed.append(finding)
            else:
                skipped.append((finding, why))
        lines = [
            f"🧹 {noun} cleanup: {len(removed)} removed · "
            f"{len(failed)} failed · {len(skipped)} skipped",
        ]
        shown = 0
        for finding in removed:
            if shown < 20:
                lines.append(f"✓ {finding['label']} — removed")
            shown += 1
        if len(removed) > 20:
            lines.append(f"… and {len(removed) - 20} more "
                         f"removed (all in /glog)")
        for finding in failed[:10]:
            lines.append(f"✗ {finding['label']} — the "
                         f"platform refused (in /glog)")
        for finding, why in skipped[:10]:
            lines.append(f"— {finding['label']} skipped: {why}")
        lines.append("kicked, not banned — and every outcome "
                     "is journaled in /glog.")
        return "\n".join(lines)

    #: How often one group's automatic deleted-account pass may
    #: run: a full member enumeration is an upkeep cost, never a
    #: per-message one (the channel analytics sampling sets the
    #: same order of cadence, ≤1/6h).
    _CLEANUP_AUTO_INTERVAL_S = 6 * 3600.0

    def _cleanup_admin_power(self, adapter, message) -> bool:  # type: ignore[no-untyped-def]
        """Does Devon hold member-removal power on this surface
        right now?  Read from the adapter's own rights surface
        where one exists (MTProto admin_my_rights / Bot API
        admin_rights).  Where none exists, True: the platform
        itself bounds what the removal call can do, and a
        refusal journals honestly through _mod_run."""
        rights = getattr(adapter, "admin_my_rights", None)
        if callable(rights):
            try:
                out = rights(message.chat)
            except Exception:  # noqa: BLE001 - unverifiable
                return False
            return bool(isinstance(out, dict)
                        and out.get("is_admin"))
        rights = getattr(adapter, "admin_rights", None)
        if callable(rights):
            try:
                out = rights(message.chat)
            except Exception:  # noqa: BLE001 - unverifiable
                return False
            if not isinstance(out, dict) or not out.get("ok"):
                return False
            return str(out.get("status") or "").lower() in (
                "creator", "administrator") and bool(
                    out.get("can_restrict_members", True))
        return True

    # ---- §16 P5 layer: duty + commands -------------------------

    def _p5_gate(self, message):  # type: ignore[no-untyped-def]
        """(key, None) for an admin in a CLAIMED Telegram
        group, else (None, text).  The §16 kinds are Telegram
        machinery (§33: her group systems act where she is
        claimed); WhatsApp never reaches them (dispatch answers
        with its own platform truth first)."""
        key = self._group_key_or_none(message)
        if key is None:
            return None, ("that's a group command — run it "
                          "inside the group.")
        platform = str(message.chat.platform or "")
        if not platform.startswith("telegram"):
            return None, ("that's a Telegram tool — this "
                          "platform has none of the systems "
                          "underneath it (member tags, channel "
                          "posts, reaction wipes, boosts).")
        if not self._group_admin_ok(message):
            return None, ("that's an admin call — I'm not "
                          "taking it from a non-admin.")
        surface = self._group_surface()
        if not surface.claim_state(key)["claimed"]:
            return None, ("this room isn't switched on for my "
                          "group systems yet — the §16 tools "
                          "run where I'm claimed (my bot "
                          "operating here, or the owner's "
                          "designation).")
        return key, None

    def _p5_topic_duty(self, surface, key, message) -> None:  # type: ignore[no-untyped-def]
        """The forum-topic auto-rename duty (§16): a topic
        Telegram auto-named (is_name_implicit — "likely needs
        to be changed by the bot") gets a proper name derived
        from its first real message.  Gates, mirroring the
        cleanup upkeep: claimed room · the group's p5 duty
        switch · defense autonomy (it is an automatic admin
        act).  Journal-only — a rename is never announced.
        Failure-isolated: the duty closes for the topic either
        way (audited), so one bad rename never retries on
        every message."""
        try:
            meta = message.meta or {}
            thread = str(getattr(message.chat, "thread_id", "")
                         or meta.get("thread_id") or "")
            if not thread:
                return
            if not (surface.store.settings(key).get("p5")
                    or {}).get("topics_rename"):
                return
            record = surface.topic_rename_candidate(key, thread)
            if record is None:
                return
            if not surface.claim_state(key)["claimed"]:
                return
            settings = surface.store.settings(key)
            if not (settings.get("defense") or {}).get(
                    "autonomy", True):
                return
            from ...social.chat.group_p5 import derive_topic_name

            name = derive_topic_name(str(message.text or ""))
            if not name:
                return  # nothing nameable yet — a later
                        # message in the topic may carry it
            adapter = self._group_adapter(message.chat.platform)
            verb = getattr(adapter, "admin_forum_topic_edit",
                           None)
            if not callable(verb):
                surface.topic_mark_renamed(
                    key, thread_id=thread,
                    title=str(record.get("title") or ""))
                surface._audit(key, "topic_rename_unsupported",
                               thread_id=thread)
                surface.store.save()
                return
            out = verb(message.chat, thread, name)
            ok = isinstance(out, dict) and bool(out.get("ok"))
            surface.topic_mark_renamed(
                key, thread_id=thread,
                title=name if ok else str(record.get("title")
                                          or ""))
            surface._audit(
                key, "topic_renamed" if ok
                else "topic_rename_failed",
                thread_id=thread, title=name,
                error="" if ok else str(
                    (out or {}).get("error") or "platform refused"))
            surface.store.save()
        except Exception:  # noqa: BLE001 - upkeep never breaks
            pass                # a group turn

    def _control_tag(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """§16 member tags: /tag @member <words> sets, /tag
        @member off clears, /tag @member shows.  The native
        tag is the BOT surface's setChatMemberTag (behind its
        can_manage_tags right); the ledger record is Devon's
        on every surface, and /member renders it."""
        key, err = self._p5_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "tag")
            if note:
                return note
            return ("who should I tag? /tag @member <words> — "
                    "or reply to their message.")
        user_id, label = res.member.user_id, res.member.label
        words = str(res.rest or "").strip()
        member = surface.member(key, user_id) or {}
        if not words:
            current = str(member.get("tag") or "")
            return (f"{label}'s tag: “{current}”." if current
                    else f"{label} has no tag — /tag {label} "
                         "<words> sets one.")
        from ...social.chat.group_p5 import validate_tag

        if words.lower() in ("off", "clear", "none", "-"):
            cleaned, clearing = "", True
        else:
            ok, cleaned, why = validate_tag(words)
            if not ok:
                return f"can't tag that: {why}"
            clearing = False
        adapter = self._group_adapter(message.chat.platform)
        verb = getattr(adapter, "admin_set_member_tag", None)
        native = None  # None: no native verb on this surface
        if callable(verb):
            out = verb(message.chat, user_id, cleaned)
            if isinstance(out, dict) and out.get("ok"):
                native = True
            elif isinstance(out, dict) and "unsupported" in str(
                    out.get("error") or ""):
                native = False  # the ledger carries it here
            else:
                reason = str((out or {}).get("error")
                             or "platform refused")
                surface._audit(key, "admin_action_failed",
                               action="tag", user_id=user_id,
                               by=str(message.sender_id or ""),
                               reason=reason)
                surface.store.save()
                hint = (" — the bot needs its manage-tags "
                        "admin right for native tags."
                        if "tag" in reason.lower() else "")
                return (f"Telegram refused the tag: "
                        f"{reason}{hint}")
        record = surface.member_set_tag(
            key, user_id, tag=cleaned,
            by=str(message.sender_id or ""))
        if not record.get("ok"):
            return (f"{label} isn't in this group's ledger "
                    "yet — tags decorate the roster, they "
                    "don't create it.")
        if clearing:
            text = f"🏷️ {label}'s tag is cleared."
        else:
            text = f"🏷️ {label} is tagged “{cleaned}”."
        if native is False:
            text += (" — in my member book; the native "
                     "in-group tag is set by my bot surface "
                     "(it needs the manage-tags right).")
        return text

    def _control_bansender(self, tail, message, *, unban=False) -> str:  # type: ignore[no-untyped-def]
        """§16 sender-chat ban: ban the CHANNEL identity
        behind a channel post — never the person who owns it.
        Target: the replied-to post's sender chat, or a
        numeric channel id named outright."""
        key, err = self._p5_gate(message)
        if err:
            return err
        surface = self._group_surface()
        meta = message.meta or {}
        rsc = meta.get("reply_to_sender_chat") or {}
        sender_chat_id = str(rsc.get("id") or "")
        title = str(rsc.get("title") or "")
        token = str(tail or "").strip().split()[0] \
            if str(tail or "").strip() else ""
        if not sender_chat_id and token.lstrip("-").isdigit():
            sender_chat_id = token
        if not sender_chat_id:
            return ("reply to a post made as a channel, then "
                    "/bansender — or name the channel's numeric "
                    "id. (A channel post is its channel's act; "
                    "there's no member to ban here.)")
        label = title or "that channel"
        verb_name = ("admin_unban_sender_chat" if unban
                     else "admin_ban_sender_chat")

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, verb_name, None)
            if fn is None:
                raise RuntimeError(
                    "this platform path cannot "
                    + ("unban" if unban else "ban")
                    + " a sender chat")
            return fn(message.chat, sender_chat_id)

        if not self._mod_run(surface, key, message,
                             "unban_sender_chat" if unban
                             else "ban_sender_chat",
                             sender_chat_id, call):
            return (f"the {'unban' if unban else 'ban'} didn't "
                    "go through — the platform refused it "
                    "(it's in /glog).")
        if unban:
            return f"📣 {label} may post here again."
        return (f"🔇 {label} is banned from posting here — "
                "its owner can't post in this group as any of "
                "their channels while it stands.")

    def _control_reactwipe(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """§16 reaction wipe: deleteAllMessageReactions on
        the bot surface; the userbot answers honestly (its
        stub's refusal is what /glog records)."""
        key, err = self._p5_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "wipe reactions for")
            if note:
                return note
            return ("whose reactions should I wipe? "
                    "/reactwipe @member — or reply to their "
                    "message.")
        user_id, label = res.member.user_id, res.member.label

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_reaction_wipe", None)
            if fn is None:
                raise RuntimeError("this platform path cannot "
                                   "wipe reactions")
            return fn(message.chat, user_id)

        if not self._mod_run(surface, key, message,
                             "reaction_wipe", user_id, call):
            return ("the wipe didn't go through — the "
                    "platform refused it (it's in /glog).")
        return (f"🧹 wiped {label}'s recent reactions here "
                "(Telegram clears up to 10,000).")

    def _control_boosters(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """§16 boost intel: the room's boost board from the
        ledger Telegram's own updates filled, or one member's
        standing plus the bot surface's live per-user lookup."""
        key, err = self._p5_gate(message)
        if err:
            return err
        surface = self._group_surface()
        token = str(tail or "").strip()

        def src_label(kind):  # type: ignore[no-untyped-def]
            return {"premium": "Premium",
                    "gift_code": "gift code",
                    "giveaway": "giveaway"}.get(
                        str(kind or ""), "boost")

        if token:
            res = self._resolver(surface).resolve_token(
                key, message, token)
            if not res.found:
                return res.exhaustion_text()
            user_id, label = res.member.user_id, res.member.label
            summary = surface.boost_summary(key, user_id)
            lines = []
            if summary["count"]:
                mix = " · ".join(
                    f"{src_label(k)} ×{v}" for k, v in
                    sorted(summary["sources"].items()))
                lines.append(f"{label} is boosting this room "
                             f"×{summary['count']} ({mix}).")
            else:
                lines.append(f"{label} has no active boosts "
                             "here that I've seen.")
            adapter = self._group_adapter(message.chat.platform)
            verb = getattr(adapter, "admin_user_boosts", None)
            if callable(verb):
                out = verb(message.chat, user_id)
                if isinstance(out, dict) and out.get("ok"):
                    result = out.get("result") or {}
                    live = result.get("boosts") or []
                    lines.append(
                        f"Telegram's own list: {len(live)} "
                        "boost(s) on record for them here.")
                elif isinstance(out, dict) and "unsupported" \
                        not in str(out.get("error") or ""):
                    lines.append("(Telegram's per-user lookup "
                                 "refused just now — the ledger "
                                 "above is what I've seen.)")
            return "\n".join(lines)
        board = surface.boosters(key)
        rows = board["rows"]
        if not rows and not board["total_active"]:
            return ("no boosts seen here yet — when members "
                    "boost this room, Telegram's updates land "
                    "them on this board.")
        lines = [f"🚀 Boosts here: {board['total_active']} "
                 "active"]
        for row in rows[:10]:
            member = surface.member(key, row["user_id"]) or {}
            name = (str(member.get("name") or "")
                    or (f"@{member['username']}"
                        if member.get("username") else "")
                    or "a member")
            mix = " · ".join(
                f"{src_label(k)} ×{v}" for k, v in
                sorted(row["sources"].items()))
            lines.append(f"· {name} — ×{row['count']} ({mix})")
        if board["unattributed"]:
            lines.append(f"· {board['unattributed']} more "
                         "boost(s) name nobody (unclaimed "
                         "giveaways) — counted for the room.")
        return "\n".join(lines)

    def _cleanup_auto_drive(self, surface, key, message) -> None:  # type: ignore[no-untyped-def]
        """Automatic deleted-account removal (owner order
        2026-11-04): on the group upkeep beat, in a claimed
        Telegram group where Devon holds admin power, accounts
        Telegram marks deleted are kicked WITHOUT waiting for a
        command — dead accounts are upkeep, not a decision.
        Journal-only: nothing is ever posted to the group about
        a removal (no meta-narration); /glog holds every
        outcome.

        Gates, in order: Telegram surface · the group's cleanup
        settings (removals on + auto-remove deleted on — both
        default ON, the owner's posture) · the defense autonomy
        switch (automatic admin acts only while it's on, like
        every other automatic enforcement) · the §33 claim ·
        her verified admin standing on the acting surface · a
        per-group 6h throttle.  Inactive members are NOT swept
        here: real people are removed by command only.  The
        pass is failure-isolated — upkeep never breaks a group
        turn, and one member's failure never stops the rest."""
        try:
            platform = str(message.chat.platform or "")
            if not platform.startswith("telegram"):
                return
            cfg = surface.cleanup_cfg(key)
            if not cfg.get("enabled", True) \
                    or not cfg.get("auto_deleted", True):
                return
            settings = surface.store.settings(key)
            if not (settings.get("defense") or {}).get(
                    "autonomy", True):
                return
            if not surface.claim_state(key)["claimed"]:
                return
            now = time.time()
            if not surface.cleanup_auto_due(
                    key, now=now,
                    interval_s=self._CLEANUP_AUTO_INTERVAL_S):
                return
            surface.note_cleanup_auto(key, ts=now)
            adapter = self._group_adapter(platform)
            if adapter is None or not self._cleanup_admin_power(
                    adapter, message):
                return
            from types import SimpleNamespace

            from ...social.chat.group_cleanup import (
                DEFAULT_INACTIVE_DAYS)

            # The acting identity for the journal: the upkeep
            # itself, never whichever member's message happened
            # to drive the beat.
            upkeep = SimpleNamespace(
                chat=message.chat, sender_id="",
                sender="member cleanup (automatic)")
            findings, _roster = self._cleanup_scan(
                surface, key, upkeep,
                int(cfg.get("inactive_days")
                    or DEFAULT_INACTIVE_DAYS))
            removed = failed = skipped = 0
            for finding in findings["deleted"]:
                outcome, _why = self._cleanup_act_on(
                    surface, key, upkeep, adapter, finding,
                    "deleted")
                if outcome == "removed":
                    removed += 1
                elif outcome == "failed":
                    failed += 1
                else:
                    skipped += 1
            if removed or failed:
                surface._audit(
                    key, "cleanup_auto",
                    by="member cleanup (automatic)",
                    removed=removed, failed=failed,
                    skipped=skipped)
                surface.store.save()
        except Exception:  # noqa: BLE001 - upkeep is isolated
            _log.warning("cleanup auto drive failed for %s",
                         key, exc_info=True)

    # Seconds between WhatsApp join-request discovery polls
    # for one group: the door list is a cheap read, but it is
    # still an upkeep cost, never a per-message one.
    _WA_JOINREQ_POLL_INTERVAL_S = 120.0

    #: §28 criteria WhatsApp never reports at request time:
    #: the platform has no usernames, period.
    _WA_NEVER_SEEN = ("username",)

    def _wa_join_request_drive(self, surface, key, message) -> None:  # type: ignore[no-untyped-def]
        """WhatsApp join-request discovery (DESIGN §28, census
        gap #5).  Telegram delivers join requests as events;
        WhatsApp holds them server-side — no event ever
        arrives — so on the group cadence (this beat, throttled
        per group inside) a claimed WhatsApp group in request
        mode polls its door through the bridge and feeds every
        NEW requester into the SAME §28 evaluation and admin
        queue the Telegram events use: clean requests
        auto-approve where configured, rule-declines decline,
        the borderline queues with its notice, and /joinq (or
        the buttons on the notice) decides — one machinery, a
        second limb.

        Platform truth, held honestly: WhatsApp admits fully
        or not at all — no restricted-admit state exists — so
        an evaluation whose enforcement would be "restricted"
        queues for a human instead of being half-approved, and
        a decline resting only on criteria WhatsApp cannot
        report (usernames don't exist here; a name the bridge
        doesn't know can't be judged) queues too — nobody is
        auto-refused on evidence the platform never offered.
        A group whose approval mode is off at WhatsApp's door
        simply never has requests (absence, not errors); a
        bridge that can't answer leaves the queue untouched —
        absence of an answer is not absence of requesters.
        Failure-isolated like every upkeep beat."""
        try:
            if str(message.chat.platform or "") != "whatsapp":
                return
            settings = surface.store.settings(key)
            if (settings.get("acceptance") or {}).get(
                    "mode") != "request":
                return
            if not surface.claim_state(key)["claimed"]:
                return
            adapter = self._group_adapter("whatsapp")
            list_fn = getattr(adapter, "admin_join_requests",
                              None)
            if not callable(list_fn):
                return
            now = time.time()
            if not surface.acceptance_poll_due(
                    key, now=now,
                    interval_s=self._WA_JOINREQ_POLL_INTERVAL_S):
                return
            surface.note_acceptance_poll(key, ts=now)
            info: dict = {}
            info_fn = getattr(adapter, "group_info", None)
            if callable(info_fn):
                try:
                    info = info_fn(message.chat) or {}
                except Exception:  # noqa: BLE001 - reads as off
                    info = {}
            if not info.get("join_approval_mode"):
                return
            out = list_fn(message.chat)
            if not isinstance(out, dict) or not out.get("ok"):
                return
            listed: dict[str, float] = {}
            names: dict[str, str] = {}
            for row in out.get("requests") or []:
                jid = str(row.get("jid") or "")
                if not jid:
                    continue
                try:
                    listed[jid] = float(
                        row.get("requested_at") or 0)
                except (TypeError, ValueError):
                    listed[jid] = 0.0
                names[jid] = str(row.get("name") or "")
            seen = surface.acceptance_wa_seen(key)
            pending_ids = {str(r.get("user_id") or "")
                           for r in surface.acceptance_queue(key)}
            surface.acceptance_wa_sync(key, listed, now=now)
            for user_id, req_ts in listed.items():
                if user_id in pending_ids:
                    continue
                if user_id in seen and seen[user_id] == req_ts:
                    continue
                self._wa_join_request_evaluate(
                    surface, key, adapter, message, user_id,
                    names.get(user_id) or "", now=now)
        except Exception:  # noqa: BLE001 - upkeep is isolated
            _log.warning("whatsapp join-request drive failed "
                         "for %s", key, exc_info=True)

    def _wa_join_request_evaluate(self, surface, key, adapter,  # type: ignore[no-untyped-def]
                                  message, user_id: str,
                                  name: str, *, now: float) -> None:
        """One newly-seen WhatsApp requester through §28: the
        shared evaluation (the surface judges), this limb
        performs — with the platform-truth adjustments from
        the drive's docstring."""
        requester = {"id": user_id, "name": name,
                     "first_name": name, "username": "",
                     "is_bot": False}
        facts = self._acceptance_facts(requester)
        criteria = ((surface.store.settings(key)
                     .get("acceptance") or {})
                    .get("criteria") or {})
        has_photo = None
        if criteria.get("photo"):
            has_photo = self._acceptance_photo(
                adapter, message.chat, user_id)
        out = surface.acceptance_on_request(
            key, facts, has_photo=has_photo, now=now)
        record = out.get("record") or {}
        if record:
            record["source"] = "wa_poll"
            surface.store.save()
        verdict = out.get("verdict")
        reasons = list(out.get("reasons") or [])
        if verdict == "approve_restricted":
            # WhatsApp has no restricted admission: the rule's
            # enforcement can't be performed, so the request
            # queues for a human instead of being half-approved.
            self._wa_request_requeue(surface, record)
            reasons.append(
                "WhatsApp can't admit a member restricted — "
                "approve admits them fully, decline keeps them "
                "out; the call is an admin's.")
            if record:
                self._send_acceptance_queue_note(
                    surface, key, message, record, reasons)
            return
        if verdict == "approve":
            done = self._acceptance_join_request(
                surface, key, adapter, message.chat, user_id,
                approve=True)
            if not done:
                return
            admitted = surface.acceptance_request_admitted(
                key, str(out.get("request_id") or ""))
            if admitted.get("welcome"):
                text = surface.welcome_text_for(
                    key, name=facts["name"],
                    username=facts["username"],
                    group_name=str(message.chat.title or ""))
                if text and self._system_send_ok(
                        surface, key, message, "welcome"):
                    if not self._rhythm_hold(
                            surface, key, "welcome", text):
                        self._send(
                            message.chat, text,
                            buttons=self._welcome_buttons(
                                surface, key),
                            parse_mode=self._rendered_mode(
                                message, text))
                        fire = getattr(self, "_fire_expression",
                                       None)
                        if fire is not None:
                            fire(message, "welcome")
                # §32: the request was admitted — the reputation
                # hook hears about any record.
                self._reputation_join_hook(
                    surface, key, message, user_id=user_id,
                    username=facts["username"],
                    name=facts["name"], held=False)
            return
        if verdict == "decline":
            if self._wa_decline_unverifiable(surface, key, out,
                                             facts):
                self._wa_request_requeue(surface, record)
                reasons.append(
                    "part of this check rests on things "
                    "WhatsApp doesn't show at the door — "
                    "usernames don't exist here, and a name the "
                    "bridge doesn't know can't be judged — so "
                    "a human decides.")
                if record:
                    self._send_acceptance_queue_note(
                        surface, key, message, record, reasons)
                return
            self._acceptance_join_request(
                surface, key, adapter, message.chat, user_id,
                approve=False)
            return
        # queued — a human decides
        if record:
            self._send_acceptance_queue_note(
                surface, key, message, record, reasons)

    @staticmethod
    def _wa_request_requeue(surface, record) -> None:  # type: ignore[no-untyped-def]
        """Undo the rule's terminal state on a WhatsApp request
        record: the platform can't carry the enforcement the
        rule chose, so the request waits for a human after
        all."""
        if record:
            record["state"] = "pending"
            record["decided_by"] = ""
            record.pop("decided_ts", None)
            surface.store.save()

    def _wa_decline_unverifiable(self, surface, key, out,  # type: ignore[no-untyped-def]
                                 facts) -> bool:
        """True when a rule-decline of a WhatsApp request rests
        ONLY on criteria the platform never reported: usernames
        don't exist on WhatsApp, and an unknown name can't fail
        the name criteria.  The previously-banned fact (her own
        ledger) and an explicit no-photo ARE real evidence
        here, so a decline they drive stands."""
        missing = [str(m) for m in (out.get("missing") or [])]
        if not missing:
            return False
        unverifiable = set(self._WA_NEVER_SEEN)
        if not str(facts.get("name") or ""):
            unverifiable |= {"firstname", "lastname", "hygiene"}
        cfg, _raid = surface.acceptance_cfg(key)
        overrides = cfg.get("overrides") or {}
        global_action = str(cfg.get("action") or "restricted")

        def _drives(criterion: str) -> bool:
            action = str(overrides.get(criterion)
                         or global_action)
            return action in ("decline", "kick")

        drivers = [k for k in missing if _drives(k)]
        return bool(drivers) and all(
            k in unverifiable for k in drivers)

    def _control_rmdeleted(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        return self._cleanup_remove(message, which="deleted")

    def _control_rminactive(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        days = None
        tail_txt = str(tail or "").strip().lower()
        if tail_txt:
            digits = tail_txt.rstrip("d").strip()
            if digits.isdigit():
                days = max(1, min(3650, int(digits)))
            else:
                from ...social.chat.group_cleanup import (
                    DEFAULT_INACTIVE_DAYS)

                surface = self._group_surface()
                key = self._group_key_or_none(message)
                current = DEFAULT_INACTIVE_DAYS
                if key is not None:
                    current = int(surface.cleanup_cfg(key).get(
                        "inactive_days") or DEFAULT_INACTIVE_DAYS)
                return (f"usage: /rminactive [days] — bare "
                        f"uses this group's window "
                        f"({current} days).")
        return self._cleanup_remove(message, which="inactive",
                                    days_override=days)

    def _control_mute(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "mute")
            if note:
                return note
            return ("who should I mute? Reply to their message with "
                    "/mute, or /mute @member [10m|2h|1d].")
        user_id, label = res.member.user_id, res.member.label
        seconds, reason = _parse_duration(res.rest)
        if seconds is None:
            # Target first, then the rest (§29): the group's
            # standing default when one is set; the standing
            # hour for a reply-target (the grammar every group
            # bot uses); and for a NAMED member with no length
            # given, ask for the length — never "who?".
            default_s = int(surface.store.settings(
                key).get("mute_default_s") or 0)
            if default_s:
                seconds = default_s
            elif res.via_reply:
                seconds = 3600
            else:
                return (f"for how long should I mute {label}? "
                        f"/mute {res.named_token} 10m|2h|1d — or "
                        f"set a standing default: /gsettings "
                        f"mute default 2h.")
        ok = self._mod_run(
            surface, key, message, "mute", user_id,
            self._mod_mute_call(message, user_id, seconds))
        if not ok:
            return "the mute didn't go through — the platform refused it (it's in /glog)."
        span = "indefinitely" if seconds == 0 else \
            f"for {_duration_text(seconds)}"
        return f"🔇 {label} muted {span}" + \
            (f" — {reason}" if reason else "")

    def _control_unmute(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "unmute")
            if note:
                return note
            return "who should I unmute? Reply to them or /unmute @member."
        user_id, label = res.member.user_id, res.member.label

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_unrestrict", None)
            if fn is None:
                raise RuntimeError("this platform path cannot unmute")
            return fn(message.chat, user_id)

        if not self._mod_run(surface, key, message, "unmute",
                             user_id, call):
            return "the unmute didn't go through — the platform refused it (it's in /glog)."
        return f"🔊 {label} can speak again."

    def _control_ban(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "ban")
            if note:
                return note
            return "who should I ban? Reply to their message or /ban @member."
        user_id, label, rest = (res.member.user_id,
                                res.member.label, res.rest)

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_ban", None)
            if fn is None:
                raise RuntimeError("this platform path cannot ban")
            return fn(message.chat, user_id)

        if not self._mod_run(surface, key, message, "ban", user_id,
                             call):
            return "the ban didn't go through — the platform refused it (it's in /glog)."
        return f"🔨 {label} is banned" + \
            (f" — {rest}" if rest.strip() else "")

    def _control_unban(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "unban")
            if note:
                return note
            return "who should I unban? /unban @member or their id."
        user_id, label = res.member.user_id, res.member.label

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_unban", None)
            if fn is None:
                raise RuntimeError("this platform path cannot unban")
            return fn(message.chat, user_id)

        if not self._mod_run(surface, key, message, "unban",
                             user_id, call):
            return "the unban didn't go through — the platform refused it (it's in /glog)."
        return f"{label} is unbanned — they can rejoin."

    def _control_kick(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "kick")
            if note:
                return note
            return "who should I kick? Reply to their message or /kick @member."
        user_id, label, rest = (res.member.user_id,
                                res.member.label, res.rest)

        def call(adapter):  # type: ignore[no-untyped-def]
            ban = getattr(adapter, "admin_ban", None)
            unban = getattr(adapter, "admin_unban", None)
            if ban is None or unban is None:
                raise RuntimeError("this platform path cannot kick")
            out = ban(message.chat, user_id)
            if isinstance(out, dict) and out.get("ok") is False:
                return out
            # A kick is a ban immediately lifted: out now, free
            # to rejoin later.
            return unban(message.chat, user_id)

        if not self._mod_run(surface, key, message, "kick",
                             user_id, call):
            return "the kick didn't go through — the platform refused it (it's in /glog)."
        return f"👢 {label} is out" + \
            (f" — {rest}" if rest.strip() else "") + \
            " (they can rejoin)."

    def _control_warn(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "warn")
            if note:
                return note
            return "who should I warn? Reply to their message or /warn @member."
        user_id, label, rest = (res.member.user_id,
                                res.member.label, res.rest)
        actor = str(message.sender_id or message.sender or "")
        reason = rest.strip()
        count = surface.record_warn(key, user_id, by=actor,
                                    reason=reason)
        # The ladder is the group's own (§30): the count a
        # warning quotes is the NEXT designed step, and landing
        # on a step fires it — an interval step restarts the
        # admin-facing warn count only (the spine keeps its
        # history); the final action resets everything, as the
        # shipped ladder always did.
        esc = surface.escalation_config(key)
        step = esc.action_at(count)
        nxt = esc.next_step(count - 1)
        denom = int(nxt["at"]) if nxt else esc.warn_limit
        if step is None:
            return (f"⚠️ {label} — warning {count}/{denom}"
                    + (f": {reason}" if reason else ""))
        ok, phrase, paid = self._fire_warn_step(
            surface, key, message, user_id, step)
        is_final = count >= esc.warn_limit and \
            step["action"] == esc.final_action
        if is_final:
            surface.clear_warns(key, user_id)
        else:
            surface.reset_warn_bookkeeping(key, user_id)
        if not ok:
            return (f"⚠️ {label}: warning {count}/{denom} — "
                    f"that's the step ({phrase}), but it was "
                    "refused (see /glog).")
        if is_final:
            tail_note = f" — that's the limit: {phrase}, " \
                        "warns reset."
        else:
            tail_note = f" — that's the step: {phrase}."
        self._post_warn_notice(surface, key, message, user_id,
                               label, step, count, paid)
        return f"⚠️ {label}: warning {count}/{denom}{tail_note}"

    def _fire_warn_step(self, surface, key, message, user_id,  # type: ignore[no-untyped-def]
                        step) -> tuple:
        """Execute one ladder step a manual warn landed on.
        Returns (ok, phrase, paid_fine) — the phrase in the
        member's terms, from what ACTUALLY executed."""
        action = str(step.get("action") or "")
        if action == "fine":
            paid = surface.apply_fine(
                key, user_id, amount=int(step.get("amount") or 0))
            if paid:
                return True, f"fined {paid} sparks", paid
            return False, (f"a {int(step.get('amount') or 0)}-spark "
                           "fine (nothing to collect)"), 0
        if action == "mute":
            minutes = int(step.get("minutes") or 60)
            ok = self._mod_run(
                surface, key, message, "mute", user_id,
                self._mod_mute_call(message, user_id, minutes * 60))
            phrase = "muted for an hour" if minutes == 60 else \
                f"muted for {minutes} min"
            return ok, phrase, 0
        if action == "restrict":
            def call(adapter):  # type: ignore[no-untyped-def]
                if not self._acceptance_restrict(
                        adapter, message.chat, user_id, on=True):
                    raise RuntimeError("restrict refused")
                return {"ok": True}

            ok = self._mod_run(surface, key, message, "restrict",
                               user_id, call)
            return ok, "restricted until an admin lifts it", 0
        if action == "kick":
            def kick_call(adapter):  # type: ignore[no-untyped-def]
                ban = getattr(adapter, "admin_ban", None)
                unban = getattr(adapter, "admin_unban", None)
                if ban is None or unban is None:
                    raise RuntimeError("platform can't kick")
                out = ban(message.chat, user_id)
                unban(message.chat, user_id)
                return out

            ok = self._mod_run(surface, key, message, "kick",
                               user_id, kick_call)
            return ok, "removed from the group", 0
        if action == "ban":
            def ban_call(adapter):  # type: ignore[no-untyped-def]
                fn = getattr(adapter, "admin_ban", None)
                if fn is None:
                    raise RuntimeError("platform can't ban")
                return fn(message.chat, user_id)

            ok = self._mod_run(surface, key, message, "ban",
                               user_id, ban_call)
            return ok, "banned from the group", 0
        return False, action, 0

    def _post_warn_notice(self, surface, key, message, user_id,  # type: ignore[no-untyped-def]
                          label, step, count, paid) -> None:
        """The §31 notice for a ladder step a manual warn fired:
        recorded on the surface, posted with its Undo."""
        from ...social.chat.group_defense import (
            consequence_text, render_warning)

        esc = surface.escalation_config(key)
        info = surface.strikes_for(key, user_id)
        strike_ts = 0.0
        for strike in reversed(info["strikes"]):
            if strike.get("rule") == "manual_warn":
                strike_ts = float(strike.get("ts") or 0)
                break
        kind = {"restrict": "acceptance_restrict"}.get(
            str(step.get("action")), str(step.get("action")))
        text = render_warning(
            f"{label} [{user_id}]", "manual_warn",
            consequence_text(
                executed="restrict" if kind
                == "acceptance_restrict" else kind,
                count=count, cfg=surface.defense_config(key),
                warn_threshold=0, esc=esc,
                fine_amount=paid if kind == "fine" else 0,
                oldest_ts=info["oldest_ts"], now=time.time()))
        nid = surface.record_notice(
            key, user_id=user_id, mention=label,
            rule="manual_warn", action=kind, count=count,
            limit=esc.warn_limit, strike_ts=strike_ts,
            strike_rule="manual_warn",
            minutes=int(step.get("minutes") or 0)
            if kind == "mute" else 0,
            fine=paid if kind == "fine" else 0, text=text)
        # The notice RECORD is the state (undo works from it
        # whenever the tap lands); the post is best-effort.
        try:
            self._send_action_notice(surface, key, message,
                                     {"notice_id": nid})
        except Exception:  # noqa: BLE001 - the record is durable
            _log.debug("warn notice post failed", exc_info=True)

    def _control_strikes(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """The strike ledger (defense brief §2.2): one member's
        record, or the group's current holders when no target is
        named.  Read from the same spine the sensors write."""
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        if not res.found:
            note = self._mod_no_target(res, "look up")
            if note:
                return note
            user_id, label = None, ""
        else:
            user_id, label = res.member.user_id, res.member.label
        if not user_id:
            rows = []
            for uid, member in surface.members(key).items():
                info = surface.strikes_for(key, uid)
                if info["count"]:
                    rows.append((info["count"], uid, member))
            if not rows:
                return "no strikes on record right now."
            rows.sort(key=lambda r: -r[0])
            from ...social.chat.typography import Card

            card = Card("moderation", title="Strike ledger",
                        icon="⚖️")
            card.stats([
                (str(member.get("name") or (
                    f"@{member['username']}"
                    if member.get("username") else uid)),
                 f"{count} strike{'s' if count != 1 else ''}")
                for count, uid, member in rows[:10]])
            card.note("live window")
            return self._card(message, card)
        info = surface.strikes_for(key, user_id)
        window = info["window_days"]
        if not info["count"]:
            return (f"{label} — clean record (no strikes in the "
                    f"last {window} days).")
        from ...social.chat.typography import Card

        card = Card("moderation", title="Strike record", icon="⚖️")
        card.line(f"**{label}** — {info['count']} strike"
                  f"{'s' if info['count'] != 1 else ''} "
                  f"(last {window} days)")
        card.block_lines([
            f"· {strike.get('rule')} ({strike.get('action')})"
            f" — {_fmt_ts(strike.get('ts'))}"
            for strike in info["strikes"]])
        # The ledger reads against the group's OWN ladder (§30):
        # the next designed step, the limit it climbs to, and
        # when the oldest active warn clears.
        from ...social.chat.group_defense import warn_expiry_text

        notes = []
        if info["next_at"]:
            notes.append(f"next: {info['next_label']} "
                         f"at {info['next_at']}")
        notes.append(f"limit {info['limit']} "
                     f"({info['final_action']})")
        expiry = warn_expiry_text(info["oldest_ts"],
                                  window_days=window, now=time.time())
        if expiry:
            notes.append(expiry)
        card.note(" · ".join(notes))
        return self._card(message, card)

    def _control_pin(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        target_id = str(getattr(message, "reply_to", "") or "")
        if not target_id:
            return "reply to the message you want pinned, with /pin."

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_pin", None)
            if fn is None:
                raise RuntimeError("this platform path cannot pin")
            return fn(message.chat, target_id)

        if not self._mod_run(surface, key, message, "pin",
                             target_id, call):
            return "the pin didn't go through — the platform refused it (it's in /glog)."
        return "📌 pinned."

    def _control_unpin(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_unpin", None)
            if fn is None:
                raise RuntimeError("this platform path cannot unpin")
            return fn(message.chat)

        if not self._mod_run(surface, key, message, "unpin", "",
                             call):
            return "the unpin didn't go through — the platform refused it (it's in /glog)."
        return "📌 unpinned."

    def _control_report(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """Member-side /report: filed into the held audit log with
        the reported member + the reporter's words; admins pull it
        with /glog.  Not admin-gated — reporting is for everyone."""
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        res = self._mod_target(surface, key, tail, message)
        user_id = res.member.user_id if res.found else ""
        label = res.member.label if res.found else (
            res.channel_label if res.channel_post else "")
        reporter = str(message.sender_id or message.sender or "")
        surface._audit(key, "report", user_id=user_id or "",
                       by=reporter, target_label=label,
                       reason=str(tail or "")[:300],
                       replied_text=str(
                           (message.meta or {}).get(
                               "reply_to_text") or "")[:200])
        surface.store.save()
        if user_id and user_id != reporter:
            # §32: a report also PROPOSES a reputation flag into
            # this group's queue — an admin confirms or discards
            # it; the report alone brands nobody.
            member = surface.member(key, user_id) or {}
            person = surface.reputation_person(
                key, user_id,
                username=str(member.get("username") or ""),
                name=str(member.get("name") or ""))
            note = str(res.rest or "").strip()[:300] \
                or "reported by a member"
            surface.stage_flag(
                key, user_id=user_id, person=person,
                category="other", note=note, by=reporter,
                by_label=str(message.sender or ""),
                direct=False, source="report")
        if user_id:
            return (f"📣 reported {label} — it's in the group log "
                    f"for the admins (/glog).")
        if res.channel_post:
            return (f"📣 reported — that message was posted as "
                    f"{res.channel_label} (a channel post, not a "
                    f"member); it's in the group log for the "
                    f"admins (/glog).")
        if res.named_token:
            return (f"📣 reported — I couldn't pin "
                    f"{res.named_token} to a member, but your "
                    f"words are in the group log for the admins "
                    f"(/glog).")
        return ("📣 reported — it's in the group log for the "
                "admins (/glog). Reply to the message next time "
                "and I'll attach who wrote it.")

    def _control_purge(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        target_id = str(getattr(message, "reply_to", "") or "")
        if not target_id:
            return "reply to the message you want gone, with /purge."

        def call(adapter):  # type: ignore[no-untyped-def]
            fn = getattr(adapter, "admin_delete", None)
            if fn is None:
                raise RuntimeError("this platform path cannot delete")
            return fn(message.chat, target_id)

        if not self._mod_run(surface, key, message, "purge",
                             target_id, call):
            from ...social.chat.tgbot_buttons import actionable

            return actionable(
                "the delete didn't go through — the platform "
                "refused it; the attempt is in the group log.",
                [[("📜 Group log", "glog")]])
        return "🧹 gone."

    # ---- economy commands (P2) --------------------------------------
    def _economy_gate(self, message):  # type: ignore[no-untyped-def]
        """Shared gate for economy commands: in-group + economy
        on + ledger healthy.  Returns (key, None) or (None, text)."""
        key = self._group_key_or_none(message)
        if key is None:
            return None, "that's a group command — run it inside the group."
        surface = self._group_surface()
        if not surface.system_output_allowed(
                key, via_platform=str(
                    getattr(message.chat, "platform", "") or "")):
            # §33: the economy is a bot system — it does not run
            # (or build state) in a room Devon-the-bot doesn't
            # operate in, whichever account is sitting there.
            return None, ("my group systems aren't running in "
                          "this group — they speak only where my "
                          "bot operates, or where the owner has "
                          "designated the room (/gsettings system "
                          "output on).")
        frozen = surface.economy_frozen(key)
        if frozen:
            return None, ("the group economy is frozen — its ledger "
                          "needs an admin's eyes before any sparks "
                          "move again.")
        if not surface.store.settings(key)["economy_enabled"]:
            return None, "the group economy is off here."
        return key, None

    def _member_ref(self, surface, key, token, message):  # type: ignore[no-untyped-def]
        """Resolve a /pay target to a member id — now the
        §29 organ's job (every path, write-back included);
        kept as the thin delegate the economy handlers call."""
        res = self._resolver(surface).resolve_token(
            key, message, str(token or ""))
        return res.member.user_id if res.found else None

    def _named_ref(self, surface, key, token, message):  # type: ignore[no-untyped-def]
        """(user_id, failure_text): resolve a named economy
        target; on a miss the failure text is the organ's
        honest exhaustion line, not a ledger-only shrug."""
        res = self._resolver(surface).resolve_token(
            key, message, str(token or ""))
        if res.found:
            return res.member.user_id, ""
        return None, res.exhaustion_text()

    def _control_balance(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...social.chat.typography import Card, sparks

        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        stmt = surface.economy.statement(key, user_id)
        member = surface.members(key).get(user_id) or {}
        name = str(member.get("name") or message.sender or "you")
        wallet = int(stmt.get("balance", 0))
        bank = surface.economy.bank_balance(key, user_id)
        card = Card("economy", title=name, icon="💰")
        rows = [("Wallet", sparks(wallet), True)]
        if bank:
            rows.append(("Banked", sparks(bank), True))
            rows.append(("Total", sparks(wallet + bank), True))
        # One purse (§25): gems + property stand on the same
        # card as the sparks — one wealth picture, three
        # denominations, each with its own faucets and sinks.
        estate = surface.estate_summary(key, user_id)
        if estate["gems"] or estate["property_count"]:
            rows.append(("Gems", f"{estate['gems']} 💎", True))
        if estate["property_count"]:
            rows.append(
                ("Property",
                 f"{estate['property_count']} · "
                 f"{sparks(estate['property_value'])}", True))
        # The exchange state (§25): the live rate + the week's
        # pool — the door between this purse and the games
        # wallet, stated on the purse's own card.
        exch_cfg = surface.exchange_cfg(key)
        if exch_cfg.get("enabled", True):
            wk = surface.exchange.week_state(key, exch_cfg)
            if wk["capacity"]:
                from ...community.group_exchange import rate_for

                rows.append((
                    "Exchange",
                    f"100 🪙 → {rate_for(exch_cfg, wk['remaining'], wk['capacity'])} "
                    f"sparks · pool {wk['remaining']:,}", True))
            else:
                rows.append(("Exchange", "closed this week", True))
        card.stats(rows)
        for badge in stmt.get("badges", []):
            card.line(f"🏅 {badge.get('badge')} "
                      f"({badge.get('season')})")
        card.stats([
            ("Group treasury",
             sparks(surface.economy.treasury_balance(key)), True),
            ("Jackpot pot",
             sparks(surface.economy.jackpot_balance(key)), True),
        ], title="The group")
        if bank:
            card.note("banked sparks are safe from robbery")
        disclaimer = str(stmt.get("disclaimer") or "")
        if disclaimer:
            card.flavor(disclaimer)
        buttons = [[("⛏️ Mine", "mine"), ("🛠️ Work", "work")],
                   [("🗺️ Quests", "quests"), ("🛒 Shop", "gshop")],
                   [("🏘️ Property", "property"),
                    ("💎 Rent", "rent")]]
        vault_row = []
        if wallet > 0:
            vault_row.append(("🏦 Deposit", "deposit all"))
        if bank > 0:
            vault_row.append(("💵 Withdraw", "withdraw all"))
        if vault_row:
            buttons.append(vault_row)
        if exch_cfg.get("enabled", True):
            buttons.append([("🔄 Convert", "convert")])
        return self._card(message, card, buttons)

    def _control_mine(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...social.chat.tgbot_buttons import actionable
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        if surface.jail_state(key, user_id):
            return "🔒 no mining from jail — /bail first."
        buttons = [[("🛠️ Work", "work"), ("🪙 My balance", "balance")]]
        entry = surface.economy.mining_claim(key, user_id)
        if entry is None:
            rate = surface.economy.claim_rate(key, user_id)
            return actionable(
                f"⛏️ nothing to claim yet — claims cool down "
                f"between hits, and the daily cap is real. "
                f"Next claim pays <b>{rate}</b> sparks.", buttons)
        rate = surface.economy.hashrate(key, user_id)
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Mining claim", icon="⛏️")
        card.line(f"**+{sparks(entry.amount)}**")
        card.stats([
            ("Balance",
             sparks(surface.economy.balance(key, user_id)), True),
            ("Hashrate", str(rate["hashrate"])),
        ])
        card.note("hashrate is engagement: real activity this "
                  "week")
        return self._card(message, card, buttons)

    def _control_work(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        if surface.jail_state(key, user_id):
            return "🔒 no shifts from jail — /bail first."
        result = surface.work_shift(key, user_id)
        if not result["ok"]:
            if result["reason"] == "cooldown":
                return (f"🛠️ you're between shifts — next one opens "
                        f"in ~{max(1, result['wait_s'] // 60)} min.")
            return "🛠️ no shift paid — the daily earn cap is hit."
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Shift done", icon="🛠️")
        card.line(f"**+{sparks(result['pay'])}**")
        card.stats([("Level", str(result["level"])),
                    ("Balance", sparks(result["balance"]), True)])
        return self._card(
            message, card,
            [[("⛏️ Mine", "mine"), ("🪙 My balance", "balance")]])

    def _control_pay(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        parts = str(tail or "").split()
        amount = None
        target_token = ""
        for part in parts:
            if part.isdigit() and amount is None:
                amount = int(part)
            elif not target_token:
                target_token = part
        if amount is None or not target_token:
            return "usage: /pay @member 50"
        recipient, miss = self._named_ref(surface, key,
                                          target_token, message)
        if not recipient:
            return miss
        sender = str(message.sender_id or message.sender or "")
        result = surface.pay(key, sender, recipient, amount)
        if not result["ok"]:
            if result["reason"] == "funds":
                return "not enough sparks for that."
            return "that payment doesn't work as written."
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Payment sent", icon="💸")
        card.line(f"**{sparks(result['net'])}** paid.")
        rows = [("Your balance", sparks(result["balance"]), True)]
        if result["fee"]:
            rows.insert(0, ("Treasury fee",
                            sparks(result["fee"]), True))
        card.stats(rows)
        return self._card(message, card,
                          [[("🪙 My balance", "balance")]])

    def _control_gshop(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...community.group_economy import STATUS_SHOP

        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        text = str(tail or "").strip()
        if text.lower().startswith("buy "):
            item_id = text[4:].strip().lower()
            from ...community.group_conflict import ARSENAL
            from ...community.group_real_estate import PREMIUM_SHELF

            if item_id in ARSENAL:
                return self._gear_buy_reply(surface, key, message,
                                            user_id, item_id)
            if item_id in PREMIUM_SHELF:
                grant = surface.estate_premium_buy(key, user_id,
                                                   item_id)
                if grant is None:
                    item = PREMIUM_SHELF[item_id]
                    gems = surface.estate.gems_balance(key,
                                                       user_id)
                    return (f"not enough gems — {item['name']} "
                            f"costs {item['cost']} 💎, you have "
                            f"{gems} 💎. Gems come from property "
                            f"rent, season rewards and "
                            f"governance gifts — never from "
                            f"sparks.")
                return (f"💎 {grant['name']} is yours — granted "
                        f"for gems and ledgered.")
            grant = surface.economy.buy_status(key, user_id, item_id)
            if grant is None:
                item = STATUS_SHOP.get(item_id)
                if item is None:
                    return "no such item — /gshop lists the shelf."
                return (f"not enough sparks — {item['name']} costs "
                        f"{item['cost']}, you have "
                        f"{surface.economy.balance(key, user_id)}.")
            if item_id == "pinned_message":
                surface.grant_pin(key, user_id)
                return ("📌 Pinned Message bought — your next "
                        "message here gets pinned for the group.")
            if item_id == "shout_out":
                who = (message.sender_username
                       and f"@{message.sender_username}") \
                    or message.sender or "a member"
                return (f"📣 SHOUT-OUT for {who}! bought with "
                        f"sparks, heard by the whole group.")
            return (f"✨ {grant['name']} is yours — granted and "
                    f"ledgered.")
        from ...social.chat.typography import Card, sparks

        balance = surface.economy.balance(key, user_id)
        card = Card("economy", title="Group shop", icon="🛒")
        gems = surface.estate.gems_balance(key, user_id)
        card.line(f"your sparks: **{sparks(balance)}** · "
                  f"your gems: **{gems} 💎**")
        shelf = []
        buy_row: list = []
        buttons: list = []
        for item_id, item in STATUS_SHOP.items():
            afford = balance >= int(item["cost"])
            shelf.append(
                f"`{item_id}` — **{item['name']}** · "
                f"{sparks(item['cost'])} · {item['blurb']}"
                + ("" if afford else " · *not yet*"))
            # Buttons are authored from the LIVE shelf + balance:
            # an item gets a buy button exactly when this member
            # can actually buy it — the buttons match the display.
            if afford:
                buy_row.append(
                    (f"Buy {item['name']} · {item['cost']}",
                     f"gshop buy {item_id}"))
                if len(buy_row) == 2:
                    buttons.append(buy_row)
                    buy_row = []
        if buy_row:
            buttons.append(buy_row)
        card.block_lines(shelf)
        # The premium shelf (§17): gems only — a second purse
        # with its own faucets, never a sparks price in disguise.
        from ...community.group_real_estate import PREMIUM_SHELF

        premium = []
        premium_row: list = []
        for item_id, item in PREMIUM_SHELF.items():
            afford = gems >= int(item["cost"])
            premium.append(
                f"`{item_id}` — **{item['name']}** · "
                f"{item['cost']} 💎 · {item['blurb']}"
                + ("" if afford else " · *not yet*"))
            if afford:
                premium_row.append(
                    (f"Buy {item['name']} · {item['cost']}💎",
                     f"gshop buy {item_id}"))
                if len(premium_row) == 2:
                    buttons.append(premium_row)
                    premium_row = []
        if premium_row:
            buttons.append(premium_row)
        card.block_lines(premium, title="Premium shelf — gems 💎")
        # The arsenal shelf (§23 addendum): gear for the
        # conflict layer, sold through this same shop — shown
        # only where the conflict layer runs, priced in sparks
        # and/or gems, rated by job so the gamble stays
        # readable before anyone pays.
        if surface._conflict_mode(key) != "off":
            from ...community.group_conflict import ARSENAL

            gear = []
            gear_row: list = []
            for item_id, item in ARSENAL.items():
                price = dict(item.get("price") or {})
                cost_bits = []
                if price.get("sparks"):
                    cost_bits.append(sparks(price["sparks"]))
                if price.get("gems"):
                    cost_bits.append(f"{price['gems']} 💎")
                afford = (not price.get("sparks")
                          or balance >= int(price["sparks"])) \
                    and (not price.get("gems")
                         or gems >= int(price["gems"]))
                ratings = ", ".join(
                    f"{job} {val}" for job, val in
                    (item.get("ratings") or {}).items() if val)
                gear.append(
                    f"`{item_id}` — {item.get('icon', '•')} "
                    f"**{item['name']}** · "
                    f"{' + '.join(cost_bits) or '—'} · "
                    f"{ratings} · {item['blurb']}"
                    + ("" if afford else " · *not yet*"))
                if afford and not item.get("admin_mint"):
                    gear_row.append(
                        (f"Buy {item['name']}",
                         f"gshop buy {item_id}"))
                    if len(gear_row) == 2:
                        buttons.append(gear_row)
                        gear_row = []
            if gear_row:
                buttons.append(gear_row)
            card.block_lines(gear, title="Arsenal — conflict gear")
            card.note("gear wears, breaks and drops — /gear "
                      "shows your loadout; defenses install "
                      "with /gear install")
        buttons.append([("🪙 My balance", "balance")])
        return self._card(message, card, buttons)

    def _control_quests(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        text = str(tail or "").strip()
        if text.lower().startswith("claim "):
            quest_id = text[6:].strip().lower()
            entry = surface.economy.claim_quest(key, user_id, quest_id)
            if entry is None:
                return ("that quest isn't claimable — either not "
                        "finished yet, or already claimed this "
                        "period.")
            from ...social.chat.typography import Card, sparks

            card = Card("economy", title="Quest claimed", icon="🏅")
            card.line(f"**+{sparks(entry.amount)}**")
            card.stats([("Balance",
                         sparks(surface.economy.balance(
                             key, user_id)), True)])
            return self._card(
                message, card,
                [[("🪙 My balance", "balance"),
                  ("🗺️ Quests", "quests")]])
        from ...social.chat.typography import Card

        progress = surface.economy.quest_progress(key, user_id)
        card = Card("economy", title="Your quests", icon="🗺️")
        mine = []
        buttons = []
        for quest_id, quest in progress.items():
            mark = "✅ claimed" if quest["claimed"] else (
                "🎁 ready — /quests claim " + quest_id
                if quest["complete"] else
                f"{quest['have']}/{quest['need']}")
            mine.append(f"`{quest_id}` — **{quest['name']}**: "
                        f"{quest['blurb']} · {mark} · "
                        f"reward {quest['reward']}")
            # A claim button exists exactly for quests this
            # member can claim RIGHT NOW — authored from the
            # same progress state the list renders.
            if quest["complete"] and not quest["claimed"]:
                buttons.append([(f"🎁 Claim: {quest['name']}",
                                  f"quests claim {quest_id}")])
        card.block_lines(mine)
        # Group quests: the COLLECTIVE board — everyone's
        # activity counts; completion pays the treasury.
        import time as _time

        from ...community.group_economy import _period_key
        group_progress = surface.economy.group_quest_progress(
            key, now=_time.time())
        paid_state = surface._extras(key).get(
            "group_quests_paid") or {}
        together = []
        for quest_id, quest in group_progress.items():
            paid = paid_state.get(quest_id) == _period_key(
                quest["period"], _time.time())
            mark = ("✅ paid to treasury" if paid else
                    "🎉 complete — bonus lands soon"
                    if quest["complete"] else
                    f"{quest['have']}/{quest['need']}")
            together.append(f"`{quest_id}` — **{quest['name']}**: "
                            f"{quest['blurb']} · {mark} · "
                            f"treasury bonus {quest['bonus']}")
        card.block_lines(together,
                         title="Group quests — everyone together")
        buttons.append([("🪙 My balance", "balance")])
        return self._card(message, card, buttons)

    def _control_treasury(self, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...social.chat.typography import Card, sparks

        surface = self._group_surface()
        health = surface.economy_health(key)
        policy = health.get("policy") or {}
        reserved = surface.reserved_treasury(key)
        card = Card("economy", title="Treasury", icon="🏦")
        rows = [("Treasury", sparks(health["treasury"]), True)]
        if reserved:
            rows.append(("Reserved for open drops",
                         sparks(reserved), True))
        rows += [
            ("Circulating", sparks(health["circulating"]), True),
            ("Holders", str(health["holders"])),
            ("Biggest holder",
             f"{int(health['top_share'] * 100)}%"),
            ("Active this week", str(health["active_7d"])),
            ("Jackpot pot", sparks(health.get("jackpot", 0)), True),
        ]
        card.stats(rows)
        if policy.get("throttled"):
            card.line(
                "📉 **monetary policy: faucets throttled** — "
                f"issuance ({sparks(policy['earned'])}) is running "
                f"hot against sinks ({sparks(policy['sunk'])}) "
                "this week, so activity pays half until sinks "
                "catch up.")
        else:
            card.line(
                f"📈 monetary policy: normal — "
                f"{sparks(policy.get('earned', 0))} issued vs "
                f"{sparks(policy.get('sunk', 0))} sunk this week.")
        proposals = surface.open_proposals(key)
        treasury_buttons = None
        if proposals:
            card.line(f"🗳️ {len(proposals)} treasury proposal(s) "
                      "waiting for a second admin.")
            treasury_buttons = [[("🗳️ Second it", "approve")]]
        card.note("triple 7️⃣ on /slots takes the jackpot pot")
        return self._card(message, card, treasury_buttons)

    def _control_drop(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "drops are started by admins."
        key, err = self._economy_gate(message)
        if err:
            return err
        words = str(tail or "").split()
        numbers = [int(p) for p in words if p.isdigit()]
        if len(numbers) < 2:
            return ("usage: /drop <total> <per claim> — or "
                    "/drop treasury <total> <per claim> to fund "
                    "it from the treasury.")
        surface = self._group_surface()
        funder = str(message.sender_id or message.sender or "")
        if any(w.lower() == "treasury" for w in words):
            threshold = int(surface.store.settings(
                key).get("governance_threshold") or 0)
            if threshold and numbers[0] > threshold \
                    and not self._operator_or_false(message):
                proposal = surface.propose_spend(
                    key, kind="drop",
                    payload={"total": numbers[0],
                             "per_claim": numbers[1]},
                    by=funder)
                from ...social.chat.tgbot_buttons import actionable
                return actionable(
                    f"🗳️ that treasury drop ({numbers[0]} sparks) "
                    f"is above the governance threshold "
                    f"({threshold}) — proposal "
                    f"<code>{proposal['id']}</code> is open. A "
                    "SECOND admin approves it:",
                    [[("✅ Approve drop", f"approve {proposal['id']}")]])
            result = surface.open_drop_from_treasury(
                key, numbers[0], numbers[1], by=funder)
            if not result["ok"]:
                if result["reason"] == "funds":
                    return (f"the treasury can't fund that — "
                            f"{result.get('available', 0)} available "
                            "after open-drop reservations.")
                return "that drop doesn't work as written."
            from ...social.chat.typography import Card, sparks

            drop = result["drop"]
            claims = drop["total"] // drop["per_claim"]
            card = Card("announcement", title="Drop", icon="🎁")
            card.line(f"**{sparks(drop['total'])}** from the "
                      f"treasury is up for grabs.")
            card.stats([("Per claim", sparks(drop["per_claim"]),
                         True),
                        ("Claims", str(claims))])
            card.note("tap Claim or send /claim")
            nid = self._record_drop_notice(
                surface, key, message, drop, treasury=True)
            return self._card(
                message, card,
                [[(f"🎁 Claim {drop['per_claim']} sparks", "claim"),
                  ("🪙 My balance", "balance")],
                 [("↩️ Cancel drop", f"undonotice {nid}")]])
        result = surface.open_drop(key, funder, numbers[0],
                                   numbers[1])
        if not result["ok"]:
            if result["reason"] == "funds":
                return "you don't hold that many sparks to fund a drop."
            return "that drop doesn't work as written."
        from ...social.chat.typography import Card, sparks

        drop = result["drop"]
        claims = drop["total"] // drop["per_claim"]
        # The button is authored from THIS drop's state: its label
        # carries the actual per-claim amount of the pool on
        # screen — never a generic claim button.
        card = Card("announcement", title="Drop", icon="🎁")
        card.line(f"**{sparks(drop['total'])}** in the pool.")
        card.stats([("Per claim", sparks(drop["per_claim"]), True),
                    ("Claims", str(claims))])
        card.note("tap Claim or send /claim")
        nid = self._record_drop_notice(
            surface, key, message, drop, treasury=False)
        return self._card(
            message, card,
            [[(f"🎁 Claim {drop['per_claim']} sparks", "claim"),
              ("🪙 My balance", "balance")],
             [("↩️ Cancel drop", f"undonotice {nid}")]])

    def _record_drop_notice(self, surface, key, message, drop,  # type: ignore[no-untyped-def]
                            *, treasury: bool) -> str:
        """Record a drop opening as a §31 notice: an admin can
        cancel the drop from its own announcement (unclaimed
        sparks refund; claims already paid stand)."""
        from ...social.chat.group_notices import drop_notice_text

        funder = str(drop.get("funder") or "")
        if treasury:
            # The notice's member is the admin who opened it
            # ("treasury" is the funding source, not a person).
            user_id = str(message.sender_id
                          or message.sender or "")
            mention = str(message.sender or user_id)
        else:
            user_id = funder
            mention = str(message.sender or funder)
        return surface.record_notice(
            key, user_id=user_id, mention=mention, rule="drop",
            action="drop",
            text=drop_notice_text(
                mention, funder, total=int(drop["total"]),
                per_claim=int(drop["per_claim"]),
                treasury=treasury),
            extra={"drop_id": str(drop["id"]),
                   "treasury": treasury})

    def _control_claim(self, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        result = surface.claim_drop(key, user_id)
        if not result["ok"]:
            return {"no_drop": "no drop is live right now.",
                    "empty": "the pool just ran dry — that was the "
                             "last of it.",
                    "already": "you already claimed from this drop.",
                    "treasury": "the pool couldn't pay out — an "
                                "admin needs to look."}.get(
                        result["reason"], "can't claim right now.")
        from ...social.chat.typography import Card, sparks

        if result.get("closed"):
            # The drop's closing record (§31): the pool is done,
            # said once, in the group where it ran.
            if self._system_send_ok(surface, key, message,
                                    "drop_close"):
                self._send_best_effort(message.chat, (
                    f"🎁 the drop is done — "
                    f"**{sparks(result.get('drop_total') or 0)}** "
                    "claimed in full."))
        card = Card("economy", title="Drop claimed", icon="🎁")
        card.line(f"**+{sparks(result['amount'])}** claimed!")
        card.stats([("Pool left", sparks(result["remaining"]),
                     True),
                    ("Your balance", sparks(result["balance"]),
                     True)])
        return self._card(message, card,
                          [[("🪙 My balance", "balance")]])

    # ---- the full economy loop (bank / daily / crime / casino /
    # boards / governance) --------------------------------------

    def _amount_arg(self, tail, wallet: int) -> int | None:  # type: ignore[no-untyped-def]
        """Parse "<n>|all" against a live balance; None = all of
        it (the Septorch grammar the owner reached for first)."""
        text = str(tail or "").strip().lower()
        if not text or text == "all":
            return None
        digits = text.split()[0]
        return int(digits) if digits.isdigit() else -1

    def _control_deposit(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        amount = self._amount_arg(
            tail, surface.economy.balance(key, user_id))
        if amount is not None and amount < 0:
            return "usage: /deposit 50 — or /deposit all"
        result = surface.deposit(key, user_id, amount)
        if not result["ok"]:
            if result["reason"] == "empty":
                return "your wallet is empty — nothing to deposit."
            return "that deposit doesn't work as written."
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Deposit", icon="🏦")
        card.line(f"deposited **{sparks(result['amount'])}** — "
                  f"now in your vault.")
        card.stats([("Wallet", sparks(result["wallet"]), True),
                    ("Banked", sparks(result["bank"]), True)])
        card.note("robbers can't touch banked sparks")
        return self._card(
            message, card,
            [[("💵 Withdraw", "withdraw all"),
              ("🪙 My balance", "balance")]])

    def _control_withdraw(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        amount = self._amount_arg(
            tail, surface.economy.bank_balance(key, user_id))
        if amount is not None and amount < 0:
            return "usage: /withdraw 50 — or /withdraw all"
        result = surface.withdraw(key, user_id, amount)
        if not result["ok"]:
            if result["reason"] == "empty":
                return "your bank vault is empty."
            return "that withdrawal doesn't work as written."
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Withdraw", icon="💵")
        card.line(f"withdrew **{sparks(result['amount'])}** — "
                  f"back in your wallet.")
        card.stats([("Wallet", sparks(result["wallet"]), True),
                    ("Banked", sparks(result["bank"]), True)])
        return self._card(message, card,
                          [[("🪙 My balance", "balance")]])

    @staticmethod
    def _convert_tail_is_currency(tail) -> bool:  # type: ignore[no-untyped-def]
        """Context-kind guard (the §17 daily/slots pattern):
        /convert is the §21 media command everywhere — but in a
        group, an amount-shaped tail is the §25 currency
        exchange.  Media tails name a conversion (sticker-to-mp4
        …) and never start with a number, a currency word or
        max/all, so the two meanings never collide."""
        text = str(tail or "").strip().lower()
        if not text:
            return True
        first = text.split()[0]
        return first.isdigit() or first in (
            "sparks", "coins", "max", "all")

    def _exchange_standing(self, player):  # type: ignore[no-untyped-def]
        """(level, mastery_steps) for the §25 allowance: the
        member's XP level + mastery tier steps summed across
        their games.  Any failure reads as standing zero — the
        base allowance is a floor, never a refusal."""
        try:
            engine = self._game_engine()
            prof = engine.store.get_for(player)
            from ...games.progression import level_for_xp

            level = level_for_xp(int(prof.xp or 0))
            steps = 0
            try:
                from ...games.achievements import get_game_stats
                from ...games.mastery import mastery_tier

                for row in get_game_stats(engine.db, player.key):
                    steps += mastery_tier(
                        str(row.get("game") or ""), row)[1]
            except Exception:  # noqa: BLE001
                steps = 0
            return level, steps
        except Exception:  # noqa: BLE001
            _log.debug("exchange standing failed", exc_info=True)
            return 1, 0

    def _exchange_quote_card(self, message, surface, quote,  # type: ignore[no-untyped-def]
                             wallet):
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Currency exchange", icon="💱")
        pool = quote["pool"]
        if quote["closed"]:
            card.line("the exchange is **closed this week** — "
                      "this group has no conversion pool "
                      "allocated. An admin opens it with "
                      "/gsettings exchange pool <sparks>.")
        elif not quote["enabled"]:
            card.line("the exchange is **off** in this group.")
        else:
            card.stats([
                ("Rate", f"100 coins → {quote['rate']} sparks",
                 True),
                ("Reverse",
                 f"100 sparks → {quote['reverse_rate_coins']} "
                 f"coins", True),
                ("Pool this week",
                 f"{pool['remaining']:,} of "
                 f"{pool['capacity']:,} sparks left", True),
                ("Your allowance",
                 f"{quote['allowance_left']:,} of "
                 f"{quote['allowance']:,} sparks left", True),
            ])
            if quote["skim_gang"]:
                card.stats([(
                    "Territory skim",
                    f"{quote['skim_pct']}% to {quote['skim_gang']}",
                    True)])
            card.stats([
                ("Your coins", f"{wallet.balance():,} 🪙", True),
                ("Your sparks", sparks(quote["sparks"]), True),
            ])
            card.note("the rate floats with the pool — a fuller "
                      "pool pays better. Unused allowance "
                      "expires when the week resets (Monday).")
        buttons = []
        if quote["enabled"] and not quote["closed"]:
            coins = wallet.balance()
            if coins >= quote["min_coins"] \
                    and quote["allowance_left"] > 0 \
                    and pool["remaining"] > 0:
                chunk = min(coins, 1000)
                buttons.append([(f"🔄 Convert {chunk:,} coins",
                                 f"convert {chunk}")])
            if quote["sparks"] >= 10:
                n = min(int(quote["sparks"]), 100)
                buttons.append([(f"💱 Convert {n} sparks",
                                 f"convert sparks {n}")])
        buttons.append([("🪙 My balance", "balance")])
        return self._card(message, card, buttons)

    def _control_exchange_convert(self, tail, message):  # type: ignore[no-untyped-def]
        """§25: /convert — games coins ⇄ this group's sparks.

        Bare = the quote (rate, pool, allowance — the numbers
        BEFORE converting).  /convert <n> converts n coins in;
        /convert sparks <n> converts n sparks back out;
        max/all takes the whole allowed amount either way."""
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        from ...games.players import Player

        player = Player.from_sender(
            str(getattr(message.chat, "platform", "") or ""),
            user_id, str(message.sender or ""))
        try:
            engine = self._game_engine()
        except Exception:  # noqa: BLE001
            return "the games wallet isn't reachable from here."
        wallet = _GamesCoinWallet(engine, player)
        level, steps = self._exchange_standing(player)
        tokens = str(tail or "").strip().lower().split()
        direction, amount = "in", ...
        if not tokens:
            quote = surface.exchange_quote(
                key, user_id, level=level, mastery_steps=steps)
            return self._exchange_quote_card(
                message, surface, quote, wallet)
        head = tokens[0]
        if head in ("max", "all") and len(tokens) == 1:
            amount = None
        elif head.isdigit() and len(tokens) == 1:
            amount = int(head)
        elif head in ("coins", "sparks") and len(tokens) == 2:
            direction = "in" if head == "coins" else "out"
            amount = None if tokens[1] in ("max", "all") \
                else int(tokens[1]) if tokens[1].isdigit() else ...
        if amount is ...:
            return ("usage: /convert — the rates · /convert "
                    "<coins> — coins → sparks · /convert sparks "
                    "<n> — sparks → coins · /convert max — "
                    "everything your allowance allows")
        result = surface.exchange_convert(
            key, user_id, direction, amount, wallet=wallet,
            level=level, mastery_steps=steps)
        if not result["ok"]:
            return self._exchange_refusal(result)
        from ...social.chat.typography import Card, sparks

        after = result["quote"]
        card = Card("economy", title="Converted", icon="💱")
        if result["direction"] == "in":
            card.line(f"**{result['coins']:,} coins → "
                      f"{sparks(result['sparks'])}**")
            rows = [("Rate", f"{result['rate']} per 100 coins",
                     True)]
            if result["skim"]:
                rows.append(("Territory skim",
                             f"{sparks(result['skim'])} to "
                             f"{result['gang']}", True))
            rows += [
                ("Coins left", f"{result['coins_left']:,} 🪙",
                 True),
                ("Sparks now", sparks(result["sparks_left"]),
                 True),
                ("Allowance left",
                 f"{after['allowance_left']:,} sparks", True),
                ("Pool left",
                 f"{after['pool']['remaining']:,} sparks", True),
            ]
        else:
            card.line(f"**{sparks(result['sparks'])} → "
                      f"{result['coins']:,} coins**")
            rows = [("Rate", f"{result['rate']} per 100 coins",
                     True)]
            if result["skim"]:
                rows.append(("Territory skim",
                             f"{sparks(result['skim'])} to "
                             f"{result['gang']}", True))
            rows += [
                ("Sparks left", sparks(result["sparks_left"]),
                 True),
                ("Coins now", f"{result['coins_left']:,} 🪙",
                 True),
            ]
        card.stats(rows)
        return self._card(message, card,
                          [[("🔄 Exchange", "convert"),
                            ("🪙 My balance", "balance")]])

    @staticmethod
    def _exchange_refusal(result) -> str:  # type: ignore[no-untyped-def]
        """Plain, specific refusals (§25): what bound, how much
        room is left, when it resets.  Nothing moved — ever."""
        reason = result.get("reason")
        quote = result.get("quote") or {}
        if reason == "off":
            return ("the currency exchange is off in this group "
                    "(an admin can open it: /gsettings exchange "
                    "on).")
        if reason == "closed":
            return ("the exchange is closed this week — this "
                    "group has no conversion pool allocated. An "
                    "admin opens it with /gsettings exchange "
                    "pool <sparks>.")
        if reason == "pool":
            left = int(result.get("pool_left") or 0)
            rate = max(1, int(quote.get("rate") or 1))
            return (f"the weekly pool can't cover that — "
                    f"{left:,} sparks left in this week's pool "
                    f"(it refills Monday). At the current rate "
                    f"that's about {left * 100 // rate:,} coins "
                    f"of conversions, total.")
        if reason == "allowance":
            left = int(result.get("allowance_left") or 0)
            return (f"that's past your weekly conversion "
                    f"allowance — {left:,} sparks of allowance "
                    f"left, and it resets Monday. Level, mastery "
                    f"and gang rank all grow it. /convert max "
                    f"takes exactly what's left.")
        if reason == "funds_coins":
            return (f"you have {int(result.get('have') or 0):,} "
                    f"games coins — not enough for that. Win "
                    f"coins at the tables, or convert less.")
        if reason == "funds_sparks":
            return (f"you have {int(result.get('have') or 0):,} "
                    f"sparks here — not enough for that.")
        if reason == "too_small":
            if "min_coins" in result:
                return (f"too few coins — at today's rate the "
                        f"smallest conversion is "
                        f"{result['min_coins']} coins for 1 "
                        f"spark.")
            return (f"too few sparks — the smallest reverse "
                    f"conversion right now is "
                    f"{result.get('min_sparks')} sparks.")
        return ("usage: /convert — the rates · /convert <coins> "
                "— coins → sparks · /convert sparks <n> — "
                "sparks → coins · /convert max")

    def _control_group_daily(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        result = surface.daily_claim(key, user_id)
        if not result["ok"]:
            if result["reason"] == "cooldown":
                return (f"🌅 already claimed today — streak at "
                        f"{result.get('streak', 0)}. Tomorrow it "
                        "grows.")
            return "🌅 no daily paid — the earn cap is hit."
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Daily", icon="🌅")
        card.line(f"daily claimed: **+{sparks(result['amount'])}**")
        card.stats([("Streak", f"{result['streak']} days"),
                    ("Balance", sparks(result["balance"]), True)])
        return self._card(
            message, card,
            [[("🪙 My balance", "balance"), ("🛠️ Work", "work")]])

    def _rob_target(self, surface, key, tail, message):  # type: ignore[no-untyped-def]
        """Who gets robbed (§29): the replied-to author, or the
        named member — by every resolution path.  (The old
        code stringified the reply-sender DICT into a phantom
        member id; the organ returns the real ID.)"""
        res = self._resolver(surface).resolve_command(
            key, message, str(tail or ""))
        return res.member.user_id if res.found else None

    def _control_rob(self, tail, message):  # type: ignore[no-untyped-def]
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        actor = str(message.sender_id or message.sender or "")
        target = self._rob_target(surface, key, tail, message)
        if not target:
            return ("usage: reply to a member with /rob, or "
                    "/rob @member")
        if target == actor:
            return "robbing yourself is just moving sparks around."
        result = surface.rob(key, actor, target,
                             rng=random.Random())
        if not result["ok"]:
            reason = result["reason"]
            if reason == "jailed":
                jail = result.get("jail") or {}
                from ...social.chat.tgbot_buttons import actionable
                return actionable(
                    f"🔒 you're in jail — bail is "
                    f"<b>{jail.get('bail', 0)} sparks</b> to the "
                    "treasury, or wait it out.",
                    [[("🔓 Pay bail", "bail")]])
            if reason == "cooldown":
                return (f"🥷 too soon — the heat dies down in "
                        f"~{max(1, result['wait_s'] // 60)} min.")
            if reason == "empty":
                return "their wallet is empty — nothing to take."
            return "that robbery doesn't work as written."
        from ...social.chat.typography import Card, sparks

        if result.get("success"):
            card = Card("economy", title="Got away", icon="🥷")
            card.line(f"**{sparks(result['amount'])}** lifted.")
            card.stats([("Balance", sparks(result["balance"]),
                         True)])
            card.note("bank it before someone returns the "
                      "favour: /deposit all")
            return self._card(
                message, card,
                [[("🏦 Deposit", "deposit all"),
                  ("🪙 My balance", "balance")]])
        card = Card("economy", title="Caught", icon="🚔")
        card.line("You're in jail for 30 minutes.")
        card.stats([("Bail", sparks(result["bail"]), True)])
        card.note("bail goes to the treasury")
        return self._card(message, card,
                          [[("🔓 Pay bail", "bail")]])

    def _control_bail(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        result = surface.bail(key, user_id)
        if not result["ok"]:
            if result["reason"] == "free":
                return "you're not in jail."
            return (f"bail is {result.get('cost', 0)} sparks and "
                    "your wallet can't cover it.")
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Bail paid", icon="🔓")
        card.line(f"**{sparks(result['cost'])}** to the treasury "
                  f"— you're out.")
        card.stats([("Balance", sparks(result["balance"]), True)])
        return self._card(message, card)

    # ---- the conflict layer (§23/§24/§35) -------------------------
    def _conflict_refusal(self, surface, key, result,  # type: ignore[no-untyped-def]
                          fallback: str = "that doesn't work "
                                            "as written."):
        """One refusal voice for the conflict facade's reasons
        — the mode gate speaks first (nothing in this class
        exists while the group's conflict mode is off)."""
        reason = str(result.get("reason") or "")
        if reason == "mode":
            need = result.get("need") or "crime"
            mode = result.get("mode") or "off"
            return (f"the streets are quiet here — conflict "
                    f"mode is {mode}, and that needs {need}. An "
                    f"admin sets it: /gsettings conflict "
                    f"crime|warfare.")
        if reason == "off":
            return "the group economy is off here."
        if reason == "frozen":
            return ("the conflict state is frozen — its records "
                    "need an admin's eyes before the street "
                    "moves again (see /glog).")
        if reason == "jailed":
            jail = result.get("jail") or {}
            from ...social.chat.tgbot_buttons import actionable
            return actionable(
                f"🔒 you're in jail — bail is "
                f"<b>{jail.get('bail', 0)} sparks</b> to the "
                "treasury, or wait it out.",
                [[("🔓 Pay bail", "bail")]])
        if reason == "hospital":
            ward = result.get("hospital") or {}
            bill = int(ward.get("bill") or 0)
            if bill:
                from ...social.chat.tgbot_buttons import actionable

                return actionable(
                    f"🏥 you're in hospital — the bill is "
                    f"{bill} sparks (a medkit discharges you "
                    f"free).",
                    [[("🔓 Pay the bill", "bail")]])
            return ("🏥 you're in hospital — a medkit "
                    "discharges you, or wait it out.")
        if reason == "cooldown":
            return (f"🥷 too soon — the heat dies down in "
                    f"~{max(1, int(result.get('wait_s') or 0) // 60)}"
                    " min.")
        if reason == "self":
            return "not against yourself."
        if reason == "target_jailed":
            return "they're in jail — out of reach for now."
        return fallback

    def _conflict_target(self, surface, key, tail, message):  # type: ignore[no-untyped-def]
        """(user_id, label) for a crime target — the same
        every-path resolution /rob uses (§29)."""
        target = self._rob_target(surface, key, tail, message)
        if not target:
            return None, ""
        return target, self._estate_member_label(surface, key,
                                                 target)

    def _control_conflict_gear(self, tail, message):  # type: ignore[no-untyped-def]
        """``/gear`` — your loadout; ``/gear buy <item>``,
        ``/gear install <item> treasury|bank <district>|vault``,
        ``/gear use <item>``.  The shelf itself lives in
        /gshop (the arsenal shelf)."""
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        gate = surface._conflict_gate(key, "crime")
        if gate:
            return self._conflict_refusal(surface, key, gate)
        words = str(tail or "").split()
        head = words[0].lower() if words else ""
        if head == "buy" and len(words) >= 2:
            return self._gear_buy_reply(surface, key, message,
                                        user_id, words[1].lower())
        if head == "install" and len(words) >= 3:
            item_id = words[1].lower()
            scope_word = words[2].lower()
            if scope_word == "treasury":
                scope = "treasury"
            elif scope_word == "bank" and len(words) >= 4:
                scope = f"bank:{words[3].lower()}"
            elif scope_word == "vault":
                scope = f"vault:{user_id}"
            else:
                return ("usage: /gear install <item> treasury · "
                        "/gear install <item> bank <district> · "
                        "/gear install <item> vault (your own)")
            can_shared = self._group_admin_ok(message) or bool(
                surface.gang_power_actor(key, user_id))
            out = surface.gear_install(key, user_id, item_id,
                                       scope,
                                       can_shared=can_shared)
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    "you don't hold that, or it isn't a defense "
                    "item — defenses install; breach gear is "
                    "carried.")
            from ...community.group_conflict import ARSENAL
            spec = ARSENAL.get(item_id) or {}
            where = {"treasury": "the group treasury"}.get(
                scope, scope.replace("bank:", "the bank in ")
                .replace("vault:", "your vault"))
            return (f"{spec.get('icon', '🛡️')} {spec.get('name', item_id)} "
                    f"installed on {where} — it stands guard "
                    "until a job wears it down.")
        if head == "use" and len(words) >= 2:
            out = surface.gear_use(key, user_id, words[1].lower())
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"not_hospital": "you're not in hospital.",
                     "not_owned": "you don't hold one of those.",
                     "nothing_worn": "nothing you carry is worn "
                                     "— no repair needed.",
                     "no_kit": "you don't hold a repair kit."}.get(
                        out.get("reason") or "",
                        "that item isn't used that way."))
            if out.get("item") == "medkit":
                return ("🧰 medkit used — you're discharged and "
                        "back on the street.")
            from ...community.group_conflict import ARSENAL
            spec = ARSENAL.get(str(out.get("item") or "")) or {}
            return (f"🔧 {spec.get('name', 'your tool')} restored "
                    "to full durability.")
        # bare /gear: the loadout card
        from ...community.group_conflict import ARSENAL
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card, sparks

        view = surface.loadout_view(key, user_id)
        card = Card("economy", title="Your gear", icon="🧰")
        card.stats([("Street power", str(view["power"])),
                    ("Street rep", str(view["rep"])),
                    ("Wanted", "★" * view["stars"] if view["stars"]
                     else "clean")])
        lines = []
        for inst in view["items"]:
            spec = ARSENAL.get(str(inst.get("item") or "")) or {}
            dur = inst.get("durability")
            state = ("single-use" if spec.get("single_use")
                     else "ready" if dur is None
                     else f"{dur} uses left")
            ratings = ", ".join(
                f"{job} {val}" for job, val in
                (spec.get("ratings") or {}).items() if val)
            lines.append(f"{spec.get('icon', '•')} "
                         f"**{spec.get('name', inst.get('item'))}**"
                         f" · {state}"
                         + (f" · {ratings}" if ratings else ""))
        if lines:
            card.block_lines(lines, title="Loadout")
        else:
            card.line("empty hands — the arsenal shelf is in "
                      "/gshop.")
        for scope, items in (view.get("defenses") or {}).items():
            names = ", ".join(
                (ARSENAL.get(str(i.get("item") or "")) or {}).get(
                    "name", str(i.get("item")))
                for i in items)
            card.line(f"🛡️ guarding your vault: {names}")
        buttons = buttons_from_spec(
            [[("gear", "Arsenal shelf", "gshop"),
              ("balance", "My balance", "balance")]],
            settings=surface.store.settings(key))
        return self._card(message, card, buttons)

    def _gear_buy_reply(self, surface, key, message, user_id,  # type: ignore[no-untyped-def]
                        item_id):
        from ...community.group_conflict import ARSENAL

        spec = ARSENAL.get(item_id)
        if spec is None:
            return "no such item — the arsenal shelf is in /gshop."
        is_admin = self._group_admin_ok(message) \
            or self._operator_or_false(message)
        out = surface.gear_buy(key, user_id, item_id,
                               is_admin=is_admin)
        if not out.get("ok"):
            reason = out.get("reason")
            if reason == "sparks":
                return (f"not enough sparks — {spec['name']} "
                        f"costs {out.get('sparks')} and you hold "
                        f"{surface.economy.balance(key, user_id)}.")
            if reason == "gems":
                return (f"not enough gems — {spec['name']} costs "
                        f"{out.get('gems')} 💎 and you hold "
                        f"{surface.estate.gems_balance(key, user_id)} 💎.")
            if reason == "admin_only":
                return ("the Nuke is minted on an admin's word "
                        "only — some things stay scarce.")
            if reason == "one_per_group":
                return ("this group's economy already holds its "
                        "one Nuke.")
            return self._conflict_refusal(surface, key, out)
        price = []
        if out.get("sparks"):
            price.append(f"{out['sparks']} sparks")
        if out.get("gems"):
            price.append(f"{out['gems']} 💎")
        return (f"{spec.get('icon', '🎁')} **{spec['name']}** is "
                f"yours — {' + '.join(price) or 'free'}, "
                "ledgered. /gear shows your loadout.")

    def _control_hit_verb(self, tail, message, verb):  # type: ignore[no-untyped-def]
        """``/shoot`` · ``/kill`` · ``/assassinate`` @member —
        the geared hit verbs beside /rob (§23): gear decides
        the odds, the odds print on the result, failure is
        jail, success can put the target in hospital."""
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        actor = str(message.sender_id or message.sender or "")
        target, label = self._conflict_target(surface, key, tail,
                                              message)
        if not target:
            return (f"usage: reply to a member with /{verb}, or "
                    f"/{verb} @member")
        out = surface.crime_verb(key, actor, target, verb,
                                 rng=random.Random())
        if not out.get("ok"):
            if out.get("reason") == "weapon":
                need = {0: "", 10: "a real weapon (street pistol "
                        "or better)", 14: "a heavy weapon (hunting "
                        "rifle, strike drone or better)"}.get(
                    int(out.get("need") or 0), "a weapon")
                return (f"you need {need} for that — the arsenal "
                        "shelf is in /gshop.")
            return self._conflict_refusal(surface, key, out)
        from ...social.chat.typography import Card, sparks

        odds = int(round(float(out.get("odds") or 0) * 100))
        if out.get("success"):
            card = Card("economy", title="It lands",
                        icon="🎯")
            card.line(f"**{label}** goes down — you lift "
                      f"**{sparks(int(out.get('take') or 0))}** "
                      f"from their wallet.")
            if verb in ("kill", "assassinate"):
                card.line(f"🏥 they're in hospital — a bill, a "
                          f"medkit, or time gets them out.")
            if out.get("bounty"):
                card.line(f"🎯 bounty collected: "
                          f"**{sparks(int(out['bounty']))}**.")
            card.stats([("The odds were", f"{odds}%")])
            return self._card(message, card)
        card = Card("economy", title="It goes wrong", icon="🚔")
        card.line(f"**{label}** gets away — and you're in jail "
                  f"for {int(out.get('jail_min') or 0)} minutes.")
        card.stats([("The odds were", f"{odds}%")])
        return self._card(message, card,
                          [[("🔓 Pay bail", "bail")]])

    def _control_shoot(self, tail, message):  # type: ignore[no-untyped-def]
        return self._control_hit_verb(tail, message, "shoot")

    def _control_kill(self, tail, message):  # type: ignore[no-untyped-def]
        return self._control_hit_verb(tail, message, "kill")

    def _control_assassinate(self, tail, message):  # type: ignore[no-untyped-def]
        return self._control_hit_verb(tail, message, "assassinate")

    def _control_sabotage(self, tail, message):  # type: ignore[no-untyped-def]
        """``/sabotage @member`` — wreck their best property;
        the estate organ bills the damage through its own
        repair machinery (insurance answers first)."""
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        actor = str(message.sender_id or message.sender or "")
        target, label = self._conflict_target(surface, key, tail,
                                              message)
        if not target:
            return ("usage: reply to a member with /sabotage, or "
                    "/sabotage @member")
        out = surface.crime_sabotage(key, actor, target,
                                     rng=random.Random())
        if not out.get("ok"):
            if out.get("reason") == "no_property":
                return f"{label} owns no property to wreck."
            return self._conflict_refusal(surface, key, out)
        from ...social.chat.typography import Card, sparks

        if out.get("success"):
            card = Card("economy", title="Wrecked", icon="🔨")
            card.line(f"**{label}**'s {out.get('property') or 'property'}"
                      f" is wrecked — a "
                      f"**{sparks(int(out.get('bill') or 0))}** "
                      "repair bill lands on them (insurance "
                      "answers first, if they hold a policy).")
            return self._card(message, card)
        card = Card("economy", title="Caught", icon="🚔")
        card.line("caught in the act — 20 minutes in jail.")
        return self._card(message, card,
                          [[("🔓 Pay bail", "bail")]])

    def _control_heist(self, tail, message):  # type: ignore[no-untyped-def]
        """``/heist start treasury|bank <district>|vault
        @member`` · ``/heist join`` · ``/heist go`` — the crew
        score (§23): the crew's combined loadout against the
        target's installed defenses, odds printed before and
        after."""
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        gate = surface._conflict_gate(key, "crime")
        if gate:
            return self._conflict_refusal(surface, key, gate)
        words = str(tail or "").split()
        head = words[0].lower() if words else ""
        if head == "start" and len(words) >= 2:
            kind = words[1].lower()
            if kind == "treasury":
                out = surface.heist_start(key, user_id,
                                          "treasury", "")
            elif kind == "bank" and len(words) >= 3:
                out = surface.heist_start(key, user_id, "bank",
                                          words[2].lower())
            elif kind == "vault":
                target, _label = self._conflict_target(
                    surface, key, " ".join(words[2:]), message)
                if not target:
                    return ("usage: /heist start vault @member")
                out = surface.heist_start(key, user_id, "vault",
                                          target)
            else:
                return ("usage: /heist start treasury · /heist "
                        "start bank <district> · /heist start "
                        "vault @member")
            if not out.get("ok"):
                if out.get("reason") == "tier":
                    return ("the big scores — the treasury and "
                            "the banks — are gang work: they need "
                            "a gang of 5+ behind the crew. Member "
                            "vaults take any crew of 2–5.")
                if out.get("reason") == "district":
                    from ...community.group_real_estate import \
                        DISTRICTS
                    return ("which bank? districts: "
                            + ", ".join(
                                f"{d} ({v['name']})"
                                for d, v in DISTRICTS.items()))
                return self._conflict_refusal(surface, key, out)
            lobby = out["heist"]
            return self._heist_card(surface, key, message, lobby,
                                    lead="crew forming — crews "
                                         "of 2–5; join below, "
                                         "then run it.")
        if head == "join":
            out = surface.heist_join(key, user_id)
            if not out.get("ok"):
                if out.get("reason") == "not_found":
                    return "no heist is forming right now."
                if out.get("reason") == "full":
                    return "that crew is full (5 ride at most)."
                if out.get("reason") in ("already", "already_crew"):
                    return "you're already on this crew."
                if out.get("reason") == "expired":
                    return "that crew waited too long — it broke up."
                return self._conflict_refusal(surface, key, out)
            return self._heist_card(surface, key, message,
                                    out["heist"],
                                    lead="you're on the crew.")
        if head == "go":
            out = surface.heist_go(key, user_id,
                                   rng=random.Random())
            if not out.get("ok"):
                if out.get("reason") == "not_found":
                    return ("you're not leading an open crew — "
                            "/heist start one first.")
                if out.get("reason") == "crew_small":
                    return ("a heist needs a crew of 2–5 — "
                            "/heist join brings riders.")
                return self._conflict_refusal(surface, key, out)
            return self._heist_result(surface, key, message, out)
        lobby = surface.heist_open(key)
        if lobby is None:
            return ("no heist is forming. Start one: /heist "
                    "start treasury · /heist start bank "
                    "<district> · /heist start vault @member — "
                    "crews of 2–5, gear decides the odds.")
        return self._heist_card(surface, key, message, lobby,
                                lead="a crew is forming:")

    def _heist_card(self, surface, key, message, lobby, *, lead):  # type: ignore[no-untyped-def]
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card

        target = lobby.get("target") or {}
        kind = str(target.get("kind") or "")
        if kind == "treasury":
            where = "the group treasury"
        elif kind == "bank":
            from ...community.group_real_estate import DISTRICTS
            where = f"the bank in {DISTRICTS.get(str(target.get('id')), {}).get('name', target.get('id'))}"
        else:
            where = f"{self._estate_member_label(surface, key, str(target.get('id') or ''))}'s vault"
        preview = surface.conflict.heist_preview(
            key, str(lobby.get("id") or ""))
        card = Card("economy", title="Heist", icon="💰")
        card.line(lead)
        card.line(f"target: **{where}** · crew: "
                  + ", ".join(self._estate_member_label(
                      surface, key, m)
                      for m in lobby.get("crew") or []))
        if preview:
            card.stats([
                ("Crew offense", str(preview["offense"])),
                ("Target defense", str(preview["defense"])),
                ("Success odds",
                 f"{int(round(preview['odds'] * 100))}%")])
            for note in preview.get("notes") or []:
                card.line(f"⚡ {note}")
            if preview.get("heat_stars"):
                card.line(f"🚔 police heat on this crew: "
                          f"{preview['heat_stars']:.0f}★ — heat "
                          "hardens the target.")
        buttons = buttons_from_spec(
            [[("heist", "Join the crew", "heist join"),
              ("heist", "Run it", "heist go")]],
            settings=surface.store.settings(key))
        return self._card(message, card, buttons)

    def _heist_result(self, surface, key, message, out):  # type: ignore[no-untyped-def]
        from ...social.chat.typography import Card, sparks

        outcome = str(out.get("outcome") or "")
        target = out.get("target") or {}
        kind = str(target.get("kind") or "")
        where = {"treasury": "the treasury",
                 "bank": "the bank"}.get(
                     kind,
                     f"{self._estate_member_label(surface, key, str(target.get('id') or ''))}'s vault")
        odds = int(round(float(out.get("odds") or 0) * 100))
        card = Card("economy", title="Heist", icon="💰")
        if outcome == "success":
            card.line(f"✅ the crew cracks {where} — "
                      f"**{sparks(int(out.get('take') or 0))}** "
                      "split between them.")
        elif outcome == "partial":
            card.line(f"🌓 the crew gets into {where} but has "
                      f"to run early — "
                      f"**{sparks(int(out.get('take') or 0))}** "
                      "grabbed, nobody caught.")
        elif outcome == "decoy":
            card.line(f"🎁 the crew cracks {where}… and finds a "
                      "decoy vault full of confetti. The real "
                      "money never moved.")
        elif outcome == "empty":
            card.line(f"🕳️ the crew gets in — {where} holds "
                      "nothing worth taking.")
        else:
            card.line(f"🚔 BUSTED at {where} — the whole crew "
                      "is in jail for 45 minutes.")
        card.stats([("Offense", str(out.get("offense"))),
                    ("Defense", str(out.get("defense"))),
                    ("The odds were", f"{odds}%")])
        for note in out.get("notes") or []:
            card.line(f"⚡ {note}")
        buttons = [[("🔓 Pay bail", "bail")]] \
            if outcome == "bust" else None
        return self._card(message, card, buttons)

    def _control_gang(self, tail, message):  # type: ignore[no-untyped-def]
        """``/gang`` — the whole gang life (§24): create
        <lifestyle> <name> · join <name> · leave · promote/
        demote @member · deposit/withdraw · job · armory add
        <item> · claim <district> · tax N (the power gang) ·
        list.  §36 adds: register · chapter <name> · hq
        [claim|build|move] · vault deposit/withdraw N ·
        armory [store|draw <item>] · code · redeem <code> ·
        side <gang>."""
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        gate = surface._conflict_gate(key, "crime")
        if gate:
            return self._conflict_refusal(surface, key, gate)
        words = str(tail or "").split()
        head = words[0].lower() if words else ""
        organ = surface.conflict
        if head == "create" and len(words) >= 3:
            from ...community.group_conflict import LIFESTYLES
            lifestyle = words[1].lower()
            if lifestyle not in LIFESTYLES:
                return ("pick a lifestyle: "
                        + " · ".join(
                            f"{k} ({v['name']})"
                            for k, v in LIFESTYLES.items())
                        + " — usage: /gang create <lifestyle> "
                          "<name>")
            name = " ".join(words[2:])
            out = surface.gang_create(key, user_id, name,
                                      lifestyle)
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"name": "gang names run 3–32 characters.",
                     "already_in": "you're already in a gang — "
                                   "leave it first.",
                     "taken": "that name is taken in this group.",
                     "lifestyle": "unknown lifestyle."}.get(
                        out.get("reason") or "",
                        "that gang doesn't form."))
            return self._gang_card(surface, key, message, user_id,
                                   lead=f"🏴 **{name}** is formed "
                                        "— recruit with /gang "
                                        "join; crews grow: 5+ is a "
                                        "gang, 15+ a syndicate.")
        if head == "join" and len(words) >= 2:
            out = surface.gang_join(key, user_id,
                                    " ".join(words[1:]))
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"not_found": "no gang by that name here — "
                                  "/gang list shows them.",
                     "cross_taken": (
                         f"your cross-group seat is taken — "
                         f"you ride with "
                         f"**{out.get('holder') or 'another gang'}** "
                         "across groups. One cross-group gang "
                         "plus one local crew is the loyalty "
                         "shape; leave that one first."),
                     "local_taken": (
                         f"your local seat is taken — you ride "
                         f"with **{out.get('holder') or 'another gang'}** "
                         "as your single-group gang. One local "
                         "gang plus one cross-group gang is "
                         "the loyalty shape; leave that one "
                         "first."),
                     "already": "you're already in that gang.",
                     "already_in": "you're already in a gang."}.get(
                        out.get("reason") or "", "can't join."))
            return self._gang_card(surface, key, message, user_id,
                                   lead="you're in. 🏴")
        if head == "leave":
            out = surface.gang_leave(key, user_id)
            if not out.get("ok"):
                return "you're not in a gang."
            if out.get("dissolved"):
                return ("the gang is dissolved — its treasury "
                        "went to its last leader, its districts "
                        "are open ground again.")
            return "you've left the gang."
        if head == "list":
            rows = surface.gang_list(key)
            if not rows:
                return ("no gangs here yet — /gang create "
                        "<lifestyle> <name> starts one.")
            from ...social.chat.typography import Card, sparks

            card = Card("economy", title="The gangs", icon="🏴")
            lines = []
            for row in rows:
                gang = row["gang"]
                from ...community.group_conflict import (
                    LIFESTYLES, gang_tier)
                life = LIFESTYLES.get(gang["lifestyle"]) or {}
                lines.append(
                    f"{life.get('icon', '🏴')} **{gang['name']}** "
                    f"· {life.get('name', gang['lifestyle'])} · "
                    f"{gang_tier(len(gang.get('members') or []))} "
                    f"of {len(gang.get('members') or [])} · "
                    f"strength {row['strength']} · treasury "
                    f"{sparks(int(gang.get('treasury') or 0))}")
            card.block_lines(lines)
            return self._card(message, card)
        if head in ("promote", "demote") and len(words) >= 2:
            target, _label = self._conflict_target(
                surface, key, " ".join(words[1:]), message)
            if not target:
                return f"usage: /gang {head} @member"
            out = organ.gang_set_role(
                key, user_id, target,
                "officer" if head == "promote" else "member",
                now=time.time())
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"not_leader": "only the leader moves ranks.",
                     "not_member": "they're not in your gang.",
                     "self": "not yourself."}.get(
                        out.get("reason") or "", "no."))
            return (f"{'⬆️' if head == 'promote' else '⬇️'} "
                    f"{self._estate_member_label(surface, key, target)}"
                    f" is now "
                    f"{'an officer' if head == 'promote' else 'a member'}"
                    " of the gang.")
        if head == "deposit":
            amount = self._amount_arg(
                words[1] if len(words) > 1 else "",
                surface.economy.balance(key, user_id))
            if amount is not None and amount < 0:
                return "usage: /gang deposit 100 — or all"
            if amount is None:
                amount = surface.economy.balance(key, user_id)
            out = organ.gang_deposit(key, user_id, amount,
                                     now=time.time())
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_gang": "you're not in a gang.",
                     "funds": "your wallet can't cover that."}.get(
                        out.get("reason") or "", "can't deposit."))
            from ...social.chat.typography import sparks
            return (f"🏦 {sparks(out['amount'])} into the gang "
                    f"treasury — it now holds "
                    f"{sparks(int(out['gang'].get('treasury') or 0))}.")
        if head == "withdraw" and len(words) >= 2 \
                and words[1].isdigit():
            out = organ.gang_withdraw(
                key, user_id, int(words[1]),
                now=time.time())
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_gang": "you're not in a gang.",
                     "rank": "the gang treasury moves on the "
                             "leader's or an officer's word.",
                     "funds": "the gang treasury doesn't hold "
                              "that."}.get(out.get("reason") or "",
                                           "can't withdraw."))
            from ...social.chat.typography import sparks
            return f"💵 {sparks(out['amount'])} from the gang treasury to you."
        if head == "job":
            import random

            out = organ.gang_job(key, user_id,
                                 rng=random.Random(),
                                 now=time.time())
            if not out.get("ok"):
                if out.get("reason") == "cooldown":
                    return ("the crew rests — gang work pays "
                            "hourly.")
                return self._conflict_refusal(
                    surface, key, out,
                    "you're not in a gang — gang work is members' "
                    "work.")
            from ...social.chat.typography import sparks
            return (f"💼 {out['job']} — "
                    f"**+{sparks(int(out['amount']))}** for the "
                    f"{out['gang']['name']}.")
        if head == "armory" and len(words) >= 3 \
                and words[1].lower() == "add":
            out = organ.gang_armory_add(
                key, user_id, words[2].lower(),
                now=time.time())
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_gang": "you're not in a gang.",
                     "not_owned": "you don't hold that item."}.get(
                        out.get("reason") or "", "can't donate."))
            from ...community.group_conflict import ARSENAL
            spec = ARSENAL.get(words[2].lower()) or {}
            return (f"{spec.get('icon', '🎁')} {spec.get('name', words[2])}"
                    " joins the gang armory — it fights the "
                    "gang's wars now.")
        if head == "claim" and len(words) >= 2:
            out = surface.territory_claim(key, user_id,
                                          words[1].lower())
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_gang": "you're not in a gang.",
                     "tier": "territory is gang work — crews "
                             "(under 5) can't hold districts.",
                     "rank": "district claims are the leader's "
                             "or an officer's call.",
                     "held": "that district is held — districts "
                             "change hands by war, not by "
                             "asking.",
                     "funds": "claiming costs 100 sparks from "
                              "the gang treasury.",
                     "district": "districts: old_town, "
                                 "market_mile, harbour, heights."}
                    .get(out.get("reason") or "", "can't claim."))
            from ...community.group_real_estate import DISTRICTS
            return (f"🚩 **{out['gang']['name']}** claims "
                    f"**{DISTRICTS[words[1].lower()]['name']}** — "
                    "it pays the gang daily while the upkeep is "
                    "paid.")
        if head == "tax" and len(words) >= 2 \
                and words[1].rstrip("%").isdigit():
            out = organ.set_gang_tax(
                key, user_id, int(words[1].rstrip("%")),
                now=time.time())
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"not_power": "only the gang IN POWER sets "
                                  "the tax — hold 2+ districts "
                                  "and be the strongest to rule.",
                     "range": "the transfer tax lives between "
                              "0% and 10% — even a ruling gang "
                              "can't strangle the economy."}.get(
                        out.get("reason") or "", "can't set tax."))
            return (f"🏛️ {out['gang']['name']} sets the group's "
                    f"transfer tax to {out['pct']}% — /pay fees "
                    "follow it while they hold power.")
        # ── §36: cross-group gangs — chapters, HQ, codes ──
        if head == "register":
            out = surface.gang_register_here(key, user_id)
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_gang": "you're not in a gang here — "
                                "there's nothing to register.",
                     "name_registered": "that name is already "
                                        "held by a registered "
                                        "gang elsewhere — this "
                                        "gang runs local-only "
                                        "under it.",
                     "frozen": "the gang registry is frozen."}
                    .get(out.get("reason") or "",
                         "can't register."))
            if out.get("already"):
                return (f"🌐 **{out['gang']['name']}** is "
                        "already in the global registry.")
            return (f"🌐 **{out['gang']['name']}** is in the "
                    "global registry now — one chapter (here), "
                    "its history and treasury exactly as they "
                    "stood. It expands as it climbs tiers: "
                    "/gang chapter <name> founds a chapter where "
                    "its officers stand.")
        if head == "chapter" and len(words) >= 2:
            name = " ".join(words[1:])
            out = surface.gang_found_chapter(key, user_id, name)
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_gang": f"no registered gang called "
                                f"{name}.",
                     "rank": "chapters are founded by the "
                             "gang's leader or an officer.",
                     "already_chapter": f"{name} already has "
                                        "a chapter here.",
                     "already_in": "you're in a gang in this "
                                   "group already — leave it "
                                   "before founding a chapter.",
                     "name_taken": "a different gang holds "
                                   "that name in this group.",
                     "capacity": (
                         f"{out.get('gang') or name} is "
                         f"{out.get('tier') or 'crew'} tier — "
                         f"{out.get('used') or 0} of "
                         f"{out.get('slots')} chapter slots "
                         "used. Grow the roster to climb a "
                         "tier and earn more reach."),
                     "frozen": "the gang registry is frozen."}
                    .get(out.get("reason") or "",
                         "can't found a chapter."))
            return (f"🌐 **{out['gang']['name']}** opens a "
                    "chapter here — you lead it locally. The "
                    "gang now stands in "
                    f"{len(out['gang'].get('chapters') or {})} "
                    "group(s); its treasury and territory here "
                    "are this chapter's own.")
        if head == "hq":
            sub = words[1].lower() if len(words) > 1 else ""
            if sub == "claim":
                out = surface.gang_hq_designate(key, user_id)
                if not out.get("ok"):
                    return self._conflict_refusal(
                        surface, key, out,
                        {"no_gang": "you're not in a gang here.",
                         "not_leader": "the HQ is designated "
                                       "by the gang's global "
                                       "leader.",
                         "no_chapter": "your gang has no "
                                       "chapter here.",
                         "hq_exists": "your gang already has "
                                      "an HQ — /gang hq move "
                                      "relocates it (a tier "
                                      "drops; it costs).",
                         "frozen": "the gang registry is "
                                   "frozen."}.get(
                            out.get("reason") or "",
                            "can't designate."))
                return ("🏰 this chapter is your gang's "
                        "headquarters now. Build it: /gang hq "
                        "build (hideout first) — the vault, the "
                        "shared armory and the defense of this "
                        "group all live here.")
            if sub == "build":
                out = surface.gang_hq_buy(key, user_id)
                if not out.get("ok"):
                    return self._conflict_refusal(
                        surface, key, out,
                        {"no_gang": "you're not in a gang here.",
                         "no_hq": "your gang has no HQ — "
                                  "/gang hq claim where it "
                                  "should stand.",
                         "rank": "HQ building is the chapter "
                                 "leader's or an officer's call.",
                         "max_tier": "fortress — the top. "
                                     "Nothing above it to buy.",
                         "funds": (
                             f"the next tier costs "
                             f"{out.get('cost')} sparks from "
                             "the gang treasury (it holds "
                             f"{out.get('treasury')})."),
                         "frozen": "the gang registry is "
                                   "frozen."}.get(
                            out.get("reason") or "",
                            "can't build."))
                return (f"🏰 HQ rises: **{out['tier_name']}** — "
                        f"{out['cost']} sparks from the gang "
                        "treasury. Its vault deepens, its walls "
                        "count in every heist and war read "
                        "here, and raiders will come for it.")
            if sub == "move":
                out = surface.gang_hq_move(key, user_id)
                if not out.get("ok"):
                    return self._conflict_refusal(
                        surface, key, out,
                        {"no_gang": "you're not in a gang here.",
                         "not_leader": "relocation is the "
                                       "gang's global leader's "
                                       "call.",
                         "no_hq": "your gang has no HQ to "
                                  "move.",
                         "no_chapter": "your gang has no "
                                       "chapter here to move "
                                       "the HQ to.",
                         "already_hq": "the HQ already stands "
                                       "here.",
                         "funds": (
                             f"relocation costs "
                             f"{out.get('cost')} sparks from "
                             "THIS chapter's treasury (it holds "
                             f"{out.get('treasury')})."),
                         "frozen": "the gang registry is "
                                   "frozen."}.get(
                            out.get("reason") or "",
                            "can't relocate."))
                return (f"🏰 the HQ relocates here — "
                        f"{out['cost']} sparks spent, the "
                        f"fortress drops to "
                        f"**{out['tier_name']}** (a fortress is "
                        "built, not carried), and the old vault "
                        f"released {out['released']} sparks to "
                        "the old chapter's treasury.")
            view = surface.gang_hq_status(key, user_id)
            if not view.get("ok"):
                return "you're not in a gang here."
            hq = view.get("hq")
            if hq is None:
                return ("🏰 your gang has no headquarters yet "
                        "— the global leader designates one "
                        "with /gang hq claim, standing in the "
                        "chapter that should hold it.")
            from ...community.group_conflict import HQ_TIERS
            return (f"🏰 **HQ** — {hq['tier_name']} "
                    f"(tier {hq['tier']} of "
                    f"{len(HQ_TIERS) - 1}) · vault "
                    f"{hq['vault']} sparks held in reserve · "
                    f"chapter treasury {hq['treasury']} · "
                    f"{len(hq['armory'])} item(s) in the "
                    "shared armory. /gang hq build raises it; "
                    "/gang vault works the reserve.")
        if head == "vault" and len(words) >= 3 \
                and words[1].lower() in ("deposit", "withdraw") \
                and words[2].isdigit():
            verb = words[1].lower()
            out = surface.gang_hq_vault(
                key, user_id, verb, int(words[2]))
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_gang": "you're not in a gang here.",
                     "no_hq": "your gang has no HQ.",
                     "hq_elsewhere": "the vault lives at the "
                                     "HQ — work it standing "
                                     "in the HQ's group.",
                     "not_member": "the vault is for HQ "
                                   "chapter members.",
                     "rank": "the vault opens for the HQ "
                             "chapter's leader and officers.",
                     "funds": "there isn't that much there.",
                     "amount": "a positive amount.",
                     "frozen": "the gang registry is frozen."}
                    .get(out.get("reason") or "",
                         "the vault won't move."))
            return (f"🏦 {verb} done — the HQ vault now holds "
                    f"{out['vault']} sparks in reserve.")
        if head == "armory":
            sub = words[1].lower() if len(words) > 1 else "list"
            if sub == "list":
                out = surface.gang_hq_armory(key, user_id,
                                             "list")
                if not out.get("ok"):
                    return ("your gang has no HQ — the shared "
                            "armory lives at the headquarters.")
                from ...community.group_conflict import ARSENAL
                names = [str((ARSENAL.get(
                    str(i.get('item') or ''), {}).get('name'))
                    or i.get("item")) for i in out["armory"]]
                return ("🧰 the shared armory (at HQ): "
                        + (", ".join(names) if names
                           else "empty — /gang armory store "
                                "<item> shelves your gear for "
                                "the whole gang")
                        + ". Draw from any chapter: /gang "
                          "armory draw <item>.")
            if sub in ("store", "draw") and len(words) >= 3:
                item = words[2].lower()
                out = surface.gang_hq_armory(
                    key, user_id, sub, item)
                if not out.get("ok"):
                    return self._conflict_refusal(
                        surface, key, out,
                        {"no_gang": "you're not in a gang "
                                    "here.",
                         "not_member": "the armory serves "
                                       "the gang's roster.",
                         "no_hq": "your gang has no HQ — "
                                  "the shared armory lives "
                                  "there.",
                         "no_chapter": "you hold no chapter "
                                       "seat here.",
                         "not_owned": "that item isn't in "
                                      "your loadout.",
                         "not_stocked": "the armory doesn't "
                                        "hold that item.",
                         "frozen": "the gang registry is "
                                   "frozen."}.get(
                            out.get("reason") or "",
                            "the armory won't move."))
                return (f"🧰 {item} "
                        + ("shelved in the shared armory — "
                           "any chapter can draw it."
                           if sub == "store" else
                           "drawn from the shared armory — "
                           "it's in your loadout here now."))
            return ("🧰 /gang armory · /gang armory store "
                    "<item> · /gang armory draw <item>")
        if head == "code":
            out = surface.gang_claim_code(key, user_id)
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_gang": "you're not in a gang here.",
                     "not_member": "codes are minted by the "
                                   "gang's roster.",
                     "frozen": "the gang registry is frozen."}
                    .get(out.get("reason") or "",
                         "can't mint a code."))
            return (f"🎟️ your gang claim code: "
                    f"`{out['code']}` — redeem it from your "
                    "other platform with /gang redeem "
                    f"{out['code']} standing in a group where "
                    "the gang has a chapter, and that identity "
                    "joins this roster. One use, 30 minutes.")
        if head == "redeem" and len(words) >= 2:
            out = surface.gang_redeem_code(
                key, user_id, words[1])
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"bad_code": "that code doesn't ring any "
                                 "bell — codes are one-shot; "
                                 "mint a fresh one with /gang "
                                 "code.",
                     "expired": "that code expired — mint a "
                                "fresh one with /gang code.",
                     "no_chapter": "the code's real — your "
                                   "identity is bound — but "
                                   "the gang has no chapter in "
                                   "this group to seat you in.",
                     "already_in": "you're in another gang in "
                                   "this group — leave it "
                                   "first; your identity stays "
                                   "bound.",
                     "gone": "the minter's roster seat is "
                             "gone.",
                     "no_gang": "the gang behind that code "
                                "is gone.",
                     "frozen": "the gang registry is frozen."}
                    .get(out.get("reason") or "",
                         "can't redeem."))
            return (f"🎟️ bound and seated — this identity now "
                    f"rides with **{out['gang']['name']}** "
                    "here, one person across platforms.")
        if head == "side" and len(words) >= 2:
            name = " ".join(words[1:])
            out = surface.gang_war_side(key, user_id, name)
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_gang": "you're not in a gang here.",
                     "no_war": "no live inter-group war "
                               "against that gang.",
                     "not_dual": "you hold one gang in this "
                                 "fight — your side is your "
                                 "gang; declarations are for "
                                 "members whose two gangs meet.",
                     "not_side": "that gang isn't a side in "
                                 "this war.",
                     "frozen": "the gang registry is frozen."}
                    .get(out.get("reason") or "",
                         "can't declare."))
            return ("⚔️ declared — in this war you fight for "
                    f"**{name}** only; /war join seats you on "
                    "that side, and the other gang gets "
                    "nothing from you.")
        gang = organ.gang_of(key, user_id)
        if gang is None:
            from ...community.group_conflict import LIFESTYLES
            return ("you're not in a gang. Form one: /gang "
                    "create <lifestyle> <name> — lifestyles: "
                    + " · ".join(f"{k}" for k in LIFESTYLES)
                    + ". Or /gang list to see who's hiring.")
        return self._gang_card(surface, key, message, user_id)

    def _gang_card(self, surface, key, message, user_id, *,  # type: ignore[no-untyped-def]
                   lead: str = ""):
        from ...community.group_conflict import LIFESTYLES
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card, sparks

        view = surface.gang_card(key, user_id)
        gang = view["gang"]
        life = LIFESTYLES.get(gang["lifestyle"]) or {}
        ranks = life.get("ranks") or ("Leader", "Officer",
                                      "Member")
        role = (ranks[0] if gang.get("leader") == user_id else
                ranks[1] if user_id in (gang.get("officers") or [])
                else ranks[2])
        card = Card("economy",
                    title=f"{life.get('icon', '🏴')} {gang['name']}",
                    icon="🏴")
        if lead:
            card.line(lead)
        card.line(f"*{life.get('blurb', '')}*")
        strength = view["strength"]
        card.stats([
            ("Lifestyle", str(life.get("name",
                                       gang["lifestyle"]))),
            ("Tier", f"{view['tier']} · "
                     f"{len(gang.get('members') or [])} members"),
            ("Your rank", role),
            ("Gang treasury",
             sparks(int(gang.get("treasury") or 0)), True),
            ("Strength", str(strength["total"])),
            ("Territory",
             ", ".join(view["territory"]) or "none"),
            ("War record",
             f"{int(gang.get('wins') or 0)}W · "
             f"{int(gang.get('losses') or 0)}L"),
            ("In power", "yes — /gang tax N sets the group's "
                         "transfer tax" if view["power"]
             else "no"),
        ])
        glob = view.get("global")
        if glob:
            # §36: the standing ABOVE this room — the gang as
            # the registry knows it, then the chapter you're
            # standing in (the stats above are this chapter's).
            hq = glob.get("hq")
            card.stats([
                ("🌐 Global tier",
                 f"{glob['tier']} · {glob['roster_size']} "
                 "members worldwide"),
                ("Prestige", str(glob["prestige"])),
                ("Chapters",
                 f"{glob['chapter_count']} "
                 f"({glob['capacity_text']})"),
                ("Your global rank", str(glob["your_role"])),
                ("HQ", (f"{hq['tier_name']} — its vault, "
                        "armory and defense stand in another "
                        "of your chapters"
                        if hq and hq["group"] != key else
                        "🏰 HERE — this chapter is the "
                        "headquarters"
                        if hq else "none designated")),
            ])
            card.note(
                "chapter perks (this room): the "
                f"{life.get('name', gang['lifestyle'])} way — "
                "gang jobs pay its rates here, its gear "
                "affinities sharpen its members, its war style "
                "and heat profile ride every local read")
        card.note("strength = members' power + armory + "
                  "territory + treasury depth + war record")
        buttons = buttons_from_spec(
            [[("gang", "Gang work", "gang job"),
              ("treasury", "Deposit 100", "gang deposit 100")],
             [("war", "War status", "war"),
              ("gang", "All gangs", "gang list")]],
            settings=surface.store.settings(key))
        return self._card(message, card, buttons)

    def _control_war(self, tail, message):  # type: ignore[no-untyped-def]
        """``/war`` — wars between gangs (§23): declare
        <gang> (here, or in another claimed group) · accept ·
        decline · join · spoils <district>.  Rounds fight
        themselves on the group cycle."""
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        gate = surface._conflict_gate(key, "crime")
        if gate:
            return self._conflict_refusal(surface, key, gate)
        words = str(tail or "").split()
        head = words[0].lower() if words else ""
        if head == "declare" and len(words) >= 2:
            name = " ".join(words[1:])
            out = surface.war_declare(key, user_id, name)
            if out.get("ok"):
                war = out["war"]
                return (f"⚔️ **{out['mine']['name']}** declares "
                        f"war on **{out['theirs']['name']}** — "
                        f"{int(war.get('escrow') or 0)} sparks "
                        "escrowed from your gang treasury. Their "
                        "leadership answers with /war accept.")
            if out.get("reason") == "not_found":
                inter = surface.war_declare_inter(key, user_id,
                                                  name)
                if inter.get("ok"):
                    war = inter["war"]
                    return (f"⚔️ **{inter['mine']['name']}** "
                            f"declares war across groups — on "
                            f"**{inter['theirs']['name']}** in "
                            "another claimed group. Escrow is "
                            "down; their leadership answers "
                            "there with /war accept.")
                return self._conflict_refusal(
                    surface, key, inter,
                    {"tier": "inter-group wars are gang-tier "
                             "work (5+ members) on both sides.",
                     "tier_target": "that gang is still a crew "
                                    "— too small for an "
                                    "inter-group war.",
                     "busy": "your gang is already at war.",
                     "funds": "the war escrow comes from your "
                              "gang treasury — it can't cover "
                              "it.",
                     "book": "the war book is unreadable — "
                             "inter-group wars pause until an "
                             "admin looks (see /glog).",
                     "frozen": "that group's conflict state is "
                               "frozen."}.get(
                        inter.get("reason") or "",
                        f"no gang called {name} here or in any "
                        "claimed group."))
            return self._conflict_refusal(
                surface, key, out,
                {"no_gang": "you're not in a gang.",
                 "rank": "war is declared by the leader or an "
                         "officer.",
                 "self": "not on your own gang.",
                 "busy": "a gang in this war can't start or "
                         "take another until it ends.",
                 "funds": "the war escrow comes from your gang "
                          "treasury — it can't cover it."}.get(
                    out.get("reason") or "", "can't declare."))
        if head in ("accept", "decline"):
            accept = head == "accept"
            out = surface.war_answer(key, user_id, accept)
            if not out.get("ok") and out.get("reason") == "no_war":
                out = surface.war_answer_inter(key, user_id,
                                               accept)
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_war": "no declared war is waiting on "
                               "your gang.",
                     "no_gang": "you're not in a gang.",
                     "rank": "the answer comes from the leader "
                             "or an officer.",
                     "funds": "matching the escrow takes gang "
                              "treasury funds you don't have."}
                    .get(out.get("reason") or "", "can't answer."))
            if accept:
                return ("⚔️ the war is ON — members opt in with "
                        "/war join; rounds fight on the cycle "
                        "until one side takes the best of it.")
            return ("🏳️ war declined — the challenger's escrow "
                    "is returned.")
        if head == "raid" and len(words) >= 2:
            name = " ".join(words[1:])
            out = surface.war_raid_hq(key, user_id, name)
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_gang": "you're not in a gang.",
                     "tier": "HQ raids are gang-tier work "
                             "(5+ members) — crews don't march "
                             "on fortresses.",
                     "not_found": f"no registered gang called "
                                  f"{name}.",
                     "self": "that's your own gang's HQ.",
                     "no_hq": f"{name} has no headquarters to "
                              "raid — yet.",
                     "local": "their HQ stands HERE — that's "
                              "an in-group war (/war declare), "
                              "not a raid.",
                     "busy": "your gang is already at war.",
                     "funds": "the raid escrow (double the "
                              "war escrow) comes from your "
                              "gang treasury — it can't "
                              "cover it.",
                     "gone": "the HQ chapter is gone.",
                     "frozen": "the war book is unreadable."}
                    .get(out.get("reason") or "",
                         "can't raid."))
            war = out["war"]
            return (f"🏰⚔️ **{out['mine']['name']}** marches "
                    f"on the **{out['theirs']['name']}** "
                    f"headquarters — a "
                    f"{out['hq']['tier_name']} fortress with "
                    f"{out['pool']} sparks of its vault in "
                    f"reach. {int(war.get('escrow') or 0)} "
                    "sparks escrowed; their leadership answers "
                    "there with /war accept.")
        if head == "join":
            out = surface.war_join(key, user_id)
            if not out.get("ok") and out.get("reason") \
                    in ("no_war", "no_gang"):
                inter = surface.war_join_inter(key, user_id)
                if inter.get("ok"):
                    return ("⚔️ you're in — your strength rides "
                            "with your gang across groups.")
                if inter.get("reason") == "already":
                    return "you're already in."
                out = inter
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_war": "your gang isn't at war.",
                     "no_gang": "you're not in a gang.",
                     "undeclared": "your two gangs are BOTH in "
                                   "this war — declare a side "
                                   "first (/gang side <gang>), "
                                   "or sit it out.",
                     "declared_other": "you declared for the "
                                       "other gang in this "
                                       "war — your strength "
                                       "fights there, not here.",
                     "already": "you're already in."}.get(
                        out.get("reason") or "", "can't join."))
            return ("⚔️ you're in — your power fights with "
                    "your gang's.")
        if head == "spoils" and len(words) >= 2:
            out = surface.war_spoils(key, user_id,
                                     words[1].lower())
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"no_spoils": "no won war is holding spoils "
                                  "for your gang.",
                     "district": "name a district the LOSER "
                                 "holds — it changes hands.",
                     "rank": "spoils are claimed by the leader "
                             "or an officer.",
                     "no_gang": "you're not in a gang."}.get(
                        out.get("reason") or "", "can't claim."))
            from ...community.group_real_estate import DISTRICTS
            return (f"🚩 **{out['gang']['name']}** takes "
                    f"**{DISTRICTS[words[1].lower()]['name']}** "
                    "as war spoils.")
        # bare /war: the status board
        view = surface.war_status(key, user_id)
        if not view.get("ok"):
            return "you're not in a gang — wars are gang business."
        from ...social.chat.typography import Card

        card = Card("economy", title="War", icon="⚔️")
        war = view.get("war")
        inter = view.get("inter")
        war_buttons = None
        if war:
            gangs = surface.conflict._load_state(key).get(
                "gangs") or {}
            a = (gangs.get(str(war.get("a"))) or {}).get(
                "name", "?")
            b = (gangs.get(str(war.get("b"))) or {}).get(
                "name", "?")
            score = war.get("score") or {}
            card.line(f"**{a}** vs **{b}** — "
                      f"{war.get('status')}")
            card.stats([("Rounds", f"{score.get('a', 0)} – "
                                    f"{score.get('b', 0)}"),
                        ("Pot", f"{int(war.get('pot') or 0)} "
                                "sparks")])
            if war.get("status") == "declared":
                card.line("waiting on the defenders' answer.")
                war_buttons = [[("⚔️ Accept war", "war accept"),
                                ("🏳️ Decline", "war decline")]]
            elif war.get("status") == "active":
                card.line("rounds fight on the cycle.")
                war_buttons = [[("⚔️ Join the fight", "war join")]]
        elif inter:
            a = (inter.get("a") or {}).get("gang_name", "?")
            b = (inter.get("b") or {}).get("gang_name", "?")
            score = inter.get("score") or {}
            card.line(f"🌍 **{a}** vs **{b}** — an inter-group "
                      f"war, {inter.get('status')}")
            card.stats([("Rounds", f"{score.get('a', 0)} – "
                                    f"{score.get('b', 0)}")])
        else:
            card.line("your gang is at peace. /war declare "
                      "<gang> starts one — escrow from the "
                      "gang treasury, best of rounds, the "
                      "winner takes the pot (and, at home, a "
                      "district).")
        return self._card(message, card, war_buttons)

    def _control_police(self, tail, message):  # type: ignore[no-untyped-def]
        """``/police`` — the precinct board (§24): who's
        wanted, which bounties stand, and your own heat.
        ``/police bribe`` buys the precinct off — a sink with
        real risk."""
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        gate = surface._conflict_gate(key, "crime")
        if gate:
            return self._conflict_refusal(surface, key, gate)
        words = str(tail or "").split()
        if words and words[0].lower() == "bribe":
            out = surface.police_bribe(
                key, user_id, rng=random.Random(),
                use_envelope=len(words) > 1
                and words[1].lower() == "envelope")
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"not_wanted": "you're not wanted — no "
                                   "precinct to pay off.",
                     "funds": "a bribe costs 40 sparks a star — "
                              "your wallet can't cover it (a "
                              "bribe envelope from /gshop "
                              "works instead)."}.get(
                        out.get("reason") or "", "no bribe."))
            if out.get("success"):
                return ("🤫 the envelope changes hands. Some of "
                        "your heat is forgotten — the journal "
                        "shows less of you now.")
            if out.get("jailed"):
                return ("🚔 the bribe blows up in your face — "
                        "more heat, and 15 minutes in jail for "
                        "trying.")
            return ("🚔 the officer pockets nothing and "
                    "remembers your face — the attempt itself "
                    "is heat now.")
        board = surface.police_board(key, user_id)
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="The precinct", icon="🚔")
        me = board["me"]
        card.stats([("Your heat",
                     f"{'★' * me['stars'] if me['stars'] else 'clean'}"
                     f" (index {me['index']:.0f})")])
        if board["wanted"]:
            lines = []
            for row in board["wanted"]:
                label = self._estate_member_label(
                    surface, key, row["member"])
                lines.append(f"{'★' * row['stars']} **{label}**"
                             f" — index {row['index']:.0f}"
                             + (f" · {'; '.join(row['inputs'])}"
                                if row["inputs"] else ""))
            card.block_lines(lines, title="Wanted")
        else:
            card.line("nobody is wanted right now — the "
                      "precinct watches excess, not play.")
        if board["bounties"]:
            lines = []
            for bounty in board["bounties"]:
                label = self._estate_member_label(
                    surface, key, str(bounty.get("target") or ""))
                lines.append(
                    f"🎯 **{label}** — "
                    f"{sparks(int(bounty.get('amount') or 0))} "
                    f"({'the precinct' if bounty.get('source') == 'police' else 'a member'}"
                    " put it up)")
            card.block_lines(lines, title="Open bounties")
        card.note("the police hunt the bully index — farming "
                  "the weak and the new; ordinary scores are "
                  "just the game. /glog shows the evidence "
                  "behind every action.")
        buttons = buttons_from_spec(
            [[("police", "Bribe (40★/star)", "police bribe"),
              ("bounty", "Bounties", "bounty")]],
            settings=surface.store.settings(key))
        return self._card(message, card, buttons)

    def _control_bounty(self, tail, message):  # type: ignore[no-untyped-def]
        """``/bounty`` — the board; ``/bounty post @member
        200`` puts your own sparks on a head; ``/bounty hunt
        @member`` is the citizen action (§24)."""
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        words = str(tail or "").split()
        head = words[0].lower() if words else ""
        if head == "post":
            amount = None
            token = ""
            for part in words[1:]:
                if part.isdigit() and amount is None:
                    amount = int(part)
                elif not token:
                    token = part
            if amount is None or not token:
                return "usage: /bounty post @member 200"
            target, label = self._conflict_target(
                surface, key, token, message)
            if not target:
                return "who? name a member of this group."
            out = surface.bounty_post(key, user_id, target,
                                      amount)
            if not out.get("ok"):
                return self._conflict_refusal(
                    surface, key, out,
                    {"funds": "your wallet can't cover that "
                              "bounty.",
                     "amount": "bounties start at 10 sparks.",
                     "self": "not on your own head."}.get(
                        out.get("reason") or "", "can't post."))
            from ...social.chat.typography import sparks
            return (f"🎯 **{sparks(amount)}** on **{label}**'s "
                    "head — whoever brings them down (a geared "
                    "hit, or /bounty hunt) collects.")
        if head == "hunt":
            target, label = self._conflict_target(
                surface, key, " ".join(words[1:]), message)
            if not target:
                return "usage: /bounty hunt @member"
            out = surface.bounty_hunt(key, user_id, target,
                                      rng=random.Random())
            if not out.get("ok"):
                if out.get("reason") == "not_wanted":
                    return (f"{label} isn't wanted and carries "
                            "no bounty — the hunt needs a mark.")
                return self._conflict_refusal(surface, key, out)
            from ...social.chat.typography import Card, sparks

            odds = int(round(float(out.get("odds") or 0) * 100))
            if out.get("success"):
                card = Card("economy", title="Brought in",
                            icon="🎯")
                card.line(f"**{label}** is brought in — 30 "
                          "minutes in jail.")
                if out.get("bounty"):
                    card.line(f"🎯 bounty paid: "
                              f"**{sparks(int(out['bounty']))}**.")
                card.stats([("The odds were", f"{odds}%")])
                return self._card(message, card)
            if out.get("jailed"):
                return ("🚔 the hunt goes wrong — no fine money "
                        "in your wallet, so it's 10 minutes in "
                        "jail instead.")
            return (f"**{label}** slips the hunt — a "
                    f"{int(out.get('fine') or 25)}-spark fine "
                    "for the trouble.")
        return self._control_police("", message)

    def _control_claimdrop(self, tail, message):  # type: ignore[no-untyped-def]
        """``/claimdrop [ref]`` — claim a §35 system drop.
        The button carries the drop's reference; bare text
        claims the live drop when exactly one is open.
        Claims are settlements: final, journaled, and the
        drop edits closed when its slots fill."""
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        ref = str(tail or "").strip().split()[0] \
            if str(tail or "").strip() else ""
        if not ref:
            state = surface.conflict._load_state(key)
            open_drops = [d for d in (state.get("drops")
                                      or {}).values()
                          if d.get("status") == "open"]
            if len(open_drops) != 1:
                return ("no single live drop to claim — when "
                        "one is announced, tap its Claim "
                        "button.")
            ref = str(open_drops[0].get("id") or "")
        out = surface.sysdrop_claim(key, user_id, ref)
        if not out.get("ok"):
            return {"gone": "that drop is gone — claimed out "
                            "or expired.",
                    "expired": "too late — that drop expired.",
                    "already": "you already claimed from this "
                               "drop.",
                    "full": "the last slot just went — the "
                            "drop is full.",
                    "cap": "your daily earn cap is full — the "
                           "slot stays open for someone else.",
                    "mode": "the conflict layer is off here.",
                    "frozen": "the conflict state is frozen "
                              "— see /glog."}.get(
                        out.get("reason") or "",
                        "can't claim that drop.")
        got = out.get("got") or {}
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Drop claimed", icon="🎁")
        if got.get("kind") in ("sparks", "sparks_big"):
            card.line(f"**+{sparks(int(got.get('amount') or 0))}**"
                      " — yours, ledgered.")
        elif got.get("kind") == "gems":
            card.line(f"**+{int(got.get('amount') or 0)} 💎** — "
                      "straight into your premium purse.")
        elif got.get("kind") in ("gear", "gear_rare"):
            from ...community.group_conflict import ARSENAL
            spec = ARSENAL.get(str(got.get("item") or "")) or {}
            card.line(f"**{spec.get('icon', '🎁')} "
                      f"{spec.get('name', 'gear')}** — into "
                      "your loadout (/gear).")
        if out.get("closed"):
            card.note("that was the last slot — the drop is "
                      "closed")
            self._close_sysdrop_message(surface, key, message,
                                        ref)
        return self._card(message, card,
                          [[("🪙 My balance", "balance")]])

    def _close_sysdrop_message(self, surface, key, message,  # type: ignore[no-untyped-def]
                               drop_id: str) -> None:
        """A filled drop edits its announcement closed (§35),
        where the platform supports editing — the state (the
        organ's record) already closed; this is presentation
        catching up with it."""
        msgs = (surface._extras(key).get("sysdrop_msgs") or {})
        message_id = str(msgs.get(drop_id) or "")
        if not message_id:
            return
        adapter = self._group_adapter(message.chat.platform)
        edit = getattr(adapter, "edit_message", None) \
            if adapter else None
        if edit is None:
            return
        try:
            edit(message.chat, message_id,
                 "🎁 **the drop is claimed in full** — every "
                 "slot went. Watch the sky; the budget rolls "
                 "again tomorrow.", buttons=[])
        except Exception:  # noqa: BLE001 - presentation only
            _log.debug("sysdrop close edit failed",
                       exc_info=True)

    def _bet_arg(self, tail, default: int = 10) -> int:  # type: ignore[no-untyped-def]
        for part in str(tail or "").split():
            if part.isdigit():
                return int(part)
        return default

    def _control_slots(self, tail, message):  # type: ignore[no-untyped-def]
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        bet = self._bet_arg(tail)
        result = surface.slots(key, user_id, bet,
                               rng=random.Random())
        if not result["ok"]:
            if result.get("reason") == "jailed":
                return "🔒 no slots from jail — /bail first."
            return "not enough sparks in your wallet for that bet."
        reels = " ".join(result["symbols"])
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card, sparks

        buttons = buttons_from_spec(
            [[("spin", f"Spin again ({bet})", f"slots {bet}"),
              ("balance", "My balance", "balance")]],
            settings=surface.store.settings(key))
        if result.get("jackpot"):
            card = Card("game", title="Slots", icon="🎰")
            card.board(reels)
            card.line(f"🎉 **JACKPOT — {sparks(result['jackpot'])}!** "
                      f"plus {sparks(result['payout'])} from the "
                      f"reels.")
            card.stats([("Balance", sparks(result["balance"]),
                         True)])
            # The jackpot moment (§21): the room celebrates in
            # media too, where expression is on and claimed.
            fire = getattr(self, "_fire_expression", None)
            if fire is not None:
                fire(message, "jackpot")
            return self._card(message, card, buttons)
        if result["payout"] > 0:
            card = Card("game", title="Slots", icon="🎰")
            card.board(reels)
            card.line(f"**+{sparks(result['payout'])}** "
                      f"(×{result['multiplier']})")
            card.stats([("Balance", sparks(result["balance"]),
                         True)])
            return self._card(message, card, buttons)
        card = Card("game", title="Slots", icon="🎰")
        card.board(reels)
        card.line("no win this spin.")
        card.stats([("Jackpot pot",
                     sparks(surface.economy.jackpot_balance(key)),
                     True)])
        card.note("the house share feeds the treasury and the "
                  "jackpot pot")
        return self._card(message, card, buttons)

    def _control_coinflip(self, tail, message):  # type: ignore[no-untyped-def]
        import random

        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        bet = self._bet_arg(tail)
        choice = "tails" if "tail" in str(tail or "").lower() \
            else "heads"
        result = surface.coinflip(key, user_id, bet, choice,
                                  rng=random.Random())
        if not result["ok"]:
            if result.get("reason") == "jailed":
                return "🔒 no coinflips from jail — /bail first."
            return "not enough sparks in your wallet for that bet."
        from ...social.chat.expression import buttons_from_spec
        from ...social.chat.typography import Card, sparks

        buttons = buttons_from_spec(
            [[("flip", f"Flip again ({bet})", f"coinflip {bet}"),
              ("balance", "My balance", "balance")]],
            settings=surface.store.settings(key))
        card = Card("game", title="Coinflip", icon="🪙")
        card.board(f"🪙 {result['flip']}")
        if result["won"]:
            card.line(f"you called it — "
                      f"**+{sparks(result['payout'])}**")
            card.stats([("Balance", sparks(result["balance"]),
                         True)])
        else:
            card.line("not your call this time.")
            card.note("the house share feeds the treasury and "
                      "the jackpot")
        return self._card(message, card, buttons)

    def _control_top(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...social.chat.typography import Card, sparks

        surface = self._group_surface()
        rows = surface.top_holders(key, limit=10)
        if not rows:
            return "no sparks held yet — /work and /mine start it."
        members = surface.members(key)
        card = Card("leaderboard", title="richest in the group",
                    icon="🏆")
        medals = ["🥇", "🥈", "🥉"]
        stats = []
        for i, row in enumerate(rows):
            record = members.get(row["member"]) or {}
            who = str(record.get("name") or record.get("username")
                      or row["member"])
            mark = medals[i] if i < 3 else f"{i + 1}."
            value = sparks(row.get("worth", row["total"]))
            if row["bank"]:
                value += f" (🏦 {sparks(row['bank'])} banked)"
            if row.get("property"):
                value += (f" (🏘️ {sparks(row['property'])} "
                          f"property)")
            stats.append((f"{mark} {who}", value, True))
        card.stats(stats)
        return self._card(message, card)

    def _control_tax(self, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        from ...social.chat.typography import Card, sparks

        surface = self._group_surface()
        s = surface.store.settings(key)
        policy = surface.economy.issuance_state(key)
        state = ("throttled (activity pays half)"
                 if policy["throttled"] else "normal")
        card = Card("economy", title="The group's monetary "
                    "policy", icon="🧾")
        card.stats([
            ("Transfer tax",
             f"{s.get('transfer_tax_pct', 2)}% to the treasury"),
            ("Moderation fine", sparks(s.get("fine_amount", 0)),
             True),
            ("Earn caps", "200/day · 800/week"),
            ("Governance",
             f"treasury spends above "
             f"{sparks(s.get('governance_threshold', 500))} need "
             f"a second admin"),
            ("Issuance this week",
             f"{sparks(policy['earned'])} issued / "
             f"{sparks(policy['sunk'])} sunk"),
        ])
        card.note(f"faucets {state}")
        return self._card(message, card)

    def _control_grant(self, tail, message):  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        sender = str(message.sender_id or message.sender or "")
        # Governance capture (§23): the gang IN POWER drives
        # the same governance organ — proposals, thresholds
        # and second-approver all unchanged; capture changes
        # WHO may drive, never the mechanism.
        if not self._group_admin_ok(message) \
                and not surface.gang_power_actor(key, sender):
            return "treasury grants are for admins."
        key, err = self._economy_gate(message)
        if err:
            return err
        raw = str(tail or "").strip()
        if raw.lower().startswith("gems "):
            return self._control_grant_gems(raw[5:].strip(),
                                            message, key, surface)
        parts = str(tail or "").split()
        amount = None
        target_token = ""
        for part in parts:
            if part.isdigit() and amount is None:
                amount = int(part)
            elif not target_token:
                target_token = part
        if amount is None or not target_token:
            return "usage: /grant @member 100"
        recipient, miss = self._named_ref(surface, key,
                                          target_token, message)
        if not recipient:
            return miss
        admin = str(message.sender_id or message.sender or "")
        threshold = int(surface.store.settings(
            key).get("governance_threshold") or 0)
        # Governance binds ADMINS spending the country's money;
        # the operator is sovereign (Septorch's owner posture)
        # and never deadlocks against his own threshold.
        if threshold and amount > threshold \
                and not self._operator_or_false(message):
            proposal = surface.propose_spend(
                key, kind="grant",
                payload={"recipient": recipient, "amount": amount,
                         "reason": "admin grant"}, by=admin)
            from ...social.chat.tgbot_buttons import actionable
            return actionable(
                f"🗳️ that grant ({amount} sparks) is above the "
                f"governance threshold ({threshold}) — proposal "
                f"<code>{proposal['id']}</code> is open. A SECOND "
                "admin approves it:",
                [[("✅ Approve grant", f"approve {proposal['id']}")]])
        result = surface.grant(key, recipient, amount, by=admin)
        if not result["ok"]:
            return (f"the treasury can't cover that — "
                    f"{result.get('available', 0)} available after "
                    "open-drop reservations.")
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Grant paid", icon="🏦")
        card.line(f"**{sparks(amount)}** granted from the "
                  f"treasury.")
        card.stats([("Treasury now", sparks(result["treasury"]),
                     True)])
        return self._card(message, card)

    def _control_grant_gems(self, tail, message, key, surface):  # type: ignore[no-untyped-def]
        """/grant gems @member 25: a gems gift — the premium
        purse's governance road (§17/§25).  Gems are never
        minted on one admin's word above the threshold: the
        gift becomes a proposal a second admin executes; the
        operator, sovereign as ever, may gift directly."""
        amount = None
        target_token = ""
        for part in str(tail or "").split():
            if part.isdigit() and amount is None:
                amount = int(part)
            elif not target_token:
                target_token = part
        if amount is None or not target_token:
            return "usage: /grant gems @member 25"
        recipient, miss = self._named_ref(surface, key,
                                          target_token, message)
        if not recipient:
            return miss
        admin = str(message.sender_id or message.sender or "")
        threshold = int(surface.store.settings(
            key).get("governance_threshold") or 0)
        if threshold and amount > threshold \
                and not self._operator_or_false(message):
            proposal = surface.propose_spend(
                key, kind="gems_gift",
                payload={"recipient": recipient,
                         "amount": amount,
                         "reason": "admin gems gift"}, by=admin)
            from ...social.chat.tgbot_buttons import actionable
            return actionable(
                f"🗳️ that gems gift ({amount} 💎) is above the "
                f"governance threshold ({threshold}) — "
                f"proposal <code>{proposal['id']}</code> is "
                f"open. A SECOND admin approves it:",
                [[("✅ Approve gift",
                   f"approve {proposal['id']}")]])
        result = surface.estate_grant_gems(
            key, recipient, amount,
            proposal_id="operator-direct", by=admin)
        if not result["ok"]:
            return "that gift can't be issued right now."
        from ...social.chat.typography import Card

        card = Card("economy", title="Gems gifted", icon="💎")
        card.line(f"**{amount} 💎** gifted from the premium "
                  f"purse — through governance, on the record.")
        card.stats([("Their gems now",
                     f"{result['gems']} 💎", True)])
        return self._card(message, card)

    def _control_approve(self, tail, message):  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        sender = str(message.sender_id or message.sender or "")
        if not self._group_admin_ok(message) \
                and not surface.gang_power_actor(key, sender):
            return "approvals are for admins."
        key, err = self._economy_gate(message)
        if err:
            return err
        pid = str(tail or "").strip().split()[0] \
            if str(tail or "").strip() else ""
        if not pid:
            open_now = surface.open_proposals(key)
            if not open_now:
                return "no treasury proposals are open."
            from ...social.chat.typography import Card

            card = Card("economy", title="Open treasury "
                        "proposals", icon="🗳️")
            card.block_lines([
                f"`{open_pid}` — {proposal.get('kind')}: "
                f"{proposal.get('payload') or {}}"
                for open_pid, proposal in open_now.items()])
            card.note("approve with /approve <id> (a different "
                      "admin than the proposer)")
            return self._card(message, card)
        admin = str(message.sender_id or message.sender or "")
        proposal = surface.open_proposals(key).get(pid) or {}
        result = surface.approve_spend(key, pid, by=admin)
        if not result["ok"]:
            return {"no_proposal": "no open proposal with that id.",
                    "self_approve": "the proposer can't approve "
                                    "their own spend — a second "
                                    "admin has to.",
                    "funds": "the treasury can't cover it any "
                             "more."}.get(result["reason"],
                                          "can't approve that.")
        # A governance execution is a §31 notice consumer: the
        # group sees what its treasury just did, on the record.
        kind = str(proposal.get("kind") or "")
        payload = proposal.get("payload") or {}
        if kind == "drop" and result.get("drop"):
            self._announce_governance_drop(surface, key, message,
                                           result["drop"])
        elif kind == "grant":
            recipient = str(payload.get("recipient") or "")
            label = self._estate_member_label(surface, key,
                                              recipient)
            text = (f"🏦 {label} [{recipient}] received "
                    f"**{int(payload.get('amount') or 0)} sparks** "
                    f"from the treasury — proposal approved by "
                    f"{message.sender or 'an admin'}.")
            surface.record_notice(
                key, user_id=recipient, mention=label,
                rule="governance", action="grant", text=text,
                extra={"reversible": False, "proposal": pid})
            if self._system_send_ok(surface, key, message,
                                    "governance"):
                self._send_best_effort(message.chat, text)
        return (f"✅ proposal <code>{pid}</code> approved and "
                "executed.")

    def _announce_governance_drop(self, surface, key, message,  # type: ignore[no-untyped-def]
                                  drop) -> None:
        """A drop a governance vote opened gets the same public
        announcement (and the same undoable notice) as an
        admin-opened one — before this slice it opened silently
        and members could not claim what they never saw."""
        from ...social.chat.typography import Card, sparks

        claims = int(drop["total"]) // int(drop["per_claim"])
        card = Card("announcement", title="Drop", icon="🎁")
        card.line(f"**{sparks(drop['total'])}** from the "
                  f"treasury is up for grabs.")
        card.stats([("Per claim", sparks(drop["per_claim"]), True),
                    ("Claims", str(claims))])
        card.note("tap Claim or send /claim")
        nid = self._record_drop_notice(surface, key, message,
                                       drop, treasury=True)
        rendered = self._card(
            message, card,
            [[(f"🎁 Claim {drop['per_claim']} sparks", "claim"),
              ("🪙 My balance", "balance")],
             [("↩️ Cancel drop", f"undonotice {nid}")]])
        if self._system_send_ok(surface, key, message, "drop"):
            self._send_best_effort(
                message.chat, rendered.text,
                buttons=rendered.buttons,
                parse_mode=self._rendered_mode(message,
                                               rendered.text))

    # ── real estate (§17 slice 2 / §20: the whole class) ──────
    def _estate_gate(self, message):  # type: ignore[no-untyped-def]
        """Economy gate + the estate freeze: (key, None) or
        (None, text)."""
        key, err = self._economy_gate(message)
        if err:
            return None, err
        if self._group_surface().estate_frozen(key):
            return None, ("the property market is frozen — its "
                          "records need an admin's eyes before "
                          "anything trades again.")
        return key, None

    def _estate_member_label(self, surface, key, user_id):  # type: ignore[no-untyped-def]
        record = surface.members(key).get(str(user_id)) or {}
        return str(record.get("name") or record.get("username")
                   or user_id)

    def _control_property(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        if str(tail or "").strip().lower() == "manager":
            return self._estate_manager(message, key, surface,
                                        user_id)
        from ...community.group_real_estate import (
            MANAGER_LEVELS, TIERS)
        from ...social.chat.typography import (
            Card, fmt_amount, sparks)

        summary = surface.estate_summary(key, user_id)
        card = Card("economy", title="Property", icon="🏘️")
        card.stats([
            ("Your gems", f"{summary['gems']} 💎", True),
            ("Property value",
             sparks(summary["property_value"]), True),
            ("Rent standing",
             f"{summary['rent_ready_gems']} 💎", True),
        ])
        wallet = surface.economy.balance(key, user_id)
        gems = summary["gems"]
        buy_buttons: list = []
        for row in surface.estate_catalog(key):
            prices = []
            for tier_id, spec in TIERS.items():
                price = row["tiers"][tier_id]
                if spec["currency"] == "gems":
                    prices.append(f"{spec['icon']} {price}💎")
                else:
                    prices.append(
                        f"{spec['icon']} {fmt_amount(price)}")
            trend = row["trend_pct"]
            arrow = "▲" if trend > 0 else (
                "▼" if trend < 0 else "▬")
            holder = surface.territory_map(key).get(
                row["district"])
            card.block_lines([
                f"**{row['name']}** · index {row['value']} "
                f"({arrow} {abs(trend)}% this week)"
                + (f" · 🚩 held by **{holder}**" if holder
                   else ""),
                " · ".join(prices),
            ])
            afford = [(t, row["tiers"][t]) for t in TIERS
                      if (TIERS[t]["currency"] == "sparks"
                          and row["tiers"][t] <= wallet)
                      or (TIERS[t]["currency"] == "gems"
                          and row["tiers"][t] <= gems)]
            if afford:
                tier_id, price = afford[0]
                cur = "💎" if TIERS[tier_id]["currency"] == "gems" \
                    else " sparks"
                buy_buttons.append(
                    (f"Buy {TIERS[tier_id]['name']} · "
                     f"{row['name']} ({fmt_amount(price)}{cur})",
                     f"buyproperty {row['district']} {tier_id}"))
        if summary["properties"]:
            lines = []
            for prop in summary["properties"]:
                spec = TIERS[prop["tier"]]
                cur = "💎" if prop["currency"] == "gems" else ""
                flags = []
                if prop.get("listed"):
                    flags.append("on the market")
                if prop.get("auction"):
                    flags.append("at auction")
                if prop.get("mortgage"):
                    flags.append("mortgaged")
                if float(prop.get("insured_until") or 0) > 0:
                    flags.append("insured")
                if int(prop.get("repair_debt") or 0):
                    flags.append(
                        f"repair debt {prop['repair_debt']}")
                lines.append(
                    f"`{prop['id']}` {spec['icon']} "
                    f"**{spec['name']}** — worth "
                    f"{fmt_amount(prop['value'])}{cur} · rent "
                    f"{prop['rent_gems']}💎 standing"
                    + (f" · {' · '.join(flags)}" if flags else ""))
            card.block_lines(lines, title="Yours")
        level = summary["manager_level"]
        mgr = MANAGER_LEVELS.get(level)
        card.note(
            f"manager: {mgr['name'] if mgr else 'none'} — "
            f"/property manager hires the next rung "
            f"(auto-collect, more rent, longer accrual)")
        buttons = []
        if summary["rent_ready_gems"] > 0:
            buttons.append([("💎 Collect rent", "rent")])
        for i in range(0, len(buy_buttons), 2):
            buttons.append(buy_buttons[i:i + 2])
        buttons.append([("🏪 Market", "listproperty"),
                        ("🔨 Auctions", "auctions")])
        return self._card(message, card, buttons)

    def _estate_manager(self, message, key, surface, user_id):  # type: ignore[no-untyped-def]
        result = surface.estate_manager_buy(key, user_id)
        if not result["ok"]:
            if result["reason"] == "top":
                return ("🧑‍💼 you already run an Estate Agency — "
                        "the top rung. Your whole portfolio is "
                        "managed.")
            return (f"🧑‍💼 the next rung costs "
                    f"{result.get('cost', 0)} sparks — you have "
                    f"{surface.economy.balance(key, user_id)}.")
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Management hired",
                    icon="🧑‍💼")
        card.line(f"**{result['name']}** now runs your "
                  f"properties.")
        card.stats([
            ("Rent", f"+{result['yield_pct']}%"),
            ("Accrual cap", f"{result['cap_days']} days"),
            ("Auto-collect", "yes — even while you're jailed "
                             "or away" if result["auto"] else "no"),
        ])
        card.note(f"paid {sparks(result['cost'])} to the "
                  f"treasury")
        return self._card(message, card,
                          [[("🏘️ My property", "property")]])

    def _control_buyproperty(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if parts and parts[0].lower() == "public":
            return self._estate_buy_public(message, key, surface,
                                           parts[1:], user_id)
        if len(parts) < 2:
            return ("usage: /buyproperty <district> <tier> — "
                    "/property shows the districts and tiers.")
        district, tier = parts[0].lower(), parts[1].lower()
        result = surface.estate_buy(key, user_id, district, tier)
        if not result["ok"]:
            return {
                "catalog": "no such district or tier — "
                           "/property shows the map.",
                "cap": f"you hold the group's property cap "
                       f"({result.get('cap')}) — sell or trade "
                       f"before buying more.",
                "funds": f"not enough sparks — that costs "
                         f"{result.get('price')} and you have "
                         f"{surface.economy.balance(key, user_id)}.",
                "gems": f"that tier is priced in gems 💎 — it "
                        f"costs {result.get('price')}💎 and you "
                        f"have "
                        f"{surface.estate.gems_balance(key, user_id)}💎. "
                        f"Gems come from property rent, season "
                        f"rewards and governance gifts — never "
                        f"from sparks.",
            }.get(result["reason"], "can't buy that.")
        from ...community.group_real_estate import DISTRICTS, TIERS
        from ...social.chat.typography import Card, sparks

        prop = result["property"]
        spec = TIERS[prop["tier"]]
        card = Card("economy", title="Property bought",
                    icon="🏘️")
        card.line(f"**{spec['name']}** in "
                  f"{DISTRICTS[prop['district']]['name']} is "
                  f"yours.")
        price_text = (f"{result['price']} 💎"
                      if prop["currency"] == "gems"
                      else sparks(result["price"]))
        card.stats([
            ("Paid", price_text, True),
            ("Rent", f"{spec['rent']} 💎 / day", True),
            ("District index",
             str(result["index"]["value"])),
            ("Property id", prop["id"]),
        ])
        card.note("rent accrues daily — /rent collects it as "
                  "gems")
        return self._card(
            message, card,
            [[("🏘️ My property", "property"),
              ("💎 Rent", "rent")]])

    def _estate_buy_public(self, message, key, surface, parts,  # type: ignore[no-untyped-def]
                           user_id):
        if not self._group_admin_ok(message):
            return "public property is an admin call."
        if len(parts) < 2:
            return ("usage: /buyproperty public <district> "
                    "<tier>")
        district, tier = parts[0].lower(), parts[1].lower()
        from ...community.group_real_estate import DISTRICTS, TIERS

        if district not in DISTRICTS or tier not in TIERS:
            return "no such district or tier — /property shows it."
        if TIERS[tier]["currency"] == "gems":
            return ("public property is bought with treasury "
                    "sparks — gem tiers are members' premium "
                    "shelf only.")
        price = surface.estate.price(key, district, tier)
        threshold = int(surface.store.settings(
            key).get("governance_threshold") or 0)
        if threshold and price > threshold \
                and not self._operator_or_false(message):
            proposal = surface.propose_spend(
                key, kind="estate_public",
                payload={"district": district, "tier": tier},
                by=user_id)
            from ...social.chat.tgbot_buttons import actionable
            return actionable(
                f"🗳️ a public {TIERS[tier]['name']} "
                f"({price} sparks of treasury money) is above "
                f"the governance threshold ({threshold}) — "
                f"proposal <code>{proposal['id']}</code> is "
                f"open. A SECOND admin approves it:",
                [[("✅ Approve purchase",
                   f"approve {proposal['id']}")]])
        result = surface.estate_buy_public(key, district, tier,
                                           by=user_id)
        if not result["ok"]:
            return (f"the treasury can't cover that — "
                    f"{result.get('available', 0)} available "
                    f"after reservations.")
        from ...community.group_real_estate import DISTRICTS
        from ...social.chat.typography import Card, sparks

        card = Card("economy", title="Public property bought",
                    icon="🏛️")
        card.line(f"**{TIERS[tier]['name']}** in "
                  f"{DISTRICTS[district]['name']} now belongs "
                  f"to the group.")
        card.stats([("Paid from treasury", sparks(price), True),
                    ("Treasury now",
                     sparks(surface.economy.treasury_balance(key)),
                     True)])
        card.note("its rent lands in the treasury daily — "
                  "funding drops and prizes")
        return self._card(message, card)

    def _control_sellproperty(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if not parts:
            return ("usage: /sellproperty <property id> — "
                    "/property lists yours with their ids.")
        pid = parts[0]
        result = surface.estate_sell(key, user_id, pid)
        if not result["ok"]:
            return {
                "not_found": "no property of yours with that "
                             "id — /property lists yours.",
                "listed": "that property is on the market — "
                          "/listproperty cancel <id> first.",
                "auction": "that property is at auction — the "
                           "auction owns it until it settles.",
                "mortgaged": "that property carries a mortgage "
                             "— repay it before selling.",
                "funds": "the treasury can't cover the "
                         "buy-back right now.",
            }.get(result["reason"], "can't sell that.")
        from ...social.chat.typography import Card, sparks

        proceeds = (f"{result['proceeds']} 💎"
                    if result["currency"] == "gems"
                    else sparks(result["proceeds"]))
        card = Card("economy", title="Property sold back",
                    icon="🏘️")
        card.line(f"the system bought it back at the "
                  f"depreciated rate.")
        card.stats([("Market value",
                     f"{result['value']}"
                     + (" 💎" if result["currency"] == "gems"
                        else " sparks"), True),
                    ("You received", proceeds, True)])
        return self._card(message, card,
                          [[("🏘️ Property", "property")]])

    def _control_rent(self, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        result = surface.estate_collect_rent(key, user_id)
        if not result["ok"]:
            if result["reason"] == "jailed":
                from ...social.chat.tgbot_buttons import actionable

                return actionable(
                    "🔒 no rent collection from jail — bail "
                    "out, or hire a manager and it collects "
                    "for you.",
                    [[("🔓 Bail", "bail")],
                     [("🏠 Hire a manager", "property manager")]])
            return "the property market is off here."
        import time as _time

        from ...community.group_real_estate import TIERS
        from ...social.chat.typography import Card

        if not result["lines"]:
            summary = surface.estate_summary(key, user_id)
            if not summary["properties"]:
                return ("you don't own property yet — /property "
                        "shows the map, /buyproperty starts you.")
            soonest = min(
                float(p.get("anchor") or 0) + 86400
                for p in summary["properties"])
            wait_h = max(1, int((soonest - _time.time()) // 3600))
            return (f"no rent standing yet — the next full day "
                    f"lands in ~{wait_h}h. Rent pays gems 💎.")
        card = Card("economy", title="Rent collected", icon="💎")
        card.line(f"**+{result['gems']} 💎**")
        lines = []
        for line in result["lines"]:
            spec = TIERS[line["tier"]]
            text = (f"{spec['icon']} **{spec['name']}** — "
                    f"{line['periods']} day(s) · "
                    f"+{line['gems']} 💎")
            if line["garnished"]:
                text += (f" ({line['garnished']} 💎 went to "
                         f"its repair debt)")
            lines.append(text)
        card.block_lines(lines)
        if result["offset"]:
            card.note(f"{result['offset']} 💎 of rent offset "
                      f"repair debt first")
        card.stats([("Gems now", f"{result['balance']} 💎", True)])
        return self._card(
            message, card,
            [[("🏘️ Property", "property"),
              ("🪙 My balance", "balance")]])

    def _control_listproperty(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if parts and parts[0].lower() == "buy" and len(parts) >= 2:
            result = surface.estate_buy_listing(key, user_id,
                                                parts[1])
            if not result["ok"]:
                return {
                    "not_found": "that listing is gone — "
                                 "/listproperty shows the live "
                                 "market.",
                    "own": "that's your own listing.",
                    "cap": "you hold the group's property cap "
                           "— sell before buying more.",
                    "funds": f"not enough sparks — the ask is "
                             f"{result.get('ask')} and you have "
                             f"{surface.economy.balance(key, user_id)}.",
                }.get(result["reason"], "can't buy that.")
            from ...social.chat.typography import Card, sparks

            card = Card("economy", title="Market purchase",
                        icon="🏪")
            card.line("the property is yours — its rent clock "
                      "kept running through the sale.")
            card.stats([("Ask paid", sparks(result["ask"]), True),
                        ("Seller received",
                         sparks(result["net"]), True),
                        ("Treasury cut", sparks(result["cut"]),
                         True)])
            return self._card(message, card,
                              [[("🏘️ My property", "property")]])
        if parts and parts[0].lower() == "cancel" \
                and len(parts) >= 2:
            result = surface.estate_unlist(key, user_id, parts[1])
            if not result["ok"]:
                return ("no listed property of yours with that "
                        "id.")
            return "🏪 listing withdrawn — the property is back in your hands."
        if len(parts) >= 2 and parts[1].isdigit():
            result = surface.estate_list(key, user_id, parts[0],
                                         int(parts[1]))
            if not result["ok"]:
                return {
                    "not_found": "no property of yours with "
                                 "that id — /property lists "
                                 "yours.",
                    "currency": "gem-priced tiers trade for "
                                "gems at auction only — the "
                                "sparks market can't list them.",
                    "listed": "that property is already on "
                              "the market.",
                    "auction": "that property is at auction.",
                    "mortgaged": "a mortgaged property can't "
                                 "be listed.",
                    "shape": "the ask must be a positive "
                             "number of sparks.",
                }.get(result["reason"], "can't list that.")
            return (f"🏪 listed for {parts[1]} sparks as "
                    f"`{result['listing']}` — /listproperty "
                    f"shows the board; the treasury takes the "
                    f"group's market cut on a sale.")
        # bare: the market board
        from ...community.group_real_estate import DISTRICTS, TIERS
        from ...social.chat.typography import Card, fmt_amount

        listings = surface.estate_market(key)
        card = Card("economy", title="Property market", icon="🏪")
        if not listings:
            card.line("nothing listed right now — list yours: "
                      "/listproperty <property id> <ask>")
        buttons = []
        lines = []
        for listing in listings:
            spec = TIERS[listing["tier"]]
            seller = self._estate_member_label(
                surface, key, listing["seller"])
            mine = listing["seller"] == user_id
            lines.append(
                f"`{listing['id']}` {spec['icon']} "
                f"**{spec['name']}** · "
                f"{DISTRICTS[listing['district']]['name']} — "
                f"**{fmt_amount(listing['ask'])} sparks** "
                f"(value ≈ {fmt_amount(listing['value'])}) · "
                f"{'yours' if mine else seller}")
            if not mine and surface.economy.balance(
                    key, user_id) >= int(listing["ask"]):
                buttons.append(
                    [(f"Buy {spec['name']} · "
                      f"{fmt_amount(listing['ask'])}",
                      f"listproperty buy {listing['id']}")])
        card.block_lines(lines)
        cut = surface.estate_cfg(key)["market_cut_pct"]
        card.note(f"the treasury takes {cut}% of every sale · "
                  f"rent clocks follow the property")
        buttons.append([("🏘️ Property", "property")])
        return self._card(message, card, buttons)

    def _control_auctions(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if parts and parts[0].lower() == "start":
            if len(parts) < 3 or not parts[2].isdigit():
                return ("usage: /auctions start <property id> "
                        "<start price> [hours]")
            hours = 24.0
            if len(parts) >= 4:
                try:
                    hours = max(0.05, float(parts[3]))
                except ValueError:
                    return "hours must be a number."
            result = surface.estate_auction_start(
                key, user_id, parts[1], int(parts[2]), hours)
            if not result["ok"]:
                return {
                    "not_found": "no property of yours with "
                                 "that id.",
                    "currency": "gem-priced tiers can't be "
                                "auctioned for sparks.",
                    "listed": "that property is on the market "
                              "— unlist it first.",
                    "auction": "that property is already at "
                               "auction.",
                    "mortgaged": "a mortgaged property can't "
                                 "be auctioned.",
                    "shape": "the start price must be "
                             "positive.",
                }.get(result["reason"], "can't start that.")
            return (f"🔨 auction `{result['auction']}` is open "
                    f"— /auctions shows it; bids are held in "
                    f"the treasury until it settles.")
        if parts and parts[0].lower() == "bid":
            if len(parts) < 3 or not parts[2].isdigit():
                return "usage: /auctions bid <auction id> <amount>"
            result = surface.estate_bid(key, user_id, parts[1],
                                        int(parts[2]))
            if not result["ok"]:
                return {
                    "not_found": "that auction is closed or "
                                 "gone — /auctions shows the "
                                 "live ones.",
                    "own": "you can't bid on your own auction.",
                    "leading": "you're already the high bidder "
                               "— your sparks are held; raise "
                               "only by being outbid first.",
                    "low": f"too low — the next bid is at "
                           f"least {result.get('floor')}.",
                    "funds": "not enough sparks in your wallet "
                             "for that bid.",
                }.get(result["reason"], "can't bid that.")
            from ...social.chat.typography import Card, sparks

            card = Card("economy", title="Bid placed", icon="🔨")
            card.line(f"you lead at **{sparks(result['high'])}** "
                      f"— held in the treasury until the "
                      f"auction settles.")
            if result["refunded"]:
                card.line("the outbid member was refunded in "
                          "full.")
            if result["extended"]:
                card.note("anti-snipe: the close moved out — "
                          "a bid inside the last 2 minutes "
                          "extends it")
            return self._card(message, card,
                              [[("🔨 Auctions", "auctions")]])
        # bare: the auction board
        import time as _time

        from ...community.group_real_estate import DISTRICTS, TIERS
        from ...social.chat.typography import Card, fmt_amount

        auctions = surface.estate_auctions(key)
        card = Card("economy", title="Auctions", icon="🔨")
        if not auctions:
            card.line("no auctions live — start one: /auctions "
                      "start <property id> <start price> [hours]")
        buttons = []
        lines = []
        for auction in auctions:
            spec = TIERS[auction["tier"]]
            left_s = max(0, int(float(auction["ends"])
                                - _time.time()))
            left = (f"{left_s // 3600}h"
                    if left_s >= 3600 else f"{left_s // 60}m")
            high = int(auction["high"] or 0)
            leader = ""
            if auction["high_bidder"]:
                leader = (" · you lead"
                          if auction["high_bidder"] == user_id
                          else " · led by " +
                          self._estate_member_label(
                              surface, key,
                              auction["high_bidder"]))
            lines.append(
                f"`{auction['id']}` {spec['icon']} "
                f"**{spec['name']}** · "
                f"{DISTRICTS[auction['district']]['name']} — "
                f"high **{fmt_amount(high or auction['start'])} "
                f"sparks** · {left} left{leader}")
            if auction["seller"] != user_id \
                    and auction["high_bidder"] != user_id:
                nxt = high + max(1, high // 10) if high \
                    else int(auction["start"])
                if surface.economy.balance(key, user_id) >= nxt:
                    buttons.append(
                        [(f"Bid {fmt_amount(nxt)} on "
                          f"{spec['name']}",
                          f"auctions bid {auction['id']} {nxt}")])
        card.block_lines(lines)
        card.note("bids are held by the treasury; outbids "
                  "refund in full; a late bid extends the close")
        buttons.append([("🏘️ Property", "property")])
        return self._card(message, card, buttons)

    def _control_mortgage(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if parts and parts[0].lower() == "borrow":
            if len(parts) < 3 or not parts[2].isdigit():
                return ("usage: /mortgage borrow <property id> "
                        "<amount>")
            result = surface.estate_borrow(key, user_id,
                                           parts[1],
                                           int(parts[2]))
            if not result["ok"]:
                return {
                    "not_found": "no property of yours with "
                                 "that id.",
                    "currency": "gem-priced tiers can't "
                                "collateralise a sparks loan.",
                    "listed": "a listed property can't be "
                              "mortgaged.",
                    "auction": "a property at auction can't "
                               "be mortgaged.",
                    "mortgaged": "that property already "
                                 "carries a mortgage.",
                    "ltv": f"too much — against a value of "
                           f"{result.get('value')}, the most "
                           f"the treasury lends is "
                           f"{result.get('max')}.",
                    "funds": f"the treasury can't lend that "
                             f"right now — "
                             f"{result.get('available', 0)} "
                             f"available after reservations.",
                    "jailed": "🔒 no borrowing from jail.",
                    "shape": "the amount must be positive.",
                }.get(result["reason"], "can't borrow that.")
            from ...social.chat.typography import Card, sparks

            m = result["mortgage"]
            card = Card("economy", title="Mortgage drawn",
                        icon="🏦")
            card.line(f"**{sparks(m['principal'])}** landed in "
                      f"your wallet, against `{m['property']}`.")
            card.stats([
                ("Interest (flat, stated)",
                 sparks(m["interest"]), True),
                ("Total debt",
                 sparks(m["principal"] + m["interest"]), True),
                ("Instalment",
                 f"{sparks(m['installment'])} / week from "
                 f"your BANK", True),
                ("Periods", str(m["periods_left"])),
            ])
            card.note("miss 3 weekly instalments and the "
                      "property is repossessed — on the record")
            return self._card(message, card,
                              [[("🏦 Mortgage", "mortgage"),
                                ("🪙 My balance", "balance")]])
        if parts and parts[0].lower() == "repay":
            if len(parts) < 3 or not parts[2].isdigit():
                return ("usage: /mortgage repay <mortgage id> "
                        "<amount>")
            result = surface.estate_repay(key, user_id, parts[1],
                                          int(parts[2]))
            if not result["ok"]:
                return {
                    "not_found": "no open mortgage of yours "
                                 "with that id — /mortgage "
                                 "lists yours.",
                    "funds": "not enough sparks in your "
                             "wallet for that payment.",
                    "shape": "the amount must be positive.",
                }.get(result["reason"], "can't repay that.")
            if result["closed"]:
                return ("🏦 mortgage cleared — the property is "
                        "yours outright again.")
            from ...social.chat.typography import sparks

            return (f"🏦 paid {sparks(result['paid'])} — "
                    f"{sparks(result['remaining'])} of debt "
                    f"left.")
        # bare: my mortgages
        from ...social.chat.typography import Card, fmt_amount

        mine = surface.estate_mortgages(key, user_id)
        card = Card("economy", title="Your mortgages", icon="🏦")
        if not mine:
            card.line("no mortgages — borrow against a "
                      "property: /mortgage borrow <property "
                      "id> <amount>")
        buttons = []
        lines = []
        import time as _time

        for m in mine:
            due_in = max(0, int(float(m["next_due"])
                                - _time.time()))
            due = (f"{due_in // 86400}d" if due_in >= 86400
                   else f"{due_in // 3600}h")
            lines.append(
                f"`{m['id']}` on `{m['property']}` — "
                f"**{fmt_amount(m['remaining'])} sparks** of "
                f"debt · instalment "
                f"{fmt_amount(m['installment'])} from your "
                f"bank in {due}"
                + (f" · ⚠️ {m['missed']} missed"
                   if m["missed"] else ""))
            pay = min(int(m["installment"]),
                      int(m["remaining"]))
            if surface.economy.balance(key, user_id) >= pay \
                    and pay > 0:
                buttons.append(
                    [(f"Repay {fmt_amount(pay)}",
                      f"mortgage repay {m['id']} {pay}")])
        card.block_lines(lines)
        cfg = surface.estate_cfg(key)
        card.note(f"the treasury lends up to "
                  f"{cfg['ltv_pct']}% of a property's value at "
                  f"{cfg['mortgage_rate_pct']}% per week, flat; "
                  f"instalments debit your bank weekly")
        buttons.append([("🏘️ Property", "property")])
        return self._card(message, card, buttons)

    def _control_insure(self, tail, message):  # type: ignore[no-untyped-def]
        key, err = self._estate_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        parts = str(tail or "").split()
        if parts and parts[0].lower() == "buy" \
                and len(parts) >= 2:
            result = surface.estate_insure(key, user_id,
                                           parts[1])
            if not result["ok"]:
                return {
                    "not_found": "no property of yours with "
                                 "that id.",
                    "funds": f"the premium is "
                             f"{result.get('premium')} sparks — "
                             f"not enough in your wallet.",
                    "gems": f"the premium is "
                            f"{result.get('premium')} 💎 — not "
                            f"enough gems (premiums price in the "
                            f"property's own currency).",
                }.get(result["reason"], "can't insure that.")
            import time as _time

            until = _time.strftime(
                "%Y-%m-%d",
                _time.gmtime(result["insured_until"]))
            premium = (f"{result['premium']} 💎"
                       if result["currency"] == "gems"
                       else f"{result['premium']} sparks")
            return (f"🛡️ insured for 30 days (until {until}) "
                    f"— premium {premium}. Repair-wave bills "
                    f"on it are covered in full.")
        if parts and parts[0].lower() == "pay" \
                and len(parts) >= 2:
            result = surface.estate_pay_repair(key, user_id,
                                               parts[1])
            if not result["ok"]:
                return {
                    "not_found": "no property of yours with "
                                 "that id.",
                    "no_debt": "that property carries no "
                               "repair debt.",
                    "funds": f"nothing payable — the debt is "
                             f"{result.get('debt')} and your "
                             f"wallet is empty.",
                }.get(result["reason"], "can't pay that.")
            from ...social.chat.typography import sparks

            tail_text = (f" — {sparks(result['remaining'])} "
                         f"still owed"
                         if result["remaining"] else
                         " — debt cleared")
            return (f"🔧 paid {sparks(result['paid'])} of "
                    f"repair debt{tail_text}.")
        # bare: my cover
        import time as _time

        from ...community.group_real_estate import (
            INSURANCE_PREMIUM_PCT, TIERS)
        from ...social.chat.typography import Card, fmt_amount

        summary = surface.estate_summary(key, user_id)
        card = Card("economy", title="Insurance", icon="🛡️")
        if not summary["properties"]:
            card.line("you don't own property yet — insurance "
                      "rides on property. /property shows the "
                      "map.")
            return self._card(message, card,
                              [[("🏘️ Property", "property")]])
        buttons = []
        lines = []
        for prop in summary["properties"]:
            spec = TIERS[prop["tier"]]
            insured_until = float(prop.get("insured_until")
                                  or 0)
            if insured_until > _time.time():
                cover = ("insured until " + _time.strftime(
                    "%Y-%m-%d", _time.gmtime(insured_until)))
            else:
                cover = "uninsured"
                premium = max(1, int(round(
                    prop["value"] * INSURANCE_PREMIUM_PCT / 100)))
                cur = "💎" if prop["currency"] == "gems" else ""
                buttons.append(
                    [(f"Insure {spec['name']} · "
                      f"{fmt_amount(premium)}{cur}",
                      f"insure buy {prop['id']}")])
            debt = int(prop.get("repair_debt") or 0)
            lines.append(
                f"`{prop['id']}` {spec['icon']} "
                f"**{spec['name']}** — {cover}"
                + (f" · 🔧 repair debt {debt}" if debt else ""))
            if debt:
                buttons.append(
                    [(f"Pay repair debt · {debt}",
                      f"insure pay {prop['id']}")])
        card.block_lines(lines)
        card.note("a policy runs 30 days; repair-wave bills "
                  "on insured property are covered in full — "
                  "market falls are not insurable")
        buttons.append([("🏘️ Property", "property")])
        return self._card(message, card, buttons)

    def _execute_group_action(self, surface, key, adapter, message,
                              action) -> bool:  # type: ignore[no-untyped-def]
        """Execute one moderation action through the adapter's
        admin surface; audit the outcome either way.  Returns True
        only when the platform actually performed it."""
        kind = str(action.get("kind") or "")
        user_id = str(action.get("user_id") or "")
        if kind == "kick":
            # A kick is a ban immediately lifted (the /kick
            # grammar): out now, free to rejoin later.
            ban = getattr(adapter, "admin_ban", None) if adapter \
                else None
            unban = getattr(adapter, "admin_unban", None) \
                if adapter else None
            if ban is None or unban is None:
                surface._audit(
                    key, "moderation_unexecuted", action=kind,
                    rule=action.get("rule"), user_id=user_id,
                    reason="adapter has no admin_ban/admin_unban")
                surface.store.save()
                return False
            try:
                out = ban(message.chat, user_id)
                if isinstance(out, dict) and out.get("ok") is False:
                    raise RuntimeError(str(out.get("error")
                                           or "platform refused"))
                unban(message.chat, user_id)
            except Exception as exc:  # noqa: BLE001 - audited
                surface._audit(
                    key, "moderation_unexecuted", action=kind,
                    rule=action.get("rule"), user_id=user_id,
                    reason=f"{type(exc).__name__}: {exc}")
                surface.store.save()
                return False
            surface._audit(key, "moderation_executed", action=kind,
                           rule=action.get("rule"), user_id=user_id)
            surface.store.save()
            return True
        if kind in ("acceptance_restrict", "acceptance_unrestrict"):
            # §28's restriction is INDEFINITE (until the member
            # fixes the miss or an admin releases them) — it has
            # its own executor because a timed mute's grammar
            # would quietly turn it into an hour.
            ok = self._acceptance_restrict(
                adapter, message.chat, user_id,
                on=(kind == "acceptance_restrict"))
            surface._audit(
                key, "moderation_executed" if ok
                else "moderation_unexecuted",
                action=kind, rule=action.get("rule"),
                user_id=user_id,
                reason="" if ok else
                "adapter restrict surface refused or missing")
            surface.store.save()
            return ok
        method = {"delete": "admin_delete", "ban": "admin_ban",
                  "mute": "admin_restrict"}.get(kind)
        if method is None:
            return kind == "warn"  # warn executes as the notice
        fn = getattr(adapter, method, None) if adapter else None
        if fn is None:
            surface._audit(
                key, "moderation_unexecuted", action=kind,
                rule=action.get("rule"), user_id=user_id,
                reason=f"adapter has no {method}")
            surface.store.save()
            return False
        try:
            if kind == "delete":
                out = fn(message.chat, str(message.message_id or ""))
            elif kind == "ban":
                out = fn(message.chat, user_id)
            else:
                # A timed mute (1h by default posture; defense
                # actions carry their configured minutes); the two
                # Telegram adapters spell it differently (minutes
                # vs until) — speak each one's own grammar.
                minutes = int(action.get("minutes") or 60)
                try:
                    out = fn(message.chat, user_id, minutes=minutes)
                except TypeError:
                    out = fn(message.chat, user_id,
                             until=int(time.time()) + minutes * 60)
            # Every adapter reports refusal as an ok:False dict,
            # not an exception (the kick branch's check, adopted
            # here): a refused delete/ban/mute is an unexecution
            # with the platform's reason, never an execution.
            if isinstance(out, dict) and out.get("ok") is False:
                raise RuntimeError(str(out.get("error")
                                       or "platform refused"))
        except Exception as exc:  # noqa: BLE001 - audited, not raised
            surface._audit(
                key, "moderation_unexecuted", action=kind,
                rule=action.get("rule"), user_id=user_id,
                reason=f"{type(exc).__name__}: {exc}")
            surface.store.save()
            return False
        surface._audit(key, "moderation_executed", action=kind,
                       rule=action.get("rule"), user_id=user_id)
        surface.store.save()
        return True


def _parse_duration(text: str) -> tuple[int | None, str]:
    """Split a leading duration token off a moderation tail:
    ``10m`` / ``2h`` / ``1d`` / ``45s`` / ``inf``.  Returns
    (seconds, remaining reason).  No token → (None, full text);
    ``inf``/``forever`` → (0, reason) = indefinite."""
    parts = str(text or "").strip().split(None, 1)
    if not parts:
        return None, ""
    token = parts[0].lower()
    rest = parts[1] if len(parts) > 1 else ""
    if token in ("inf", "forever", "permanent"):
        return 0, rest
    import re

    match = re.fullmatch(r"(\d+)([smhd])", token)
    if not match:
        return None, str(text or "")
    scale = {"s": 1, "m": 60, "h": 3600, "d": 86400}[match.group(2)]
    return int(match.group(1)) * scale, rest


def _duration_text(seconds: int) -> str:
    if seconds >= 86400:
        return f"{seconds // 86400}d"
    if seconds >= 3600:
        return f"{seconds // 3600}h"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _fmt_ts(ts) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M UTC",
                             time.gmtime(float(ts)))
    except (TypeError, ValueError):
        return "?"


def _member_card(surface, key: str, member: dict) -> str:
    """The member card, through the typography organ (§27,
    moderation class): identity fields are literals — a hostile
    display name renders as text, never markup."""
    from ...social.chat.typography import Card, render_card

    name = str(member.get("name") or member.get("username")
               or "member")
    handle = (f"@{member['username']}" if member.get("username")
              else "no username")
    card = Card("moderation", title=name, icon="👤")
    joined = member.get("joined_ts") or member.get("first_seen")
    rows = [("Handle", handle),
            ("Joined", _fmt_ts(joined)),
            ("Messages", int(member.get("message_count") or 0)),
            ("Last seen", _fmt_ts(member.get("last_seen")))]
    # §19 recognition: the ladder rung is part of who a member
    # is here — and a worn-face flag is shown, never hidden.
    if member.get("role"):
        rows.append(("Standing", str(member["role"])))
    # §16: the member's tag (set by /tag; literal text, like
    # every identity field here) and their boost standing.
    if member.get("tag"):
        rows.append(("Tag", str(member["tag"])))
    _boosts = surface.boost_summary(
        key, str(member.get("user_id") or ""))
    if _boosts["count"]:
        rows.append(("Boosts", f"×{_boosts['count']}"))
    if member.get("impersonation_flag"):
        rows.append(("⚠️ Flag",
                     str(member["impersonation_flag"])))
    if member.get("left_ts"):
        rows.append(("Left", _fmt_ts(member["left_ts"])))
    card.stats(rows)
    # The estate rides the member card (§20f): what they hold
    # is part of who they are in the group.
    estate = surface.estate_summary(
        key, str(member.get("user_id") or ""))
    if estate["property_count"] or estate["gems"]:
        from ...social.chat.typography import sparks as _sparks

        card.stats([
            ("Properties",
             f"{estate['property_count']} · "
             f"{_sparks(estate['property_value'])}"),
            ("Gems", f"{estate['gems']} 💎"),
        ], title="Estate")
    changes = [c for c in surface.changes(key, limit=200)
               if c.get("user_id") == member.get("user_id")]
    if changes:
        card.board("\n".join(
            f"{change.get('kind', 'changed')}: "
            f"{change.get('old') or '—'} → "
            f"{change.get('new') or '—'} "
            f"({_fmt_ts(change.get('ts'))})"
            for change in changes[-3:]),
            title="Identity changes")
    return render_card(
        card, platform=str(key).split(":", 1)[0],
        settings=surface.store.settings(key)).text


def _audit_line(entry: dict) -> str:
    kind = entry.get("kind", "?")
    bits = [_fmt_ts(entry.get("ts")), kind]
    for field_name in ("action", "rule", "user_id", "change_kind",
                       "reason"):
        if entry.get(field_name):
            bits.append(f"{field_name}={entry[field_name]}")
    if entry.get("count"):
        bits.append(f"count={entry['count']}")
    if entry.get("suppressed"):
        bits.append(f"suppressed={entry['suppressed']}")
    if entry.get("signature"):
        bits.append(f"sig={entry['signature']}")
    if entry.get("excerpt"):
        bits.append(f"“{str(entry['excerpt'])[:60]}”")
    if entry.get("old") or entry.get("new"):
        bits.append(f"{entry.get('old') or '—'} → "
                    f"{entry.get('new') or '—'}")
    return " · ".join(str(b) for b in bits)
