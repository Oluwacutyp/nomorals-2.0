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


class RuntimeGroupMixin:
    """Group-surface turn for PartnerRuntime._process_inner."""

    def _group_surface(self):  # type: ignore[no-untyped-def]
        surface = getattr(self, "_group_surface_obj", None)
        if surface is None:
            from ...social.chat.group_surface import default_surface

            ws_dir = getattr(self.context.settings, "workspace_dir", ".")
            surface = default_surface(ws_dir)
            self._group_surface_obj = surface
        return surface

    def _send(self, chat, text: str, *, buttons=None,  # type: ignore[no-untyped-def]
              parse_mode: str = "") -> None:
        """The mixin's own outbound channel (welcomes, goodbyes,
        notices).  This method was CALLED by every welcome path
        since P1 but defined NOWHERE on PartnerRuntime — the live
        bug found 2026-11-03: every join's welcome died on
        AttributeError, caught upstream and logged, never sent.
        Defined here, on the mixin that calls it, riding the same
        gateway contract (buttons + parse_mode) as every surface."""
        self.gateway.send(chat.platform, chat, str(text),
                          buttons=buttons, parse_mode=parse_mode)

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

    def _card(self, message, card, buttons=None):  # type: ignore[no-untyped-def]
        """Render a typography Card (§27) for this message's
        platform under the group's presentation settings, and
        bundle the buttons the handler authored from the same
        state.  This is the ONE way group control replies are
        composed now: handlers name what renders, the group's
        theme decides how it looks."""
        from ...social.chat.tgbot_buttons import actionable
        from ...social.chat.typography import render_card

        surface = self._group_surface()
        key = message.chat.key
        rendered = render_card(
            card, platform=str(message.chat.platform or ""),
            settings=surface.store.settings(key))
        if buttons is not None:
            return actionable(rendered.text, buttons)
        return rendered.text

    def _welcome_buttons(self, surface: Any, key: str) -> list:
        """Buttons authored from the group's LIVE state: rules are
        always reachable; the sparks row appears only when the
        economy actually runs here."""
        rows = [[("📜 Rules", "rules")]]
        try:
            if surface.economy_on(key):
                rows.append([("🪙 My sparks", "balance"),
                             ("🗺️ Quests", "quests")])
        except Exception:  # noqa: BLE001 - buttons never block a welcome
            pass
        return rows

    def _group_adapter(self, platform: str):  # type: ignore[no-untyped-def]
        gateway = getattr(self, "gateway", None)
        adapters = getattr(gateway, "adapters", {}) or {}
        return adapters.get(platform)

    def _group_surface_turn(self, message) -> bool:  # type: ignore[no-untyped-def]
        """Run one group message through the surface.  Returns
        True when the message is fully consumed (service event,
        or moderated away)."""
        surface = self._group_surface()
        key = message.chat.key
        meta = message.meta or {}
        service = str(meta.get("service") or "")

        if service == "join":
            welcomed = []
            for member in meta.get("new_members") or []:
                if not isinstance(member, dict):
                    continue
                text = surface.record_join(
                    key, str(member.get("id") or ""),
                    username=str(member.get("username") or ""),
                    name=str(member.get("name") or ""),
                    group_name=str(message.chat.title or ""))
                if text:
                    welcomed.append(text)
            for text in welcomed:
                self._send(
                    message.chat, text,
                    buttons=self._welcome_buttons(surface, key),
                    parse_mode=self._rendered_mode(message, text))
            self._clean_service_message(surface, message)
            return True
        if service == "leave":
            left = meta.get("left_member") or {}
            if isinstance(left, dict) and left.get("id"):
                text = surface.record_leave(
                    key, str(left.get("id") or ""),
                    group_name=str(message.chat.title or ""))
                if text:
                    self._send(
                        message.chat, text,
                        parse_mode=self._rendered_mode(message,
                                                       text))
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
                    if callout.get("to") == "group":
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
        result = surface.handle_message(
            key, user_id,
            username=str(message.sender_username or ""),
            name=str(message.sender or ""),
            text=str(message.text or ""),
            message_id=str(message.message_id or ""),
            envelope=meta.get("envelope"),
            sender_is_admin=self._group_admin_ok(message),
            sender_is_operator=self._operator_or_false(message))
        for notice in result.get("notices") or []:
            self._send(message.chat, str(notice))
        self._execute_pending_pin(surface, message)
        actions = result.get("actions") or []
        if not actions:
            return False
        adapter = self._group_adapter(message.chat.platform)
        consumed = False
        for action in actions:
            kind = str(action.get("kind") or "")
            executed = self._execute_group_action(
                surface, key, adapter, message, action)
            warning = str(action.get("warning") or "")
            if warning and executed:
                # The organ's one public warning IS the record
                # (no meta-narration): name + rule + consequence.
                self._send(message.chat, warning)
            elif kind == "warn" and executed:
                who = (message.sender_username
                       and f"@{message.sender_username}") \
                    or message.sender or "there"
                self._send(
                    message.chat,
                    f"⚠️ {who} — {action.get('reason') or 'warning'}")
            if kind in ("delete", "ban", "kick") and executed:
                consumed = True
        return consumed

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
                return
            fn(message.chat, str(message.message_id))
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

    def _clean_service_message(self, surface, message) -> None:  # type: ignore[no-untyped-def]
        """Service-message hygiene (P1): the raw "X joined the
        group" service message is deleted after the surface has
        handled it — the welcome is the visible artifact, the
        platform's own notice is clutter.  Setting-gated,
        best-effort, audited on success."""
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
            fn(message.chat, str(message.message_id))
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
        theme decides the form."""
        from ...social.chat.typography import Card

        key = message.chat.key
        surface = self._group_surface()
        settings = surface.store.settings(key)
        members = surface.members(key)
        eco = surface.economy_on(key)
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
            "/help /balance — the full page for any command")
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
            "moderation, by hand: reply to a member's message "
            "with /warn, /mute 10m|2h|1d, /ban or /kick — or name "
            "@member instead. /unmute /unban undo them; /pin and "
            "/purge work on the message you reply to; warns pile "
            "up to an automatic mute (limit in /gsettings).",
            "/glog — the group audit log (I keep it; I only show "
            "it when an admin asks)",
            "/strikes — the defense strike ledger: who is on "
            "the ladder, and how close the rungs are",
            "I moderate on my own: floods, repeat spam and "
            "off-allowlist links get deleted; crash payloads are "
            "quarantined, channel injections and mass tags are "
            "struck; repeat offenders are muted, kicked, then "
            "banned — all tuned in /gsettings.",
        ], title="for admins:")
        card.note(
            f"{len(members)} members in my ledger · "
            f"{len(settings.get('rules') or [])} rules · welcome "
            f"{'on' if settings['welcome_enabled'] else 'off'} · "
            f"link moderation "
            f"{'allowlist' if (settings['moderation'].get('link_allowlist')) else 'open'}")
        buttons = [[("📜 Rules", "rules"), ("🎮 Games", "game")]]
        if eco:
            buttons.append([("🪙 My sparks", "balance"),
                            ("🗺️ Quests", "quests"),
                            ("🏦 Treasury", "treasury")])
        buttons.append([("⚙️ Group settings", "gsettings"),
                        ("📋 Audit log", "glog")])
        return self._card(message, card, buttons)

    def _group_key_or_none(self, message):  # type: ignore[no-untyped-def]
        if message is None:
            return None
        from ...social.chat.base import ChatKind

        if message.chat.kind != ChatKind.GROUP:
            return None
        return message.chat.key

    def _control_rules(self, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        return self._group_surface().rules_text(key)

    def _control_member(self, arg, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        surface = self._group_surface()
        members = surface.members(key)
        target = None
        wanted = str(arg or "").strip().lstrip("@").lower()
        if wanted:
            for uid, member in members.items():
                if uid == wanted or str(
                        member.get("username") or "").lower() == wanted \
                        or str(member.get("name") or "").lower() == wanted:
                    target = member
                    break
            if target is None:
                return (f"I haven't seen {arg} in this group yet — "
                        f"the ledger only knows people who've been "
                        f"active or joined since I started watching.")
        else:
            target = members.get(str(message.sender_id or ""))
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

            def _toggle(label: str, kind: str, on_now: bool) -> tuple:
                # The button SHOWS the current state and carries
                # the flip — authored from the live settings, so
                # it can never disagree with the summary above it.
                return (f"{label}: {'ON' if on_now else 'OFF'}",
                        f"gsettings {kind} "
                        f"{'off' if on_now else 'on'}")

            pres = s.get("presentation") or {}
            theme_cycle = {"house": "clean", "clean": "board",
                           "board": "house"}
            theme_now = str(pres.get("theme") or "house")
            buttons = [
                [_toggle("👋 Welcome", "welcome",
                         bool(s["welcome_enabled"])),
                 _toggle("🚪 Goodbye", "goodbye",
                         bool(s["goodbye_enabled"]))],
                [_toggle("🧹 Cleanup", "clean",
                         bool(s["clean_service"])),
                 _toggle("🪙 Economy", "economy",
                         bool(s["economy_enabled"]))],
                [(f"🎨 Theme: {theme_now}",
                  f"gsettings theme "
                  f"{theme_cycle.get(theme_now, 'house')}"),
                 _toggle("🔖 Stamp", "stamp",
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
                f" ({d_ed.get('mode', 'log')}) · strikes: mute at "
                f"{s.get('warn_threshold', 3)} · kick at "
                f"{d_esc.get('kick_at', 5)} · ban at "
                f"{d_esc.get('ban_at', 7)} "
                f"({d_esc.get('window_days', 7)}-day window)",
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
                "`/gsettings warns N`",
                "`/gsettings fine N` · "
                "`/gsettings governance N`",
                "`/gsettings filter add WORD` · "
                "`/gsettings filter action delete|warn|mute`",
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
            ], title="Change it")
            card.note("tap a button to flip a switch, or type a "
                      "command")
            return self._card(message, card, buttons)
        parts = text.split(None, 2)
        head = parts[0].lower()
        rest = parts[1] if len(parts) > 1 else ""
        rest2 = parts[2] if len(parts) > 2 else ""
        on = rest.lower() == "on"
        if head == "warns" and rest.isdigit():
            surface.update_settings(
                key, warn_threshold=max(1, int(rest)))
            return f"warn limit {max(1, int(rest))} before an auto-mute."
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
        return ("I didn't follow that — /gsettings alone shows the "
                "current settings and the change grammar.")

    # ---- manual moderation (DESIGN §4: the admin command set) ------
    def _mod_gate(self, message):  # type: ignore[no-untyped-def]
        """(key, None) for an admin in a group, else (None, text)."""
        key = self._group_key_or_none(message)
        if key is None:
            return None, "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return None, "that's an admin call — I'm not taking it from a non-admin."
        return key, None

    def _mod_target(self, surface, key, tail, message):  # type: ignore[no-untyped-def]
        """Resolve who a moderation command acts on, in the
        grammar every group bot uses: (1) the AUTHOR of the
        message the admin replied to (adapter meta, then the
        ledger's last-message map), (2) @username or id in the
        command text.  Returns (user_id, label, rest)."""
        meta = message.meta or {}
        rts = meta.get("reply_to_sender") or {}
        if isinstance(rts, dict) and rts.get("id"):
            label = (rts.get("username")
                     and f"@{rts['username']}") or rts.get("name") \
                or str(rts["id"])
            return str(rts["id"]), label, str(tail or "")
        if getattr(message, "reply_to", ""):
            for uid, member in surface.members(key).items():
                if str(member.get("last_message_id") or "") == str(
                        message.reply_to):
                    label = (member.get("username")
                             and f"@{member['username']}") \
                        or member.get("name") or uid
                    return uid, label, str(tail or "")
        parts = str(tail or "").split(None, 1)
        if parts:
            uid = self._member_ref(surface, key, parts[0], message)
            if uid:
                return uid, parts[0], (parts[1] if len(parts) > 1
                                       else "")
        return None, "", str(tail or "")

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

    def _control_mute(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id, label, rest = self._mod_target(
            surface, key, tail, message)
        if not user_id:
            return ("who should I mute? Reply to their message with "
                    "/mute, or /mute @member [10m|2h|1d].")
        seconds, reason = _parse_duration(rest)
        if seconds is None:
            seconds = 3600
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
        user_id, label, _ = self._mod_target(surface, key, tail,
                                             message)
        if not user_id:
            return "who should I unmute? Reply to them or /unmute @member."

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
        user_id, label, rest = self._mod_target(
            surface, key, tail, message)
        if not user_id:
            return "who should I ban? Reply to their message or /ban @member."

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
        user_id, label, _ = self._mod_target(surface, key, tail,
                                             message)
        if not user_id:
            return "who should I unban? /unban @member or their id."

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
        user_id, label, rest = self._mod_target(
            surface, key, tail, message)
        if not user_id:
            return "who should I kick? Reply to their message or /kick @member."

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
        user_id, label, rest = self._mod_target(
            surface, key, tail, message)
        if not user_id:
            return "who should I warn? Reply to their message or /warn @member."
        actor = str(message.sender_id or message.sender or "")
        count = surface.record_warn(key, user_id, by=actor,
                                    reason=rest.strip())
        threshold = int(surface.store.settings(
            key).get("warn_threshold") or 3)
        if count >= threshold:
            ok = self._mod_run(
                surface, key, message, "mute", user_id,
                self._mod_mute_call(message, user_id, 3600))
            surface.clear_warns(key, user_id)
            tail_note = (" — that's the limit: muted for an hour, "
                         "warns reset.") if ok else \
                " — that's the limit, but the mute was refused (see /glog)."
            return f"⚠️ {label}: warning {count}/{threshold}{tail_note}"
        return (f"⚠️ {label} — warning {count}/{threshold}"
                + (f": {rest.strip()}" if rest.strip() else ""))

    def _control_strikes(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        """The strike ledger (defense brief §2.2): one member's
        record, or the group's current holders when no target is
        named.  Read from the same spine the sensors write."""
        key, err = self._mod_gate(message)
        if err:
            return err
        surface = self._group_surface()
        user_id, label, _ = self._mod_target(
            surface, key, tail, message)
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
        cfg = surface.defense_config(key)
        threshold = int(surface.store.settings(
            key).get("warn_threshold") or 3)
        rungs = []
        if threshold and info["count"] < threshold:
            rungs.append(f"mute at {threshold}")
        if cfg.kick_at and info["count"] < cfg.kick_at:
            rungs.append(f"removal at {cfg.kick_at}")
        if cfg.ban_at and info["count"] < cfg.ban_at:
            rungs.append(f"ban at {cfg.ban_at}")
        if rungs:
            card.note("next: " + " · ".join(rungs))
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
        user_id, label, rest = self._mod_target(
            surface, key, tail, message)
        reporter = str(message.sender_id or message.sender or "")
        surface._audit(key, "report", user_id=user_id or "",
                       by=reporter, target_label=label,
                       reason=str(tail or "")[:300],
                       replied_text=str(
                           (message.meta or {}).get(
                               "reply_to_text") or "")[:200])
        surface.store.save()
        if user_id:
            return (f"📣 reported {label} — it's in the group log "
                    f"for the admins (/glog).")
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
            return "the delete didn't go through — the platform refused it (it's in /glog)."
        return "🧹 gone."

    # ---- economy commands (P2) --------------------------------------
    def _economy_gate(self, message):  # type: ignore[no-untyped-def]
        """Shared gate for economy commands: in-group + economy
        on + ledger healthy.  Returns (key, None) or (None, text)."""
        key = self._group_key_or_none(message)
        if key is None:
            return None, "that's a group command — run it inside the group."
        surface = self._group_surface()
        frozen = surface.economy_frozen(key)
        if frozen:
            return None, ("the group economy is frozen — its ledger "
                          "needs an admin's eyes before any sparks "
                          "move again.")
        if not surface.store.settings(key)["economy_enabled"]:
            return None, "the group economy is off here."
        return key, None

    def _member_ref(self, surface, key, token, message):  # type: ignore[no-untyped-def]
        """Resolve a /pay target: @username from the member
        ledger, or a raw member id."""
        token = str(token or "").strip()
        if token.startswith("@"):
            want = token[1:].lower()
            for uid, member in surface.members(key).items():
                if str(member.get("username") or "").lower() == want:
                    return uid
            return None
        return token or None

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
                   [("🗺️ Quests", "quests"), ("🛒 Shop", "gshop")]]
        vault_row = []
        if wallet > 0:
            vault_row.append(("🏦 Deposit", "deposit all"))
        if bank > 0:
            vault_row.append(("💵 Withdraw", "withdraw all"))
        if vault_row:
            buttons.append(vault_row)
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
        recipient = self._member_ref(surface, key, target_token,
                                     message)
        if not recipient:
            return f"I don't have {target_token} in this group's ledger."
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
                    f"ledgered ({grant['ledger_id']}).")
        from ...social.chat.typography import Card, sparks

        balance = surface.economy.balance(key, user_id)
        card = Card("economy", title="Group shop", icon="🛒")
        card.line(f"your sparks: **{sparks(balance)}**")
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
        buttons.append([("🪙 My balance", "balance")])
        card.block_lines(shelf)
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
        if proposals:
            card.line(f"🗳️ {len(proposals)} treasury proposal(s) "
                      "waiting for a second admin (/approve).")
        card.note("triple 7️⃣ on /slots takes the jackpot pot")
        return self._card(message, card)

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
            return self._card(
                message, card,
                [[(f"🎁 Claim {drop['per_claim']} sparks", "claim"),
                  ("🪙 My balance", "balance")]])
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
        return self._card(
            message, card,
            [[(f"🎁 Claim {drop['per_claim']} sparks", "claim"),
              ("🪙 My balance", "balance")]])

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
        meta = getattr(message, "meta", None) or {}
        replied = meta.get("reply_to_sender")
        if replied:
            return str(replied)
        for part in str(tail or "").split():
            if part.startswith("@") or part.isdigit():
                return self._member_ref(surface, key, part, message)
        return None

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
        from ...social.chat.typography import Card, sparks

        buttons = [[(f"🎰 Spin again ({bet})", f"slots {bet}"),
                    ("🪙 My balance", "balance")]]
        if result.get("jackpot"):
            card = Card("game", title="Slots", icon="🎰")
            card.board(reels)
            card.line(f"🎉 **JACKPOT — {sparks(result['jackpot'])}!** "
                      f"plus {sparks(result['payout'])} from the "
                      f"reels.")
            card.stats([("Balance", sparks(result["balance"]),
                         True)])
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
        from ...social.chat.typography import Card, sparks

        buttons = [[(f"🪙 Flip again ({bet})", f"coinflip {bet}"),
                    ("🪙 My balance", "balance")]]
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
            value = sparks(row["total"])
            if row["bank"]:
                value += f" (🏦 {sparks(row['bank'])} banked)"
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
        if not self._group_admin_ok(message):
            return "treasury grants are for admins."
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
            return "usage: /grant @member 100"
        recipient = self._member_ref(surface, key, target_token,
                                     message)
        if not recipient:
            return (f"I don't have {target_token} in this "
                    "group's ledger.")
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

    def _control_approve(self, tail, message):  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "approvals are for admins."
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
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
        result = surface.approve_spend(key, pid, by=admin)
        if not result["ok"]:
            return {"no_proposal": "no open proposal with that id.",
                    "self_approve": "the proposer can't approve "
                                    "their own spend — a second "
                                    "admin has to.",
                    "funds": "the treasury can't cover it any "
                             "more."}.get(result["reason"],
                                          "can't approve that.")
        return (f"✅ proposal <code>{pid}</code> approved and "
                "executed.")

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
                fn(message.chat, str(message.message_id or ""))
            elif kind == "ban":
                fn(message.chat, user_id)
            else:
                # A timed mute (1h by default posture; defense
                # actions carry their configured minutes); the two
                # Telegram adapters spell it differently (minutes
                # vs until) — speak each one's own grammar.
                minutes = int(action.get("minutes") or 60)
                try:
                    fn(message.chat, user_id, minutes=minutes)
                except TypeError:
                    fn(message.chat, user_id,
                       until=int(time.time()) + minutes * 60)
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
    if member.get("left_ts"):
        rows.append(("Left", _fmt_ts(member["left_ts"])))
    card.stats(rows)
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
