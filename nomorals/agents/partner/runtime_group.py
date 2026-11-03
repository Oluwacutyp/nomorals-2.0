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
                self._send(message.chat, text,
                           buttons=self._welcome_buttons(surface, key))
            self._clean_service_message(surface, message)
            return True
        if service == "leave":
            left = meta.get("left_member") or {}
            if isinstance(left, dict) and left.get("id"):
                text = surface.record_leave(
                    key, str(left.get("id") or ""),
                    group_name=str(message.chat.title or ""))
                if text:
                    self._send(message.chat, text)
            self._clean_service_message(surface, message)
            return True

        user_id = str(message.sender_id or message.sender or "")
        if not user_id:
            return False
        result = surface.handle_message(
            key, user_id,
            username=str(message.sender_username or ""),
            name=str(message.sender or ""),
            text=str(message.text or ""),
            message_id=str(message.message_id or ""))
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
            if kind == "warn" and executed:
                who = (message.sender_username
                       and f"@{message.sender_username}") \
                    or message.sender or "there"
                self._send(
                    message.chat,
                    f"⚠️ {who} — {action.get('reason') or 'warning'}")
            if kind in ("delete", "ban") and executed:
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

    # ---- group commands (P1 slice 2) --------------------------------
    def _control_group_help(self, message):  # type: ignore[no-untyped-def]
        """Bare /help INSIDE a group (owner order 2026-11-03):
        list what is actually available in this group — the member
        surface and the admin surface side by side, rendered from
        this group's live state, not the global owner catalog.
        Everyone sees both sections; the admin commands themselves
        stay gated at execution.

        Presentation build: the reply is an ActionableMessage —
        HTML text and buttons authored from the SAME state (the
        sparks row exists exactly when the economy runs here; the
        buttons mirror the sections on screen)."""
        import html as _html

        from ...social.chat.tgbot_buttons import actionable
        key = message.chat.key
        surface = self._group_surface()
        settings = surface.store.settings(key)
        members = surface.members(key)
        eco = surface.economy_on(key)
        title = _html.escape(message.chat.title or "this group")
        lines = [
            f"🛠️ <b>{title}</b> — what I do here",
            "",
            "<b>for everyone:</b>",
            "  /rules — the group rules",
            "  /report — flag a message or member for the admins "
            "(reply to it, or /report @member why)",
            "  /member [@name] — a member's card: tenure, activity, "
            "username/name changes",
            "  /ping — check I'm alive, and how fast I answered",
            "  🎮 games — /ttt /connect4 /reversi /checkers /mines "
            "and the rest of the board; /game lists them, moves are "
            "plain text (a1, b2 …). Some games queue you against "
            "players on other platforms.",
        ]
        if eco:
            lines.append(
                "  🪙 sparks (the group purse) — /balance · /daily · "
                "/mine · /work · /pay @member 50 · /deposit · "
                "/withdraw · /gshop · /quests · /treasury · /top · "
                "/tax · /claim when a drop is live. Chatting, "
                "streaks, quests and in-group game wins all earn.")
            lines.append(
                "  🥷 the risky side — /rob @member (fail and it's "
                "jail; /bail gets you out) · 🎰 /slots 10 · 🪙 "
                "/coinflip 10 — losses feed the treasury and the "
                "jackpot pot; triple 7️⃣ takes the pot.")
        lines += [
            "  /help /balance — the full page for any command",
            "",
            "<b>for admins:</b>",
            "  /gsettings — welcome & goodbye, the rules list, "
            "flood/link moderation, service-message cleanup, the "
            "economy switch and transfer tax",
            "  /drop 100 10 — fund a giveaway pool from your own "
            "sparks (or /drop treasury 100 10 from the treasury); "
            "members claim with /claim",
            "  /grant @member 100 — pay a member from the treasury; "
            "above the governance threshold a second admin must "
            "/approve it first",
            "  moderation, by hand: reply to a member's message "
            "with /warn, /mute 10m|2h|1d, /ban or /kick — or name "
            "@member instead. /unmute /unban undo them; /pin and "
            "/purge work on the message you reply to; warns pile "
            "up to an automatic mute (limit in /gsettings).",
            "  /glog — the group audit log (I keep it; I only show "
            "it when an admin asks)",
            "  I moderate on my own: floods, repeat spam and "
            "off-allowlist links get deleted; repeat offenders are "
            "muted, then banned — all tuned in /gsettings.",
            "",
            f"<i>{len(members)} members in my ledger · "
            f"{len(settings.get('rules') or [])} rules · welcome "
            f"{'on' if settings['welcome_enabled'] else 'off'} · "
            f"link moderation "
            f"{'allowlist' if (settings['moderation'].get('link_allowlist')) else 'open'}</i>",
        ]
        buttons = [[("📜 Rules", "rules"), ("🎮 Games", "game")]]
        if eco:
            buttons.append([("🪙 My sparks", "balance"),
                            ("🗺️ Quests", "quests"),
                            ("🏦 Treasury", "treasury")])
        buttons.append([("⚙️ Group settings", "gsettings"),
                        ("📋 Audit log", "glog")])
        return actionable("\n".join(lines), buttons)

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
        import html as _html

        entries = self._group_surface().audit(key, limit=max(1, min(n, 50)))
        if not entries:
            return "nothing in the group log yet."
        lines = ["📋 <b>group log</b> (held by me, shown on demand):"]
        for entry in entries:
            lines.append(_html.escape(_audit_line(entry)))
        return "\n".join(lines)

    def _control_gsettings(self, tail, message) -> str:  # type: ignore[no-untyped-def]
        key = self._group_key_or_none(message)
        if key is None:
            return "that's a group command — run it inside the group."
        if not self._group_admin_ok(message):
            return "group settings are for admins."
        surface = self._group_surface()
        text = str(tail or "").strip()
        if not text:
            from ...social.chat.tgbot_buttons import actionable
            s = surface.store.settings(key)
            mod = s["moderation"]

            def _toggle(label: str, kind: str, on_now: bool) -> tuple:
                # The button SHOWS the current state and carries
                # the flip — authored from the live settings, so
                # it can never disagree with the summary above it.
                return (f"{label}: {'ON' if on_now else 'OFF'}",
                        f"gsettings {kind} "
                        f"{'off' if on_now else 'on'}")

            buttons = [
                [_toggle("👋 Welcome", "welcome",
                         bool(s["welcome_enabled"])),
                 _toggle("🚪 Goodbye", "goodbye",
                         bool(s["goodbye_enabled"]))],
                [_toggle("🧹 Cleanup", "clean",
                         bool(s["clean_service"])),
                 _toggle("🪙 Economy", "economy",
                         bool(s["economy_enabled"]))],
            ]
            return actionable(
                f"⚙️ <b>group settings</b>\n"
                f"welcome: {'on' if s['welcome_enabled'] else 'off'} · "
                f"goodbye: {'on' if s['goodbye_enabled'] else 'off'} · "
                f"service cleanup: {'on' if s['clean_service'] else 'off'}\n"
                f"rules: {len(s.get('rules') or [])} · "
                f"flood: {mod['flood_max']} per "
                f"{int(mod['flood_window_s'])}s · repeats ×"
                f"{mod['repeat_threshold']} · links: "
                f"{', '.join(mod.get('link_allowlist') or []) or 'open'}"
                f" · filters: {len(mod.get('word_filters') or [])} "
                f"({mod.get('filter_action') or 'delete'})\n"
                f"economy: {'on' if s['economy_enabled'] else 'off'}"
                f" · transfer tax {s.get('transfer_tax_pct', 2)}%"
                f" · fine {s.get('fine_amount', 0)}"
                f" · governance ≥{s.get('governance_threshold', 500)}\n"
                f"tap a button to flip a switch, or type:\n"
                f"<code>/gsettings welcome on|off</code> · "
                f"<code>/gsettings goodbye on|off</code> · "
                f"<code>/gsettings clean on|off</code>\n"
                f"<code>/gsettings rules add TEXT</code> · "
                f"<code>/gsettings rules clear</code>\n"
                f"<code>/gsettings flood N</code> · "
                f"<code>/gsettings links DOMAINS|off</code>\n"
                f"<code>/gsettings economy on|off</code> · "
                f"<code>/gsettings tax PCT</code> · "
                f"<code>/gsettings warns N</code>\n"
                f"<code>/gsettings fine N</code> · "
                f"<code>/gsettings governance N</code>\n"
                f"<code>/gsettings filter add WORD</code> · "
                f"<code>/gsettings filter action "
                f"delete|warn|mute</code>",
                buttons)
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
        import html as _html

        from ...social.chat.tgbot_buttons import actionable
        surface = self._group_surface()
        user_id = str(message.sender_id or message.sender or "")
        stmt = surface.economy.statement(key, user_id)
        member = surface.members(key).get(user_id) or {}
        name = _html.escape(
            str(member.get("name") or message.sender or "you"))
        wallet = int(stmt.get("balance", 0))
        bank = surface.economy.bank_balance(key, user_id)
        lines = [f"💰 <b>{name}</b> — wallet <b>{wallet}</b> sparks"]
        if bank:
            lines.append(f"🏦 banked: <b>{bank}</b> sparks "
                         f"(safe from robbery) · total "
                         f"<b>{wallet + bank}</b>")
        for badge in stmt.get("badges", []):
            lines.append(f"  🏅 {_html.escape(str(badge.get('badge')))}"
                         f" ({_html.escape(str(badge.get('season')))})")
        lines.append(
            f"🏦 group treasury: "
            f"{surface.economy.treasury_balance(key)} sparks · "
            f"jackpot pot: {surface.economy.jackpot_balance(key)}")
        lines.append(f"<i>{_html.escape(str(stmt.get('disclaimer') or ''))}</i>")
        buttons = [[("⛏️ Mine", "mine"), ("🛠️ Work", "work")],
                   [("🗺️ Quests", "quests"), ("🛒 Shop", "gshop")]]
        vault_row = []
        if wallet > 0:
            vault_row.append(("🏦 Deposit", "deposit all"))
        if bank > 0:
            vault_row.append(("💵 Withdraw", "withdraw all"))
        if vault_row:
            buttons.append(vault_row)
        return actionable("\n".join(lines), buttons)

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
        return actionable(
            f"⛏️ <b>+{entry.amount} sparks</b> "
            f"(balance {surface.economy.balance(key, user_id)}). "
            f"Your hashrate is {rate['hashrate']} — it's "
            f"engagement: real activity this week.", buttons)

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
        from ...social.chat.tgbot_buttons import actionable
        return actionable(
            f"🛠️ shift done: <b>+{result['pay']} sparks</b> "
            f"(level {result['level']}, balance "
            f"{result['balance']}).",
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
        fee_line = (f" ({result['fee']} to the treasury)"
                    if result["fee"] else "")
        from ...social.chat.tgbot_buttons import actionable
        return actionable(
            f"💸 paid <b>{result['net']} sparks</b>{fee_line}. "
            f"Your balance: {result['balance']}.",
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
        from ...social.chat.tgbot_buttons import actionable
        balance = surface.economy.balance(key, user_id)
        lines = [f"🛒 <b>group shop</b> — your sparks: "
                 f"<b>{balance}</b>"]
        buy_row: list = []
        buttons: list = []
        for item_id, item in STATUS_SHOP.items():
            afford = balance >= int(item["cost"])
            lines.append(
                f"<code>{item_id}</code> — <b>{item['name']}</b> · "
                f"{item['cost']} sparks · {item['blurb']}"
                + ("" if afford else " · <i>not yet</i>"))
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
        return actionable("\n".join(lines), buttons)

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
            from ...social.chat.tgbot_buttons import actionable
            return actionable(
                f"🏅 quest claimed: <b>+{entry.amount} sparks</b> "
                f"(balance "
                f"{surface.economy.balance(key, user_id)}).",
                [[("🪙 My balance", "balance"),
                  ("🗺️ Quests", "quests")]])
        from ...social.chat.tgbot_buttons import actionable
        progress = surface.economy.quest_progress(key, user_id)
        lines = ["🗺️ <b>your quests:</b>"]
        buttons = []
        for quest_id, quest in progress.items():
            mark = "✅ claimed" if quest["claimed"] else (
                "🎁 ready — /quests claim " + quest_id
                if quest["complete"] else
                f"{quest['have']}/{quest['need']}")
            lines.append(f"<code>{quest_id}</code> — "
                         f"<b>{quest['name']}</b>: "
                         f"{quest['blurb']} · {mark} · "
                         f"reward {quest['reward']}")
            # A claim button exists exactly for quests this
            # member can claim RIGHT NOW — authored from the
            # same progress state the list renders.
            if quest["complete"] and not quest["claimed"]:
                buttons.append([(f"🎁 Claim: {quest['name']}",
                                  f"quests claim {quest_id}")])
        # Group quests: the COLLECTIVE board — everyone's
        # activity counts; completion pays the treasury.
        import time as _time

        from ...community.group_economy import _period_key
        group_progress = surface.economy.group_quest_progress(
            key, now=_time.time())
        paid_state = surface._extras(key).get(
            "group_quests_paid") or {}
        lines.append("🏛️ <b>group quests</b> (everyone together):")
        for quest_id, quest in group_progress.items():
            paid = paid_state.get(quest_id) == _period_key(
                quest["period"], _time.time())
            mark = ("✅ paid to treasury" if paid else
                    "🎉 complete — bonus lands soon"
                    if quest["complete"] else
                    f"{quest['have']}/{quest['need']}")
            lines.append(f"<code>{quest_id}</code> — "
                         f"<b>{quest['name']}</b>: "
                         f"{quest['blurb']} · {mark} · "
                         f"treasury bonus {quest['bonus']}")
        buttons.append([("🪙 My balance", "balance")])
        return actionable("\n".join(lines), buttons)

    def _control_treasury(self, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        health = surface.economy_health(key)
        policy = health.get("policy") or {}
        reserved = surface.reserved_treasury(key)
        lines = [
            f"🏦 <b>treasury: {health['treasury']} sparks</b>"
            + (f" ({reserved} reserved for open drops)"
               if reserved else ""),
            f"circulating: {health['circulating']} across "
            f"{health['holders']} holders · biggest holder "
            f"{int(health['top_share'] * 100)}% · "
            f"{health['active_7d']} members active this week.",
            f"🎰 jackpot pot: {health.get('jackpot', 0)} sparks "
            f"(triple 7️⃣ on /slots takes it)",
        ]
        if policy.get("throttled"):
            lines.append(
                "📉 <b>monetary policy: faucets throttled</b> — "
                f"issuance ({policy['earned']}) is running hot "
                f"against sinks ({policy['sunk']}) this week, so "
                "activity pays half until sinks catch up.")
        else:
            lines.append(
                f"📈 monetary policy: normal — {policy.get('earned', 0)} "
                f"issued vs {policy.get('sunk', 0)} sunk this week.")
        proposals = surface.open_proposals(key)
        if proposals:
            lines.append(f"🗳️ {len(proposals)} treasury proposal(s) "
                         "waiting for a second admin (/approve).")
        return "\n".join(lines)

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
            from ...social.chat.tgbot_buttons import actionable
            drop = result["drop"]
            claims = drop["total"] // drop["per_claim"]
            return actionable(
                f"🎁 <b>DROP: {drop['total']} sparks from the "
                f"treasury</b>\n{drop['per_claim']} per claim · "
                f"{claims} claims · tap Claim or send /claim",
                [[(f"🎁 Claim {drop['per_claim']} sparks", "claim"),
                  ("🪙 My balance", "balance")]])
        result = surface.open_drop(key, funder, numbers[0],
                                   numbers[1])
        if not result["ok"]:
            if result["reason"] == "funds":
                return "you don't hold that many sparks to fund a drop."
            return "that drop doesn't work as written."
        from ...social.chat.tgbot_buttons import actionable
        drop = result["drop"]
        claims = drop["total"] // drop["per_claim"]
        # The button is authored from THIS drop's state: its label
        # carries the actual per-claim amount of the pool on
        # screen — never a generic claim button.
        return actionable(
            f"🎁 <b>DROP: {drop['total']} sparks in the pool</b>\n"
            f"{drop['per_claim']} per claim · {claims} claims · "
            f"tap Claim or send /claim",
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
        from ...social.chat.tgbot_buttons import actionable
        return actionable(
            f"🎁 <b>+{result['amount']} sparks claimed!</b> "
            f"(pool left: {result['remaining']} · your balance: "
            f"{result['balance']})",
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
        from ...social.chat.tgbot_buttons import actionable
        return actionable(
            f"🏦 deposited <b>{result['amount']} sparks</b> — "
            f"wallet {result['wallet']} · banked "
            f"<b>{result['bank']}</b> (robbers can't touch it).",
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
        from ...social.chat.tgbot_buttons import actionable
        return actionable(
            f"💵 withdrew <b>{result['amount']} sparks</b> — "
            f"wallet <b>{result['wallet']}</b> · banked "
            f"{result['bank']}.",
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
        from ...social.chat.tgbot_buttons import actionable
        return actionable(
            f"🌅 daily claimed: <b>+{result['amount']} sparks</b> "
            f"(streak {result['streak']} · balance "
            f"{result['balance']}).",
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
        from ...social.chat.tgbot_buttons import actionable
        if result.get("success"):
            return actionable(
                f"🥷 got away with <b>{result['amount']} "
                f"sparks</b> (balance {result['balance']}). "
                "Bank it before someone returns the favour: "
                "/deposit all",
                [[("🏦 Deposit", "deposit all"),
                  ("🪙 My balance", "balance")]])
        return actionable(
            f"🚔 caught. You're in jail for 30 minutes — bail is "
            f"<b>{result['bail']} sparks</b> to the treasury.",
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
        return (f"🔓 bail paid — <b>{result['cost']} sparks</b> to "
                f"the treasury. You're out (balance "
                f"{result['balance']}).")

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
        from ...social.chat.tgbot_buttons import actionable
        buttons = [[(f"🎰 Spin again ({bet})", f"slots {bet}"),
                    ("🪙 My balance", "balance")]]
        if result.get("jackpot"):
            return actionable(
                f"🎰 {reels}\n🎉 <b>JACKPOT — "
                f"{result['jackpot']} sparks!</b> plus "
                f"{result['payout']} from the reels. Balance "
                f"{result['balance']}.", buttons)
        if result["payout"] > 0:
            return actionable(
                f"🎰 {reels}\n<b>+{result['payout']} sparks</b> "
                f"(×{result['multiplier']}). Balance "
                f"{result['balance']}.", buttons)
        return actionable(
            f"🎰 {reels}\nno win — the house share feeds the "
            f"treasury and the jackpot pot "
            f"({surface.economy.jackpot_balance(key)} and "
            "growing).", buttons)

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
        from ...social.chat.tgbot_buttons import actionable
        buttons = [[(f"🪙 Flip again ({bet})", f"coinflip {bet}"),
                    ("🪙 My balance", "balance")]]
        if result["won"]:
            return actionable(
                f"🪙 it landed <b>{result['flip']}</b> — you "
                f"called it: <b>+{result['payout']} sparks</b>. "
                f"Balance {result['balance']}.", buttons)
        return actionable(
            f"🪙 it landed <b>{result['flip']}</b> — not your "
            "call. The house share feeds the treasury and the "
            "jackpot.", buttons)

    def _control_top(self, message):  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        import html as _html

        surface = self._group_surface()
        rows = surface.top_holders(key, limit=10)
        if not rows:
            return "no sparks held yet — /work and /mine start it."
        members = surface.members(key)
        lines = ["🏆 <b>richest in the group</b>"]
        medals = ["🥇", "🥈", "🥉"]
        for i, row in enumerate(rows):
            record = members.get(row["member"]) or {}
            who = _html.escape(str(
                record.get("name") or record.get("username")
                or row["member"]))
            mark = medals[i] if i < 3 else f"{i + 1}."
            bank = (f" (🏦 {row['bank']})" if row["bank"] else "")
            lines.append(f"{mark} {who} — <b>{row['total']}</b>"
                         f"{bank}")
        return "\n".join(lines)

    def _control_tax(self, message) -> str:  # type: ignore[no-untyped-def]
        key, err = self._economy_gate(message)
        if err:
            return err
        surface = self._group_surface()
        s = surface.store.settings(key)
        policy = surface.economy.issuance_state(key)
        state = ("throttled (activity pays half)"
                 if policy["throttled"] else "normal")
        return (
            f"🧾 <b>the group's monetary policy</b>\n"
            f"transfer tax: {s.get('transfer_tax_pct', 2)}% to "
            f"the treasury · moderation fine: "
            f"{s.get('fine_amount', 0)} sparks\n"
            f"earn caps: 200/day · 800/week · governance: "
            f"treasury spends above "
            f"{s.get('governance_threshold', 500)} need a second "
            f"admin\nissuance this week: {policy['earned']} "
            f"issued / {policy['sunk']} sunk — faucets {state}.")

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
        return (f"🏦 granted <b>{amount} sparks</b> from the "
                f"treasury (treasury now {result['treasury']}).")

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
            lines = ["🗳️ <b>open treasury proposals:</b>"]
            for open_pid, proposal in open_now.items():
                payload = proposal.get("payload") or {}
                lines.append(
                    f"<code>{open_pid}</code> — "
                    f"{proposal.get('kind')}: {payload}")
            lines.append("approve with /approve <id> (a different "
                         "admin than the proposer).")
            return "\n".join(lines)
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
                # A one-hour mute; the two Telegram adapters spell
                # it differently (minutes vs until) — speak each
                # one's own grammar.
                try:
                    fn(message.chat, user_id, minutes=60)
                except TypeError:
                    fn(message.chat, user_id,
                       until=int(time.time()) + 3600)
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
    import html as _html

    name = _html.escape(
        str(member.get("name") or member.get("username") or "member"))
    handle = (f"@{_html.escape(str(member['username']))}"
              if member.get("username") else "no username")
    lines = [f"👤 <b>{name}</b> ({handle})"]
    joined = member.get("joined_ts") or member.get("first_seen")
    lines.append(f"joined: {_fmt_ts(joined)} · messages: "
                 f"{member.get('message_count') or 0} · last seen: "
                 f"{_fmt_ts(member.get('last_seen'))}")
    if member.get("left_ts"):
        lines.append(f"left: {_fmt_ts(member['left_ts'])}")
    changes = [c for c in surface.changes(key, limit=200)
               if c.get("user_id") == member.get("user_id")]
    for change in changes[-3:]:
        lines.append(
            f"{change.get('kind', 'changed')}: "
            f"{_html.escape(str(change.get('old') or '—'))} → "
            f"{_html.escape(str(change.get('new') or '—'))} "
            f"({_fmt_ts(change.get('ts'))})")
    return "\n".join(lines)


def _audit_line(entry: dict) -> str:
    kind = entry.get("kind", "?")
    bits = [_fmt_ts(entry.get("ts")), kind]
    for field_name in ("action", "rule", "user_id", "change_kind",
                       "reason"):
        if entry.get(field_name):
            bits.append(f"{field_name}={entry[field_name]}")
    if entry.get("old") or entry.get("new"):
        bits.append(f"{entry.get('old') or '—'} → "
                    f"{entry.get('new') or '—'}")
    return " · ".join(str(b) for b in bits)
