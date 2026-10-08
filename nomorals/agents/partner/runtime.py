"""PartnerRuntime: the loop that keeps the partner alive."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable
from ...core.logging_setup import get_logger
from ...partner.gating import gate_decision, is_owner_chat, is_restricted
from ...partner.presence import Presence, decide_presence, human_typing_seconds
from ...social.chat.base import ChatKind, ChatMessage, ChatRef
from ...social.chat.gateway import ChatGateway
from .brain import PartnerBrain
from .approvals import _direct_approve, _direct_deny, _key_set
from .runtime_live import RuntimeLiveMixin
from .runtime_meta import RuntimeMetaMixin
from .runtime_mission import RuntimeMissionMixin
from .runtime_build import RuntimeBuildMixin
from .runtime_schedule import RuntimeScheduleMixin
from .runtime_voice import RuntimeVoiceMixin
from .runtime_memory import RuntimeMemoryMixin
from .runtime_games import RuntimeGamesMixin
from .runtime_system import RuntimeSystemMixin
from .runtime_media import RuntimeMediaMixin
from .runtime_intel import RuntimeIntelMixin
from .runtime_search import RuntimeSearchMixin
from .runtime_wisdom import RuntimeWisdomMixin
_log = get_logger(__name__)

#: telegram-bot is the public endpoint: conversation + games only. A known
#: non-game slash command there gets this redirect instead of running —
#: owner/system controls live on the userbot (``telegram`` endpoint).
_PUBLIC_BOT_NOT_AVAILABLE = (
    "that's not something I can do here — this is my public chat, "
    "so I keep it to conversation and games. "
    "try /game list to see what's on, or just talk to me."
)


class PartnerRuntime(
    RuntimeLiveMixin,
    RuntimeMetaMixin,
    RuntimeMissionMixin,
    RuntimeBuildMixin,
    RuntimeScheduleMixin,
    RuntimeVoiceMixin,
    RuntimeMemoryMixin,
    RuntimeGamesMixin,
    RuntimeSystemMixin,
    RuntimeMediaMixin,
    RuntimeIntelMixin,
    RuntimeSearchMixin,
    RuntimeWisdomMixin,
):
    """Runs the partner across all configured platforms, at the same time."""

    def __init__(
        self,
        context: Any,
        *,
        gateway: ChatGateway | None = None,
        brain: PartnerBrain | None = None,
        dry_run: bool = False,
        session_bridge: Any = None,
    ) -> None:
        """Create the runtime.

        ``session_bridge`` is an optional ``os.SessionBridge`` (injected —
        this module is L5 and must not import ``os``/L6). When present it is
        passed to the ChatGateway so every inbound message is attached to
        its OS Session before the brain runs.
        """
        self.context = context
        self.settings = context.settings
        self.brain = brain or PartnerBrain(context)
        #: Monotonic boot time — drives the console dashboard uptime.
        import time as _time

        self._boot_mono = _time.monotonic()
        # wave 87: the Core Mind — the always-on layer that routes a
        # natural-language goal to the right organ (owner DMs only;
        # commands remain the manual override everywhere).
        from ..coremind import CoreMind

        self.mind = CoreMind(context, runtime=self)
        self.gateway = gateway
        self.dry_run = dry_run
        #: Optional os.SessionBridge (L6), injected to keep layering clean.
        #: Also published on context.extras so CLI/tools can reach it.
        self._session_bridge = session_bridge
        if session_bridge is not None:
            try:
                context.extras["session_bridge"] = session_bridge
            except Exception:  # noqa: BLE001 - extras is best-effort
                pass
        partner_cfg = self.settings.partner

        # Profile-aware: the chat pool follows the runtime tune (phone →
        # 2 parallel chats, desktop → 4). An explicit partner.max_parallel_chats
        # in settings already won inside the tune.
        tune = context.extras.get("tune")
        max_chats = partner_cfg.max_parallel_chats if tune is None else tune.max_parallel_chats
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, int(max_chats)),
            thread_name_prefix="partner-chat",
        )
        self._queues: dict[str, deque[ChatMessage]] = {}
        self._draining: dict[str, bool] = {}
        self._queue_guard = threading.Lock()
        self._stopped = threading.Event()
        #: In-flight chat-pool futures (per-chat pumps + discord greeters).
        #: Drained (bounded) in stop() so the context's Database is never
        #: closed while a worker is mid-write — the "database is closed"
        #: shutdown race seen on the phone gateway.
        self._inflight: set[Future] = set()
        self._inflight_guard = threading.Lock()
        self._autonomy: Any = None
        self.stats = {"messages": 0, "replies": 0, "errors": 0, "controls": 0}
        #: Guards ``stats``: the counters are bumped from every chat-pool
        #: worker, the delayed-reply threads, and the control path. ``+=``
        #: on a dict value is a read-modify-write and lost increments
        #: otherwise (silent undercounting in /status).
        self._stats_lock = threading.Lock()
        self._owner_chats = _key_set(partner_cfg.owner_chats)
        self._book_busy: set[str] = set()  # slugs with a book pipeline running
        self._apply_persisted_mode()

        if self.gateway is None:
            from ...social.chat import build_adapters

            adapters, skipped = build_adapters(self.settings, on_new_member=self._on_new_discord_member)
            if skipped:
                _log.warning("skipped chat adapters: %s", skipped)
            if not adapters:
                from ...social.chat.local import LocalAdapter

                adapters = {"local": LocalAdapter(mood_label_provider=lambda: self.brain.mood.current().label)}
            owner_chats = _key_set(partner_cfg.owner_chats)
            us_chats = _key_set(partner_cfg.us_chats)
            self.gateway = ChatGateway(
                adapters,
                db=context.db,
                dry_run=dry_run,
                owner_chats=owner_chats,
                us_chats=us_chats,
                adapter_builder=self._build_adapter,
                session_bridge=self._session_bridge,
            )

        # Tools that deliver files into chats (file_send, report_publish)
        # reach the live gateway through the context, not a second wire.
        context.extras["gateway"] = self.gateway

        # Console-only commands (dashboard, status, jobs, …) for the local
        # terminal adapter. Wired here — not just in run_chat_bot.py — so
        # EVERY entrypoint gets a working `dashboard --watch`, not only the
        # one script that remembered to set the hook.
        self._wire_console_commands()

        # Power mode persists across restarts: re-widen on boot if unlocked.
        # Runs AFTER the gateway exists — calling it earlier hit
        # `'NoneType' object has no attribute 'set_rate_limit'` on boots
        # where no gateway was injected.
        self._adopt_power_mode()

    # ── Console commands: `dashboard`, `dashboard --watch`, `status`… ─────
    def _wire_console_commands(self) -> None:
        """Attach the console command handler to the local adapter.

        This is what makes `dashboard --watch` enter the live GodScreen.
        It lives on the runtime (not just run_chat_bot.py) so every
        entrypoint gets it. Best-effort: never breaks boot.
        """
        try:
            from ...console import ConsoleCommands

            local = None
            try:
                local = self.gateway.adapters.get("local")
            except Exception:  # noqa: BLE001 - gateway internals are best-effort
                local = None
            if local is None:
                return
            # Don't clobber a hook an entrypoint set deliberately (e.g.
            # run_chat_bot.py wires its own richer snapshot with history).
            if getattr(local, "command_hook", None) is not None:
                return
            local.command_hook = ConsoleCommands(self.console_snapshot).handle
        except Exception:  # noqa: BLE001 - console commands are optional
            _log.debug("console commands unavailable", exc_info=True)

    def console_snapshot(self) -> dict:
        """Live-status snapshot for the console dashboard / GodScreen.

        Same shape as scripts/run_chat_bot.py's provider (all keys
        optional — the renderers degrade gracefully).
        """
        import os
        import time as _time

        snap: dict = {}
        snap["uptime_s"] = _time.monotonic() - getattr(self, "_boot_mono", _time.monotonic())
        # ── adapters ──
        adapters: dict = {}
        try:
            status = self.gateway.status()
            for name, info in status.items():
                if str(name).startswith("_"):
                    continue
                adapters[str(name)] = {
                    "running": bool(info.get("running_in_session", True)),
                    "received": info.get("received", "—"),
                    "sent": info.get("sent", "—"),
                }
        except Exception:  # noqa: BLE001 - dashboard is best-effort
            pass
        snap["adapters"] = adapters
        # ── traffic ──
        try:
            with self._stats_lock:
                snap["traffic"] = dict(self.stats)
        except Exception:  # noqa: BLE001
            snap["traffic"] = dict(getattr(self, "stats", {}))
        # ── scheduler ──
        try:
            sched = getattr(self, "_scheduler", None)
            jobs = []
            running = False
            if sched is not None:
                running = bool(sched.running())
                for job in sched.list_jobs():
                    jobs.append(
                        {
                            "name": job.get("name") or job.get("id"),
                            "spec": job.get("spec") or job.get("schedule") or "",
                            "enabled": job.get("enabled", True),
                            "next_run": job.get("next_run"),
                        }
                    )
            snap["scheduler"] = {"running": running, "jobs": jobs}
        except Exception:  # noqa: BLE001
            snap["scheduler"] = {}
        # ── games ──
        try:
            row = self.context.db.query_one("SELECT COUNT(*) AS n FROM game_players")
            players = (row or {}).get("n", "—")
        except Exception:  # noqa: BLE001
            players = "—"
        snap["games"] = {"players": players}
        # ── brain (LLM router health) ──
        try:
            router = getattr(getattr(self, "_brain", None), "router", None) or getattr(
                self, "_router", None
            )
            if router is not None and hasattr(router, "stats_snapshot"):
                snap["llm"] = router.stats_snapshot()
            else:
                snap["llm"] = {}
        except Exception:  # noqa: BLE001
            snap["llm"] = {}
        # ── theme ──
        snap["theme"] = os.environ.get("NM_CONSOLE_THEME", "ocean")
        # ── extras ──
        extras: dict = {}
        try:
            extras["autonomy"] = "on" if getattr(self, "_autonomy", None) else "off"
        except Exception:  # noqa: BLE001
            pass
        try:
            extras["arena"] = "on" if getattr(self, "_arena", None) else "off"
        except Exception:  # noqa: BLE001
            pass
        snap["extras"] = extras
        return snap

    def _tuned_autonomy_caps(self) -> tuple[int, int]:
        """Daily proactive-volume caps, scaled by the profile's mission
        aggressiveness. A phone (0.5) gets half the background volume of a
        workstation (1.0). Explicit 0 (unlimited) and non-default configured
        caps are respected as-is — the tune only scales the defaults.
        """
        import math

        partner_cfg = self.settings.partner
        tune = self.context.extras.get("tune")
        scale = float(getattr(tune, "mission_autonomy", 1.0) or 1.0)
        out: list[int] = []
        for value, default in (
            (partner_cfg.max_proactive_dm_per_day, 6),
            (partner_cfg.max_group_posts_per_day, 2),
        ):
            if value == 0 or value != default or scale >= 1.0:
                out.append(int(value))
            else:
                out.append(max(1, math.ceil(value * scale)))
        return out[0], out[1]

    # ── Discord: say hi first when someone new joins a server ──────────────
    def _on_new_discord_member(self, guild_name: str, member: dict) -> None:
        """Adapter hook (already deduped per person): decide on a first DM.

        Runs on the chat pool — a brain call must not block the adapter.
        """
        if self._stopped.is_set() or self.dry_run:
            return
        try:
            self._track_inflight(
                self._pool.submit(self._greet_discord_newcomer, guild_name, member)
            )
        except RuntimeError:
            _log.debug("discord greeter dropped: pool shut down")

    def _greet_discord_newcomer(self, guild_name: str, member: dict) -> None:
        from ...social.chat.base import ChatMessage, ChatRef

        adapter = self.gateway.adapters.get("discord")
        if adapter is None:
            return
        try:
            chat: ChatRef | None = adapter.start_dm(str(member.get("id", "")))
        except Exception as exc:  # noqa: BLE001
            _log.debug("discord newcomer DM open failed: %s", exc)
            return
        if chat is None:
            return  # DMs closed or unresolved — nothing to say first
        prompt = (
            f"[a new person just joined your Discord server {guild_name!r}: "
            f"{member.get('name', 'someone')} ({member.get('tag', '')}). "
            "You could say a short, friendly hi so they know who you are — "
            "but only if it feels right; staying quiet is fine too.]"
        )
        message = ChatMessage(chat=chat, incoming=True, text=prompt,
                              sender=str(member.get("name", "")), ts=time.time())
        try:
            outcome = self.brain.handle_message(message)
        except Exception as exc:  # noqa: BLE001
            _log.exception("discord newcomer greeting failed: %s", exc)
            return
        if outcome.parts:
            self._bump("replies")
            self._send_reply(message, outcome.parts)

    # ── inbound fan-in: per-chat FIFO, cross-chat parallel ──────────────────
    def _bump(self, key: str, amount: int = 1) -> None:
        """Thread-safe stats increment (see ``_stats_lock``)."""
        with self._stats_lock:
            self.stats[key] = self.stats.get(key, 0) + amount

    def _stats_snapshot(self) -> dict[str, int]:
        """A consistent copy of the counters for status/log lines."""
        with self._stats_lock:
            return dict(self.stats)

    def _release_db_thread(self) -> None:
        """Drop the calling thread's DB connection (see
        ``Database.release_thread``): one-shot workers (typing keepalive,
        delayed replies, book builds) each open a per-thread sqlite
        connection, and without this every such thread leaks one fd until
        process exit. Long-lived pool threads never call this."""
        from ...storage.db import release_thread_connection

        release_thread_connection(getattr(self.context, "db", None))

    def on_message(self, message: ChatMessage) -> None:
        key = message.chat.key
        with self._queue_guard:
            queue = self._queues.setdefault(key, deque())
            queue.append(message)
            draining = self._draining.get(key, False)
            self._draining[key] = True
        if draining:
            return  # a pump is already working this chat; it will pick the message up
        try:
            self._track_inflight(self._pool.submit(self._pump, key))
        except RuntimeError:
            # The pool is shut down (runtime stopping): reset the flag or
            # the queued message is orphaned — no pump will ever pick it up.
            _log.warning("on_message: pool shut down, dropping inbound for %s", key)
            with self._queue_guard:
                self._draining[key] = False

    def _track_inflight(self, fut: Future) -> None:
        """Remember a chat-pool future so stop() can drain it."""
        with self._inflight_guard:
            self._inflight.add(fut)
        fut.add_done_callback(self._untrack_inflight)

    def _untrack_inflight(self, fut: Future) -> None:
        with self._inflight_guard:
            self._inflight.discard(fut)

    def _drain_chat_pool(self, timeout: float = 30.0) -> None:
        """Bounded wait for in-flight chat handlers before teardown.

        stop() runs while the AgentContext (and its Database) is still
        alive; the context closes right after. A pump mid-reply holds no
        lock the closer needs, but it *writes* — closing the DB first
        turns its persist tail into StorageError("database is closed").
        The timeout covers one interactive reply budget (25s); stragglers
        past it are left to the brain's shutdown-tolerant persist tail.
        """
        with self._inflight_guard:
            pending = [f for f in self._inflight if not f.done()]
        if not pending:
            return
        _log.info("draining %d in-flight chat handler(s) (%.0fs max)",
                  len(pending), timeout)
        _done, not_done = wait(pending, timeout=timeout)
        if not_done:
            _log.warning("%d chat handler(s) still running after %.0fs drain; "
                         "continuing shutdown", len(not_done), timeout)

    def _pump(self, key: str) -> None:
        try:
            while not self._stopped.is_set():
                with self._queue_guard:
                    queue = self._queues.get(key)
                    message = queue.popleft() if queue else None
                    if message is None:
                        # Queue drained: drop the entries. A long-lived
                        # runtime otherwise keeps one dict slot per chat it
                        # ever saw, forever. Safe under the guard: a racing
                        # on_message setdefaults a fresh queue and (seeing
                        # no _draining entry) submits a new pump.
                        self._queues.pop(key, None)
                        self._draining.pop(key, None)
                        break
                self._process(message)
        except Exception as exc:  # noqa: BLE001 - a bad chat must not kill the pool
            _log.exception("pump for %s crashed: %s", key, exc)
            with self._queue_guard:
                self._draining[key] = False

    def _process(self, message: ChatMessage) -> None:
        self._bump("messages")
        # Vision flag: with it off, inbound media is dropped before the brain
        # (no download-to-understanding pipeline, no token cost).
        if message.media:
            from ..features import feature_enabled

            if not feature_enabled(self.context, "vision"):
                import dataclasses

                message = dataclasses.replace(message, media=[])
        # Prompt 09 attachment glue: with vision on, the current message's
        # image attachments are addressable as attachment:<n> by the vision
        # tool. The chat media dirs sit outside the workspace sandbox — this
        # is their explicit intake allowance. (Shared context: concurrent
        # chats can clobber the list; the window is one brain turn.)
        if message.media:
            self.context.extras["attachments"] = [
                {"path": m.path, "name": m.name or f"image-{i}", "mime": m.mime}
                for i, m in enumerate(message.media)
                if m.kind == "image"
            ]
        else:
            self.context.extras.pop("attachments", None)
        # Voice notes use the typed-intent path: if the owner's voice note
        # transcribes to a slash command, it becomes the message text so the
        # whole downstream pipeline (command parsing, routing, gating) treats
        # it exactly like typed text. Anything else stays conversation
        # context via _media_notes as before.
        if message.incoming and not message.text.strip().startswith("/"):
            spoken = self.brain._spoken_command(message)
            if spoken:
                import dataclasses

                message = dataclasses.replace(message, text=spoken)
                _log.info("voice note in %s transcribed to command %r",
                          message.chat.key, spoken[:40])
        # Voice-note ping-pong (inbound): a voice note that isn't a slash
        # command still becomes spoken text, so the brain hears it.  Gated
        # on the "voice" feature; transcription is best-effort.
        if message.incoming and not message.text.strip():
            from ..features import feature_enabled as _fe
            if _fe(self.context, "voice"):
                from ...voice.pingpong import (
                    has_voice_media, transcribe_voice_media)
                if has_voice_media(message):
                    heard = transcribe_voice_media(self.context, message)
                    if heard:
                        import dataclasses

                        message = dataclasses.replace(message, text=heard)
                        message.meta["voice_in"] = True
                        _log.info("voice note in %s heard: %r",
                                  message.chat.key, heard[:60])
        # Trigger engine hook (nomorals/triggers): message-source triggers
        # evaluate the final inbound text here.  One call, no fork of the
        # dispatch path below; a no-op when no engine is attached, and a
        # trigger failure here never breaks message intake.
        if message.incoming:
            try:
                from ...triggers.engine import message_hook
                message_hook(self.context, message.text, message.chat.key,
                             platform=message.chat.platform,
                             sender=getattr(message, "sender", ""))
            except Exception:  # noqa: BLE001 - triggers must never break intake
                _log.exception("trigger message hook failed")
        # Games: while a game is live in this chat, the room owns the
        # conversation — plain messages are moves and the in-game commands
        # (/status /pass /shop /leave …) work for every participant.
        # /game itself stays the control plane (the router lets it through).
        if message.incoming:
            move_reply = self._route_game_move(
                message.chat.key, message.text,
                player=self._game_player(message), kind=message.chat.kind)
            if move_reply is not None:
                try:
                    self.gateway.send(message.chat.platform, message.chat, move_reply)
                except Exception:  # noqa: BLE001
                    _log.exception("game reply send failed")
                return
        # /game AND the direct game commands (/hangman, /mafia, …) work in
        # EVERY chat, for EVERY participant — games are social by nature, so
        # a member of any group can start and play one. These are also the
        # ONLY game start triggers outside the owner's DMs: natural language
        # never launches a game in a non-owner chat (wave 87). All other
        # slash commands stay owner-only: for everyone else a slash falls
        # through to her as text.
        if message.incoming and message.text.strip().startswith("/"):
            from ...social.chat.control import GAME_COMMANDS, parse_control

            command = parse_control(message.text)
            _log.debug("game command check: text=%r, command=%s", 
                      message.text[:40], command)
            if (command is not None
                    and (command.kind == "game" or command.kind in GAME_COMMANDS)):
                self._bump("controls")
                _log.info("game command %r from %s in %s (any-chat dispatch)",
                          message.text[:40], message.sender or "?", message.chat.key)
                try:
                    if message.chat.kind == ChatKind.CHANNEL:
                        # Telegram doesn't identify individual senders in
                        # channels (the sender IS the channel), so per-player
                        # game state can't work there. Friendly redirect
                        # instead of minting a channel-named phantom profile.
                        reply = ("Games work in groups and DMs, not channels — "
                                 "Telegram doesn't tell me who's sending in a channel.")
                    elif command.kind in ("inventory", "equip", "unequip",
                                        "repair"):
                        reply = self._control_gear(
                            command.kind, command.tail or command.arg,
                            player=self._game_player(message))
                    elif command.kind == "level":
                        reply = self._control_level(
                            player=self._game_player(message))
                    elif command.kind == "skill":
                        reply = self._control_skills(
                            command.tail or command.arg,
                            player=self._game_player(message))
                    elif command.kind == "title":
                        reply = self._control_titles(
                            command.tail or command.arg,
                            player=self._game_player(message))
                    elif command.kind == "stats":
                        reply = self._control_stats(
                            command.tail or command.arg,
                            player=self._game_player(message))
                    elif command.kind == "daily":
                        reply = self._control_daily(
                            player=self._game_player(message))
                    elif command.kind == "mastery":
                        reply = self._control_mastery(
                            command.tail or command.arg,
                            player=self._game_player(message))
                    elif command.kind == "gift":
                        reply = self._control_gift(
                            command.tail or command.arg,
                            player=self._game_player(message))
                    else:
                        verb = (command.tail or command.arg) \
                            if command.kind == "game" \
                            else command.kind + ((" " + command.tail)
                                                 if command.tail else "")
                        reply = self._control_game(
                            verb, chat_key=message.chat.key,
                            player=self._game_player(message),
                            kind=message.chat.kind)
                    _log.debug("game command reply: %r", reply[:100] if reply else "")
                except Exception as exc:  # noqa: BLE001
                    _log.exception("game command failed: %s", exc)
                    reply = (f"that move broke something on my end: {exc} — "
                             f"try /help {command.kind} for the usage, "
                             f"or /game quit and start it fresh")
                if reply:
                    try:
                        self._typing_for(message.chat, reply)
                        result = self.gateway.send(message.chat.platform, message.chat, reply)
                        if not result.ok:
                            _log.warning("game reply send failed in %s: %s",
                                         message.chat.key, result.error)
                        else:
                            _log.info("game reply sent to %s", message.chat.key)
                    except Exception as exc:  # noqa: BLE001
                        _log.warning("game reply send failed: %s", exc)
                else:
                    _log.warning("game command produced empty reply in %s", message.chat.key)
                return
            # telegram-bot is the public endpoint: conversation + games
            # only. A known non-game slash here gets a friendly redirect
            # instead of running — owner/system controls live on the
            # userbot. Unknown slashes keep falling through to her as
            # ordinary text, as before.
            if message.chat.platform == "telegram-bot" and command is not None:
                self._bump("controls")
                _log.info("public-bot gate: blocked /%s from %s in %s",
                          command.kind, message.sender or "?",
                          message.chat.key)
                try:
                    self.gateway.send(message.chat.platform, message.chat,
                                      _PUBLIC_BOT_NOT_AVAILABLE)
                except Exception:  # noqa: BLE001
                    _log.warning("public-bot gate reply failed")
                return
        # WisdomKeeper practice sessions own their chats while live: plain
        # words (pause/resume/stop) steer the run and the reply after a
        # session closes is journaled. Only active sessions and pending
        # journal prompts intercept — everything else falls through to
        # normal flow untouched.
        if message.incoming:
            wisdom_reply = self._wisdom_incoming(message.chat.key, message.text)
            if wisdom_reply is not None:
                try:
                    self.gateway.send(message.chat.platform, message.chat,
                                      wisdom_reply)
                except Exception:  # noqa: BLE001
                    _log.exception("wisdom reply send failed")
                return
        # Explicit-override (user directive 2026-10-07): the owner saying
        # "send it" right after a draft preview sends that draft with no
        # re-confirmation. Owner chats only; never fires without an armed
        # draft, and the draft preview already showed the exact payload.
        if message.incoming and self._is_operator(message):
            try:
                from ..email_triage import check_explicit_send

                send_reply = check_explicit_send(message.text,
                                                 context=self.context)
            except Exception:  # noqa: BLE001 - never eat the chat
                send_reply = None
            if send_reply is not None:
                try:
                    self.gateway.send(message.chat.platform, message.chat,
                                      send_reply)
                except Exception:  # noqa: BLE001
                    _log.warning("explicit-send reply failed")
                return
        # Control commands: from the console or the owner chat, a *known*
        # slash command is a command, not conversation. Unknown slashes and
        # stray / from non-operators fall through to her as ordinary text.
        if message.text.strip().startswith("/") and self._is_operator(message):
            from ...social.chat.control import parse_control

            if parse_control(message.text) is not None:
                self._bump("controls")
                # Feed the arena's interest profiler: what the owner runs
                # most shapes future topic picks. Never raises.
                try:
                    from ..arena import activity as _arena_activity

                    verb = message.text.strip().split()[0].lstrip("/").lower()
                    _arena_activity.record(self.context.db, "command", verb)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    reply = self.handle_control(message.text, message.chat.key,
                                                message=message)
                except Exception as exc:  # noqa: BLE001
                    _log.exception("control command failed: %s", exc)
                    reply = (f"/{command.kind} blew up on my end: {exc} — "
                             f"/help {command.kind} shows the right usage")
                if reply:
                    try:
                        self._typing_for(message.chat, reply)
                        # Septorch-style outputs carry HTML tags — render
                        # them formatted, everything else stays plain text.
                        _pm = "HTML" if ("<b>" in reply or "<code>" in reply) else ""
                        self.gateway.send(message.chat.platform, message.chat,
                                          reply, parse_mode=_pm)
                    except Exception as exc:  # noqa: BLE001
                        _log.warning("control reply send failed: %s", exc)
                return
        # wave 87: the Core Mind. A natural-language goal in the owner's DM
        # routes to the right organ (research, builder, browser, downloader,
        # missions, games). Structurally owner-DM-only: in every other chat
        # this returns None and the message falls through to conversation.
        if message.incoming and not message.text.strip().startswith("/"):
            try:
                mind_reply = self.mind.handle(
                    message.text, message=message, chat_key=message.chat.key)
            except Exception:  # noqa: BLE001 - the mind must never eat the chat
                _log.exception("core mind failed on %s", message.chat.key)
                mind_reply = None
            if mind_reply is not None:
                self._bump("controls")
                _log.info("core mind routed %r in %s",
                          message.text[:40], message.chat.key)
                # The fast path skips the brain (zero model calls) — but the
                # turn still counts: persist it and run the downstream hooks
                # (curator, training pairs) so history never silently gaps.
                try:
                    self.brain.note_fast_turn(message, mind_reply)
                except Exception as exc:  # noqa: BLE001 - never eat the chat
                    _log.warning("fast turn hook failed: %s", exc)
                try:
                    self._typing_for(message.chat, mind_reply)
                    self.gateway.send(message.chat.platform, message.chat,
                                      mind_reply)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("mind reply send failed: %s", exc)
                return
        # While the brain works, the indicator stays up (own thread — the
        # model call must not be delayed by a blocking typing window).
        # In groups she only speaks when addressed — decide BEFORE showing
        # any typing indicator, so a message she'll stay quiet on never
        # flashes "typing..." with no reply after it.
        keepalive_stop = threading.Event()
        _will_answer = (
            message.chat.kind != ChatKind.GROUP
            or self.brain.should_speak_in_group(message)
        )
        if self.settings.partner.typing_while_thinking and not self.dry_run and _will_answer:
            threading.Thread(
                target=self._typing_keepalive,
                args=(message.chat, keepalive_stop),
                name=f"typing-keepalive-{message.chat.key}",
                daemon=True,
            ).start()
        try:
            outcome = self.brain.handle_message(message)
        except Exception as exc:  # noqa: BLE001
            self._bump("errors")
            _log.exception("brain failed on %s: %s", message.chat.key, exc)
            keepalive_stop.set()
            return
        finally:
            keepalive_stop.set()
        if not outcome.parts:
            if outcome.presence.delay_seconds > 0:
                self._schedule_delayed(message, outcome.presence)
            return
        self._bump("replies")
        self._send_reply(message, outcome.parts)

    def _schedule_delayed(self, message: ChatMessage, presence: Presence) -> None:
        """She's busy: the reply lands later, on its own daemon thread.

        A dedicated thread (not the chat pool) — a half-hour gap must not
        occupy one of the ``max_parallel_chats`` workers. The brain's state
        already heard the message; ``deliver_reply`` only generates + sends.
        """
        delay = presence.delay_seconds

        def _job() -> None:
            try:
                if self._stopped.is_set():
                    return
                time.sleep(delay)
                if self._stopped.is_set():
                    return
                try:
                    parts = self.brain.deliver_reply(message)
                    if parts:
                        self._bump("replies")
                        self._send_reply(message, parts)
                except Exception:  # noqa: BLE001 - a late reply must not crash the thread
                    _log.exception("delayed reply for %s failed", message.chat.key)
            finally:
                self._release_db_thread()

        threading.Thread(target=_job, name=f"delayed-reply-{message.chat.key}", daemon=True).start()

    def _send_voice_reply(self, message: ChatMessage,
                            parts: list[str]) -> bool:
        """Reply with voice note(s).  True when at least one went out.

        Shows the "recording voice" indicator (not "typing") for the
        actual length of the note, and sends it as a voice bubble.
        """
        from ...voice.pingpong import synthesize_voice_reply, ogg_duration
        from ...social.chat.base import MediaRef
        sent_any = False
        for part in parts:
            text = (part or "").strip()
            if not text:
                continue
            try:
                ogg_path = synthesize_voice_reply(self.context, text)
                # recording indicator matches the note's real length
                try:
                    secs = max(1.0, min(60.0, ogg_duration(ogg_path)))
                    self.gateway.typing(message.chat.platform, message.chat,
                                        seconds=secs, action="recording")
                except Exception:  # noqa: BLE001 - cosmetic
                    pass
                media = MediaRef(path=ogg_path, kind="audio",
                                 name="reply.ogg", mime="audio/ogg",
                                 voice_note=True)
                adapter = self.gateway._adapter_for(message.chat.platform)
                if adapter is None or self.gateway.dry_run:
                    return sent_any
                result = adapter.send_media(message.chat, media)
                if result.ok:
                    sent_any = True
                    _log.info("voice reply sent in %s", message.chat.key)
                else:
                    _log.warning("voice reply send failed: %s", result.error)
            except Exception as exc:  # noqa: BLE001 - fall back to text
                _log.warning("voice reply failed, falling back to text: %s",
                             exc)
                return sent_any
        return sent_any

    def _send_reply(self, message: ChatMessage, parts: list[str]) -> None:
        # Voice-note ping-pong (outbound): the user spoke, so Devon answers
        # in voice.  Best-effort — any TTS/ffmpeg failure falls back to the
        # normal text path below.
        if message.meta.get("voice_in"):
            from ..features import feature_enabled as _fe
            if _fe(self.context, "voice") and self._send_voice_reply(message, parts):
                return
        partner_cfg = self.settings.partner
        values = self.brain.mood.current().values
        # Typing indicator on every kind of chat, per part: each chunk of a
        # split reply gets its own realistic, length-scaled typing run.
        typing_on = (
            message.chat.kind != ChatKind.GROUP
            or partner_cfg.typing_in_groups
        )
        _log.info("_send_reply: %d part(s), typing_on=%s, chat=%s", 
                  len(parts), typing_on, message.chat.key)
        try:
            for i, part in enumerate(parts):
                if i:
                    # Between parts: a breath, a re-think, thumbs back on the
                    # keys. The configured delay is a base, not a metronome.
                    delay = partner_cfg.part_delay_seconds * self.brain.presence_rng.uniform(0.6, 2.4)
                    _log.debug("_send_reply: sleeping %.1fs between parts", delay)
                    time.sleep(delay)
                if typing_on:
                    # Human typing pace: the indicator stays up as long as it
                    # takes to actually type THIS part — length- and
                    # mood-dependent, jittered (presence.human_typing_seconds).
                    typing_seconds = human_typing_seconds(
                        part,
                        mood=values,
                        rng=self.brain.presence_rng,
                        minimum=partner_cfg.typing_seconds,
                        cap=partner_cfg.typing_cap_seconds,
                    )
                    _log.info("_send_reply: calling typing for part %d/%d, seconds=%.1f, text_len=%d",
                              i+1, len(parts), typing_seconds, len(part))
                    typing_result = self.gateway.typing(
                        message.chat.platform, message.chat,
                        seconds=typing_seconds,
                    )
                    _log.info("_send_reply: typing returned %s", typing_result)
                result = self.gateway.send(
                    message.chat.platform, message.chat, part,
                    reply_to=message.reply_to if message.chat.kind == ChatKind.GROUP else "",
                )
                _log.info("_send_reply: send part %d/%d, ok=%s", i+1, len(parts), result.ok)
                if not result.ok:
                    _log.warning("send failed on %s: %s", message.chat.platform, result.error)
        except Exception as exc:  # noqa: BLE001
            _log.exception("send loop failed: %s", exc)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> list[str]:
        started = self.gateway.start(self.on_message)
        if self.settings.partner.enabled and not self.dry_run:
            try:
                from ..autonomy import AutonomyAgent

                dm_cap, group_cap = self._tuned_autonomy_caps()
                self._autonomy = AutonomyAgent(
                    self.context, self.brain, self.gateway,
                    mode=self.settings.partner.autonomy_mode,
                    owner_chats=_key_set(self.settings.partner.owner_chats),
                    group_chats=_key_set(self.settings.partner.group_chats),
                    quiet_start=self.settings.partner.quiet_start,
                    quiet_end=self.settings.partner.quiet_end,
                    max_dm_per_day=dm_cap,
                    max_group_per_day=group_cap,
                )
                self._autonomy.start()
            except Exception as exc:  # noqa: BLE001 - autonomy is optional
                _log.warning("autonomy agent failed to start: %s", exc)
                self._autonomy = None
        # Arena: the autonomous research/build loop. Off unless the owner
        # opted in with NM_ARENA_ENABLED=1; even then each cycle re-checks the
        # ``arena`` feature flag and power mode before doing anything.
        self._arena: Any = None
        if getattr(self.settings, "arena", None) is not None \
                and self.settings.arena.enabled and not self.dry_run:
            try:
                from ..arena import Arena

                self._arena = Arena(self.context)
                self._arena.start_loop(notify=self._arena_notify)
            except Exception as exc:  # noqa: BLE001 - arena is optional
                _log.warning("arena failed to start: %s", exc)
                self._arena = None
        # Research: always-on research & suggestions (opt-in via the real
        # `research` feature flag — /features or `nm features`; the
        # per-cycle feature flag + daily cap still apply inside the agent).
        # NOTE: there is no `settings.research` section — the flag lives in
        # kv_store via FeatureRegistry (nomorals/agents/features.py).
        self._research: Any = None
        from ..features import feature_enabled as _research_flag_on
        if _research_flag_on(self.context, "research") and not self.dry_run:
            try:
                from ..notifier import Notifier
                from ..researcher import ResearchAgent

                self._research = ResearchAgent(
                    self.context, notifier=Notifier(self.context, self.gateway)
                )
                self._research.start_loop()
            except Exception as exc:  # noqa: BLE001 - research is optional
                _log.warning("research agent failed to start: %s", exc)
                self._research = None
        # Scheduler: durable at/every/daily jobs (message/tool/command payloads),
        # outcomes published through the notifier.
        self._scheduler: Any = None
        sched_settings = getattr(self.settings, "scheduler", None)
        # The job table is durable regardless of the tick-loop switch: we
        # build the scheduler whenever we're live, but only spin its thread
        # when the section is enabled — booting from another process must
        # still see (and idempotently extend) the registered jobs.
        if sched_settings is not None and not self.dry_run:
            try:
                from ..scheduler import Scheduler

                self._scheduler = Scheduler(
                    self.context,
                    gateway=self.gateway,
                    tick_seconds=getattr(sched_settings, "tick_seconds", 60.0),
                    max_concurrent=getattr(sched_settings, "max_concurrent", 2),
                    wall_seconds=getattr(sched_settings, "wall_seconds", 300.0),
                )
                if sched_settings.enabled:
                    self._scheduler.start()
                # expose to agents: the cognitive loop's cost-aware cadence
                # (wave 63) reschedules its own job through this handle
                self.context.extras["scheduler"] = self._scheduler
                # Closed-loop self-improvement tick: when the owner opts in
                # (improvement.auto_tick) and the mode is on, run a cycle on a
                # schedule. Idempotent — one durable job by name.
                imp = getattr(self.settings, "improvement", None)
                if imp is not None and imp.auto_tick and imp.mode != "off":
                    try:
                        have = [j for j in self._scheduler.list_jobs()
                                if j.get("name") == "improvement tick"]
                        if not have:
                            self._scheduler.add(
                                "improvement tick", "every 6h", "tool",
                                {"tool": "improve", "args": {"action": "cycle"}},
                            )
                    except Exception as exc:  # noqa: BLE001 - optional
                        _log.warning(
                            "improvement tick job not registered: %s", exc)
                # Prompt 01: skill self-rewrite loop + synthesis scan jobs.
                # ensure_improvement_schedule is idempotent by name; both
                # jobs are no-ops while settings.improvement.mode == "off".
                try:
                    from ..skill_evolution import ensure_improvement_schedule
                    ensure_improvement_schedule(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "self-improvement schedule not registered: %s", exc)
                # Prompt 03: watchers sweeper — one durable "watchers sweep"
                # job (every 1m) that runs the watch tool's sweep action.
                # ensure_sweeper_job is idempotent by name; the sweep itself
                # is a no-op when no watchers exist.
                try:
                    from ..watchers import ensure_sweeper_job
                    ensure_sweeper_job(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "watchers sweeper job not registered: %s", exc)
                # Prompt 05: rooms tick — one durable "rooms tick" job
                # (every 5m) that runs the room tool's tick action, advancing
                # each active room's linked goal/project.  Idempotent by
                # name; a no-op when no rooms exist.
                try:
                    from ...workspace.rooms import ensure_rooms_tick_job
                    ensure_rooms_tick_job(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "rooms tick job not registered: %s", exc)
                # Prompt 11: persona rebuild + curation — two durable daily
                # jobs (rebuild at 03:30, curate at 04:00) that keep the
                # user model fresh and memory hygienic.  Idempotent by name;
                # no-ops when memory is empty.
                try:
                    from ..scheduler import ensure_persona_jobs
                    ensure_persona_jobs(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "persona jobs not registered: %s", exc)
                # Prompt 04: morning briefing — one durable "briefing" job
                # (daily at settings.briefing.time, default 07:00, owner
                # tz) that composes + delivers the overnight digest.
                # Idempotent by name; catch-up runs once on boot if Devon
                # was down at briefing time.
                try:
                    from ..morning_briefing import (
                        ensure_briefing_job, check_catchup)
                    ensure_briefing_job(self.context)
                    check_catchup(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("briefing job not registered: %s", exc)
                # Wave E: always-on research loop — one durable
                # "research loop" job (every NM_RESEARCH_LOOP_HOURS,
                # default 6h) ticking the Wave C organs (swarm -> digest
                # -> upgrade queue). Idempotent; the loop itself
                # re-checks the research feature flag, the proactive
                # master switch, and quiet hours before running.
                try:
                    from ..research_loop import ensure_research_job
                    ensure_research_job(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("research loop job not registered: %s", exc)
                # The cognitive loop (wave 51): one heartbeat that ticks
                # goals (driving linked projects), improvement, and the
                # personal-model fine-tune. On when the autonomy dial is on
                # (config, power mode, or the durable `nm autonomy on`).
                # Idempotent.
                from ..cognition import autonomy_enabled as _auto_on
                if _auto_on(self.context):
                    try:
                        have = [j for j in self._scheduler.list_jobs()
                                if j.get("name") == "cognitive loop"]
                        if not have:
                            try:
                                from ..cognition import CognitiveLoop

                                hours = CognitiveLoop(self.context) \
                                    .effective_interval()
                            except Exception:  # noqa: BLE001
                                _auto = getattr(self.settings, "autonomy", None)
                                hours = max(0.25, float(getattr(
                                    _auto, "interval_hours", 6.0) or 6.0))
                            hours = max(0.25, hours)
                            self._scheduler.add(
                                "cognitive loop", f"every {hours:g}h", "tool",
                                {"tool": "autonomy", "args": {"action": "tick"}},
                            )
                    except Exception as exc:  # noqa: BLE001 - optional
                        _log.warning(
                            "cognitive loop job not registered: %s", exc)
                # Wave 71: change monitors — a short fixed-interval job
                # ticks every active watch while the bot runs. Idempotent.
                try:
                    from ..monitor import MonitorAgent as _MonitorAgent

                    if _MonitorAgent(self.context).list():
                        have = [j for j in self._scheduler.list_jobs()
                                if j.get("name") == "monitor tick"]
                        if not have:
                            self._scheduler.add(
                                "monitor tick", "every 5m", "tool",
                                {"tool": "monitor",
                                 "args": {"action": "tick"}},
                            )
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("monitor tick job not registered: %s", exc)
            except Exception as exc:  # noqa: BLE001 - scheduler is optional
                _log.warning("scheduler failed to start: %s", exc)
                self._scheduler = None
        # Trial assist: background signups killed by a restart are marked
        # interrupted and reported once, so no run silently disappears.
        # Idempotent — only non-terminal rows are touched.
        # Trial SMS watches: background verification-code watches killed
        # by a restart are resumed (deadline ahead) or closed as
        # timed-out, and reported once — same guarantee as assist runs.
        if not self.dry_run:
            try:
                from ..trial import TrialFlow

                TrialFlow.recover_interrupted_runs(
                    self.context, gateway=self.gateway)
                TrialFlow.recover_interrupted_watches(
                    self.context, gateway=self.gateway)
            except Exception as exc:  # noqa: BLE001 - recovery is optional
                _log.warning("trial assist recovery failed: %s", exc)
        return started

    def _arena_notify(self, text: str) -> None:
        """Deliver an arena build review to the owner, on every live channel."""
        from ...social.chat.base import ChatRef

        for key in _key_set(self.settings.partner.owner_chats):
            plat, _, cid = key.partition(":")
            if not cid:
                continue
            try:
                self.gateway.send(plat, ChatRef(platform=plat, chat_id=cid), text)
            except Exception:  # noqa: BLE001 - best-effort delivery
                pass

    def _tick_beacon(self, force: bool = False) -> None:
        """Write the status beacon on cadence (wave 94).  Never raises."""
        from ..beacon import BEACON_INTERVAL_S

        now = time.time()
        if not force and now < self._beacon_next:
            return
        self._beacon_next = now + BEACON_INTERVAL_S
        try:
            from ..beacon import build_beacon_state, write_status

            write_status(self.settings.home, build_beacon_state(self))
        except Exception:  # noqa: BLE001 - beacon must never take the bot down
            _log.debug("status beacon write failed", exc_info=True)

    def run(self, *, duration: float | None = None) -> None:
        self._started = time.time()
        self._beacon_next = 0.0  # write one immediately at startup
        # SIGTERM (supervisor stop, `kill`, phone process kill) must take
        # the same path as Ctrl-C: the loop exits and `finally` runs
        # stop(), which writes the final beacon with stopped=True so
        # `nm status` reports a clean stop instead of a stale death.
        from ...core.shutdown import install_sigterm_as_interrupt

        install_sigterm_as_interrupt()
        started = self.start()
        _log.info("partner runtime running on: %s (Ctrl-C to stop)", ",".join(started) or "none")
        try:
            if duration is not None:
                deadline = time.time() + duration
                while not self._stopped.is_set() and time.time() < deadline:
                    self._tick_beacon()
                    time.sleep(0.5)
            else:
                while not self._stopped.is_set():
                    self._tick_beacon()
                    time.sleep(0.5)
        except KeyboardInterrupt:  # noqa: E103, E106 - interrupt ends the beacon loop; finally stops it
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._stopped.set()
        if getattr(self, "_arena", None) is not None:
            try:
                self._arena.stop_loop()
            except Exception:  # noqa: BLE001
                pass
        if getattr(self, "_research", None) is not None:
            try:
                self._research.stop_loop()
            except Exception:  # noqa: BLE001
                pass
        if self._autonomy is not None:
            try:
                self._autonomy.stop()
            except Exception:  # noqa: BLE001
                pass
        if getattr(self, "_scheduler", None) is not None:
            try:
                self._scheduler.stop()
            except Exception:  # noqa: BLE001
                pass
        engine = getattr(self, "_game_engine_obj", None)
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:  # noqa: BLE001
                pass
        self.gateway.stop()
        # Drain (bounded) BEFORE the pool shuts down and the context's DB
        # closes: an in-flight pump writing after db.close() raises
        # StorageError("database is closed") and drops the reply.
        self._drain_chat_pool()
        self._pool.shutdown(wait=False)
        # final beacon, marked stopped — `nm status` then reports a clean
        # "stopped" instead of "went stale"
        try:
            from ..beacon import build_beacon_state, write_status

            state = build_beacon_state(self)
            state["stopped"] = True
            write_status(self.settings.home, state)
        except Exception:  # noqa: BLE001
            pass
        _log.info("partner runtime stopped: %s", self._stats_snapshot())

    def status(self) -> dict[str, Any]:
        return {
            "stats": self._stats_snapshot(),
            "mood": self.brain.mood.current().to_dict(),
            "relationship": {
                "stage": self.brain.relationship.stage,
                "trust": self.brain.relationship.trust,
                "milestones": len(self.brain.relationship.milestones),
                "fights": len(self.brain.relationship.fights),
            },
            "platforms": self.gateway.status(),
            "autonomy": self._autonomy.status() if self._autonomy else "off",
            # Dynamic lexicon voice feed: per-category term counts so the
            # owner can see what the feed is working with (deeper inspection
            # via the research_lexicon tool: action=terms|stats).
            "lexicon": (
                self.brain.lexicon.status()
                if self.brain.lexicon is not None
                else {"available": False, "enabled": False, "module": "partner"}
            ),
            # Wave D: how many interactive replies hit the reply budget and
            # fell back instead of stalling the chat.
            "reply_timeouts": self.brain.responder.reply_timeouts,
        }

    def say(self, chat_key: str, text: str, *, reply_to: str = "") -> Any:
        """Owner-initiated send (CLI `partner --say`): goes through the gateway,
        which keeps it rate-limited and audited like everything else."""
        chat = ChatRef.parse(chat_key)
        return self.gateway.send(chat.platform, chat, text, reply_to=reply_to)

    def _adopt_power_mode(self) -> None:
        """Power mode persists across restarts: re-widen on boot if unlocked."""
        if self.gateway is None:
            # Defensive: the gateway is built in __init__ before this runs,
            # but never crash boot on a None gateway — just skip the widen.
            _log.warning("power-mode restore skipped: gateway not ready")
            return
        try:
            from ..power import power_mode_for

            power = power_mode_for(self.context)
            power.adopt_persisted_state()
            if power.active:
                # unlimited inbound while power is active (0 = no cap)
                self.gateway.set_rate_limit(0)
        except Exception as exc:  # noqa: BLE001
            _log.warning("power-mode restore failed: %s", exc)

    def _apply_persisted_mode(self) -> None:
        """A /mode switch (or `nm partner --mode`) persists; honor it on boot."""
        try:
            row = self.context.db.query_one(
                "SELECT value FROM kv_store WHERE key = 'partner.autonomy_mode'"
            )
            if row:
                persisted = str(json.loads(row["value"]).get("mode") or "")
                if persisted in {"off", "suggest", "auto"}:
                    self.settings.partner.autonomy_mode = persisted
        except Exception:  # noqa: BLE001 - a stale kv row must not break boot
            pass

    # ── control commands (live steering) ─────────────────────────────────────
    def _is_operator(self, message: ChatMessage) -> bool:
        # Same single owner test as reply gating — a control command from a
        # chat the gating layer doesn't call owner must not run.
        # Prefer the gateway's verdict (stashed in meta) which includes the
        # DB-registered owner chats; fall back to the config-only check.
        meta = getattr(message, "meta", None)
        if isinstance(meta, dict) and meta.get("is_owner") is True:
            return True
        return is_owner_chat(message.chat, owner_chats=self._owner_chats)

    def _progression_gate(self, kind: str, message: Any) -> str | None:
        """Locked-command reply when a player lacks a progression unlock.

        Returns the reply text to send instead of dispatching, or None when
        the command is open / unlocked / the sender is the owner. Never
        raises — on any failure the command dispatches normally (fail-open
        for the gate itself; the unlock check inside is fail-closed).
        """
        try:
            from ...games.unlocks import command_unlock_required, locked_reply

            required = command_unlock_required(kind)
            if required is None:
                return None
            if message is not None and self._is_operator(message):
                return None  # owner bypass: progression never gates the owner
            player = None
            try:
                player = self._game_player(message)
            except Exception:  # noqa: BLE001
                player = None
            if player is None:
                # No resolvable player (e.g. console call without a sender):
                # treat as locked only when a sender exists, else let the
                # command's own handler complain about the missing player.
                if message is None or not getattr(message, "sender", None):
                    return None
                player_key = "unknown"
            else:
                player_key = player.key
            db = getattr(self.context, "db", None)
            return locked_reply(kind, db, player_key)
        except Exception:  # noqa: BLE001
            return None

    def _build_adapter(self, name: str) -> Any:
        """Factory for hot starts. Returns the adapter, None for an unknown
        or disabled platform, and raises with a readable message when a
        platform is enabled but unavailable (missing dependency/credentials) —
        the gateway reports that message to the owner."""
        from ...social.chat import build_adapter

        return build_adapter(self.settings, name)

    def handle_control(self, text: str, chat_key: str,
                       message: Any = None) -> str:
        """Parse + dispatch one control command. Returns the reply to send.

        A registered kind that has no dispatch branch can only happen through
        a code inconsistency (``parse_control`` only returns registered kinds);
        it falls back to an ``"unknown command /<kind>"`` reply rather than
        raising, so chat never dies on a dispatch-table gap.
        """
        from ...social.chat.control import detailed_help, parse_control
        from ..power import power_mode_for

        command = parse_control(text)
        if command is None:
            return ""  # not actually a control command; let normal flow handle it
        kind, arg = command.kind, command.arg
        if kind == "error":
            return arg

        # Progression-gated commands: beating challenges unlocks commands.
        # Checked before dispatch; the owner bypasses (their capabilities
        # come from role config — progression is additive for players only).
        locked = self._progression_gate(kind, message)
        if locked is not None:
            return locked

        if kind in {"list", "commands", "menu"}:
            # wave 68: /list — every executable chat command, categorized
            # with a one-line "what it does"; /list <group> filters.
            from ...social.chat.control import list_catalog
            return list_catalog(arg)
        if kind == "help":
            # wave 67: detailed help — /help for the catalog,
            # /help <command> for one command's full page, /help budget
            # etc. for topic pages.
            return detailed_help(arg or command.tail)

        if kind == "profile":
            # wave 86: the profile-aware runtime, in chat (owner-only —
            # control commands never reach non-operators)
            from ...core.runtune import build_tune

            tune = self.context.extras.get("tune") or build_tune(self.settings)
            p = tune.profile
            lines = [
                f"running on: {p.kind} ({p.detail}) — {p.cpu} cores, "
                f"{p.memory_mb // 1024 if p.memory_mb else '?'} GB",
                f"threads {tune.threads} · vcpus {tune.vcpu_min}/{tune.vcpu_target}/{tune.vcpu_max} "
                f"· chats {tune.max_parallel_chats} · downloads "
                f"{tune.max_concurrent_downloads}×{tune.max_download_mb:g}MB",
                f"memory {tune.context_budget_tokens:,} tokens ({tune.memory_pressure}) · "
                f"missions {tune.mission_max_concurrent}@{tune.mission_autonomy:.0%} · "
                f"model pref {tune.model_pref}",
                "override: runtime.<knob> in config or NM_RUNTIME_<KNOB>",
            ]
            return "\n".join(lines)

        if kind == "quit":
            threading.Thread(target=self.stop, name="partner-quit", daemon=True).start()
            return "shutting down. the rest of you is on disk."

        if kind == "status":
            return self._control_status()

        if kind == "platforms":
            lines = ["platforms:"]
            for name, info in sorted(self.gateway.status().items()):
                if name == "_stats":
                    continue
                state = "running" if info.get("running_in_session") else "not running"
                suffix = "" if info.get("running") else " (adapter thread down)"
                lines.append(f"  {name}: {state}{suffix}")
            return "\n".join(lines)

        if kind == "start":
            result = self.start_platform(arg)
            return (f"{arg}: started" if result.get("ok")
                    else f"{arg}: {result.get('error', 'could not start')}")

        if kind == "stop":
            result = self.stop_platform(arg)
            return (f"{arg}: stopped" if result.get("ok")
                    else f"{arg}: {result.get('error', 'could not stop')}")

        if kind == "mood":
            return self._control_mood(command.tail or arg)

        if kind == "mode":
            return self._control_mode(arg)

        if kind == "model":
            return self._control_model(command.tail)

        if kind == "providers":
            return self._control_providers()

        if kind == "say":
            target, sep, payload = command.tail.partition(" ")
            payload = payload.strip()
            if not sep or not payload or ":" not in target:
                return "usage: /say telegram:123 the text to send"
            platform, _, chat_id = target.partition(":")
            if not chat_id:
                return "usage: /say telegram:123 the text to send"
            from ...social.chat.base import ChatRef

            result = self.gateway.send(
                platform, ChatRef(platform=platform, chat_id=chat_id), payload
            )
            return "sent." if result.ok else f"send failed: {result.error}"

        if kind == "proposals":
            rows = self.proposals()
            if not rows:
                return "no pending proposals."
            lines = [f"pending proposals ({len(rows)}):"]
            for row in rows[:10]:
                lines.append(f"  {row['id']} [{row['kind']}→{row['platform']}:{row['chat_id']}] {str(row['content'])[:70]}")
            return "\n".join(lines)

        if kind == "approve":
            result = self.approve(arg)
            if result.get("ok"):
                return f"approved {arg} → {result.get('status', 'sent')}"
            err = result.get("error") or "no reason given"
            return f"approve failed: {err} — /proposals to check what's still pending"

        if kind == "deny":
            result = self.deny(arg)
            if result.get("ok"):
                return "denied."
            err = result.get("error") or "no reason given"
            return f"deny failed: {err} — /proposals to check what's still pending"

        if kind == "stage":
            rel = self.brain.relationship
            if not arg:
                return f"stage: {rel.stage} (trust {rel.trust:.0f})"
            from ...partner.relationship import STAGES

            arg = arg.strip().lower()
            if arg not in STAGES:
                return f"unknown stage {arg!r}. use one of: {', '.join(STAGES)}"
            from ...partner.relationship import STAGE_ORDER

            target = STAGE_ORDER[arg]
            current = STAGE_ORDER.get(rel.stage, 1)
            while target > current and rel.stage_index < len(STAGES) - 1:
                rel.advance_stage(reason="owner command")
                current += 1
            while target < current and rel.stage_index > 0:
                rel.regress_stage(reason="owner command")
                current -= 1
            try:
                rel.save(self.context.db)
            except Exception:  # noqa: BLE001
                pass
            return f"stage: {rel.stage}"

        if kind == "power":
            power = power_mode_for(self.context)
            parts = (command.tail or arg).split()
            verb = parts[0].lower() if parts else ""
            if verb == "on":
                if len(parts) < 2:
                    return ("usage: /power on <owner-key> — or: "
                            "/power on <identity> <passphrase>  (owner seal; "
                            "safer via `nm power unlock`, which hides input)")
                if len(parts) >= 3:
                    result = power.unlock(parts[2], actor=chat_key,
                                          identity=parts[1])
                else:
                    result = power.unlock(parts[1], actor=chat_key)
                if result.get("ok"):
                    # Power mode = unlimited: the inbound rate cap lifts too.
                    self.gateway.set_rate_limit(0)
                return result.get("message", "")
            if verb == "off":
                result = power.lock(actor=chat_key)
                self.gateway.set_rate_limit(self._base_rate_limit())
                return result.get("message", "")
            if verb == "status":
                s = power.status()
                if not s["key_configured"]:
                    return "power mode: locked (no owner key configured)"
                if not s["active"]:
                    return "power mode: locked"
                if not s.get("applied_in_process"):
                    return f"power mode: active (persisted, by {s['unlocked_by']}; this process has the base dials)"
                fields = ", ".join(c["field"] for c in s["changes"])
                return f"power mode: ACTIVE (by {s['unlocked_by']}). widened: {fields}"
            return "usage: /power on <key> | /power off | /power status"

        # ── search engine ────────────────────────────────────────────────────
        if kind == "search":
            return self._control_search(command.tail, mode="quick", chat_key=chat_key)
        if kind == "searchdeep":
            return self._control_search(command.tail, mode="deep", chat_key=chat_key)
        if kind == "searchleads":
            return self._control_search_leads()
        if kind == "searchhist":
            return self._control_search_hist(arg)
        if kind == "redteam":
            # Defensive self-attack suite: sandboxed, never touches externals.
            from ..redteam import control_redteam

            return control_redteam(command.tail or arg, context=self.context)
        if kind == "memories":
            # Trust view: what she remembers. Summaries only, never raw dumps.
            from ..proactive import control_memory

            return control_memory(command.tail or arg, context=self.context)
        if kind == "spend":
            # Naira-first expense log: /spend 5k on transport
            from ...finance.commands import control_spend

            return control_spend(command.tail or arg, context=self.context)
        if kind == "budget":
            # Monthly budgets: /budget set food 50k | /budget list
            from ...finance.commands import control_budget

            return control_budget(command.tail or arg, context=self.context)
        if kind == "spending":
            # Spending summary vs budgets: /spending [week|month]
            from ...finance.commands import control_spending

            return control_spending(command.tail or arg, context=self.context)
        if kind == "predict":
            # Prediction pit: unlocked by beating challenges (see
            # nomorals/games/unlocks.py). Gated above in _progression_gate.
            return self._control_predict(
                command.tail or arg, chat_key, message=message)
        if kind == "miniapp":
            # Group mini-apps: architecturally isolated from owner memory,
            # accounts and vaults (see nomorals/community/__init__.py).
            # Group chats only — control_miniapp refuses DMs itself.
            from ...community.miniapps import control_miniapp

            chat = message.chat if message is not None else None
            return control_miniapp(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "email":
            # AI email triage. Drafts are never auto-sent; sending needs
            # an approved draft or the owner's explicit "send it".
            from ..email_triage import control_email

            return control_email(command.tail or arg, context=self.context)
        if kind == "book":
            return self._control_book(tail=command.tail or arg, chat_key=chat_key)
        if kind in {"wisdom", "wis"}:
            # WisdomKeeper: /wisdom and its /wis alias share one handler.
            return self._control_wisdom(command.tail or arg, chat_key=chat_key,
                                        message=message)
        if kind == "decode":
            return self._control_decode(command.tail or arg, chat_key=chat_key)
        if kind == "cookies":
            return self._control_cookies(command.tail or arg)
        if kind == "structure":
            return self._control_structure(command.tail or arg)
        if kind == "investigate":
            return self._control_investigate(command.tail or arg)
        if kind == "monitor":
            return self._control_monitor(command.tail or arg)
        if kind == "cipher":
            return self._control_cipher(command.tail or arg)
        if kind == "music":
            return self._control_music(command.tail or arg, chat_key=chat_key)
        if kind == "play":
            return self._control_play(command.tail or arg)
        if kind == "video":
            return self._control_video(command.tail or arg)
        if kind == "exec":
            return self._control_exec(command.tail or arg)
        if kind == "zip":
            return self._control_zip(command.tail or arg)
        if kind == "apps":
            return self._control_apps(command.tail or arg)
        if kind == "hub":
            return self._control_hub(command.tail or arg, chat_key=chat_key)
        if kind == "podcast":
            return self._control_podcast(command.tail or arg, chat_key=chat_key)
        if kind == "fix":
            return self._control_fix(command.tail or arg)

        # ── feature flags ────────────────────────────────────────────────────
        if kind == "features":
            return self._control_features(command.tail or arg)

        # ── self-improvement arena ───────────────────────────────────────────
        if kind == "arena":
            return self._control_arena(
                command.tail or arg, chat_key=chat_key,
                player=self._game_player_for_key(chat_key), kind="dm")

        # ── single-account trials ────────────────────────────────────────────
        if kind == "trial":
            return self._control_trial(command.tail or arg, chat_key=chat_key)

        # ── identity bank (profile for signups) ──────────────────────────────
        if kind == "identity":
            return self._control_identity(command.tail or arg)

        # ── expansion wave ───────────────────────────────────────────────────
        if kind == "game":
            # console path: no inbound message, so the player is derived from
            # the chat key (stable per chat) and the kind defaults to dm.
            return self._control_game(
                command.tail or arg, chat_key=chat_key,
                player=self._game_player_for_key(chat_key), kind="dm")
        if kind == "npc":
            # the living cast — game-scoped via the live room in this chat
            return self._control_npc(command.tail or arg, chat_key=chat_key)
        if kind == "dm":
            # the narrator's persona for the current game
            return self._control_dm(command.tail or arg, chat_key=chat_key)
        # ── arena gear: persistent equipment ─────────────────────────────────
        if kind in ("inventory", "equip", "unequip", "repair"):
            return self._control_gear(
                kind, command.tail or arg,
                player=self._game_player_for_key(chat_key))
        # ── progression ──────────────────────────────────────────────────────
        if kind == "level":
            return self._control_level(
                player=self._game_player_for_key(chat_key))
        # ── battle skills ────────────────────────────────────────────────────
        if kind == "skill":
            return self._control_skills(
                command.tail or arg,
                player=self._game_player_for_key(chat_key))
        # ── titles ───────────────────────────────────────────────────────────
        if kind == "title":
            return self._control_titles(
                command.tail or arg,
                player=self._game_player_for_key(chat_key))
        # ── RPG attributes ───────────────────────────────────────────────────
        if kind == "stats":
            return self._control_stats(
                command.tail or arg,
                player=self._game_player_for_key(chat_key))
        # ── daily hunt ───────────────────────────────────────────────────────
        if kind == "daily":
            return self._control_daily(
                player=self._game_player_for_key(chat_key))
        # ── per-game mastery tiers ───────────────────────────────────────────
        if kind == "mastery":
            return self._control_mastery(
                command.tail or arg,
                player=self._game_player_for_key(chat_key))
        # ── player-to-player gifting ─────────────────────────────────────────
        if kind == "gift":
            return self._control_gift(
                command.tail or arg,
                player=self._game_player_for_key(chat_key))
        if kind == "news":
            return self._control_news(command.tail or arg)
        if kind == "research":
            return self._control_research(command.tail or arg,
                                          chat_key=chat_key, message=message)
        if kind == "code":
            return self._control_code(command.tail, chat_key=chat_key)
        if kind == "py":
            return self._control_py(command.tail, chat_key=chat_key)
        if kind == "remember":
            return self._control_remember(command.tail or arg, chat_key=chat_key)
        if kind == "recall":
            return self._control_recall(command.tail or arg)
        if kind == "forget":
            return self._control_forget(command.tail or arg)
        if kind == "tts":
            return self._control_tts(command.tail or arg, chat_key=chat_key)
        if kind == "stt":
            return self._control_stt(command.tail or arg)
        if kind == "look":
            return self._control_look(command.tail or arg, chat_key=chat_key)
        if kind == "schedule":
            return self._control_schedule(command.tail or arg)
        if kind == "db":
            return self._control_db(command.tail or arg)
        if kind == "api":
            return self._control_api(command.tail or arg)
        if kind == "swarm":
            return self._control_swarm(command.tail or arg, chat_key=chat_key)
        if kind == "dns":
            return self._control_dns(command.tail or arg)
        if kind == "scan":
            return self._control_scan(command.tail or arg)
        if kind == "whois":
            return self._control_whois(command.tail or arg)
        if kind == "ports":
            return self._control_ports(command.tail or arg)
        if kind == "proxy":
            return self._control_proxy(command.tail or arg)
        if kind == "workspace":
            return self._control_workspace(command.tail or arg)
        if kind == "gen":
            return self._control_gen(command.tail or arg)
        if kind == "osint":
            return self._control_osint(command.tail or arg, chat_key=chat_key)
        if kind == "record":
            return self._control_record(command.tail or arg)
        if kind == "macro":
            return self._control_macro(command.tail or arg, chat_key=chat_key)
        if kind == "file":
            return self._control_file(command.tail or arg)
        if kind == "publish":
            return self._control_publish(command.tail or arg)
        if kind == "deliver":
            return self._control_deliver(command.tail or arg, chat_key)
        if kind == "data":
            return self._control_data(command.tail or arg)
        if kind == "evolve":
            return self._control_evolve(command.tail or arg, chat_key=chat_key)
        if kind == "upgrade":
            # the originating chat rides along so _control_upgrade can
            # re-check the owner gate itself (defense in depth — a direct
            # call with a non-owner chat is denied, fail-closed).
            chat = getattr(message, "chat", None) if message is not None else None
            return self._control_upgrade(command.tail or arg, _chat=chat)
        if kind == "speak":
            return self._control_speak(command.tail or arg, chat_key=chat_key)
        if kind == "voice":
            return self._control_voice(command.tail, chat_key,
                                       message=message)
        if kind == "bet":
            return self._control_bet(command.tail or arg, chat_key=chat_key)
        if kind == "money":
            from ..opportunities import handle_money_command
            return handle_money_command(command.tail or arg, self.context)
        if kind == "finance":
            return self._control_finance(command.tail or arg, chat_key=chat_key)
        if kind == "weather":
            return self._control_weather(command.tail or arg)
        if kind == "tz":
            return self._control_tz(command.tail or arg)
        if kind == "task":
            return self._control_task(command.tail or arg, chat_key=chat_key)
        if kind == "mind":
            return self._control_mind(command.tail or arg, chat_key=chat_key)
        if kind == "notify":
            return self._control_notify(arg)
        if kind == "proactive":
            return self._control_proactive(arg)
        if kind == "mission":
            return self._control_mission(command.tail or arg, chat_key=chat_key)
        if kind == "image":
            return self._control_image(command.tail or arg)
        if kind == "lens":
            return self._control_lens(command.tail or arg)

        # ── devon: the autonomous dev agent ─────────────────────────────────
        if kind == "devon":
            return self._control_devon(command.tail, chat_key=chat_key)

        # ── reasoning: explicit, auditable multi-step thought ───────────────
        if kind == "think":
            return self._control_think(command.tail or arg, chat_key=chat_key)
        if kind == "benchmark":
            return self._control_benchmark(command.tail or arg)

        return f"unknown command /{kind}"

    # ── search control commands ──────────────────────────────────────────────
    def _base_rate_limit(self) -> int:
        try:
            return int(self.settings.chat.max_per_hour)
        except Exception:  # noqa: BLE001
            return 60

    @staticmethod
    def _ref_from_key(key: str):
        from ...social.chat.base import ChatRef

        plat, _, cid = key.partition(":")
        return ChatRef(platform=plat or "local", chat_id=cid or key)

    def _typing_for(self, chat: ChatRef, text: str, *, cap: float = 10.0) -> None:
        """Length-scaled typing before a short outgoing reply (control
        answers, game moves). Skipped in groups when configured off; never
        raises — typing is cosmetic."""
        partner_cfg = self.settings.partner
        if chat.kind == ChatKind.GROUP and not partner_cfg.typing_in_groups:
            return
        try:
            self.gateway.typing(
                chat.platform, chat,
                seconds=human_typing_seconds(
                    text, minimum=1.5, cap=cap, rng=self.brain.presence_rng,
                ),
            )
        except Exception:  # noqa: BLE001
            _log.debug("typing indicator failed (cosmetic)", exc_info=True)

    def _typing_keepalive(self, chat: ChatRef, stop: threading.Event) -> None:
        """Hold "typing…" up while the brain is working (own daemon thread).

        Platforms expire the indicator (Telegram ~5s, Discord ~10s), so it
        is refreshed on a short cadence until the reply starts or the
        patience budget runs out. Works with both blocking adapters
        (Telegram/Discord hold the call for the window) and fire-and-forget
        ones (the bridge), since the wait tops up the cycle to the tick.
        """
        try:
            self._typing_keepalive_loop(chat, stop)
        finally:
            # One-shot thread: don't leak its DB connection (gateway.send ->
            # touch opens one on the first successful send).
            self._release_db_thread()

    def _typing_keepalive_loop(self, chat: ChatRef, stop: threading.Event) -> None:
        """Body of :meth:`_typing_keepalive` (split out so the wrapper can
        own the thread-teardown ``finally``)."""
        partner_cfg = self.settings.partner
        if chat.kind == ChatKind.GROUP and not partner_cfg.typing_in_groups:
            return
        tick = max(0.2, partner_cfg.typing_keepalive_seconds)
        deadline = time.time() + partner_cfg.typing_keepalive_budget
        # One "still thinking" line in a DM when the answer is taking a long
        # time — on a phone a cold 7B needs minutes, and silence reads as
        # death (which is exactly when owners reach for Ctrl+C).
        notice_at = partner_cfg.slow_reply_notice_seconds
        noticed = False
        started_all = time.time()
        while not stop.is_set() and time.time() < deadline:
            if (not noticed and notice_at > 0
                    and chat.kind != ChatKind.GROUP
                    and time.time() - started_all >= notice_at):
                noticed = True
                try:
                    self.gateway.send(chat.platform, chat,
                                      "one sec — still thinking, the model is slow on the phone")
                except Exception:  # noqa: BLE001 - the notice must never kill the reply
                    _log.warning("slow-reply notice failed for %s", chat.key)
            started = time.time()
            if not self.gateway.typing(chat.platform, chat, seconds=tick):
                return  # no adapter / platform down — stop, don't retry blindly
            stop.wait(max(0.05, tick - (time.time() - started)))

    def _send_long(self, platform: str, chat, text: str, limit: int = 3800) -> int:
        """Send a long text in chunks (Telegram's hard cap is 4096 chars).

        Each chunk gets its own length-scaled typing run, so a report split
        into five pages reads as five real typing sessions, not a telegraph.

        Returns the number of chunks actually delivered. A failed chunk is
        a warning log, never silent — callers must not treat the return as
        "delivered" without checking it (see :meth:`_send_long_checked`).
        """
        partner_cfg = self.settings.partner
        typing_on = (
            getattr(chat, "kind", ChatKind.DM) != ChatKind.GROUP
            or partner_cfg.typing_in_groups
        )
        sent = 0
        chunks = max(1, (len(text) + limit - 1) // limit)
        for i in range(0, max(1, len(text)), limit):
            chunk = text[i:i + limit]
            if typing_on:
                self.gateway.typing(
                    platform, chat,
                    seconds=human_typing_seconds(
                        chunk,
                        mood=self.brain.mood.current().values,
                        rng=self.brain.presence_rng,
                        minimum=partner_cfg.typing_seconds,
                        cap=partner_cfg.typing_cap_seconds,
                    ),
                )
            try:
                result = self.gateway.send(platform, chat, chunk)
                if result.ok:
                    sent += 1
                else:
                    _log.warning("send_long: chunk %d/%d to %s failed: %s",
                                 sent + 1, chunks, getattr(chat, "key", chat),
                                 result.error)
                    break
            except Exception as exc:  # noqa: BLE001
                _log.warning("send_long: chunk %d/%d to %s raised %s: %s",
                             sent + 1, chunks, getattr(chat, "key", chat),
                             type(exc).__name__, exc)
                break
        return sent

    def _send_long_checked(self, platform: str, chat, text: str,
                           limit: int = 3800) -> str:
        """Send a long text; ``""`` when delivered, an honest failure note
        when NOTHING got through.

        Callers that used to do ``self._send_long(...); return ""`` ("report
        already delivered in chunks") must use this: a total send failure
        used to vanish silently, leaving the owner with no reply at all.
        """
        sent = self._send_long(platform, chat, text, limit=limit)
        if sent == 0 and (text or "").strip():
            return ("⚠️ delivery failed — I couldn't send that message "
                    "(gateway down?). Nothing was delivered; check the logs.")
        return ""

    def _notify(self, chat: Any, text: str) -> None:
        """Best-effort chat notification (background book pipeline)."""
        try:
            self.gateway.send(chat.platform, chat, text)
        except Exception:  # noqa: BLE001
            pass

    def start_platform(self, name: str) -> dict[str, Any]:
        return self.gateway.start_one(name.strip().lower(), self.on_message)

    def stop_platform(self, name: str) -> dict[str, Any]:
        return self.gateway.stop_one(name.strip().lower())

    def proposals(self, status: str = "pending") -> list[dict[str, Any]]:
        return self.context.db.query(
            "SELECT * FROM proactive_log WHERE status = ? ORDER BY decided_at DESC LIMIT 50",
            (status,),
        )

    def approve(self, proposal_id: str) -> dict[str, Any]:
        if self._autonomy is not None:
            return self._autonomy.approve(proposal_id)
        return _direct_approve(self.context, self.gateway, proposal_id)

    def deny(self, proposal_id: str) -> dict[str, Any]:
        if self._autonomy is not None:
            return self._autonomy.deny(proposal_id)
        return _direct_deny(self.context, proposal_id)

    # ── power layer: network / proxy / gen / osint / record / macro ──────────

    def _tool_reply(self, tool: str, chat_key: str, **kwargs: Any) -> str:
        """Run one tool and stream its JSON back to the chat (or the error)."""
        outcome = self.context.tools.call(tool, **kwargs)
        chat = self._ref_from_key(chat_key)
        if not outcome.ok:
            return f"{tool} failed: {getattr(outcome.error, 'message', outcome.error)}"
        text = json.dumps(outcome.value, default=str, indent=1)
        return self._send_long_checked(chat.platform, chat, text[:6000])
