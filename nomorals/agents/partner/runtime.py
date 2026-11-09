"""PartnerRuntime: the loop that keeps the partner alive."""

from __future__ import annotations

import contextvars
import json
import re
import threading
import time
from datetime import datetime
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import Any, Callable
from ...llm.brain import brain_for
from ...core.logging_setup import get_logger
from ...partner.chat_modes import (
    feature_disabled_reply as _chat_modes_disabled_reply,
    feature_for_command as _chat_modes_feature_for,
    game_disabled_reply as _chat_modes_game_disabled_reply,
    is_feature_disabled as _chat_modes_is_feature_disabled,
    is_game_disabled as _chat_modes_is_game_disabled,
    is_games_disabled as _chat_modes_is_games_disabled,
    is_games_only as _chat_modes_is_games_only)
from ...partner.gating import (
    gate_decision, is_gated_surface, is_owner_chat, is_restricted)
from ...partner.presence import Presence, decide_presence, human_typing_seconds
from ...social.chat.base import ChatKind, ChatMessage, ChatRef
from ...social.chat.gateway import ChatGateway
from ...social.chat.typography import apply_stamp
from ...storage.kv import KVStore
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
from .runtime_provision import RuntimeProvisionMixin
from .runtime_group import RuntimeGroupMixin
from .runtime_expression import RuntimeExpressionMixin
from .runtime_fun import RuntimeFunMixin
from .runtime_system import RuntimeSystemMixin
from .runtime_media import RuntimeMediaMixin, play_owner_var
from .runtime_intel import RuntimeIntelMixin
from .runtime_search import RuntimeSearchMixin
from .runtime_wisdom import RuntimeWisdomMixin
from .runtime_reach import RuntimeReachMixin
from .runtime_anime import RuntimeAnimeMixin
from .runtime_chatops import RuntimeChatOpsMixin
from .runtime_makers import RuntimeMakersMixin
from .failures import failure_count, recent_failures, record_failure
from .reply_guard import (
    deliberation_reason,
    guard_deliberation_parts,
    guard_reply_parts,
    scrub_narration,
)
from .brain_gate import BrainGate, is_personal_platform
_log = get_logger(__name__)

#: Living-card tap stamp (Living Chat P2): a card button's
#: callback payload — "cardtap <card_id> <revision> <index>" —
#: resolved against the durable card before anything executes.
_CARD_TAP_RE = re.compile(r"^/cardtap ([0-9A-Za-z_-]+) (\d+) (\d+)$")


def _update_recovery_notice(receipt: dict) -> tuple[str, str]:
    """Owner-facing (title, body) for an update crash recovery that acted.

    Only called when the receipt says the recovery DID something
    (``recovered`` is true); a clean boot never reaches this, so the
    text can state plainly what happened and what state Devon is
    running on now.
    """
    action = str(receipt.get("action", ""))
    snap = str(receipt.get("restored_snapshot")
               or receipt.get("snapshot") or "")
    if action == "rolled_back":
        return (
            "interrupted update rolled back at startup",
            "Devon restarted while a self-update was mid-flight — it "
            "died during or after migrations, so at startup I rolled "
            f"it back: restored the pre-update snapshot ({snap}) and "
            "put the code tree back to the pre-update version. You "
            "are running the state from before that update. The "
            "interrupted run is closed out and will not retry on its "
            "own — run /update when you're ready.",
        )
    if action == "rolled_back_state_only":
        return (
            "interrupted update: state restored, code NOT restored",
            "Devon restarted while a self-update was mid-flight. At "
            f"startup I restored the pre-update snapshot ({snap}), "
            "but the code tree could not be put back "
            f"({receipt.get('code_restore_error') or 'unknown error'})"
            " — the code may still be the interrupted update's. "
            "Check the install before relying on it.",
        )
    if action == "rollback_failed":
        return (
            "interrupted update recovery FAILED at startup",
            "Devon restarted while a self-update was mid-flight, and "
            "the startup recovery could not restore the pre-update "
            f"state ({receipt.get('error') or 'unknown error'}). "
            "Devon is booting on the possibly half-updated state — "
            "database and/or code from an update that never finished. "
            "Check the install before trusting it; a manual snapshot "
            "restore may be needed.",
        )
    if action == "pending_restore_applied":
        return (
            "deferred database restore applied at startup",
            "A failed self-update could not restore the database "
            "while Devon was running (the running process holds the "
            "database open), so the restore was deferred. At this "
            f"startup I applied it: the database is back to the "
            f"pre-update snapshot ({snap}). Nothing else is pending "
            "from that update.",
        )
    if action == "pending_restore_failed":
        # The receipt carries the ACTUAL state (which update the
        # deferred restore belongs to, which version is running, and
        # whether that code is the rolled-back version) — state it,
        # never assert a pairing the facts don't support. The static
        # fallback covers receipts from older builds that carry no
        # state note.
        state = str(receipt.get("state_note") or "")
        if state:
            state_line = f"Actual state: {state}. "
        else:
            state_line = ("Devon is booting on the post-update "
                          "database paired with rolled-back code. ")
        return (
            "deferred database restore FAILED at startup",
            "A failed self-update deferred its database restore to "
            "this startup because the database was locked while Devon "
            f"ran, and the restore still could not be applied "
            f"({receipt.get('error') or 'unknown error'}). "
            f"{state_line}"
            "Check the install before trusting it; a manual "
            "snapshot restore may be needed.",
        )
    if action == "pending_restore_discarded":
        # The marker's premise was provably gone (the running code
        # is not the rolled-back version the deferred restore pairs
        # with), so recovery discarded it BEFORE attempting the
        # restore. Information, not a failure: the live database —
        # the correct state — was kept untouched. The receipt's
        # state note carries the facts, as on the failure notice.
        state = str(receipt.get("state_note") or "")
        state_line = f"Actual state: {state}. " if state else ""
        return (
            "stale deferred database restore discarded at startup",
            "A failed self-update left a database restore pending, "
            "but that rollback no longer applies — the code Devon "
            f"is running is not the version it was meant to pair "
            f"with. {state_line}"
            "At this startup I discarded the pending restore "
            "instead of applying it: the live database is kept, "
            "exactly as it is, and nothing is pending from that "
            "update. No action needed.",
        )
    if action == "pending_restore_unfulfillable":
        # The marker's snapshot was deleted after the restore was
        # deferred, so the restore can never be applied and the
        # marker was discarded. The live database was kept — but
        # unlike the premise-gone discard, the rollback's database
        # half is unrecoverable, so no "no action needed": state
        # the pairing the facts support (via the state note) and
        # name the consequence.
        state = str(receipt.get("state_note") or "")
        state_line = f"Actual state: {state}. " if state else ""
        return (
            "deferred database restore can no longer be applied",
            "A failed self-update deferred its database restore to "
            "this startup, but the snapshot it would restore "
            f"({receipt.get('snapshot') or 'unknown'}) no longer "
            "exists — it was deleted after the restore was "
            "deferred. There is nothing left to apply, so at this "
            "startup I discarded the pending restore and kept the "
            f"live database exactly as it is. {state_line}"
            "The rollback that update performed is complete in "
            "code; only its database half is unrecoverable. If the "
            "database misbehaves under the rolled-back code, "
            "restore a snapshot manually.",
        )
    if action == "restart_confirmed":
        return (
            "update restart confirmed — running the new version",
            "Devon restarted into the self-update and this process "
            "is back up on version "
            f"{str(receipt.get('running_sha') or '?')[:12]} — the "
            "version the update landed. The restart is confirmed "
            "by the process that actually came up, not assumed by "
            "the one that went down.",
        )
    if action == "restart_unconfirmed":
        return (
            "update restart NOT confirmed",
            "Devon was restarted for a self-update, but the "
            "version now running is "
            f"{str(receipt.get('running_sha') or 'unreadable')[:12]}, "
            "not the version the update landed "
            f"({str(receipt.get('to_sha') or '?')[:12]}). The new "
            "code may not be running — check the install before "
            "trusting it, and run /update again when ready.",
        )
    if action == "tree_restored":
        return (
            "interrupted update: half-applied pull put back at startup",
            "Devon restarted while a self-update was mid-pull. A "
            "pull writes the new code file by file, so the tree "
            "may have been half-written; at startup I put the "
            "working tree back to exactly its pre-update state "
            "(verified against the status the update recorded "
            "before pulling). The database was never touched — "
            "the update died before migrations — and your local "
            "changes are untouched. Run /update when you're ready.",
        )
    if action == "tree_restore_failed":
        return (
            "interrupted update: code tree could NOT be put back",
            "Devon restarted while a self-update was mid-pull, "
            "and the startup recovery could not restore the "
            f"working tree ({receipt.get('tree_restore') or receipt.get('error') or 'unknown error'}). "
            "The code on disk may be a half-written mix of the "
            "old and new versions — check the install (git status "
            "in the repo) before trusting it.",
        )
    # aborted (and any future close-out action): died before the
    # dangerous steps, so nothing was changed and nothing restored.
    return (
        "interrupted update closed out at startup",
        "A self-update was interrupted by a restart before it "
        "changed anything (it died before migrations), so at "
        "startup I marked it aborted. Nothing was restored and "
        "nothing is pending — no action needed.",
    )


class _VoiceUnavailable(Exception):
    """Voice synthesis degraded (the voice ladder's silent rung served).

    Carries the ladder's honesty clause in ``honesty_note`` so the text
    fallback can tell the chat why the reply is text, not voice.
    """

    def __init__(self, note: str = "") -> None:
        super().__init__(note or "voice synthesis unavailable")
        self.honesty_note = note

#: telegram-bot is the public endpoint: conversation + games only. A known
#: non-game slash command there gets this redirect instead of running —
#: owner/system controls live on the userbot (``telegram`` endpoint).
_PUBLIC_BOT_NOT_AVAILABLE = (
    "that's not something I can do here — this is my public chat, "
    "so I keep it to conversation and games. "
    "try /game list to see what's on, or just talk to me."
)


def _partner_failures_status(limit: int = 5) -> dict[str, Any]:
    """Ledger snapshot for :meth:`PartnerRuntime.status`.

    Total ever recorded plus the newest entries (site, type, message).
    The ledger's accessors only lock and read, so this never raises.
    """
    return {
        "total": failure_count(),
        "recent": [
            {"site": f.site, "exc_type": f.exc_type,
             "message": f.message}
            for f in recent_failures(limit)
        ],
    }



_BUTTON_CLAIM_RE = re.compile(
    r"tap (?:a )?(?:button|tool) below|buttons? (?:are|on each screen)"
    r"|tap (?:the )?\S+ to run",
    re.IGNORECASE)


def _claims_buttons(text: str) -> bool:
    """True when reply copy tells the reader to tap buttons."""
    return bool(_BUTTON_CLAIM_RE.search(str(text or "")))


def _strip_button_claims(text: str) -> str:
    """Remove button-claim lines (platform takes no buttons and
    none were authored): the actions stay named in the body copy,
    the false instruction goes."""
    kept = [ln for ln in str(text or "").split("\n")
            if not _BUTTON_CLAIM_RE.search(ln)]
    return "\n".join(kept).strip("\n")


def _pick_owner_key(message: Any) -> str:
    """The operation-slot owner for a message's cards: the stable
    sender id when the adapter provides one, else the display
    sender, else "" (the chat's shared slot)."""
    return str(getattr(message, "sender_id", "")
               or getattr(message, "sender", "") or "")


def _numbered_button_fallback(runtime: Any, message: Any, text: str,
                              rows: list) -> str:
    """Rewrite a button claim into a numbered action list.

    The authored buttons become ``reply 1 — <label>`` lines; the
    picks are registered on the runtime so the owner's bare-number
    reply runs the same command a tap would have (register #27:
    the fallback must actually execute, not just describe).
    """
    flat = [pair for row in rows for pair in row]
    lines = [ln for ln in str(text or "").split("\n")
             if not _BUTTON_CLAIM_RE.search(ln)]
    body = "\n".join(lines).rstrip("\n")
    picks: list[str] = []
    rendered = []
    for i, (label, data) in enumerate(flat, start=1):
        rendered.append(f"{i} — {label}")
        picks.append(str(data))
    register = getattr(runtime, "_register_button_picks", None)
    if callable(register):
        try:
            register(message.chat.key, picks,
                     owner_key=_pick_owner_key(message))
        except Exception as exc:  # noqa: BLE001 - the list still reads honestly
            _log.debug("button pick registration failed: %s", exc)
    return (body + "\n\nno buttons on this surface — reply with a "
            "number:\n" + "\n".join(rendered))


#: Delivery mark for the #32 no-empty wrapper in handle_control:
#: _send_long bumps this whenever it actually delivers chunks, so a
#: handler that answered by SENDING (the _send_long_checked
#: discipline — "" means delivered; a failed delivery returns its
#: own ⚠️ note instead) is never mistaken for a silent handler and
#: followed by a false "bug on my end" message. A ContextVar, so
#: concurrent dispatches each see only their own deliveries.
_control_deliveries: contextvars.ContextVar[int] = (
    contextvars.ContextVar("nm_control_deliveries", default=0))


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
    RuntimeProvisionMixin,
    RuntimeGroupMixin,
    RuntimeExpressionMixin,
    RuntimeFunMixin,
    RuntimeMediaMixin,
    RuntimeIntelMixin,
    RuntimeSearchMixin,
    RuntimeWisdomMixin,
    RuntimeReachMixin,
    RuntimeAnimeMixin,
    RuntimeMakersMixin,
    RuntimeChatOpsMixin,
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
        resource_advisor: Callable[[], Any] | None = None,
    ) -> None:
        """Create the runtime.

        ``session_bridge`` is an optional ``os.SessionBridge`` (injected —
        this module is L5 and must not import ``os``/L6). When present it is
        passed to the ChatGateway so every inbound message is attached to
        its OS Session before the brain runs.

        ``resource_advisor`` is an optional factory returning the resource
        manager (``os.resources.default_manager``, injected for the same
        layering reason). When None, the scheduler runs ungated — the
        same fallback as a missing advisor.
        """
        self.context = context
        self.settings = context.settings
        self.brain = brain or PartnerBrain(context)
        # Register #28: the brain consults the SAME owner-replied
        # ledger the runtime feeds from adapter own-echoes.
        try:
            self.brain._owner_reply_ledger_obj = self._owner_reply_ledger()
        except Exception as exc:  # noqa: BLE001 - the brain forms its own ledger then
            record_failure("runtime.PartnerRuntime.brain_ledger", exc)
        #: Two-tier memory (build-map #36): lazy, opt-in via
        #: memory.two_tier_enabled. None until the first observed exchange.
        self._two_tier: Any = None
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
        #: Optional resource-advisor factory (os.resources, L6), injected
        #: to keep layering clean. None means the scheduler runs ungated.
        self._resource_advisor = resource_advisor
        #: Optional os.SessionBridge (L6), injected to keep layering clean.
        #: Also published on context.extras so CLI/tools can reach it.
        #: (Restored: the resource_advisor commit accidentally replaced
        #: this assignment instead of adding alongside it, breaking
        #: gateway construction with a bridge + the bridge tests.)
        self._session_bridge = session_bridge
        if session_bridge is not None:
            try:
                context.extras["session_bridge"] = session_bridge
            except Exception as e:
                record_failure("runtime.PartnerRuntime.__init__", e)
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
        #: THE autonomy side-ledger handle (idle clock + pattern /
        #: interest writers). One lazily-opened Database for the whole
        #: process, via autonomy.idle.workspace_db — never a fresh
        #: Database per message (see _autonomy_db for the failure
        #: history that rule comes from).
        self._autonomy_db_handle: Any = None
        self._autonomy_db_guard = threading.Lock()
        #: Monotonic deadline before which a failed ledger open/write
        #: is not retried: one journal entry per outage episode.
        self._autonomy_db_retry_at: float = 0.0
        self._autonomy_db_episode: bool = False
        #: Boot mission-resumption thread (see _start_boot_mission_resume).
        self._boot_resume_thread: Any = None
        self.stats = {"messages": 0, "replies": 0, "errors": 0, "controls": 0}
        #: Guards ``stats``: the counters are bumped from every chat-pool
        #: worker, the delayed-reply threads, and the control path. ``+=``
        #: on a dict value is a read-modify-write and lost increments
        #: otherwise (silent undercounting in /status).
        self._stats_lock = threading.Lock()
        #: Competence gate state (brain_gate): degraded-episode
        #: bookkeeping for personal-surface reply holds.
        self._brain_gate = BrainGate()
        self._owner_chats = _key_set(partner_cfg.owner_chats)
        # Sender-id allowlist (NM_PARTNER_OWNER_SENDER_IDS): the owner's
        # Telegram user id. The chat-key test above fails when the owner
        # configured a bare id; the sender id is platform-independent,
        # so it also covers the personal bot and userbot DMs where the
        # chat key carries the adapter prefix.
        self._owner_sender_ids = {
            s.strip()
            for s in str(getattr(partner_cfg, "owner_sender_ids", "") or "").split(",")
            if s.strip()
        }
        self._book_busy: set[str] = set()  # slugs with a book pipeline running
        self._apply_persisted_mode()

        if self.gateway is None:
            from ...social.chat import build_adapters
            from ...social.identity import IdentityStore

            adapters, skipped = build_adapters(
                self.settings, on_new_member=self._on_new_discord_member,
                db=getattr(self.context, "db", None))
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
                # Cross-platform identity: inbound senders register in
                # the store and carry a unified person_id downstream.
                identity_store=IdentityStore(),
                # Ordinary chat path gets durable delivery (row first,
                # receipt after) whenever a db exists; without a db the
                # gateway reports durable-requested-unavailable and each
                # send carries delivery_degraded=True (seam A).
                durable_outbox=True,
            )

        # Tools that deliver files into chats (file_send, report_publish)
        # reach the live gateway through the context, not a second wire.
        context.extras["gateway"] = self.gateway
        # Unsolicited-send paths that deliver outside a live turn (the
        # Notifier above all) consult the competence gate through the
        # context the same way — one gate instance, the runtime's own.
        context.extras["brain_gate"] = self._get_brain_gate()
        # Degradation-ladder drain: messages the telegram ladder queued
        # into the KV store while no gateway existed (gateway-less
        # processes) move into this gateway's durable outbox now that a
        # gateway DOES exist — the outbox's own drain delivers them.
        try:
            from ...core.error_system import drain_tg_retry_queue
            from ...storage.kv import KVStore

            drained = drain_tg_retry_queue(
                context, self.gateway, kv_factory=lambda db: KVStore(db))
            if drained:
                _log.info("drained %d ladder-queued message(s) into the "
                          "durable outbox", drained)
        except Exception as exc:  # noqa: BLE001 — drain is best-effort
            _log.debug("tg_retry drain skipped: %s", exc)
        # The spine's social tools (social_send, social_history, …) reach
        # the same live gateway. Bound once here so the brain's hands are
        # live on every platform from boot.
        try:
            from ...tools.social import set_gateway as _bind_social_gateway
            _bind_social_gateway(self.gateway)
            from ...tools.whatsapp import set_gateway as _bind_wa_gateway
            _bind_wa_gateway(self.gateway)
        except Exception as e:
            record_failure("runtime.PartnerRuntime.__init__:329", e)

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
            except Exception as e:
                record_failure("runtime.PartnerRuntime._wire_console_commands", e)
                local = None
            if local is None:
                return
            # Don't clobber a hook an entrypoint set deliberately (e.g.
            # run_chat_bot.py wires its own richer snapshot with history).
            if getattr(local, "command_hook", None) is not None:
                return
            local.command_hook = ConsoleCommands(self.console_snapshot).handle
        except Exception as e:
            record_failure("runtime.PartnerRuntime._wire_console_commands:367", e)
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
        except Exception as e:
            record_failure("runtime.PartnerRuntime.console_snapshot", e)
        snap["adapters"] = adapters
        # ── traffic ──
        try:
            with self._stats_lock:
                snap["traffic"] = dict(self.stats)
        except Exception as e:
            record_failure("runtime.PartnerRuntime.console_snapshot:400", e)
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
                            "routine": bool(job.get("routine")),
                            "next_run": job.get("next_run"),
                        }
                    )
            snap["scheduler"] = {"running": running, "jobs": jobs}
        except Exception as e:
            record_failure("runtime.PartnerRuntime.console_snapshot:420", e)
            snap["scheduler"] = {}
        # ── games ──
        try:
            row = self.context.db.query_one("SELECT COUNT(*) AS n FROM game_players")
            players = (row or {}).get("n", "—")
        except Exception as e:
            record_failure("runtime.PartnerRuntime.console_snapshot:426", e)
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
        except Exception as e:
            record_failure("runtime.PartnerRuntime.console_snapshot:438", e)
            snap["llm"] = {}
        # ── theme ──
        snap["theme"] = os.environ.get("NM_CONSOLE_THEME", "ocean")
        # ── extras ──
        extras: dict = {}
        try:
            extras["autonomy"] = "on" if getattr(self, "_autonomy", None) else "off"
        except Exception as e:
            record_failure("runtime.PartnerRuntime.console_snapshot:446", e)
        try:
            extras["arena"] = "on" if getattr(self, "_arena", None) else "off"
        except Exception as e:
            record_failure("runtime.PartnerRuntime.console_snapshot:450", e)
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
            # A first-DM greeting is an UNSOLICITED send — Devon
            # speaks first, from a synthetic turn. Its degraded
            # state rides through like any other outcome's, so the
            # competence gate judges it by the same provenance as a
            # live reply (before this, the flag was dropped here and
            # a fallback greeting could go out ungated).
            self._send_reply(message, outcome.parts,
                             degraded=outcome.degraded,
                             degraded_note=outcome.degraded_note)

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
        # /ping stage stamp: the moment the runtime took this message
        # in (receipt). The platform's own timestamp on the message is
        # the arrival stage; _process stamps the processing stage.
        try:
            message.meta["received_at"] = time.time()
        except Exception as e:  # noqa: BLE001 - a stamp must never break intake
            record_failure("runtime.PartnerRuntime.on_message:stamp", e)
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

    def _passes_games_only(self, message: ChatMessage) -> bool:
        """Does this message pass a games-only chat? Game moves were
        already handled by the game router above; this decides the rest.

        Passes: owner control commands (any registered kind — so the
        owner can /chatmode normal), and game commands the sender's
        grant allows. Everything else (plain chat, non-game slashes,
        owner-only commands from non-owners) drops silently.
        """
        from ...social.chat.control import (
            CONTROL_COMMANDS, GAME_COMMANDS, caller_may_run, parse_control)

        text = (message.text or "").strip()
        if not text.startswith("/"):
            return False
        command = parse_control(text)
        if command is None or command.kind == "error":
            return False
        is_operator = self._is_operator(message)
        if is_operator and command.kind in CONTROL_COMMANDS:
            return True
        # The games-only pass set: the game hub, every direct game
        # command (from the registry, never a pasted list), plus /arena
        # and /play (the owner's games-adjacent commands — /arena stays
        # owner-only via caller_may_run, /play is public).
        games_only_pass = (
            frozenset({"game", "arena", "play"}) | frozenset(GAME_COMMANDS))
        if command.kind in games_only_pass and caller_may_run(
                command.kind, is_operator):
            return True
        return False

    def _handle_music_voice_note(self, message: ChatMessage) -> bool:
        """Music voice-note intercept. Returns True when consumed.

        A bare voice note whose transcription already failed may have
        failed because it contains MUSIC, not speech — STT on a song
        returns an empty transcript. The caller runs this before the
        honest STT-failure reply, or a music note would get "couldn't
        transcribe" and never reach the ask.

        * Music + OWNER → send the Find/Create ask with inline
          buttons, remember the clip as this chat's pending music
          note, consume the message (no brain turn, no STT reply).
          The ask IS the gate: no production runs until the owner
          taps Create.
        * Music + NON-OWNER DM → NO ask, NO buttons. The note is
          re-presented to the brain as music context and the message
          falls through to normal conversation — Devon decides her
          own reply (voice or text) from there.
        * Not music → False: the STT-failure reply below runs as usual.

        Owner gating uses :meth:`_is_operator` (owner identity —
        meta / sender-id allowlist / owner chats), never chat type
        alone. Never raises; any failure means "not music".
        """
        try:
            from ...voice import music_note as _music_note
            from ...voice.pingpong import has_voice_media as _has_voice
        except Exception as e:  # noqa: never fatal
            record_failure("runtime.PartnerRuntime._handle_music_voice_note", e)
            return False
        try:
            if not _has_voice(message):
                return False
            # Only notes transcription already failed on: a transcribed
            # note is speech by definition, and this hook must never
            # re-examine (or delay) the normal speech path.
            if not (message.meta or {}).get("stt_failure_reason"):
                return False
            path = ""
            for media in message.media or []:
                kind = str(getattr(media, "kind", "") or "")
                mime = str(getattr(media, "mime", "") or "")
                if (kind in ("audio", "voice") or mime.startswith("audio")) \
                        and getattr(media, "path", ""):
                    path = str(media.path)
                    break
            if not path:
                return False
            found = _music_note.detect_music(path)
            if not found.get("is_music"):
                return False
            heard = str(found.get("heard") or "")
            if self._is_operator(message):
                _music_note.PENDING.remember(message.chat.key, path, heard)
                try:
                    self._bump("replies")
                    self.gateway.send(
                        message.chat.platform, message.chat,
                        _music_note.ASK_TEXT,
                        buttons=_music_note.ask_buttons())
                except Exception:  # noqa: BLE001 - ask send is best-effort
                    _log.exception("music voice-note ask send failed")
                return True
            # Non-owner: no ask, no buttons — Devon decides. Stamp the
            # clip as music (so the brain's _media_notes presents it
            # as music context, never as untranscribed speech), clear
            # the STT reason (so the STT-failure reply below does not
            # fire), and give the brain a bracketed description to
            # react to. voice_in stays unset: a voice reply is her
            # choice via the normal voice-mode rules, never forced.
            try:
                message.meta["music_note"] = {"path": path, "heard": heard}
                message.meta.pop("stt_failure_reason", None)
                message.meta.pop("voice_untranscribed_reason", None)
                message.text = _music_note.music_context_text(heard)
            except Exception as e:
                record_failure("runtime.PartnerRuntime._handle_music_voice_note:719", e)
            return False
        except Exception as e:
            record_failure("runtime.PartnerRuntime._handle_music_voice_note:722", e)
            _log.debug("music voice-note detection failed", exc_info=True)
            return False

    #: Seconds before a failed autonomy-ledger open/write is retried.
    #: The ledger is best-effort bookkeeping: a dead workspace must
    #: cost one journal entry per outage episode, not one per message.
    _AUTONOMY_DB_RETRY_S = 300.0

    def _autonomy_db_lock(self) -> threading.Lock:
        guard = getattr(self, "_autonomy_db_guard", None)
        if guard is None:
            guard = self._autonomy_db_guard = threading.Lock()
        return guard

    def _autonomy_db_outage(self, exc: BaseException, site: str) -> None:
        """Enter/refresh the ledger outage episode.

        Drops the handle (a later retry reopens from scratch), arms
        the retry deadline, and journals the episode exactly once —
        repeated failures inside the cooldown stay silent.
        """
        with self._autonomy_db_lock():
            handle = getattr(self, "_autonomy_db_handle", None)
            self._autonomy_db_handle = None
            self._autonomy_db_retry_at = (
                time.monotonic() + self._AUTONOMY_DB_RETRY_S)
            first = not getattr(self, "_autonomy_db_episode", False)
            self._autonomy_db_episode = True
        if handle is not None:
            try:
                handle.close()
            except Exception:  # noqa: BLE001 - teardown is best-effort
                pass
        if first:
            record_failure(site, exc)

    def _autonomy_db(self) -> Any:
        """The process-lifetime autonomy ledger handle, or None.

        ONE lazily-opened Database for the idle/pattern/interest
        writers, opened through :func:`autonomy.idle.workspace_db` —
        the canonical seam mapping a workspace DIRECTORY to its DB
        file (``<dir>/nomorals.db``, the same file IdleMonitor,
        IdleCoordinator and presence read). The primary context DB
        is a different file (``data/nomorals.db``), so it cannot
        substitute: the monitor would never see these writes.

        History: the per-message block in ``_process_inner`` used to
        construct ``Database(workspace_dir)`` directly — handing
        SQLite a directory, which cannot open (OperationalError
        "unable to open database file") on every single message,
        each one journaled as a failure. The writes never landed
        and the failure ledger filled with the same row forever.

        Failure contract: never raises. A failed open journals ONE
        entry for the whole outage episode (via _autonomy_db_outage)
        and callers skip quietly until the retry deadline passes; a
        successful open after an episode logs the recovery.
        """
        handle = getattr(self, "_autonomy_db_handle", None)
        if handle is not None:
            return handle
        outage: BaseException | None = None
        with self._autonomy_db_lock():
            handle = getattr(self, "_autonomy_db_handle", None)
            if handle is not None:
                return handle
            if time.monotonic() < getattr(
                    self, "_autonomy_db_retry_at", 0.0):
                return None
            try:
                from pathlib import Path as _Path

                from ...autonomy.idle import workspace_db
                ws_dir = getattr(
                    self.context.settings, "workspace_dir", ".")
                # The DB file's parent IS the workspace dir; ensure
                # it exists at open (fresh installs, bare settings).
                _Path(ws_dir).mkdir(parents=True, exist_ok=True)
                handle = workspace_db(ws_dir)
            except Exception as e:  # noqa: BLE001 - best-effort ledger
                outage = e
            else:
                self._autonomy_db_handle = handle
                if getattr(self, "_autonomy_db_episode", False):
                    self._autonomy_db_episode = False
                    _log.info("autonomy ledger open again after an "
                              "outage episode")
                return handle
        # Journal/arm the cooldown outside the lock: record_failure
        # does its own locking and disk I/O.
        self._autonomy_db_outage(
            outage, "runtime.PartnerRuntime.autonomy_ledger_open")
        return None

    def _note_autonomy_activity(self, message: ChatMessage) -> None:
        """Feed the autonomy idle/pattern/interest models one message.

        Cheap, best-effort, never blocks and never raises — the
        ``_process_inner`` contract. Rides the one shared ledger
        handle from :meth:`_autonomy_db`; while the ledger is in a
        failure cooldown the note is skipped silently.
        """
        db = self._autonomy_db()
        if db is None:
            return
        try:
            from ...autonomy.idle import note_activity
            from ...autonomy.patterns import (
                record_activity,
                record_interests,
            )
            note_activity(db)
            if message.incoming:
                record_activity(db)
                if message.text:
                    record_interests(db, message.text)
        except Exception as e:  # noqa: BLE001 - never break intake
            self._autonomy_db_outage(
                e, "runtime.PartnerRuntime.autonomy_ledger_write")

    def _process(self, message: ChatMessage) -> None:
        """Inbound entry: process one message under its idempotency scope.

        The scope (chat + platform message id) names every tool effect
        this message can cause, so a redelivered message — gateway
        retry, adapter replay, process restart — replays recorded
        results instead of re-applying effects (see
        :mod:`nomorals.core.idempotency_keys`). A message with no
        platform id gets no scope and runs unkeyed: inventing an
        identity could collide two distinct messages into a false
        replay.
        """
        from ...core.idempotency_keys import idempotency_scope

        scope = ""
        try:
            chat_key = str(getattr(message.chat, "key", "") or "")
            message_id = str(getattr(message, "message_id", "") or "")
            if chat_key and message_id:
                scope = f"chat:{chat_key}:{message_id}"
        except Exception as e:
            record_failure("runtime.PartnerRuntime._process", e)
            scope = ""
        if scope:
            with idempotency_scope(scope):
                self._process_inner(message)
            return
        self._process_inner(message)

    def _process_inner(self, message: ChatMessage) -> None:
        # /ping stage stamp: processing starts now (after any queue wait).
        try:
            message.meta["process_at"] = time.time()
        except Exception as e:  # noqa: BLE001 - a stamp must never break intake
            record_failure("runtime.PartnerRuntime._process_inner:stamp", e)
        self._bump("messages")
        # Chat-title index (register #30): record this chat's title
        # so owner commands can target chats BY NAME later. Writes
        # only on change; failure-isolated like every intake beat.
        try:
            from ...partner.chat_targets import note_chat_title
            note_chat_title(self.context.db, message.chat)
        except Exception as e:  # noqa: BLE001
            record_failure("runtime.PartnerRuntime._process_inner:title", e)
        # Background /play + /download jobs: surface any that died in
        # this chat without reporting (crashed worker, heartbeat
        # silence).  The happy path delivers itself from the job
        # thread; this only handles outcomes the job never posted.
        try:
            self._harvest_play_downloads(message.chat.key, message.chat)
        except Exception as e:
            record_failure("runtime.PartnerRuntime._process_inner", e)
        # Startup journey (doc 4): if boot could not deliver the
        # readiness card (no transport was up yet), the owner's
        # first interaction on a live transport delivers the
        # retained record. Cheap no-op once delivered.
        try:
            self._deliver_pending_startup_card(message)
        except Exception as e:  # noqa: BLE001 - a surface, not intake
            record_failure(
                "runtime.PartnerRuntime._process_inner:startup_card", e)
        # Housekeeping beat (§15): the supervision sweep, the
        # WhatsApp live-attach reconcile and the story sweep —
        # throttled inside _housekeeping_beat, and driven from
        # here so ANY inbound message (bot platforms included)
        # can drive it.  It used to ride only the personal-surface
        # turn below, whose platform gate a bot-only deployment
        # never passes — a paired WhatsApp account stayed dark
        # from cold start until a /pair or /start triggered the
        # reconcile by hand.  Failure-isolated, like every cycle
        # here: housekeeping never breaks intake.
        try:
            beat = getattr(self, "_housekeeping_beat", None)
            if beat is not None:
                beat()
        except Exception as e:  # noqa: BLE001
            record_failure(
                "runtime.PartnerRuntime._housekeeping_beat", e)
        # Personal surfaces (§15 slices 2–3): on the owner's own
        # WhatsApp / Telegram-userbot accounts, status posts and DM
        # delete events are consumed here (saved / journaled for the
        # owner alone — never conversation, never replied to), and
        # ordinary DMs are observed for the antidelete cache and
        # view-once preservation before flowing on.  As-him hat: the
        # §33 system-output gate does not apply — nothing here posts
        # anything anywhere.
        try:
            personal_turn = getattr(self, "_personal_surface_turn", None)
            if personal_turn is not None and personal_turn(message):
                return
        except Exception as e:  # noqa: BLE001 - the personal
            record_failure(           # surfaces never break intake
                "runtime.PartnerRuntime._personal_surface_turn", e)
        # Stale replays (adapters stamp meta["stale_replay"]): a
        # message composed while a surface was offline, delivered at
        # (re)connect as if fresh. Everything above has already
        # observed it — housekeeping, the personal-surface hooks,
        # persistence — and that is ALL that happens: no command
        # executes, no reply is composed, no game moves. A stale
        # /help detonating as a 13-chunk catalog at boot (owner's
        # WhatsApp self-chat, 2026-11-04) is the incident this ends;
        # silence over stale performance is the owner's standing law.
        if message.meta.get("stale_replay"):
            self._bump("stale_held")
            _log.info(
                "stale replay observed, not answered: %s in %s "
                "(message ts %.0f, arrived %.0f)",
                (message.text or "")[:40], message.chat.key,
                float(message.ts or 0), time.time())
            return
        # Page-watch beat (§20b): due webshot watches re-shoot
        # on the message cycle (throttled inside; no daemon).
        # The driver owns its sends (§33 re-checked per alert);
        # a failure here never breaks intake.
        try:
            watch_drive = getattr(self, "_drive_webshot_watches",
                                  None)
            if watch_drive is not None:
                watch_drive()
        except Exception as e:  # noqa: BLE001
            record_failure(
                "runtime.PartnerRuntime._drive_webshot_watches",
                e)
        # Security-audit beat: queued /scan + /audit jobs run
        # one per cycle and deliver their report to the DM they
        # were requested from — no daemon, the watch discipline.
        try:
            audit_drive = getattr(self, "_drive_security_audits",
                                  None)
            if audit_drive is not None:
                audit_drive()
        except Exception as e:  # noqa: BLE001
            record_failure(
                "runtime.PartnerRuntime._drive_security_audits",
                e)
        # Channel posture (owner order, 2026-11-02): a channel is a
        # broadcast surface, not a conversation. Devon never replies in
        # one — no commands, no catalog, no games, no persona chat, not
        # even for the owner. Channel control happens in DM or the
        # linked discussion group; the channel system (programming,
        # analytics) reads posts through its own path. Ingest stops
        # here on purpose: channel traffic must not feed the interest
        # models, the memory, or the brain.
        if message.chat.kind == ChatKind.CHANNEL:
            _log.info("channel post in %s noted; channels are silent "
                      "by posture", message.chat.key)
            # Owner-side control plane (P4a): the sighting is
            # noted in the channel registry, the bot-claim probe
            # runs, and the posting queue takes its beat.  The
            # message itself still feeds nothing — no memory,
            # no brain, no reply; control stays in DM.
            note = getattr(self, "_channel_note_and_drive", None)
            if note is not None:
                try:
                    note(message)
                except Exception as e:  # noqa: BLE001 - the
                    # control plane never breaks the silence
                    record_failure(
                        "runtime.PartnerRuntime.channel_note", e)
            return
        # Group surface (DESIGN §1, P1): group chats are claimed by
        # the surface before conversation — service events (join /
        # leave) run the welcome and member-ledger organs, ordinary
        # messages are observed into the ledger and moderated. A
        # message the surface consumes (service event, or moderated
        # away) never reaches the conversation pipeline.
        if message.chat.kind == ChatKind.GROUP:
            try:
                if self._group_surface_turn(message):
                    return
            except Exception as e:  # noqa: BLE001 - surface never
                # breaks the chat; the failure is recorded, the
                # message flows on unmoderated rather than dropped.
                record_failure(
                    "runtime.PartnerRuntime._group_surface_turn", e)
        # One-time login intake (armed by /provision login): the
        # owner's next message IS a password. Consumed here — before
        # the autonomy interest recorder and before the brain — so the
        # secret never reaches history, logs, or interest models.
        try:
            if self._try_consume_login_intake(message):
                return
        except Exception:  # noqa: BLE001 - intake never breaks chat
            _log.exception("login intake consume failed")
        # Autonomy: every inbound message resets the idle clock and feeds
        # the pattern/interest models. Cheap, never blocks — and never
        # raises (the shared-ledger helper owns its failure episodes).
        self._note_autonomy_activity(message)
        # Pending voice-clone naming: /voice clone with no name holds the
        # audio and asks; the next short text message in that chat is the
        # name. Checked before anything else so it can't be misread as chat.
        if message.incoming and not (message.text or "").strip().startswith("/"):
            pending = getattr(self, "_pending_voice_clones", None) or {}
            hold = pending.get(message.chat.key)
            if hold:
                name = (message.text or "").strip()
                # A name is short and single-line; anything else isn't a name.
                if name and len(name) <= 40 and "\n" not in name:
                    import time as _t
                    if _t.time() - hold.get("ts", 0) < 600:  # 10 min window
                        del pending[message.chat.key]
                        reply = self._finish_voice_clone(
                            message.chat.key, message.chat, name, hold["path"])
                        try:
                            self._typing_for(message.chat, reply)
                            self.gateway.send(message.chat.platform,
                                              message.chat, reply)
                        except Exception as e:
                            record_failure("runtime.PartnerRuntime._process_inner:807", e)
                        return
                    del pending[message.chat.key]  # expired
        # Live DJ room sensing: if DJ Vrede is performing in this chat,
        # every incoming message feeds the room sensor (non-blocking).
        # The brain still processes the message normally — the DJ listens
        # alongside, it doesn't replace her.
        if message.incoming and (message.text or "").strip():
            try:
                from ...media.dj_live import notify_live_message
                notify_live_message(message.chat.key,
                                    message.text, message.sender)
            except Exception as e:
                record_failure("runtime.PartnerRuntime._process_inner:820", e)
        # SMS fallback surface (build-map #19): SMS can't carry media into
        # the brain. Answer honestly instead of letting her hallucinate
        # about a picture she never received.
        if (message.incoming and message.chat.platform == "sms"
                and message.meta.get("sms_has_media")):
            from ...social.chat.sms import SMS_MEDIA_REPLY

            try:
                self.gateway.send(message.chat.platform, message.chat,
                                  SMS_MEDIA_REPLY)
            except Exception:  # noqa: BLE001 - reply is best-effort
                _log.exception("sms media reply send failed")
            return
        # Vision flag: with it off, inbound VISUAL media is dropped
        # before the brain (no download-to-understanding pipeline, no
        # token cost). Audio is NOT vision — voice notes survive so STT
        # still runs (a vision-off chat must not silently kill voice).
        if message.media:
            from ..features import feature_enabled

            if not feature_enabled(self.context, "vision"):
                import dataclasses

                kept = [m for m in message.media
                        if str(getattr(m, "kind", "") or "").lower() in ("audio", "voice")
                        or str(getattr(m, "mime", "") or "").lower().startswith("audio")]
                if len(kept) != len(message.media):
                    message = dataclasses.replace(message, media=kept)
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
        # Per-chat voice toggle, decision D3 (text-replies-only): with
        # voice off in this chat, inbound voice notes STILL transcribe
        # (transcription is understanding, not a voice reply) — only
        # the outbound voice reply is suppressed (see _send_reply).
        # There is deliberately no refusal here: a voice note in a
        # voice-off chat is answered in text. Games-only chats stay
        # silent (their gate owns them).
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
                    has_voice_media, transcribe_voice_media_with_reason)
                if has_voice_media(message):
                    heard, _stt_reason = transcribe_voice_media_with_reason(
                        self.context, message)
                    if heard:
                        import dataclasses

                        message = dataclasses.replace(message, text=heard)
                        message.meta["voice_in"] = True
                        _log.info("voice note in %s heard: %r",
                                  message.chat.key, heard[:60])
                        # Voice money commands (owner only): a confident
                        # money intent short-circuits here so a transfer
                        # hits the fingerprint gate before the brain ever
                        # sees it. Anything else falls through untouched.
                        if self._is_operator(message):
                            from ...voice.money import voice_money_precheck
                            try:
                                money_reply = voice_money_precheck(
                                    heard, self.context)
                            except Exception:  # noqa: BLE001
                                _log.exception("voice money pre-check failed")
                                money_reply = None
                            if money_reply:
                                try:
                                    self.gateway.send(
                                        message.chat.platform,
                                        message.chat, money_reply)
                                except Exception:  # noqa: BLE001
                                    _log.exception(
                                        "voice money reply send failed")
                                return
                            # Ekiti pronunciation check (owner only): an armed
                            # /ekiti check consumes the next voice note here.
                            from ...learn.dialect import (
                                consume_check, pending_check)
                            if pending_check(message.chat.key):
                                try:
                                    scored = consume_check(
                                        message.chat.key, heard)
                                except Exception:  # noqa: BLE001
                                    _log.exception("ekiti check failed")
                                    scored = None
                                if scored:
                                    try:
                                        self.gateway.send(
                                            message.chat.platform,
                                            message.chat, scored)
                                    except Exception:  # noqa: BLE001
                                        _log.exception(
                                            "ekiti check reply failed")
                                    return
        # Meal photo logging (owner only): a photo with meal intent — or an
        # armed /health meal — starts the 2-question flow. The next text
        # message answers the questions and completes the log.
        if message.incoming and self._is_operator(message):
            from ...health import nutrition as _nut
            try:
                images = [m for m in (message.media or [])
                          if getattr(m, "kind", "") == "image"]
                chat_key = message.chat.key
                text = (message.text or "").strip()
                if images and (_nut.meal_flow_armed(chat_key)
                               or _nut.meal_intent_in_text(text)):
                    _nut.disarm_meal_flow(chat_key)
                    draft = _nut.log_meal(getattr(images[0], "path", ""))
                    if draft.ok:
                        _nut.arm_meal(chat_key, draft)
                        reply = _nut.format_draft_message(draft)
                    else:
                        reply = (f"couldn't log this one — {draft.error}. "
                                 "try a clearer photo of the food.")
                    try:
                        self.gateway.send(message.chat.platform,
                                          message.chat, reply)
                    except Exception:  # noqa: BLE001
                        _log.exception("meal photo reply failed")
                    return
                pending = _nut.pending_meal(chat_key)
                if (pending and text and not text.startswith("/")
                        and not images):
                    questions = _nut.follow_up_questions(pending)
                    answers = _nut.parse_meal_answers(text, questions)
                    meal_log = _nut.complete_meal(pending, answers)
                    _nut.consume_meal(chat_key)
                    try:
                        self.gateway.send(message.chat.platform,
                                          message.chat, meal_log.summary())
                    except Exception:  # noqa: BLE001
                        _log.exception("meal summary reply failed")
                    return
                # /distribute walk-through: plain replies advance the draft
                dist_reply = self._distribute_answer(chat_key, text or "")
                if (dist_reply is not None and text
                        and not text.startswith("/") and not images):
                    try:
                        self.gateway.send(message.chat.platform,
                                          message.chat, dist_reply)
                    except Exception:  # noqa: BLE001
                        _log.exception("distribute reply failed")
                    return
            except Exception:  # noqa: BLE001
                _log.exception("meal photo flow failed")
        # Owner-replied suppression ledger (register #28, restored
        # 2026-11-05): every adapter's owner-sent echo lands here
        # before anything else reads the turn, so the brain's
        # conversational path can stay quiet on a message/thread
        # the owner already answered himself. Control/command
        # handling never consults the ledger — commands answer.
        if (message.meta or {}).get("own_message"):
            try:
                self._owner_reply_ledger().record_owner_reply(
                    message.chat.key,
                    reply_to=str(
                        (message.meta or {}).get("reply_to") or ""))
            except Exception as exc:  # noqa: BLE001 - bookkeeping never eats a turn
                record_failure(
                    "runtime.PartnerRuntime.owner_reply_ledger", exc)
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
                # Games-off / games-disabled: a live game in this chat
                # is dead — a move or in-game command gets the honest
                # refusal, never a game turn. Plain messages return
                # None from the router and flow to normal chat below.
                if _chat_modes_is_games_disabled(message.chat.key):
                    move_reply = _chat_modes_disabled_reply("games")
                try:
                    self.gateway.send(message.chat.platform, message.chat, move_reply)
                except Exception:  # noqa: BLE001
                    _log.exception("game reply send failed")
                return
        # Per-chat games-only mode: this chat stays silent for normal
        # chat, but game commands and owner control commands still work
        # (so the owner can unset it). Game moves were handled above —
        # a plain message reaching here is not a move, so it drops.
        if message.incoming and _chat_modes_is_games_only(message.chat.key):
            if not self._passes_games_only(message):
                self._bump("dropped")
                _log.info("games-only: dropped non-game message in %s",
                          message.chat.key)
                return
        # Music voice note (owner: Find/Create ask; non-owner: music
        # context, Devon decides). A voice note whose transcription
        # failed may have failed because it contains MUSIC, not
        # speech — STT on a song returns an empty transcript. Check
        # BEFORE the honest STT-failure reply below, or a music note
        # would get "couldn't transcribe" and never reach the ask.
        # Games-only chats never reach here (their gate returned
        # above), so the ask stays silent there like every reply.
        if message.incoming and not message.text.strip():
            try:
                if self._handle_music_voice_note(message):
                    return
            except Exception:  # noqa: BLE001 - music detection never eats chat
                _log.exception("music voice-note hook failed")
        # STT failure, answered honestly — but ONLY in non-games-only
        # chats: the games-only gate above already returned, so silence
        # wins there and this never replies in a games-only chat. A
        # voice note whose transcription failed (no backend, API error,
        # empty transcript) gets ONE honest reply instead of silence
        # or a confused brain turn. Non-voice messages never reach
        # here with a reason recorded (only the STT paths set it).
        if message.incoming and not message.text.strip():
            _stt_reason = (message.meta or {}).get("stt_failure_reason", "")
            if _stt_reason:
                from ...voice.pingpong import has_voice_media as _has_voice
                if _has_voice(message):
                    self._bump("replies")
                    _log.info("STT failed in %s: %s",
                              message.chat.key, _stt_reason)
                    try:
                        self.gateway.send(
                            message.chat.platform, message.chat,
                            f"I couldn't transcribe that voice note "
                            f"({_stt_reason}).")
                    except Exception:  # noqa: BLE001 - honest reply is best-effort
                        _log.exception("STT failure reply send failed")
                    return
        # /game AND the direct game commands (/hangman, /mafia, …) work in
        # EVERY chat, for EVERY participant — games are social by nature, so
        # a member of any group can start and play one. These are also the
        # ONLY game start triggers outside the owner's DMs: natural language
        # never launches a game in a non-owner chat (wave 87). All other
        # slash commands stay owner-only: for everyone else a slash falls
        # through to her as text.
        if message.incoming and message.text.strip().startswith("/"):
            from ...social.chat.control import (
                GAME_COMMANDS, caller_may_run, parse_control,
                split_command_stack)

            command = parse_control(message.text)
            _log.debug("game command check: text=%r, command=%s",
                      message.text[:40], command)
            # One command, one handling, across surfaces (owner
            # order 2026-11-05): this message may be another
            # surface's copy of the same command — addressed to
            # it, in its DM, in a group where the bot's standing
            # is verified, or already claimed by it. Then this
            # copy stands down silently: observed and persisted
            # above, never handled, never answered here.
            if command is not None and self._command_stands_down(
                    message, command):
                self._bump("controls")
                _log.info(
                    "command /%s in %s stands down on %s — another "
                    "surface owns it", command.kind, message.chat.key,
                    message.chat.platform)
                return
            # /math in an outsider DM is conversation only (social_gate:
            # games are public in groups, not in DMs). Skip the game
            # branch entirely so the message falls through to the brain
            # as conversation — the gate inside _control_math alone
            # would return None and the dispatch would send nothing.
            _math_dm_outsider = (
                command is not None
                and command.kind == "math"
                and not self._is_operator(message)
                and message.chat.kind == ChatKind.DM)
            if (command is not None
                    and (command.kind == "game" or command.kind in GAME_COMMANDS)
                    and not _math_dm_outsider):
                # Per-chat games toggle: games-off / games-disabled chats
                # refuse game commands honestly — never silently dropped,
                # never run. /chatmode (the control plane) is not a game
                # command, so the owner can always re-enable from here.
                if _chat_modes_is_games_disabled(message.chat.key):
                    self._bump("controls")
                    try:
                        self.gateway.send(
                            message.chat.platform, message.chat,
                            _chat_modes_disabled_reply("games"))
                    except Exception as e:
                        record_failure("runtime.PartnerRuntime._process_inner:1110", e)
                    return
                # Stacking never includes games (interactive, stateful):
                # a stack starting with a game is rejected explicitly
                # instead of running its first game and silently
                # dropping the rest. Stacks with a game in a later
                # step are rejected the same way by handle_control.
                _stacked = split_command_stack(message.text)
                if _stacked is not None and len(_stacked) > 1:
                    self._bump("controls")
                    try:
                        self.gateway.send(
                            message.chat.platform, message.chat,
                            "stacks can't include games — send one game "
                            "command per message. Nothing ran.")
                    except Exception as e:
                        record_failure("runtime.PartnerRuntime._process_inner:1126", e)
                    return
                # Capability grant, stated explicitly: games are social,
                # so any caller may run them (caller_may_run returns True
                # for every game kind). The check documents the grant
                # instead of leaving it implicit in the dispatch below.
                if not caller_may_run(command.kind,
                                      self._is_operator(message)):
                    self._bump("controls")
                    _log.info("game command /%s denied for %s in %s",
                              command.kind, message.sender or "?",
                              message.chat.key)
                    try:
                        self.gateway.send(message.chat.platform, message.chat,
                                          "that one's just for the owner.")
                    except Exception as e:
                        record_failure("runtime.PartnerRuntime._process_inner:1142", e)
                    return
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
                            kind=message.chat.kind,
                            is_owner=self._is_operator(message))
                    _log.debug("game command reply: %r",
                               str(getattr(reply, "text", reply))[:100]
                               if reply else "")
                except Exception as exc:  # noqa: BLE001
                    _log.exception("game command failed: %s", exc)
                    reply = (f"that move broke something on my end: {exc} — "
                             f"try /help {command.kind} for the usage, "
                             f"or /game quit and start it fresh")
                if reply:
                    try:
                        result = self._send_control_reply(message, reply)
                        if result is not None and not result.ok:
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
            # + the PUBLIC command list only. A public command here goes
            # through handle_control (which re-checks the caller's grant);
            # a known non-public slash gets a friendly redirect instead of
            # running — owner/system controls live on the userbot.
            # Unknown slashes keep falling through to her as ordinary
            # text, as before.
            if message.chat.platform == "telegram-bot" and command is not None:
                from ...social.chat.control import (
                    GROUP_DECISION_COMMANDS,
                    PUBLIC_COMMANDS,
                )

                # Bare /start and bare /menu are menu triggers (the menu
                # text is public) — bare by the PARSED command, so the
                # addressed form Telegram's command menu inserts
                # (/menu@Nomorals_bot) is the same trigger when the
                # addressee is THIS surface (a foreign addressee never
                # reaches here: the stand-down arbitration already
                # stood this copy down). /menu <group> and /start
                # <platform> carry a tail, so they are NOT bare and
                # fall to the redirect below — list_catalog would leak
                # the owner-only catalog.
                bare_menu = (
                    command.kind in ("menu", "start")
                    and not command.tail
                    and (not command.addressee
                         or self._surface_usernames().get(
                             command.addressee)
                         == message.chat.platform))
                if bare_menu or command.kind in PUBLIC_COMMANDS \
                        or (message.chat.kind == ChatKind.GROUP
                            and command.kind
                            in GROUP_DECISION_COMMANDS):
                    self._bump("controls")
                    _log.info("public-bot command /%s from %s in %s",
                              command.kind, message.sender or "?",
                              message.chat.key)
                    try:
                        reply = self.handle_control(
                            message.text, message.chat.key, message=message)
                    except Exception as exc:  # noqa: BLE001
                        _log.exception("public-bot command failed: %s", exc)
                        reply = "that broke something on my end — try again."
                    if reply:
                        try:
                            # The persistent catalog reply-keyboard
                            # grid is retired (2026-11-06 menufix):
                            # the inline menu is the action surface.
                            # Bare-menu replies clear any grid a
                            # client still holds — once per chat.
                            if bare_menu:
                                self._clear_menu_grid_once(message)
                            self._send_control_reply(message, reply)
                        except Exception:  # noqa: BLE001
                            _log.warning("public-bot reply send failed")
                    return
                self._bump("controls")
                _log.info("public-bot gate: blocked /%s from %s in %s",
                          command.kind, message.sender or "?",
                          message.chat.key)
                # The gate itself stands (owner ruling,
                # 2026-10-20: no operator power on the public
                # bot, the owner included) — but when the caller
                # IS verifiably the owner (sender-id allowlist,
                # an identity check only; it grants nothing here),
                # the stranger text ("try /game list") reads as
                # "that command doesn't exist", which is how
                # /statusreact reported dead (truth pass,
                # 2026-11-07).  He gets told where the command
                # actually lives instead.
                refusal = _PUBLIC_BOT_NOT_AVAILABLE
                if str(getattr(message, "sender_id", "") or "") \
                        in getattr(self, "_owner_sender_ids", set()):
                    refusal = (
                        f"that one's yours, but it doesn't run "
                        f"here — this is my public bot, and "
                        f"personal controls stay off it. "
                        f"/{command.kind} answers in my private "
                        f"chats with you: your personal bot chat, "
                        f"your userbot DM, or your WhatsApp "
                        f"self-chat.")
                try:
                    self._typing_for(message.chat, refusal)
                    self.gateway.send(message.chat.platform, message.chat,
                                      refusal)
                except Exception:  # noqa: BLE001
                    _log.warning("public-bot gate reply failed")
                return
            # Claimed once, groups on every surface (group build P0):
            # the public-bot branch above claims recognized commands on
            # telegram-bot. A GROUP on any other surface (the userbot's
            # groups, WhatsApp, Discord) reaches here with a recognized
            # non-game command from a non-operator still unclaimed —
            # and would otherwise fall through to the persona and get a
            # conversational reply, the double-claim the owner banned.
            # Claim it the same way: a public command runs
            # (handle_control re-checks the grant), an owner-only kind
            # gets one honest redirect. Operators skip this — their own
            # control branch below owns their commands. DMs keep the
            # standing behavior; channels are their own posture.
            if (command is not None
                    and message.chat.kind == ChatKind.GROUP
                    and not self._is_operator(message)):
                from ...social.chat.control import (
                    GROUP_DECISION_COMMANDS,
                    PUBLIC_COMMANDS,
                )

                if (command.kind in PUBLIC_COMMANDS
                        or command.kind in GROUP_DECISION_COMMANDS
                        or command.kind == "error"):
                    self._bump("controls")
                    _log.info("group command /%s from %s in %s",
                              command.kind, message.sender or "?",
                              message.chat.key)
                    try:
                        reply = self.handle_control(
                            message.text, message.chat.key, message=message)
                    except Exception as exc:  # noqa: BLE001
                        _log.exception("group command failed: %s", exc)
                        reply = "that broke something on my end — try again."
                    if reply:
                        try:
                            self._send_control_reply(message, reply)
                        except Exception:  # noqa: BLE001
                            _log.warning("group command reply send failed")
                    return
                self._bump("controls")
                _log.info("group gate: blocked /%s from %s in %s",
                          command.kind, message.sender or "?",
                          message.chat.key)
                try:
                    self.gateway.send(
                        message.chat.platform, message.chat,
                        "that one's owner-only — I run it in the owner's "
                        "own chats. Games and the public list work here: "
                        "/game list shows what's on.")
                except Exception:  # noqa: BLE001
                    _log.warning("group gate reply failed")
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
            except Exception as e:
                record_failure("runtime.PartnerRuntime._process_inner:1294", e)
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

            _parsed = parse_control(message.text)
            # One command, one handling, across surfaces: an own
            # echo can be another surface's command (the owner's
            # /play in the BOT's DM arrives here through the
            # userbot and used to mint a second job). Standing
            # down is silent — the owning surface answers.
            if _parsed is not None and self._command_stands_down(
                    message, _parsed):
                self._bump("controls")
                _log.info(
                    "command /%s in %s stands down on %s — another "
                    "surface owns it", _parsed.kind, message.chat.key,
                    message.chat.platform)
                return
            if _parsed is not None:
                self._bump("controls")
                # Feed the arena's interest profiler: what the owner runs
                # most shapes future topic picks. Never raises.
                try:
                    from ..arena import activity as _arena_activity

                    verb = message.text.strip().split()[0].lstrip("/").lower()
                    _arena_activity.record(self.context.db, "command", verb)
                except Exception as e:
                    record_failure("runtime.PartnerRuntime._process_inner:1318", e)
                # The kind is parsed HERE, in this scope: the
                # exception path below names it, and it used to read
                # a stale `command` left over from an earlier scope
                # (a NameError waiting inside the error handler).
                _cmd = parse_control(message.text)
                _kind = _cmd.kind if _cmd is not None else "command"
                try:
                    reply = self.handle_control(message.text, message.chat.key,
                                                message=message)
                except Exception as exc:  # noqa: BLE001
                    _log.exception("control command failed: %s", exc)
                    try:
                        from ...core.error_doctor import diagnose as _diagnose
                        _diag = _diagnose(exc, context={"command": f"/{_kind}"})
                        _log.info("error diagnosis for /%s: %s",
                                  _kind, _diag["summary"])
                        _why = _diag["summary"]
                    except Exception as e:
                        record_failure("runtime.PartnerRuntime._process_inner:1331", e)
                        _why = f"{type(exc).__name__}: {exc}"
                    reply = (f"/{_kind} blew up on my end: {_why} — "
                             f"/help {_kind} shows the right usage")
                if reply:
                    try:
                        # Bare /start and /menu carry the bottom menu
                        # keyboard on operator chats too (any platform).
                        # Retired 2026-11-06 (menufix): the grid is
                        # abolished — the inline menu is the action
                        # surface; bare-menu replies clear any grid a
                        # client still holds, once per chat.
                        if message.text.strip() in ("/start", "/menu"):
                            self._clear_menu_grid_once(message)
                        self._send_control_reply(message, reply)
                    except Exception as exc:  # noqa: BLE001
                        _log.warning("control reply send failed: %s", exc)
                return
        # Build-map #43 — the explicit-override rule (user's standing
        # directive 2026-10-07): the owner's explicit "post this" / "send
        # it" / "post at 6" / "publish now" / "reply now" means EXECUTE
        # immediately. No re-confirmation, no hesitation, no policy text.
        # Owner-DM/operator only; never fires on other people's messages.
        if (message.incoming and self._is_operator(message)
                and not message.text.strip().startswith("/")):
            try:
                override_reply = self._explicit_post_override(message)
            except Exception:  # noqa: BLE001 - the override must never eat chat
                _log.exception("explicit-post override failed")
                override_reply = None
            if override_reply is not None:
                self._bump("controls")
                try:
                    self._typing_for(message.chat, override_reply)
                    self.gateway.send(message.chat.platform, message.chat,
                                      override_reply)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("explicit-post reply send failed: %s", exc)
                return
        # Build-map #49: conversational money — "send 5k to Mama" is a
        # first-class chat primitive (owner only). Guard warnings surface
        # here with [Send anyway] [Cancel]; biometric gates the movement.
        if (message.incoming and self._is_operator(message)
                and not message.text.strip().startswith("/")):
            try:
                money_reply = self._money_send_hook(message)
            except Exception:  # noqa: BLE001 - the hook must never eat chat
                _log.exception("money-send hook failed")
                money_reply = None
            if money_reply is not None:
                self._bump("controls")
                try:
                    self._typing_for(message.chat, money_reply)
                    self.gateway.send(message.chat.platform, message.chat,
                                      money_reply)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("money-send reply send failed: %s", exc)
                return
        # Build-map #58: "make me a song about X" → full vocal pipeline
        # (bed + DiffSinger + RVC + master). Owner only, non-slash.
        if (message.incoming and self._is_operator(message)
                and not message.text.strip().startswith("/")):
            try:
                song_reply = self._music_full_hook(message)
            except Exception:  # noqa: BLE001 - the hook must never eat chat
                _log.exception("music-full hook failed")
                song_reply = None
            if song_reply is not None:
                self._bump("controls")
                try:
                    self._typing_for(message.chat, song_reply)
                    self.gateway.send(message.chat.platform, message.chat,
                                      song_reply)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("music-full reply send failed: %s", exc)
                return
        # Build-map #71: "track this flight for me" → price watcher.
        # Owner only, non-slash.
        if (message.incoming and self._is_operator(message)
                and not message.text.strip().startswith("/")):
            try:
                track_reply = self._track_hook(message)
            except Exception:  # noqa: BLE001 - the hook must never eat chat
                _log.exception("track hook failed")
                track_reply = None
            if track_reply is not None:
                self._bump("controls")
                try:
                    self._typing_for(message.chat, track_reply)
                    self.gateway.send(message.chat.platform, message.chat,
                                      track_reply)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("track reply send failed: %s", exc)
                return
            # Build-map #72: forwarded booking confirmation → itinerary.
            try:
                trip_reply = self._trip_hook(message)
            except Exception:  # noqa: BLE001 - the hook must never eat chat
                _log.exception("trip hook failed")
                trip_reply = None
            if trip_reply is not None:
                self._bump("controls")
                try:
                    self._typing_for(message.chat, trip_reply)
                    self.gateway.send(message.chat.platform, message.chat,
                                      trip_reply)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("trip reply send failed: %s", exc)
                return
            # Producer NL: "make me something for the gym" -> compose;
            # "I like this" / "too slow" right after a production -> feedback.
            # Owner only, non-slash, never eats chat.
            try:
                produce_reply = self._produce_hook(message)
            except Exception:  # noqa: BLE001 - the hook must never eat chat
                _log.exception("produce hook failed")
                produce_reply = None
            if produce_reply is not None:
                self._bump("controls")
                try:
                    self._typing_for(message.chat, produce_reply)
                    self.gateway.send(message.chat.platform, message.chat,
                                      produce_reply)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("produce reply send failed: %s", exc)
                return
            # /play pick-list: a bare number reply ("3") right after a
            # vague /play resolves the pending pick-list choice and
            # downloads that track — old-school WhatsApp-bot style.
            # Owner only, non-slash, never eats chat.
            try:
                pick_reply = self._button_pick_hook(message)
            except Exception:  # noqa: BLE001 - the hook must never eat chat
                _log.exception("button-pick hook failed")
                pick_reply = None
            if pick_reply is None:
                try:
                    pick_reply = self._play_pick_hook(message)
                except Exception:  # noqa: BLE001 - the hook must never eat chat
                    _log.exception("play-pick hook failed")
                    pick_reply = None
            if pick_reply is not None:
                self._bump("controls")
                try:
                    self._typing_for(message.chat, pick_reply)
                    self.gateway.send(message.chat.platform, message.chat,
                                      pick_reply)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("play-pick reply send failed: %s", exc)
                return
        # Conversational exactly-once (reply claims, 2026-11-06):
        # everything above this seam has OBSERVED the message
        # (housekeeping, personal-surface hooks, persistence,
        # command arbitration); everything below GENERATES a
        # conversational reply (Core Mind, the brain, the delayed
        # path). One platform message identity gets one turn at
        # that generation: a redelivery — bridge replay inside the
        # stale window, adapter seen-set lost to a restart — is
        # held here instead of answering twice. (The owner's
        # Telegram, 2026-11-06: the same photo processed twice
        # sent the same line twice in one minute.) The delayed
        # reply is part of the same claimed turn and never
        # re-enters this seam.
        if self._reply_already_claimed(message):
            self._bump("reply_claim_held")
            _log.info(
                "reply claim: redelivered message held in %s "
                "(id=%s) — already answered",
                message.chat.key,
                str(getattr(message, "message_id", "") or ""))
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
                self._send_mind_reply(message, mind_reply)
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
        self._send_reply(message, outcome.parts,
                         degraded=outcome.degraded,
                         degraded_note=outcome.degraded_note)
        # Two-tier memory (build-map #36): opt-in parallel layer. Runs AFTER
        # the reply is sent and AFTER all existing memory work — it never
        # replaces or precedes the trusted recall path.
        self._two_tier_observe(message, outcome)

    def _two_tier_observe(self, message: Any, outcome: Any) -> None:
        """Feed one exchange to the two-tier memory layer. Opt-in via
        ``memory.two_tier_enabled`` (default off). Never raises — a memory
        failure must never break chat."""
        try:
            mem_settings = getattr(self.settings, "memory", None)
            if not getattr(mem_settings, "two_tier_enabled", False):
                return
            if not getattr(message, "incoming", False):
                return
            user_text = (getattr(message, "text", "") or "").strip()
            parts = getattr(outcome, "parts", None) or []
            reply_text = "\n".join(str(p) for p in parts).strip()
            if not user_text and not reply_text:
                return
            from ...memory.tiers import TwoTierMemory
            if self._two_tier is None:
                # BM25 recall spine: the documents/ types are injected
                # into HybridMemoryIndex (memory L3 never imports
                # documents L4); see nomorals/memory/hybrid.py.
                from ...documents.index import DocumentIndex
                from ...documents.model import Document, Section
                from ...memory.hybrid import HybridMemoryIndex
                self._two_tier = TwoTierMemory(
                    settings=self.settings, llm_fn=self._two_tier_llm,
                    hybrid_index_factory=lambda: HybridMemoryIndex(
                        index_factory=DocumentIndex,
                        document_factory=Document,
                        section_factory=Section))
            exchange = f"user: {user_text}\ndevon: {reply_text}"
            self._two_tier.observe(exchange, tags=f"chat:{message.chat.platform}")
        except Exception:  # noqa: BLE001 — memory never breaks chat
            _log.exception("two-tier observe failed")

    def _two_tier_llm(self, prompt: str) -> str:
        """Synchronous LLM call for the Muninn extraction pass."""
        from ...llm.base import Message, SamplingParams
        response = brain_for(self.context).chat(
            [Message.system("Output JSON only. No prose around it."),
             Message.user(prompt)],
            SamplingParams(temperature=0.2, max_tokens=500),
        task_kind="chat")
        if not response.ok or not response.text:
            raise RuntimeError(response.error or "empty LLM response")
        return response.text

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
                    outcome = self.brain.deliver_reply(message)
                    if outcome.parts:
                        self._bump("replies")
                        self._send_reply(
                            message, outcome.parts,
                            degraded=outcome.degraded,
                            degraded_note=outcome.degraded_note)
                except Exception:  # noqa: BLE001 - a late reply must not crash the thread
                    _log.exception("delayed reply for %s failed", message.chat.key)
            finally:
                self._release_db_thread()

        threading.Thread(target=_job, name=f"delayed-reply-{message.chat.key}", daemon=True).start()

    def _synthesize_voice_part(self, text: str) -> str:
        """Synthesize one voice-reply part → path to an .ogg voice note.

        Routes through the voice degradation ladder when the error
        system is wired: rung 0 is the real TTS engine (injected at
        the composition root), rung 1 is the silent rung — a TTS
        failure raises :class:`_VoiceUnavailable` carrying the ladder's
        honesty clause, so the text fallback in :meth:`_send_voice_reply`
        can say WHY the reply is text. Without a wired ladder the
        direct synthesis call runs, exactly as before.
        """
        from ...voice.pingpong import synthesize_voice_reply

        ladder = None
        es = getattr(self.context, "error_system", None)
        if es is not None:
            try:
                ladder = es.ladders.get("voice")
            except Exception as e:
                record_failure("runtime.PartnerRuntime._synthesize_voice_part", e)
                ladder = None
        if ladder is None:
            return synthesize_voice_reply(self.context, text)
        result, meta = ladder.call(text, context=self.context)
        if isinstance(result, dict) and result.get("ok") \
                and result.get("path"):
            return str(result["path"])
        note = ""
        if isinstance(meta, dict):
            note = str(meta.get("honesty") or "")
        if isinstance(result, dict):
            note = note or str(result.get("note") or "")
        raise _VoiceUnavailable(note)

    def _send_voice_reply(self, message: ChatMessage,
                            parts: list[str]) -> bool:
        """Reply with voice note(s).  True when at least one went out.

        Shows the "recording voice" indicator (not "typing") for the
        actual length of the note, and sends it as a voice bubble.

        If TTS/ffmpeg throws part-way through, the failed part and every
        part after it fall back to TEXT — a mid-reply voice failure must
        never silently drop the rest of the reply.
        """
        from ...voice.pingpong import ogg_duration
        from ...social.chat.voice_transport import send_voice_note
        sent_any = False
        part_list = list(parts or [])
        for idx, part in enumerate(part_list):
            text = (part or "").strip()
            if not text:
                continue
            try:
                ogg_path = self._synthesize_voice_part(text)
                # recording indicator matches the note's real length —
                # unless this chat has the typing simulation off (owner
                # order: userbot owner DM + personal bot send at once).
                if self._typing_enabled(message.chat):
                    try:
                        secs = max(1.0, min(60.0, ogg_duration(ogg_path)))
                        self.gateway.typing(message.chat.platform, message.chat,
                                            seconds=secs, action="recording")
                    except Exception as e:
                        record_failure("runtime.PartnerRuntime._send_voice_reply", e)
                adapter = self.gateway._adapter_for(message.chat.platform)
                if adapter is None or self.gateway.dry_run:
                    return sent_any
                # The shared voice transport owns the send: it converts,
                # uses the surface's native voice type, and reports a
                # voice note only when a real bubble landed (a rejected
                # bubble degrades to a labelled file inside the
                # transport — still "went out", but never AS a voice
                # note in the log or anywhere else).
                delivery = send_voice_note(adapter, message.chat, ogg_path)
                if delivery.ok:
                    sent_any = True
                    if delivery.voice_note:
                        _log.info("voice reply sent in %s", message.chat.key)
                    else:
                        _log.warning(
                            "voice reply for %s went out as a file, "
                            "not a voice note: %s",
                            message.chat.key, delivery.reason)
                else:
                    # The transport delivered nothing for this part —
                    # treat it like a synthesis failure (the except
                    # below): this part and everything after it fall
                    # back to text instead of vanishing silently.
                    raise RuntimeError(
                        f"voice send failed: {delivery.reason}")
            except Exception as exc:  # noqa: BLE001 - fall back to text
                _log.warning("voice reply failed, falling back to text: %s",
                             exc)
                # When the degradation came through the voice ladder,
                # its honesty clause rides on the first text part — the
                # chat is told the reply is text because voice is down,
                # not left to guess.
                honesty_note = str(getattr(exc, "honesty_note", "") or "")
                note_pending = bool(honesty_note)
                for rest in part_list[idx:]:
                    rest_text = (rest or "").strip()
                    if not rest_text:
                        continue
                    if note_pending:
                        rest_text = f"{honesty_note}\n{rest_text}"
                        note_pending = False
                    try:
                        tts_result = self.gateway.send(
                            message.chat.platform, message.chat, rest_text)
                        if tts_result.ok:
                            sent_any = True
                    except Exception:  # noqa: BLE001 - degraded but not silent
                        _log.exception("voice-fallback text send failed")
                return sent_any
        return sent_any

    def _get_brain_gate(self) -> BrainGate:
        """The competence-gate state, built on first use.

        ``__init__`` sets it; the getattr covers runtimes assembled
        without ``__init__`` (test fixtures that hand-wire a brain).
        """
        gate = getattr(self, "_brain_gate", None)
        if gate is None:
            gate = BrainGate()
            self._brain_gate = gate
        return gate

    def _send_reply(self, message: ChatMessage, parts: list[str], *,
                    degraded: bool = False,
                    degraded_note: str = "") -> None:
        # Competence gate (brain_gate): a PERSONAL-surface reply the
        # degraded brain produced from its template bank — no model
        # rung answered — is never sent AS the owner. The inbound
        # turn was already observed and persisted upstream; only the
        # reply is held, journaled once per degraded episode, and
        # replies resume automatically on the first healthy send.
        # Control/command replies never cross this seam. Bot-surface
        # replies are out of scope (theirs is degraded liveness, not
        # impersonation) and pass regardless of brain state.
        gate = self._get_brain_gate()
        # Every observed outcome feeds the gate's degraded state —
        # unsolicited-send paths (Notifier pushes, greetings) consult
        # it before speaking as the owner between live turns.
        gate.observe_outcome(degraded)
        if degraded and is_personal_platform(message.chat.platform):
            gate.note_hold(platform=message.chat.platform,
                           chat_key=message.chat.key,
                           note=degraded_note)
            return
        if not degraded:
            gate.note_healthy()
        # Deliberation hold (reply_guard, 2026-11-06): the third
        # check of this one gate. A healthy model can still leak its
        # raw deliberation as the "reply" — reasoning inlined into
        # the content channel, untagged, narrating the conversation
        # ("the user seems to be continuing… let me re-read…") — and
        # neither the competence gate (the brain WAS competent) nor
        # the scaffold guard (no scaffold furniture) stops it. On
        # personal surfaces, where Devon speaks AS the owner, a
        # deliberation-shaped part is HELD: journaled, counted, and
        # never sent — and a turn whose every part is held resolves
        # to silence, never a substitute line. Bot surfaces are out
        # of scope (her own voice, her own chats) and control/command
        # replies never cross this seam.
        if is_personal_platform(message.chat.platform):
            parts, held = guard_deliberation_parts(
                parts, platform=message.chat.platform,
                chat_key=message.chat.key)
            if held:
                self._bump("deliberation_held", held)
            if not parts:
                return
        # Send-boundary scaffold guard (reply_guard): the model's raw
        # analysis of the incoming message is not a reply and must
        # never reach a chat — least of all a personal-account surface
        # where she posts AS the owner. Runs before EVERY outbound
        # branch below (voice included: a scaffold must not be spoken
        # either). Clean parts pass untouched; a fully-scaffolded turn
        # resolves to one in-character reaction, journaled in the
        # partner failure ledger.
        parts = guard_reply_parts(
            parts, kind=message.chat.kind, rng=self.brain.presence_rng)
        if not parts:
            return
        # Voice-note ping-pong (outbound): the user spoke, so Devon answers
        # in voice.  Best-effort — any TTS/ffmpeg failure falls back to the
        # normal text path below. Per-chat voice toggle wins: with voice
        # off in this chat, replies stay text.
        if message.meta.get("voice_in"):
            from ..features import feature_enabled as _fe
            if (_fe(self.context, "voice")
                    and not _chat_modes_is_feature_disabled(
                        message.chat.key, "voice")
                    and self._send_voice_reply(message, parts)):
                return
        # Voice conversation mode: chat explicitly set to voice — replies go
        # out as voice notes even when the input was typed. Per-chat voice
        # toggle wins over the mode: voice off means text replies.
        try:
            from ...voice.conversation import voice_mode_on
            if (voice_mode_on(message.chat.key)
                    and not _chat_modes_is_feature_disabled(
                        message.chat.key, "voice")):
                if self._send_voice_reply(message, parts):
                    return
        except Exception as e:
            record_failure("runtime.PartnerRuntime._send_reply", e)
        partner_cfg = self.settings.partner
        values = self.brain.mood.current().values
        # Typing indicator on every kind of chat, per part: each chunk of a
        # split reply gets its own realistic, length-scaled typing run —
        # except chats where the owner ordered the simulation off
        # (userbot owner DM + personal bot: send at once, no delay).
        typing_on = (
            self._typing_enabled(message.chat)
            and (message.chat.kind != ChatKind.GROUP
                 or partner_cfg.typing_in_groups)
        )
        _log.info("_send_reply: %d part(s), typing_on=%s, chat=%s", 
                  len(parts), typing_on, message.chat.key)
        try:
            for i, part in enumerate(parts):
                if i and typing_on:
                    # Between parts: a breath, a re-think, thumbs back on the
                    # keys. The configured delay is a base, not a metronome.
                    # Same simulation as the indicator — off where the
                    # owner ordered it off, so parts land back-to-back.
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
                # Conversational replies are NEVER stamped or
                # otherwise mutated at this boundary: the personal-
                # surface constitution (brain-gate + reply-guard
                # pins, post-2026-11-04 harm) is byte-identical
                # passage — Devon speaking as the owner in his own
                # chats carries no brand, on any surface.  The
                # always-present stamp (owner law, 2026-11-05)
                # governs SYSTEM/PRESENTED output: control replies
                # (_send_control_reply), scheduled reporters and
                # presented cards carry it; free conversation does
                # not.
                send_text = part
                result = self.gateway.send(
                    message.chat.platform, message.chat, send_text,
                    reply_to=message.reply_to if message.chat.kind == ChatKind.GROUP else "",
                    # A live turn's reply, first dispatch attempt:
                    # send-once guards must never swallow it.
                    fresh=True,
                )
                _log.info("_send_reply: send part %d/%d, ok=%s", i+1, len(parts), result.ok)
                if not result.ok:
                    _log.warning("send failed on %s: %s", message.chat.platform, result.error)
        except Exception as exc:  # noqa: BLE001
            _log.exception("send loop failed: %s", exc)

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> list[str]:
        if not self.dry_run:
            # Boot mission resumption kicks off BEFORE the gateway
            # begins dispatch: its claims must land before any
            # /mission status read can reconcile-flip a crashed
            # mission to FAILED (after which resume is a no-op). The
            # sweep runs on a background thread — startup and chat
            # readiness are never blocked by it.
            try:
                self._start_boot_mission_resume()
            except Exception as exc:  # noqa: BLE001 - boot must not fail here
                _log.warning("boot mission resume not started: %s", exc)
        # Self-provisioned credentials: bridge vault keys into the env
        # before anything can fail for a missing one. Best-effort — a
        # locked vault skips the bridge, never the boot.
        self._bridge_provisioned_credentials()
        started = self.gateway.start(self.on_message)
        # WhatsApp cold-start attach (§15): gateway start builds
        # the whatsapp adapter once, against the static config
        # endpoint, and drops it when no bridge is listening yet —
        # an account paired in a PREVIOUS run (registry: linked,
        # its bridge relaunched by the supervisor) would stay
        # conversationally dark until some inbound message drove
        # the housekeeping beat.  Reconcile once here, now that
        # the gateway is live and its builders exist: a qualifying
        # account raises its surface with zero inbound messages.
        # Failure-isolated — the reconcile notes a failure on the
        # account record itself (visible in /pair), and a raise
        # here is recorded, never a broken boot.
        if not self.dry_run:
            try:
                self._sync_whatsapp_surface()
            except Exception as e:  # noqa: BLE001 - attach is
                record_failure(           # never load-bearing
                    "runtime.PartnerRuntime.start:wa_attach", e)
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
        # Arena: the autonomous research/build loop. ON by default
        # (NM_ARENA_ENABLED=0 turns it off); each cycle re-checks the
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

                # Resource-aware scheduling: heavy jobs defer (never drop)
                # while the machine is under pressure.  Best-effort — the
                # scheduler runs ungated when no advisor is available.
                _resources = None
                if self._resource_advisor is not None:
                    try:
                        _resources = self._resource_advisor()
                    except Exception as e:
                        record_failure("runtime.PartnerRuntime.start", e)
                        _log.debug("no resource advisor for scheduler",
                                   exc_info=True)
                self._scheduler = Scheduler(
                    self.context,
                    gateway=self.gateway,
                    tick_seconds=getattr(sched_settings, "tick_seconds", 60.0),
                    max_concurrent=getattr(sched_settings, "max_concurrent", 2),
                    wall_seconds=getattr(sched_settings, "wall_seconds", 300.0),
                    resources=_resources,
                )
                if sched_settings.enabled:
                    self._scheduler.start()
                # expose to agents: the cognitive loop's cost-aware cadence
                # (wave 63) reschedules its own job through this handle
                self.context.extras["scheduler"] = self._scheduler
                # Closed-loop self-improvement tick: ON by default
                # (improvement.auto_tick, mode != "off"); run a cycle on a
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
                                routine=True,
                            )
                    except Exception as exc:  # noqa: BLE001 - optional
                        _log.warning(
                            "improvement tick job not registered: %s", exc)
                # Mining session: daily conversation→training-data mining.
                # Idempotent — one durable job by name. Mines new chat
                # pairs into scored bundles for nm train; manual runs
                # still work via /data mine.
                try:
                    have_mine = [j for j in self._scheduler.list_jobs()
                                 if j.get("name") == "mining session"]
                    if not have_mine:
                        self._scheduler.add(
                            "mining session", "daily 03:00", "tool",
                            {"tool": "train_mine", "args": {}},
                        )
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("mining session job not registered: %s", exc)
                # Autonomous organs: research + wisdom run their own ticks on
                # the durable scheduler. Idempotent by name; the tools do one
                # full organ cycle each. No commands needed — this is the
                # "system works together without tripping commands" wiring.
                # Research tick is light (due-check + gaps); wisdom tick is
                # heavy (fetch + parse + index).
                for _name, _spec, _tool, _heavy, _routine in (
                    ("research organ tick", "every 30m",
                     "research_organ_tick", False, True),
                    ("wisdom organ tick", "every 6h",
                     "wisdom_organ_tick", True, True),
                    # Memory consolidation: additive distillation on a
                    # schedule. She reviews and distills without being asked.
                    # Never deletes — consolidation is additive only.
                    ("memory consolidation", "every 12h",
                     "memory_consolidate", False, False),
                ):
                    try:
                        _have = [j for j in self._scheduler.list_jobs()
                                 if j.get("name") == _name]
                        if not _have:
                            self._scheduler.add(
                                _name, _spec, "tool",
                                {"tool": _tool, "args": {}},
                                heavy=_heavy, routine=_routine,
                            )
                    except Exception as exc:  # noqa: BLE001 - optional
                        _log.warning("%s job not registered: %s", _name, exc)
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
                # Briefing slots — morning pulse (news → briefing → voice,
                # daily 23:00 America/Denver = 06:00 WAT) plus afternoon
                # (markets + fresh news, 13:00), evening (day wrap +
                # tomorrow, 19:00) and midnight (overnight fresh news,
                # 00:00).  Idempotent by name; catch-up covers
                # slots whose time passed without delivery.
                try:
                    from ..morning_pulse import (
                        ensure_pulse_jobs, check_all_catchup)
                    ensure_pulse_jobs(self.context)
                    check_all_catchup(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("briefing-slot jobs not registered: %s",
                                 exc)
                # Finance: alert watchtower (every 15m, delivers via
                # Notifier) + weekly money digest (Sundays 09:00 local).
                # Idempotent by name; no-ops when no alerts/goals exist.
                try:
                    from ...finance.jobs import ensure_finance_jobs
                    ensure_finance_jobs(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("finance jobs not registered: %s", exc)
                # Social: the weekly content pipeline (Mondays 08:00 —
                # research -> drafts queued for review at the best
                # windows; never auto-posts). Idempotent by name.
                try:
                    from ...social.jobs import ensure_weekly_content_jobs
                    ensure_weekly_content_jobs(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("weekly content job not registered: %s",
                                 exc)
                # Price sources: the quote-provider health loop
                # (every 6h — one bounded real quote probe per
                # keyless provider; keyed providers never probed;
                # pass #91).
                try:
                    from ...finance.price_jobs import (
                        ensure_price_source_health_jobs)
                    ensure_price_source_health_jobs(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("price source health job not "
                                 "registered: %s", exc)
                # Trading system (Pass 6): the standing pipeline —
                # tick (default every 15m: read → setups → forecast
                # → judgment shadow → management poll) + audit cycle
                # (default every 6h: retrain + self-audit).  The
                # jobs call the trading tool; the pipeline's own
                # control state (start/stop, HALT switch) lives in
                # the trading journal and the tick honors it.
                # Idempotent by name.
                try:
                    from ...finance.trading_jobs import ensure_trading_jobs
                    ensure_trading_jobs(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("trading jobs not registered: %s", exc)
                # News: the continuous poll (every 30m — fetch, store,
                # interest-score, push the high tier).  The digests
                # ride the briefing slots above; this job is what
                # keeps the store current between them (pass #85).
                try:
                    from ..news_jobs import ensure_news_jobs
                    ensure_news_jobs(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("news jobs not registered: %s", exc)
                # Betting desk: study cycle (every 30m — collect, lab,
                # library) + signal evaluation (every 10m — timed
                # calls off lab-confirmed time structure only).
                # Idempotent by name; both routine (the desk's output
                # rides its own surfaces).
                try:
                    from ..betdesk.jobs import ensure_betdesk_jobs
                    ensure_betdesk_jobs(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("betdesk jobs not registered: %s", exc)
                # Opportunity hunt: the weekly paid-task scan
                # (Mondays 05:30 — ahead of the 07:00 briefing) that
                # vets the OpportunityHunter's finds for Nigerian
                # signup + payout and stocks the briefing's Paid
                # Tasks section.  Idempotent by name; routine (its
                # output rides the briefing).
                try:
                    from ..opportunity_jobs import ensure_opportunity_jobs
                    ensure_opportunity_jobs(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("opportunity hunt job not registered: "
                                 "%s", exc)
                # Pump desk: watch cycle (every 5m — intake, dossiers,
                # gate, calls) + study cycle (every 30m — grade calls,
                # dev profiles).  Idempotent by name; both routine
                # (calls ride their own surface).  The standing
                # firehose consumer starts best-effort alongside.
                try:
                    from ..pumpdesk.jobs import ensure_pumpdesk_jobs
                    ensure_pumpdesk_jobs(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("pumpdesk jobs not registered: %s", exc)
                try:
                    from ..pumpdesk import PumpDesk
                    PumpDesk(getattr(self, "context", None)) \
                        .ensure_stream()
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("pumpdesk stream not started: %s", exc)
                # Motivation class: randomized Lagos-anchored one-shots
                # (self-rescheduling inside the 05:00–06:00 WAT morning
                # window + the evening window) drawing on the Teachings
                # ledger; catch-up covers an overnight-off host.
                try:
                    from ..motivation import (
                        check_motivation_catchup,
                        ensure_motivation_jobs,
                    )
                    ensure_motivation_jobs(self.context)
                    check_motivation_catchup(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("motivation jobs not registered: %s", exc)
                # Download sources: the registry health loop (every
                # 6h — re-verify sources + discovery candidates,
                # promote the verified, shed the dead; pass #86).
                try:
                    from ...media.source_jobs import (
                        ensure_source_health_jobs)
                    ensure_source_health_jobs(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("download source health job not "
                                 "registered: %s", exc)
                # Story chapter discovery: one routine sweep job
                # (every 30m; 1h on termux) that only ASKS whether
                # followed stories gained chapters — the reading
                # engine below does the reading itself, immediately
                # on the discovery event.  Idempotent by name.
                try:
                    from ...books.jobs import ensure_story_jobs
                    ensure_story_jobs(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("story discovery job not "
                                 "registered: %s", exc)
                # Autonomous nervous system (nomorals/autonomy): the organs
                # work without commands. Idle monitor watches for quiet
                # periods; presence heartbeat notices/prepares every 30m.
                # Idempotent; safe to call on every boot.
                try:
                    from ...autonomy.idle import IdleMonitor, note_activity
                    from ...autonomy.presence import ensure_heartbeat_job
                    from ...autonomy.coordinator import IdleCoordinator
                    ws_dir = getattr(self.context.settings,
                                     "workspace_dir", ".")
                    self._idle_monitor = IdleMonitor(ws_dir)
                    self._idle_monitor.start()
                    self._idle_coordinator = IdleCoordinator(ws_dir)
                    self._idle_coordinator.subscribe()
                    ensure_heartbeat_job(self._scheduler)
                    # Seed the activity clock so we don't start "idle".
                    # Rides the same shared ledger handle as the
                    # per-message writers (see _autonomy_db).
                    _adb = self._autonomy_db()
                    if _adb is not None:
                        try:
                            note_activity(_adb)
                        except Exception as e:  # noqa: BLE001
                            self._autonomy_db_outage(
                                e, "runtime.PartnerRuntime.start:2149")
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("autonomy nervous system not started: %s",
                                 exc)
                # Agency layer (nomorals/agency): initiative backlog +
                # boot reconcile (lease recovery, missed-check coalesce,
                # idempotent watch re-registration). Best-effort: agency
                # must never break boot. Deliberation/execution run via
                # the existing scheduler facade + idle coordinator.
                try:
                    from ...agency.wiring import boot_agency
                    # context= routes the boot through the ONE agency
                    # composition root (F09): the runtime's systems
                    # get the same real gateway/search/organs the
                    # scheduled shift composes, not an organ-less twin.
                    self._agency = boot_agency(
                        self.context.db,
                        scheduler=getattr(self, "_scheduler", None),
                        context=self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("agency layer not started: %s", exc)
                    self._agency = None
                # Trigger engine (nomorals/triggers): the automation layer
                # that lets systems wake each other.  Message-source
                # triggers evaluate inbound chat via message_hook (wired
                # below through attach()); bus-source triggers fire on the
                # shared event bus (scheduler.job.finished, mission.terminal,
                # trigger.fired, …) through attach_bus() — that is the real
                # cross-system wiring.  schedule/file/price/webhook sources
                # run on the engine's own scheduler worker + poll thread.
                # Best-effort: automation must never break boot.
                try:
                    from ...triggers.engine import TriggerEngine, attach

                    _engine = TriggerEngine(
                        self.context.db, self.context,
                        resources=_resources,
                        send_message=self.say,
                    )
                    attach(_engine, self.context)
                    _engine.attach_bus()
                    _engine.start()
                    self._trigger_engine = _engine
                    _log.info("trigger engine started (bus-attached)")
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("trigger engine not started: %s", exc)
                    self._trigger_engine = None
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
                # Memory consolidation cadence: one durable
                # "memory-consolidation" job (every
                # consolidation_interval_seconds, default 1h) running the
                # *additive* consolidation tick.  Idempotent; the tick
                # itself never deletes anything.
                try:
                    from ...memory.cadence import ensure_consolidation_job
                    from ..scheduler import Scheduler

                    ensure_consolidation_job(
                        self.context, scheduler_factory=Scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("consolidation job not registered: %s", exc)
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
                            except Exception as e:
                                record_failure("runtime.PartnerRuntime.start:2232", e)
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
                                routine=True,
                            )
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("monitor tick job not registered: %s", exc)
                # Agent modes: the autonomous-modes engine gets its
                # proactive bridge (ProactiveEngine suggestions become
                # initiatives) + one durable "agent modes tick" job on
                # this same scheduler, so chats with autonomous mode
                # get cycles without anyone poking them.  Idempotent
                # by source name / job name; best-effort, never breaks
                # boot.  Under /mode suggest the bridge's initiatives
                # land in /proposals for approval; under /mode auto
                # they execute immediately (existing policy).
                try:
                    modes = self.agent_modes
                    if "proactive" not in \
                            modes.autonomous.source_names():
                        from ..modes import proactive_source
                        from ..proactive import ProactiveEngine

                        modes.autonomous.add_source(
                            "proactive",
                            proactive_source(ProactiveEngine(), "owner"))
                    from ..modes import ensure_agent_modes_tick_job

                    ensure_agent_modes_tick_job(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "agent modes tick not registered: %s", exc)
                # Immune adaptive gate: one durable "immune sampler
                # tick" job (every 1m) collecting real process
                # samples into the persisted adaptive baseline, so the
                # cancer gate learns from live behaviour and can
                # actually arm — before this wiring nothing recorded
                # samples at runtime and the baseline stayed empty
                # forever.  Idempotent; best-effort.
                try:
                    from ...immune.wiring import ensure_immune_sampler_job

                    ensure_immune_sampler_job(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "immune sampler job not registered: %s", exc)
                # Cognition failure loop: one durable "failure loop
                # tick" job (every 1h) clustering the recorded
                # failures into sandbox-checked, owner-gated fix
                # proposals filed on the FailureKB — before this
                # wiring the loop had no production caller and no
                # proposal was ever built.  Idempotent; best-effort.
                try:
                    from ...cognition.failure_loop import ensure_failure_loop_job

                    ensure_failure_loop_job(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "failure loop job not registered: %s", exc)
                # Games seasons (AA-16): one durable "season tick"
                # job (every 1h) rolling each group's quarterly
                # season over at the boundary and paying closed
                # seasons' prizes through the group economy — the
                # engine folds finished group games into the season
                # boards, but before this wiring nothing ever closed
                # a season or moved a prize.  Idempotent; best-effort.
                try:
                    from ...games.tournaments_seasons import ensure_season_tick_job

                    ensure_season_tick_job(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "season tick job not registered: %s", exc)
                # Community orchestration (AA-18): one durable
                # "community orchestration tick" job (every 1m)
                # heartbeating this device into the fleet session
                # registry and dispatching due scheduled group posts
                # through the router — before this wiring the device
                # Router had no production caller and scheduled posts
                # were never sent.  Idempotent; best-effort.
                try:
                    from ...community.orchestration import ensure_orchestration_job

                    ensure_orchestration_job(self._scheduler)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "orchestration job not registered: %s", exc)
            except Exception as exc:  # noqa: BLE001 - scheduler is optional
                _log.warning("scheduler failed to start: %s", exc)
                self._scheduler = None
        # Story reading engine: the standing consumer that turns a
        # follow (story.followed) and a chapter discovery
        # (story.chapters_discovered) into her actually reading the
        # story — fetch + bible digestion on its own worker, pinging
        # the chat that asked.  Independent of the scheduler (its
        # triggers are events, not ticks); the boot catch-up reads
        # whatever a follow left undigested while no engine ran.
        if not self.dry_run:
            try:
                from ...books.reading_engine import ensure_reading_engine

                _story_engine = ensure_reading_engine(
                    self.context, chat_notify=self._story_chat_notify)
                _story_engine.catch_up_all()
            except Exception as exc:  # noqa: BLE001 - optional
                _log.warning("story reading engine not started: %s", exc)
        # Supervision: watched threads get restarted on death, with incident
        # recording and restart budgets. The supervisor watches; components
        # keep owning their threads (no invasive lifecycle rewrites).
        try:
            es = getattr(self.context, "error_system", None)
            if es is not None and not self.dry_run:
                if self._scheduler is not None:
                    _sched = self._scheduler
                    es.supervisor_for("scheduler").watch(
                        "tick_loop",
                        is_alive=lambda: _sched.running(),
                        restart=lambda: _sched.start(),
                        check_interval_s=30.0,
                    )
                    _log.info("scheduler tick loop under supervision")
                if getattr(self, "_autonomy", None) is not None:
                    _auto = self._autonomy
                    es.supervisor_for("autonomy").watch(
                        "agent_loop",
                        is_alive=lambda: getattr(_auto, "_running", True),
                        restart=lambda: _auto.start(),
                        check_interval_s=60.0,
                    )
                if getattr(self, "_trigger_engine", None) is not None:
                    _eng = self._trigger_engine
                    es.supervisor_for("triggers").watch(
                        "engine",
                        is_alive=lambda: getattr(_eng, "_running", True),
                        restart=lambda: _eng.start(),
                        check_interval_s=60.0,
                    )
        except Exception as exc:  # noqa: BLE001 - supervision is optional
            _log.warning("worker supervision failed to start: %s", exc)
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
        # Self-update crash recovery: an update interrupted mid-run
        # (crash / power loss) leaves a journal with no terminal
        # event; replay it once here so a run that died at/after a
        # dangerous step is rolled back (state snapshot AND code
        # tree) before Devon serves on a possibly half-migrated
        # database paired with new code. A clean journal is a cheap
        # no-op (one journal read, no notice). Same guarantee class
        # as the trial recovery above: reported once when it acts,
        # and a failure inside recovery never blocks boot.
        if not self.dry_run:
            try:
                self._boot_update_receipt = \
                    self._recover_interrupted_update()
            except Exception as exc:  # noqa: BLE001 - recovery is optional
                _log.warning("update crash recovery failed: %s", exc)
        # Startup journey (doc 4, Living Chat P2): ONE readiness
        # card to the owner's surface, built from the facts this
        # boot actually established and revised from the durable
        # record — not a boot log dump. Best-effort like every
        # boot surface: a failure here never breaks the boot, and
        # an undelivered card stays persisted for delivery on the
        # first owner interaction.
        if not self.dry_run:
            try:
                self._publish_startup_card(started)
            except Exception as exc:  # noqa: BLE001 - surface, not boot
                _log.warning("startup card not published: %s", exc)
        return started

    def _recover_interrupted_update(
            self, *, repo_root: Any = None) -> dict[str, Any] | None:
        """Replay the self-update journal once at boot (pass 46).

        Builds the UpdateManager exactly as the chat ``/update``
        branch does (same SnapshotManager from the runtime's real
        settings, same repo-root derivation) and calls
        :meth:`UpdateManager.recover_interrupted`. On a clean journal
        this is the cheap no-op that method already is — one journal
        read, nothing restored, no owner notice. When the receipt
        says the recovery acted, the owner is told via
        :meth:`_report_update_recovery`.

        Returns the receipt, or None when the check itself could not
        run. NEVER raises — a recovery failure is logged and boot
        continues on the state as found.
        """
        try:
            from pathlib import Path

            from ...os.snapshots import SnapshotManager
            from ...os.update import UpdateManager

            settings = self.context.settings
            if repo_root is None:
                repo_root = Path(__file__).resolve().parents[3]
            snapshots = SnapshotManager(
                home=settings.home_path, db_path=settings.db_path,
                blob_dir=getattr(settings, "blob_dir", None))
            manager = UpdateManager(repo_root, snapshots)
            receipt = manager.recover_interrupted()
            # Settle the restart journey: if the previous process
            # restarted into an update, THIS process is the only
            # witness that can confirm it came up on the landed
            # version. Its verdict is reported on its own, whatever
            # the journal recovery above found.
            try:
                intent_receipt = manager.consume_restart_intent()
            except Exception:  # noqa: BLE001 - never blocks the boot
                _log.warning("update restart-intent check could not "
                             "run", exc_info=True)
                intent_receipt = None
            # The readiness facts keep the restart verdict beside
            # the journal receipt (an unconfirmed restart is a
            # recovery-required fact about THIS boot).
            self._boot_intent_receipt = intent_receipt
        except Exception as exc:  # noqa: BLE001 - boot must not fail here
            _log.warning("update crash recovery could not run: %s", exc)
            return None
        if intent_receipt is not None:
            self._report_update_recovery(intent_receipt)
            # Restart journey (doc 4): THIS process — never the
            # one that went down — writes the journey card's final
            # state, from the same receipt.
            try:
                self._finalize_restart_card(intent_receipt)
            except Exception as exc:  # noqa: BLE001 - never blocks boot
                _log.warning("restart card not finalized: %s", exc)
        else:
            # No restart intent: a PLAIN restart (or one whose
            # intent write failed) may still have left a journey
            # card open. This process closes it with the only
            # witness it has — its own existence — claiming no
            # version truth.
            try:
                self._finalize_restart_card({})
            except Exception as exc:  # noqa: BLE001 - never blocks boot
                _log.warning("restart card sweep failed: %s", exc)
        if not receipt.get("recovered"):
            return receipt
        self._report_update_recovery(receipt)
        return receipt

    def _report_update_recovery(self, receipt: dict[str, Any]) -> None:
        """Surface an acting update crash recovery to the owner.

        Log at warning level (error for ``rollback_failed`` — Devon
        is then booting on the unrecovered state) plus one durable
        Notifier publish, the same report-once mechanism the
        TrialFlow boot recovery beside this uses: the alert persists
        even if live delivery fails and is redelivered later. Never
        raises.
        """
        title, body = _update_recovery_notice(receipt)
        failed = receipt.get("action") in (
            "rollback_failed", "pending_restore_failed",
            "tree_restore_failed", "restart_unconfirmed")
        if failed:
            _log.error("update crash recovery FAILED: %s", body)
        else:
            _log.warning("update crash recovery acted: %s", body)
        try:
            from ..notifier import Notifier

            Notifier(self.context, gateway=self.gateway).publish(
                "update", title, body, force=True, critical=failed,
                alert_hints=({"failed": True, "subject": "self-update"}
                             if failed else
                             {"notice": True, "subject": "self-update"}))
        except Exception as exc:  # noqa: BLE001 - reporting is best-effort
            _log.warning("update recovery notice not delivered: %s", exc)

    # ── living cards (Living Chat P2) ───────────────────────────

    def _living_card_store(self) -> Any:
        """The durable living-card store over the context db
        (in-memory fallback when the context carries no db)."""
        store = getattr(self, "_living_card_store_obj", None)
        if store is None:
            from ...social.chat.living_cards import CardStore

            store = CardStore(getattr(self.context, "db", None))
            self._living_card_store_obj = store
        return store

    def _living_card_host(self) -> Any:
        """The card host wired to this runtime's real verbs: the
        gateway send for first delivery / superseding sends, and
        the live-session edit verb (the same one interactive
        flows edit through) for in-place updates."""
        host = getattr(self, "_living_card_host_obj", None)
        if host is None:
            from ...social.chat.live_session import edit_session_message
            from ...social.chat.living_cards import CardHost

            def _send(platform: str, chat: Any, text: str,
                      buttons: Any = None) -> tuple[bool, str]:
                try:
                    res = self.gateway.send(
                        platform, chat, text, buttons=buttons)
                except Exception:  # noqa: BLE001 - host degrades
                    return (False, "")
                if res is None or not getattr(res, "ok", False):
                    return (False, "")
                return (True, str(getattr(res, "message_id", "")
                                  or ""))

            def _edit(chat: Any, message_id: str, text: str,
                      buttons: Any = None) -> bool:
                adapter = self._chat_adapter(chat.platform)
                if adapter is None:
                    return False
                return edit_session_message(
                    adapter, chat, message_id, text, buttons=buttons)

            def _register_picks(chat_key: str, commands: list,
                                owner_key: str,
                                operation_id: str) -> str:
                # Living Chat P3 (P2 remainder #1): a card that
                # landed in the numbered form registers its
                # actions under its own identity at send time,
                # so a bare number resolves against the card —
                # whichever producer built it.
                return self._button_pick_store().register_operation(
                    chat_key, list(commands), owner_key=owner_key,
                    operation_id=operation_id, kind="living-card")

            host = CardHost(self._living_card_store(),
                            send=_send, edit=_edit,
                            register_picks=_register_picks)
            self._living_card_host_obj = host
        return host

    @staticmethod
    def _card_chat_ref(card: Any) -> Any:
        from ...social.chat.base import ChatRef

        return ChatRef(platform=card.platform, chat_id=card.chat_id)

    # ── startup journey ─────────────────────────────────────────

    def _readiness_facts(self, started: list[str]) -> dict[str, Any]:
        """The facts the startup readiness card reports — each
        gathered from the subsystem that owns it, at boot."""
        facts: dict[str, Any] = {
            "storage": getattr(self.context, "db", None) is not None,
            "recovery": None, "platforms": [], "brain": None,
            "scheduler": None, "agency": None, "missions": None}
        intent = getattr(self, "_boot_intent_receipt", None)
        journal = getattr(self, "_boot_update_receipt", None)
        if isinstance(intent, dict):
            facts["recovery"] = intent
        elif isinstance(journal, dict) and journal.get("recovered"):
            facts["recovery"] = journal
        try:
            known = list(self.gateway.known_platforms() or [])
        except Exception:  # noqa: BLE001 - facts never break boot
            known = []
        up = {str(s) for s in (started or [])}
        seen = set()
        for name in known:
            seen.add(str(name))
            facts["platforms"].append((
                str(name), "up" if str(name) in up else "down",
                "" if str(name) in up else "not started this boot"))
        for name in sorted(up - seen):
            facts["platforms"].append((name, "up", ""))
        try:
            from ..notifier import resolve_brain_gate

            gate = resolve_brain_gate(self.context)
            if gate is not None and gate.degraded_now():
                facts["brain"] = (
                    "degraded",
                    "the brain's last observed outcome was a "
                    "failure — personal surfaces stay observe-only "
                    "until it answers again")
            else:
                facts["brain"] = ("ready", "")
        except Exception:  # noqa: BLE001 - facts never break boot
            facts["brain"] = None
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is None:
            facts["scheduler"] = ("down", "no scheduler this boot")
        else:
            try:
                running = bool(scheduler.running())
            except Exception:  # noqa: BLE001
                running = False
            facts["scheduler"] = (
                ("running", "") if running else
                ("down", "scheduler not running"))
        agency = getattr(self, "_agency", None)
        if isinstance(agency, dict):
            facts["agency"] = agency.get("readiness")
        return facts

    def _publish_startup_card(self, started: list[str]) -> None:
        """Deliver the boot's ONE readiness surface to the owner
        chats (doc 4): persisted first (the retained record),
        then published through the living-card machinery.
        Never raises."""
        from ...social.chat.living_cards import build_readiness_card

        facts = self._readiness_facts(started)
        store = self._living_card_store()
        host = self._living_card_host()
        for key in _key_set(self.settings.partner.owner_chats):
            plat, _, cid = key.partition(":")
            if not cid:
                continue
            card = build_readiness_card(
                facts, chat_key=key, platform=plat, chat_id=cid)
            store.save(card)  # the retained record, before delivery
            host.publish(card, self._card_chat_ref(card))
            _log.info("startup card %s for %s: verdict %s, "
                      "delivered %s", card.card_id, key,
                      card.status, card.delivered)

    def _deliver_pending_startup_card(self, message: Any) -> None:
        """Show the startup record when the transport actually
        connects (doc 4): an owner chat's first interaction
        delivers any readiness card boot could not. Never
        raises (the caller failure-isolates it too)."""
        chat_key = str(getattr(message.chat, "key", "") or "")
        if not chat_key:
            return
        if chat_key not in _key_set(self.settings.partner.owner_chats):
            return
        store = self._living_card_store()
        pending = [c for c in store.live_for_chat(chat_key)
                   if c.family == "readiness" and not c.delivered]
        if not pending:
            return
        host = self._living_card_host()
        for card in pending:
            host.deliver(card, message.chat)

    # ── restart journey ─────────────────────────────────────────

    def _open_restart_card(self, message: Any, chat_key: str,
                           kind: str) -> Any:
        """Open the restart journey card in the requesting chat.
        Returns the card, or None when this turn has no chat
        surface to hold it. Never raises."""
        if message is None or getattr(message, "chat", None) is None:
            return None
        try:
            from ...social.chat.living_cards import build_restart_card

            chat = message.chat
            card = build_restart_card(
                chat_key=chat_key,
                owner_key=_pick_owner_key(message),
                platform=str(getattr(chat, "platform", "") or ""),
                chat_id=str(getattr(chat, "chat_id", "") or ""),
                kind=kind)
            card.state = {**card.state,
                          "boot_id": self._play_dl_boot_id()}
            self._living_card_store().save(card)
            self._living_card_host().publish(card, chat)
            return card
        except Exception as exc:  # noqa: BLE001 - journey is a surface
            _log.warning("restart card not opened: %s", exc)
            return None

    def _revise_restart_card(self, card: Any, message: Any,
                             **changes: Any) -> Any:
        """Deliver one journey revision to the card's chat.
        Never raises; returns the (possibly revised) card."""
        if card is None or message is None:
            return card
        try:
            return self._living_card_host().revise(
                card, message.chat, **changes)
        except Exception as exc:  # noqa: BLE001 - journey is a surface
            _log.warning("restart card revision failed: %s", exc)
            return card

    def _finalize_restart_card(self, receipt: dict[str, Any]) -> None:
        """Write the restart journey's final state — called by
        the NEW process with the restart-truth receipt (and, for
        a plain restart that left no receipt, by the boot sweep
        below with the process-up witness only)."""
        from ...social.chat.living_cards import (
            finalize_plain_restart_card,
            finalize_restart_card,
        )

        store = self._living_card_store()
        card = store.latest("restart")
        if card is None or card.terminal:
            return
        opened_boot = str((card.state.get("restart") or {})
                          .get("boot_id") or card.state.get(
                              "boot_id") or "")
        if opened_boot and opened_boot == self._play_dl_boot_id():
            return  # opened by THIS process; its journey is live
        if receipt:
            done = finalize_restart_card(card, receipt)
        else:
            done = finalize_plain_restart_card(
                card, boot_id=self._play_dl_boot_id())
        old = done.revision
        done.revision = old + 1
        if not store.save(done, expect_revision=old):
            return
        try:
            self._living_card_host().deliver(
                done, self._card_chat_ref(done))
        except Exception as exc:  # noqa: BLE001 - surface, not boot
            _log.warning("restart card final delivery failed: %s",
                         exc)

    # ── fenced card taps ────────────────────────────────────────

    def _living_card_tap(self, card_id: str, revision: int,
                         index: int, chat_key: str,
                         message: Any = None) -> str:
        """Resolve one ``cardtap`` tap (Living Chat P2 fencing).

        Current revision → the action's command dispatches
        through the ordinary control path (grants re-checked
        there). Superseded revision → the honest stale answer
        with the CURRENT card rendered, and the live card is
        re-delivered so the current state is what is on screen.
        Unknown card → a plain "gone" answer. Nothing stale ever
        executes.
        """
        from ...social.chat.living_cards import (
            STALE_CARD_MESSAGE,
            TapStatus,
            render_card,
            resolve_card_tap,
        )

        owner = _pick_owner_key(message) if message is not None \
            else ""
        out = resolve_card_tap(
            self._living_card_store(), card_id, revision, index,
            chat_key=chat_key, owner_key=owner)
        if out.status is TapStatus.OK and out.command:
            return self.handle_control(
                "/" + out.command.lstrip("/"), chat_key,
                message=message)
        if out.status is TapStatus.STALE and out.card is not None:
            card = out.card
            if message is not None:
                try:
                    self._living_card_host().deliver(
                        card, message.chat)
                except Exception as exc:  # noqa: BLE001
                    _log.debug("stale card re-delivery failed: %s",
                               exc)
            platform = (chat_key or "").partition(":")[0]
            return (STALE_CARD_MESSAGE + "\n"
                    + render_card(card, platform).text)
        if out.status is TapStatus.OUT_OF_RANGE and out.card is not None:
            return (f"that card has {len(out.card.actions)} "
                    "action(s) — the buttons on its current "
                    "version are the ones that work.")
        return ("that card is gone — it was replaced, expired, or "
                "belonged to an earlier run, and nothing was run "
                "from it.")

    # ── boot mission resumption ─────────────────────────────────────────

    def _start_boot_mission_resume(self) -> None:
        """Spawn the boot mission-resumption sweep on a daemon thread.

        The pass #54 run-claim is the arbiter: the sweep, a chat
        ``/mission resume``, the ops watcher's self-heal and a CLI
        resume can never double-drive a mission — whoever claims it
        first drives it, the rest are refused. This also composes
        with the self-update restart path (:meth:`_recover_interrupted_update`):
        an update restart is just another boot to this sweep, and if
        update recovery later restores a state snapshot, any mission
        this sweep drove reverts with that snapshot and is simply
        resumable again on the next boot.
        """

        def _job() -> None:
            try:
                self._resume_interrupted_missions()
            except Exception:  # noqa: BLE001 - boot continues regardless
                _log.exception("boot mission resumption failed")
            finally:
                self._release_db_thread()

        self._boot_resume_thread = threading.Thread(
            target=_job, name="mission-boot-resume", daemon=True)
        self._boot_resume_thread.start()

    @staticmethod
    def _mission_driver_live(mission: Any) -> bool:
        """True when a live driver holds ``mission``, by the runner's
        own claim-liveness rule (pass #54 ``_claim_is_live``): a
        foreign live pid is a live driver however fresh its heartbeat,
        a dead pid's claim is dead however fresh, and a same-process
        claim is live only while it keeps refreshing. Probes the
        persisted run claim, falling back to the heartbeat (a driver
        that died before its claim write — or an older row — still
        left one)."""
        from ...missions.runner import MissionRunner

        state = mission.state or {}
        claim = state.get("run_claim")
        if not (isinstance(claim, dict) and claim.get("token")):
            hb = state.get("heartbeat") or {}
            claim = {"pid": hb.get("pid"), "at": hb.get("at")}
        return MissionRunner._claim_is_live(claim)

    def _resume_interrupted_missions(self) -> dict[str, Any]:
        """Resume the missions a previous process left interrupted.

        Selection — deliberately narrower than ``MissionStore.resumable()``
        (which is every non-terminal row):

        * RUNNING with a dead/stale driver (per the run-claim liveness
          rule) → driven to completion.
        * PENDING with drive evidence (a heartbeat, a run claim, or
          completed steps — it died between create and its first
          heartbeat flip) → driven; a pristine PENDING was never
          started and is left alone.
        * RUNNING with a live driver → skipped; the claim refuses a
          second driver anyway.
        * PAUSED → left PAUSED (an explicit owner hold is not boot's
          to override).
        * Terminal rows are never revived — including the ones a
          reconcile-on-read already flipped to FAILED (recognizable
          by ``state["status_note"]`` starting ``"reconciled:"``);
          reviving those is the owner's deliberate ``/mission replan``.

        Terminal catch-up rides along first: a mission that finished
        while Devon was down whose terminal push was never marked
        delivered gets it re-delivered exactly once (the runner's
        own catch-up checks the persisted milestone marks).
        Reconcile-marked failures are excluded — a surface already
        surfaced that failure when it flipped the row.

        Drives run one thread + one production wired runner per
        mission (``max_iterations=0``: unlimited, budget-bound — the
        chat spawner's semantics), so a long mission never delays
        the fleet's claims and one bad mission never blocks the
        others. Returns the summary dict; when the sweep actually
        did something it also emits ONE owner summary through the
        durable Notifier (see :meth:`_report_boot_mission_resume`).
        Silence when nothing was resumed.
        """
        from ...core.errors import ValidationError
        from ...missions import MissionStatus, MissionStore, wired_runner

        summary: dict[str, Any] = {
            "resumed": [], "failed": [], "skipped_live": [],
            "held_paused": 0, "terminal_caught_up": []}
        db = getattr(self.context, "db", None)
        if db is None:
            return summary
        store = MissionStore(db)

        # 1) Terminal catch-up.
        for status in sorted(MissionStatus.TERMINAL):
            try:
                rows = store.list(status=status, limit=50)
            except Exception as _partner_exc:
                record_failure("runtime.PartnerRuntime._resume_interrupted_missions", _partner_exc)
                continue
            for mission in rows:
                state = mission.state or {}
                if str(state.get("status_note") or "").startswith(
                        "reconciled:"):
                    continue
                marks = state.get("milestones") or []
                if any(isinstance(e, dict)
                       and e.get("event") == f"terminal:{mission.status}"
                       for e in marks):
                    continue
                try:
                    wired_runner(self.context, store=store).run(mission)
                    summary["terminal_caught_up"].append(mission.id)
                except Exception as exc:  # noqa: BLE001 - isolated per mission
                    summary["failed"].append({
                        "id": mission.id, "name": mission.name,
                        "error": f"terminal catch-up: {exc}"})

        # 2) Select interrupted missions.
        targets: list[Any] = []
        for mission in store.resumable():
            if mission.status == MissionStatus.PAUSED:
                summary["held_paused"] += 1
            elif mission.status == MissionStatus.RUNNING:
                if self._mission_driver_live(mission):
                    summary["skipped_live"].append(mission.id)
                else:
                    targets.append(mission)
            elif mission.status == MissionStatus.PENDING:
                state = mission.state or {}
                driven = bool(state.get("heartbeat")
                              or state.get("run_claim")
                              or state.get("completed_steps"))
                if driven and not self._mission_driver_live(mission):
                    targets.append(mission)

        # 3) Drive them — one thread + wired runner per mission.
        lock = threading.Lock()

        def _drive(mission: Any) -> None:
            try:
                result = wired_runner(self.context, store=store).run(
                    mission, max_iterations=0)
                with lock:
                    summary["resumed"].append({
                        "id": mission.id, "name": mission.name,
                        "goal": mission.goal, "status": result.status})
            except ValidationError:
                # The claim did its job: another driver got there
                # first. Not a failure.
                with lock:
                    summary["skipped_live"].append(mission.id)
            except Exception as exc:  # noqa: BLE001 - isolated per mission
                with lock:
                    summary["failed"].append({
                        "id": mission.id, "name": mission.name,
                        "error": str(exc) or type(exc).__name__})
            finally:
                self._release_db_thread()

        threads = [
            threading.Thread(target=_drive, args=(mission,),
                             name=f"mission-boot-resume-{mission.id[:8]}",
                             daemon=True)
            for mission in targets]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        if (summary["resumed"] or summary["failed"]
                or summary["terminal_caught_up"]):
            self._report_boot_mission_resume(summary)
        else:
            _log.info("boot mission resume: nothing interrupted to resume")
        return summary

    def _report_boot_mission_resume(self, summary: dict[str, Any]) -> None:
        """Surface a boot resumption that acted to the owner: one
        durable Notifier publish (the alert persists even if live
        delivery fails) plus a log line — the same report-once shape
        as :meth:`_report_update_recovery`. Never raises."""
        resumed = summary.get("resumed") or []
        failed = summary.get("failed") or []
        caught = summary.get("terminal_caught_up") or []
        lines: list[str] = []
        for entry in resumed:
            lines.append(
                f"▶ {entry['name']} [{entry['id'][:8]}] — "
                f"{str(entry.get('goal') or '')[:80]} → {entry['status']}")
        for entry in failed:
            lines.append(
                f"✖ {entry['name']} [{entry['id'][:8]}] — resume "
                f"failed: {entry['error']}")
        if caught:
            lines.append(
                f"🔔 {len(caught)} finished mission(s) had an "
                "undelivered completion notice — re-sent.")
        if summary.get("held_paused"):
            lines.append(
                f"⏸ {summary['held_paused']} paused mission(s) left paused.")
        title = f"missions: resumed {len(resumed)} at boot"
        if failed:
            title += f", {len(failed)} failed"
        body = "\n".join(lines)
        if failed:
            _log.warning("boot mission resume acted: %s", title)
        else:
            _log.info("boot mission resume acted: %s", title)
        try:
            from ..notifier import Notifier

            Notifier(self.context, gateway=self.gateway).publish(
                "missions", title, body, force=True,
                alert_hints=({"failed": True,
                              "subject": "boot mission resume"}
                             if failed else
                             {"notice": True,
                              "subject": "boot mission resume"}))
        except Exception as exc:  # noqa: BLE001 - reporting is best-effort
            _log.warning("boot mission resume notice not delivered: %s", exc)

    def _arena_notify(self, text: str) -> None:
        """Deliver an arena build review to the owner, on every live channel."""
        from ...social.chat.base import ChatRef

        for key in _key_set(self.settings.partner.owner_chats):
            plat, _, cid = key.partition(":")
            if not cid:
                continue
            try:
                self.gateway.send(plat, ChatRef(platform=plat, chat_id=cid), text)
            except Exception as e:
                record_failure("runtime.PartnerRuntime._arena_notify", e)

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
        except Exception as e:
            record_failure("runtime.PartnerRuntime._tick_beacon", e)
            _log.debug("status beacon write failed", exc_info=True)

    def run(self, *, duration: float | None = None) -> None:
        self._started = time.time()
        self._beacon_next = 0.0  # write one immediately at startup
        # SIGTERM (supervisor stop, `kill`, phone process kill) must take
        # the same path as Ctrl-C: the loop exits and `finally` runs
        # stop(), which writes the final beacon with stopped=True so
        # `nm status` reports a clean stop instead of a stale death.
        from ...core.shutdown import (
            install_sigint_as_interrupt,
            install_sigterm_as_interrupt,
            install_windows_console_handler,
        )

        install_sigterm_as_interrupt()
        # Ctrl-C gets the managed handler too: the interpreter default
        # only fires when the disposition is default, and a launcher
        # that ignores SIGINT hands the ignore down — on the EC2 host
        # Ctrl-C did nothing at all and the owner had to close the
        # console window to stop the runtime (2026-10-26).
        install_sigint_as_interrupt()
        # Windows console reality: Ctrl-Break and the console window's
        # close button are not signals at all — they arrive as console
        # control events on an OS thread. The handler flips the same
        # stop event this loop polls, so it lands even when the main
        # thread is slow to reach a bytecode boundary. (No-op off
        # Windows.)
        install_windows_console_handler(
            lambda _event: self._stopped.set())
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
        except KeyboardInterrupt as e:  # noqa: E103, E106 - interrupt ends the beacon loop; finally stops it
            record_failure("runtime.PartnerRuntime.run", e)
        finally:
            self.stop()

    def stop(self) -> None:
        self._stopped.set()
        if getattr(self, "_arena", None) is not None:
            try:
                self._arena.stop_loop()
            except Exception as e:
                record_failure("runtime.PartnerRuntime.stop", e)
        if getattr(self, "_research", None) is not None:
            try:
                self._research.stop_loop()
            except Exception as e:
                record_failure("runtime.PartnerRuntime.stop:2723", e)
        if self._autonomy is not None:
            try:
                self._autonomy.stop()
            except Exception as e:
                record_failure("runtime.PartnerRuntime.stop:2728", e)
        _ledger = getattr(self, "_autonomy_db_handle", None)
        if _ledger is not None:
            self._autonomy_db_handle = None
            try:
                _ledger.close()
            except Exception as e:  # noqa: BLE001 - best-effort teardown
                record_failure("runtime.PartnerRuntime.stop:ledger", e)
        if getattr(self, "_scheduler", None) is not None:
            try:
                self._scheduler.stop()
            except Exception as e:
                record_failure("runtime.PartnerRuntime.stop:2733", e)
        if getattr(self, "_trigger_engine", None) is not None:
            try:
                self._trigger_engine.stop()
            except Exception as e:
                record_failure("runtime.PartnerRuntime.stop:2738", e)
        engine = getattr(self, "_game_engine_obj", None)
        if engine is not None:
            try:
                engine.shutdown()
            except Exception as e:
                record_failure("runtime.PartnerRuntime.stop:2744", e)
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
        except Exception as e:
            record_failure("runtime.PartnerRuntime.stop:2760", e)
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
            # Effective delivery mode (seam A): durable / direct /
            # durable-requested-unavailable — visible here, not just
            # in a log line.
            "delivery": {
                "mode": self.gateway.delivery_mode,
                "durable_outbox": bool(self.gateway.durable_outbox),
                "outbox": self.gateway.outbox is not None,
            },
            # Degradation ladders (llm / telegram / voice): which rung
            # each subsystem is serving from right now. Empty when the
            # error system never wired (shouldn't happen in production).
            "degradation": self._degradation_status(),
            "autonomy": self._autonomy.status() if self._autonomy else "off",
            # Dynamic lexicon voice feed: per-category term counts so the
            # owner can see what the feed is working with (deeper inspection
            # via the research_lexicon tool: action=terms|stats).
            "lexicon": (
                self.brain.lexicon.status()
                if self.brain.lexicon is not None
                else {"available": False, "enabled": False, "module": "partner"}
            ),
            # Partner failure ledger: best-effort fallbacks that fired
            # instead of raising. Additive key; absent when the ledger
            # module is unavailable (never breaks status).
            "failures": _partner_failures_status(),
            # Wave D: how many interactive replies hit the reply budget and
            # fell back instead of stalling the chat.
            "reply_timeouts": self.brain.responder.reply_timeouts,
        }

    def _degradation_status(self) -> dict[str, Any]:
        """Ladder snapshot for status surfaces. Never raises."""
        es = getattr(self.context, "error_system", None)
        if es is None:
            return {"wired": False, "degraded": [], "ladders": {}}
        try:
            return {
                "wired": True,
                "degraded": es.ladders.degraded_subsystems(),
                "ladders": es.ladders.status_all(),
            }
        except Exception as e:
            record_failure("runtime.PartnerRuntime._degradation_status", e)
            return {"wired": True, "degraded": [], "ladders": {}}

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
            data = KVStore(self.context.db).get("partner.autonomy_mode", default={})
            persisted = str(data.get("mode") or "")
            if persisted in {"off", "suggest", "auto"}:
                self.settings.partner.autonomy_mode = persisted
        except Exception as e:
            record_failure("runtime.PartnerRuntime._apply_persisted_mode", e)

    # ── control commands (live steering) ─────────────────────────────────────
    def _immune_event_sink(self):
        """Organ-bus sink for immune events (AA-42).

        Immune events (threat detected / contained, antibody created,
        false positive retired) persist into the cross-organ event bus
        (``organ_events``, dst ``brain`` — the same dst research and
        wisdom emit decision-relevant events to). Returns None when
        the context carries no db; the immune surface then keeps its
        events on its own state record instead.
        """
        db = getattr(self.context, "db", None)
        if db is None:
            return None
        from ... import organs

        def _sink(kind: str, payload: dict) -> None:
            organs.emit(db, "immune", "brain", kind, payload)

        return _sink

    def _is_operator(self, message: ChatMessage) -> bool:
        # Surface rule (owner's model, 2026-10-20): operator power is a
        # property of the surface, not just the person. On the gated
        # public bot (telegram-bot) NOBODY is an operator — the owner
        # included; owner power lives on the personal bot and the
        # userbot. This check comes first, before every identity test,
        # so no identity path (meta stamp, sender allowlist, chat key)
        # can re-open operator power on the public surface.
        if is_gated_surface(getattr(message, "chat", None)):
            return False
        # Same single owner test as reply gating — a control command from a
        # chat the gating layer doesn't call owner must not run.
        # Prefer the gateway's verdict (stashed in meta) which includes the
        # DB-registered owner chats; fall back to the config-only check.
        meta = getattr(message, "meta", None)
        if isinstance(meta, dict) and meta.get("is_owner") is True:
            return True
        # The WhatsApp self-chat verdict (adapter-stamped): the paired
        # account's own "Message yourself" chat is an owner surface by
        # construction — only the account owner can write there. The
        # gateway's inbound owner test honors the same stamp, so the
        # two can never disagree.
        if isinstance(meta, dict) and meta.get("self_chat") is True:
            return True
        # The WhatsApp owner-command verdict (adapter-stamped): the
        # bridge reported one of the paired account's OWN messages
        # in a group, and the adapter promoted it to a turn because
        # it parses as a registered command. The bridge emits
        # own_message events only for the paired account itself, so
        # the stamp carries the same provenance as the self-chat
        # stamp above — the owner's identity on his own account,
        # proven by the pairing — and the gateway's inbound owner
        # test honors it identically. This is what lets the owner
        # command her from inside a WhatsApp group, exactly as on
        # the userbot. His non-command group text is never promoted
        # (the adapter keeps it persist-only), so no conversation
        # path opens with this — commands only.
        if isinstance(meta, dict) and meta.get("owner_command") is True:
            return True
        # Sender-id allowlist: the owner's Telegram user id, same number
        # on every Telegram surface. Covers the personal bot and userbot,
        # where the chat key carries an adapter prefix the owner never
        # wrote in config. Strangers' sender ids never match.
        sender_id = str(getattr(message, "sender_id", "") or "").strip()
        if sender_id and sender_id in getattr(self, "_owner_sender_ids", set()):
            return True
        return is_owner_chat(message.chat, owner_chats=self._owner_chats)

    # ── one command, one handling, across surfaces ─────────────
    # (owner order 2026-11-05; see command_claims.py for the
    # incident). A Telegram message reaches this runtime once per
    # attached surface; without arbitration every surface that
    # parses a command handles it. The checkpoints in
    # _process_inner consult _command_stands_down before any
    # command branch runs.

    def _command_claims(self):  # type: ignore[no-untyped-def]
        ledger = getattr(self, "_command_claims_obj", None)
        if ledger is None:
            from .command_claims import CommandClaimLedger

            ledger = CommandClaimLedger()
            self._command_claims_obj = ledger
        return ledger

    def _reply_claims(self):  # type: ignore[no-untyped-def]
        ledger = getattr(self, "_reply_claims_obj", None)
        if ledger is None:
            from .reply_claims import ReplyClaimLedger

            ledger = ReplyClaimLedger()
            self._reply_claims_obj = ledger
        return ledger

    def _reply_already_claimed(self, message) -> bool:  # type: ignore[no-untyped-def]
        """True when this message identity already had its
        conversational turn claimed — a redelivery stands down.

        The conversational counterpart of the command claims
        (see :mod:`reply_claims` for the incident and the law).
        Never raises: a ledger fault must not silence chat, so a
        failure here lets the turn proceed unclaimed."""
        try:
            from .reply_claims import reply_claim_key

            key = reply_claim_key(message)
            if not key:
                return False
            return not self._reply_claims().claim(key)
        except Exception as exc:  # noqa: BLE001 - arbitration
            record_failure(               # never eats a turn
                "runtime.PartnerRuntime._reply_already_claimed", exc)
            return False

    def _surface_usernames(self) -> dict[str, str]:
        """{username (lowercase, no @): platform} for every
        attached surface that knows its own handle — the bot
        adapters learn theirs from getMe, the userbot from its
        session. Plus the configured main-bot username as a
        fallback for a bot that has not connected yet."""
        out: dict[str, str] = {}
        gateway = getattr(self, "gateway", None)
        adapters = getattr(gateway, "adapters", {}) or {}
        for name, adapter in adapters.items():
            for attr in ("_bot_username", "_me_username"):
                value = getattr(adapter, attr, None)
                if value:
                    out[str(value).strip().lower().lstrip("@")] = \
                        str(name)
        try:
            web = getattr(self.context.settings, "web", None)
            configured = str(
                getattr(web, "bot_username", "") or ""
            ).strip().lower().lstrip("@")
            if configured:
                out.setdefault(configured, "telegram-bot")
        except Exception:  # noqa: BLE001 - a fallback, not a need
            pass
        return out

    def _bot_account_ids(self) -> set[str]:
        """The numeric account ids of Devon's own bots, from the
        attached adapters: each bot adapter's own id, and the
        userbot's companion-bot registry (it is configured with
        BOTH bot ids so it never talks over them)."""
        ids: set[str] = set()
        gateway = getattr(self, "gateway", None)
        adapters = getattr(gateway, "adapters", {}) or {}
        for adapter in adapters.values():
            bot_id = getattr(adapter, "_bot_id", None)
            if bot_id:
                ids.add(str(bot_id))
            companions = getattr(adapter, "companion_bot_ids", None)
            if companions:
                for cand in companions:
                    try:
                        ids.add(str(int(cand)))
                    except (TypeError, ValueError):
                        continue
        return ids

    def _command_stands_down(self, message, command) -> bool:  # type: ignore[no-untyped-def]
        """True when THIS surface must not handle ``command``:
        another surface owns it. Silent by design — standing down
        never replies; the owning surface's answer is the only
        one the room sees. Never raises."""
        try:
            return self._command_stands_down_inner(message, command)
        except Exception as exc:  # noqa: BLE001 - arbitration
            record_failure(               # never eats a command
                "runtime.PartnerRuntime._command_stands_down", exc)
            return False

    def _command_stands_down_inner(self, message, command) -> bool:  # type: ignore[no-untyped-def]
        chat = message.chat
        platform = str(chat.platform or "")
        bot_surface = platform in ("telegram-bot", "telegram-personal")
        # (1) Addressed commands belong to the addressed account's
        # surface, everywhere: /cmd@Nomorals_bot is the bot's, on
        # the userbot's copy too — and a command addressed to an
        # account that is not one of ours is not ours at all.
        addressee = str(getattr(command, "addressee", "")
                        or "").strip().lower()
        if addressee:
            owners = self._surface_usernames()
            if owners:
                owner_surface = owners.get(addressee)
                if owner_surface is None \
                        or owner_surface != platform:
                    return True
        # (2) A bot's DM belongs to that bot: a non-bot surface
        # never handles commands in a DM whose peer is one of
        # Devon's bots (the userbot sees the owner's bot chats in
        # his dialogs — the bot's copy is the handling).
        if chat.kind == ChatKind.DM and not bot_surface \
                and str(chat.chat_id or "") in self._bot_account_ids():
            return True
        # (3) In a group where the bot's standing is VERIFIED
        # (the merged §33 claim says bot_admin), group systems
        # speak through the bot: a non-bot surface leaves group
        # commands to it. Designated rooms without a bot keep
        # their personal-surface handling — designation is not
        # bot standing, and those rooms have no bot copy coming.
        if chat.kind == ChatKind.GROUP and not bot_surface:
            try:
                state = self._group_surface().claim_state(chat.key)
            except Exception:  # noqa: BLE001 - unproven, no defer
                state = None
            if state and state.get("bot_admin") is True:
                return True
        # (4) One handling: claim this message identity. The same
        # message carries the same platform message id on every
        # surface; the family collapses Telegram's three platform
        # names. A prior claim by another surface stands this one
        # down; this surface's own claim re-checks as its own.
        message_id = str(getattr(message, "message_id", "") or "")
        if message_id:
            family = ("telegram" if platform.startswith("telegram")
                      else platform)
            key = f"{family}:{chat.chat_id}:{message_id}"
            if not self._command_claims().claim(key, platform):
                return True
        return False

    def _money_send_hook(self, message: ChatMessage) -> str | None:
        """Build-map #49: "send 5k to Mama" → staged transfer + guard.

        Returns a reply string when a money-send intent fired, else None.
        Never raises — failures return honest error strings.
        """
        from ...finance.send import parse_send_request, send_money
        req = parse_send_request(message.text)
        if req is None:
            return None
        _log.info("money-send intent: %s → %s", req["amount_kobo"], req["to"])
        # Explicit "send it anyway" after a warning → override the guard.
        override = "send it anyway" in message.text.lower()
        result = send_money(req["to"], req["amount_kobo"], context=self.context,
                            override_warning=override)
        if result.get("warning"):
            w = result["warning"]
            return (w["message"] + "\n\nReply 'send it anyway' to proceed, "
                    "or 'cancel' to drop it.")
        if result.get("needs") == "recipient":
            return result["ask"]
        if result.get("needs") == "biometric":
            return (result["prompt"] + " (confirm on your phone to complete)")
        if result.get("error"):
            return f"couldn't send: {result['error']}"
        return None

    def _track_hook(self, message: ChatMessage) -> str | None:
        """Build-map #71: "track LOS LHR flights for me" → price watcher.

        Returns a reply string when a tracking intent fired, else None.
        Never raises.
        """
        import re
        text = (message.text or "").strip()
        low = text.lower()
        if not re.search(r"\btrack\b.*\b(flight|fare|price)", low):
            return None
        from ...travel.watchers import PriceWatcher, parse_watch_request
        parsed = parse_watch_request(text)
        if parsed is None or not parsed.get("departure_date"):
            return ("sure — which route and date? e.g. "
                    "'track LOS to LHR on 2026-12-01 under 400k'.")
        try:
            watcher = getattr(self, "_price_watcher", None)
            if watcher is None:
                watcher = PriceWatcher()
                self._price_watcher = watcher
            w = watcher.watch(
                parsed["origin"], parsed["destination"],
                parsed["departure_date"],
                target_kobo=parsed["target_kobo"])
        except ValueError as exc:
            return f"couldn't start the watch: {exc}"
        try:
            alert = watcher.check(w)
        except Exception as e:
            record_failure("runtime.PartnerRuntime._track_hook", e)
            alert = None
        msg = f"👀 watching {w.route} on {w.departure_date} (id {w.id})."
        if alert is not None:
            msg += "\n" + alert.text
        return msg

    def _trip_hook(self, message: ChatMessage) -> str | None:
        """Build-map #72: forwarded booking confirmation → auto itinerary.

        Returns a reply string when confirmation-like text was ingested,
        else None. Never raises.
        """
        text = (message.text or "").strip()
        if not text or text.startswith("/"):
            return None
        try:
            from ...travel.itinerary import (
                ItineraryBuilder, confirmation_hook, added_message,
            )
        except Exception as e:
            record_failure("runtime.PartnerRuntime._trip_hook", e)
            return None
        if not confirmation_hook(text):
            return None
        try:
            builder = getattr(self, "_trip_builder", None)
            if builder is None:
                builder = ItineraryBuilder()
                self._trip_builder = builder
            trip = builder.ingest_email(text)
        except Exception:  # noqa: BLE001
            _log.exception("trip hook ingest failed")
            return None
        if trip.is_empty():
            return None
        return added_message(trip) + f"\n(trip id: {trip.id})"

    def _music_full_hook(self, message: ChatMessage) -> str | None:
        """Build-map #58: "make me a song about X" → full vocal pipeline.

        Returns a reply string when a full-song intent fired, else None.
        Never raises — failures return honest error strings.
        """
        from ...media.vocals import (
            RVCVoiceRegistry, parse_full_request, make_full_song)
        from ...media.music import STYLES
        req = parse_full_request(message.text, STYLES)
        if req is None:
            return None
        _log.info("music-full intent: %s [%s]", req["topic"], req["style"])
        registry = RVCVoiceRegistry()
        default = registry.default()
        # No RVC voice? No problem — make_full_song falls back to TTS
        # vocals automatically (the phone path). Never dead-end.
        voice_id = default.voice_id if default is not None else ""
        try:
            res = make_full_song(req["topic"], style=req["style"],
                                 voice_id=voice_id,
                                 audience="private", context=self.context)
        except Exception as exc:  # noqa: BLE001 - both model errors
            return f"🎵 couldn't make the full song:\n{exc}"
        vlabel = voice_id or "tts vocals"
        text = (f"🎵 “{res.title}” — full song [{vlabel}]\n"
                f"{res.note}")
        if res.master_path:
            vocal_txt = "AI vocals" if res.vocal_path else "instrumental"
            try:
                self.gateway.send_file(
                    message.chat.platform,
                    f"{message.chat.platform}:{message.chat.chat_id}",
                    res.master_path,
                    caption=f"🎵 {res.title} — AI instrumental + {vocal_txt}", kind="audio")
                text += "\nsent the master to this chat."
            except Exception as e:
                record_failure("runtime.PartnerRuntime._music_full_hook", e)
                text += f"\nmaster: {res.master_path}"
        return text

    def _explicit_post_override(self, message: ChatMessage) -> str | None:
        """Build-map #43: the owner's explicit post instruction = execute.

        Returns a reply string when explicit-post intent fired (the post or
        schedule already happened), else None so the message falls through
        to the normal pipeline. Never raises — failures return an honest
        error string, never silence.
        """
        from ...social.drafts import (
            DraftQueue, detect_explicit_post, execute_post, schedule_post,
        )
        intent = detect_explicit_post(message.text)
        if intent is None:
            return None
        _log.info("explicit-post override fired (%r) in %s",
                  intent.get("matched"), message.chat.key)
        try:
            queue = DraftQueue()
        except Exception as exc:  # noqa: BLE001
            return f"couldn't open the draft queue: {exc}"
        try:
            from ...social import SocialManager
            manager = SocialManager(self.context).register_builtins()
        except Exception as exc:  # noqa: BLE001
            return f"can't post — no publisher wired: {exc}"
        # Prefer the owner's pending draft; otherwise the message text
        # after the command is the content ("post this: <text>").
        pending = [d for d in queue.pending()
                   if d.status in ("pending_review", "approved", "draft")]
        draft = pending[0] if pending else None
        if draft is not None:
            content, platforms, draft_id = (
                draft.content, draft.platforms or None, draft.id)
        else:
            content = re.sub(r"(?i)^(post|publish|send|reply)\b[^:]*:?\s*",
                             "", message.text.strip()).strip()
            if not content or detect_explicit_post(content) is not None:
                return ("what should I post? Send the text first, then say"
                        " 'post this'.")
            platforms, draft_id = None, None
        try:
            if intent["action"] == "schedule" and intent.get("at"):
                if draft is None:
                    new_draft = queue.create_draft(content, list(platforms or []))
                    queue.propose(new_draft.id)
                    draft_id = new_draft.id
                scheduler = getattr(self, "_scheduler", None) or \
                    self.context.extras.get("scheduler")
                if scheduler is None:
                    return "can't schedule — the scheduler isn't running."
                schedule_post(scheduler, queue, draft_id, intent["at"],
                              platforms=platforms)
                when = datetime.fromtimestamp(intent["at"]).strftime(
                    "%I:%M %p, %b %d")
                return f"scheduled for {when} — it'll go out on its own."
            outcome = execute_post(manager, content, platforms=platforms,
                                   draft_id=draft_id, queue=queue)
            posted = len(getattr(outcome, "posted", []) or [])
            failed = len(getattr(outcome, "failed", []) or [])
            if posted and not failed:
                return f"posted to {posted} platform(s)."
            if posted:
                return (f"posted to {posted}, {failed} failed: "
                        f"{getattr(outcome, 'error', '')[:120]}")
            return (f"post failed: "
                    f"{getattr(outcome, 'error', 'unknown error')[:160]}")
        except Exception as exc:  # noqa: BLE001
            return f"post failed: {exc}"
        finally:
            queue.close()

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
            except Exception as e:
                record_failure("runtime.PartnerRuntime._progression_gate", e)
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
        except Exception as e:
            record_failure("runtime.PartnerRuntime._progression_gate:3112", e)
            return None

    def _build_adapter(self, name: str) -> Any:
        """Factory for hot starts. Returns the adapter, None for an unknown
        or disabled platform, and raises with a readable message when a
        platform is enabled but unavailable (missing dependency/credentials) —
        the gateway reports that message to the owner."""
        from ...social.chat import build_adapter

        return build_adapter(self.settings, name)

    def _confirm_gate(self, kind: str, tail: str, chat_key: str,
                      ) -> tuple[Any, bool, str]:
        """Confirm-flow gate for CONFIRMABLE commands.

        Returns ``(reply, proceed, effective_tail)``: ``proceed`` False
        means handle_control should return ``reply`` (confirm prompt, or
        cancel/expired/nothing-to-confirm message); True means dispatch
        with ``effective_tail`` (the ORIGINAL tail stored when the
        request was first made — so ``/forget abc`` + ``/forget confirm``
        forgets ``abc``). Pending confirmations are keyed by chat_key,
        expire after CONFIRM_TTL_S, and are single-use. The prompt
        DECLARES its Confirm / Cancel buttons (presentation
        architecture — an ActionableMessage, no longer derived from
        the ``⚠️ confirm: /<kind>`` text by the adapter); typed
        ``/<kind> confirm`` / ``/<kind> cancel`` resolve identically.
        """
        from ...social.chat.control import CONFIRM_TTL_S
        import time as _time

        pending: dict = getattr(self, "_pending_confirms", None)
        if pending is None:
            pending = self._pending_confirms = {}
        verb = (tail or "").strip().lower()
        if verb == "confirm":
            # Peek, don't pop: a confirm for the WRONG kind must neither
            # authorize anything nor destroy the pending request (a
            # mismatched "/quit confirm" after "/forget abc" used to eat
            # the forget confirmation). Only a kind-matching confirm
            # consumes the token — single-use, exact-bound (§5.3).
            stored = pending.get(chat_key)
            if stored is None or stored[0] != kind:
                return (f"nothing to confirm — send /{kind} … first, "
                        f"then /{kind} confirm within 5 minutes.", False, "")
            _, original_tail, expires_at = stored
            if expires_at < _time.time():
                pending.pop(chat_key, None)
                return (f"that confirmation expired — send /{kind} … again "
                        f"if you still want it.", False, "")
            pending.pop(chat_key, None)
            return (None, True, original_tail)
        if verb == "cancel":
            stored = pending.get(chat_key)
            if stored is None or stored[0] != kind:
                return (f"nothing to cancel — no pending /{kind}.",
                        False, "")
            pending.pop(chat_key, None)
            return (f"cancelled — /{kind} did not run.", False, "")
        # Fresh request: /quit takes no arguments, so a non-verb tail is
        # a usage error, not a confirmable request.
        if kind == "quit" and tail.strip():
            return ("usage: /quit — no arguments (confirm with "
                    "/quit confirm).", False, "")
        pending[chat_key] = (kind, tail, _time.time() + CONFIRM_TTL_S)
        if kind == "quit":
            what = "shut Devon down"
        else:
            what = f"forget {tail.strip()!r}"
        from ...social.chat.tgbot_buttons import actionable

        return (
            actionable(
                f"⚠️ confirm: /{kind} — this will {what}.",
                [[("❌ Cancel", f"{kind} cancel"),
                  ("✅ Confirm", f"{kind} confirm")]]),
            False, "")

    def _run_command_stack(self, segments: list[str], chat_key: str,
                           message: Any = None) -> str:
        """Run a validated command stack in order; one combined reply.

        Pre-validation runs FIRST: every step's kind is checked for
        stackability, the caller's per-command grant, the SMS surface
        and the progression lock — before anything runs. A single bad
        step rejects the whole stack with an explicit message and
        nothing runs, so an outsider can never smuggle an owner-only
        command behind a public one.

        Steps then run in order via :meth:`handle_control` with
        ``_no_stack=True`` (a ``{prev}`` expansion carries the previous
        step's output, newlines included — it must be one command's
        argument, never fresh commands). A step that raises, is denied,
        or asks for confirmation stops the stack: the failure is
        reported with its step header and the remaining steps are named
        as not run. Commands return prose, not exit codes, so a step
        that ran and reported its own problem in its reply is shown
        verbatim under its header and the stack continues — nothing is
        ever silent, because every step's output is in the reply.
        """
        from ...social.chat.control import (
            NON_STACKABLE_COMMANDS, STACK_DENIAL_REPLY,
            expand_prev_placeholders, parse_control, caller_may_run)
        from ...social.chat.sms import SMS_BLOCKED_REPLY, sms_allows

        is_owner = True if message is None else self._is_operator(message)
        platform = (chat_key or "").partition(":")[0].strip().lower()
        parsed = []
        for i, seg in enumerate(segments):
            cmd = parse_control(seg)
            if cmd is None or cmd.kind == "error":
                return (f"stack stopped before it started — step {i + 1} "
                        "did not parse as a command. Nothing ran.")
            if cmd.kind in NON_STACKABLE_COMMANDS:
                return (f"stack stopped before it started — step {i + 1} "
                        f"/{cmd.kind} can't be stacked (games, confirmations "
                        "and lifecycle commands go one per message). "
                        "Nothing ran — send it on its own.")
            if not caller_may_run(cmd.kind, is_owner):
                return (f"stack stopped before it started — step {i + 1} "
                        f"/{cmd.kind} is owner-only, and you're not the "
                        "owner. Nothing ran.")
            if platform == "sms" and not sms_allows(cmd.kind):
                return (SMS_BLOCKED_REPLY + " — stack stopped before it "
                        "started. Nothing ran.")
            locked = self._progression_gate(cmd.kind, message)
            if locked is not None:
                return (locked + "\n— stack stopped before it started. "
                        "Nothing ran.")
            parsed.append(cmd)

        total = len(segments)
        outputs: list[str] = []
        parts: list[str] = []
        for i, (seg, cmd) in enumerate(zip(segments, parsed)):
            expanded = expand_prev_placeholders(seg, outputs)
            try:
                reply = self._handle_control_inner(
                    expanded, chat_key, message=message, _no_stack=True)
                # A stacked reply is composite TEXT — a step that
                # returned an ActionableMessage contributes its text;
                # its buttons belong to its own send, not the stack.
                if hasattr(reply, "text"):
                    reply = str(reply.text)
            except Exception as exc:  # noqa: BLE001 - one bad step stops the stack, loudly
                parts.append(
                    f"── {i + 1}/{total} /{cmd.kind} ──\n"
                    f"⛔ failed: {type(exc).__name__}: {exc}")
                if i + 1 < total:
                    parts.append(
                        f"— stack stopped at step {i + 1}/{total}; "
                        f"steps {i + 2}–{total} did not run —")
                return "\n".join(parts)
            if reply == STACK_DENIAL_REPLY:
                parts.append(
                    f"── {i + 1}/{total} /{cmd.kind} ──\n"
                    "⛔ denied — that one's just for the owner.")
                if i + 1 < total:
                    parts.append(
                        f"— stack stopped at step {i + 1}/{total}; "
                        f"steps {i + 2}–{total} did not run —")
                return "\n".join(parts)
            if isinstance(reply, str) and reply.startswith("⚠️ confirm:"):
                # A bespoke confirm prompt inside a step (e.g. a
                # multi-step subcommand): never auto-answer it, and do
                # not echo the prompt line — the adapter turns it into
                # Confirm/Cancel buttons for a confirmation that was
                # never armed.
                parts.append(
                    f"── {i + 1}/{total} /{cmd.kind} ──\n"
                    "⛔ needs confirmation — send it on its own, then "
                    "answer its confirm prompt.")
                if i + 1 < total:
                    parts.append(
                        f"— stack stopped at step {i + 1}/{total}; "
                        f"steps {i + 2}–{total} did not run —")
                return "\n".join(parts)
            outputs.append(reply or "")
            parts.append(
                f"── {i + 1}/{total} /{cmd.kind} ──\n"
                f"{reply if reply else '(no output)'}")
        return "\n".join(parts)

    def _menu_session_reply(self, message, chat_key, command):  # type: ignore[no-untyped-def]
        """The live menu render for /menu, /list, /commands and bare
        /help — or None when this surface keeps the catalog text path.

        The session render applies only where the one-live-message
        law can hold: a real inbound message, a DM or a group (the
        2026-11-07 paging law moved groups onto the session too —
        the catalog pages inside the one live message there as
        well), and a platform whose limits support buttons AND
        text edit. WhatsApp takes the session render too since
        the buttons completion (2026-11-08): its catalog was
        authored per-platform all along (menu_session_render
        renders the whatsapp section tree), its keyboards ride
        the bridges' interactive form (3 native + text-line
        overflow), and its edits now carry keyboards — the
        text-only whatsapp_menu() frame remains the fallback for
        buttonless contexts below. Buttonless surfaces and
        internal (message=None) calls keep the text path. An
        unknown topic also falls back — the catalog's "no group"
        reply stays the answer. Channels never reach dispatch at
        all.
        """
        if message is None:
            return None
        platform = (chat_key or "").partition(":")[0].strip().lower()
        chat = getattr(message, "chat", None)
        if chat is None or getattr(chat, "kind", None) not in (
                ChatKind.DM, ChatKind.GROUP):
            return None
        from ...social.chat.control import menu_session_render
        from ...social.chat.platform_limits import for_platform
        limits = for_platform(platform)
        if limits is None or not (limits.supports_buttons
                                  and limits.supports_text_edit):
            return None
        tail = str(command.tail or "").split()
        topic = (command.arg or "").strip()
        page = 1
        if len(tail) > 1 and tail[1].isdigit():
            page = max(1, int(tail[1]))
        meta = getattr(message, "meta", None) or {}
        tapped = bool(meta.get("tapped_message_id"))
        public_only = is_gated_surface(platform)
        if topic.lower() in ("home", "index"):
            return menu_session_render(
                platform=platform, public_only=public_only,
                home=True, continuation=True,
                media=self._menu_head_media(platform))
        if not topic:
            if tapped and command.kind in ("list", "commands"):
                # The category menu's "All commands" tap: start the
                # linear page walk at the first section, editing the
                # menu message — never a fresh catalog wall.
                return menu_session_render(
                    platform=platform, public_only=public_only)
            return menu_session_render(
                platform=platform, public_only=public_only,
                home=True, continuation=tapped,
                media=self._menu_head_media(platform))
        return menu_session_render(
            topic, page, platform=platform, public_only=public_only,
            continuation=True)

    def _menu_head_media(self, platform: str):  # type: ignore[no-untyped-def]
        """The menu home's picture head (owner, 2026-11-07:
        "no pic no vid" — the working menu was media-dead):
        her banner, from her media library (rendered by her
        own studio the first time, then shelved).  The Telegram
        surfaces take it natively; WhatsApp takes it since the
        buttons completion (2026-11-08) — its media sends carry
        the keyboard in the interactive header, so photo +
        caption + buttons is a real form there too. The MTProto
        menu keeps the text head (capability-checked, never
        faked).  None on any failure: the menu never breaks for
        its picture."""
        if platform not in ("telegram-bot", "telegram-personal",
                            "whatsapp"):
            return None
        try:
            item = self._expression_service().menu_banner_item()
        except Exception as exc:  # noqa: BLE001 - menu survives it
            record_failure(
                "runtime.PartnerRuntime._menu_head_media", exc)
            return None
        if not item or not item.get("path"):
            return None
        return {"path": str(item["path"]), "kind": "image"}

    def _clear_menu_grid_once(self, message) -> None:
        """Retire the persistent catalog reply keyboard in this chat.

        The grid attached on every bare /start|/menu reply until the
        2026-11-06 menufix; a persistent keyboard survives in the
        client until a remove lands, so the first bare-menu reply in
        each Telegram chat carries one short carrier message with
        the remove verb (an empty reply_keyboard), KV-flagged so it
        happens exactly once per chat, ever. Best-effort: a failed
        send leaves the flag unset and the next menu retries; a
        failure here never breaks the menu reply itself.
        """
        try:
            chat = getattr(message, "chat", None)
            if chat is None:
                return
            platform = str(getattr(chat, "platform", "") or "")
            if platform not in ("telegram", "telegram-bot",
                                "telegram-personal"):
                return
            kv = KVStore(self.context.db)
            flag = f"menu.grid_cleared:{chat.key}"
            if kv.get(flag, default=False):
                return
            result = self.gateway.send(
                platform, chat,
                "⌨️ the bottom command grid is retired — "
                "the menu carries the buttons now.",
                reply_keyboard=[])
            if result is not None and getattr(result, "ok", False):
                kv.set(flag, True)
        except Exception as exc:  # noqa: BLE001 - cosmetic cleanup
            record_failure(
                "runtime.PartnerRuntime._clear_menu_grid_once", exc)

    def _send_control_reply(self, message, reply, *,  # type: ignore[no-untyped-def]
                            reply_keyboard=None):
        """The ONE send channel for control-command replies, on
        every bot and surface (presentation build, owner order
        2026-11-03: buttons + formatting are platform-wide, not a
        per-surface improvisation).

        A handler returns either a plain string or an
        ActionableMessage (tgbot_buttons.actionable): text and its
        buttons authored together from the same state that
        rendered the display, so the buttons always match what's
        on screen.  Platforms without buttons ignore the keyboard.

        Formatting convention: reply text carrying <b> or <code>
        goes out as HTML (the adapter falls back to plain text if
        the platform rejects the markup — formatting never eats
        the message)."""
        text, buttons = reply, None
        if hasattr(reply, "text") and hasattr(reply, "buttons"):
            text, buttons = str(reply.text), reply.buttons
        # Live-session declaration (presentation architecture):
        # an interactive flow's renders belong to ONE live message
        # per chat, edited in place — see social/chat/live_session.
        session = str(getattr(reply, "session", "") or "")
        session_edit = bool(getattr(reply, "session_edit", False))
        # A declared media head (the menu's picture): the fresh
        # send goes out as a media message — photo + this text
        # as its caption + the same keyboard — where the
        # surface's adapter carries it; session renders then
        # edit the caption in place.  Any failure falls through
        # to the plain text send below, unchanged.
        media = getattr(reply, "media", None)
        if not isinstance(media, dict) or not media.get("path"):
            media = None
        text = str(text or "")
        if not text and not buttons:
            return None
        # Button honesty (register #27, 2026-11-05): copy may claim
        # buttons only when buttons actually ride this send. When
        # the platform cannot take them (limits say so, or none were
        # attached to a reply whose copy claims them), the copy is
        # rewritten HERE — before the gateway — to an honest
        # numbered action list, and the picks are registered so a
        # bare-number reply executes the same command a tap would.
        # The failure this kills: /toolhub on WhatsApp said "tap a
        # button below" while the gateway stripped the bundle.
        btn_rows = None
        if buttons:
            try:
                btn_rows = [
                    [(str(label), str(data))
                     for label, data in row]
                    for row in buttons if row]
            except Exception:  # noqa: BLE001 - malformed bundle -> honesty path
                btn_rows = None
            btn_rows = [r for r in (btn_rows or []) if r] or None
        _limits = None
        try:
            from ...social.chat.platform_limits import for_platform
            _limits = for_platform(message.chat.platform)
        except Exception:  # noqa: BLE001 - unknown platform: honesty by copy
            _limits = None
        _takes_buttons = bool(
            _limits is not None and _limits.supports_buttons)
        if _claims_buttons(text) and not (btn_rows and _takes_buttons):
            if btn_rows:
                text = _numbered_button_fallback(self, message, text,
                                                 btn_rows)
                btn_rows = None
                buttons = None
            elif _limits is not None and not _takes_buttons:
                text = _strip_button_claims(text) or (
                    "buttons don't exist on this surface — the same "
                    "actions run by typing their commands "
                    "(/help lists them).")
            # A button-capable platform with claim copy but no
            # authored buttons is the adapter-derivation path
            # (Telegram derives hub keyboards from the text):
            # the claim stays, the adapter owns the buttons.
        # The stamp (owner law, 2026-11-05): always present on
        # presented output — this channel is where control replies
        # become output, so it stamps here, once (the transports
        # render it down: Telegram italic entity, WhatsApp _…_,
        # plain = the bare line).  Cards already stamped upstream
        # pass unchanged.  A plain unadorned reply (no markup, no
        # buttons) takes the stamp's LIGHTEST form — one line,
        # no blank-line ceremony.
        try:
            text = apply_stamp(
                text, light=not buttons and "<" not in text)
        except Exception as e:  # noqa: BLE001 - the reply must land
            record_failure("runtime.PartnerRuntime._send_control_reply.stamp", e)
        # §16 ephemeral answers: handle_control stamps replies
        # of sensitive kinds in groups that opted in (p5
        # ephemeral).  The ladder, in order: the adapter's
        # ephemeral send (visible only to the asker + the bot
        # in-group) → the same text by DM → a bare pointer in
        # the group.  The content itself is NEVER posted
        # publicly as a fallback — that would defeat the point
        # silently.
        _eph = (getattr(message, "meta", None) or {}).get(
            "ephemeral_reply")
        if isinstance(_eph, dict) and text:
            _uid = str(_eph.get("user_id") or "")
            _adapter = (getattr(self.gateway, "adapters", {})
                        or {}).get(message.chat.platform)
            _verb = getattr(_adapter, "send_ephemeral", None)
            if _uid and callable(_verb):
                try:
                    _out = _verb(message.chat, _uid, text)
                except Exception:  # noqa: BLE001 - fall to DM
                    _out = None
                if isinstance(_out, dict) and _out.get("ok"):
                    return _out
            if _uid:
                from ...social.chat.base import (
                    ChatKind as _CK, ChatRef as _CR)
                try:
                    _sent = self.gateway.send(
                        message.chat.platform,
                        _CR(platform=message.chat.platform,
                            chat_id=_uid, kind=_CK.DM),
                        text, fresh=True)
                except Exception:  # noqa: BLE001 - fall to pointer
                    _sent = None
                if _sent is not None and getattr(_sent, "ok", False):
                    return _sent
            return self.gateway.send(
                message.chat.platform, message.chat,
                "that's yours alone — check your DMs with me.",
                fresh=True)
        self._typing_for(message.chat, text)
        parse_mode = "HTML" if ("<b>" in text or "<code>" in text
                                or "<i>" in text) else ""
        if session:
            _was_media = self._live_session_registry().is_media(
                str(message.chat.key or ""), session)
            edited = self._try_session_edit(
                message, session, session_edit=session_edit,
                text=text, buttons=btn_rows or [],
                parse_mode=parse_mode, media=_was_media)
            if edited is not None:
                return edited
        if media is not None:
            sent_media = self._send_reply_media_head(
                message, media, text, btn_rows or [],
                parse_mode=parse_mode)
            if sent_media is not None:
                if session:
                    _mid = str(getattr(sent_media, "message_id", "")
                               or "")
                    if _mid:
                        self._live_session_registry().record(
                            str(message.chat.key or ""), session,
                            _mid, media=True)
                return sent_media
        result = self.gateway.send(message.chat.platform, message.chat,
                                   text, buttons=buttons,
                                   parse_mode=parse_mode,
                                   reply_keyboard=reply_keyboard,
                                   # Solicited control reply to a live
                                   # turn: its first dispatch attempt is
                                   # fresh — a send-once guard may hold
                                   # retries, never the answer itself.
                                   fresh=True)
        if session and result is not None \
                and getattr(result, "ok", False):
            _mid = str(getattr(result, "message_id", "") or "")
            if _mid:
                self._live_session_registry().record(
                    str(message.chat.key or ""), session, _mid)
        return result

    def _live_session_registry(self):  # type: ignore[no-untyped-def]
        """The runtime's one live-session map (presentation
        architecture): (chat, session) → the platform id of the
        session's ONE live message. Lazily built like the other
        runtime registries; in-memory — a restart forgets it and
        the next render posts fresh (see live_session)."""
        registry = getattr(self, "_live_sessions_obj", None)
        if registry is None:
            from ...social.chat.live_session import LiveSessionRegistry

            registry = self._live_sessions_obj = LiveSessionRegistry()
        return registry

    def _send_reply_media_head(self, message, media, text,  # type: ignore[no-untyped-def]
                               buttons, *, parse_mode=""):
        """Send a reply's declared media head through the
        surface adapter's native media verb — photo with the
        reply text as caption and the reply's keyboard riding
        it.  Returns the SendResult on success, None to fall
        back to the text send: a caption over the surface's
        caption limit, an adapter without a button-carrying
        media verb, or a failed send all degrade honestly —
        the reply still lands as text, never lost."""
        try:
            from ...social.chat.base import MediaRef
            from ...social.chat.platform_limits import for_platform

            platform = str(message.chat.platform or "")
            limits = for_platform(platform)
            if limits is not None and len(text) > limits.max_caption:
                return None
            adapter = (getattr(self.gateway, "adapters", {})
                       or {}).get(platform)
            fn = getattr(adapter, "send_media", None) \
                if adapter is not None else None
            if fn is None:
                return None
            import inspect as _inspect

            params = _inspect.signature(fn).parameters
            if "buttons" not in params:
                return None
            ref = MediaRef(path=str(media.get("path") or ""),
                           kind=str(media.get("kind") or "image"),
                           name="menu")
            result = fn(message.chat, ref, caption=text,
                        buttons=buttons or None,
                        parse_mode=parse_mode)
            if result is not None and getattr(result, "ok", False):
                return result
            return None
        except Exception as exc:  # noqa: BLE001 - degrade to text
            record_failure(
                "runtime.PartnerRuntime._send_reply_media_head", exc)
            return None

    def _try_session_edit(self, message, session, *,  # type: ignore[no-untyped-def]
                          session_edit, text, buttons, parse_mode,
                          media=False):
        """Render a live-session state change by EDITING the
        session's live message instead of posting a follow-up.
        Returns a SendResult when the edit landed; None falls
        through to the fresh send — a refused edit is never
        reported as an edit (the fallback send re-records the
        live message). Never raises."""
        try:
            from ...social.chat.base import SendResult
            from ...social.chat.live_session import edit_session_message, edit_target
            from ...social.chat.platform_limits import for_platform

            limits = for_platform(message.chat.platform)
            if limits is None or not limits.supports_text_edit:
                return None
            registry = self._live_session_registry()
            target = edit_target(registry, message, session,
                                 session_edit=session_edit)
            if not target:
                return None
            adapter = (getattr(self.gateway, "adapters", {})
                       or {}).get(message.chat.platform)
            if adapter is None:
                return None
            if not edit_session_message(
                    adapter, message.chat, target, text,
                    buttons=buttons, parse_mode=parse_mode,
                    media=media):
                return None
            registry.record(str(message.chat.key or ""), session,
                            target, media=media)
            return SendResult(ok=True,
                              platform=message.chat.platform,
                              message_id=target)
        except Exception as exc:  # noqa: BLE001 - degrade to the send
            record_failure(
                "runtime.PartnerRuntime._try_session_edit", exc)
            return None

    def handle_control(self, text: str, chat_key: str,
                       message: Any = None, _no_stack: bool = False) -> str:
        """Control-command entry point: every PARSED command answers.

        The dispatch lives in :meth:`_handle_control_inner`; this
        wrapper is the no-empty class fix (owner order 2026-11-05,
        register #32: ``/sudo`` met total silence in the owner's
        WhatsApp self-chat — the kind had no parse entry, and several
        handler shapes return ``""` as a success sentinel). A parsed
        command whose dispatch produced an empty reply gets an
        honest plain-words fallback — named, journaled, never
        silent. The ``""` contract for a NON-command (parse None —
        ordinary chat text) is preserved untouched.
        """
        from ...social.chat.control import parse_control as _parse
        delivered_before = _control_deliveries.get()
        reply = self._handle_control_inner(
            text, chat_key, message=message, _no_stack=_no_stack)
        command = _parse(text)
        if command is None or command.kind == "error":
            return reply
        empty = reply is None or (
            isinstance(reply, str) and not reply.strip()) or (
            not isinstance(reply, str)
            and not str(getattr(reply, "text", "") or "").strip())
        if not empty:
            return reply
        if _control_deliveries.get() > delivered_before:
            # The handler answered by SENDING during this dispatch
            # (the _send_long_checked discipline: "" means the answer
            # was delivered; a failed delivery returns its own ⚠️
            # note, never ""). The #32 law targets silence, not a
            # delivered answer — converting this "" would post a
            # false "bug on my end" after a successful send (the
            # self-sending shape: /vision, /code, /search, /swarm,
            # /evolve, long reports — nav audit, Devon half).
            return reply
        from ...social.chat.expression import (
            MEDIA_DELIVERED_KINDS,
            REACTION_GIF_COMMANDS,
        )

        if command.kind in MEDIA_DELIVERED_KINDS \
                or command.kind in REACTION_GIF_COMMANDS:
            # These handlers answer by SENDING media — the
            # GIF / sticker / reaction WAS the reply, and ""
            # is the only shape their success takes (see
            # expression.MEDIA_DELIVERED_KINDS).  The #32 law
            # targets silence, not a delivered media message;
            # converting this "" into the fallback text would
            # post a false "bug on my end" after every
            # successful send (probe-proven on /gif, 2026-11-05).
            return reply
        try:
            record_failure(
                "runtime.PartnerRuntime.handle_control:empty_reply",
                ValueError(
                    f"/{command.kind} produced an empty control reply "
                    f"(chat {chat_key})"))
        except Exception as exc:  # noqa: BLE001 - journaling never breaks a reply
            _log.debug("empty-reply journal failed: %s", exc)
        return (
            f"/{command.kind} ran but produced no reply — that's a "
            f"bug on my end, not something you did. "
            f"/help {command.kind} shows what it should do.")

    def _handle_control_inner(self, text: str, chat_key: str,
                              message: Any = None,
                              _no_stack: bool = False) -> str:
        """Parse + dispatch one control command (or a stack). Returns the reply.

        A stacked message (one command per line — see
        ``split_command_stack``) runs its steps in order via
        :meth:`_run_command_stack` and returns one combined reply.
        ``_no_stack`` is set by the stack runner itself: a ``{prev}``
        expansion may contain newlines from a previous step's output,
        and those must never re-split into injected commands.

        A registered kind that has no dispatch branch can only happen through
        a code inconsistency (``parse_control`` only returns registered kinds);
        it falls back to an ``"unknown command /<kind>"`` reply rather than
        raising, so chat never dies on a dispatch-table gap.
        """
        from ...social.chat.control import (
            CONFIRMABLE_COMMANDS, ControlCommand, GAME_START_COMMANDS,
            TELEGRAM_CLEANUP_KINDS, TELEGRAM_P5_KINDS,
            WHATSAPP_UNSUPPORTED_KINDS,
            caller_may_run, detailed_help, parse_control,
            public_menu_text)
        from ..power import power_mode_for

        # Bare /start is a menu trigger, not the /start <platform>
        # hot-start (which needs an argument). Checked before
        # parse_control, which would error on the missing argument.
        if text.strip() == "/start":
            # Personal bot (owner-only surface): the hub card, whose
            # inline buttons the adapter derives from the card's
            # header (gated path) / the hub origin (personal path).
            # Everywhere else the public menu stays — the gated bot
            # never shows owner-only tools to anyone.
            _platform = (chat_key or "").partition(":")[0].strip().lower()
            if _platform == "telegram-personal":
                from ...social.chat.hub import hub_text
                return hub_text()
            if _platform in ("whatsapp", "wa"):
                # WhatsApp gets its OWN menu (owner rule): the
                # public menu is the Telegram-shaped one — wrong
                # command set for this surface, and it tells the
                # reader to tap a button WhatsApp doesn't have.
                from ...social.chat.control import help_menu
                return help_menu("whatsapp")
            return public_menu_text()

        # Natural phrasing for the representation review: "/how did I do"
        # Natural phrasing for the representation review: "/how did I do"
        # (and variants) routes to the review command.  Checked before
        # parse_control because multi-word kinds aren't slash commands.
        _how = text.strip().lower().rstrip("?").rstrip()
        if _how in ("/how did i do", "/how did i do?",
                    "how did i do", "how did i do?"):
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_review("", chat_key=chat_key)

        # Living-card taps (Living Chat P2): a card button's
        # payload is a fencing stamp, not a command —
        # "cardtap <card_id> <revision> <index>". Checked before
        # the catalog parse (the stamp is deliberately NOT a
        # catalog kind): resolution reads the DURABLE card, so a
        # tap cut at a superseded revision answers with the
        # current card state instead of executing stale intent,
        # and the first tap after a restart resolves exactly
        # like the last one before it.
        _tap = _CARD_TAP_RE.match(text.strip())
        if _tap is not None:
            return self._living_card_tap(
                _tap.group(1), int(_tap.group(2)),
                int(_tap.group(3)), chat_key, message=message)

        # Command stacking: one message, one command per line (see
        # split_command_stack). Checked before the single-command parse
        # so a stack whose first line would fail the whole-text arg
        # count (e.g. /status followed by /mood) still stacks.
        if not _no_stack:
            from ...social.chat.control import (
                MAX_STACK_DEPTH, split_command_stack)
            _segments = split_command_stack(text)
            if _segments is not None and len(_segments) > 1:
                if len(_segments) > MAX_STACK_DEPTH:
                    return (f"stack too deep: {len(_segments)} commands, "
                            f"max is {MAX_STACK_DEPTH} — split it into "
                            "smaller stacks. Nothing ran.")
                return self._run_command_stack(_segments, chat_key, message)

        command = parse_control(text)
        if command is None:
            return ""  # not actually a control command; let normal flow handle it
        kind, arg = command.kind, command.arg
        if kind == "error":
            return arg

        # Bare /menu is a menu trigger, not /list (which would leak the
        # owner-only catalog to non-owners). /menu <group> still goes to
        # list_catalog below — and the telegram-bot gate redirects it.
        if kind == "menu" and not command.tail:
            # WhatsApp gets its OWN menu (owner rule) — same
            # reasoning as bare /start above: never the
            # Telegram-shaped public menu on this surface.
            if (chat_key or "").partition(":")[0].strip().lower() \
                    in ("whatsapp", "wa"):
                from ...social.chat.control import help_menu
                return help_menu("whatsapp")
            # Bot-DM button surfaces get the live menu session's home
            # (one message; categories page inside it) — the text menu
            # stays for groups and buttonless surfaces.
            home = self._menu_session_reply(message, chat_key, command)
            if home is not None:
                return home
            return public_menu_text()

        # The originating surface, from the chat key — one computation,
        # used by the grant denial below, the SMS surface gate and the
        # catalog dispatch (all surface decisions key off it).
        platform = (chat_key or "").partition(":")[0].strip().lower()

        # Capability grant: every command checks the caller's grant —
        # deny without grant. message=None means an internal call
        # (tour/heal/self-heal probe), which runs as the owner.
        is_owner = True if message is None else self._is_operator(message)
        # Group governance decisions (approve/deny of treasury
        # proposals) are the GROUP's commands, not the owner's: in
        # a group chat the grant passes and the handler enforces
        # the admin rule — a proposal the bot itself posted in the
        # room must be decidable in the room (owner report
        # 2026-11-07: the public surface refused the very /approve
        # its own gift proposal instructed).
        from ...social.chat.control import GROUP_DECISION_COMMANDS

        _group_decision = (
            kind in GROUP_DECISION_COMMANDS
            and message is not None
            and self._group_key_or_none(message) is not None)
        if not caller_may_run(kind, is_owner) and not _group_decision:
            # On the gated public surface the honest denial names the
            # surface, not the person: the owner is denied there too
            # (operator power is a surface property — see _is_operator).
            if is_gated_surface(platform):
                return _PUBLIC_BOT_NOT_AVAILABLE
            return "that one's just for the owner."

        # Per-chat feature gate: the shared choke point for every slash
        # command that reaches dispatch — owner DM, public bot, stack
        # steps (each re-enters here), direct handle_control calls. A
        # disabled feature in this chat never runs; the caller gets the
        # honest refusal, not silence and not a run. /chatmode itself is
        # exempt (it is the control plane). message=None is an internal
        # probe (tour/heal), not chat traffic — it bypasses the gate so
        # probes measure the command, not the chat's toggles.
        if message is not None and kind != "chatmode":
            _feature = _chat_modes_feature_for(kind)
            if (_feature is not None
                    and _chat_modes_is_feature_disabled(chat_key, _feature)):
                return _chat_modes_disabled_reply(_feature)
            # Per-game toggle (D5): a single game off in this chat —
            # the game-specific slash command (e.g. /chess) is refused
            # honestly; /game <name> is gated in _control_game, plain
            # moves in _route_game_move.
            from ...social.chat.control import GAME_COMMANDS as _GAME_CMDS
            if (kind in _GAME_CMDS
                    and _chat_modes_is_game_disabled(chat_key, kind)):
                return _chat_modes_game_disabled_reply(kind)

        # Confirm-flow gate for CONFIRMABLE commands (quit/forget):
        # bare request → prompt + store; confirm → dispatch with the
        # stored original tail; cancel/expired → honest message.
        if kind in CONFIRMABLE_COMMANDS:
            reply, proceed, effective = self._confirm_gate(
                kind, command.tail, chat_key)
            if not proceed:
                return reply or ""
            command = ControlCommand(
                kind=kind,
                arg=effective.split()[0] if effective.strip() else "",
                tail=effective)

        # Progression-gated commands: beating challenges unlocks commands.
        # Checked before dispatch; the owner bypasses (their capabilities
        # come from role config — progression is additive for players only).
        locked = self._progression_gate(kind, message)
        if locked is not None:
            return locked

        # WhatsApp has no per-person mute (owner ruling
        # 2026-11-04): the mute family's handlers bottom out in
        # admin_restrict / admin_unrestrict, which WhatsApp
        # cannot execute — a command the platform can't run
        # doesn't sit on its surface.  Typed anyway (group or
        # DM), it gets the platform truth in one line, here at
        # dispatch: it never enters _mod_run or the warn
        # machinery, and no moderation attempt is audited for
        # an execution the platform cannot perform.
        if platform in ("whatsapp", "wa") \
                and kind in WHATSAPP_UNSUPPORTED_KINDS:
            if kind in TELEGRAM_CLEANUP_KINDS:
                # Member cleanup is Telegram machinery: it
                # reads Telegram's member list and its
                # deleted-account markers.  WhatsApp exposes
                # neither — same platform-truth treatment as
                # the mute family, its own line.
                return ("group cleanup is a Telegram tool — "
                        "WhatsApp gives me no member list to "
                        "scan and no deleted-account markers "
                        "to find.")
            if kind in TELEGRAM_P5_KINDS:
                # The §16 P5 kinds are Telegram affordances
                # (member tags, channel-identity bans, reaction
                # wipes, boosts) — WhatsApp has none of the
                # four systems underneath them.
                return ("that's a Telegram tool — WhatsApp "
                        "has no member tags, channel posts, "
                        "reaction wipes or boosts to run it on.")
            return ("WhatsApp has no per-person mute — a "
                    "member here can only be removed, or left "
                    "as they are.")

        # SMS fallback surface (build-map #19): restricted command set over
        # SMS — games, media, exec and everything heavy get an honest
        # "needs the app" reply instead of running.
        if platform == "sms":
            from ...social.chat.sms import SMS_BLOCKED_REPLY, sms_allows

            if not sms_allows(kind):
                return SMS_BLOCKED_REPLY

        # Default chat ref for every dispatch branch.  Several branches
        # (miniapp/gtrip/upgrade/...) rebind `chat` from the message for a
        # richer ref; branches that don't (watch/aeo/passport/...) previously
        # crashed with UnboundLocalError.  This default keeps them working.
        chat = self._ref_from_key(chat_key)

        if kind in {"list", "commands", "menu"}:
            # wave 68: /list — every executable chat command, categorized
            # with a one-line "what it does"; /list <group> filters.
            # On the gated public surface the catalog is the PUBLIC
            # list only — the full catalog names owner-only commands
            # (platform control, money, trial tooling) that must never
            # be advertised there.
            #
            # Surfaces with buttons + text edit get the LIVE menu
            # instead (2026-11-06 menufix; groups joined it under
            # the 2026-11-07 paging law): one message, paged in
            # place via the menu session — the catalog text below
            # stays the path for WhatsApp and buttonless surfaces.
            session_reply = self._menu_session_reply(
                message, chat_key, command)
            if session_reply is not None:
                return session_reply
            from ...social.chat.control import list_catalog
            return list_catalog(arg, public_only=is_gated_surface(platform))
        if kind == "help":
            # wave 67: detailed help — /help for the catalog,
            # /help <command> for one command's full page, /help budget
            # etc. for topic pages.
            # Bare /help inside a GROUP lists what is available in
            # that group — member and admin surfaces alike, from the
            # group's live state (owner order 2026-11-03).  A topic
            # still gets its full command page.  The group check
            # comes FIRST on every surface: on WhatsApp the
            # platform check below used to pre-empt it, so a
            # WhatsApp group got the DM menu and group help was
            # unreachable there (command census, gap #5).
            if not (arg or command.tail) \
                    and self._group_key_or_none(message) is not None:
                return self._control_group_help(message)
            # A bare /help on a buttons+edit surface opens the same
            # live menu home as /menu (2026-11-07 paging law): the
            # catalog's front door pages inside one message instead
            # of arriving as the multi-message text dump. A topic
            # keeps its full command page below; the session helper
            # self-gates (WhatsApp, buttonless, message=None keep
            # the paths they have always had).
            if not (arg or command.tail):
                session_reply = self._menu_session_reply(
                    message, chat_key, command)
                if session_reply is not None:
                    return session_reply
            # WhatsApp gets its OWN menu (owner rule): the bare catalog
            # on WhatsApp serves the WhatsApp-native, text-first menu —
            # control.help_menu existed for exactly this but had no
            # production caller, so WhatsApp users got the
            # Telegram-styled catalog. Topic pages stay detailed_help
            # (content; the outbound formatter converts per platform).
            if platform in ("whatsapp", "wa") and not (arg or command.tail):
                from ...social.chat.control import help_menu
                return help_menu("whatsapp")
            return detailed_help(arg or command.tail,
                                 public_only=is_gated_surface(platform))

        if kind == "tour":
            # smoke-test every command with safe read-only probes.
            return self._control_tour(command.tail or arg, chat_key=chat_key)

        if kind == "heal":
            # self-healing loop: diagnose + auto-fix broken commands.
            return self._control_heal(command.tail or arg, chat_key=chat_key)

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

        if kind == "ping":
            return self._control_ping(message)

        if kind == "sudo":
            return self._control_sudo(command.tail, message)

        if kind == "pair":
            return self._control_pair(command.tail, chat_key=chat_key,
                                      message=message)

        if kind in ("massreact", "pollvote", "chfollow"):
            # §14 mass tools over the §15 pool — owner-only like
            # /pair (control dispatch never reaches non-operators).
            return self._control_mass(kind, command.tail)

        if kind == "rules":
            return self._control_rules(message)

        if kind == "member":
            return self._control_member(arg, message)

        if kind == "glog":
            return self._control_glog(arg, message)

        if kind == "strikes":
            return self._control_strikes(arg, message)

        if kind == "recheck":
            return self._control_recheck(message)

        if kind == "joinq":
            return self._control_joinq(command.tail, message)
        if kind == "console":
            return self._control_console(command.tail, message)
        if kind == "undonotice":
            return self._control_undonotice(command.tail, message)
        if kind == "flag":
            return self._control_flag(command.tail, message)
        if kind == "rep":
            return self._control_rep(command.tail, message)
        if kind == "repclear":
            return self._control_repclear(command.tail, message)
        if kind == "repdismiss":
            return self._control_repdismiss(command.tail, message)

        if kind == "gsettings":
            return self._control_gsettings(command.tail, message)

        if kind == "balance":
            reply = self._control_balance(message)
            # §16 ephemeral answers: a sensitive kind's group
            # reply goes only to the asker when the group's p5
            # settings say so (bot surface).  The stamp is read
            # by _send_control_reply, which owns the fallback
            # ladder (ephemeral → DM → pointer) — the reply is
            # never silently public.
            if platform == "telegram-bot":
                try:
                    _key = self._group_key_or_none(message)
                    if _key is not None and (
                            self._group_surface().store.settings(
                                _key).get("p5") or {}).get(
                                    "ephemeral") \
                            and isinstance(
                                getattr(message, "meta", None),
                                dict):
                        message.meta["ephemeral_reply"] = {
                            "user_id": str(message.sender_id or "")}
                except Exception:  # noqa: BLE001 - delivery detail
                    pass            # never blocks the reply itself
            return reply

        if kind == "mine":
            return self._control_mine(message)

        if kind == "work":
            return self._control_work(message)

        if kind == "pay":
            return self._control_pay(command.tail, message)

        if kind == "gshop":
            return self._control_gshop(command.tail, message)

        if kind == "quests":
            return self._control_quests(command.tail, message)

        if kind == "treasury":
            return self._control_treasury(message)

        if kind == "drop":
            return self._control_drop(command.tail, message)

        if kind == "claim":
            return self._control_claim(message)

        if kind == "deposit":
            return self._control_deposit(command.tail, message)

        if kind == "withdraw":
            return self._control_withdraw(command.tail, message)

        # /convert is a CONTEXT kind (§17 pattern): the §21
        # media command owns it everywhere — except in a group,
        # where an amount-shaped tail is the §25 currency
        # exchange (games coins ⇄ group sparks).  Media tails
        # name a conversion and never match the guard, so the
        # media meaning is untouched in groups too.
        if kind == "convert" \
                and self._group_key_or_none(message) is not None \
                and self._convert_tail_is_currency(command.tail):
            return self._control_exchange_convert(command.tail,
                                                  message)

        # daily / slots / approve are CONTEXT kinds: in a group
        # they are the economy's (daily claim, purse slots,
        # treasury approvals); everywhere else the existing
        # owners of those names (arena daily, game slots,
        # proposal approvals) keep them — fall through untouched.
        if kind == "daily" \
                and self._group_key_or_none(message) is not None:
            return self._control_group_daily(message)

        if kind == "rob":
            return self._control_rob(command.tail, message)

        if kind == "bail":
            return self._control_bail(message)

        # The conflict layer (§23/§24/§35): group-only kinds —
        # in DM they meet the economy gate's group text like
        # every other group kind.
        if kind == "gear":
            return self._control_conflict_gear(command.tail, message)

        if kind == "heist":
            return self._control_heist(command.tail, message)

        if kind == "shoot":
            return self._control_shoot(command.tail, message)

        if kind == "kill":
            return self._control_kill(command.tail, message)

        if kind == "assassinate":
            return self._control_assassinate(command.tail,
                                             message)

        if kind == "sabotage":
            return self._control_sabotage(command.tail, message)

        if kind == "gang":
            return self._control_gang(command.tail, message)

        if kind == "war":
            return self._control_war(command.tail, message)

        if kind == "police":
            return self._control_police(command.tail, message)

        if kind == "bounty":
            return self._control_bounty(command.tail, message)

        if kind == "claimdrop":
            return self._control_claimdrop(command.tail, message)

        if kind == "poll":
            return self._control_poll(command.tail, message)

        if kind == "vote":
            return self._control_vote(command.tail, message)

        if kind == "milestones":
            return self._control_milestones(command.tail, message)

        if kind == "contributions":
            return self._control_contributions(command.tail, message)

        if kind == "boards":
            return self._control_boards(command.tail, message)

        if kind == "webshot":
            return self._control_webshot(command.tail, message)

        if kind == "webrec":
            return self._control_webrec(command.tail, message)

        if kind == "audit":
            return self._control_audit(command.tail, message)

        if kind == "owner":
            return self._control_owner(command.tail, message)

        if kind == "slots" \
                and self._group_key_or_none(message) is not None:
            return self._control_slots(command.tail, message)

        if kind == "coinflip":
            return self._control_coinflip(command.tail, message)

        if kind == "top":
            return self._control_top(message)

        if kind == "tax":
            return self._control_tax(message)

        if kind == "grant":
            return self._control_grant(command.tail, message)

        if kind == "approve" \
                and self._group_key_or_none(message) is not None:
            return self._control_approve(command.tail, message)

        if kind == "deny" \
                and self._group_key_or_none(message) is not None:
            return self._control_deny(command.tail, message)

        # Real estate (§17 slice 2 / §20): group-only kinds —
        # in DM they fall to the unknown-command honesty like
        # every other group kind's gate text.
        if kind == "property":
            return self._control_property(command.tail, message)

        if kind == "buyproperty":
            return self._control_buyproperty(command.tail, message)

        if kind == "sellproperty":
            return self._control_sellproperty(command.tail,
                                              message)

        if kind == "rent":
            return self._control_rent(message)

        if kind == "listproperty":
            return self._control_listproperty(command.tail,
                                              message)

        if kind == "auctions":
            return self._control_auctions(command.tail, message)

        if kind == "mortgage":
            return self._control_mortgage(command.tail, message)

        if kind == "insure":
            return self._control_insure(command.tail, message)

        if kind == "mute":
            return self._control_mute(command.tail, message)

        if kind == "unmute":
            return self._control_unmute(command.tail, message)

        if kind == "ban":
            return self._control_ban(command.tail, message)

        if kind == "unban":
            return self._control_unban(command.tail, message)

        if kind == "kick":
            return self._control_kick(command.tail, message)

        if kind == "warn":
            return self._control_warn(command.tail, message)

        if kind == "pin":
            return self._control_pin(command.tail, message)

        if kind == "unpin":
            return self._control_unpin(command.tail, message)

        if kind == "purge":
            return self._control_purge(command.tail, message)

        if kind == "report":
            return self._control_report(command.tail, message)

        if kind == "tag":
            return self._control_tag(command.tail, message)

        if kind in ("bansender", "unbansender"):
            return self._control_bansender(command.tail, message,
                                           unban=kind == "unbansender")

        if kind == "reactwipe":
            return self._control_reactwipe(command.tail, message)

        if kind == "boosters":
            return self._control_boosters(command.tail, message)

        if kind == "cleanup":
            return self._control_cleanup(command.tail, message)

        if kind == "rmdeleted":
            return self._control_rmdeleted(command.tail, message)

        if kind == "rminactive":
            return self._control_rminactive(command.tail, message)

        if kind == "platforms":
            lines = ["platforms:"]
            status = self.gateway.status()
            # Platforms under an owner stop order render AS stopped
            # — a distinct truth from merely "not running": the
            # reconcile is holding them down on purpose, and only
            # an owner start lifts it.
            stopped_by_owner: set[str] = set()
            try:
                registry = self._pairing_supervisor().registry
                for pname in status:
                    if not pname.startswith("_") and \
                            registry.owner_stopped(pname):
                        stopped_by_owner.add(pname)
            except Exception:  # noqa: BLE001 - listing never dies
                stopped_by_owner = set()
            # WhatsApp's truth includes its bridge: an adapter can
            # be "running" (thread alive, reconnecting) while the
            # bridge process behind it is dead — the boot that
            # live-attached to a dead port read exactly like a
            # healthy surface here.  When the pool's resolved
            # account has no serving bridge, the line says
            # "bridge down" with the reason, never a bare
            # "running" / "not running".
            wa_bridge_down = ""
            try:
                from ...social.pairing import resolve_account

                sup = self._pairing_supervisor()
                wa_rec = resolve_account(sup.registry, "whatsapp")
                if wa_rec is not None \
                        and "whatsapp" not in stopped_by_owner \
                        and not sup.bridge_serving(wa_rec):
                    reason = sup.pairing_error(wa_rec.id) \
                        or (wa_rec.last_error or "")
                    prefix = getattr(
                        self, "_WA_SURFACE_NOTE_PREFIX", "")
                    if prefix and reason.startswith(prefix):
                        reason = reason[len(prefix):]
                    wa_bridge_down = reason or (
                        "the bridge process is not running")
            except Exception:  # noqa: BLE001 - listing never dies
                wa_bridge_down = ""
            seen: set[str] = set()
            for name, info in sorted(status.items()):
                if name.startswith("_") or not isinstance(info, dict):
                    continue
                seen.add(name)
                if name in stopped_by_owner and \
                        not info.get("running_in_session"):
                    lines.append(
                        f"  {name}: stopped (by you) — "
                        f"/start {name} to resume")
                    continue
                if name == "whatsapp" and wa_bridge_down and \
                        not info.get("connected"):
                    lines.append(
                        f"  whatsapp: bridge down — "
                        f"{wa_bridge_down}")
                    continue
                state = "running" if info.get("running_in_session") else "not running"
                suffix = "" if info.get("running") else " (adapter thread down)"
                if (state == "running" and is_personal_platform(name)
                        and self._get_brain_gate().episode_open):
                    # Competence gate state, honestly on the surface:
                    # the platform runs, but personal replies are held
                    # while no model rung answers (brain_gate).
                    suffix += " — observe-only (brain unavailable)"
                if is_personal_platform(name):
                    # Always-online truth (§15 presence pack) rides
                    # the same line when the owner flagged the
                    # platform's account — asserted, or why not.
                    presence_suffix = getattr(
                        self, "_presence_suffix", None)
                    if callable(presence_suffix):
                        suffix += presence_suffix(name)
                lines.append(f"  {name}: {state}{suffix}")
            # Known-but-never-built platforms (not in the boot set,
            # never hot-started) are invisible in gateway.status()
            # — list them from the catalogue with the way to start
            # them, so the switch is findable from this surface.
            try:
                known = self.gateway.known_platforms()
            except Exception:  # noqa: BLE001 - listing never dies
                known = []
            try:
                registry = self._pairing_supervisor().registry
                for pname in known:
                    if registry.owner_stopped(pname):
                        stopped_by_owner.add(pname)
            except Exception:  # noqa: BLE001 - listing never dies
                pass
            for name in known:
                if name not in seen:
                    if name in stopped_by_owner:
                        lines.append(
                            f"  {name}: stopped (by you) — "
                            f"/start {name} to resume")
                    elif name == "whatsapp" and wa_bridge_down:
                        lines.append(
                            f"  whatsapp: bridge down — "
                            f"{wa_bridge_down}")
                    else:
                        lines.append(
                            f"  {name}: not running — "
                            f"start with /start {name}")
            return "\n".join(lines)

        if kind == "start":
            result = self.start_platform(arg)
            return (f"{arg}: started" if result.get("ok")
                    else f"{arg}: {result.get('error', 'could not start')}")

        if kind == "stop":
            result = self.stop_platform(arg)
            return (f"{arg}: stopped" if result.get("ok")
                    else f"{arg}: {result.get('error', 'could not stop')}")

        if kind in {"update", "pull", "deploy", "restart"}:
            # Step 9: chat self-update — drain → checkpoint → restart → resume.
            # Owner-only: caller_may_run already denied non-owners above
            # (these kinds are absent from PUBLIC_COMMANDS).
            from pathlib import Path

            from ...missions.mission import MissionStore
            from ...os.snapshots import SnapshotManager
            from ...os.update import UpdateManager
            from ...social.chat.update_control import ChatUpdateController

            try:
                settings = self.context.settings
                repo_root = Path(__file__).resolve().parents[3]
                snapshots = SnapshotManager(
                    home=settings.home_path, db_path=settings.db_path,
                    blob_dir=getattr(settings, "blob_dir", None))
                updater = UpdateManager(
                    repo_root, snapshots,
                    quiesce=self._quiesce_bridges_for_update)
                controller = ChatUpdateController(
                    mission_store=MissionStore(self.context.db),
                    update_runner=lambda pull: updater.run(pull=pull))
                # Restart journey (doc 4, Living Chat P2): the
                # journey card opens BEFORE the work, at preflight,
                # in the chat that asked — every later phase
                # revises this one card.
                journey = self._open_restart_card(
                    message, chat_key, kind)
                try:
                    result = controller.execute(kind)
                finally:
                    # The update is over either way: bridge
                    # relaunches are the reconcile's business again
                    # (on success the restart below replaces this
                    # process and the hold would die with it).
                    self._release_bridge_launch_hold()
            except Exception as exc:  # noqa: BLE001 - update wiring failure is a reply
                return f"/{kind} failed to start: {type(exc).__name__}: {exc}"
            # The journey follows the outcome: the report advances
            # preflight/swap; a journey that ends WITHOUT a restart
            # (a pull, an up-to-date tree, a refusal) closes here,
            # honestly — only a restart handoff stays open, and
            # only the NEXT process may close that.
            if journey is not None:
                from ...social.chat.living_cards import advance_restart_card

                _report = result.get("update_report")
                if _report is not None:
                    advance_restart_card(journey, _report)
                    journey = self._revise_restart_card(
                        journey, message, body=journey.body,
                        status=journey.status, state=journey.state)
                if not result.get("restart_requested"):
                    if _report is None and not result.get("ok"):
                        journey = self._revise_restart_card(
                            journey, message, status="attention",
                            terminal=True,
                            body=("the update did not run: "
                                  + str(result.get("message", ""))
                                  ))
                    elif _report is not None:
                        journey = self._revise_restart_card(
                            journey, message,
                            status=("applied" if result.get("ok")
                                    else "attention"),
                            terminal=True,
                            body=(str(result.get("message", ""))
                                  ))
            if result.get("restart_requested"):
                # Record the restart intent BEFORE this process dies:
                # the reply below promises a restart, and only the
                # NEXT process can confirm it actually came up on the
                # version the update landed (boot consumes the intent
                # in _recover_interrupted_update). A plain /restart
                # carries no update report and writes no intent.
                report_obj = result.get("update_report")
                if report_obj is not None:
                    try:
                        updater.note_restart_intent(report_obj)
                    except Exception:  # noqa: BLE001 - never blocks the restart
                        _log.warning("could not record the update "
                                     "restart intent", exc_info=True)
                # The handoff lands on the journey card BEFORE
                # this process dies — and that is as far as this
                # process may write: the confirmed phase belongs
                # to the process that comes up (advance_restart_card
                # structurally cannot write it).
                if journey is not None:
                    from ...social.chat.living_cards import advance_restart_card

                    advance_restart_card(journey, phase="restart")
                    self._revise_restart_card(
                        journey, message, body=journey.body,
                        status=journey.status, state=journey.state)
                # The reply goes out first; the restart thread stops the
                # runtime and re-execs the same argv. Boot resumes from
                # the checkpoints the controller just took.
                import os as _os
                import sys as _sys

                def _restart() -> None:
                    try:
                        time.sleep(2.0)
                        self.stop()
                    finally:
                        _os.execv(_sys.executable,
                                  [_sys.executable, *_sys.argv])
                threading.Thread(target=_restart, name="partner-restart",
                                 daemon=True).start()
            return str(result.get("message", ""))

        if kind == "mood":
            return self._control_mood(command.tail or arg)

        if kind == "mode":
            return self._control_mode(arg)

        if kind == "agent":
            return self._control_agent(command.tail or arg, chat_key)

        if kind == "chatmode":
            return self._control_chatmode(command, chat_key, message)

        if kind == "style":
            return self._control_style(command, chat_key, message)

        if kind == "stylepin":
            return self._control_stylepin(command.tail or "", message)

        if kind == "pack":
            return self._control_pack(command.tail or "", message)

        if kind == "skills":
            return self._control_skillbook(command.tail or "", message)

        if kind == "delegate":
            return self._control_delegate(command.tail or "", message)

        if kind == "computer":
            return self._control_computer(command.tail or "", message)

        if kind == "doctor":
            return self._control_doctor()

        if kind == "model":
            return self._control_model(command.tail)

        if kind == "providers":
            return self._control_providers()

        if kind == "say":
            # ONE branch for all three /say forms (a second
            # `if kind == "say"` below would be unreachable — the first
            # branch always returns). Routed on the tail shape:
            #   [direction] / accent=… / ambient=… / room=… / emotion=…
            #       → rich voice (_control_say)
            #   <platform:chat> <text> → gateway send as her
            #   <voice> <text> → named-voice TTS (_control_say_tts)
            _tail = (command.tail or "").strip()
            _first = _tail.split(None, 1)[0] if _tail else ""
            _rich = _tail.startswith("[") or (
                "=" in _first
                and _first.split("=", 1)[0].lower()
                in {"accent", "ambient", "room", "emotion", "direction"})
            if _rich:
                return self._control_say(command.tail or arg,
                                         chat_key=chat_key)
            target, sep, payload = command.tail.partition(" ")
            payload = payload.strip()
            if not sep or not payload:
                return ("usage: /say telegram:123 the text to send — "
                        "or /say <voice> <text> to speak it as a voice note — "
                        "or /say [direction] [accent=…] <text> for rich voice")
            if ":" not in target:
                # TTS path: /say <voice> <text> — voice names never contain ":"
                return self._control_say_tts(target, payload, chat_key)
            platform, _, chat_id = target.partition(":")
            if not chat_id:
                return "usage: /say telegram:123 the text to send"
            from ...social.chat.base import ChatRef

            result = self.gateway.send(
                platform, ChatRef(platform=platform, chat_id=chat_id), payload
            )
            return "sent." if result.ok else f"send failed: {result.error}"

        if kind == "proposals":
            # Allow /proposals approve <id> and /proposals deny <id> [reason]
            # as natural subcommands. Deny takes an optional reason (required
            # for upgrade tickets — UpgradeQueue.deny demands one).
            parts = (arg or "").strip().split(None, 2)
            if parts and parts[0] in ("approve", "deny"):
                sub_id = parts[1] if len(parts) > 1 else ""
                if not sub_id:
                    return f"usage: /proposals {parts[0]} <proposal_id>"
                if parts[0] == "approve":
                    result = self.approve(sub_id)
                else:
                    result = self.deny(sub_id, parts[2] if len(parts) > 2 else "")
                if result.get("ok"):
                    return f"{parts[0]}d {sub_id} → {result.get('status', 'done')}"
                err = result.get("error") or "no reason given"
                return f"{parts[0]} failed: {err}"
            rows = self.proposals()
            if not rows:
                return "no pending proposals."
            lines = [f"pending proposals ({len(rows)}):"]
            upgrades = [r for r in rows if r.get("queue") == "upgrade"]
            proactive = [r for r in rows if r.get("queue") != "upgrade"]
            if upgrades:
                lines.append("  upgrade proposals (research → approve → evolve):")
                for row in upgrades[:10]:
                    lines.append(f"    {row.get('id')} {str(row.get('title') or '')[:70]}")
                lines.append("  tip: /upgrade show <id> | /upgrade diff <id> | /upgrade approve <id>")
            if proactive:
                lines.append("  proactive actions:")
                for row in proactive[:10]:
                    lines.append(f"    {row['id']} [{row['kind']}→{row['platform']}:{row['chat_id']}] {str(row['content'])[:70]}")
            lines.append("tip: /proposals approve <id> or /approve <id>  ·  /proposals deny <id> <reason>")
            return "\n".join(lines)

        if kind == "approve":
            result = self.approve(arg)
            if result.get("ok"):
                return f"approved {arg} → {result.get('status', 'sent')}"
            err = result.get("error") or "no reason given"
            return f"approve failed: {err} — /proposals to check what's still pending"

        if kind == "deny":
            sub = (arg or "").strip().split(None, 1)
            pid = sub[0] if sub else ""
            reason = sub[1] if len(sub) > 1 else ""
            result = self.deny(pid, reason)
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
            except Exception as e:
                record_failure("runtime.PartnerRuntime.handle_control", e)
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
            return "usage: /power on <key> | /power on <identity> <passphrase> | /power off | /power status"

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
        if kind == "mandate":
            # Payment mandates: /mandate issue|list|revoke|revoke-all.
            # AA-51 (2026-11-05): this is the ONE dispatch branch.  A
            # second branch further down this chain (to the mixin's
            # _control_mandate) could never execute — this branch
            # returns first — so the dead branch and its duplicate
            # handler are gone.  The operator gate is repeated here
            # in the same idiom as the gated neighbours below:
            # caller_may_run already refuses non-owners for this kind
            # at the top of this function, and the branch keeps the
            # invariant locally rather than depending on the
            # capability table never changing.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...finance.commands import control_mandate

            return control_mandate(command.tail or arg, context=self.context)
        if kind == "balances":
            # Unified money view: /balances [mono,binance]
            from ...finance.commands import control_balances

            return control_balances(command.tail or arg, context=self.context)
        if kind == "alert":
            # Price & money alerts: /alert add|list|remove|check
            from ...finance.commands import control_alert

            return control_alert(command.tail or arg, context=self.context)
        if kind == "goal":
            # Savings goals: /goal add|list|done|remove
            from ...finance.commands import control_goal

            return control_goal(command.tail or arg, context=self.context)
        if kind == "insights":
            # Native money analytics: /insights [days]
            from ...finance.commands import control_insights

            return control_insights(command.tail or arg, context=self.context)
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

            if message is not None:
                chat = message.chat
            return control_miniapp(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "panel":
            # Persistent live-data panels — owner's personal data, owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...community.miniapps import control_panel
            return control_panel(
                command.tail or arg,
                user_id="owner",
                user_name=getattr(message, "sender", "") if message else "",
            )
        if kind == "contract":
            # Contract risk review — owner's private documents, owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...legal.contracts import control_contract
            return control_contract(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "legal":
            # Legal hub — subcommands route to the matching legal control
            # (contract / portfolio / regulatory / aid / research); a bare
            # question keeps the existing aid behaviour (control_legal).
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...legal.chat import control_legal_full
            return control_legal_full(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "learn":
            # One front door over learn/ — owner-only, like /tutor.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...learn.chat import control_learn
            return control_learn(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
                chat_key=chat_key,
            )
        if kind == "impossible":
            # Impossible Made Possible — owner-only, like /immune.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...impossible.chat import control_impossible
            return control_impossible(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
                chat_key=chat_key,
            )
        if kind == "immune":
            # Defensive immune system — owner-only, like /learn.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...immune.chat import control_immune
            return control_immune(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
                chat_key=chat_key,
                event_sink=self._immune_event_sink(),
            )
        if kind == "research":
            # Web research — swarm -> digest -> brief. Owner-only.
            # (Legal research lives under /legal; /research is the web.)
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_web_research(
                command.tail or arg or "", chat_key=chat_key, chat=chat)
        if kind == "contracts":
            # Contract portfolio — owner's private documents, owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...legal.portfolio import control_contracts
            return control_contracts(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "regwatch":
            # Regulatory monitoring — business compliance, owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...legal.regulatory import control_regwatch
            return control_regwatch(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "uprofile":
            # Social profiles + element-level likes — anyone can drive it
            # (profiles belong to their owners; likes are public signals).
            from ...social.profiles import control_uprofile
            return control_uprofile(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "vnote":
            # Voice notes — not owner-gated (anyone in the chat sends them),
            # community (#12) and gig (#1) chats alike.
            from ...social.voice_notes import control_vnote
            return control_vnote(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "match":
            # Curated batches + stable matching — not owner-gated (shared
            # surfaces: gigs, communities, mentoring). Anyone drives it.
            from ...matching.chat import control_match
            return control_match(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "cgroup":
            # Community groups + events + meetups (#84) — not owner-gated
            # (shared surfaces). Anyone drives it.
            from ...community.groups import control_cgroup
            return control_cgroup(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "room":
            # Live audio rooms (#107) — not owner-gated (shared surfaces,
            # community-scoped). Anyone in the community drives it.
            from ...community.audio_rooms import control_room
            return control_room(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "party":
            # Group parties (AA-05) — not owner-gated (shared surfaces,
            # group-scoped). Anyone in the group drives it; trivia wins
            # credit the one group economy through the token facade.
            from ...community.parties import control_party
            return control_party(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "ajo":
            # Rotating savings circles (AA-22) — not owner-gated
            # (shared surfaces, group-scoped). Anyone in the group
            # drives it.
            from ...community.ajo import control_ajo
            return control_ajo(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "challenge":
            # Photo-proof challenges + streaks + squads (#87) — not
            # owner-gated (owner + community). Anyone drives it.
            from ...health.challenges import control_challenge
            return control_challenge(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "scamcheck":
            # Nigerian rental scam check (#93) — not owner-gated
            # (owner + community). Anyone drives it.
            from ...property.scam import control_scamcheck
            return control_scamcheck(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "passport":
            # Rental passport (#94) — owner's private identity data,
            # owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...property.passport import control_passport
            return control_passport(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "watch":
            # Persistent match alerts (#96) — owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...triggers.saved_search import control_watch
            return control_watch(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "aeo":
            # AEO/GEO visibility tracking (#97) — share of answer, owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...marketing.aeo import control_aeo
            return control_aeo(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "send":
            # Self-hosted send layer (#98) — Devon owns the mailgun,
            # owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...marketing.send import control_send
            return control_send(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "leads":
            # Comment → DM → lead loop (#100) — ManyChat pattern, owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...social.leads import control_leads
            return control_leads(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "brief":
            # Brief-first content pipeline (#99) — SERP-derived briefs,
            # owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...marketing.briefs import control_brief
            return control_brief(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "content":
            # Brief-first content pipeline (#99) — score/predict/schedule,
            # owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...marketing.briefs import control_content
            return control_content(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "guardrails":
            # Rule-based ad guardrails (#101) — autonomous-spend safety,
            # owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...marketing.guardrails import control_guardrails
            return control_guardrails(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "competitor":
            # No-access competitor intel (#102) — public data only,
            # owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...marketing.competitor import control_competitor
            return control_competitor(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "cost":
            # Transparent cost display (extension #7) — toggleable
            # per-answer spend footer, today's spend, research budgets.
            # Owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...llm.cost_display import control_cost
            return control_cost(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "truecost":
            # True-cost calculator (#94) — pairs with the passport,
            # owner-only.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...property.passport import control_truecost
            return control_truecost(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "value":
            # DIY AVM (#95) — not owner-gated (owner + community).
            # Estimates inform; humans commit.
            from ...property.value import control_value
            return control_value(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "valuepick":
            # Comp-picker (#95) — not owner-gated (owner + community).
            from ...property.value import control_valuepick
            return control_valuepick(
                command.tail or arg,
                context=self.context,
                chat=chat,
            )
        if kind == "gtrip":
            # Group-travel stack: NOT owner-gated (every member drives it),
            # group chats only — control_gtrip refuses DMs itself.
            from ...travel.groups import control_gtrip

            if message is not None:
                chat = message.chat
            return control_gtrip(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind in {"email", "gmail", "mail", "inbox"}:
            # Full mailbox surface. /gmail and /mail are aliases of /email;
            # bare /inbox lists the inbox instead of running triage.
            # Drafts are never auto-sent; sending needs the owner's
            # explicit "send it" after /email send <id> arms one draft.
            from ..email_triage import control_email

            tail = command.tail or arg
            if kind == "inbox":
                tail = f"inbox {tail}" if tail else "inbox"
            return control_email(tail, context=self.context)
        if kind in {"book", "books"}:
            return self._control_book(tail=command.tail or arg, chat_key=chat_key)
        if kind == "novel":
            return self._control_novel(tail=command.tail or arg, chat_key=chat_key)
        if kind in {"wisdom", "wis"}:
            # WisdomKeeper: /wisdom and its /wis alias share one handler.
            return self._control_wisdom(command.tail or arg, chat_key=chat_key,
                                        message=message)
        if kind == "decode":
            return self._control_decode(command.tail or arg, chat_key=chat_key)
        if kind == "cookies":
            return self._control_cookies(command.tail or arg,
                                         message=message)
        if kind == "cookielab":
            return self._control_cookielab(command.tail or arg)
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
        if kind == "distribute":
            return self._control_distribute(command.tail or arg,
                                           chat_key=chat_key)
        if kind == "play":
            # The owner key binds any pick-list this play opens to
            # the requesting user (Living Chat P1 targeting). It
            # rides a ContextVar into _control_play rather than a
            # new call argument: that seam is doubled in tests and
            # downstream variants with the two-argument shape, and
            # they must keep working unchanged.
            _owner = _pick_owner_key(message) \
                if message is not None else ""
            _tok = play_owner_var.set(_owner)
            try:
                return self._control_play(command.tail or arg,
                                          chat_key=chat_key)
            finally:
                play_owner_var.reset(_tok)
        if kind in {"anime", "manga", "anichar", "charpic", "staff",
                    "studio", "episodes", "animechars", "trailer",
                    "streaming", "animereview", "genres", "genre",
                    "topanime", "topmanga", "popular", "trending",
                    "airing", "upcoming", "season", "topchars",
                    "anischedule", "nextep", "recommend", "randomanime",
                    "randommanga", "randomchar", "animequote", "favorite",
                    "whatanime", "waifu", "neko", "kitsune", "husbando"}:
            # The anime organ (wave B): lookup/schedules/pictures/
            # favorites, one mixin entry for the whole family.
            return self._control_anime(
                kind, command.tail or arg, chat_key=chat_key,
                message=message)
        if kind == "download":
            return self._control_download(command.tail or arg, chat_key=chat_key)
        if kind in {"tiktok", "ttdl"}:
            return self._control_tiktok(command.tail or arg, chat_key=chat_key)
        if kind == "instagram":
            return self._control_instagram(command.tail or arg, chat_key=chat_key)
        if kind == "spotify":
            return self._control_spotify(command.tail or arg, chat_key=chat_key)
        if kind == "aio":
            return self._control_aio(command.tail or arg, chat_key=chat_key)
        if kind == "video":
            return self._control_video(command.tail or arg)
        if kind == "director":
            return self._control_director(command.tail or arg)
        if kind == "caption":
            return self._control_caption(command.tail or arg,
                                         chat_key=chat_key,
                                         message=message)
        if kind == "vision":
            return self._control_vision(command.tail or arg,
                                        chat_key=chat_key,
                                        message=message)
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
        if kind == "dj":
            return self._control_dj(command.tail or arg, chat_key=chat_key)
        if kind == "produce":
            return self._control_produce(command.tail or arg,
                                         chat_key=chat_key)
        if kind == "like":
            return self._control_like(command.tail or arg)
        if kind == "dislike":
            return self._control_dislike(command.tail or arg)
        if kind == "sham":
            return self._control_sham(command.tail or arg, chat_key=chat_key,
                                      message=message)
        if kind == "musicnote":
            # The music voice-note ask's buttons (owner-only — the
            # grant check above already denied non-owners). The
            # pending clip was stashed when the ask went out; the tap
            # consumes it.
            return self._control_musicnote(command.tail or arg,
                                           chat_key=chat_key)
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

        # ── reachability wave: temp mail/SMS, browser, paper, people,
        # commitments, focus — capabilities that existed with no chat
        # surface. Temp inboxes are per-user: the message (sender id)
        # scopes the read, so one user can never read another's inbox.
        if kind == "tempmail":
            return self._control_tempmail(
                command.tail or arg, chat_key=chat_key, message=message)
        if kind == "checkmail":
            return self._control_checkmail(
                command.tail or arg, chat_key=chat_key, message=message)
        if kind == "vnumber":
            return self._control_vnumber(
                command.tail or arg, chat_key=chat_key, message=message)
        if kind == "checksms":
            return self._control_checksms(
                command.tail or arg, chat_key=chat_key, message=message)
        if kind == "browser":
            return self._control_browser(command.tail or arg, chat_key=chat_key,
                                         message=message)
        if kind == "paper":
            return self._control_paper(command.tail or arg, chat_key=chat_key)
        if kind == "people":
            return self._control_people(command.tail or arg, chat_key=chat_key)
        if kind == "commitments":
            return self._control_commitments(command.tail or arg, chat_key=chat_key)
        if kind == "decisions":
            return self._control_decisions(command.tail or arg, chat_key=chat_key)
        if kind == "iddocs":
            # Identity document tracker (AA-31) — the owner's private
            # documents, owner-only like /contracts.
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...identity.chat import control_iddocs
            return control_iddocs(
                command.tail or arg,
                context=self.context,
                chat=chat,
                sender_id=getattr(message, "sender_id", "") if message else "",
                sender=getattr(message, "sender", "") if message else "",
            )
        if kind == "focus":
            return self._control_focus(command.tail or arg, chat_key=chat_key)
        if kind == "backup":
            return self._control_backup(command.tail or arg, chat_key=chat_key)

        # ── direct game-start commands (/hangman, /mafia, …) ─────────────────
        # The any-chat path in _on_message normally intercepts these
        # before handle_control, but handle_control stays total: direct
        # callers (stacks, console, tour/heal) get the real game, never
        # a lying "unknown command /<kind>". The tour skips these kinds
        # explicitly — starting a game is interactive state, not a
        # read-only probe.
        if kind in GAME_START_COMMANDS:
            return self._control_game(
                kind + (" " + (command.tail or arg)
                        if (command.tail or arg) else ""),
                chat_key=chat_key,
                player=self._game_player_for_key(chat_key), kind="dm",
                is_owner=is_owner)

        # ── expansion wave ───────────────────────────────────────────────────
        if kind == "game":
            # console path: no inbound message, so the player is derived from
            # the chat key (stable per chat) and the kind defaults to dm.
            return self._control_game(
                command.tail or arg, chat_key=chat_key,
                player=self._game_player_for_key(chat_key), kind="dm",
                is_owner=is_owner)
        if kind == "math":
            # Math Class via its standalone handler (same as the
            # any-chat game dispatch). A None return means the
            # outsider-in-DM gate fired — an honest redirect, since
            # handle_control has no conversation fall-through.
            res = self._control_math(
                command.tail or arg, chat_key,
                player=self._game_player_for_key(chat_key),
                kind=getattr(getattr(message, "chat", None), "kind", "dm"),
                is_owner=is_owner)
            if res is None:
                return ("Math Class is a group game — play it in a group "
                        "with /math.")
            return res
        if kind == "npc":
            # the living cast — game-scoped via the live room in this chat
            return self._control_npc(command.tail or arg, chat_key=chat_key)
        if kind == "charagent":
            # character agents — persistent entities beyond any one game
            return self._control_charagent(command.tail or arg,
                                           chat_key=chat_key)
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
        if kind in {"translate", "tl"}:
            return self._control_translate(command.tail or "", message)
        # ── media expression (§21 + §22) ─────────────────────────
        if kind == "steal":
            return self._control_steal(command.tail or "", message)
        if kind == "sticker":
            return self._control_sticker(command.tail or "", message)
        if kind == "convert":
            return self._control_convert(command.tail or "", message)
        if kind == "gif":
            return self._control_gif(command.tail or "", message)
        if kind == "emoji":
            return self._control_emoji(command.tail or "", message)
        if kind == "customemoji":
            return self._control_customemoji(command.tail or "", message)
        if kind == "emojiart":
            return self._control_emojiart(command.tail or "", message)
        if kind == "react":
            return self._control_react(command.tail or "", message)
        if kind == "create":
            return self._control_create(command.tail or "", message)
        if kind in {
                "awoo", "baka", "bite", "blush", "bonk", "bored",
                "bully", "cringe", "cry", "cuddle", "dance",
                "facepalm", "feed", "glomp", "handhold",
                "handshake", "happy", "highfive", "hug",
                "kiss", "laugh", "lick", "nom", "pat",
                "poke", "pout", "shrug", "slap", "sleep", "smile",
                "smug", "stare", "thumbsup", "tickle", "wave",
                "wink", "yeet"}:
            # The per-emotion reaction GIFs (wave C): the set
            # is spelled out so the catalog's handler-path scan
            # sees every kind; expression.REACTION_GIFS is the
            # same set with its sources + captions, and
            # tests/test_reaction_gifs.py pins the two equal.
            return self._control_reaction(
                kind, command.tail or "", message)
        if kind in {
                "joke", "dadjoke", "progjoke", "chuck", "fact", "fact2",
                "catfact", "numfact", "advice", "quote",
                "dog", "cat", "fox", "meme", "riddle", "8ball",
                "choose", "activity", "pokemon", "history", "nasa",
                "calc"}:
            # The fun one-shots (FEI gap wave D): the set is
            # spelled out so the catalog's handler-path scan
            # sees every kind; social/chat/fun.py's FUN_COMMANDS
            # is the same set with its sources + chains, and
            # tests/test_fun_oneshots.py pins the two equal.
            return self._control_fun(
                kind, command.tail or "", message)
        if kind in {
                "qcard", "canvasquote", "iqc", "wndwqute", "meme2",
                "fakemsg", "fakettc", "nokiamsg", "fakeff",
                "fakeffduo", "ffgirl",
                "removebg", "qr", "shorturl", "unshorten", "bible",
                "couplepp", "pp"}:
            # The makers & tools (FEI gap wave E): the set is
            # spelled out so the catalog's handler-path scan
            # sees every kind; social/chat/makers.py's
            # MAKER_COMMANDS is the same set with its sources,
            # rungs and renderer, and tests/test_maker_tools.py
            # pins the two equal.
            return self._control_makers(
                kind, command.tail or "", message)
        if kind in {
                "archive", "unarchive", "mutechat", "unmutechat",
                "readchat", "unreadchat", "star", "unstar",
                "forward",
                "unsend", "edit", "editmsg", "antiedit",
                "block", "unblock", "anticall", "disappear",
                "statusreact", "poststatus", "chanstory",
                "banchecker", "gcstatus",
                "statusinfo", "pollres", "tagall", "hidetag",
                "antitag", "viewonce", "vv",
                "joinlink", "newgroup", "newchannel", "leave",
                "myprofile", "privacy", "makepoll", "vcf",
                "addcontact"}:
            # The chat & privacy set (FEI gap wave F): the set
            # is spelled out so the catalog's handler-path scan
            # sees every kind; social/chat/chatops.py's
            # CHATOPS_COMMANDS is the same set, and
            # tests/test_chatops_wavef.py pins the two equal.
            # Owner-only account ops on his personal surfaces —
            # none of these kinds is in PUBLIC_COMMANDS.
            return self._control_chatops(
                kind, command.tail or "", message)
        if kind == "pulse":
            return self._control_pulse(command.tail or arg)
        if kind == "code":
            return self._control_code(command.tail, chat_key=chat_key)
        if kind == "checkpoint":
            return self._control_checkpoint(command.tail, chat_key=chat_key)
        if kind == "rewind":
            return self._control_rewind(command.tail, chat_key=chat_key)
        if kind == "checkpoints":
            return self._control_checkpoints(command.tail, chat_key=chat_key)
        if kind == "py":
            return self._control_py(command.tail, chat_key=chat_key)
        if kind == "remember":
            return self._control_remember(command.tail or arg, chat_key=chat_key)
        if kind == "recall":
            return self._control_recall(command.tail or arg)
        if kind == "forget":
            return self._control_forget(command.tail or arg)
        if kind == "review":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_review(command.tail or arg,
                                        chat_key=chat_key)
        if kind == "tutor":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_tutor(command.tail or arg,
                                       chat_key=chat_key)
        if kind == "ekiti":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_ekiti(command.tail or arg,
                                       chat_key=chat_key)
        if kind == "course":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_course(command.tail or arg,
                                        chat_key=chat_key)
        if kind == "health":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_health(command.tail or arg,
                                        chat_key=chat_key)
        if kind == "train":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_train(command.tail or arg,
                                       chat_key=chat_key)
        if kind == "form":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_form(command.tail or arg,
                                      chat_key=chat_key)
        if kind == "film":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_film(command.tail or arg,
                                      chat_key=chat_key)
        if kind == "audio":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_audio(command.tail or arg,
                                       chat_key=chat_key)
        if kind == "overview":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_overview(command.tail or arg,
                                          chat_key=chat_key)
        if kind == "character":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_character(command.tail or arg,
                                           chat_key=chat_key)
        if kind == "audiobook":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_audiobook(command.tail or arg,
                                           chat_key=chat_key)
        if kind == "graph":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_graph(command.tail or arg,
                                       chat_key=chat_key)
        if kind == "route":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_route(command.tail or arg,
                                       chat_key=chat_key)
        if kind == "eta":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_eta(command.tail or arg,
                                     chat_key=chat_key)
        if kind == "congestion":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_congestion(command.tail or arg,
                                            chat_key=chat_key)
        if kind == "routine":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_routine(command.tail or arg,
                                         chat_key=chat_key)
        if kind == "home":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_home(command.tail or arg,
                                      chat_key=chat_key)
        if kind == "store":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_store(command.tail or arg,
                                       chat_key=chat_key)
        if kind == "track":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_track(command.tail or arg,
                                       chat_key=chat_key)
        if kind == "untrack":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_untrack(command.tail or arg,
                                         chat_key=chat_key)
        if kind == "trip":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            return self._control_trip(command.tail or arg,
                                      chat_key=chat_key)
        if kind == "travelclient":
            if message is not None and not self._is_operator(message):
                return "that one's just for the owner."
            from ...travel.whitelabel import _control_travelclient
            return _control_travelclient(self, command.tail or arg,
                                         chat_key=chat_key)
        if kind == "tts":
            return self._control_tts(command.tail or arg, chat_key=chat_key)
        if kind == "sing":
            return self._control_sing(command.tail or arg, chat_key=chat_key)
        # (no second `if kind == "say"` here: the single /say branch above
        # routes all three forms, including rich voice via _control_say —
        # a second branch would be unreachable dead code.)
        if kind == "stt":
            return self._control_stt(command.tail or arg)
        if kind == "look":
            return self._control_look(command.tail or arg, chat_key=chat_key)
        if kind == "schedule":
            return self._control_schedule(command.tail or arg)
        if kind == "db":
            return self._control_db(command.tail or arg)
        if kind == "artifact":
            return self._control_artifact(command.tail or arg,
                                          chat_key=chat_key)
        if kind == "api":
            return self._control_api(command.tail or arg)
        if kind == "connectors":
            return self._control_connectors(command.tail or arg)
        if kind == "provision":
            return self._control_provision(command.tail or arg,
                                           chat_key=chat_key,
                                           message=message)
        if kind == "notion":
            return self._control_notion(command.tail or arg)
        if kind == "gcal":
            return self._control_gcal(command.tail or arg)
        if kind == "trello":
            return self._control_trello(command.tail or arg)
        if kind == "exness":
            return self._control_exness(command.tail or arg)
        if kind == "stripe":
            return self._control_stripe(command.tail or arg)
        if kind == "swarm":
            return self._control_swarm(command.tail or arg, chat_key=chat_key)
        if kind == "dns":
            return self._control_dns(command.tail or arg)
        if kind == "scan":
            _tail = command.tail or arg
            _first = (_tail or "").split()[0] if (_tail or "").split() \
                else ""
            if _first.lower().startswith(("http://", "https://")) \
                    and message is not None:
                # A URL target is the security-audit exposure
                # scan (owner, 2026-11-03), not a port scan.
                return self._control_audit_scan(_tail, message)
            return self._control_scan(_tail)
        if kind == "whois":
            return self._control_whois(command.tail or arg)
        if kind == "ports":
            return self._control_ports(command.tail or arg)
        if kind == "proxy":
            return self._control_proxy(command.tail or arg)
        if kind == "toolhub":
            return self._control_toolhub(command.tail or arg, chat_key)
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
            return self._control_file(command.tail or arg, chat_key)
        if kind == "publish":
            return self._control_publish(command.tail or arg)
        if kind == "deliver":
            return self._control_deliver(command.tail or arg, chat_key)
        if kind == "data":
            return self._control_data(command.tail or arg, chat_key=chat_key)
        if kind == "evolve":
            return self._control_evolve(command.tail or arg, chat_key=chat_key)
        if kind == "upgrade":
            # the originating chat rides along so _control_upgrade can
            # re-check the owner gate itself (defense in depth — a direct
            # call with a non-owner chat is denied, fail-closed).
            if message is not None:
                chat = getattr(message, "chat", None) or chat
            return self._control_upgrade(command.tail or arg, _chat=chat)
        if kind == "speak":
            return self._control_speak(command.tail or arg, chat_key=chat_key)
        if kind == "voice":
            return self._control_voice(command.tail, chat_key,
                                       message=message)
        if kind == "voices":
            return self._control_voices(command.tail or arg,
                                        chat_key=chat_key)
        if kind == "clonevoice":
            return self._control_clonevoice(command.tail, chat_key,
                                            message=message)
        if kind == "bet":
            return self._control_bet(command.tail or arg, chat_key=chat_key)
        if kind == "desk":
            return self._control_desk(command.tail or arg, chat_key=chat_key)
        if kind == "apk":
            return self._control_apk(command.tail or arg, chat_key=chat_key)
        if kind == "re":
            return self._control_re(command.tail or arg, chat_key=chat_key)
        if kind == "pump":
            return self._control_pump(command.tail or arg, chat_key=chat_key)
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
            return self._control_image(command.tail or arg,
                                       chat_key=chat_key, message=message)
        if kind == "imggen":
            return self._control_imggen(command.tail or arg)
        if kind == "shorts":
            return self._control_shorts(command.tail or arg)
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
        except Exception as e:
            record_failure("runtime.PartnerRuntime._base_rate_limit", e)
            return 60

    @staticmethod
    def _ref_from_key(key: str):
        from ...social.chat.base import ChatRef

        plat, _, cid = key.partition(":")
        return ChatRef(platform=plat or "local", chat_id=cid or key)

    def _typing_enabled(self, chat: ChatRef) -> bool:
        """False where the owner ordered the typing simulation off.

        - Personal bot (``telegram-personal``) is owner-only: no typing
          delay / fake indicator on any of its chats.
        - Userbot (``telegram`` MTProto) in the owner's DM: no typing
          delay / fake indicator. Other userbot chats are unchanged.
        - Everything else — main gated bot (``telegram-bot``) included,
          in owner and non-owner chats — keeps the existing behavior.
        """
        if chat.platform == "telegram-personal":
            return False
        if chat.platform == "telegram" and is_owner_chat(
                chat, owner_chats=getattr(self, "_owner_chats", ())):
            return False
        return True

    def _owner_reply_ledger(self) -> Any:
        """The owner-replied suppression ledger (register #28).

        One per runtime, formed lazily so test doubles and partial
        constructions still get the real behavior. The brain reaches
        it through ``context.ledger``-style duck typing via
        ``PartnerBrain.owner_reply_ledger``.
        """
        ledger = getattr(self, "_owner_reply_ledger_obj", None)
        if ledger is None:
            from .owner_replies import OwnerReplyLedger
            ledger = OwnerReplyLedger()
            self._owner_reply_ledger_obj = ledger
        return ledger

    def _button_pick_store(self) -> Any:
        """The numbered button-fallback pick store (register #27)."""
        store = getattr(self, "_button_pick_store_obj", None)
        if store is None:
            from .button_picks import ButtonPickStore
            store = ButtonPickStore()
            self._button_pick_store_obj = store
        return store

    def _register_button_picks(self, chat_key: str,
                               commands: list, *,
                               owner_key: str = "") -> str:
        return self._button_pick_store().register(
            chat_key, list(commands), owner_key=owner_key)

    def _button_pick_hook(self, message: ChatMessage) -> str | None:
        """Bare number reply after a numbered button fallback.

        Mirrors the /play pick hook's contract: a bare number with
        a live pick list executes that pick's command through the
        control dispatch (grant re-checked there); anything else
        returns None and the turn flows on untouched. Never raises.

        Targeting (Living Chat P1): the pick resolves in the
        SENDER's (chat, owner) slot — another user's card in the
        same group is never consumed — and a pick against a card
        that was replaced or expired answers honestly instead of
        flowing on in silence.
        """
        text = (message.text or "").strip()
        if not text.isdigit():
            return None
        from ...core.operations import PickStatus
        from .button_picks import STALE_PICK_MESSAGE
        try:
            res = self._button_pick_store().take_result(
                message.chat.key, int(text),
                owner_key=_pick_owner_key(message))
        except Exception as exc:  # noqa: BLE001 - hook never eats chat
            record_failure(
                "runtime.PartnerRuntime._button_pick_hook", exc)
            return None
        if res.status is PickStatus.STALE:
            return STALE_PICK_MESSAGE
        if res.status is not PickStatus.OK or not res.command:
            return None
        return self.handle_control(
            "/" + res.command.lstrip("/"), message.chat.key,
            message=message)

    def _send_mind_reply(self, message: ChatMessage, reply: str) -> None:
        """Send a CoreMind reply through the narration scrub.

        Personal surfaces (the owner's own accounts — wa.me,
        Telegram MTProto, Discord personal) never see internal
        narration: the router's ``(route: …)`` parentheticals and
        "I'll report back" scaffold promises are stripped at this
        boundary whatever composer produced them (owner order
        2026-11-05, register #33 — the raw started notes used to be
        sent verbatim). A reply that is narration through and
        through sends nothing: the job's own artifact is the
        message. Bot surfaces keep their text byte-identical;
        degenerate-brain gating upstream is untouched.
        """
        if is_personal_platform(message.chat.platform):
            reply = scrub_narration(reply)
            if not str(reply or "").strip():
                return
            # Deliberation hold (2026-11-06): the same third check
            # as _send_reply's gate — a mind reply composed with a
            # model can carry the model's deliberation about the
            # conversation, and on a personal surface that is held
            # to silence, never sent in the owner's voice. Journaled
            # hash-only, like every hold at this boundary.
            _delib_reason = deliberation_reason(reply)
            if _delib_reason:
                import hashlib

                record_failure(
                    "runtime.PartnerRuntime._send_mind_reply:"
                    "deliberation_guard",
                    ValueError(
                        f"mind reply deliberation held "
                        f"({_delib_reason}) on "
                        f"{message.chat.platform} in "
                        f"{message.chat.key}: sha256="
                        f"{hashlib.sha256(reply.encode('utf-8')).hexdigest()[:16]} "
                        f"len={len(reply)}"))
                self._bump("deliberation_held")
                return
        try:
            self._typing_for(message.chat, reply)
            self.gateway.send(message.chat.platform, message.chat, reply)
        except Exception as exc:  # noqa: BLE001
            _log.warning("mind reply send failed: %s", exc)

    def _typing_for(self, chat: ChatRef, text: str, *, cap: float = 10.0) -> None:
        """Length-scaled typing before a short outgoing reply (control
        answers, game moves). Skipped in groups when configured off; never
        raises — typing is cosmetic."""
        partner_cfg = self.settings.partner
        if not self._typing_enabled(chat):
            return
        if chat.kind == ChatKind.GROUP and not partner_cfg.typing_in_groups:
            return
        try:
            self.gateway.typing(
                chat.platform, chat,
                seconds=human_typing_seconds(
                    text, minimum=1.5, cap=cap, rng=self.brain.presence_rng,
                ),
            )
        except Exception as e:
            record_failure("runtime.PartnerRuntime._typing_for", e)
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
        typing_on = self._typing_enabled(chat)
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
            if typing_on and not self.gateway.typing(chat.platform, chat, seconds=tick):
                return  # no adapter / platform down — stop, don't retry blindly
            stop.wait(max(0.05, tick - (time.time() - started)))

    def _send_long(self, platform: str, chat, text: str, limit: int = 3800) -> int:
        """Send a long text in chunks (Telegram's hard cap is 4096 chars).

        Each chunk gets its own length-scaled typing run, so a report split
        into five pages reads as five real typing sessions, not a telegraph.

        Chunks split at paragraph → sentence → word boundaries via the
        shared lossless chunker (never mid-word when a boundary fits) —
        the old hard ``text[i:i+limit]`` slicing cut words and sentences
        in half. The join of the chunks is exactly ``text``.

        Returns the number of chunks actually delivered. A failed chunk is
        a warning log, never silent — callers must not treat the return as
        "delivered" without checking it (see :meth:`_send_long_checked`).
        """
        from ...social.chat.chunking import chunk_message

        partner_cfg = self.settings.partner
        typing_on = (
            self._typing_enabled(chat)
            and (getattr(chat, "kind", ChatKind.DM) != ChatKind.GROUP
                 or partner_cfg.typing_in_groups)
        )
        sent = 0
        pieces = chunk_message(text, max_text=limit)
        total = len(pieces)
        for i, chunk in enumerate(pieces):
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
                                 sent + 1, total, getattr(chat, "key", chat),
                                 result.error)
                    break
            except Exception as exc:  # noqa: BLE001
                _log.warning("send_long: chunk %d/%d to %s raised %s: %s",
                             sent + 1, total, getattr(chat, "key", chat),
                             type(exc).__name__, exc)
                break
        if sent:
            try:
                _control_deliveries.set(_control_deliveries.get() + sent)
            except Exception as exc:  # noqa: BLE001 - marking never breaks a send
                _log.debug("control delivery mark failed: %s", exc)
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
        except Exception as e:
            record_failure("runtime.PartnerRuntime._notify", e)

    def start_platform(self, name: str) -> dict[str, Any]:
        name = name.strip().lower()
        # An owner start revokes any standing owner stop order for
        # the platform (durable intent, pairing registry) BEFORE
        # the gateway moves — a reconcile beat landing between the
        # two must already see the order lifted.
        self._set_platform_owner_stopped(name, False)
        if name == "whatsapp":
            # The WhatsApp surface starts through its reconcile:
            # the resolved account's bridge is ensured through
            # the supervisor and the adapter attaches to THAT
            # bridge — the generic gateway start would raise an
            # adapter against the static config endpoint with no
            # bridge behind it (the dead-port attach this fixes).
            # None = no account paired; the generic start owns
            # that case as before.
            routed = self._start_whatsapp_surface()
            if routed is not None:
                return routed
        return self.gateway.start_one(name, self.on_message)

    def stop_platform(self, name: str) -> dict[str, Any]:
        name = name.strip().lower()
        # Record the owner's stop order FIRST, durably: the
        # WhatsApp live-attach reconcile (and its startup twin)
        # honor it, so this stop sticks across beats and restarts
        # instead of lasting only until the pool next looks live.
        self._set_platform_owner_stopped(name, True)
        return self.gateway.stop_one(name)

    # ── agent modes: one persistent engine for /agent, /proposals, ticks ──

    @property
    def agent_modes(self) -> Any:
        """The runtime's ONE persistent :class:`AgentModes`.

        Shared by ``/agent``, ``/proposals`` approve/deny, and the
        scheduled tick — a fresh engine per command would have no
        sources and no pending initiatives.  Its engine files
        approval-gated initiatives into ``proactive_log`` via
        :meth:`_file_initiative_proposal`, and its approval policy is
        the existing autonomy mode (:meth:`_initiative_requires_approval`).
        A half-built runtime (tests using ``__new__``) falls back to a
        bare ``AgentModes`` so mode commands never crash chat.
        """
        cached = self.__dict__.get("_agent_modes")
        if cached is not None:
            return cached
        try:
            from ..modes import AgentModes

            modes = AgentModes(
                proposal_sink=self._file_initiative_proposal,
                requires_approval=self._initiative_requires_approval)
        except Exception as e:
            record_failure("runtime.PartnerRuntime.agent_modes", e)
            from ..modes import AgentModes

            modes = AgentModes()
        self.__dict__["_agent_modes"] = modes
        try:
            self.context.extras["agent_modes"] = modes
        except Exception as e:
            record_failure("runtime.PartnerRuntime.agent_modes:4831", e)
        return modes

    def _initiative_requires_approval(self, spec: dict[str, Any],
                                      ctx: dict[str, Any]) -> bool:
        """Approval policy for modes initiatives: the existing autonomy mode.

        ``/mode suggest`` → initiatives wait for approval in
        ``/proposals``; ``/mode auto`` (default) → they run
        immediately.  A spec-level ``requires_approval`` bool is
        honoured by the engine before this callable is consulted.
        """
        try:
            settings = (getattr(self, "settings", None)
                        or getattr(self.context, "settings", None))
            mode = getattr(getattr(settings, "partner", None),
                           "autonomy_mode", "auto")
        except Exception as e:
            record_failure("runtime.PartnerRuntime._initiative_requires_approval", e)
            mode = "auto"
        return str(mode or "auto").strip().lower() == "suggest"

    def _file_initiative_proposal(self, initiative: Any,
                                  spec: dict[str, Any],
                                  ctx: dict[str, Any]) -> str:
        """File a held initiative into the proposals queue (``proactive_log``).

        ``kind="initiative"`` so ``/proposals`` lists it and
        :meth:`approve` / :meth:`deny` route it to the modes engine
        instead of the message-send path — approving an initiative
        RUNS its action; sending the proposal text as a chat message
        would be wrong for an action.  Returns the proposal id
        (= the initiative id, so routing is a direct lookup).
        """
        chat_key = (ctx or {}).get("chat_key", "")
        platform, chat_id = "local", ""
        try:
            ref = ChatRef.parse(chat_key)
            platform, chat_id = ref.platform, ref.chat_id
        except Exception as e:
            record_failure("runtime.PartnerRuntime._file_initiative_proposal", e)
        content = str(spec.get("proposal")
                      or f"autonomous initiative: {initiative.label}")
        reason = f"autonomous initiative from {initiative.source}"
        self.context.db.execute(
            "INSERT INTO proactive_log (id, kind, platform, chat_id, "
            "content, status, reason, decided_at, acted_at) "
            "VALUES (?, 'initiative', ?, ?, ?, 'pending', ?, ?, NULL)",
            (initiative.id, platform, chat_id, content, reason,
             time.time()))
        return initiative.id

    def proposals(self, status: str = "pending") -> list[dict[str, Any]]:
        """Everything awaiting owner approval, unified across both queues.

        Two proposal systems exist, and /proposals used to show only one:
        * proactive sends (``proactive_log``, status ``pending``) — the
          autonomy suggest-mode queue.
        * research upgrade tickets (``upgrade_proposals``, status
          ``proposed``) — filed by the research loop via UpgradePipeline.

        The research-loop notification says "N proposal(s)", so the owner
        checks /proposals — which read only proactive_log and always came
        back empty for research proposals. This returns both, each row
        tagged with ``queue`` ("proactive" | "upgrade"), upgrades first
        (they are the ones the notification points at).
        """
        rows: list[dict[str, Any]] = []
        try:
            from ..upgrade_queue import UpgradeQueue

            for row in UpgradeQueue(self.context).list(status="proposed"):
                r = dict(row)
                r["queue"] = "upgrade"
                rows.append(r)
        except Exception as e:
            record_failure("runtime.PartnerRuntime.proposals", e)
        for row in self.context.db.query(
            "SELECT * FROM proactive_log WHERE status = ? ORDER BY decided_at DESC LIMIT 50",
            (status,),
        ):
            r = dict(row)
            r["queue"] = "proactive"
            rows.append(r)
        return rows

    def approve(self, proposal_id: str) -> dict[str, Any]:
        # Upgrade tickets (upg_…) live in upgrade_proposals, not
        # proactive_log — route them to approve+implement (same as
        # /upgrade approve), not the autonomy path (which would fail
        # with "unknown proposal").
        try:
            from ..upgrade_queue import UpgradePipeline, UpgradeQueue

            if UpgradeQueue(self.context).get(proposal_id) is not None:
                try:
                    done = UpgradePipeline(self.context).approve_and_implement(
                        proposal_id, by="owner")
                except Exception as exc:  # noqa: BLE001 — already recorded failed
                    return {"ok": False,
                            "error": f"{type(exc).__name__}: {exc}"}
                return {"ok": True,
                        "status": done.get("status", "implemented"),
                        "proposal": done}
        except Exception as e:
            record_failure("runtime.PartnerRuntime.approve", e)
        # Initiatives (kind="initiative") live in proactive_log but are
        # ACTIONS, not messages — approving one RUNS the action via the
        # modes engine.  This routes BEFORE the autonomy/direct paths,
        # which would send the proposal text as a chat message.
        try:
            row = self.context.db.query_one(
                "SELECT * FROM proactive_log WHERE id = ?", (proposal_id,))
            if row is not None and row.get("kind") == "initiative":
                try:
                    init = self.agent_modes.approve_proposal(proposal_id)
                except KeyError:
                    # Orphaned by a restart: the in-memory action is gone.
                    # Leave the row pending and say so honestly — the
                    # owner can /deny it to clear it.
                    return {"ok": False,
                            "error": f"initiative {proposal_id!r} is no "
                                     "longer pending (restarted since it was "
                                     "proposed) — deny it to clear it"}
                status = "approved" if init.status == "done" else "failed"
                try:
                    self.context.db.execute(
                        "UPDATE proactive_log SET status = ?, acted_at = ? "
                        "WHERE id = ?",
                        (status, time.time(), proposal_id))
                except Exception as e:
                    record_failure("runtime.PartnerRuntime.approve:4961", e)
                return {"ok": init.status == "done", "status": status,
                        "error": init.error or None,
                        "initiative": init.to_dict()}
        except Exception as e:
            record_failure("runtime.PartnerRuntime.approve:4966", e)
        if self._autonomy is not None:
            return self._autonomy.approve(proposal_id)
        return _direct_approve(self.context, self.gateway, proposal_id)

    def deny(self, proposal_id: str, reason: str = "") -> dict[str, Any]:
        # Upgrade tickets need a denial reason (UpgradeQueue.deny
        # requires one) — ask for it instead of failing silently.
        try:
            from ..upgrade_queue import UpgradeQueue

            row = UpgradeQueue(self.context).get(proposal_id)
            if row is not None:
                if not (reason or "").strip():
                    return {"ok": False,
                            "error": "upgrade deny needs a reason: "
                                     "/deny <id> <reason>"}
                try:
                    denied = UpgradeQueue(self.context).deny(
                        proposal_id, reason.strip(), by="owner")
                except Exception as exc:  # noqa: BLE001 — status/validation
                    return {"ok": False,
                            "error": f"{type(exc).__name__}: {exc}"}
                return {"ok": True,
                        "status": denied.get("status", "denied"),
                        "proposal": denied}
        except Exception as e:
            record_failure("runtime.PartnerRuntime.deny", e)
        # Initiatives: denying one discards its action (never runs it)
        # and clears the row — including rows orphaned by a restart.
        try:
            row = self.context.db.query_one(
                "SELECT * FROM proactive_log WHERE id = ?", (proposal_id,))
            if row is not None and row.get("kind") == "initiative":
                try:
                    self.agent_modes.deny_proposal(proposal_id)
                except KeyError as e:
                    record_failure("runtime.PartnerRuntime.deny:5003", e) # orphaned by a restart: still clear the row
                try:
                    self.context.db.execute(
                        "UPDATE proactive_log SET status = 'denied', "
                        "acted_at = ? WHERE id = ?",
                        (time.time(), proposal_id))
                except Exception as e:
                    record_failure("runtime.PartnerRuntime.deny:5010", e)
                return {"ok": True, "status": "denied"}
        except Exception as e:
            record_failure("runtime.PartnerRuntime.deny:5013", e)
        if self._autonomy is not None:
            return self._autonomy.deny(proposal_id)
        return _direct_deny(self.context, proposal_id)

    # ── power layer: network / proxy / gen / osint / record / macro ──────────

    def _tool_reply(self, tool: str, chat_key: str, **kwargs: Any) -> str:
        """Run one tool and stream its styled output back to the chat.

        Human-readable (header + structured ``key: value`` lines), never
        a raw JSON dump.  Formatting F7: the rendering converged onto
        the presentation layer — the result becomes a
        :func:`~nomorals.agents.job_reports.tool_result_doc` (the
        F3-style generic Doc: summary first, noise suppressed, detail
        expandable) rendered in the canonical dialect, which the
        gateway's F1 presentation converts per platform at send.
        ``core.output_style.render_output`` stays as a module; this
        path no longer renders through it.
        """
        from ...core.idempotency_keys import scoped_idempotency_key

        # Control commands are re-runnable by redelivery of the same
        # inbound message: under that message's scope (set by
        # _process), an identical command replays its recorded result
        # instead of re-applying the effect (e.g. /proxy set landing
        # twice). Outside message processing there is no scope and the
        # call runs unkeyed, as before.
        call_kwargs = dict(kwargs)
        explicit = call_kwargs.pop("idempotency_key", None)
        key = explicit or scoped_idempotency_key(tool, call_kwargs)
        if key:
            call_kwargs["idempotency_key"] = key
        outcome = self.context.tools.call(tool, **call_kwargs)
        chat = self._ref_from_key(chat_key)
        if not outcome.ok:
            return f"{tool} failed: {getattr(outcome.error, 'message', outcome.error)}"
        from ...social.chat.present import canonical_text
        from ..job_reports import tool_result_doc

        text = canonical_text(tool_result_doc(tool, outcome.value))
        return self._send_long_checked(chat.platform, chat, text[:6000])
