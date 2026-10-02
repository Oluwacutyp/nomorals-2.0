"""The partner runtime: the brain, and the loop that keeps it alive.

Two objects:

* :class:`PartnerBrain` — stateless-per-message cognition. For one inbound
  message it: advances the mood clock, detects emotional signals, recalls
  shared memory, selects gated background knowledge, calls the model, guards
  the output, and persists everything (messages, training pairs).
* :class:`PartnerRuntime` — the loop. It owns the :class:`ChatGateway`, runs
  every platform at once, guarantees **per-chat FIFO ordering** while letting
  different chats be processed **in parallel** (bounded by
  ``partner.max_parallel_chats``), adds typing indicators, splits replies
  into short human-sized sends, and spawns the curator sub-agent on segments.

Conversation rules per chat kind:

* **DM (the partner / owner)** — always answered.
* **DM (anyone else)** — answered in character, but the context tells her this
  is not the partner, so the intimacy stays where it belongs.
* **Group** — she listens; she only speaks when mentioned, when someone
  replies to her, or when the partner is talking there.
* **Channel** — read-only for her; posting to channels is the autonomy
  agent's job (deliberately, on its own schedule).
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.config import Settings
from ..core.ids import ulid_now
from ..core.logging_setup import get_logger
from ..core.text import truncate
from ..llm.base import Message, SamplingParams
from ..partner.background import BackgroundSelector
from ..partner.lexicon_feed import LexiconFeed, seed_partner_lexicon
from ..partner.mood import MoodEngine
from ..partner.persona import Persona, default_persona
from ..partner.gating import gate_decision, is_owner_chat, is_restricted
from ..partner.presence import Presence, decide_presence, human_typing_seconds
from ..partner.relationship import Relationship
from ..partner.responder import PartnerResponder, detect_signals
from ..social.chat.base import ChatKind, ChatMessage, ChatRef
from ..social.chat.gateway import ChatGateway

__all__ = ["PartnerBrain", "PartnerRuntime", "PresenceOutcome"]

_log = get_logger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class PresenceOutcome:
    """What one inbound message produced, presence decisions included.

    * ``parts`` non-empty — reply to send now.
    * ``parts`` empty + ``presence.delay_seconds`` — she's busy; the runtime
      schedules :meth:`PartnerBrain.deliver_reply` after the gap.
    * ``parts`` empty, no delay — read and left (low stakes) or a chat kind
      she stays quiet in.
    """

    parts: list[str] = field(default_factory=list)
    presence: Presence = field(default_factory=lambda: Presence(reply=True))


# ── the brain ──────────────────────────────────────────────────────────────────


class PartnerBrain:
    """All the cognition. The runtime calls :meth:`handle_message` per message."""

    def __init__(
        self,
        context: Any,
        persona: Persona | None = None,
        *,
        settings: Settings | None = None,
        curator: Callable[[str], None] | None = None,
    ) -> None:
        self.context = context
        self.settings = settings or context.settings
        self.persona = persona or default_persona()
        partner_cfg = self.settings.partner
        if partner_cfg.persona_name and partner_cfg.persona_name != self.persona.name:
            self.persona.name = partner_cfg.persona_name
            self.persona.one_liner = self.persona.one_liner  # name swap only
        self.mood = MoodEngine(self.persona.baselines, db=context.db)
        self.relationship = Relationship.load(context.db)
        if partner_cfg.disclosure:
            self.persona.disclosure = partner_cfg.disclosure
        self.background = BackgroundSelector(mode=partner_cfg.background_gate)
        # Dynamic voice feed: scored lexicon terms blended into her phrasing.
        # None when the owner disabled it — lexical influence never overrides
        # owner-set persona/style preferences, and off means off.
        self.lexicon: LexiconFeed | None = None
        if partner_cfg.lexicon_voice:
            self.lexicon = LexiconFeed(context.db)
            if partner_cfg.lexicon_seed and context.db is not None:
                try:
                    seed_partner_lexicon(context.db)
                except Exception as exc:  # noqa: BLE001 - seeding must never break boot
                    _log.warning("partner lexicon seed failed: %s", exc)
        self.responder = PartnerResponder(
            context.router,
            self.persona,
            self.mood,
            self.relationship,
            context.memory,
            self.background,
            lexicon=self.lexicon,
        )
        self.curator_hook = curator
        self._last_user_seen: dict[str, float] = {}
        # Presence (typing pace, busy gaps) gets its own rng so the mood and
        # sampling randomness stay independent and tests can seed just one.
        self.presence_rng = random.Random()
        # wave 94: the status beacon — how `nm status` (a separate process)
        # knows this one is alive, what is answering, and the last error.
        self._last_reply: dict[str, Any] = {}
        self._last_error = ""

    # ── chat classification ──────────────────────────────────────────────────
    def _chat_flags(self, chat: ChatRef) -> dict[str, Any]:
        row = self.context.db.query_one("SELECT * FROM chats WHERE id = ?", (chat.key,)) or {}
        # Single owner test (nomorals/partner/gating.py) — the gateway's
        # inbound path uses the same function, so rate-limit exemption and
        # reply gating can never disagree about who the owner is.
        is_owner = is_owner_chat(
            chat,
            owner_chats=_key_set(self.settings.partner.owner_chats),
            db_is_owner=bool(row.get("is_owner")),
        )
        in_us = bool(row.get("in_us"))
        last_active = float(row.get("last_active") or 0.0)
        return {
            "is_owner": is_owner,
            "in_us": in_us,
            "last_active": last_active,
        }

    # ── media ────────────────────────────────────────────────────────────────
    def _media_notes(self, message: ChatMessage) -> list[str]:
        notes: list[str] = []
        tools = getattr(self.context, "tools", None)
        for media in message.media[:3]:
            if media.kind == "image" or media.mime.startswith("image/"):
                if tools is not None:
                    result = tools.call("vision_describe", path=media.path,
                                        prompt="Describe this photo in one or two plain sentences: what's in it, and the mood of it.")
                    if result.ok:
                        payload = result.value
                        description = payload.get("description") if isinstance(payload, dict) else str(payload)
                        notes.append(f"photo — {description}")
                        continue
                notes.append("photo (could not see it)")
            elif media.kind == "video":
                notes.append(f"video: {media.name or media.path}")
            elif media.kind == "audio" or media.mime.startswith("audio/"):
                note = self._transcribe_media_note(media)
                notes.append(note or f"audio: {media.name or media.path}")
            else:
                notes.append(f"file: {media.name or media.path}")
        return notes

    def _spoken_command(self, message: ChatMessage) -> str:
        """Raw transcript if an audio attachment transcribes to a /command.

        Returns "" when there is no audio, no transcript, or the transcript
        isn't a command — the caller leaves the message untouched.
        """
        for media in message.media or []:
            kind = str(getattr(media, "kind", "") or "")
            mime = str(getattr(media, "mime", "") or "")
            if kind not in ("audio", "voice") and not mime.startswith("audio"):
                continue
            note = self._transcribe_media_note(media)
            text = note.split("—", 1)[-1].strip() if "—" in note else ""
            if text.startswith("/"):
                return text
        return ""

    def _transcribe_media_note(self, media: Any) -> str:
        """Voice input: transcribe an incoming voice note so the model HEARS it.

        Best-effort — if no STT backend is configured/available this returns
        "" and the caller falls back to the plain "audio: name" note.
        """
        if media.path is None:
            return ""
        try:
            settings = Settings.load()
            has_api = bool(settings.audio.stt_api_key or settings.llm.api_key)
            has_whisper_cpp = bool(os.environ.get("NM_WHISPER_CPP_BIN"))
            if not (has_api or has_whisper_cpp):
                return ""
            outcome = self.context.tools.call("transcribe", path=str(media.path),
                                              provider="auto")
            if not outcome.ok:
                return ""
            text = (outcome.value or {}).get("text", "").strip()
            return f"voice note (transcribed) — {text}" if text else ""
        except Exception:  # noqa: BLE001 - hearing is a bonus, never a failure
            _log.debug("voice-note transcription failed", exc_info=True)
            return ""

    # ── history ─────────────────────────────────────────────────────────────
    def _history(self, chat_key: str, limit: int) -> list[Message]:
        rows = self.context.db.query(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?",
            (chat_key, limit * 2),
        )
        rows.reverse()
        out: list[Message] = []
        for row in rows:
            if row["role"] not in {"user", "assistant"} or not (row["content"] or "").strip():
                continue
            out.append(Message.user(row["content"]) if row["role"] == "user"
                         else Message.assistant(row["content"]))
        return out

    # ── persistence ──────────────────────────────────────────────────────────
    def _persist_turn(
        self, message: ChatMessage, reply_parts: list[str], model: str
    ) -> None:
        """Inbound + outbound in one go (the no-presence fast path)."""
        self._persist_inbound(message)
        if reply_parts:
            self._persist_outbound(message.chat, reply_parts, model)

    def _persist_inbound(self, message: ChatMessage) -> None:
        """The user's message — written the moment she READS it, even if the
        reply comes minutes later or not at all."""
        chat = message.chat
        db = self.context.db
        title = chat.title or chat.peer or chat.chat_id
        try:
            with db.transaction():
                db.execute(
                    """INSERT INTO conversations (id, title, agent, channel, created_at, updated_at)
                       VALUES (?, ?, 'partner', ?, 0, ?)
                       ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at,
                                                    title = CASE WHEN excluded.title != '' THEN excluded.title ELSE conversations.title END""",
                    (chat.key, title, chat.platform, time.time()),
                )
                db.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, name, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ulid_now(), chat.key, "user", message.text, message.sender, "", time.time()),
                )
        except Exception as exc:  # noqa: BLE001 - persistence must never block a reply
            _log.warning("persist turn failed: %s", exc)

    def _persist_outbound(self, chat: ChatRef, reply_parts: list[str], model: str) -> None:
        if not reply_parts:
            return
        try:
            with self.context.db.transaction():
                self.context.db.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, name, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ulid_now(), chat.key, "assistant", "\n".join(reply_parts), self.persona.name, model, time.time()),
                )
        except Exception as exc:  # noqa: BLE001 - persistence must never block a reply
            _log.warning("persist turn failed: %s", exc)

    def _note_reply(self, bundle: Any, chat: ChatRef) -> None:
        """Record the reply for the status beacon (which model answered,
        latency, whether it fell back).  Fallback replies also capture the
        router's latest provider error so `nm status` can say WHY."""
        self._last_reply = {
            "ts": time.time(),
            "model": getattr(bundle, "model", "") or "",
            "chat": chat.key,
            "latency_ms": round(float(getattr(bundle, "latency_ms", 0.0) or 0.0), 1),
            "fallback": bool(getattr(bundle, "fallback", False)),
            "lexicon_dynamic": bool(getattr(bundle, "lexicon_dynamic", False)),
            "lexicon_terms_used": int(getattr(bundle, "lexicon_terms_used", 0) or 0),
        }
        if not self._last_reply["fallback"]:
            return
        try:
            router = getattr(self.context, "router", None)
            snap = getattr(router, "stats_snapshot", lambda: {})() or {}
            health = snap.get("health") or {}
            # the provider that is actually failing right now: in cooldown,
            # or on a run of consecutive failures — with a recorded error
            failing = [h for h in health.values()
                       if h.get("last_error") and
                       (h.get("cooling_down") or
                        int(h.get("consecutive_failures") or 0) > 0)]
            if not failing:
                failing = [h for h in health.values() if h.get("last_error")]
            if failing:
                latest = max(failing,
                             key=lambda h: int(h.get("consecutive_failures") or 0))
                self._last_error = f"{latest.get('name', '?')}: " \
                    f"{latest.get('last_error', '')}"[:200]
        except Exception:  # noqa: BLE001 - beacon bookkeeping never raises
            self._last_error = "all providers failed"

    def _log_training_pair(self, chat: ChatRef, user_text: str, reply: str, model: str, label: str) -> None:
        if not self.settings.partner.train_collect:
            return
        try:
            data_dir = self.settings.resolve(self.settings.training.data_dir)
            data_dir.mkdir(parents=True, exist_ok=True)
            record = {
                "ts": time.time(),
                "chat": chat.key,
                "user": user_text,
                "assistant": reply,
                "mood": label,
                "model": model,
            }
            with open(data_dir / "conversations.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:  # noqa: BLE001
            _log.debug("training pair log failed: %s", exc)

    # ── the per-message pipeline ─────────────────────────────────────────────
    def handle_message(self, message: ChatMessage) -> PresenceOutcome:
        chat = message.chat
        flags = self._chat_flags(chat)
        is_owner = flags["is_owner"]

        # 1. Advance time: decay, circadian energy, and the cost of silence.
        self.mood.tick()
        if is_owner:
            now = time.time()
            last = flags["last_active"] or self._last_user_seen.get(chat.key, 0.0)
            if last:
                hours_idle = max(0.0, (now - last) / 3600.0)
                if hours_idle >= 1.0:
                    self.mood.note_silence(hours_idle)
            self._last_user_seen[chat.key] = now

        # 2. Conversation rules per chat kind. Channels are read-only; in
        #    groups she only speaks when addressed. Checked BEFORE generation
        #    so a message we'll stay quiet on never spends a model call.
        if chat.kind == ChatKind.CHANNEL:
            self._persist_inbound(message)
            return PresenceOutcome()
        if chat.kind == ChatKind.GROUP and not self._should_speak_in_group(message):
            self._persist_inbound(message)
            _log.info("group message in %s: staying quiet", chat.key)
            return PresenceOutcome()

        # 3. Read the message for emotional events; move the state first.
        events = detect_signals(message.text)
        for event in events:
            self.mood.note_event(event.kind, event.intensity, note=event.note)
        if any(e.kind == "fight" for e in events) and self.mood.open_fight is None:
            self.mood.open_fight_now(message.text[:80])
        if any(e.kind == "apology" for e in events) and self.mood.open_fight is not None:
            reason = self.mood.open_fight.get("reason", "a fight")
            self.mood.resolve_fight(repaired=True, repaired_by="partner")
            try:
                self.relationship.record_fight(reason, repaired=True, repaired_by="partner")
            except Exception:  # noqa: BLE001
                pass

        # 3b. Presence: a real person is sometimes busy. A low-stakes message
        #     may be read and left; a substantive one may be answered minutes
        #     later. Either way she HEARD it — the mood state above already
        #     moved, so a busy gap never costs the conversation its history.
        presence = decide_presence(
            self.mood,
            message.text,
            chat_kind=chat.kind,
            is_owner=is_owner,
            rng=self.presence_rng,
        )
        self._persist_inbound(message)
        if not presence.reply:
            _log.info("presence in %s: %s", chat.key, presence.reason)
            return PresenceOutcome(presence=presence)
        if presence.delay_seconds > 0:
            _log.info("presence in %s: %s (%.0fs)", chat.key, presence.reason,
                      presence.delay_seconds)
            return PresenceOutcome(presence=presence)

        parts = self._generate_and_persist(message, flags)
        return PresenceOutcome(parts=parts, presence=presence)

    def deliver_reply(self, message: ChatMessage) -> list[str]:
        """The generation half of :meth:`handle_message`, for the busy case.

        The mood clock is NOT re-ticked and signals are NOT re-detected — she
        already heard the message when it arrived; this is just the reply
        landing once she's free.
        """
        return self._generate_and_persist(message, self._chat_flags(message.chat))

    def _generate_and_persist(self, message: ChatMessage, flags: dict[str, Any]) -> list[str]:
        chat = message.chat
        is_owner = flags["is_owner"]

        # Ownership-aware gating: the full version of her lives in the
        # owner's DMs (and the owner's messages anywhere). Everyone else
        # gets the restricted, human-like version — with the owner-private
        # context (shared memories, cross-platform continuity, romantic
        # background) kept OUT of the prompt, not merely covered by a
        # "don't tell them" instruction.
        gate_mode = gate_decision(
            chat,
            is_owner=is_owner,
            restricted_enabled=self.settings.partner.gate_restricted_chats,
        )
        restricted = is_restricted(gate_mode)

        # Media, memory, background.
        media_notes = self._media_notes(message)
        memories = () if restricted else self.responder.recall(message.text, limit=5)
        romantic = is_owner and self.relationship.is_romantic()
        background_lines = self.background.context(
            message.text,
            user_in_us=flags["in_us"],
            romantic=romantic,
        )

        # Generate.
        history = self._history(chat.key, limit=self.settings.partner.history_window)
        continuity = () if restricted else self._continuity_lines(chat.key)
        if is_owner:
            digest = self._reasoning_digest(message.text)
            if digest:
                continuity = list(continuity) + [digest]
        # Interactive budget: the provider chain behind router.chat can stall
        # for minutes; the chat thread gets INTERACTIVE_REPLY_TIMEOUT_S and
        # then an honest fallback — never a ~90s polite wait.
        bundle = self.responder.respond_bounded(
            chat_platform=chat.platform,
            user_text=message.text,
            gate_mode=gate_mode,
            history=history,
            memories=memories,
            background_lines=background_lines,
            continuity_lines=continuity,
            media_notes=media_notes,
        )

        # Persist + downstream hooks.
        parts = bundle.parts
        if getattr(bundle, "degraded", False) and not restricted:
            # Provider failover: the owner sees WHICH provider failed and
            # WHAT fallback ran — the final answer alone would hide the
            # degradation. Restricted (non-owner) chats keep the human
            # surface clean; the bundle + logs still carry it.
            note = getattr(bundle, "degraded_note", "") or "provider failover"
            _log.warning("reply to %s served degraded: %s", chat.key, note)
            parts = parts + [f"⏬ {note}"]
        self._persist_outbound(chat, parts, bundle.model)
        self._note_reply(bundle, chat)
        self._log_training_pair(chat, message.text, "\n".join(parts), bundle.model,
                                self.mood.current().label)
        self.relationship.save(self.context.db)
        self._maybe_curate(chat.key, is_owner)
        self._maybe_extract(message, chat.key, is_owner)
        return parts

    def note_fast_turn(self, message: "ChatMessage", reply_text: str) -> None:
        """Record a Core-Mind fast-path turn without any model call.

        The fast path answers trivially ("hey", "what time is it") with zero
        LLM latency, but the turn must still be persisted and run the
        downstream hooks (training pairs, curator, memory extraction) —
        otherwise history, training data, and the curator silently lose
        turns. Persistence must never break the reply.
        """
        try:
            chat = message.chat
            is_owner = self._chat_flags(chat)["is_owner"]
            self._persist_inbound(message)
            self._persist_outbound(chat, [reply_text], "fast-path")
            self._log_training_pair(chat, message.text, reply_text, "fast-path",
                                    self.mood.current().label)
            self.relationship.save(self.context.db)
            self._maybe_curate(chat.key, is_owner)
            self._maybe_extract(message, chat.key, is_owner)
        except Exception as exc:  # noqa: BLE001 - persistence must never block a reply
            _log.warning("fast turn persist failed: %s", exc)

    def _maybe_extract(self, message: "ChatMessage", chat_key: str, is_owner: bool) -> None:
        """Mine the turn for durable memories — off-thread, never fatal."""
        try:
            mem_settings = getattr(self.settings, "memory", None)
            if not getattr(mem_settings, "extract_enabled", True):
                return
            if self.context.memory is None:
                return
            speaker = message.sender or message.chat.peer or message.chat.title
            assistant_text = ""

            def _work() -> None:
                try:
                    from ..memory.extract import MemoryExtractor

                    actions = MemoryExtractor(self.context).extract_turn(
                        message.text,
                        assistant_text=assistant_text,
                        chat_key=chat_key,
                        speaker=speaker,
                        is_owner=is_owner,
                    )
                    stored = [a for a in actions if a["action"] == "stored"]
                    if stored:
                        self.context.metrics.incr("memory.extracted", len(stored))
                        _log.info(
                            "memory: stored %d from %s: %s",
                            len(stored), chat_key,
                            "; ".join(f"{a['kind']}: {a['content'][:50]}" for a in stored),
                        )
                except Exception:  # noqa: BLE001 - extraction never breaks chat
                    _log.exception("memory extraction thread crashed (%s)", chat_key)

            threading.Thread(target=_work, name=f"memory-extract-{chat_key[:24]}", daemon=True).start()
        except Exception:  # noqa: BLE001
            _log.exception("memory extraction hook failed to start")

    def _continuity_lines(self, chat_key: str, *, hours: float = 6.0) -> list[str]:
        """What happened on the OTHER platforms recently, plus open threads.

        This is the cross-platform glue: she remembers that she told him on
        WhatsApp what's happening, even when he asks her about it on Discord.
        """
        db = self.context.db
        lines: list[str] = []
        cutoff = time.time() - hours * 3600
        try:
            rows = db.query(
                "SELECT conversation_id, role, content, created_at FROM messages "
                "WHERE conversation_id != ? AND role IN ('user','assistant') AND created_at > ? "
                "ORDER BY created_at DESC LIMIT 8",
                (chat_key, cutoff),
            )
            by_chat: dict[str, list[str]] = {}
            for row in rows:
                content = " ".join((row["content"] or "").split())
                if not content:
                    continue
                platform = row["conversation_id"].split(":", 1)[0]
                who = "you" if row["role"] == "assistant" else "them"
                by_chat.setdefault(row["conversation_id"], []).append(
                    f"({platform}) {who}: {content[:90]}"
                )
            for chat_id, msgs in list(by_chat.items())[:2]:
                lines.extend(msgs[-2:])
        except Exception:  # noqa: BLE001 - continuity is a niceness, never fatal
            pass
        try:
            row = db.query_one("SELECT value FROM kv_store WHERE key = 'partner.open_loops'")
            if row:
                loops = json.loads(row["value"] or "[]")
                fresh = [
                    loop for loop in loops
                    if isinstance(loop, dict)
                    and time.time() - float(loop.get("ts", 0.0)) < 48 * 3600
                ]
                for loop in fresh[-3:]:
                    platform = loop.get("platform", "somewhere")
                    text = str(loop.get("text", "")).strip()
                    if text:
                        lines.append(f"open thread from {platform}: {text[:90]}")
        except Exception:  # noqa: BLE001
            pass
        return lines[-5:]

    def _reasoning_digest(self, text: str) -> str:
        """A private reasoning pass before she answers a complex question.

        Mode comes from ``settings.partner.reasoning``: off | auto | always.
        In auto mode only messages that look genuinely complex trigger it.
        The digest is folded into her context so her reply is *informed by*
        the reasoning — in her own voice, never a raw chain-of-thought dump.
        A failed pass is invisible: reasoning must never break the chat.
        """
        mode = (self.settings.partner.reasoning or "auto").lower()
        if mode == "off":
            return ""
        from .reasoning import ReasoningEngine, looks_complex
        if mode != "always" and not looks_complex(text):
            return ""
        try:
            engine = ReasoningEngine(self.context, max_llm_calls=6,
                                     max_seconds=45.0)
            result = engine.reason(text, strategy="auto")
            if not result.answer or result.stopped != "complete":
                return ""
            steps = " | ".join(
                s.text[:110] for s in result.trace
                if s.kind in {"note", "subgoal", "critique"}
            )[:400]
            return (f"[my working reasoning on this, verify it before "
                    f"answering] {steps or 'straight chain of steps'} → "
                    f"working answer: {result.answer[:300]}")
        except Exception:  # noqa: BLE001 — never let a bad pass kill the reply
            _log.debug("partner reasoning pass failed", exc_info=True)
            return ""

    def _should_speak_in_group(self, message: ChatMessage) -> bool:
        """Groups: she listens more than she talks. She speaks when her name
        comes up, the platform tags/mentions the account, or someone is
        replying to a message in that chat."""
        text = message.text.lower()
        name = self.persona.name.lower()
        if message.mentioned:
            return True
        if message.reply_to:
            return True
        if name in text:
            return True
        return False

    # ── curator ──────────────────────────────────────────────────────────────
    def _maybe_curate(self, chat_key: str, is_owner: bool) -> None:
        if not is_owner:
            return
        db = self.context.db
        try:
            row = db.query_one("SELECT value FROM kv_store WHERE key = ?", (f"curated.{chat_key}",))
            last = float(json.loads(row["value"]).get("ts", 0.0)) if row else 0.0
        except Exception:  # noqa: BLE001
            last = 0.0
        count = int(db.scalar(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ? AND role = 'user' "
            "AND created_at > ?", (chat_key, last),
            default=0,
        ))
        if count < 6 or (time.time() - last) < 900:
            return
        try:
            db.execute(
                "INSERT INTO kv_store (key, value, kind, updated_at) VALUES (?, ?, 'json', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (f"curated.{chat_key}", json.dumps({"ts": time.time()}), time.time()),
            )
        except Exception:  # noqa: BLE001
            pass
        if self.curator_hook is not None:
            try:
                self.curator_hook(chat_key)
            except Exception as exc:  # noqa: BLE001
                _log.warning("curator hook failed: %s", exc)
        else:
            self.curate(chat_key)

    def curate(self, chat_key: str) -> dict[str, Any]:
        """Distill a conversation segment into long-term memory. Sub-agent work."""
        db = self.context.db
        rows = db.query(
            "SELECT role, content FROM messages WHERE conversation_id = ? AND role IN ('user','assistant') "
            "ORDER BY created_at DESC LIMIT 24",
            (chat_key,),
        )
        rows.reverse()
        if len(rows) < 4:
            return {"curated": False, "reason": "too short"}
        transcript = "\n".join(f"{'them' if r['role'] == 'user' else self.persona.name}: {r['content']}"
                               for r in rows if r["content"])
        prompt = (
            "You are a memory-distillation service for a companion AI. Summarize this "
            "conversation segment into long-term memory. Respond with ONLY JSON:\n"
            '{"summary": "<2-4 sentences, third person, what happened and how it felt>", '
            '"facts": [{"subject": "user|partner|world", "predicate": "<verb phrase>", '
            '"object": "<value>", "confidence": 0.0-1.0}], '
            '"milestone": "<a relationship milestone, or null>", '
            '"open_loops": ["<a thread left hanging — a promise, a plan not made, an '
            'unanswered question, something to follow up on. None if nothing is open>"], '
            '"mood_after": "<one word>"}\n\n'
            f"Segment:\n{transcript[:6000]}"
        )
        try:
            response = self.context.router.chat(
                [Message.system("Output JSON only. No prose around it."), Message.user(prompt)],
                SamplingParams(temperature=0.2, max_tokens=700),
            )
        except Exception as exc:  # noqa: BLE001
            _log.warning("curator LLM failed: %s", exc)
            return {"curated": False, "error": str(exc)}
        if not response.ok or not response.text:
            return {"curated": False, "error": response.error or "empty"}
        match = _JSON_BLOCK.search(response.text)
        if not match:
            return {"curated": False, "error": "no json in response"}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            return {"curated": False, "error": f"bad json: {exc}"}

        summary = str(data.get("summary") or "").strip()
        written: dict[str, Any] = {"curated": bool(summary), "summary": summary[:200]}
        if summary:
            try:
                self.context.memory.remember(summary, kind="episode", source=f"curate:{chat_key}")
            except Exception as exc:  # noqa: BLE001
                _log.warning("curator memory write failed: %s", exc)
        for fact in (data.get("facts") or [])[:8]:
            if not isinstance(fact, dict):
                continue
            subject = str(fact.get("subject") or "user").lower()
            predicate = str(fact.get("predicate") or "").strip()
            object_ = str(fact.get("object") or "").strip()
            confidence = max(0.0, min(1.0, float(fact.get("confidence") or 0.7)))
            if subject == "user" and predicate and object_:
                try:
                    self.relationship.note_user_fact(predicate, object_)
                except Exception:  # noqa: BLE001
                    pass
            if predicate and object_:
                try:
                    db.execute(
                        "INSERT INTO facts (id, subject, predicate, object, confidence, provenance, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (ulid_now(), subject, predicate, object_, confidence, f"curate:{chat_key}", time.time()),
                    )
                except Exception as exc:  # noqa: BLE001
                    _log.debug("curator fact write failed: %s", exc)
        milestone = data.get("milestone")
        if isinstance(milestone, str) and milestone.strip():
            try:
                self.relationship.add_milestone(milestone.strip()[:200], kind="moment")
                self.mood.note_event("milestone", intensity=0.6)
            except Exception:  # noqa: BLE001
                pass
        written["open_loops"] = self._merge_open_loops(data.get("open_loops"), chat_key)
        try:
            self.relationship.save(db)
        except Exception:  # noqa: BLE001
            pass
        return written

    def _merge_open_loops(self, new_loops: Any, chat_key: str) -> list[str]:
        """Fold the curator's fresh open threads into the running list.

        Loops age out after 48h; duplicates (same thread re-noticed) are
        refreshed instead of duplicated. Kept under kv 'partner.open_loops'.
        """
        db = self.context.db
        try:
            row = db.query_one("SELECT value FROM kv_store WHERE key = 'partner.open_loops'")
            existing = json.loads(row["value"] or "[]") if row else []
        except Exception:  # noqa: BLE001
            existing = []
        if not isinstance(existing, list):
            existing = []
        now = time.time()
        fresh = [loop for loop in existing if isinstance(loop, dict) and now - float(loop.get("ts", 0)) < 48 * 3600]
        for text in new_loops if isinstance(new_loops, list) else []:
            if not isinstance(text, str) or not text.strip():
                continue
            text = text.strip()[:140]
            for loop in fresh:
                if str(loop.get("text", "")).lower() == text.lower():
                    loop["ts"] = now
                    break
            else:
                platform = chat_key.split(":", 1)[0]
                fresh.append({"text": text, "platform": platform, "ts": now})
        fresh = fresh[-10:]
        try:
            db.execute(
                "INSERT INTO kv_store (key, value, kind, updated_at) VALUES (?, ?, 'json', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                ("partner.open_loops", json.dumps(fresh), now),
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("open_loops write failed: %s", exc)
        return [str(loop.get("text", "")) for loop in fresh]


# ── the runtime ────────────────────────────────────────────────────────────────


class PartnerRuntime:
    """Runs the partner across all configured platforms, at the same time."""

    def __init__(
        self,
        context: Any,
        *,
        gateway: ChatGateway | None = None,
        brain: PartnerBrain | None = None,
        dry_run: bool = False,
    ) -> None:
        self.context = context
        self.settings = context.settings
        self.brain = brain or PartnerBrain(context)
        # wave 87: the Core Mind — the always-on layer that routes a
        # natural-language goal to the right organ (owner DMs only;
        # commands remain the manual override everywhere).
        from .coremind import CoreMind

        self.mind = CoreMind(context, runtime=self)
        self.gateway = gateway
        self.dry_run = dry_run
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
        self._autonomy: Any = None
        self.stats = {"messages": 0, "replies": 0, "errors": 0, "controls": 0}
        self._owner_chats = _key_set(partner_cfg.owner_chats)
        self._book_busy: set[str] = set()  # slugs with a book pipeline running
        self._apply_persisted_mode()
        self._adopt_power_mode()

        if self.gateway is None:
            from ..social.chat import build_adapters

            adapters, skipped = build_adapters(self.settings, on_new_member=self._on_new_discord_member)
            if skipped:
                _log.warning("skipped chat adapters: %s", skipped)
            if not adapters:
                from ..social.chat.local import LocalAdapter

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
            )

        # Tools that deliver files into chats (file_send, report_publish)
        # reach the live gateway through the context, not a second wire.
        context.extras["gateway"] = self.gateway

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
        self._pool.submit(self._greet_discord_newcomer, guild_name, member)

    def _greet_discord_newcomer(self, guild_name: str, member: dict) -> None:
        from ..social.chat.base import ChatMessage, ChatRef

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
            self.stats["replies"] += 1
            self._send_reply(message, outcome.parts)

    # ── inbound fan-in: per-chat FIFO, cross-chat parallel ──────────────────
    def on_message(self, message: ChatMessage) -> None:
        key = message.chat.key
        with self._queue_guard:
            queue = self._queues.setdefault(key, deque())
            queue.append(message)
            draining = self._draining.get(key, False)
            self._draining[key] = True
        if draining:
            return  # a pump is already working this chat; it will pick the message up
        self._pool.submit(self._pump, key)

    def _pump(self, key: str) -> None:
        try:
            while not self._stopped.is_set():
                with self._queue_guard:
                    queue = self._queues.get(key)
                    message = queue.popleft() if queue else None
                    if message is None:
                        self._draining[key] = False
                        break
                self._process(message)
        except Exception as exc:  # noqa: BLE001 - a bad chat must not kill the pool
            _log.exception("pump for %s crashed: %s", key, exc)
            with self._queue_guard:
                self._draining[key] = False

    def _process(self, message: ChatMessage) -> None:
        self.stats["messages"] += 1
        # Vision flag: with it off, inbound media is dropped before the brain
        # (no download-to-understanding pipeline, no token cost).
        if message.media:
            from .features import feature_enabled

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
            from ..social.chat.control import GAME_COMMANDS, parse_control

            command = parse_control(message.text)
            _log.debug("game command check: text=%r, command=%s", 
                      message.text[:40], command)
            if (command is not None
                    and (command.kind == "game" or command.kind in GAME_COMMANDS)):
                verb = (command.tail or command.arg) if command.kind == "game" \
                    else command.kind + ((" " + command.tail) if command.tail else "")
                self.stats["controls"] += 1
                _log.info("game command %r from %s in %s (any-chat dispatch)",
                          message.text[:40], message.sender or "?", message.chat.key)
                try:
                    reply = self._control_game(
                        verb, chat_key=message.chat.key,
                        player=self._game_player(message), kind=message.chat.kind)
                    _log.debug("game command reply: %r", reply[:100] if reply else "")
                except Exception as exc:  # noqa: BLE001
                    _log.exception("game command failed: %s", exc)
                    reply = f"control error: {exc}"
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
        # Control commands: from the console or the owner chat, a *known*
        # slash command is a command, not conversation. Unknown slashes and
        # stray / from non-operators fall through to her as ordinary text.
        if message.text.strip().startswith("/") and self._is_operator(message):
            from ..social.chat.control import parse_control

            if parse_control(message.text) is not None:
                self.stats["controls"] += 1
                # Feed the arena's interest profiler: what the owner runs
                # most shapes future topic picks. Never raises.
                try:
                    from .arena import activity as _arena_activity

                    verb = message.text.strip().split()[0].lstrip("/").lower()
                    _arena_activity.record(self.context.db, "command", verb)
                except Exception:  # noqa: BLE001
                    pass
                try:
                    reply = self.handle_control(message.text, message.chat.key,
                                                message=message)
                except Exception as exc:  # noqa: BLE001
                    _log.exception("control command failed: %s", exc)
                    reply = f"control error: {exc}"
                if reply:
                    try:
                        self._typing_for(message.chat, reply)
                        self.gateway.send(message.chat.platform, message.chat, reply)
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
                self.stats["controls"] += 1
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
        keepalive_stop = threading.Event()
        if self.settings.partner.typing_while_thinking and not self.dry_run:
            threading.Thread(
                target=self._typing_keepalive,
                args=(message.chat, keepalive_stop),
                name=f"typing-keepalive-{message.chat.key}",
                daemon=True,
            ).start()
        try:
            outcome = self.brain.handle_message(message)
        except Exception as exc:  # noqa: BLE001
            self.stats["errors"] += 1
            _log.exception("brain failed on %s: %s", message.chat.key, exc)
            keepalive_stop.set()
            return
        finally:
            keepalive_stop.set()
        if not outcome.parts:
            if outcome.presence.delay_seconds > 0:
                self._schedule_delayed(message, outcome.presence)
            return
        self.stats["replies"] += 1
        self._send_reply(message, outcome.parts)

    def _schedule_delayed(self, message: ChatMessage, presence: Presence) -> None:
        """She's busy: the reply lands later, on its own daemon thread.

        A dedicated thread (not the chat pool) — a half-hour gap must not
        occupy one of the ``max_parallel_chats`` workers. The brain's state
        already heard the message; ``deliver_reply`` only generates + sends.
        """
        delay = presence.delay_seconds

        def _job() -> None:
            if self._stopped.is_set():
                return
            time.sleep(delay)
            if self._stopped.is_set():
                return
            try:
                parts = self.brain.deliver_reply(message)
                if parts:
                    self.stats["replies"] += 1
                    self._send_reply(message, parts)
            except Exception:  # noqa: BLE001 - a late reply must not crash the thread
                _log.exception("delayed reply for %s failed", message.chat.key)

        threading.Thread(target=_job, name=f"delayed-reply-{message.chat.key}", daemon=True).start()

    def _send_reply(self, message: ChatMessage, parts: list[str]) -> None:
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
                from .autonomy import AutonomyAgent

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
                from .arena import Arena

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
        from .features import feature_enabled as _research_flag_on
        if _research_flag_on(self.context, "research") and not self.dry_run:
            try:
                from .notifier import Notifier
                from .researcher import ResearchAgent

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
                from .scheduler import Scheduler

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
                    from .skill_evolution import ensure_improvement_schedule
                    ensure_improvement_schedule(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "self-improvement schedule not registered: %s", exc)
                # Prompt 03: watchers sweeper — one durable "watchers sweep"
                # job (every 1m) that runs the watch tool's sweep action.
                # ensure_sweeper_job is idempotent by name; the sweep itself
                # is a no-op when no watchers exist.
                try:
                    from .watchers import ensure_sweeper_job
                    ensure_sweeper_job(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "watchers sweeper job not registered: %s", exc)
                # Prompt 05: rooms tick — one durable "rooms tick" job
                # (every 5m) that runs the room tool's tick action, advancing
                # each active room's linked goal/project.  Idempotent by
                # name; a no-op when no rooms exist.
                try:
                    from ..workspace.rooms import ensure_rooms_tick_job
                    ensure_rooms_tick_job(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning(
                        "rooms tick job not registered: %s", exc)
                # Prompt 11: persona rebuild + curation — two durable daily
                # jobs (rebuild at 03:30, curate at 04:00) that keep the
                # user model fresh and memory hygienic.  Idempotent by name;
                # no-ops when memory is empty.
                try:
                    from .scheduler import ensure_persona_jobs
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
                    from .morning_briefing import (
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
                    from .research_loop import ensure_research_job
                    ensure_research_job(self.context)
                except Exception as exc:  # noqa: BLE001 - optional
                    _log.warning("research loop job not registered: %s", exc)
                # The cognitive loop (wave 51): one heartbeat that ticks
                # goals (driving linked projects), improvement, and the
                # personal-model fine-tune. On when the autonomy dial is on
                # (config, power mode, or the durable `nm autonomy on`).
                # Idempotent.
                from .cognition import autonomy_enabled as _auto_on
                if _auto_on(self.context):
                    try:
                        have = [j for j in self._scheduler.list_jobs()
                                if j.get("name") == "cognitive loop"]
                        if not have:
                            try:
                                from .cognition import CognitiveLoop

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
                    from .monitor import MonitorAgent as _MonitorAgent

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
        return started

    def _arena_notify(self, text: str) -> None:
        """Deliver an arena build review to the owner, on every live channel."""
        from ..social.chat.base import ChatRef

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
        from .beacon import BEACON_INTERVAL_S

        now = time.time()
        if not force and now < self._beacon_next:
            return
        self._beacon_next = now + BEACON_INTERVAL_S
        try:
            from .beacon import build_beacon_state, write_status

            write_status(self.settings.home, build_beacon_state(self))
        except Exception:  # noqa: BLE001 - beacon must never take the bot down
            _log.debug("status beacon write failed", exc_info=True)

    def run(self, *, duration: float | None = None) -> None:
        self._started = time.time()
        self._beacon_next = 0.0  # write one immediately at startup
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
        self._pool.shutdown(wait=False)
        # final beacon, marked stopped — `nm status` then reports a clean
        # "stopped" instead of "went stale"
        try:
            from .beacon import build_beacon_state, write_status

            state = build_beacon_state(self)
            state["stopped"] = True
            write_status(self.settings.home, state)
        except Exception:  # noqa: BLE001
            pass
        _log.info("partner runtime stopped: %s", self.stats)

    def status(self) -> dict[str, Any]:
        return {
            "stats": dict(self.stats),
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
        try:
            from .power import power_mode_for

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
        return is_owner_chat(message.chat, owner_chats=self._owner_chats)

    def _build_adapter(self, name: str) -> Any:
        """Factory for hot starts. Returns the adapter, None for an unknown
        or disabled platform, and raises with a readable message when a
        platform is enabled but unavailable (missing dependency/credentials) —
        the gateway reports that message to the owner."""
        from ..social.chat import build_adapter

        return build_adapter(self.settings, name)

    def handle_control(self, text: str, chat_key: str,
                       message: Any = None) -> str:
        """Parse + dispatch one control command. Returns the reply to send."""
        from ..social.chat.control import detailed_help, parse_control
        from .power import power_mode_for

        command = parse_control(text)
        if command is None:
            return ""  # not actually a control command; let normal flow handle it
        kind, arg = command.kind, command.arg
        if kind == "error":
            return arg

        if kind in {"list", "commands", "menu"}:
            # wave 68: /list — every executable chat command, categorized
            # with a one-line "what it does"; /list <group> filters.
            from ..social.chat.control import list_catalog
            return list_catalog(arg)
        if kind == "help":
            # wave 67: detailed help — /help for the catalog,
            # /help <command> for one command's full page, /help budget
            # etc. for topic pages.
            return detailed_help(arg or command.tail)

        if kind == "profile":
            # wave 86: the profile-aware runtime, in chat (owner-only —
            # control commands never reach non-operators)
            from ..core.runtune import build_tune

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

        if kind == "say":
            target, sep, payload = command.tail.partition(" ")
            payload = payload.strip()
            if not sep or not payload or ":" not in target:
                return "usage: /say telegram:123 the text to send"
            platform, _, chat_id = target.partition(":")
            if not chat_id:
                return "usage: /say telegram:123 the text to send"
            from ..social.chat.base import ChatRef

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
            return (f"approved {arg} → {result.get('status', 'sent')}" if result.get("ok")
                    else f"approve failed: {result.get('error', '?')}")

        if kind == "deny":
            result = self.deny(arg)
            return "denied." if result.get("ok") else f"deny failed: {result.get('error', '?')}"

        if kind == "stage":
            rel = self.brain.relationship
            if not arg:
                return f"stage: {rel.stage} (trust {rel.trust:.0f})"
            from ..partner.relationship import STAGES

            arg = arg.strip().lower()
            if arg not in STAGES:
                return f"unknown stage {arg!r}. use one of: {', '.join(STAGES)}"
            from ..partner.relationship import STAGE_ORDER

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
        if kind == "book":
            return self._control_book(tail=command.tail or arg, chat_key=chat_key)
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
            return self._control_music(command.tail or arg)
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
            return self._control_arena(command.tail or arg, chat_key=chat_key)

        # ── single-account trials ────────────────────────────────────────────
        if kind == "trial":
            return self._control_trial(command.tail or arg, chat_key=chat_key)

        # ── expansion wave ───────────────────────────────────────────────────
        if kind == "game":
            # console path: no inbound message, so the player is derived from
            # the chat key (stable per chat) and the kind defaults to dm.
            return self._control_game(
                command.tail or arg, chat_key=chat_key,
                player=self._game_player_for_key(chat_key), kind="dm")
        if kind == "news":
            return self._control_news(command.tail or arg)
        if kind == "research":
            return self._control_research(command.tail or arg)
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
            from .opportunities import handle_money_command
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
        from ..social.chat.base import ChatRef

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

    def _control_search(self, tail: str, mode: str, chat_key: str) -> str:
        from .features import feature_enabled
        from .power import power_mode_for
        from .search.engine import SearchEngine

        if not feature_enabled(self.context, "search"):
            return "search is off. turn it on: /features search on"
        query = (tail or "").strip()
        if not query:
            return "usage: /search <what to research>"
        if mode == "deep" and not power_mode_for(self.context).active:
            return "deep research is a power-mode capability: /power on <key> first"
        chat = self._ref_from_key(chat_key)
        try:
            self.gateway.send(chat.platform, chat,
                               f"⏳ {mode} research: {query[:80]}\nthis takes a moment — the report comes right here.")
        except Exception:  # noqa: BLE001 - progress note is best-effort
            pass
        started = time.time()
        try:
            report = SearchEngine(self.context).run(query, mode=mode,
                                                    pages=8 if mode == "deep" else 3)
        except Exception as exc:  # noqa: BLE001
            return f"search failed: {exc}"
        elapsed = time.time() - started
        text = self._format_search_report(report, query, mode, elapsed)
        return self._send_long_checked(chat.platform, chat, text)  # report already delivered in chunks

    @staticmethod
    def _format_search_report(report: dict, query: str, mode: str, elapsed: float) -> str:
        pages_read = report.get("pages_read") or []
        n_pages = len(pages_read) if isinstance(pages_read, list) else int(pages_read or 0)
        sub_queries = report.get("sub_queries") or []
        lines = [
            f"🔎 {mode} research: {query[:90]}",
            f"({n_pages} pages read · {elapsed:.0f}s · "
            f"{'model-sourced' if report.get('model_summary') else 'extractive'} summary)",
        ]
        if len(sub_queries) > 1:
            lines.append("sub-queries: " + " | ".join(s[:48] for s in sub_queries[:5]))
        summary = str(report.get("summary") or "").strip()
        if summary:
            lines.append("")
            lines.append(summary)
        numbered = report.get("sources")
        if numbered:
            lines.append("")
            lines.append("sources:")
            for s in numbered[:8]:
                title = (s.get("title") or "").strip()
                lines.append(f"  [{s.get('n')}] {title[:70]}\n      {s.get('url')}")
        else:
            sources = report.get("results") or []
            if sources:
                lines.append("")
                lines.append("main sources:")
                for s in sources[:6]:
                    lines.append(f"  • {s.get('title') or s.get('url')}\n    {s.get('url')}")
        return "\n".join(lines)

    def _control_search_leads(self) -> str:
        from .features import feature_enabled
        from .search.engine import SearchEngine

        if not feature_enabled(self.context, "search"):
            return "search is off. turn it on: /features search on"
        try:
            leads = SearchEngine(self.context).leads()
        except Exception as exc:  # noqa: BLE001
            return f"leads research failed: {exc}"
        if not leads:
            return ("no leads surfaced (endpoints may be unreachable from here).\n"
                    "try again later, or /search <platform> to check one directly.")
        lines = [f"legit paid-task platforms — {len(leads)} (research report, pick one you trust):"]
        for lead in leads[:12]:
            lines.append(f"  • {lead.get('title') or lead.get('domain')} — {lead.get('url')}")
        lines.append("")
        lines.append("to try one: /trial start <name>")
        return "\n".join(lines)

    def _control_search_hist(self, arg: str) -> str:
        from .features import feature_enabled
        from .search.engine import SearchEngine

        if not feature_enabled(self.context, "search"):
            return "search is off. turn it on: /features search on"
        limit = int(arg) if arg.isdigit() else 5
        rows = SearchEngine(self.context).history(limit)
        if not rows:
            return "no research runs yet."
        lines = [f"recent research ({len(rows)}):"]
        for row in rows:
            when = time.strftime("%m-%d %H:%M", time.localtime(row.get("created_at", 0)))
            lines.append(f"  {when}  {row.get('mode', 'quick')}  {str(row.get('query', ''))[:60]}")
        return "\n".join(lines)

    # ── BookForge ────────────────────────────────────────────────────────────
    def _control_book(self, tail: str, chat_key: str) -> str:
        """/book <topic> [chapters] — writes a real book in the background and
        sends the finished PDF to this chat when done.  Also: list / status /
        build / send for books already on disk."""
        from ..books import BookForge
        from ..books.model import BookError, slugify

        query = (tail or "").strip()
        parts = query.split()
        verb = parts[0].lower() if parts else ""

        def _one_line(slug: str) -> str:
            try:
                b = forge.load(slug)
                return (f"  {slug} — {b.status} · {b.chapters_written}/{len(b.chapters)} ch · "
                        f"{b.total_words} words")
            except Exception:  # noqa: BLE001
                return f"  {slug}"

        try:
            forge = BookForge(self.context)
        except Exception as exc:  # noqa: BLE001
            return f"book system failed: {exc}"

        try:
            if verb == "list" or not verb:
                books = forge.list_books()
                if not books:
                    return "no books yet. /book <topic> [chapters] writes one and sends the pdf."
                lines = ["books:"]
                for b in books:
                    lines.append(f"  {b['slug']} — {b['status']} · {b['written']}/{b['chapters']} ch · "
                                 f"{b['words']} words")
                lines.append("\n/book status [slug] for details · /book build <slug> for the pdf")
                return "\n".join(lines)

            if verb == "status":
                if len(parts) > 1:
                    return _one_line(parts[1])
                books = forge.list_books()
                if not books:
                    return "no books on disk yet."
                return "\n".join([f"{b['slug']} — {b['status']} · {b['written']}/{b['chapters']} ch · "
                                  f"{b['words']} words" for b in books])

            if verb == "build":
                if len(parts) < 2:
                    return "usage: /book build <slug>"
                r = forge.build(parts[1])
                return (f"built {r['slug']}: {r['pages']} pages · {r['pdf_bytes']} bytes\n"
                        f"{r['pdf']}")

            if verb == "send":
                if len(parts) < 4:
                    return "usage: /book send <slug> <platform> <chat_id>"
                r = forge.send(parts[1], parts[2], parts[3])
                return f"sent {r['slug']} to {parts[2]}:{parts[3]}"

            # ── new book ────────────────────────────────────────────────────
            chapters = 8
            if parts and parts[-1].isdigit():
                chapters = max(3, min(int(parts[-1]), 16))
                topic = " ".join(parts[:-1]).strip()
            else:
                topic = query
            if not topic:
                return ("usage: /book <topic> [chapters] — e.g. /book eBPF for system security 8\n"
                        "it researches the topic, plans chapters, writes them, builds a real "
                        "pdf, and sends it here when done. /book list · /book status")
            slug = slugify(topic)
            with self._queue_guard:
                if slug in self._book_busy:
                    return f"already writing {slug!r} — check /book status {slug}"
                self._book_busy.add(slug)

            chat = self._ref_from_key(chat_key)
            try:
                self.gateway.send(
                    chat.platform, chat,
                    f"✍️ writing “{topic[:70]}” — {chapters} chapters. "
                    "research → outline → write → pdf → straight to this chat. "
                    "check progress: /book status " + slug,
                )
            except Exception:  # noqa: BLE001 - start note is best-effort
                pass

            def _work() -> None:
                try:
                    try:
                        book = forge.create(topic, chapters=chapters, research=True)
                        title = book.display_title
                    except BookError as exc:
                        self._notify(chat, f"⚠️ book failed: {exc}")
                        return
                    # mid-run note at the half-chapter mark, so a long book
                    # doesn't look dead (chapters can take real minutes)
                    half = max(1, len(book.chapters) // 2)
                    written = 0
                    while True:
                        r = forge.write_next(slug)
                        written += 1
                        if r.get("done") and not r.get("chapter"):
                            break
                        if written == half:
                            self._notify(chat, (
                                f"✍️ {title[:60]} — halfway: "
                                f"{r.get('chapters_written', written)}/{r.get('total_chapters', len(book.chapters))} "
                                f"chapters, {r.get('total_words', 0)} words so far."))
                    built = forge.build(slug)
                    caption = (f"📕 {title} — {built.get('chapters_written')} chapters, "
                               f"{built.get('words')} words, {built.get('pages')} pages")
                    try:
                        self.gateway.send_file(chat.platform,
                                               f"{chat.platform}:{chat.chat_id}",
                                               built["pdf"], caption=caption)
                        self._notify(chat, f"📕 done — “{title[:70]}” is on its way. "
                                           f"/book status {slug} for the record.")
                    except Exception as send_exc:  # noqa: BLE001 - the book exists either way
                        self._notify(chat, (f"📕 “{title[:70]}” is finished but the send "
                                            f"failed ({send_exc}).\n"
                                            f"pdf: {built['pdf']}"))
                except Exception as exc:  # noqa: BLE001
                    self._notify(chat, f"⚠️ book {slug} failed: {exc}")
                finally:
                    with self._queue_guard:
                        self._book_busy.discard(slug)

            threading.Thread(target=_work, name=f"book-{slug}", daemon=True).start()
            return ""  # the start note is already on its way
        except BookError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001
            return f"book failed: {exc}"

    def _notify(self, chat: Any, text: str) -> None:
        """Best-effort chat notification (background book pipeline)."""
        try:
            self.gateway.send(chat.platform, chat, text)
        except Exception:  # noqa: BLE001
            pass

    # ── universal decoder ────────────────────────────────────────────────────
    def _control_decode(self, tail: str, chat_key: str) -> str:
        """/decode <data> | /decode file:<path> | /decode hash <digest> | decoders.

        Runs the Universal Decoder; if the winning decode is binary it is
        saved and sent to the chat as a file."""
        tail = (tail or "").strip()
        if not tail:
            return ("usage: /decode <data>  |  /decode file:<path>  |  "
                    "/decode hash <digest>  |  /decode decoders  |  "
                    "/decode history [query]  |  /decode show <report-id>")
        from .decoder import DecoderAgent

        if tail == "decoders":
            from ..core.decoder import DECODERS
            names = ", ".join(d.name for d in DECODERS)
            return f"{len(DECODERS)} decoders: {names}"
        if tail.startswith("history") or tail.startswith("show "):
            from ..core.decoder import decode_history, get_report

            if tail.startswith("show "):
                rid = tail[5:].strip()
                row = get_report(getattr(self.context, "db", None), rid)
                if row is None:
                    return f"no report {rid!r} — /decode history"
                rep = row.get("report") or {}
                best = rep.get("best") or {}
                return (f" {row['id']} · {row['source']} · "
                        f"{best.get('chain') if isinstance(best, dict) else '—'}\n"
                        f"  input: {row['input_head'][:160]}")
            query = tail[7:].strip()
            rows = decode_history(getattr(self.context, "db", None),
                                  query=query, limit=10)
            if not rows:
                return "no archived decodes yet"
            lines = [f"📜 decode archive ({len(rows)} most recent):"]
            for r in rows:
                lines.append(
                    f"  {r['id']} {time.strftime('%m-%d %H:%M', time.localtime(r['ts']))} "
                    f"{r['source'][:20]:<20} {r['best_name'] or '—':<9} "
                    f"{(r['input_head'] or '').replace(chr(10), ' ')[:40]}")
            lines.append("  · /decode show <id>")
            return "\n".join(lines)
        if tail.startswith("hash "):
            from ..core.decoder import (identify_hash,
                                        known_hash_lookup_chained)
            digest = tail[5:].strip()
            cands = identify_hash(digest)
            known = (known_hash_lookup_chained(
                getattr(self.context, "db", None), digest)
                if cands else None)
            line = (f"{digest[:16]}… → {', '.join(cands) or 'unknown'}")
            if known:
                line += f"  ·  KNOWN: {known['algorithm']} of {known['plaintext']!r}"
            return line

        spec: dict[str, Any] = {"save": True, "explain": True}
        if tail.startswith("file:"):
            spec["path"] = tail[5:].strip()
        else:
            spec["data"] = tail
        agent = DecoderAgent(context=self.context, name="chat-decoder")
        result = agent.run(spec)
        if not result.ok:
            return f"decode failed: {result.error}"
        v = result.output
        best = v["report"].get("best") or {}
        out = best.get("output")
        text = v["explanation"]
        if isinstance(out, str) and out:
            snippet = truncate(out, 800)
            text += f"\ndecoded:\n{snippet}"
        sent = ""
        if v.get("saved_to"):
            chat = self._ref_from_key(chat_key)
            try:
                res = self.gateway.send_file(
                    chat.platform, chat, v["saved_to"],
                    caption=f"decoded from {v['report'].get('target')!r}"
                    f" → {best.get('output', {}).get('magic')}"
                    if isinstance(best.get("output"), dict) else "decoded file")
                if getattr(res, "ok", False):
                    sent = " (file sent)"
            except Exception:  # noqa: BLE001
                sent = f" (saved at {v['saved_to']})"
        return text + sent

    def _control_cookies(self, tail: str) -> str:
        """``/cookies <header>`` / ``/cookies file:<path>`` /
        ``/cookies ingest <header>`` — the CookieLab report."""
        tail = (tail or "").strip()
        if not tail:
            return "usage: /cookies <cookie-header>  |  /cookies file:<path>" \
                   "  |  /cookies ingest <cookie-header>"
        ingest = False
        if tail.lower().startswith("ingest "):
            ingest = True
            tail = tail[7:].strip()
        try:
            from pathlib import Path

            from ..core.cookies import CookieLab
            lab = CookieLab()
            raw = tail
            if raw.startswith("file:"):
                p = Path(raw[5:])
                if not p.is_file():
                    return f"no such file: {p}"
                raw = p.read_text(encoding="utf-8", errors="replace")
            rep = lab.report(raw)
        except Exception as exc:  # noqa: BLE001
            return f"cookie analysis failed: {exc}"
        lines = [f"cookies: {rep['count']} parsed"]
        if rep["services"]:
            lines.append("services: " + ", ".join(rep["services"]))
        kinds = rep.get("kinds", {})
        if kinds:
            lines.append("kinds: " + ", ".join(f"{k}={n}"
                                               for k, n in kinds.items()))
        for c in rep.get("cookies", []):
            flags = c.get("flags", {})
            f_txt = ""
            if flags:
                f_txt = " [" + ", ".join(k for k in
                                         ("httponly", "secure", "samesite")
                                         if k in flags) + "]"
            dec = c.get("decoded_value")
            d_txt = ""
            if dec is not None:
                d_txt = f"  decoded({c.get('decode_via')}): {str(dec)[:80]}"
            svc = f" <{c['service']}>" if c.get("service") else ""
            lines.append(f"  {c['name']} = {str(c['value'])[:48]}  "
                         f"[{c.get('kind')}]{svc}{f_txt}{d_txt}")
        sec = rep.get("security", {})
        warns = sec.get("plaintext_auth") or []
        if warns:
            lines.append("security: " + ", ".join(warns)
                         + " lack HttpOnly+Secure")
        if ingest:
            try:
                from .kg import KnowledgeGraph
                ing = lab.ingest(self.context, raw, source="chat-cookies",
                                 graph=KnowledgeGraph(self.context.db))
                lines.append(f"ingested: {ing.get('nodes', 0)} graph nodes"
                             + (f", report {ing.get('report_id')}"
                                if ing.get("report_id") else ""))
            except Exception as exc:  # noqa: BLE001
                lines.append(f"ingest failed: {exc}")
        return "\n".join(lines)

    def _control_structure(self, tail: str) -> str:
        """``/structure <objective>`` — the structuring sub-agent's brief."""
        tail = (tail or "").strip()
        if not tail:
            return "usage: /structure <objective>"
        try:
            from ..agents.structuring import structure_text
            brief = structure_text(self.context, tail, for_="chat",
                                   polish=True)
        except Exception as exc:  # noqa: BLE001
            return f"structuring failed: {exc}"
        return brief.get("brief", "(no brief)")

    def _control_investigate(self, tail: str) -> str:
        """/investigate <artifact> [file] — decode→crack→OSINT→KG in one pass."""
        from .investigate import InvestigateAgent

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /investigate <hash|jwt|cookie|url|blob|file> "
                    "[file]  — one pass: decode → crack → OSINT → knowledge "
                    "graph")
        is_file = tail.split()[-1].lower() == "file"
        art = tail[:-4].strip() if is_file else tail
        try:
            rep = InvestigateAgent(self.context).run(
                art, file=is_file, source="chat-investigate")
        except Exception as exc:  # noqa: BLE001
            return f"investigate error: {exc}"
        if not rep.get("ok", True):
            return f"investigate failed: {rep.get('error')}"
        lines = [f"🔬 {rep['kind']}: {rep['artifact_head'][:70]}"]
        lines.append("  " + " → ".join(rep["steps"]))
        for d, p_ in (rep.get("cracked") or {}).items():
            lines.append(f"  cracked: {d[:24]}… = {p_!r}")
        o = rep.get("osint") or {}
        if isinstance(o, dict) and o.get("persons_found"):
            lines.append(f"  identities: {', '.join(o['persons'][:6])}")
        if isinstance(o, dict) and o.get("domains"):
            lines.append(f"  domains:    {', '.join(o['domains'][:6])}")
        kg = rep.get("kg") or {}
        if isinstance(kg, dict) and kg.get("added_nodes") is not None:
            lines.append(f"  knowledge graph: +{kg.get('added_nodes', 0)} "
                         f"nodes, +{kg.get('added_links', 0)} links")
        if rep.get("report_id"):
            lines.append(f"  report: {rep['report_id']}  "
                         "(/decode show <id>)")
        return "\n".join(lines)

    def _control_monitor(self, tail: str) -> str:
        """/monitor add <target> [every Ns] | list | tick | rm <ref> | status."""
        tail = (tail or "").strip()
        from .monitor import MonitorAgent
        from .notifier import Notifier

        agent = MonitorAgent(self.context, notifier=Notifier(
            self.context))
        if not tail or tail == "list":
            rows = agent.list()
            if not rows:
                return "no monitors — /monitor add <url-or-file> [every 300s] [--webhook URL] [--min-gap 60]"
            lines = [f"{r['target'][:40]} · {r['kind']} · every "
                     f"{r['interval_s']:.0f}s · {'on' if r['enabled'] else 'off'}"
                     + (f" · webhook→{r['webhook_url'][-30:]}"
                        if r.get("webhook_url") else "")
                     + (f" · gap {r['min_alert_gap_s']:.0f}s"
                        if r.get("min_alert_gap_s") else "")
                     for r in rows]
            return f"{len(rows)} monitor(s):\n" + "\n".join(lines)
        if tail.startswith(("add ", "alert ", "webhook-test ")):
            verb, body = tail.split(None, 1)
            body = body.strip()
            import re

            def _flag(flag: str):
                m = re.search(rf"(?<![\w./-]){flag}\s+(\S+)", body)
                if not m:
                    return None
                body_ = (body[:m.start()] + " " + body[m.end():]).strip()
                return m.group(1), body_

            def _flag_on(flag: str):
                m = re.search(rf"(?<![\w./-]){flag}(?:\s|$)", body)
                if not m:
                    return False, body
                body_ = (body[:m.start()] + " " + body[m.end():]).strip()
                return True, body_

            webhook = ""
            m = _flag("--webhook")
            if m:
                webhook, body = m
            secret = ""
            m = _flag("--secret")
            if m:
                secret, body = m
            no_decode, body = _flag_on("--no-decode")
            min_gap = -1.0
            m = _flag("--min-gap")
            if m:
                try:
                    min_gap = float(m[0])
                except ValueError:
                    min_gap = -1.0
                body = m[1]
            if verb == "webhook-test":
                if not body:
                    return "usage: /monitor webhook-test <ref>"
                res = agent.webhook_test(body)
                if res is None:
                    return f"no monitor {body!r}"
                if not res.get("ok"):
                    err = res.get("error") or (res.get("result") or {}).get(
                        "error", "")
                    return f"webhook test FAILED: {err or 'no response'}"
                r = res["result"]
                return (f"webhook test OK — HTTP {r.get('status')} "
                        f"({r.get('attempts', 1)} attempt(s)) → "
                        f"{res['target']}")
            if verb == "add":
                interval = 300.0
                m = re.search(r"\bevery\s+(\d+(?:\.\d+)?)\s*s?\b", body)
                if m:
                    interval = float(m.group(1))
                    body = (body[:m.start()] + body[m.end():]).strip()
                target = body
                if not target:
                    return ("usage: /monitor add <target> [every 300s] "
                            "[--webhook URL] [--min-gap 60]")
                gap = 60.0 if min_gap < 0 else max(0.0, min_gap)
                info = agent.add(target, interval=interval,
                                 webhook=webhook, secret=secret,
                                 min_gap=gap, auto_decode=not no_decode)
                return (f"watching {target} ({info['kind']}, every "
                        f"{info['interval_s']:.0f}s) — I'll alert you on change"
                        + (f" · webhook → {info['webhook_url']}"
                           if info.get("webhook_url") else "")
                        + (f" · at most one alert per "
                           f"{info['min_alert_gap_s']:.0f}s"
                           if info.get("min_alert_gap_s") else ""))
            # alert
            if not body:
                return ("usage: /monitor alert <ref> [--webhook URL] "
                        "[--secret S] [--min-gap 60] [--no-decode]")
            row = agent.set_alerting(
                body,
                webhook=webhook or None,
                secret=secret or None,
                min_gap=min_gap if min_gap >= 0 else None,
                auto_decode=False if no_decode else None)
            if row is None:
                return f"no monitor {body!r}"
            return (f"alerting for {row['target']}:"
                    f" webhook={'on → ' + row['webhook_url'] if row['webhook_url'] else 'off'}"
                    f" · gap={row['min_alert_gap_s']:.0f}s"
                    + (" (every change)" if not row["min_alert_gap_s"] else ""))
        if tail == "tick":
            res = agent.tick()
            parts = [f"checked {res['checked']} due"]
            for c in res["changed"]:
                parts.append(f"changed: {c['target']}")
            for e in res["errors"]:
                parts.append(f"error: {e['target']} ({e['error'][:60]})")
            return "; ".join(parts)
        if tail.startswith("rm "):
            ref = tail[3:].strip()
            return "removed" if agent.remove(ref) else f"no monitor {ref!r}"
        if tail == "status":
            st = agent.status()
            return f"{st['enabled']}/{st['total']} monitors active"
        return "usage: /monitor add <target> [every Ns] | list | tick | rm <ref>"

    def _control_cipher(self, tail: str) -> str:
        """/cipher enc <data> with <passphrase> | /cipher dec <blob> with <passphrase>
        | /cipher vault put <name> <secret> with <passphrase>
        | /cipher vault get <name> with <passphrase>
        | /cipher vault list | /cipher vault rm <name> with <passphrase>."""
        import re

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /cipher enc <data> with <passphrase>  |  "
                    "/cipher dec <blob> with <passphrase>  |  "
                    "/cipher vault put|get|list|rm …")
        from .cipher import CipherAgent
        from ..core.cipher import CipherError

        agent = CipherAgent(context=self.context, name="chat-cipher")

        # ── named secrets vault ──
        if tail.startswith("vault "):
            body = tail[6:].strip()
            parts = body.split(" with ", 1)
            head, passphrase = (parts[0].strip(), parts[1].strip()) \
                if len(parts) == 2 else (body, "")
            tokens = head.split()
            verb = tokens[0].lower() if tokens else ""
            rest = tokens[1:]
            try:
                if verb == "list":
                    r = agent.run({"action": "vault_list"})
                    if not r.ok:
                        return f"vault failed: {r.error}"
                    entries = r.output.get("entries") or [
                        {"name": n, "key_scheme": "pass"}
                        for n in r.output.get("names", [])]
                    return ("🗝 vault:\n" +
                            ("\n".join(f"  {e['name']}  [{e['key_scheme']}]"
                                       for e in entries)
                             if entries else "  (empty)"))
                if verb == "export" and rest:
                    # /cipher vault export <file> with <key> [entry <epass>]
                    m = re.match(
                        r"^(?P<file>\S+)(?:\s+entry\s+(?P<epass>\S+))?$",
                        " ".join(rest))
                    if not m:
                        return ("usage: /cipher vault export <file> "
                                "with <export-key> [entry <epass>]")
                    r = agent.run({"action": "vault_export",
                                   "path": m.group("file"),
                                   "passphrase": passphrase,
                                   "entry_pass": m.group("epass") or ""})
                    if not r.ok:
                        return f"vault export failed: {r.error}"
                    skipped = f"  (skipped: {', '.join(r.output['skipped'])})" \
                        if r.output.get("skipped") else ""
                    return (f"📦 exported {r.output['exported']} entries "
                            f"→ {r.output['path']}{skipped}")
                if verb == "import" and rest:
                    r = agent.run({"action": "vault_import",
                                   "path": rest[0],
                                   "passphrase": passphrase})
                    if not r.ok:
                        return f"vault import failed: {r.error}"
                    names = ", ".join(r.output["imported"][:8])
                    return (f"📥 imported {r.output['count']} entries "
                            f"({names})")
                if verb in ("put", "get", "rm") and rest:
                    name = rest[0]
                    if verb == "put":
                        if len(rest) < 2:
                            return ("usage: /cipher vault put <name> "
                                    "<secret> with <passphrase>  (omit "
                                    "'with …' to seal under NM_VAULT_KEY)")
                        secret = " ".join(rest[1:])
                        r = agent.run({"action": "vault_put", "name": name,
                                       "data": secret,
                                       "passphrase": passphrase})
                        if not r.ok:
                            return f"vault failed: {r.error}"
                        return (f"🗝 stored {name} "
                                f"({r.output['bytes']} B sealed, AES-256, "
                                f"key: {r.output.get('key_scheme', 'pass')})")
                    if verb == "rm":
                        r = agent.run({"action": "vault_rm", "name": name})
                        if not r.ok:
                            return f"vault failed: {r.error}"
                        return (f"🗝 removed {name}"
                                if r.output["removed"]
                                else f"no entry {name!r}")
                    if not passphrase:
                        return f"usage: /cipher vault {verb} <name> with <passphrase>"
                    r = agent.run({"action": f"vault_{verb}", "name": name,
                                   "passphrase": passphrase})
                    if not r.ok:
                        return f"vault failed: {r.error}"
                    if verb == "get":
                        return f"🔓 {name}: " + r.output.get(
                            "data", r.output.get("hex", ""))
                return ("usage: /cipher vault put <name> <secret> with <pass>  |  "
                        "/cipher vault get <name> with <pass>  |  "
                        "/cipher vault list  |  /cipher vault rm <name>  |  "
                        "/cipher vault export <file> with <key> [entry <epass>]  |  "
                        "/cipher vault import <file> with <pass>")
            except CipherError as exc:
                return f"vault error: {exc}"

        m = re.match(r"^(enc|decrypt|dec)\s+(.+?)\s+with\s+(\S+)\s*$",
                     tail, re.I)
        if not m:
            return ("usage: /cipher enc <data> with <passphrase>  |  "
                    "/cipher dec <blob> with <passphrase>  |  "
                    "/cipher vault put|get|list|rm …")
        verb, body, passphrase = m.group(1).lower(), m.group(2).strip(), m.group(3)
        try:
            if verb == "enc":
                r = agent.run({"action": "encrypt", "data": body,
                               "passphrase": passphrase})
                if not r.ok:
                    return f"cipher failed: {r.error}"
                return f"🔒 sealed ({r.output['bytes']} bytes):\n{r.output['blob']}"
            r = agent.run({"action": "decrypt", "blob": body,
                           "passphrase": passphrase})
            if not r.ok:
                return f"cipher failed: {r.error}"
            return f"🔓 {r.output.get('data', r.output.get('hex', ''))}"
        except CipherError as exc:
            return f"cipher error: {exc}"

    # ── wave 72 systems: media · execution · archives · builders ───────────

    def _control_music(self, tail: str) -> str:
        """/music <topic> [style] | /music styles | /music song [slug]."""
        from ..media.music import STYLES, MusicCreator

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /music <topic> [style]  |  /music styles  |  "
                    "/music song [slug]\nstYLES: " + ", ".join(STYLES))
        words = tail.split()
        if words[0].lower() == "styles":
            return ("styles:\n" + "\n".join(
                f"  {k:12s} {v.label}  ({v.mode}, {v.tempo[0]}-{v.tempo[1]} bpm)"
                for k, v in STYLES.items()))
        if words[0].lower() == "song":
            from ..media.music import _saved_songs

            lookup = " ".join(words[1:]).strip()
            out = _saved_songs(self.context, lookup)
            if lookup:
                if out.get("found"):
                    s = out["song"]
                    return f"found “{s['title']}” → {s['midi']}\n/play it to play."
                names = ", ".join(x["title"] for x in out.get("songs", [])[:8])
                return (f"no saved song matching {lookup!r}. Saved: "
                        f"{names or '(none)'}")
            names = [x["title"] for x in out.get("songs", [])]
            return ("saved songs:\n" + "\n".join(f"  {n}" for n in names)
                    or "saved songs: (none)\n/music <topic> composes one.")
        topic = tail
        style = "pop"
        if words and words[-1].lower() in STYLES and len(words) > 1:
            style = words[-1].lower()
            topic = " ".join(words[:-1])
        if not topic:
            return "usage: /music <topic> [style]"
        try:
            song = MusicCreator(self.context).compose(topic, style=style)
        except Exception as exc:  # noqa: BLE001
            return f"music error: {exc}"
        n_lines = sum(len(sec.lyrics) for sec in song.sections)
        text = (f"🎵 “{song.title}”  [{song.style}, {song.key}, {song.tempo} bpm]\n"
                f"{n_lines} lyric lines across {len(song.sections)} sections\n"
                f"{song.midi_path}\nqueue it with: /play {song.midi_path}")
        return text

    def _control_play(self, tail: str) -> str:
        """/play <paths…> | status | queue | pause | … (transport)."""
        from ..media.playback import PlaybackEngine

        tail = (tail or "").strip()
        actions = {"add", "pause", "resume", "stop", "seek", "volume",
                   "next", "prev", "queue", "remove", "clear", "status"}
        words = tail.split()
        action = words[0].lower() if words and words[0].lower() in actions \
            else "play"
        rest = words[1:] if action != "play" else words
        try:
            engine = PlaybackEngine(self.context)
            if action in ("play", "add"):
                if not rest:
                    st = engine.status()
                    return (f"queue ({st['queue']}): "
                            + (f"now {st['current']}" if st["queue"] else "empty")
                            + "\n/play <path-or-url…> to queue and play")
                added = []
                for ref in rest:
                    res = engine.add(ref)
                    added.extend(res.get("added", []))
                queue_len = len(engine.queue())
                out = (f"queued {len(added)} (queue {queue_len}):\n"
                       + "\n".join(f"  - {a.get('title') or a['path']}"
                                   for a in added))
                if action == "play":
                    st = engine.play()
                    if st.get("status") == "playing":
                        out += f"\n▶ playing “{st.get('current', '')}” " \
                               f"[{st.get('backend')}]"
                    else:
                        out += (f"\nstatus: {st.get('status')} "
                                f"{st.get('error') or st.get('hint', '')}")
                return out
            if action == "seek" and rest:
                return f"seeked to {rest[0]}s: {engine.seek(float(rest[0]))}"
            if action == "volume" and rest:
                return f"volume: {engine.volume(int(rest[0]))}"
            if action == "remove" and rest:
                return f"removed: {engine.remove(int(rest[0]))}"
            if action == "queue":
                items = engine.queue()
                return (f"queue ({len(items)}):\n"
                        + "\n".join(f"  {i:2d}. {it['title']} [{it['kind']}]"
                                    for i, it in enumerate(items))
                        or "queue (0): empty")
            if action in ("pause", "resume", "stop", "next", "prev", "clear"):
                return f"{action}: {getattr(engine, action)()}"
            st = engine.status()
            return (f"backend: {st['backend'].get('name') if isinstance(st['backend'], dict) else st['backend']}\n"
                    f"playing: {st['playing']}  paused: {st['paused']}\n"
                    f"current: {st['current'] or '(none)'} "
                    f"[{st['position']}/{st['queue']} in queue]\n"
                    f"volume: {st['volume']}")
        except Exception as exc:  # noqa: BLE001
            return f"play error: {exc}"

    def _control_video(self, tail: str) -> str:
        """/video <query> [platform] | /video download <url> [audio] | platforms."""
        from ..media.video import _PLATFORM_SITES, VideoFinder

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /video <query> [platform]  |  "
                    "/video download <url> [audio]  |  /video platforms")
        words = tail.split()
        if words[0].lower() == "platforms":
            return "platforms: " + ", ".join(sorted(_PLATFORM_SITES))
        if words[0].lower() == "download":
            rest = words[1:]
            audio = any(w.lower() == "audio" for w in rest)
            url = " ".join(w for w in rest if w.lower() != "audio").strip()
            if not url:
                return "usage: /video download <url> [audio]"
            try:
                out = VideoFinder(self.context).download(url,
                                                         audio_only=audio)
                return f"⬇ downloaded: {out.get('path', out)}"
            except Exception as exc:  # noqa: BLE001
                return f"download failed: {exc}"
        query = tail
        platform = ""
        if words[-1].lower() in _PLATFORM_SITES and len(words) > 1:
            platform = words[-1].lower()
            query = " ".join(words[:-1])
        try:
            out = VideoFinder(self.context).find(query, max_results=8,
                                                 platform=platform)
        except Exception as exc:  # noqa: BLE001
            return f"video error: {exc}"
        if not out["results"]:
            return (f"no video results for “{query}”"
                    + (f" on {platform}" if platform else "")
                    + " — try a broader query or another platform.")
        lines = [f"🎬 {out['count']} result(s) for “{query}”:"]
        for i, r in enumerate(out["results"][:6], 1):
            title = (r.get("title") or r["url"])[:64]
            dur = f"  [{r['duration']}]" if r.get("duration") else ""
            extra = f" · {r['author']}" if r.get("author") else ""
            lines.append(f"  {i}. {title}{dur}{extra}\n     {r['url']}")
        lines.append("download: /video download <url>")
        return "\n".join(lines)

    def _control_exec(self, tail: str) -> str:
        """/exec <code> | /exec until_green <code|file|project> [lang] | /exec languages."""
        from ..execbox import CodeRunner

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /exec <code> [lang]  |  "
                    "/exec until_green <code|file|project>  |  /exec languages")
        if tail.lower() == "languages":
            box = CodeRunner(self.context)
            langs = box.languages()
            avail = ", ".join(l.name for l in langs.values() if l.available)
            return f"installed: {avail}\nmissing: " \
                   f"{', '.join(l.name for l in langs.values() if not l.available) or '(none)'}"
        if tail.split()[0].lower() == "until_green" and len(tail.split()) > 1:
            target = " ".join(tail.split()[1:])
            try:
                out = CodeRunner(self.context).run_project_until_green(target)
            except Exception as exc:  # noqa: BLE001
                return f"until_green error: {exc}"
            mark = "🟢 GREEN" if out["green"] else "🔴 RED"
            text = (f"{mark} — {out['target']}\n"
                    f"  criterion: {out['criterion']}  "
                    f"({out['rounds']}/{out['max_rounds']} rounds)")
            if out.get("files_changed"):
                text += ("\n  files rewritten by model: "
                         + ", ".join(out["files_changed"][:6]))
            fr = out.get("final_run") or {}
            if fr.get("stderr"):
                text += "\n(stderr tail)\n" + fr["stderr"][-600:]
            elif fr.get("stdout"):
                text += "\n(stdout tail)\n" + fr["stdout"][-400:]
            return text
        code = tail
        lang = ""
        words = tail.split()
        known = set(CodeRunner(self.context).languages())
        if len(words) > 1 and words[-1].lower() in known:
            lang = words[-1].lower()
            code = " ".join(words[:-1])
        try:
            out = CodeRunner(self.context).run(code, lang=lang)
        except Exception as exc:  # noqa: BLE001
            return f"exec error: {exc}"
        mark = "✅" if out["ok"] else "❌"
        text = (f"{mark} [{out['language']}] exit={out['exit_code']} "
                f"{out['seconds']}s" + ("  ⏱ timed out" if out["timed_out"] else ""))
        if out["stdout"]:
            text += "\n" + out["stdout"].rstrip()
        if out["stderr"]:
            text += "\n(stderr)\n" + out["stderr"].rstrip()
        if out["files"]:
            text += "\nwrote: " + ", ".join(f["path"] for f in out["files"][:8])
        return text

    def _control_zip(self, tail: str) -> str:
        """/zip <paths…> --dest x.zip | /zip list|info|extract <archive> | compress <file> [fmt]."""
        from ..archives import Archivist

        tail = (tail or "").strip()
        if not tail:
            return ("usage: /zip <paths…> --dest x.zip [--fmt tar.gz]  |  "
                    "/zip list|info|extract <archive>  |  /zip compress <file> [fmt]")
        tokens = tail.split()
        dest = ""
        fmt = "zip"
        if "--dest" in tokens:
            i = tokens.index("--dest")
            if i + 1 < len(tokens):
                dest = tokens[i + 1]
            tokens = tokens[:i] + tokens[i + 2:]
        if "--fmt" in tokens:
            i = tokens.index("--fmt")
            if i + 1 < len(tokens):
                fmt = tokens[i + 1]
            tokens = tokens[:i] + tokens[i + 2:]
        try:
            a = Archivist(self.context)
            if tokens and tokens[0].lower() in ("list", "info", "extract",
                                                 "digest", "compress") and len(tokens) > 1:
                action, path = tokens[0].lower(), tokens[1]
                if action == "info":
                    out = a.info(path)
                    return (f"{out['path']} → {out['format']} · "
                            f"{out.get('entries', '?')} entries · "
                            f"{out['compressed_bytes']} B"
                            + (f" · {out['note']}" if "note" in out else ""))
                if action == "list":
                    out = a.list(path)
                    return (f"{out['format']}, {out['count']} entries:\n"
                            + "\n".join(f"  {e['name']} ({e['size']} B)"
                                        for e in out["entries"][:30]))
                if action == "extract":
                    out = a.extract(path)
                    return (f"extracted {out['extracted']} file(s) → {out['dest']}"
                            + (f"\nskipped unsafe: {', '.join(out['skipped'][:5])}"
                               if out["skipped"] else ""))
                if action == "digest":
                    from ..tools.filesystem import safe_path

                    rp = safe_path(self.context, path, must_exist=True)
                    if rp.is_dir():
                        out = a.digest_directory(str(rp))
                    else:
                        out = a.digest(str(rp))
                    kg = out.get("kg", {})
                    lines = [f"📚 {out['path']} → {out['format']}: "
                             f"{out['text_files']} text file(s), {out['chars']:,} chars"]
                    lines.append(f"  knowledge graph: +{kg.get('added_nodes', 0)} "
                                 f"nodes, +{kg.get('added_links', 0)} links")
                    lines.append(f"  memory: {'episode stored' if out.get('memory_stored') else 'not stored'}")
                    if out.get("extracted_to"):
                        lines.append(f"  extracted to: {out['extracted_to']}")
                    if out.get("kg_error"):
                        lines.append(f"  kg error: {out['kg_error']}")
                    return "\n".join(lines)
                if action == "compress":
                    f2 = tokens[2] if len(tokens) > 2 else "gz"
                    out = a.compress(path, fmt=f2)
                    return f"compressed → {out['path']} ({out['compressed_bytes']} B)"
            if not dest:
                return ("usage: /zip <paths…> --dest x.zip [--fmt zip|tar.gz]  |  "
                        "/zip digest <archive>")
            out = a.create(tokens, dest, fmt=fmt)
            return (f"📦 created {out['path']} — {out['entries']} entries, "
                    f"{out['compressed_bytes']} B")
        except Exception as exc:  # noqa: BLE001
            return f"zip error: {exc}"

    def _control_apps(self, tail: str) -> str:
        """/apps [list|stacks|build <name> --stack …|info <name>]."""
        from ..builders import AppBuilder, STACKS

        tail = (tail or "").strip()
        tokens = tail.split() if tail else []
        action = tokens[0].lower() if tokens and tokens[0].lower() in (
            "list", "stacks", "build", "info", "serve", "stop", "served",
            "deploy", "stop_deploy", "deployed") else "list"
        rest = tokens[1:] if action != "list" else tokens
        def _port_flag():
            if "--port" in rest:
                i = rest.index("--port")
                if i + 1 < len(rest):
                    try:
                        return int(rest[i + 1])
                    except ValueError:
                        return 0
            return 0
        try:
            b = AppBuilder(self.context)
            if action == "served":
                out = b.served()
                entries = out.get("served") or []
                if not entries:
                    return "no apps currently serving"
                return (f"serving ({out['count']}):\n" + "\n".join(
                    f"  {e['app']}  {e['url']}  (pid {e['pid']}, "
                    f"alive={e['alive']})" for e in entries))
            if action == "serve":
                if not rest:
                    return "usage: /apps serve <name> [--port N]"
                out = b.serve(rest[0], port=_port_flag())
                health = out.get("health") or {}
                text = (f"🖥 serving {out['app']} ({out['stack']}) → "
                        f"{out['url']}  (pid {out.get('pid')}, "
                        f"health={'ok' if health.get('ok') else health.get('error', 'unknown')})")
                if out.get("note"):
                    text += f"\n{out['note']}"
                return text
            if action == "stop":
                if not rest:
                    return "usage: /apps stop <name>"
                out = b.stop(rest[0])
                return (f"stopped {out['app']} (was pid {out['was_pid']}) "
                        f"→ {'stopped' if out['stopped'] else 'already dead'}")
            if action == "stacks":
                return "stacks: " + ", ".join(STACKS)
            if action == "info":
                if not rest:
                    return "usage: /apps info <name>"
                out = b.info(rest[0])
                return (f"{out.get('name')} ({out.get('stack')}) · "
                        f"{len(out.get('files', []))} files\nrun: {out.get('run')}")
            if action == "build":
                name = " ".join(
                    w for w in rest
                    if not w.startswith("--")).strip()
                def _flag(flag, default=""):
                    if flag in rest:
                        i = rest.index(flag)
                        if i + 1 < len(rest):
                            return rest[i + 1]
                    return default
                if not name or name.startswith("--"):
                    return ("usage: /apps build <name> --stack "
                            "static|flask|fastapi|express|react-vite|cli-python|"
                            "django|nextjs|bot-telegram|go-cli "
                            "[--title T] [--features a,b,c]")
                stack = _flag("--stack", "static")
                feats = [f.strip() for f in _flag("--features", "").split(",")
                         if f.strip()]
                out = b.build({"name": name, "stack": stack,
                               "title": _flag("--title", name),
                               "features": feats,
                               "overwrite": True})
                v = out["validation"]
                return (f"🏗 built {out['app']} ({out['stack']}) → {out['dir']}\n"
                        f"files: {len(out['files'])} · validation: "
                        f"{'OK' if v['ok'] else 'FAILED ' + str(v['failed'])}\n"
                        f"run: {out['run']}")
            if action == "deployed":
                out = b.deployed()
                entries = out.get("deployments") or []
                if not entries:
                    return "no apps currently deployed"
                return (f"deployed ({out['count']}):\n" + "\n".join(
                    f"  {e['app']}  {e['url']}  (proxy pid {e['pid']}, "
                    f"→ :{e.get('backend_port')}, alive={e['alive']})"
                    for e in entries))
            if action == "deploy":
                def _flag(flag, default=""):
                    if flag in rest:
                        i = rest.index(flag)
                        if i + 1 < len(rest):
                            return rest[i + 1]
                    return default
                name = " ".join(
                    w for w in rest
                    if not w.startswith("--")).strip()
                if not name:
                    return ("usage: /apps deploy <name> [--domain d] "
                            "[--path /x] [--port N]")
                out = b.deploy(name, host=_flag("--host", "0.0.0.0"),
                               port=int(_flag("--port", "0") or 0),
                               domain=_flag("--domain"),
                               path=_flag("--path"))
                health = out.get("health") or {}
                text = (f"🌐 deployed {out['app']} → {out['url']}\n"
                        f"  real reverse proxy (pid {out.get('pid')}) "
                        f"→ 127.0.0.1:{out.get('backend_port')}\n"
                        f"  domain: {out.get('domain') or '(none)'}   "
                        f"path: {out.get('path') or '/'}   "
                        f"health={'ok' if health.get('ok') else health.get('error', 'unknown')}")
                if out.get("note"):
                    text += f"\n{out['note']}"
                return text
            if action == "stop_deploy":
                if not rest:
                    return "usage: /apps stop_deploy <name>"
                out = b.stop_deploy(rest[0])
                return (f"stopped proxy for {out['app']} "
                        f"({'stopped' if out['stopped'] else 'already dead'}) "
                        f"was {out.get('url', '')}")
            out = b.list_apps()
            if not out["apps"]:
                return ("no apps yet — /apps build myapp --stack flask "
                        "(stacks: " + ", ".join(STACKS) + ")")
            return (f"apps ({out['count']}):\n" + "\n".join(
                f"  {a['name']}  [{a['stack']}]  →  {a['run']}"
                for a in out["apps"]))
        except Exception as exc:  # noqa: BLE001
            return f"apps error: {exc}"

    # ── wave 73: media hub / podcast / CI fix ──────────────────────────────
    def _control_hub(self, tail: str, chat_key: str = "") -> str:
        """/hub song <topic…> [style] | video <query…> [platform] |
        podcast <query…> | status — the one-call media orchestrator."""
        from ..media import MediaHub

        tail = (tail or "").strip()
        tokens = tail.split() if tail else []
        if tokens and tokens[0].lower() in ("song", "video", "podcast",
                                            "status"):
            action = tokens[0].lower()
            rest = " ".join(tokens[1:]).strip()
        else:
            action = "song"
            rest = tail
        try:
            hub = MediaHub(self.context)
            if action == "status":
                out = hub.status()
                backend = out.get("backend") or {}
                bname = (backend.get("name")
                         if isinstance(backend, dict) else str(backend))
                lines = [f"player: {bname or 'unknown'}  "
                         f"playing={out.get('playing')}"]
                if out.get("path"):
                    lines.append(f"  path: {out['path']}")
                if out.get("current"):
                    lines.append(f"  current: {out['current']}")
                return "\n".join(lines)
            if not rest:
                return (f"usage: /hub {action} <topic…>  |  /hub video "
                        "<query…> [platform]  |  /hub podcast <query…>  |  "
                        "/hub status")
            send_to = (chat_key.split(":", 1)[0] if ":" in chat_key else "",
                       chat_key.split(":", 1)[1] if ":" in chat_key else "")
            if action == "song":
                words = rest.split()
                style = words[-1] if len(words) > 1 else "pop"
                if len(words) > 1:
                    topic = " ".join(words[:-1])
                else:
                    topic, style = words[0], "pop"
                out = hub.run("song", topic=topic, style=style)
                song = out.get("song", {})
                lines = [f"🎵 {song.get('title', topic)}  "
                         f"({song.get('style', style)})"]
                if song.get("midi_path"):
                    lines.append(f"  midi: {song['midi_path']}")
                pb = out.get("playback")
                if pb:
                    lines.append(f"  playback: {pb.get('status', pb)}")
                return "\n".join(lines)
            if action == "video":
                words = rest.split()
                platform = ""
                if len(words) > 1 and words[-1].lower() in (
                        "youtube", "vimeo", "tiktok", "dailymotion",
                        "twitch", "rumble"):
                    platform = words[-1].lower()
                    query = " ".join(words[:-1])
                else:
                    query = rest
                out = hub.run("video", query=query, platform=platform)
                pick = out.get("pick") or {}
                fc = (out.get("found_count")
                      or (out.get("found") or {}).get("count", "?"))
                lines = [f"🎬 found {fc}, picked "
                         f"“{(pick.get('title') or pick.get('url') or query)[:70]}”"]
                if out.get("download", {}).get("path"):
                    lines.append(f"  file: {out['download']['path']}")
                if out.get("playback"):
                    lines.append(f"  playback: "
                                 f"{out['playback'].get('status', out['playback'])}")
                if out.get("error"):
                    lines.append(f"  {out['error']}")
                return "\n".join(lines)
            # podcast — transcript auto-sends to your newest live chat on
            # any connected platform (send_transcript=None = auto)
            out = hub.run("podcast", query=rest, send_transcript=None,
                          send_to=send_to)
            pick = out.get("pick") or {}
            fc = (out.get("found_count")
                  or (out.get("found") or {}).get("count", "?"))
            lines = [f"🎙 {rest}: found {fc}, picked "
                     f"“{(pick.get('title') or pick.get('url') or rest)[:70]}”"]
            if out.get("download", {}).get("path"):
                lines.append(f"  file: {out['download']['path']}")
            if out.get("transcript"):
                lines.append(f"  transcript: {len(out['transcript'])} chars "
                             f"(stt: {out.get('stt_provider', 'n/a')})")
            if out.get("summary"):
                lines.append(f"  summary: {out['summary'][:160]}")
            if out.get("chapters"):
                lines.append(f"  chapters: {len(out['chapters'])}")
                for ch in out["chapters"][:8]:
                    lines.append(f"    [{ch['start']}-{ch['end']}] {ch['title']}")
            if out.get("transcript_path"):
                lines.append(f"  saved: {out['transcript_path']}")
            s = out.get("send")
            if isinstance(s, dict):
                lines.append(f"  transcript → {s.get('platform')}:"
                             f"{s.get('chat')} "
                             f"({'sent' if s.get('ok') else 'FAILED: ' + str(s.get('error'))})")
            elif s:
                lines.append(f"  transcript: {s}")
            if out.get("stt_error"):
                lines.append(f"  stt: {out['stt_error']}")
            if out.get("note"):
                lines.append(f"  note: {out['note']}")
            if out.get("error"):
                lines.append(f"  {out['error']}")
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            return f"hub error: {exc}"

    def _control_podcast(self, tail: str, chat_key: str = "") -> str:
        """/podcast <query…> [platform] — the podcast pipeline, one call."""
        return self._control_hub(f"podcast {tail}", chat_key=chat_key)

    def _control_fix(self, tail: str) -> str:
        """/fix <code> [lang] [--rounds N] — CI loop until green."""
        from ..execbox import CodeRunner

        tail = (tail or "").strip()
        if not tail:
            return "usage: /fix <code> [lang] [--rounds N]"
        tokens = tail.split()
        rounds = 4
        if "--rounds" in tokens:
            i = tokens.index("--rounds")
            if i + 1 < len(tokens):
                try:
                    rounds = int(tokens[i + 1])
                except ValueError:
                    rounds = 4
            tokens = tokens[:i] + tokens[i + 2:]
        known = set(CodeRunner(self.context).languages())
        lang = ""
        if len(tokens) > 1 and tokens[-1].lower() in known:
            lang = tokens[-1].lower()
            tokens = tokens[:-1]
        code = " ".join(tokens)
        try:
            out = CodeRunner(self.context).run_until_green(
                code, lang=lang, max_rounds=rounds)
        except Exception as exc:  # noqa: BLE001
            return f"fix error: {exc}"
        mark = "🟢 GREEN" if out["green"] else "🔴 RED"
        text = f"{mark} after {out['rounds']}/{out['max_rounds']} rounds"
        for h in out["history"]:
            line = (f"  {'✓' if h['ok'] else '✗'} round {h['round']}: "
                    f"exit={h['exit_code']} in {h['seconds']}s")
            if not h["ok"] and h.get("fix_note"):
                line += f"  → {h['fix_note']}"
            if not h["ok"] and not h.get("fix_note") and h.get("stderr_tail"):
                last = h["stderr_tail"].strip().splitlines()
                if last:
                    line += f"\n      {last[-1][:120]}"
            text += line + "\n"
        fr = out["final_run"] or {}
        if fr.get("stdout"):
            text += "--- stdout ---\n" + fr["stdout"].rstrip()
        if fr.get("stderr"):
            text += "--- stderr ---\n" + fr["stderr"].rstrip()
        if out["note"]:
            text += f"\n{out['note']}"
        return text

    # ── feature flags ────────────────────────────────────────────────────────
    def _control_features(self, tail: str) -> str:
        from .features import FeatureRegistry

        reg = FeatureRegistry(self.context.db)
        if not tail:
            lines = ["features — toggle with /features <name> on|off:"]
            for f in reg.list():
                lines.append(f"  {'✓' if f['on'] else '✗'} {f['name']} — {f['description']}")
            return "\n".join(lines)
        parts = tail.split()
        if len(parts) != 2 or parts[1].lower() not in {"on", "off"}:
            names = ", ".join(f["name"] for f in reg.list())
            return f"usage: /features <name> on|off — names: {names}"
        name = parts[0].strip().lower()
        on = parts[1].lower() == "on"
        if not reg.set(name, on):
            names = ", ".join(f["name"] for f in reg.list())
            return f"unknown feature {name!r}. available: {names}"
        return f"{name}: {'on' if on else 'off'}"

    # ── arena ────────────────────────────────────────────────────────────────
    def _control_arena(self, tail: str, chat_key: str) -> str:
        from .arena import Arena
        from .features import feature_enabled

        # Reuse the runtime's arena (loop state lives per instance).
        arena = getattr(self, "_arena", None) or Arena(self.context)
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        if verb == "stats":
            s = arena.stats()
            builds = ", ".join(f"{k}={v}" for k, v in s["builds"].items())
            last = time.strftime("%m-%d %H:%M", time.localtime(s["last_cycle"])) if s["last_cycle"] else "never"
            interval = arena.interval_hours()
            interval_txt = f"{interval:g}h" if interval else "config default"
            return (f"arena stats — cycles: {s['cycles']} · knowledge rows: {s['knowledge']} · "
                    f"builds: {builds or '0'}\n"
                    f"last cycle: {last} · loop interval: {interval_txt}")
        if verb == "digest":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            rows = arena.digests(n)
            if not rows:
                return "no digests yet — run a cycle first (/arena run)"
            lines = []
            for r in rows:
                when = time.strftime("%m-%d %H:%M", time.localtime(r.get("created_at", 0)))
                lines.append(f"── {when} · {r['topic']} ({r.get('category', '?')})\n{str(r.get('digest', ''))[:900]}")
            text = "\n\n".join(lines)
            if len(text) > 1800:
                chat = self._ref_from_key(chat_key)
                return self._send_long_checked(chat.platform, chat, text)
            return text
        if verb == "schedule":
            if len(parts) > 1 and parts[1].replace(".", "", 1).isdigit():
                hours = float(parts[1])
                if not arena.set_interval(hours):
                    return "could not save the interval"
                return f"loop interval set to {hours:g}h (live — the running loop picks it up next cycle)"
            live = arena.interval_hours()
            live_txt = f"{live:g}h" if live else "config default"
            return f"loop interval: {live_txt} — /arena schedule <hours> to change"
        if verb == "loop":
            mode = parts[1].lower() if len(parts) > 1 else "status"
            if mode == "on":
                if not feature_enabled(self.context, "arena"):
                    return "loop blocked: feature off — /features arena on first"
                from .power import power_mode_for
                if not power_mode_for(self.context).active:
                    return "loop blocked: power mode is locked (the loop is a power-mode capability)"
                if arena.start_loop():
                    persistent = "" if getattr(self, "_arena", None) is arena else (
                        "\n(note: this session only — add NM_ARENA_ENABLED=1 to ~/.nomorals/.env "
                        "to auto-start it on every boot)"
                    )
                    live = arena.interval_hours()
                    live_txt = f"{live:g}h" if live else "config default"
                    return (f"background loop running — interval {live_txt}, "
                            f"it researches + digests on its own{persistent}")
                return "loop is already running"
            if mode == "off":
                arena.stop_loop()
                return "background loop stopped"
            state = "running" if arena.loop_running() else "idle"
            live = arena.interval_hours()
            live_txt = f"{live:g}h" if live else "config default"
            return f"loop: {state} (interval {live_txt})"
        if verb == "topic":
            sub = parts[1].lower() if len(parts) > 1 else "list"
            if sub == "add":
                text = " ".join(parts[2:]).strip()
                if not text:
                    return "usage: /arena topic add <topic>"
                return arena.add_topic(text)
            custom = arena.custom_topics()
            if not custom:
                return "topic bank is empty — /arena topic add <topic> to add one"
            return f"topic bank ({len(custom)}): " + " · ".join(custom[:15])
        if verb == "builds":
            rows = arena.all_builds(20)
            if not rows:
                return "no arena builds yet (power mode + /features arena on + arena.build config)"
            lines = ["arena builds:"]
            for r in rows:
                when = time.strftime("%m-%d %H:%M", time.localtime(r.get("created_at", 0)))
                lines.append(f"  {r['id']} [{r['status']}] {r.get('name', '?')} — {str(r.get('purpose', ''))[:60]} ({when})")
            return "\n".join(lines)
        if verb == "status":
            pending = arena.builds()
            stream = arena.stream(1)
            loop = "loop: running" if arena.loop_running() else "loop: idle"
            lines = [f"arena — {loop}"]
            lines.append(f"feature: {'on' if feature_enabled(self.context, 'arena') else 'off'} "
                         f"(/features arena on|off)")
            lines.append(f"knowledge rows: {self._arena_knowledge_count()}")
            if pending:
                lines.append("pending builds:")
                for row in pending[:5]:
                    lines.append(f"  {row['id']} — {row.get('name')} ({row.get('purpose', '')[:50]})")
            else:
                lines.append("pending builds: none")
            if stream:
                last = stream[0]
                lines.append(f"last event: {last['kind']} {str(last.get('payload', {}).get('topic', ''))[:50]}")
            return "\n".join(lines)
        if verb == "run":
            if not feature_enabled(self.context, "arena"):
                return "arena is off. /features arena on"
            topic = " ".join(parts[1:]).strip() or None
            chat = self._ref_from_key(chat_key)

            def _notify(text: str) -> None:
                try:
                    self.gateway.send(chat.platform, chat, text)
                except Exception:  # noqa: BLE001
                    pass

            try:
                self.gateway.send(chat.platform, chat,
                                  f"⏳ arena cycle starting{': ' + topic[:60] if topic else ''}…")
            except Exception:  # noqa: BLE001
                pass
            result = arena.run_cycle(topic=topic, notify=_notify)
            if not result.get("ok"):
                return f"arena cycle failed: {result.get('error')}"
            build = result.get("build")
            build_line = (f"build {build['id']} sent for review." if build
                          else "no build this cycle (power mode + arena.build required).")
            return (f"arena cycle done: “{result['topic']}” — digested into knowledge "
                    f"in {result.get('seconds', 0)}s. {build_line}")
        if verb == "topics":
            from .arena.topics import category_table, topics_table

            if tail and tail.strip().lower() not in {"", "all"}:
                return category_table(tail.strip().lower()[:40])
            try:
                profile = arena.interest_profile()
            except Exception:  # noqa: BLE001
                profile = None
            return topics_table(profile=profile, db=self.context.db)
        if verb == "surprise":
            if not feature_enabled(self.context, "arena"):
                return "arena is off. /features arena on"
            seed = None
            if tail and tail.strip().lstrip("-").isdigit():
                seed = int(tail.strip())
            chat = self._ref_from_key(chat_key)

            def _notify2(text: str) -> None:
                try:
                    self.gateway.send(chat.platform, chat, text)
                except Exception:  # noqa: BLE001
                    pass

            result = arena.run_cycle(notify=_notify2, surprise=True, seed=seed)
            if not result.get("ok"):
                return f"surprise cycle failed: {result.get('error')}"
            return (f"🎲 surprise cycle done: “{result['topic']}” "
                    f"[{result.get('category')}] — digested into knowledge "
                    f"in {result.get('seconds', 0)}s.")
        if verb == "stream":
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
            kind_filter = parts[1].lower() if len(parts) > 1 and not parts[1].isdigit() else None
            events = arena.stream(limit if not kind_filter else limit * 10)
            if kind_filter:
                events = [e for e in events if e.get("kind") == kind_filter][:limit]
            if not events:
                return "arena stream is empty — run a cycle first (/arena run)."
            lines = [f"arena stream (last {len(events)}):"]
            for ev in events:
                payload = ev.get("payload") or {}
                detail = (payload.get("topic") or payload.get("name")
                          or payload.get("error") or ev.get("kind", ""))
                when = time.strftime("%m-%d %H:%M", time.localtime(ev.get("ts", 0)))
                lines.append(f"  {when}  {ev.get('kind', '?'):<8} {str(detail)[:70]}")
            return "\n".join(lines)
        if verb == "export":
            limit = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 200
            text = arena.export(limit)
            if not text.strip():
                return "arena knowledge is empty — run some cycles first."
            try:
                path = self.context.settings.home_path / f"arena_export_{int(time.time())}.jsonl"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text + "\n", "utf-8")
            except Exception as exc:  # noqa: BLE001
                return f"export failed: {exc}"
            return (f"exported {len(text.splitlines())} training rows → {path}\n"
                    "that's the live stream, in the export format — ready for the Colab training run.")
        if verb == "ship":
            from .arena.ship import ship_queue

            try:
                queue = ship_queue(self.context.db, self.context)
            except Exception as exc:  # noqa: BLE001
                return f"ship queue failed: {exc}"
            if not queue:
                return ("ship queue is empty — promote a build first "
                        "(/arena promote <build-id>)")
            lines = ["arena ship queue:"]
            for q in queue:
                files = q["edits"]
                shown = ", ".join(files[:4])
                if len(files) > 4:
                    shown += f" (+{len(files) - 4} more)"
                lines.append(f"  {q['id']} [{q['status']}] "
                             f"{q['instruction'][:60]} :: {shown}")
            return "\n".join(lines)
        if verb == "promote":
            if len(parts) < 2:
                return "usage: /arena promote <build-id>"
            from .arena.ship import promote_build
            from .evolution import EvolutionAgent, _REPO_ROOT

            try:
                pid = promote_build(self.context.db, self.context, parts[1],
                                    _REPO_ROOT)
                prop = EvolutionAgent(self.context)._load(pid)
                files = [e["path"] for e in (prop.edits if prop else [])
                         if isinstance(e, dict)]
            except Exception as exc:  # noqa: BLE001
                return f"promote failed: {exc}"
            shown = ", ".join(files[:6])
            if len(files) > 6:
                shown += f" (+{len(files) - 6} more)"
            return (f"promoted build {parts[1]} → proposal {pid} "
                    f"({len(files)} files: {shown})\n"
                    f"verify with /arena apply {pid}")
        if verb == "apply":
            if len(parts) < 2:
                return "usage: /arena apply <proposal-id>"
            from .arena.ship import approve_ship
            from .evolution import _REPO_ROOT

            try:
                result = approve_ship(self.context.db, self.context, parts[1],
                                      _REPO_ROOT, commit=True, full_suite=False)
            except Exception as exc:  # noqa: BLE001
                return f"apply failed: {exc}"
            if result.get("applied"):
                files = ", ".join(result.get("edits", []))
                return (f"shipped {result['proposal']} → {files} "
                        f"(commit {result.get('commit', 'n/a')})")
            return (f"not applied: {result.get('reason', 'unknown')} — "
                    f"{str(result.get('report', ''))[:400]}")
        if verb == "reject":
            if len(parts) < 3:
                return "usage: /arena reject <proposal-id> <reason>"
            from .arena.ship import deny_ship

            if deny_ship(self.context.db, self.context, parts[1],
                         " ".join(parts[2:])):
                return f"rejected {parts[1]}"
            return f"no proposal {parts[1]!r}"
        if verb == "scores":
            try:
                from .arena import scoring
            except ImportError:
                return ("arena scoring isn't available — the arena_scores "
                        "table exists (migration 61) but no scoring module "
                        "could be imported; nothing to show")
            try:
                table = scoring.category_scores(self.context.db)
            except Exception as exc:  # noqa: BLE001
                return f"scores failed: {exc}"
            if not table:
                return ("no arena scores recorded yet — the scoring worker "
                        "logs to arena_scores after each build")
            lines = ["arena category scores:"]
            for cat in sorted(table):
                s = table[cat]
                avg = (f"{s['avg']:.2f}" if s.get("avg") is not None
                       else "n/a")
                lat = (f"{s['avg_latency']:.1f}s"
                       if s.get("avg_latency") is not None else "n/a")
                lines.append(f"  {cat:<16} runs={s['runs']:<4} avg={avg:<6} "
                             f"latency={lat}")
            return "\n".join(lines)
        if verb == "sample":
            from .arena.sampling import sample_challenge

            try:
                profile = arena.interest_profile()
            except Exception:  # noqa: BLE001
                profile = None
            # db=None: a dry run — sample_challenge never touches the
            # database then, so no anti-repeat state is written and the
            # topic bank stays untouched.
            cat, topic, entry = sample_challenge(db=None, profile=profile,
                                                 anti_repeat=0)
            return (f"arena sample (preview — nothing recorded):\n"
                    f"  category:   {cat}\n"
                    f"  difficulty: {entry.get('d', '?')}\n"
                    f"  kind:       {entry.get('kind', '?')}\n"
                    f"  topic:      {topic}\n"
                    f"  verify:     {str(entry.get('verify', ''))[:120]}")
        if verb in {"approve", "deny"}:
            if len(parts) < 2:
                return f"usage: /arena {verb} <build-id>"
            return arena.approve(parts[1]) if verb == "approve" else arena.deny(parts[1])
        # An unknown verb is a topic: /arena hacking == /arena run hacking.
        if (tail or "").strip():
            return self._control_arena(f"run {tail.strip()}", chat_key=chat_key)
        return ("usage: /arena [status|run [topic]|surprise [seed]|stats|digest [n]|schedule [h]|loop on|off|"
                "topic add <t>|builds|topics [category]|stream [kind|n]|export [n]|approve <id>|deny <id>|"
                "promote <build-id>|ship|apply <proposal-id>|reject <proposal-id> <reason>|scores|sample]")

    def _arena_knowledge_count(self) -> int:
        try:
            row = self.context.db.query_one("SELECT COUNT(*) AS n FROM arena_knowledge")
            return int(row.get("n", 0)) if row else 0
        except Exception:  # noqa: BLE001
            return 0

    # ── trial accounts ───────────────────────────────────────────────────────
    def _control_trial(self, tail: str, chat_key: str) -> str:
        from .trial import TrialFlow

        flow = TrialFlow(self.context)
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "list"
        if verb == "list":
            return flow.list()
        if verb == "start":
            if len(parts) < 2:
                return "usage: /trial start <platform>"
            return flow.start(" ".join(parts[1:]))
        if verb == "save":
            if len(parts) < 4:
                return "usage: /trial save <platform> <login> <password>"
            platform, login = parts[1], parts[2]
            secret = " ".join(parts[3:])
            flow.save(platform, login, secret)
            try:
                sent = flow.deliver(platform, gateway=self.gateway, via=chat_key)
            except Exception as exc:  # noqa: BLE001
                sent = f"stored, but delivery failed: {exc}"
            return f"stored the {platform.lower()} trial account (encrypted, local only). {sent}"
        if verb == "send":
            if len(parts) < 2:
                return "usage: /trial send <platform>"
            return flow.deliver(parts[1], gateway=self.gateway, via=chat_key)
        if verb == "rm":
            if len(parts) < 2:
                return "usage: /trial rm <platform>"
            return flow.remove(parts[1])
        return "usage: /trial [list|start <p>|save <p> <login> <pass>|send <p>|rm <p>]"

    # ── games ────────────────────────────────────────────────────────────────
    def _game_engine(self) -> Any:
        """The multi-player GameEngine — ONE instance serves every platform
        at once. It never touches a platform: it sees chat keys, sender
        keys and a send callback, and the gateway feeds it inbound text."""
        engine = getattr(self, "_game_engine_obj", None)
        if engine is None:
            from ..games.engine import GameEngine

            engine = GameEngine(
                self.context,
                send=self._game_send,
                suggest=self._game_suggest(),
            )
            self._game_engine_obj = engine
        return engine

    def _game_relay(self) -> Any:
        """The game relay system — one per engine, so the ticker reaps
        expired invites and every platform shares the same relay table."""
        return self._game_engine().relay

    def _relay_send(self, chat_key: str, text: str) -> bool:
        """Send to a relay chat on its own platform. Chat keys are
        ``platform:chat_id`` — never assume telegram."""
        platform = (chat_key or "").partition(":")[0].strip() or "telegram"
        try:
            result = self.gateway.send(platform, chat_key, text)
            return bool(getattr(result, "ok", False))
        except Exception:  # noqa: BLE001
            _log.warning("relay send failed for %s", chat_key,
                         exc_info=True)
            return False

    @staticmethod
    def _resolve_invite_target(inviter_chat_key: str,
                               to_label: str) -> str | None:
        """Best-effort chat key for a direct invite DM on the inviter's
        own platform. Returns None when the label can't be resolved
        (then the invite code is the delivery mechanism)."""
        label = (to_label or "").strip()
        if not label:
            return None
        platform = (inviter_chat_key or "").partition(":")[0].strip()
        if platform == "telegram":
            return f"telegram:{label.lstrip('@')}"
        if platform == "whatsapp":
            digits = "".join(c for c in label if c.isdigit())
            # a phone number → DM chat; the adapter normalizes the JID
            if len(digits) >= 7:
                return f"whatsapp:{digits}"
            return None
        if platform == "discord":
            digits = "".join(c for c in label if c.isdigit())
            if len(digits) >= 5:
                return f"discord:{digits}"
            return None
        return None

    def _game_suggest(self) -> Callable[[str], str] | None:
        """Optional LLM brain for the game AI — the same router the legacy
        bridge used. Without one the AI plays seeded deterministic moves,
        which is fully functional on its own."""
        router = getattr(self.context, "router", None)
        if router is None:
            return None

        def _call(prompt: str) -> str:
            from ..llm.base import Message

            response = router.chat([Message(role="user", content=prompt)])
            return getattr(response, "text", "") or ""

        return _call

    def _game_send(self, chat_key: str, text: str) -> None:
        """Engine output → the same chat it came from, on any platform."""
        ref = ChatRef.parse(chat_key)
        for chunk in self._game_chunks(text):
            try:
                result = self.gateway.send(ref.platform, ref, chunk)
                if not result.ok:
                    _log.warning("game send failed in %s: %s",
                                 chat_key, result.error)
            except Exception:  # noqa: BLE001
                _log.exception("game send failed in %s", chat_key)

    @staticmethod
    def _game_chunks(text: str, limit: int = 3800) -> list[str]:
        """Split long game reports at newlines for platform limits."""
        text = text or ""
        if not text:
            return []
        if len(text) <= limit:
            return [text]
        chunks: list[str] = []
        while len(text) > limit:
            cut = text.rfind("\n", 0, limit)
            if cut < limit // 2:
                cut = limit
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        if text:
            chunks.append(text)
        return chunks

    @staticmethod
    def _game_player(message: Any) -> Any:
        from ..games.players import Player

        sender = (message.sender or "").strip() or "unknown"
        return Player.from_sender(message.chat.platform, sender, sender)

    @staticmethod
    def _game_player_for_key(chat_key: str) -> Any:
        """A stable player for console-style calls that only carry a key."""
        from ..games.players import Player

        try:
            ref = ChatRef.parse(chat_key)
        except Exception:  # noqa: BLE001
            ref = ChatRef(platform="local", chat_id="console")
        return Player.from_sender(ref.platform, ref.chat_id, ref.chat_id[:40])

    @staticmethod
    def _profile_line(prof: Any) -> str:
        items = ", ".join(f"{k}×{v}" for k, v in (prof.items or {}).items()) or "none"
        games = prof.games_played or 0
        pct = round(100 * prof.wins / games) if games else 0
        if prof.streak > 0:
            streak = f"win streak {prof.streak}"
        elif prof.streak < 0:
            streak = f"loss streak {-prof.streak}"
        else:
            streak = "no streak"
        return (f"📊 {prof.name or prof.key} — {prof.wins}W {prof.losses}L "
                f"{prof.draws}D · {pct}% · {streak}\n"
                f"   {games} games · {prof.coins} coins · {prof.points} points "
                f"· items: {items}")

    def _control_game(self, tail: str, chat_key: str, *,
                       player: Any = None, kind: str = "dm") -> str:
        from .features import feature_enabled

        _log.debug("_control_game: tail=%r, chat_key=%s, player=%s, kind=%s",
                  tail, chat_key, player, kind)
        if not feature_enabled(self.context, "games"):
            _log.debug("games feature is disabled")
            return "games are off. /features games on"
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else ""
        _log.debug("_control_game: verb=%r", verb)
        engine = self._game_engine()
        live = engine.live(chat_key)
        _log.debug("_control_game: live=%s", live)

        if verb in {"", "list", "help"}:
            lines = []
            if live is not None:
                lines.append(
                    f"live game: {live.game} — send your move (or /game quit)")
            lines.append(engine.list_games())
            lines.append("  /game <name> — start · /game rematch — run it back")
            lines.append("  /game invite <game> [who] · /game accept <code> — DM duels")
            return "\n".join(lines)
        if verb == "quit":
            # capture the relay (if any) BEFORE quitting — the engine
            # tears it down, and the opponent deserves to hear about it
            relay = self._game_relay()
            doomed = relay.get_relay_for_chat(chat_key)
            out = engine.quit(chat_key)
            if doomed is not None:
                who = player.name if player is not None else "your opponent"
                self._relay_send(
                    doomed.other_chat(chat_key),
                    f"🏁 {who} closed the {doomed.game_name} duel.")
            return "\n".join(out)
        if verb == "rematch":
            room, msgs = engine.rematch(chat_key)
            return "\n".join(msgs)
        if verb in {"leaderboard", "board", "ranks"}:
            game = parts[1].lower() if len(parts) > 1 else ""
            return engine.board.render(10, game=game)
        if verb == "stats":
            want = " ".join(parts[1:]).strip().lower()
            if want:
                for prof in engine.store.all():
                    if want in prof.key.lower() or want in prof.name.lower():
                        return self._profile_line(prof)
                return f"no player found matching {want!r}."
            if player is not None:
                return self._profile_line(engine.store.get(player.key))
            return "usage: /game stats [name]"
        if verb == "balance":
            if player is None:
                return "balance needs a chat sender — run it where you play."
            prof = engine.store.get(player.key)
            items = ", ".join(f"{k}×{v}" for k, v in (prof.items or {}).items()) or "none"
            return f"🪙 {prof.coins} coins · {prof.points} points · items: {items}"
        if verb == "shop":
            rest = " ".join(parts[1:]).strip()
            if rest.startswith("buy ") and player is not None:
                ok, msg = engine.economy.purchase(player, rest[4:].strip())
                return msg
            return engine.economy.catalog_text("", player)
        if verb == "join":
            if player is None:
                return "join needs a chat sender — say it where the game is live."
            return "\n".join(engine.join(chat_key, player)) or "joined."
        if verb == "invite":
            if player is None:
                return "invite needs a chat sender."
            if len(parts) < 2:
                return ("usage: /game invite <game_name> [who]\n"
                        "the code is the invite — your friend accepts with "
                        "/game accept <code> from any chat.")
            game_name = parts[1].lower()
            to_label = parts[2] if len(parts) > 2 else ""
            try:
                relay = self._game_relay()
                invite = relay.create_invite(
                    chat_key, player, game_name, to_label=to_label)
            except ValueError as exc:
                return str(exc)
            share = (f"invite ready for {invite.game_name} — share this:\n"
                     f"/game accept {invite.code}\n"
                     f"(expires in 1 hour, works from any chat)")
            target = self._resolve_invite_target(chat_key, to_label)
            if target is not None:
                sent = self._relay_send(
                    target,
                    f"🎮 {player.name} invited you to play "
                    f"{invite.game_name}!\n"
                    f"to accept: /game accept {invite.code}\n"
                    f"(expires in 1 hour)")
                if sent:
                    return (f"invite sent to {to_label} for "
                            f"{invite.game_name}.\n{share}")
            return share
        if verb == "accept":
            if player is None:
                return "accept needs a chat sender."
            if len(parts) < 2:
                return "usage: /game accept <invite_code>"
            code = parts[1]
            try:
                relay = self._game_relay()
                relay_room = relay.accept_invite(code, chat_key, player)
                # tell the inviter their friend joined (the accepter is
                # already looking at this chat)
                self._relay_send(
                    relay_room.chat_a,
                    f"🎮 {player.name} accepted your "
                    f"{relay_room.game_name} invite — game on! play in "
                    f"your DM, moves are relayed.")
                return ("game started! play here in this chat — your moves "
                        "are relayed to your opponent.")
            except ValueError as exc:
                return str(exc)
        if verb in engine.games:
            if player is None:
                return "start a game from a chat — I need to know who's at the table."
            # "/hangman daily" — same word for everyone, all day
            # "/game case timed" — countdown mode with a speed bonus
            words = [p.lower() for p in parts[1:]]
            daily = "daily" in words
            timed = "timed" in words
            try:
                room, msgs = engine.start(chat_key, verb, player, kind=kind,
                                          daily=daily, timed=timed)
            except ValueError as exc:
                return str(exc)
            if msgs and not msgs[0].startswith("🎮"):
                msgs[0] = f"🎮 {msgs[0]}"  # the table-opening banner
            return "\n".join(msgs) or engine.describe(room)
        return (f"unknown game {verb!r} — /game list to see the table.")

    def _route_game_move(self, chat_key: str, text: str, *,
                         player: Any = None, kind: str = "dm") -> str | None:
        """While a game is live in this chat, plain messages are game moves.

        The multi-player engine owns every game (39 and counting, every
        platform). Relay rooms (DM-to-DM multiplayer) are checked first.
        """
        try:
            from .features import feature_enabled

            if not text:
                return None
            if not feature_enabled(self.context, "games"):
                return None
            
            # Check for relay room first (DM-to-DM multiplayer)
            relay = self._game_relay()
            relay_room = relay.get_relay_for_chat(chat_key)
            if relay_room is not None and player is not None:
                # route through the virtual room; the reply goes back to
                # this chat via the normal path, the opponent gets theirs
                # on their own platform
                msgs = relay.relay_move(chat_key, text, player)
                if not msgs:
                    return None
                reply = "\n".join(msgs)
                self._relay_send(relay_room.other_chat(chat_key), reply)
                return reply
            
            engine = self._game_engine()
            room = engine.live(chat_key)
            if room is not None:
                # A live room owns plain messages (moves) and the engine's
                # own commands (/status /pass /shop /leave …).  /game is the
                # control plane, and any OTHER slash command (/mood, /status
                # the bot, …) falls through to the normal control/brain flow.
                if text.startswith("/"):
                    from ..games.games.base import parse_command

                    cmd, _rest = parse_command(text)
                    if cmd in ("", "game"):
                        return None  # not an engine command — let it pass
                    is_engine_command = True
                else:
                    is_engine_command = False
                if player is None:
                    return None
                msgs: list[str] = []
                # a group auto-seats a new human the moment they SPEAK —
                # engine commands don't seat, they just work
                if (room.kind == "group" and room.player(player.key) is None
                        and not is_engine_command):
                    msgs.extend(engine.join(chat_key, player))
                msgs.extend(engine.move(chat_key, text, player, kind=kind))
                return "\n".join(msgs) or None
            return None
        except Exception:  # noqa: BLE001 - a game bug must never eat the chat
            _log.exception("game move failed")
            return None

    # ── news ─────────────────────────────────────────────────────────────────
    def _control_news(self, tail: str) -> str:
        from .features import feature_enabled
        from .news import NewsAgent
        from .notifier import Notifier

        if not feature_enabled(self.context, "news"):
            return "news is off. /features news on"
        agent = NewsAgent(self.context, notifier=Notifier(self.context, self.gateway))
        parts = tail.split()
        verb = parts[0].lower() if parts else "run"
        if verb == "status":
            rows = agent.recent(8)
            if not rows:
                return "no news stored yet — /news run to fetch the feeds."
            lines = ["recent news:"]
            for row in rows:
                lines.append(f"  [{row['source']}] {row['title'][:70]}")
            return "\n".join(lines)
        if verb == "run":
            cap = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            report = agent.run(cap)
            if not report["ok"]:
                errs = "; ".join(report["errors"][:3])
                return f"news run got nothing readable ({report['items']} items)" + (f" — {errs}" if errs else "")
            return (f"news digest — {report['items']} new items (of {report['fresh']} fresh) "
                    f"— delivered where a channel is live.\n\n{report['digest'][:3000]}")
        return "usage: /news [run [n]|status]"

    # ── always-on research ───────────────────────────────────────────────────
    def _control_research(self, tail: str) -> str:
        from .features import feature_enabled
        from .news import NewsAgent  # noqa: F401 - keeps imports local & symmetric
        from .notifier import Notifier
        from .researcher import DOMAINS, ResearchAgent

        if not feature_enabled(self.context, "research"):
            return "research is off. /features research on"
        agent = ResearchAgent(self.context, notifier=Notifier(self.context, self.gateway))
        parts = tail.split()
        verb = parts[0].lower() if parts else "status"
        if verb == "status":
            counts: dict[str, int] = {}
            try:
                for row in self.context.db.query("SELECT domain, COUNT(*) AS n FROM research_log GROUP BY domain"):
                    counts[str(row.get("domain"))] = int(row.get("n", 0))
            except Exception:  # noqa: BLE001
                pass
            lines = [f"research feature: on — domains: {', '.join(sorted(DOMAINS))}"]
            lines.append(f"stored digests: {sum(counts.values())}" +
                         (f" ({', '.join(f'{d} {n}' for d, n in sorted(counts.items()))})" if counts else ""))
            lines.append(f"background loop: {'running' if agent.loop_running() else 'idle'}")
            return "\n".join(lines)
        if verb == "run":
            domain = parts[1] if len(parts) > 1 and parts[1] in DOMAINS else None
            result = agent.run_cycle(domain)
            if not result.get("ok"):
                return f"research failed: {result.get('error')}"
            return (
                f"researched ({result['domain']} / {result.get('researcher', '?')}): {result['topic']}\n"
                f"suggestion: {result['suggestion']}\n"
                f"score: {result.get('score', 0):.2f} → {result.get('status', '?')}\n"
                f"({result.get('seconds', 0)}s — {self._status_note(result.get('status'))})"
            )
        if verb in {"ideas", "queue", "pending"}:
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            rows = agent.list_proposals(limit=n)
            if not rows:
                return "no proposals yet — /research run"
            lines = [f"top proposals ({len(rows)}):"]
            for row in rows:
                lines.append(
                    f"· {row['id'][:8]}  {row['score']:.2f}  [{row['status']}] "
                    f"{row['domain']} — {str(row['topic'])[:52]}\n"
                    f"  {str(row['suggestion'])[:120]}"
                )
            lines.append("approve: /research approve <id|latest>   deny: /research deny <id|latest>")
            return "\n".join(lines)
        if verb == "approve" and len(parts) > 1:
            return agent.approve(parts[1])
        if verb == "deny" and len(parts) > 1:
            return agent.deny(parts[1])
        if verb == "history":
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 5
            if self.context.db is None:
                return "no database"
            rows = self.context.db.query(
                "SELECT id, domain, topic, score, status, created_at FROM research_log "
                "ORDER BY created_at DESC LIMIT ?", (max(1, min(n, 20)),))
            if not rows:
                return "no research history yet"
            lines = ["recent research:"]
            for row in rows:
                lines.append(
                    f"· {row['id'][:8]}  {float(row.get('score') or 0):.2f}  "
                    f"[{row.get('status', 'pending')}] {row['domain']} — "
                    f"{str(row.get('topic'))[:56]}"
                )
            return "\n".join(lines)
        return "usage: /research [run [lifestyle|tech|cyber]|status|ideas [n]|approve <id|latest>|deny <id|latest>|history [n]]"

    @staticmethod
    def _status_note(status: str) -> str:
        return {
            "notified": "delivered",
            "pending": "valuable but over the daily cap — queued (see /research ideas)",
            "skipped": "below the quality bar — logged, not delivered",
        }.get(status, "stored")

    # ── the coding bot ───────────────────────────────────────────────────────
    def _control_code(self, tail: str, chat_key: str) -> str:
        from .coding import CodingAgent

        task = (tail or "").strip()
        if not task:
            return "usage: /code <what to build> — e.g. /code write a script that renames all .JPG files to .jpg"
        chat = self._ref_from_key(chat_key)
        try:
            self.gateway.send(chat.platform, chat, f"⏳ coding bot: {task[:70]}\n(draft → run → fix, up to 5 tries)")
        except Exception:  # noqa: BLE001
            pass
        started = time.time()
        try:
            result = CodingAgent(self.context).run(task, max_iterations=5, timeout=120.0)
        except Exception as exc:  # noqa: BLE001
            return f"coding bot crashed: {exc}"
        elapsed = time.time() - started
        if result.ok:
            files = ", ".join(getattr(result, "files", []) or []) or "output"
            output = str(getattr(result, "output", ""))[:600]
            text = (f"✅ code done in {result.iterations} iteration(s), {elapsed:.0f}s — {files}\n"
                    f"run output:\n{output}")
        else:
            text = (f"❌ coding bot gave up after {result.iterations} iteration(s), {elapsed:.0f}s\n"
                    f"{str(getattr(result, 'error', ''))[:600]}\n"
                    f"it's in the workspace if you want to take over.")
        return self._send_long_checked(chat.platform, chat, text)

    # ── code interpreter (run python, keep a session) ───────────────────────
    def _control_py(self, tail: str, chat_key: str) -> str:
        """/py <python code> — run code in the sandbox and get the result.

        ``/py -s <name> <code>`` keeps variables in a named session;
        ``/py -r [name]`` resets a session. Runs inside the sandbox
        (resource-limited, no network) — same trust level as /code.
        """
        from ..tools.sandbox_code import CodeInterpreter

        raw = (tail or "").strip()
        if not raw:
            return ("usage: /py <python code>\n"
                    "  /py -s <name> <code>   keep variables in a session\n"
                    "  /py -r [name]          reset a session (default: console)\n"
                    "e.g. /py sum(range(100)) or /py -s work x = 10")
        session, reset = "console", False
        parts = raw.split(None, 2)
        if parts[0] == "-r":
            reset, session = True, (parts[1] if len(parts) > 1 else "console")
        elif parts[0] == "-s":
            if len(parts) < 3:
                return "usage: /py -s <name> <code>"
            session, raw = parts[1], parts[2].strip()
            if not raw:
                return "usage: /py -s <name> <code>"
        else:
            raw = raw.strip()

        try:
            if reset:
                CodeInterpreter().run("", session=session, reset=True)
                return f"session '{session}' reset."
            result = CodeInterpreter().run(raw, session=session, timeout=60.0)
        except Exception as exc:  # noqa: BLE001
            return f"code error: {exc}"

        if not result.get("ok") and result.get("error"):
            return f"code error: {result['error']}"
        lines = []
        if result.get("result") is not None:
            lines.append(f"→ {str(result['result'])[:400]}")
        if result.get("stdout"):
            lines.append(result["stdout"].strip()[:1200])
        if result.get("stderr"):
            lines.append("⚠ " + result["stderr"].strip()[-600:])
        if result.get("files"):
            lines.append("files: " + ", ".join(result["files"][:12]))
        if not lines:
            lines.append("(ran clean, no output)")
        head = f"[{session}] exit={result.get('exit_code')}, {result.get('seconds', 0):.2f}s"
        if result.get("timed_out"):
            head += " (timed out)"
        body = head + "\n" + "\n".join(l for l in lines if l)
        chat = self._ref_from_key(chat_key)
        return self._send_long_checked(chat.platform, chat, body)

    # ── long-term memory: /remember /recall /forget ──────────────────────────
    def _control_remember(self, tail: str, *, chat_key: str = "") -> str:
        """Explicitly store something: /remember <text> [kind] [tags:a,b]."""
        from ..memory.base import ALL_KINDS

        raw = (tail or "").strip()
        if not raw:
            return ("usage: /remember <what to remember> [kind] [tags:a,b]\n"
                    "kinds: fact preference decision relationship episode skill lesson")
        tokens = raw.split()
        kind = "episode"
        for i, tok in enumerate(tokens):
            if tok.lower() in ALL_KINDS:
                kind = tok.lower()
                tokens = tokens[:i] + tokens[i + 1:]
                break
        content = " ".join(tokens)
        tags = ""
        if " " in content and content.rsplit(" ", 1)[-1].startswith("tags:"):
            content, tagpart = content.rsplit(" ", 1)
            tags = tagpart[len("tags:"):]
        content = content.strip()
        if not content:
            return "usage: /remember <what to remember> [kind] [tags:a,b]"
        record_id = self.context.memory.remember(
            content, kind=kind, importance=0.9,
            source="user:command", origin=f"chat:{chat_key or 'command'}",
            tags=tags,
        )
        return f"remembered it. [{kind}] {content[:120]} (id {record_id[:8]})"

    def _control_recall(self, tail: str) -> str:
        """Show the top memories matching the query (or the freshest, if none)."""
        query = (tail or "").strip()
        if self.context.memory is None:
            return "memory is off in this session."
        result = self.context.memory.recall(query, limit=5)
        if not result.records:
            return "nothing in memory matches that yet — try /remember."
        lines = ["from memory:"]
        now = time.time()
        for r in result.records:
            days = (now - r.created_at) / 86400.0
            age = f"today" if days < 0.05 else f"{days:.0f}d ago" if days < 60 else f"{days / 30.0:.0f}mo ago"
            score = f" {r.score:.2f}" if r.score else ""
            tags = f" #{r.tags}" if r.tags else ""
            lines.append(f"  [{r.kind}]{tags} {r.content[:130]}  ({age}, id {r.id[:6]})")
        return "\n".join(lines)

    def _control_forget(self, tail: str) -> str:
        """Forget by id, or by the best matching description."""
        target = (tail or "").strip()
        if not target:
            return "usage: /forget <id or description>"
        if self.context.memory is None:
            return "memory is off in this session."
        if re.fullmatch(r"[0-9a-f]{12}", target):
            removed = self.context.memory.forget(target)
            return "forgotten." if removed else f"no memory with id {target}."
        record = self.context.memory.find_one(target)
        if record is None:
            return "didn't find that in memory — give me the id from /recall, or better words."
        self.context.memory.forget(record.id)
        return f"forgotten: [{record.kind}] {record.content[:110]}"

    # ── voice: /tts /stt ─────────────────────────────────────────────────────
    def _control_tts(self, tail: str, *, chat_key: str) -> str:
        """Speak text: synthesize with the best available engine, send the file."""
        text = (tail or "").strip()
        if not text:
            return "usage: /tts <text to speak>"
        outcome = self.context.tools.call("speak", text=text[:4000])
        if not outcome.ok:
            return f"tts failed: {getattr(outcome.error, 'message', outcome.error)}"
        info = outcome.value
        chat = self._ref_from_key(chat_key)
        try:
            result = self.gateway.send_file(chat.platform, chat, info["path"],
                                            caption=f"[{info.get('engine')}]")
            if getattr(result, "ok", False):
                return f"🔊 spoken ({info.get('engine')}, {info.get('bytes', 0) // 1024} KB, {info.get('seconds')}s)"
        except Exception:  # noqa: BLE001 - console and other adapters
            pass
        return f"🔊 spoken ({info.get('engine')}) — saved at {info.get('path')}"

    def _control_stt(self, tail: str) -> str:
        """Transcribe an audio file with the best available STT backend."""
        path = (tail or "").strip()
        if not path:
            return "usage: /stt <audio file path>"
        outcome = self.context.tools.call("transcribe", path=path)
        if not outcome.ok:
            return f"stt failed: {getattr(outcome.error, 'message', outcome.error)}"
        value = outcome.value
        text = value.get("text") or "(no speech detected)"
        return f"🎧 [{value.get('provider')}, {value.get('seconds')}s]\n{text[:1500]}"

    # ── vision: /look (screen reader) ────────────────────────────────────────
    def _control_look(self, tail: str, *, chat_key: str) -> str:
        """Screen-reader analysis of a screenshot (path or URL)."""
        parts = (tail or "").strip().split(None, 1)
        if not parts:
            return "usage: /look <path or url> [what to focus on]"
        target = parts[0]
        focus = parts[1] if len(parts) > 1 else ""
        kwargs = {"url": target} if target.startswith(("http://", "https://")) else {"path": target}
        outcome = self.context.tools.call("vision_screen", focus=focus, **kwargs)
        if not outcome.ok:
            return f"vision failed: {getattr(outcome.error, 'message', outcome.error)}"
        value = outcome.value
        description = value.get("description") or "(vision model unavailable — metadata only)"
        ocr_text = str(value.get("ocr_text") or "").strip()
        text = (f"👁 screen read ({value.get('format', '?')}, {value.get('width', '?')}x{value.get('height', '?')}) "
                f"[{value.get('provider') or 'no provider'}]\n{description}")
        if ocr_text:
            text += f"\n\n— verbatim text from the pixels (OCR) —\n{ocr_text[:2000]}"
        chat = self._ref_from_key(chat_key)
        return self._send_long_checked(chat.platform, chat, text[:6000])

    # ── scheduler: /schedule ─────────────────────────────────────────────────
    def _scheduler_or_build(self) -> Any:
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is None:
            from .scheduler import Scheduler

            scheduler = Scheduler(self.context, gateway=self.gateway)
            self._scheduler = scheduler
        return scheduler

    def _control_schedule(self, tail: str) -> str:
        from ..core.errors import AmbiguousRef
        scheduler = self._scheduler_or_build()
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        if verb in {"status", "list"}:
            jobs = scheduler.list_jobs()
            if not jobs:
                return "no scheduled jobs — /schedule add <name> <when> <action>"
            lines = ["scheduled jobs:"]
            for job in jobs[:15]:
                state = "on " if job["enabled"] else "off"
                nxt = job.get("next_run_iso") or ("past" if job["kind"] == "at" else "—")
                lines.append(f"  [{state}] {job['name']} — {job['kind']} {job['spec']} (next: {nxt})")
                if job.get("last_result"):
                    lines.append(f"        last: {job['last_result'][:80]}")
            return "\n".join(lines)
        if verb == "add":
            return self._schedule_add(parts[1:], scheduler)
        if verb in {"rm", "remove"}:
            ref = " ".join(parts[1:])
            if not ref:
                return "usage: /schedule rm <name or id>"
            try:
                return "removed." if scheduler.remove(ref) else f"no job named {ref!r}"
            except AmbiguousRef as exc:
                return str(exc)
        if verb in {"enable", "disable"}:
            ref = " ".join(parts[1:])
            if not ref:
                return f"usage: /schedule {verb} <name or id>"
            try:
                job = scheduler.set_enabled(ref, enabled=(verb == "enable"))
            except (LookupError, AmbiguousRef) as exc:
                return str(exc)
            return f"{job['name']} {'enabled' if job['enabled'] else 'disabled'} (next {job.get('next_run_iso') or '—'})."
        if verb == "run":
            ref = " ".join(parts[1:])
            if not ref:
                return "usage: /schedule run <name or id>"
            try:
                outcome = scheduler.run_now(ref)
            except (LookupError, AmbiguousRef) as exc:
                return str(exc)
            return f"ran {outcome['name']}: {outcome['result'][:400]}"
        return ("usage: /schedule add <name> <when> <message|tool|command> <...> | list | "
                "rm <name> | enable|disable <name> | run <name>\n"
                "  when: 'at 2026-12-25 09:00' | 'every 30m' | '22:00'\n"
                "  action: message goodnight | tool web_research {\"query\":\"ai news\"} | command python3 -V")

    def _schedule_add(self, parts: list[str], scheduler: Any) -> str:
        """parts = the tail after 'add': [name, spec..., verb, payload...]"""
        if len(parts) < 3:
            return ("usage: /schedule add <name> <when> <message|tool|command> <...>\n"
                    "  e.g. /schedule add goodnight 22:00 message goodnight 🌙")
        name = parts[0]
        rest = parts[1:]
        payload_kind = ""
        split_at = -1
        for i, tok in enumerate(rest):
            if tok.lower() in {"message", "tool", "command"}:
                payload_kind = tok.lower()
                split_at = i
                break
        if split_at < 0:
            return "add needs an action: message <text> | tool <name> <json> | command <cmd>"
        spec = " ".join(rest[:split_at])
        payload_parts = rest[split_at + 1:]
        payload: dict[str, Any]
        if payload_kind == "message":
            payload = {"text": " ".join(payload_parts)}
        elif payload_kind == "tool":
            if not payload_parts:
                return "tool action needs: <tool name> <json args>"
            args: dict[str, Any] = {}
            if len(payload_parts) > 1:
                try:
                    parsed = json.loads(" ".join(payload_parts[1:]))
                    if isinstance(parsed, dict):
                        args = parsed
                except (ValueError, TypeError):
                    return "tool args must be a JSON object, e.g. {\"query\":\"ai news\"}"
            payload = {"tool": payload_parts[0], "args": args}
        else:
            if not payload_parts:
                return "command action needs the command text"
            payload = {"command": " ".join(payload_parts)}
        try:
            job = scheduler.add(name, spec, payload_kind, payload)
        except (ValueError, RuntimeError) as exc:
            return f"scheduling failed: {exc}"
        when = time.strftime("%m-%d %H:%M", time.localtime(job["next_run"]))
        return f"⏰ scheduled {job['name']} — {job['kind']} ({spec}) — next {when}"

    # ── database: /db ────────────────────────────────────────────────────────
    def _control_db(self, tail: str) -> str:
        parts = (tail or "").split(None, 1)
        verb = parts[0].lower() if parts else "counts"
        tools = self.context.tools
        if verb == "tables":
            outcome = tools.call("db_tables")
            if not outcome.ok:
                return f"db failed: {getattr(outcome.error, 'message', outcome.error)}"
            value = outcome.value
            lines = [f"tables ({value['count']}):"]
            for table in value["tables"][:100]:
                lines.append(f"  {table['name']} — {table['rows']} rows")
            return "\n".join(lines)
        if verb == "schema":
            if len(parts) < 2:
                return "usage: /db schema <table>"
            outcome = tools.call("db_schema", table=parts[1].strip())
            if not outcome.ok:
                return f"db failed: {getattr(outcome.error, 'message', outcome.error)}"
            value = outcome.value
            lines = [f"{value['table']} ({value['rows']} rows):"]
            for col in value["columns"]:
                flags = "".join(flag for cond, flag in (
                    (col["pk"], "PK"), (col["not_null"], "NN"),
                ) if cond)
                default = f" default={col['default']}" if col.get("default") is not None else ""
                lines.append(f"  {col['name']} {col['type']} {flags}{default}")
            return "\n".join(lines)
        if verb == "query":
            if len(parts) < 2:
                return "usage: /db query <select sql>"
            outcome = tools.call("db_query", sql=parts[1].strip())
            if not outcome.ok:
                return f"db failed: {getattr(outcome.error, 'message', outcome.error)}"
            value = outcome.value
            rows = value["data"][:20]
            if not rows:
                return f"{value['sql']}\n(no rows)"
            cols = value["columns"]
            lines = [" ".join(cols)]
            for row in rows:
                lines.append(" | ".join(str(row.get(c, ""))[:40] for c in cols))
            more = "" if len(rows) == value["rows"] else f"\n… {value['rows'] - len(rows)} more (limit forced ≤ 200)"
            return "\n".join(lines) + more
        if verb == "counts":
            outcome = tools.call("db_counts")
            if not outcome.ok:
                return f"db failed: {getattr(outcome.error, 'message', outcome.error)}"
            lines = ["largest tables:"]
            for table in outcome.value["tables"]:
                lines.append(f"  {table['rows']:>8,}  {table['name']}")
            return "\n".join(lines)
        return "usage: /db tables | schema <table> | query <select sql> | counts"

    # ── api connectors: /api ─────────────────────────────────────────────────
    def _control_api(self, tail: str) -> str:
        parts = (tail or "").split(None, 1)
        if not parts or parts[0].lower() == "list":
            outcome = self.context.tools.call("api_list")
            if not outcome.ok:
                return f"api failed: {getattr(outcome.error, 'message', outcome.error)}"
            lines = ["API connectors:"]
            for conn in outcome.value["connectors"]:
                params = ", ".join(conn["params"]) or "no params"
                lines.append(f"  {conn['name']} — {conn['description'][:70]}\n      params: {params[:120]}")
            return "\n".join(lines)[:4000]
        name = parts[0].strip()
        params = parts[1].strip() if len(parts) > 1 else ""
        outcome = self.context.tools.call("api_call", connector=name, params=params)
        if not outcome.ok:
            return f"api failed: {getattr(outcome.error, 'message', outcome.error)}"
        return f"{name}:\n" + json.dumps(outcome.value, default=str, indent=1)[:4000]

    # ── swarm: /swarm ────────────────────────────────────────────────────────
    def _control_swarm(self, tail: str, *, chat_key: str) -> str:
        stripped = (tail or "").strip()
        # wave 86: /swarm research <topic> — the research swarm (parallel
        # researchers, conflict-aware synthesis) instead of devon builders.
        if stripped.lower().startswith("research"):
            topic = stripped[8:].strip()
            if not topic:
                return "usage: /swarm research <topic> — parallel research swarm"
            return self._control_swarm_research(topic, chat_key=chat_key)
        from .swarm import SwarmAgent

        goal, workers = stripped, 3
        parts = stripped.rsplit(None, 1)
        if len(parts) == 2 and parts[1].isdigit() and 1 <= int(parts[1]) <= 5:
            goal, workers = parts[0].strip(), int(parts[1])
        if not goal:
            return ("usage: /swarm <goal> [workers 1-5] — parallel devon agents + fusion\n"
                    "        /swarm research <topic> — parallel research swarm")
        chat = self._ref_from_key(chat_key)
        try:
            self.gateway.send(chat.platform, chat,
                              f"🐝 swarm: {goal[:80]}\n{workers} worker(s) — this takes a few minutes.")
        except Exception:  # noqa: BLE001
            pass
        started = time.time()
        try:
            result = SwarmAgent(self.context, brain=self.brain, gateway=self.gateway).run(
                goal, workers=workers)
        except Exception as exc:  # noqa: BLE001
            return f"swarm crashed: {exc}"
        elapsed = time.time() - started
        ok_legs = sum(1 for leg in result.legs if leg.get("ok"))
        text = (f"🐝 swarm done: {ok_legs}/{len(result.legs)} legs ok, {elapsed:.0f}s\n\n"
                f"{result.synthesis}")
        return self._send_long_checked(chat.platform, chat, text)

    def _control_swarm_research(self, topic: str, *, chat_key: str) -> str:
        """wave 86: /swarm research <topic> — parallel research swarm."""
        from .research_swarm import ResearchSwarm

        chat = self._ref_from_key(chat_key)
        try:
            self.gateway.send(chat.platform, chat,
                              f"🔬 research swarm: {topic[:80]}\n"
                              f"several angles at once — a minute or two.")
        except Exception:  # noqa: BLE001
            pass
        started = time.time()
        try:
            swarm = ResearchSwarm(self.context)
            report = swarm.run(topic, save_memory=True)
        except Exception as exc:  # noqa: BLE001
            return f"research swarm crashed: {exc}"
        elapsed = time.time() - started
        text = (f"🔬 swarm: {len(report.findings)} findings, "
                f"{len(report.angles)} angles, {len(report.sources)} sources, "
                f"{elapsed:.0f}s\n\n" + report.to_text(3200))
        return self._send_long_checked(chat.platform, chat, text)

    # ── power layer: network / proxy / gen / osint / record / macro ──────────

    def _tool_reply(self, tool: str, chat_key: str, **kwargs: Any) -> str:
        """Run one tool and stream its JSON back to the chat (or the error)."""
        outcome = self.context.tools.call(tool, **kwargs)
        chat = self._ref_from_key(chat_key)
        if not outcome.ok:
            return f"{tool} failed: {getattr(outcome.error, 'message', outcome.error)}"
        text = json.dumps(outcome.value, default=str, indent=1)
        return self._send_long_checked(chat.platform, chat, text[:6000])

    def _control_dns(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split()
        if not parts:
            return "usage: /dns <domain> [record type]"
        domain = parts[0]
        record = parts[1].upper() if len(parts) > 1 else "A"
        outcome = self.context.tools.call("dns_lookup", domain=domain, record=record)
        if not outcome.ok:
            return f"dns failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        answers = ", ".join(v["answers"][:16]) or "(none)"
        return f"{v['domain']} {v['record']}: {answers}  ({v['seconds']}s)"

    def _control_scan(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split()
        if not parts:
            return "usage: /scan <target> [ports] [banner] — own infrastructure only"
        target = parts[0]
        rest = parts[1:]
        banner = ""
        if rest and rest[-1].lower() in {"banner", "1", "true", "yes", "on"}:
            banner = "true"
            rest = rest[:-1]
        ports = rest[0] if rest else ""
        outcome = self.context.tools.call("port_scan", target=target, ports=ports, banner=banner)
        if not outcome.ok:
            return f"scan failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        open_ports = ", ".join(str(p) for p in v["open"]) or "none"
        return (f"scan {v['scanned']} ({v['seconds']}s): open = {open_ports} | "
                f"closed {v['closed']} | filtered {v['filtered']}  [scope: {v['scope']}]")

    def _control_whois(self, tail: str, chat_key: str = "") -> str:
        domain = (tail or "").strip()
        if not domain:
            return "usage: /whois <domain>"
        outcome = self.context.tools.call("whois_lookup", domain=domain)
        if not outcome.ok:
            return f"whois failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        lines = [f"whois {v.get('domain')}"]
        for key in ("registrar", "registration", "expiration", "last_changed"):
            if v.get(key):
                lines.append(f"  {key}: {v[key]}")
        if v.get("nameservers"):
            lines.append(f"  nameservers: {', '.join(v['nameservers'][:6])}")
        if v.get("status"):
            lines.append(f"  status: {', '.join(v['status'][:5])}")
        return "\n".join(lines)

    def _control_ports(self, tail: str, chat_key: str = "") -> str:
        outcome = self.context.tools.call("local_services")
        if not outcome.ok:
            return f"ports failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        if not v["listeners"]:
            return "nothing listening (or /proc not available here)"
        lines = [f"listening on this machine ({v['count']}):"]
        for item in v["listeners"][:30]:
            lines.append(f"  {item['port']:<6} {item['address']}")
        return "\n".join(lines)

    # ── wave 84: the virtual CPU farm ─────────────────────────────────────────
    def _control_workspace(self, tail: str) -> str:
        from ..workspace import Workspace
        ws = getattr(self.context, "workspace", None)
        if ws is None:
            # Cache on the context so context.close() can shut it down;
            # spawning one per call would leak a thread fleet per message.
            ws = Workspace(self.context)
            try:
                setattr(self.context, "workspace", ws)
            except Exception:  # noqa: BLE001
                pass
        parts = [t for t in (tail or "").split() if t]
        verb = (parts[0].lower() if parts else "status")
        value = parts[1] if len(parts) > 1 else ""
        if verb == "scale" and value.isdigit():
            return f"farm resized to {ws.scale_to(int(value))} vcpu(s) — {ws.summary_line()}"
        if verb == "up":
            return f"farm: {ws.summary_line()}" if ws.scale_up(1) == len(ws.vcpus()) \
                else f"vcpu added — {ws.summary_line()}"
        if verb == "down":
            ws.scale_down(1)
            return f"vcpu retired (drained) — {ws.summary_line()}"
        if verb in {"pause", "resume"} and value:
            vcpu = ws.get(value)
            if vcpu is None:
                return f"no core named {value!r} — {ws.summary_line()}"
            (vcpu.pause() if verb == "pause" else vcpu.resume())
            return f"{value} is now {vcpu.status} — {ws.summary_line()}"
        if verb == "add" and value in {"io", "cpu", "balanced"}:
            v = ws.add_vcpu(kind=value)
            return f"attached {v.name} ({v.kind}) — {ws.summary_line()}"
        if verb == "remove" and value:
            ok = ws.remove_vcpu(value)
            return (f"{value} detached — {ws.summary_line()}" if ok
                    else f"could not remove {value!r} (unknown, or below the "
                         f"farm minimum)")
        st = ws.status()
        prof = st["profile"]
        lines = [f"🖥 {ws.summary_line()}",
                 f"profile {prof['kind']} ({prof['detail']}) — envelope "
                 f"{st['min_vcpus']}-{st['target_vcpus']}-{st['max_vcpus']}, "
                 f"autoscale {'on' if st['autoscale'] else 'off'}"]
        for d in st["details"]:
            flag = "⏸" if d["status"] == "paused" else \
                   "⛔" if d["status"] == "error" else "  "
            lines.append(f"  {flag}{d['id']:8s} [{d['status']:6s}] {d['kind']:8s} "
                         f"load {d['load']:.0%} queue {d['queue']} "
                         f"tasks {d['stats']['tasks_run']}")
        if st["last_scale"]:
            lines.append(f"last scale: {st['last_scale']}")
        lines.append("controls: /workspace scale <n> · up · down · "
                     "pause|resume <vcpu> · add <io|cpu|balanced> · remove <vcpu>")
        return "\n".join(lines)

    def _control_proxy(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        if verb == "status":
            outcome = self.context.tools.call("proxy_status")
            if not outcome.ok:
                return f"proxy failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            known = ", ".join(v["known"]) or "none"
            raw = "raw sockets routed too" if v.get("raw_sockets_routed") \
                else ("socks needs PySocks (pip install pysocks)"
                      if v["active"].lower().startswith(("socks5", "socks5h"))
                      else "http proxies route urllib traffic too")
            return (f"outbound: {v['active']}\nknown: {known}\n"
                    f"{raw} — all clients covered")
        if verb == "list":
            outcome = self.context.tools.call("proxy_list")
            if not outcome.ok:
                return f"proxy failed: {getattr(outcome.error, 'message', outcome.error)}"
            proxies = outcome.value["proxies"]
            return "proxies: " + (", ".join(proxies) if proxies else "none — /proxy set <url>")
        if verb == "test":
            return self._tool_reply("proxy_test", chat_key, proxy=" ".join(parts[1:]))
        if verb == "set":
            url = " ".join(parts[1:])
            if not url:
                return "usage: /proxy set http://host:port | socks5://host:port"
            return self._tool_reply("proxy_set", chat_key, proxy=url)
        if verb == "clear":
            return self._tool_reply("proxy_clear", chat_key)
        # ── proxy lab: free-proxy discovery + SSH tunnels ──────────────────
        if verb == "scrape":
            outcome = self.context.tools.call("proxy_scrape")
            if not outcome.ok:
                return f"scrape failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            lines = [f"🕸️ scraped {v['total']} unique proxies "
                     f"({v['seconds']}s, {len(v['by_source'])} sources)"]
            for src, n in sorted(v["by_source"].items(), key=lambda kv: -kv[1]):
                lines.append(f"  {src}: {n}")
            for src, err in list(v["errors"].items())[:4]:
                lines.append(f"  {src}: FAILED ({err[:60]})")
            disabled = v.get("disabled_sources") or []
            if disabled:
                lines.append(f"  (retired for 24h, dead too often: "
                             f"{', '.join(disabled[:4])})")
            stats = v.get("source_stats") or {}
            if stats.get("discovered"):
                lines.append(f"  sources: {stats['active']} active "
                             f"({stats['discovered']} learned from the "
                             f"internet)")
            if len("\n".join(lines)) > 900:
                return self._send_long_checked(self._ref_from_key(chat_key).platform,
                                self._ref_from_key(chat_key), "\n".join(lines))
            return "\n".join(lines)
        if verb == "discover":
            seeds = " ".join(parts[1:])
            outcome = self.context.tools.call(
                "proxy_discover", seeds=seeds)
            if not outcome.ok:
                return (f"discover failed: "
                        f"{getattr(outcome.error, 'message', outcome.error)}")
            v = outcome.value
            reg = v.get("registered") or []
            lines = [f"🔎 mined {v.get('endpoints_tested', 0)} endpoint(s) "
                     f"from {len(v.get('seeds') or [])} seed page(s) — "
                     f"registered {len(reg)} new source(s)"]
            for r in reg[:8]:
                lines.append(f"  + {r['name']} ({r['kind']}, "
                             f"{r['found']} proxies)")
            for seed, err in list((v.get("seed_errors") or {}).items())[:3]:
                lines.append(f"  {seed.split('/')[2] if seed.startswith('http') else seed}: "
                             f"FAILED ({err[:60]})")
            if not reg:
                lines.append("  no new working list endpoints found this "
                             "run — the catalog keeps what already works")
            stats = v.get("source_stats") or {}
            if stats:
                lines.append(f"  catalog now: {stats.get('active', 0)} "
                             f"active ({stats.get('discovered', 0)} learned)")
            return "\n".join(lines)
        if verb == "sources":
            outcome = self.context.tools.call("proxy_sources")
            if not outcome.ok:
                return f"sources failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            rows = v.get("sources") or []
            stats = v.get("stats") or {}
            lines = [f"proxy sources: {stats.get('active', len(rows))} active, "
                     f"{stats.get('disabled', 0)} retired "
                     f"({stats.get('discovered', 0)} learned)"]
            for row in rows[:12]:
                flag = "⛔" if row.get("disabled") else "  "
                found = row.get("last_found")
                lines.append(f"  {flag}{row['name']:28s} "
                             f"tests={row.get('tests', 0)} "
                             f"last={'dead' if row.get('fails') else str(found or 0) + ' found'}")
            if len(rows) > 12:
                lines.append(f"  … {len(rows) - 12} more")
            return "\n".join(lines)
        if verb == "refresh":
            return self._tool_reply("proxy_refresh", chat_key)
        if verb == "pool":
            scheme = parts[1] if len(parts) > 1 else ""
            outcome = self.context.tools.call(
                "proxy_pool", action="urls", scheme=scheme, limit="8")
            if not outcome.ok:
                return f"pool failed: {getattr(outcome.error, 'message', outcome.error)}"
            urls = outcome.value["urls"]
            if not urls:
                return "pool empty — /proxy refresh to scrape+test"
            lines = [f"🧱 working proxies ({outcome.value['count']}):"]
            lines += [f"  {u}" for u in urls]
            lines.append("use: /proxy set <url>  (routes all outbound traffic)")
            return "\n".join(lines)
        if verb == "file":
            # /proxy file — send a CLEAN file of the working proxies.
            outcome = self.context.tools.call("proxy_file")
            if not outcome.ok:
                return f"proxy file failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            if not v.get("written"):
                return f"no working proxies right now — {v.get('note', '/proxy refresh first')}"
            chat = self._ref_from_key(chat_key)
            sent = False
            try:
                result = self.gateway.send_file(chat.platform, chat, v["path"],
                                                caption=f"{v['count']} working proxies "
                                                        f"(fastest first)")
                sent = bool(getattr(result, "ok", False))
            except Exception:  # noqa: BLE001 - console adapter
                sent = False
            if sent:
                return (f"🧱 sent the clean file — {v['count']} working proxies "
                        f"(fastest: {v.get('fastest', '?')})")
            return (f"🧱 clean file ready ({v['count']} proxies): {v['path']} "
                    f"\n(plain chat — no file sending on this adapter)")
        if verb == "rotate":
            sub = parts[1].lower() if len(parts) > 1 else "status"
            if sub in {"on", "start"}:
                strategy = parts[2] if len(parts) > 2 else ""
                return self._tool_reply("proxy_rotate", chat_key,
                                        action="start", strategy=strategy)
            if sub in {"off", "stop"}:
                return self._tool_reply("proxy_rotate", chat_key,
                                        action="stop")
            if sub == "next":
                return self._tool_reply("proxy_rotate", chat_key,
                                        action="next")
            return self._tool_reply("proxy_rotate", chat_key, action="status")
        if verb == "ssh":
            sub = parts[1].lower() if len(parts) > 1 else "list"
            if sub == "start":
                # /proxy ssh start <name> <host> <user> [key-path] [port]
                if len(parts) < 5:
                    return ("usage: /proxy ssh start <name> <host> <user> "
                            "[key-path] [port]")
                return self._tool_reply("ssh_socks", chat_key,
                                        action="start",
                                        name=parts[2], host=parts[3],
                                        user=parts[4],
                                        key=parts[5] if len(parts) > 5 else "",
                                        port=parts[6] if len(parts) > 6 else "")
            return self._tool_reply("ssh_socks", chat_key,
                                    action=sub, name=parts[1] if len(parts) > 1 else "")
        return ("usage: /proxy status | list | test [url] | set <url> | clear |"
                " scrape | refresh | pool [scheme] | file | rotate on [strategy]"
                " | ssh start <name> <host> <user>")

    def _control_gen(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split(None, 2)
        if len(parts) < 2:
            outcome = self.context.tools.call("script_kinds")
            kinds = outcome.value["kinds"] if outcome.ok else {}
            lines = ["usage: /gen <kind> <name> [json config]"]
            for kind, cfg in sorted(kinds.items()):
                lines.append(f"  {kind}: " + ", ".join(cfg))
            return "\n".join(lines)
        kind, name = parts[0], parts[1]
        config = parts[2].strip() if len(parts) > 2 else ""
        outcome = self.context.tools.call("script_gen", kind=kind, name=name, config=config)
        if not outcome.ok:
            return f"gen failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        return (f"✅ generated {v['path']} ({v['bytes']} B, validated {v['ext']})\n"
                f"{v['preview']}")

    def _control_osint(self, tail: str, chat_key: str) -> str:
        target = (tail or "").strip()
        if not target:
            return ("usage: /osint <domain|ip|url|email> — public intel\n"
                    "  /osint campaign <seeds…> — automated investigation walk\n"
                    "  /osint graph clusters | node <ref> | merge <a> <b> | stats")
        # campaign: automated walk over the discovered web
        if target.startswith("campaign"):
            seeds = target[len("campaign"):].strip()
            if not seeds:
                return "usage: /osint campaign <phone|email|username|domain|ip>…"
            chat = self._ref_from_key(chat_key)
            started = time.time()
            outcome = self.context.tools.call("osint_campaign", seeds=seeds)
            if not outcome.ok:
                return f"campaign failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            lines = [f"🕸️ campaign: {v['steps']} investigations, "
                     f"{v['entities_investigated']} entities, "
                     f"{time.time() - started:.0f}s — {v['stopped']}"]
            for c in v.get("clusters", [])[:5]:
                ids = ", ".join(f"{e['kind']}:{e['value'][:28]}"
                                for e in c["entities"][:6])
                lines.append(f"  cluster [{c['size']}] conf {c['confidence']}: {ids}")
            for f in v.get("findings", [])[:8]:
                if f.get("error"):
                    lines.append(f"  ! {f['kind']}:{f['value'][:30]} — {f['error'][:60]}")
                else:
                    summary = f.get("summary") or {}
                    lines.append(f"  · {f['kind']}:{f['value'][:30]} — "
                                 f"{str(summary)[:90]}")
            return self._send_long_checked(chat.platform, chat, "\n".join(lines))  # report already delivered in chunks
        # graph: identity correlation queries
        if target.startswith("graph"):
            parts = target.split()
            sub = parts[1].lower() if len(parts) > 1 else "stats"
            if sub == "clusters":
                outcome = self.context.tools.call("osint_graph", action="clusters")
                if not outcome.ok:
                    return f"graph failed: {getattr(outcome.error, 'message', outcome.error)}"
                clusters = outcome.value["clusters"]
                if not clusters:
                    return "identity graph is empty — /osint campaign <seed> to build it"
                lines = ["🕸️ identity clusters:"]
                for c in clusters[:10]:
                    ids = ", ".join(f"{e['value'][:26]}" for e in c["entities"][:5])
                    lines.append(f"  [{c['size']}] conf {c['confidence']}: {ids}")
                return "\n".join(lines)
            if sub == "node" and len(parts) >= 3:
                outcome = self.context.tools.call(
                    "osint_graph", action="node", node=" ".join(parts[2:]))
                if not outcome.ok:
                    return f"graph failed: {getattr(outcome.error, 'message', outcome.error)}"
                n = outcome.value
                lines = [f"🧬 {n['kind']}:{n['value']} — conf {n['confidence']}, "
                         f"sources: {len(n['sources'])}"]
                for link in n["links"][:12]:
                    lines.append(f"  [{link['edge']}/{link['weight']}] "
                                 f"{link['kind']}:{link['value'][:36]}")
                return "\n".join(lines)
            if sub == "merge" and len(parts) >= 4:
                outcome = self.context.tools.call(
                    "osint_graph", action="merge",
                    a=" ".join(parts[2:3]), b=" ".join(parts[3:4]))
                if not outcome.ok:
                    return f"merge failed: {getattr(outcome.error, 'message', outcome.error)}"
                v = outcome.value
                return (f"merged {v.get('alias')} → {v.get('representative')}"
                        if v.get("merged") else v.get("note", "no-op"))
            if sub == "decoder" and len(parts) >= 3:
                # feed a Universal Decoder report (inline JSON or a
                # workspace file) into the identity graph
                payload = " ".join(parts[2:])
                try:
                    from ..tools.filesystem import safe_path

                    payload = safe_path(self.context, payload,
                                        must_exist=True).read_text()
                except Exception:  # noqa: BLE001
                    pass  # not a file — treat as inline JSON
                outcome = self.context.tools.call(
                    "osint_graph", action="ingest_decoder", report=payload)
                if not outcome.ok:
                    return ("decoder ingest failed: "
                            f"{getattr(outcome.error, 'message', outcome.error)}")
                v = outcome.value["ingest_decoder"]
                persons = ", ".join(v["persons"][:6]) or "—"
                domains = ", ".join(v["domains"][:6]) or "—"
                return (f"🕸️ decoder findings ingested ({v['source']}):\n"
                        f"  persons: {persons}\n  domains: {domains}")
            outcome = self.context.tools.call("osint_graph", action="stats")
            if not outcome.ok:
                return f"graph failed: {getattr(outcome.error, 'message', outcome.error)}"
            s = outcome.value
            return (f"identity graph: {s['nodes']} entities, {s['edges']} edges, "
                    f"{s['clusters']} clusters — {s['by_kind']}")
        chat = self._ref_from_key(chat_key)
        started = time.time()
        try:
            outcome = self.context.tools.call("osint_report", target=target)
        except Exception as exc:  # noqa: BLE001
            return f"osint failed: {exc}"
        if not outcome.ok:
            return f"osint failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        lines = [f"🔎 osint report: {v.get('target')} ({v.get('kind')}) — "
                 f"{time.time() - started:.1f}s"]
        for key, value in v.items():
            if key in {"target", "kind", "note"}:
                continue
            if isinstance(value, dict):
                lines.append(f"· {key}:\n" + json.dumps(value, default=str, indent=1)[:1200])
            elif isinstance(value, list):
                joined = ", ".join(map(str, value))[:400]
                lines.append(f"· {key}: {joined or '—'}")
            elif value:
                lines.append(f"· {key}: {str(value)[:300]}")
        return self._send_long_checked(chat.platform, chat, "\n".join(lines))

    def _control_record(self, tail: str, chat_key: str = "") -> str:
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        if verb == "start":
            if len(parts) < 2:
                return "usage: /record start <name>"
            outcome = self.context.tools.call("record_start", name=parts[1])
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            return (f"● recording {parts[1]} — every tool call you send me is captured. "
                    f"/record stop when done; it becomes /macro {parts[1]}.")
        if verb == "stop":
            outcome = self.context.tools.call("record_stop")
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            return (f"■ saved macro {v['steps']} steps → call it with "
                    f"/macro {v['saved']} (or ask me to run it)")
        if verb == "step":
            if len(parts) < 2:
                return "usage: /record step <tool> [json args]"
            args = " ".join(parts[2:]) if len(parts) > 2 else ""
            outcome = self.context.tools.call("record_step", tool=parts[1], args=args)
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            return f"step {outcome.value['steps']} added ({parts[1]})"
        if verb == "status":
            outcome = self.context.tools.call("record_status")
            if not outcome.ok:
                return f"record failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            if not v.get("recording"):
                return "not recording — /record start <name>"
            return (f"● recording {v['name']}: {v['steps']} steps "
                    f"({v['seconds']}s, last: {v.get('last')})")
        return "usage: /record start <name> | stop | step <tool> [json] | status"

    def _control_macro(self, tail: str, chat_key: str) -> str:
        parts = (tail or "").split(None, 1)
        if not parts or parts[0].lower() in {"list", ""}:
            outcome = self.context.tools.call("macro_list")
            if not outcome.ok:
                return f"macro failed: {getattr(outcome.error, 'message', outcome.error)}"
            macros = outcome.value["macros"]
            if not macros:
                return "no macros yet — /record start <name>, do things, /record stop"
            lines = ["macros:"]
            for m in macros:
                tools = ", ".join(m["tools"])
                lines.append(f"  {m['name']} — {m['steps']} steps ({tools}) "
                             f"[runs: {m['runs']}]")
            return "\n".join(lines)
        name = parts[0]
        overrides = parts[1].strip() if len(parts) > 1 else ""
        outcome = self.context.tools.call("macro_run", name=name, overrides=overrides)
        if not outcome.ok:
            return f"macro failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        lines = [f"▶ {v['macro']}: {v['summary']} ({v['seconds']}s)"]
        for step in v["steps"]:
            mark = "✓" if step["ok"] else "✗"
            lines.append(f"  {mark} {step['step']}: {step['tool']} — {step['result'][:120]}")
        chat = self._ref_from_key(chat_key)
        return self._send_long_checked(chat.platform, chat, "\n".join(lines))

    def _control_file(self, tail: str, chat_key: str) -> str:
        parts = (tail or "").split()
        if len(parts) < 3:
            return "usage: /file <platform> <chat_id> <path> [caption]"
        platform, chat_id, path = parts[0], parts[1], parts[2]
        caption = " ".join(parts[3:]) if len(parts) > 3 else ""
        outcome = self.context.tools.call(
            "file_send", platform=platform, chat_id=chat_id,
            path=path, caption=caption)
        if not outcome.ok:
            return f"file send failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        return (f"📎 sent {os.path.basename(v['path'])} "
                f"({v['bytes'] // 1024} KB) → {platform}:{chat_id}")

    def _control_publish(self, tail: str, chat_key: str) -> str:
        parts = (tail or "").split()
        if len(parts) < 3:
            return "usage: /publish <platform> <chat_id> <markdown file> [format]"
        platform, chat_id, path = parts[0], parts[1], parts[2]
        fmt = parts[3].lower() if len(parts) > 3 else "pdf"
        try:
            from ..tools.filesystem import safe_path

            source = safe_path(self.context, path, must_exist=True)
        except Exception as exc:  # noqa: BLE001
            return f"publish failed: cannot read {path!r} ({exc})"
        content = source.read_text(encoding="utf-8", errors="replace")
        title = source.stem.replace("-", " ").replace("_", " ").strip() or "Report"
        outcome = self.context.tools.call(
            "report_publish", content=content, title=title, platform=platform,
            chat_id=chat_id, format=fmt, name=source.stem)
        if not outcome.ok:
            return (f"publish failed: {getattr(outcome.error, 'message', outcome.error)}")
        v = outcome.value
        size = v.get("bytes", 0)
        where = f" → {platform}:{chat_id}" if v.get("sent") else " (not sent)"
        return (f"📄 published {os.path.basename(v['path'])} "
                f"({size // 1024} KB, {v['format']}){where}")

    def _control_deliver(self, tail: str, chat_key: str) -> str:
        """`/deliver report <topic> [--section "T::body"] [--to p:c] [--no-pdf]`.

        Create-and-deliver: styled HTML report + real PDF → zip → the live
        gateway's file-send path.  Default target is the chat the command
        came from; `--to platform:chat` overrides.
        """
        import shlex

        usage = ('usage: /deliver report <topic> --section "Title::body" '
                 "[--to platform:chat] [--no-pdf]")
        try:
            tokens = shlex.split(tail or "")
        except ValueError as exc:
            return f"{usage} (could not parse: {exc})"
        if not tokens or tokens[0].lower() != "report":
            return usage
        topic_parts: list[str] = []
        sections: list[tuple[str, str]] = []
        target = ""
        include_pdf = True
        i = 1
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--section" and i + 1 < len(tokens):
                raw = tokens[i + 1]
                sec_title, sep, sec_body = raw.partition("::")
                if not sep or not sec_title.strip():
                    return f"{usage} (bad --section {raw!r}; use 'Title::body')"
                sections.append((sec_title.strip(), sec_body.strip()))
                i += 2
            elif tok == "--to" and i + 1 < len(tokens):
                target = tokens[i + 1].strip()
                i += 2
            elif tok == "--no-pdf":
                include_pdf = False
                i += 1
            else:
                topic_parts.append(tok)
                i += 1
        topic = " ".join(topic_parts).strip()
        if not topic:
            return usage + " — a topic is required"
        if not sections:
            return (f"{usage} — at least one section is required, e.g.\n"
                    f'/deliver report "{topic}" '
                    '--section "Overview::The key points…"')
        ref = self._ref_from_key(chat_key)
        platform, chat_id = ref.platform, ref.chat_id
        if target:
            if ":" in target:
                platform, chat_id = (p.strip() for p in target.split(":", 1))
            else:
                chat_id = target
        try:
            from ..tools.deliver_report import deliver_report

            out = deliver_report(self.context, topic, sections, platform,
                                 chat_id, include_pdf=include_pdf)
        except Exception as exc:  # noqa: BLE001 - the reply carries the failure
            return f"deliver failed: {exc}"
        name = out["zip_path"].rsplit("/", 1)[-1]
        return (f"📄 delivered {name} → {platform}:{chat_id} "
                f"({out['zip_bytes']} B, {len(out['sections'])} sections, "
                f"message {out['message_id']})")

    def _control_evolve(self, tail: str, chat_key: str) -> str:
        from .evolution import EvolutionAgent
        from .power import power_mode_for

        parts = (tail or "").split()
        verb = parts[0].lower() if parts else ""
        agent = EvolutionAgent(self.context)
        if verb == "audit":
            started = time.time()
            report = agent.audit()
            lines = [f"🔬 audit ({time.time() - started:.1f}s): {report['summary']}"]
            for f in report["findings"][:10]:
                lines.append(f"  [{f['severity']}] {f['where']} — {f['what']}")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), "\n".join(lines))
        if verb == "research" and len(parts) >= 2:
            topic = " ".join(parts[1:])
            started = time.time()
            outcome = self.context.tools.call("evolve_research", topic=topic)
            if not outcome.ok:
                return f"research failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            lines = [f"📚 research: {v['topic']} ({time.time() - started:.1f}s)",
                     f"evidence: {v['evidence_summary']}"]
            for r in v.get("recommendations", [])[:5]:
                targets = ", ".join(r.get("targets", [])[:3])
                lines.append(f"  • {r.get('title', '?')} → {targets or 'framework'} "
                             f"(risk: {r.get('risk', '?')})")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), "\n".join(lines))
        if verb == "revert" and len(parts) >= 2:
            started = time.time()
            try:
                out = agent.revert(parts[1])
            except Exception as exc:  # noqa: BLE001
                return f"revert failed: {exc}"
            if out.get("ok"):
                return (f"↩️ evolution {parts[1]} rolled back via "
                        f"{out['method']} ({time.time() - started:.0f}s, "
                        f"test gate re-verified: {out.get('verified')})")
            return f"revert problem: {out.get('report', 'see logs')[-400:]}"
        if verb == "git":
            st = agent.git_status()
            lines = [
                f"🌿 evolution git — on {st['branch'] or '(detached)'}",
                f"  remote: {st['remote'] or '(none — local only)'}",
                f"  work branch: {st['work_branch']}",
                f"  main branch: {st['main_branch']}",
                f"  push on publish: {st['push_on_publish']}",
                f"  tree clean: {st['clean']}",
            ]
            if "ahead" in st:
                lines.append(f"  vs upstream: +{st['ahead']} / -{st['behind']}")
            lines.append("  /evolve publish [branch] [--push] to make it permanent")
            return "\n".join(lines)
        if verb == "publish":
            target = parts[1] if len(parts) > 1 and not parts[1].startswith("--") else ""
            want_push = any(p.startswith("--push") for p in parts[1:])
            started = time.time()
            try:
                out = agent.git.publish(target,
                                        push=True if want_push else None)
            except Exception as exc:  # noqa: BLE001
                return f"publish failed: {exc}"
            elapsed = time.time() - started
            if not out.get("published"):
                return f"publish: {out.get('reason', 'nothing to do')} " \
                       f"({elapsed:.0f}s)"
            push_note = ""
            if want_push:
                if out.get("pushed"):
                    push_note = " — pushed to origin"
                else:
                    push_note = (f" — push skipped: "
                                 f"{out.get('reason', 'no origin remote')}")
            return (f"🚀 evolution published to {out['target']} "
                    f"({out['method']}, {elapsed:.0f}s)"
                    f"{push_note}: {out.get('commit', '')}")
        if verb == "auto":
            steps = parts[1] if len(parts) > 1 else "3"
            started = time.time()
            outcome = self.context.tools.call("evolve_auto", steps=steps)
            if not outcome.ok:
                return f"autopilot: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            lines = [f"🤖 autopilot ({time.time() - started:.0f}s): "
                     f"{len(v['applied'])} applied, {len(v['reverted'])} reverted, "
                     f"stopped: {v['stopped_reason']}"]
            for a in v["applied"]:
                lines.append(f"  ✅ {a['source']} → {a['instruction'][:60]} "
                             f"({a.get('tag') or a.get('commit') or 'on disk'})")
            for r in v["reverted"]:
                lines.append(f"  ↩️ {r['source']} → {r['instruction'][:60]} "
                             f"(gate failed, rolled back)")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), "\n".join(lines))
        if verb == "queue":
            sub = parts[1].lower() if len(parts) > 1 else "list"
            if sub == "add" and len(parts) >= 3:
                outcome = self.context.tools.call(
                    "evolve_queue", action="add", instruction=" ".join(parts[2:]))
                if not outcome.ok:
                    return f"queue failed: {getattr(outcome.error, 'message', outcome.error)}"
                return f"🎯 queued for autopilot ({outcome.value['queued']} total): " \
                       f"{' '.join(parts[2:])[:80]}"
            outcome = self.context.tools.call("evolve_queue", action=sub)
            if not outcome.ok:
                return f"queue failed: {getattr(outcome.error, 'message', outcome.error)}"
            goals = outcome.value["goals"]
            if not goals:
                return "autopilot queue is empty — /evolve queue add <goal>"
            lines = ["autopilot queue:"]
            lines += [f"  {i + 1}. {g[:80]}" for i, g in enumerate(goals[:10])]
            return "\n".join(lines)
        if verb in {"list", ""}:
            proposals = agent.list(8)
            if not proposals:
                return "no evolution proposals yet — /evolve <what to improve>"
            lines = ["evolution proposals:"]
            for p in proposals:
                files = ", ".join(e["path"] for e in p.edits[:4]) or "—"
                lines.append(f"  {p.id} [{p.status}] {p.instruction[:70]} → {files}")
            return "\n".join(lines)
        if verb == "apply" and len(parts) >= 2:
            proposal_id = parts[1]
            commit = len(parts) > 2 and parts[2].lower() in {"commit", "1", "true", "yes"}
            started = time.time()
            try:
                out = agent.apply(proposal_id, verify=True, commit=commit)
            except Exception as exc:  # noqa: BLE001
                return f"evolve failed: {exc}"
            elapsed = time.time() - started
            if out.get("applied"):
                return (f"🧬 evolution applied to the framework ({elapsed:.0f}s, "
                        f"full test suite passed): {', '.join(out['edits'])}\n"
                        f"commit: {out.get('commit') or 'on disk (not committed)'}")
            return (f"🧬 evolution REVERTED — verification failed, the bot is "
                    f"untouched ({elapsed:.0f}s):\n{out.get('report', '')[-1200:]}")
        if verb:
            instruction = " ".join(parts)
            plan_outcome = self.context.tools.call("evolve_plan", instruction=instruction)
            if not plan_outcome.ok:
                return f"evolve failed: {getattr(plan_outcome.error, 'message', plan_outcome.error)}"
            proposal = plan_outcome.value
            lines = [
                f"🧬 evolution plan {proposal['id']} ({len(proposal['edits'])} edits):",
            ]
            for edit in proposal["edits"][:6]:
                lines.append(f"  • {edit['path']}")
            if proposal.get("rationale"):
                lines.append(f"why: {proposal['rationale'][:200]}")
            power = power_mode_for(self.context).active
            if power:
                lines.append("power mode on — applying now (full test suite first)…")
                started = time.time()
                apply_outcome = self.context.tools.call(
                    "evolve_apply", proposal_id=proposal["id"], commit="")
                if apply_outcome.ok and apply_outcome.value.get("applied"):
                    lines.append(
                        f"✅ applied, all tests passed ({time.time() - started:.0f}s) — "
                        "restart to load it")
                elif apply_outcome.ok:
                    lines.append(
                        "↩️ verification failed — REVERTED, the framework is untouched:\n"
                        + apply_outcome.value.get("report", "")[-900:])
                else:
                    lines.append(
                        f"apply failed: {getattr(apply_outcome.error, 'message', apply_outcome.error)}")
            else:
                lines.append(
                    f"NOT applied — normal mode. Approve with: "
                    f"/evolve apply {proposal['id']}   (or turn power mode on)")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), "\n".join(lines))
        return ("usage: /evolve <instruction> | /evolve apply <id> [commit] "
                "| /evolve list")

    def _control_upgrade(self, tail: str, *,
                         _chat: Any | None = None,
                         _pipeline: Any | None = None) -> str:
        """The research → approve → evolve loop, from chat.

        /upgrade list            pending proposals, one line each
        /upgrade show <id>       full ticket: problem, patch plan, risk
        /upgrade diff <id>       preview the actual patch before approving
        /upgrade approve <id>    approve → test-gated apply → what-changed digest
        /upgrade deny <id> <reason>
        /upgrade applied         recently applied, with what-changed digests

        Owner-only, enforced at two layers: ``on_message`` only routes
        slash commands into ``handle_control`` for operator chats (the
        ``_is_operator`` gate), and this method re-checks
        :func:`is_owner_chat` itself before touching the queue — a direct
        call from a non-owner chat is denied, fail-closed.  ``_chat`` is
        the originating chat (``message.chat``); ``None`` means no chat
        was proven and is denied.

        ``_pipeline`` is a test seam (a mock pipeline); production always
        uses the real UpgradePipeline.
        """
        if not is_owner_chat(_chat,
                             owner_chats=getattr(self, "_owner_chats", ())):
            return "owner-only: /upgrade is not available in this chat."

        import time

        from ..core.errors import NoMoralsError
        from .upgrade_chat import (
            UPGRADE_USAGE,
            render_applied_digest,
            render_upgrade_diff,
            render_upgrade_list,
            render_upgrade_show,
            resolve_proposal,
        )
        from .upgrade_queue import UpgradePipeline, UpgradeQueue

        parts = (tail or "").strip().split(None, 1)
        verb = (parts[0] if parts else "list").lower()
        rest = parts[1] if len(parts) > 1 else ""
        queue = UpgradeQueue(self.context)
        pipeline = (_pipeline if _pipeline is not None
                    else UpgradePipeline(self.context))

        if verb == "list":
            return render_upgrade_list(queue.list(status="proposed"))

        if verb in ("show", "diff"):
            proposal, err = resolve_proposal(queue, rest)
            if err:
                return err
            if verb == "show":
                return render_upgrade_show(proposal)
            evo = None
            plan = proposal.get("patch_plan") or {}
            evo_id = (plan.get("evolution_proposal_id")
                      if isinstance(plan, dict) else "")
            if evo_id:
                # the ticket references a planned evolution proposal — load
                # its real edits so the preview shows actual hunks. Any
                # load problem falls back to the ticket text.
                try:
                    agent = getattr(pipeline, "evolution", None)
                    evo = agent._load(str(evo_id)) if agent is not None else None
                except Exception:  # noqa: BLE001 — fallback is fine
                    evo = None
            return render_upgrade_diff(proposal, evo_proposal=evo)

        if verb == "approve":
            proposal, err = resolve_proposal(queue, rest)
            if err:
                return err
            if (proposal.get("status") or "") != "proposed":
                return (f"{proposal.get('id')} is "
                        f"'{proposal.get('status')}' — only 'proposed' "
                        "tickets can be approved.")
            started = time.time()
            try:
                done = pipeline.approve_and_implement(
                    str(proposal.get("id")), by="owner")
            except Exception as exc:  # noqa: BLE001 — already recorded failed
                return (f"❌ apply failed and was recorded as failed: "
                        f"{type(exc).__name__}: {exc}")
            digest = render_applied_digest(done)
            return f"{digest}\n  ⏱️ took {time.time() - started:.0f}s"

        if verb == "deny":
            sub = rest.split(None, 1)
            if len(sub) < 2:
                return "usage: /upgrade deny <id> <reason>"
            ref, reason = sub
            proposal, err = resolve_proposal(queue, ref)
            if err:
                return err
            try:
                denied = pipeline.deny_with_reason(
                    str(proposal.get("id")), reason, by="owner")
            except NoMoralsError as exc:
                return f"deny failed: {exc}"
            return (f"🚫 upgrade denied: {denied.get('title') or denied.get('id')}\n"
                    f"reason: {reason.strip()}")

        if verb == "applied":
            rows = queue.list(status="implemented", limit=10)
            if not rows:
                return "no upgrades applied yet."
            lines = [f"applied upgrades ({len(rows)}):"]
            lines.extend(render_applied_digest(p) for p in rows)
            return "\n\n".join(lines)

        return UPGRADE_USAGE

    def _control_data(self, tail: str, chat_key: str) -> str:
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "list"
        if verb == "mine":
            name = parts[1] if len(parts) > 1 else ""
            outcome = self.context.tools.call("train_mine", name=name)
            if not outcome.ok:
                return (f"mine failed: {getattr(outcome.error, 'message', outcome.error)}")
            v = outcome.value
            stats = v.get("stats", {})
            hist = v.get("score_histogram", {})
            text = (f"🧠 mined {v['examples']} training pairs from {stats.get('conversations', 0)} "
                    f"conversations ({stats.get('pairs_seen', 0)} pairs seen, "
                    f"noise {stats.get('noise_dropped', 0)}, dupes {stats.get('duplicates', 0)}, "
                    f"below-score {stats.get('below_score', 0)})\n"
                    f"quality: high {hist.get('high', 0)} / mid {hist.get('mid', 0)} / low {hist.get('low', 0)}\n"
                    f"bundle: {v['dir']}/{v['name']}.* "
                    f"(alpaca + sharegpt + chatml)\n"
                    f"dataset id: {v.get('dataset_id') or '(not registered)'}")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), text)
        if verb in {"list", ""}:
            outcome = self.context.tools.call("train_datasets")
            if not outcome.ok:
                return (f"data failed: {getattr(outcome.error, 'message', outcome.error)}")
            v = outcome.value
            stats = v.get("registry_stats", {})
            lines = [f"datasets: {stats.get('datasets', 0)} registered "
                     f"({stats.get('rows', 0)} rows, {stats.get('bytes', 0) // 1024} KB)"]
            for row in v.get("datasets", [])[:12]:
                lines.append(f"  {row['id']} — {row['name']} "
                             f"({row['rows']} rows, {row['bytes'] // 1024} KB)")
            mined = v.get("recently_mined", [])
            if mined:
                lines.append(f"recently mined: {mined[0]['name']} "
                             f"({mined[0]['examples'] if 'examples' in mined[0] else ''})")
            free = v.get("free_catalog", [])
            if free:
                lines.append("free datasets (fetch with /data fetch <name>):")
                for e in free[:8]:
                    lines.append(f"  {e['name']} — {e['kind']} · {e['license']}")
            return "\n".join(lines)
        if verb == "fetch" and len(parts) >= 2:
            ref = parts[1]
            max_rows = parts[2] if len(parts) > 2 else ""
            outcome = self.context.tools.call(
                "dataset_fetch", ref=ref, max_rows=max_rows)
            if not outcome.ok:
                return f"fetch failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            return (f"📦 fetched {v['rows']} rows from {v['source']} "
                    f"({v.get('license', '?')}) → {v['path']}\n"
                    f"registered as {v['name']}-fetched — nm train can use it")
        if verb == "mix" and len(parts) >= 2:
            # /data mix <name1,name2,...> [rows] — the fine-tune bundle
            sources = parts[1]
            rows = parts[2] if len(parts) > 2 else ""
            outcome = self.context.tools.call(
                "train_mix", sources=sources, rows=rows)
            if not outcome.ok:
                return f"mix failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            per_src = ", ".join(f"{n}: {d.get('rows', 0)}"
                                for n, d in v.get("per_source", {}).items())
            text = (f"🧬 persona mix ready — {v['rows']} rows "
                    f"({per_src})\n"
                    f"bundle: {v['outputs']['messages_jsonl']} "
                    f"(+ alpaca/sharegpt/chatml)\n")
            if v.get("colab_script"):
                text += (f"colab script: {v['colab_script']} — upload it + "
                         f"the .jsonl to Colab free and run (auto-resumes "
                         f"after kills)\n")
            if v.get("missing_sources"):
                text += f"not fetched (skipped): {', '.join(v['missing_sources'])}\n"
            text += ("next: /file <platform> <chat> <path> to get the files, "
                     "or nm data on the console")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), text)
        return ("usage: /data mine [name] | /data list | /data fetch <name> [rows] "
                "| /data mix <name1,name2> [rows]")

    def _control_speak(self, tail: str, chat_key: str) -> str:
        """`/speak <text>` — I say it: neural TTS voice note sent back here."""
        text = (tail or "").strip()
        if not text:
            return ("usage: /speak <text> — I'll say it as a voice note "
                    "(tags like [happy] [whisper] [laughs] [pause:300] work)")
        # bridge the current mood into the voice, best-effort
        mood, mood_level = "", 5
        mood_box = self.context.extras.get("mood")
        if mood_box is not None:
            mood = str(getattr(mood_box, "mood", "") or
                       (mood_box.get("mood") if isinstance(mood_box, dict) else "") or "")
            try:
                mood_level = int(getattr(mood_box, "level", 5) or 5)
            except (TypeError, ValueError):
                mood_level = 5
        outcome = self.context.tools.call(
            "tts_say", text=text, mood=mood, mood_level=str(mood_level))
        if not outcome.ok:
            return f"speak failed: {getattr(outcome.error, 'message', outcome.error)}"
        v = outcome.value
        ref = self._ref_from_key(chat_key)
        gateway = self.context.extras.get("gateway")
        sent = False
        if gateway is not None and ref.platform in getattr(gateway, "adapters", {}):
            try:
                # Caption ceiling is ~1024 chars on the platforms; the full
                # transcript rides beside the voice note so it's readable
                # and searchable in chat.
                result = gateway.send_file(ref.platform, ref, v["path"],
                                           caption=text[:1024])
                sent = bool(getattr(result, "ok", False))
                if sent and len(text) > 1024:
                    try:
                        gateway.send(ref.platform, ref, text)
                    except Exception:  # noqa: BLE001 - transcript is a bonus
                        _log.debug("voice-note transcript send failed",
                                   exc_info=True)
            except Exception:  # noqa: BLE001 - report the file, don't crash
                sent = False
        backend = v.get("backend", "?")
        return self._deliver_voice_note(chat_key, v["path"], text, backend,
                                        size_kb=v["bytes"] // 1024)

    def _deliver_voice_note(self, chat_key: str, path: str, text: str,
                            backend: str, size_kb: int = 0) -> str:
        """Send a wav back into the chat as a voice note; reply text."""
        ref = self._ref_from_key(chat_key)
        gateway = self.context.extras.get("gateway")
        sent = False
        if gateway is not None and ref.platform in getattr(gateway, "adapters", {}):
            try:
                # Caption ceiling is ~1024 chars on the platforms; the full
                # transcript rides beside the voice note so it's readable
                # and searchable in chat.
                result = gateway.send_file(ref.platform, ref, path,
                                           caption=text[:1024])
                sent = bool(getattr(result, "ok", False))
                if sent and len(text) > 1024:
                    try:
                        gateway.send(ref.platform, ref, text)
                    except Exception:  # noqa: BLE001 - transcript is a bonus
                        _log.debug("voice-note transcript send failed",
                                   exc_info=True)
            except Exception:  # noqa: BLE001 - report the file, don't crash
                sent = False
        if not size_kb and os.path.exists(path):
            size_kb = os.path.getsize(path) // 1024
        if sent:
            return f"🎙️ said it ({backend}, {size_kb} KB voice note)"
        return (f"🎙️ voice note ready ({backend}, {size_kb} KB): {path}")

    # ── voice catalogue: /voice (Telegram / WhatsApp / console) ─────────────
    def _control_voice(self, tail: str, chat_key: str,
                       message: Any = None) -> str:
        """The voice catalogue, live from any chat.

        /voice list                              catalogue + active voice
        /voice use <name>                        switch this chat's voice now
        /voice say <text>                        speak as the chat's voice
        /voice clone <name> [path]               clone a voice note / file
        /voice transcript <name> <text>          set a clone's prompt text
        /voice describe <name> <text>            describe a catalogue voice
        /voice backend <name> <backend>          change a voice's backend
        /voice rm <name>                         drop a catalogue voice
        """
        from ..voice.catalogue import default_catalogue

        cat = default_catalogue()
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        if verb in ("list", "ls"):
            voices = cat.list()
            if not voices:
                return ("no voices in the catalogue yet — clone one:\n"
                        "/voice clone <name> (attach a voice note to the "
                        "command message, or give a file path)")
            lines = ["voices:"]
            for v in voices:
                mark = "●" if v["active"] else "○"
                chat_mark = (" [this chat]"
                             if cat.chat_overrides.get(chat_key) == v["name"]
                             else "")
                desc = f" — {v['description']}" if v["description"] else ""
                prof = f" (profile: {v['profile']})" if v["profile"] else ""
                lines.append(f"  {mark} {v['name']} [{v['backend']}]"
                             f"{prof}{chat_mark}{desc}")
            return "\n".join(lines)

        if verb == "use":
            name = rest.strip()
            if not name:
                return "usage: /voice use <name>"
            try:
                cat.set_chat_voice(chat_key, name)
            except KeyError:
                return f"unknown voice {name!r} — /voice list"
            return f"🎙️ this chat now speaks as '{name}'"

        if verb == "say":
            text = rest.strip()
            if not text:
                return "usage: /voice say <text>"
            mood = self._voice_mood()
            try:
                out = cat.speak(text, chat_key=chat_key, mood=mood or "neutral")
            except Exception as exc:  # noqa: BLE001
                return f"voice say failed: {exc}"
            return self._deliver_voice_note(chat_key, out["path"], text,
                                            out.get("backend", "?"))

        if verb == "clone":
            cparts = rest.strip().split(None, 1)
            if not cparts:
                return ("usage: /voice clone <name> [audio path] — attach a "
                        "voice note to this message or pass a file path")
            name, path = cparts[0], (cparts[1] if len(cparts) > 1 else "")
            if not path and message is not None:
                for media in getattr(message, "media", None) or []:
                    if getattr(media, "kind", "") in ("audio", "voice"):
                        path = getattr(media, "path", "")
                        break
            if not path or not os.path.exists(path):
                return ("no audio found — attach a voice note to the /voice "
                        "clone message or pass a file path")
            transcript = ""
            try:
                outcome = self.context.tools.call("transcribe", path=path)
                if outcome.ok and outcome.value:
                    transcript = outcome.value.get("text", "") or ""
            except Exception:  # noqa: BLE001 - transcript is a bonus
                transcript = ""
            try:
                voice = cat.clone(name, path, transcript=transcript,
                                  consent_confirmed=True,
                                  description="cloned by owner via /voice clone")
            except Exception as exc:  # noqa: BLE001
                return f"clone failed: {exc}"
            note = (f" (transcript: {transcript[:80]}…)"
                    if transcript else " (no transcript — set one with "
                    "/voice transcript <name> <text>)")
            return (f"🎙️ cloned '{voice.name}' from your voice note{note}\n"
                    f"say something: /voice say hello there")

        if verb == "transcript":
            tparts = rest.strip().split(None, 1)
            if len(tparts) < 2:
                return "usage: /voice transcript <name> <text>"
            name, text = tparts
            profile = cat.library.get(name)
            if profile is None:
                return f"unknown voice {name!r} — /voice list"
            profile.prompt_text = text.strip()
            cat.library._save_index()
            return f"transcript set for '{name}' ({len(text)} chars)"

        if verb == "describe":
            dparts = rest.strip().split(None, 1)
            if len(dparts) < 2:
                return "usage: /voice describe <name> <text>"
            voice = cat.get(dparts[0])
            if voice is None:
                return f"unknown voice {dparts[0]!r} — /voice list"
            voice.description = dparts[1].strip()
            cat._save()
            return f"description set for '{voice.name}'"

        if verb == "backend":
            bparts = rest.strip().split(None, 1)
            if len(bparts) < 2:
                return "usage: /voice backend <name> <backend>"
            voice = cat.get(bparts[0])
            if voice is None:
                return f"unknown voice {bparts[0]!r} — /voice list"
            voice.backend = bparts[1].strip()
            cat._save()
            return f"'{voice.name}' now prefers backend '{voice.backend}'"

        if verb in ("rm", "remove", "delete"):
            name = rest.strip()
            if not name:
                return "usage: /voice rm <name>"
            if not cat.remove(name):
                return f"unknown voice {name!r} — /voice list"
            return f"removed '{name}' from the catalogue"

        return ("usage: /voice list | use <name> | say <text> | clone <name> "
                "[path] | transcript <name> <text> | describe <name> <text> | "
                "backend <name> <backend> | rm <name>")

    def _voice_mood(self) -> str:
        """Best-effort current mood label for voice performances."""
        mood_box = self.context.extras.get("mood")
        if mood_box is None:
            return ""
        return str(getattr(mood_box, "mood", "") or
                   (mood_box.get("mood") if isinstance(mood_box, dict)
                    else "") or "")

    # ── devon: the autonomous dev & investigation agent ──────────────────────
    def _control_devon(self, tail: str, chat_key: str) -> str:
        """Owner types ``/devon <free text>``: Devon plans the tools that fit
        the question, runs them within a budget, journals every step into his
        own memory box, and digests the findings into a plain-English reply.

        With no task, it's a status line: what he's on and his last digests.
        """
        from .devon import DevonAgent

        task = (tail or "").strip()
        agent = DevonAgent(self.context, brain=self.brain, gateway=self.gateway)
        if not task:
            digests = agent.recent_digests(3)
            if not digests:
                return (
                    "devon: give me a task — e.g. /devon check if the brain replied to the last messages\n"
                    "tools: read/grep code, git status+diff, live db, message flow, turn traces, "
                    "run tests, log tail, mood, chat+game stats, model chain health, config snapshot, "
                    "his own history, web research, scheduled watches, background missions.\n"
                    "every run is journaled in his memory box, so he remembers what he already checked."
                )
            lines = ["devon — recent digests:"]
            for d in digests:
                lines.append(f"  · {d['task'][:60]} → {d['digest'][:160]}")
            return "\n".join(lines)

        result = agent.run(task, chat_key=chat_key)
        tools = ", ".join(dict.fromkeys(s.tool for s in result.steps)) or "none"
        header = f"🔧 devon (planned by {result.planned_by}, {result.seconds:.0f}s · tools: {tools})"
        # wave 68: when the plan degraded, say WHY — an honest
        # "planned by heuristic (model down)" beats a silent fallback.
        if result.planned_by == "heuristic" and result.plan_error:
            header += f" — {result.plan_error[:90]}"
        if result.plan_error:
            # Persist EVERY plan degradation (heuristic AND reasoning-engine
            # fallback) so `nm mind` shows the last plan_error with a
            # timestamp (wave E router telemetry). Goes through the CoreMind
            # helper built for exactly this — never raises, never breaks
            # the reply.
            try:
                self.mind.record_plan_error(result.plan_error, route="devon")
            except Exception:  # noqa: BLE001 - telemetry never breaks a reply
                _log.debug("devon plan-error telemetry failed", exc_info=True)
        text = f"{header}\n{result.digest}"
        if len(text) > 1800:
            chat = self._ref_from_key(chat_key)
            return self._send_long_checked(chat.platform, chat, text)  # report already delivered in chunks
        return text

    # ── reasoning: /think — explicit, auditable multi-step thought ──────────
    def _control_think(self, tail: str, chat_key: str) -> str:
        """Owner types ``/think <question> [strategy]``: the reasoning
        engine works it out with an explicit trace — plan, subgoals,
        actions, observations, critiques, verdict — and answers with the
        full work shown, not just the conclusion."""
        from .reasoning import ReasoningEngine, trace_text

        parts = (tail or "").strip().split(None, 1)
        if not parts or not parts[0].strip():
            return ("usage: /think <question> [strategy]\n"
                    "strategies: auto (default) · cot · decompose · "
                    "hypothesize · critique · tree\n"
                    "it shows the work — every step, then the answer.")
        question = parts[0].strip()
        strategy = parts[1].strip() if len(parts) > 1 and parts[1].strip() \
            else "auto"
        if strategy not in {"auto", "cot", "decompose", "hypothesize",
                            "critique", "tree"}:
            return (f"unknown strategy {strategy!r} — use auto, cot, "
                    "decompose, hypothesize, critique, or tree")
        engine = ReasoningEngine(self.context, max_llm_calls=14,
                                 max_seconds=120.0)
        try:
            result = engine.reason(question, strategy=strategy)
        except Exception as exc:  # noqa: BLE001 — never let a bad run kill chat
            return f"reasoning failed: {type(exc).__name__}: {exc}"
        text = trace_text(result)
        if len(text) > 3500:
            text = text[:3480] + "\n…(trace trimmed)"
        if len(text) > 1800:
            chat = self._ref_from_key(chat_key)
            return self._send_long_checked(chat.platform, chat, text)  # report already delivered in chunks
        return text

    # ── /benchmark — how sharp is the system right now ──────────────────────
    def _control_benchmark(self, tail: str) -> str:
        """Owner types ``/benchmark [dimension]``: runs the agent benchmark
        (1 task per dimension for speed in-chat) and reports the score."""
        from .benchmark import run_benchmark

        dim = (tail or "").strip()
        dims = [dim] if dim in {"reasoning", "planning", "tool_use",
                                "self_correction"} else None
        if tail and not dims:
            return "dimensions: reasoning, planning, tool_use, self_correction"
        report = run_benchmark(self.context, dimensions=dims, limit=1)
        if not report.measurable:
            return (f"can't benchmark with the {report.provider!r} provider "
                    f"(mock/offline) — switch to a real model and ask again")
        overall = report.overall if report.overall is not None else 0.0
        lines = [f"benchmark: {overall:.2f} ({report.provider}, "
                 f"{report.seconds:.0f}s)"]
        for name, d in report.scores.items():
            score = "n/a" if d.score is None else f"{d.score:.0%}"
            lines.append(f"  {name}: {score} ({d.passed}/{d.total})")
        return "\n".join(lines)

    # ── weather awareness + USA situations + timezones ───────────────────
    def _control_weather(self, tail: str) -> str:
        """/weather [place] | usa | forecast <place> | alerts <place> — keyless live weather."""
        from .weather import Weather, USASituations, owner_tz, tz_note

        w = Weather()
        sub = (tail or "").strip()
        if not sub or sub.lower() == "now":
            return w.now()["text"]
        low = sub.lower()
        if low == "usa":
            return USASituations(w).overview()["text"]
        if low.startswith("forecast"):
            place = sub[8:].strip() or None
            return w.forecast(place, days=3)["text"]
        if low.startswith("alerts"):
            place = sub[6:].strip() or None
            res = w.now(place)
            alerts = res.get("alerts") or []
            if not alerts:
                return f"no active alerts — {res['text']}"
            return "\n".join(f"• [{a.get('severity','?')}] {a.get('title','')}"
                             for a in alerts[:8])
        if low.startswith("tz"):
            return tz_note(owner_tz())
        return w.now(sub)["text"]

    def _control_tz(self, tail: str) -> str:
        """/tz [YYYY-MM-DD HH:MM [from-zone] [to-zone]] — convert time zones."""
        from .weather import convert_time, owner_tz, tz_note

        toks = (tail or "").strip().split()
        if not toks:
            return tz_note(owner_tz())
        # optional "YYYY-MM-DD HH:MM" glued at the front
        raw = " ".join(toks[:2]) if len(toks) > 1 and toks[1].count(":") else toks[0]
        rest = toks[2:] if len(toks) > 1 and toks[1].count(":") else toks[1:]
        from_zone = rest[0] if len(rest) > 0 else owner_tz()
        to_zone = rest[1] if len(rest) > 1 else owner_tz()
        res = convert_time(raw, from_zone, to_zone)
        return res.get("text") or f"couldn't parse that: {res.get('error','')}"

    # ── sports bet analyst: /bet (analysis only — never places bets) ─────────
    def _control_bet(self, tail: str, chat_key: str) -> str:
        """The ensemble-ML sports bet analyst.

        /bet analyze <home> vs <away> [h d a] [--league L]
        /bet bankroll [set <amount>]
        /bet backtest [n] [--seed S]
        /bet record <home> <away> <hg>-<ag> [--league L]
        """
        from .sports_bet import (BetStore, Fixture, OddsSnapshot, backtest,
                                 render_analysis, render_backtest,
                                 synthetic_history)

        store = BetStore()
        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        def _usage() -> str:
            return (
                "/bet analyze <home> vs <away> [home_odds draw_odds away_odds] "
                "[--league L]\n"
                "/bet bankroll [set <amount>]  — the paper bankroll\n"
                "/bet backtest [n] [--seed S]  — walk-forward backtest on "
                "synthetic history\n"
                "/bet record <home> <away> <hg>-<ag> [--league L]  — feed a "
                "result back in\n"
                f"bankroll: {store.bankroll:.2f}")

        if verb in ("", "help"):
            return _usage()

        if verb == "bankroll":
            if rest.lower().startswith("set"):
                try:
                    amt = float(rest.split()[1])
                except (IndexError, ValueError):
                    return "usage: /bet bankroll set <amount>"
                store.bankroll = amt
                store.save()
                return f"bankroll set to {amt:.2f}"
            return f"paper bankroll: {store.bankroll:.2f}"

        if verb == "backtest":
            n = 400
            seed = 7
            toks = rest.split()
            i = 0
            while i < len(toks):
                t = toks[i]
                if t == "--seed" and i + 1 < len(toks):
                    try:
                        seed = int(toks[i + 1])
                    except ValueError:  # noqa: E103 - unparseable --seed keeps default 7
                        pass
                    i += 2
                    continue
                if t.startswith("--seed="):
                    try:
                        seed = int(t.split("=", 1)[1])
                    except ValueError:  # noqa: E103 - unparseable --seed keeps default 7
                        pass
                elif t.isdigit():
                    n = int(t)
                i += 1
            entries = synthetic_history(n=n, seed=seed)
            r = backtest(entries, bankroll=store.bankroll, seed=seed)
            return render_backtest(r)

        if verb == "record":
            toks = rest.split()
            if len(toks) < 3:
                return "usage: /bet record <home> <away> <hg>-<ag> [--league L]"
            home, away = toks[0], toks[1]
            try:
                hg_s, ag_s = toks[2].split("-")
                hg, ag = int(hg_s), int(ag_s)
            except ValueError:
                return "usage: /bet record <home> <away> <hg>-<ag> [--league L]"
            league = "GEN"
            if "--league" in toks:
                try:
                    league = toks[toks.index("--league") + 1]
                except IndexError:  # noqa: E103 - missing --league value keeps default GEN
                    pass
            store.record(Fixture(home=home, away=away, league=league,
                                 home_goals=hg, away_goals=ag))
            elo_h = store.analyst.elo.rating(home)
            elo_a = store.analyst.elo.rating(away)
            return (f"recorded: {home} {hg}-{ag} {away}  "
                    f"(elo {elo_h:.0f} / {elo_a:.0f})")

        if verb == "analyze":
            league = "GEN"
            if "--league" in rest:
                toks = rest.split()
                try:
                    league = toks[toks.index("--league") + 1]
                except IndexError:  # noqa: E103 - missing --league value keeps default GEN
                    pass
                rest = " ".join(t for i, t in enumerate(toks)
                                if t != "--league" and
                                (i == 0 or toks[i - 1] != "--league"))
            import re as _re
            m = _re.split(r"\s+vs\.?\s+", rest, maxsplit=1, flags=_re.I)
            if len(m) < 2:
                return ("usage: /bet analyze <home> vs <away> "
                        "[home_odds draw_odds away_odds]")
            home = m[0].strip()
            tail2 = m[1].strip().split()
            away_parts: list = []
            odds: list = []
            for t in tail2:
                try:
                    odds.append(float(t))
                except ValueError:
                    if odds:
                        break  # odds started, team name done
                    away_parts.append(t)
                if len(odds) == 3:
                    break
            away = " ".join(away_parts).strip()
            if not home or not away:
                return ("usage: /bet analyze <home> vs <away> "
                        "[home_odds draw_odds away_odds]")
            snaps = []
            if len(odds) == 3:
                snaps = [OddsSnapshot(bookmaker="chat", home=odds[0],
                                      draw=odds[1], away=odds[2])]
            elif odds:
                return "give all three odds (home draw away) or none"
            a = store.analyst.analyze(
                home, away, league=league, odds=snaps,
                fixtures=store.fixtures())
            return render_analysis(a)

        return f"unknown /bet verb {verb!r}\n{_usage()}"

    # ── finance: the native-TA FinancialExpert ─────────────────────────────
    def _control_finance(self, tail: str, chat_key: str) -> str:
        """Conversational finance over free market data.

        /finance quote <symbol> [market]
        /finance analyze <symbol> [market] [timeframe]
        /finance signal <symbol> [market]
        /finance idea <symbol> [market] [--profile default|aggressive|conservative]
        /finance backtest <symbol> [market] [--profile P] [--strategy NAME]
        /finance strategies — the native strategy zoo
        /finance compare <sym1,sym2,..> [market]
        /finance watch <symbol> <above|below> <price> [market]
        /finance doctor
        """
        from .financial_expert import FinancialExpert
        from ..integrations import sentinel_bridge as bridge

        parts = (tail or "").strip().split(None, 1)
        verb = parts[0].lower() if parts else ""
        rest = parts[1] if len(parts) > 1 else ""

        def _usage() -> str:
            return (
                "/finance quote <symbol> [market] — spot price, no engine\n"
                "/finance analyze <symbol> [market] [timeframe] — regime + bias\n"
                "/finance signal <symbol> [market] — directional call\n"
                "/finance idea <symbol> [market] [--profile P] — full trade plan\n"
                "/finance backtest <symbol> [market] [--profile P] [--strategy NAME]\n"
                "/finance strategies — list the strategy zoo\n"
                "/finance compare <s1,s2,..> [market]\n"
                "/finance watch <symbol> <above|below> <price> [market] — price alert\n"
                "/finance doctor — integration health\n"
                "markets: crypto (default) | forex | stocks")

        if verb in ("", "help"):
            return _usage()

        def _market(args: list[str], default: str = "crypto") -> tuple[str, list[str]]:
            if args and args[-1].lower() in ("crypto", "forex", "stocks"):
                return args[-1].lower(), args[:-1]
            return default, args

        try:
            if verb == "doctor":
                return bridge.doctor().summary_text()

            if verb == "strategies":
                rows = FinancialExpert(self.context).strategies()
                lines = ["strategy zoo (native, no submodule needed):"]
                for r in rows:
                    params = ", ".join(f"{k}={v}"
                                       for k, v in r["params"].items())
                    lines.append(f"  {r['name']:15s} [{r['kind']:9s}] "
                                 f"{r['blurb']} ({params})")
                return "\n".join(lines)

            if verb == "quote":
                toks = rest.split()
                if not toks:
                    return "usage: /finance quote <symbol> [market]"
                market, toks = _market(toks)
                q = FinancialExpert(self.context).quote(toks[0], market)
                if not q:
                    return f"no quote for {toks[0]} [{market}]"
                chg = q.get("change_pct_24h")
                chg_s = f" ({chg:+.2f}% 24h)" if isinstance(chg, (int, float)) else ""
                return (f"{q['symbol']} {q['price']:,.2f} "
                        f"{q.get('currency', '')}{chg_s} — via {q['source']}")

            if verb in ("analyze", "signal", "backtest"):
                toks = rest.split()
                if not toks:
                    return f"usage: /finance {verb} <symbol> [market] [timeframe]"
                market, toks = _market(toks)
                symbol = toks[0]
                expert = FinancialExpert(self.context)
                if verb == "analyze":
                    tf = toks[1] if len(toks) > 1 else "1h"
                    return expert.analyze(symbol, market, timeframe=tf).summary_text()
                if verb == "signal":
                    return expert.signal(symbol, market).summary_text()
                profile = "default"
                strategy = None
                for flag, slot in (("--profile", "profile"),
                                   ("--strategy", "strategy")):
                    if flag in toks:
                        try:
                            val = toks[toks.index(flag) + 1]
                        except IndexError:
                            val = None
                        if slot == "profile":
                            profile = val or profile
                        else:
                            strategy = val
                return expert.backtest(symbol, market, strategy=strategy,
                                       profile=profile).summary_text()

            if verb == "idea":
                toks = rest.split()
                if not toks:
                    return "usage: /finance idea <symbol> [market] [--profile P]"
                profile = "default"
                if "--profile" in toks:
                    i = toks.index("--profile")
                    try:
                        profile = toks[i + 1]
                    except IndexError:  # noqa: E103 - missing --profile value keeps default
                        pass
                    toks = toks[:i] + toks[i + 2:]
                market, toks = _market(toks)
                if not toks:
                    return "usage: /finance idea <symbol> [market] [--profile P]"
                return FinancialExpert(self.context).trade_idea(
                    toks[0], market, profile=profile).summary_text()

            if verb == "compare":
                toks = rest.split()
                if not toks:
                    return "usage: /finance compare <s1,s2,..> [market]"
                market, toks = _market(toks)
                symbols = [s.strip() for s in " ".join(toks).split(",") if s.strip()]
                if not symbols:
                    return "usage: /finance compare <s1,s2,..> [market]"
                return FinancialExpert(self.context).compare(symbols, market).summary_text()

            if verb == "watch":
                toks = rest.split()
                # /finance watch BTC below 60000 [crypto]
                if len(toks) < 3:
                    return "usage: /finance watch <symbol> <above|below> <price> [market]"
                market, toks = _market(toks)
                symbol, direction, price_s = toks[0], toks[1].lower(), toks[2]
                if direction not in ("above", "below"):
                    return "usage: /finance watch <symbol> <above|below> <price> [market]"
                try:
                    price = float(price_s.replace(",", ""))
                except ValueError:
                    return f"not a price: {price_s!r}"
                op = "gt" if direction == "above" else "lt"
                res = FinancialExpert(self.context).watch_price(
                    symbol, market,
                    condition={"op": op, "field": "value", "value": price},
                    name=f"{symbol.upper()} {direction} {price:,.4g}")
                if not res.get("ok"):
                    return f"couldn't create the alert: {res}"
                w = res["watcher"]
                return (f"watching {symbol.upper()} [{market}] — alert when "
                        f"price goes {direction} {price:,.4g} "
                        f"(watcher {w.get('id')}). {res.get('echo', '')}".strip())
        except bridge.SentinelUnavailable as exc:
            return str(exc)
        except bridge.SentinelError as exc:
            return f"finance error: {exc}"
        except Exception as exc:  # noqa: BLE001 - chat must never traceback
            return f"finance error: {exc}"
        return f"unknown /finance verb {verb!r}\n{_usage()}"

    # ── directives (direct instructions to the core) ─────────────────────────
    def _control_task(self, tail: str, chat_key: str) -> str:
        from .directives import DirectivesAgent
        from .notifier import Notifier

        agent = DirectivesAgent(self.context, notifier=Notifier(self.context, self.gateway))
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "list"
        if verb == "add":
            text = " ".join(parts[1:]).strip()
            if not text:
                return "usage: /task add <instruction>"
            # wave 68 (F2): substantial task instructions pass through the
            # structuring sub-agent — the stored directive becomes the
            # god-tier spec (objective, steps, done-means), not the raw line.
            # wave 68 (F2): substantial task instructions pass through the
            # structuring sub-agent — the queued directive becomes the
            # god-tier spec (objective, steps, done-means), not the raw line.
            queued, brief_note = text, ""
            try:
                from .brief import BriefAgent, should_brief
                if should_brief(text):
                    brief = BriefAgent(self.context).refine(text, kind="task")
                    if brief.by == "model" and brief.objective:
                        queued = brief.as_goal()
                        brief_note = f" (structured: {len(brief.steps)} steps)"
                    elif brief.by == "heuristic":
                        brief_note = " (structured heuristically)"
            except Exception:  # noqa: BLE001 - structuring must never block the queue
                pass
            result = agent.add(queued)
            if not result.get("ok"):
                return f"could not add: {result.get('error')}"
            return (f"queued {result['id'][:8]}: “{text[:70]}”{brief_note} — "
                    f"/task run to execute now")
        if verb == "run":
            did = parts[1] if len(parts) > 1 else ""
            chat = self._ref_from_key(chat_key)
            try:
                self.gateway.send(chat.platform, chat, "⏳ executing directive…")
            except Exception:  # noqa: BLE001
                pass
            result = agent.run(did)
            if not result.get("ok"):
                return f"task: {result.get('error')}"
            text = f"✅ {result['id'][:8]} done:\n{str(result.get('result', ''))[:3000]}"
            return self._send_long_checked(chat.platform, chat, text)
        if verb == "list":
            return agent.format_list(agent.list(15))
        return "usage: /task add <instruction> | /task run [id] | /task list"

    # ── core mind (wave 87) ──────────────────────────────────────────────────
    def _control_mind(self, tail: str, chat_key: str) -> str:
        """/mind — the manual override over the Core Mind."""
        tail = (tail or "").strip()
        if tail in ("", "status"):
            return self.mind.status()
        if tail == "clear":
            n = self.mind.clear(chat_key)
            return f"mind: {n} pending clarification(s) cleared."
        if tail == "pending":
            pend = self.mind.pending()
            if not pend:
                return "no open clarifications."
            lines = []
            for key, p in pend.items():
                lines.append(f"  {key}: “{p.get('question', '')[:80]}”")
            return "open clarifications:\n" + "\n".join(lines)
        # /mind <goal> — force the routing and show the decision
        intent = self.mind.decide(tail, live_game=self.mind._live_game(chat_key))
        if intent.kind == "chat":
            return (f"the mind reads no goal in “{tail[:60]}” — “{tail[:30]}” "
                    f"stays conversation. Try a verb: research/build/browse/"
                    f"download/mission: …")
        if intent.action == "ask":
            question = self.mind._question_for(intent)
            self.mind._set_pending(chat_key, intent, question)
            return f"decision: {intent.kind} (ask) — {intent.why}\n{question}"
        reply = self.mind._dispatch(intent, chat_key, self._fake_message(chat_key))
        return f"decision: {intent.kind} (route {intent.route}) — {intent.why}\n" + \
            (reply or "(no reply)")

    @staticmethod
    def _fake_message(chat_key: str) -> Any:
        """A minimal message shell for console-style /mind dispatch."""
        from dataclasses import dataclass, field

        @dataclass
        class _Msg:
            text: str = ""
            sender: str = "console"
            chat: Any = None

        @dataclass
        class _Chat:
            key: str
            platform: str = "local"
            kind: str = "dm"

        return _Msg(chat=_Chat(key=chat_key))

    # ── notifications ────────────────────────────────────────────────────────
    def _control_notify(self, arg: str) -> str:
        from .notifier import Notifier

        notifier = Notifier(self.context, self.gateway)
        limit = int(arg) if arg.isdigit() else 10
        rows = notifier.recent(limit)
        if not rows:
            return "no notifications yet — arena builds, research, news and tasks land here."
        marks = {"sent": "✓", "failed": "✗", "pending": "…",
                 "held-quiet-hours": "⏸", "disabled": "⊘",
                 "muted": "⊘", "deduped": "⤺"}
        lines = [f"notifications ({len(rows)}):"]
        for row in rows:
            when = time.strftime("%m-%d %H:%M", time.localtime(row.get("created_at", 0)))
            state = row.get("delivery_state") or (
                "sent" if row.get("delivered") else "pending")
            mark = marks.get(state, "?")
            body = str(row.get("body", "")).replace("\n", " ")[:80]
            lines.append(f"  {when} [{row.get('kind')}] {mark} {state} "
                         f"{row.get('title', '')[:50]}")
            if body:
                lines.append(f"      {body}")
        lines.append("states: ✓sent ✗failed …pending ⏸held-quiet-hours "
                     "⊘disabled/muted ⤺deduped — see `nm briefing status`")
        return "\n".join(lines)

    def _control_proactive(self, arg: str) -> str:
        """Owner-only: show the proactive push-send switches and the
        delivery states of recent proactive sends (briefing + watcher
        alerts).  Proactive messages go to the owner's DMs only — this
        command just reports; toggles are env vars (see /help)."""
        from .morning_briefing import proactive_status

        payload = proactive_status(self.context)
        s = payload["settings"]
        marks = {"sent": "✓", "failed": "✗", "pending": "…",
                 "held-quiet-hours": "⏸", "disabled": "⊘",
                 "muted": "⊘", "deduped": "⤺"}
        lines = [
            "she speaks first — proactive push sends (owner DMs only):",
            f"  master:   {'ON' if s['proactive_enabled'] else 'OFF'}"
            "  (NM_PARTNER_PROACTIVE_ENABLED=0 silences everything)",
            f"  briefing: {'ON' if s['proactive_briefing'] else 'OFF'}"
            f"  daily {s['briefing_time']} ({s['timezone']})",
            f"  watchers: {'ON' if s['proactive_watchers'] else 'OFF'}",
            f"  quiet hours: {s['quiet_hours']} — watcher alerts hold, "
            "the scheduled briefing still goes out",
        ]
        recent = payload["recent"]
        counts = payload.get("counts") or {}
        if counts:
            agg = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            lines.append(f"  last 24h: {agg}")
        health = payload.get("health") or {}
        for reason in health.get("degraded", []):
            lines.append(f"  ⚠️ degraded: {reason}")
        if not recent:
            lines.append("no proactive sends recorded yet")
        else:
            lines.append("recent sends:")
            for r in recent:
                when = time.strftime(
                    "%m-%d %H:%M", time.localtime(r.get("created_at", 0)))
                state = r.get("delivery_state", "?")
                lines.append(f"  {when} [{r.get('kind')}] "
                             f"{marks.get(state, '?')} {state} — "
                             f"{r.get('title', '')[:60]}")
        return "\n".join(lines)

    # ── missions: chat-visible progress ──────────────────────────────────────
    def _control_mission(self, tail: str, chat_key: str = "") -> str:
        """Mission progress and control, from persisted state.

        /mission status [id|name] — % complete, current step, ETA, stall reason
        /mission list             — active missions at a glance
        /mission stall <id> <code> <message> — record a concrete blocker
        /mission clear <id>       — drop the stall record (progress resumed)
        /mission pause <id>       — freeze it (status → paused)
        /mission resume <id>      — continue it in the background
        /mission cancel <id> [reason] — stop it for good (terminal)
        /mission retry <id>       — fresh attempt: cancel + requeue the goal
        /mission watch <id>       — this chat gets milestone updates
        /mission unwatch <id>     — stop milestone updates in this chat
        /mission new <template> <args> — create from a template
            (research <topic> | build <what> | fix <target>)
        """
        from ..core.errors import AmbiguousRef, NoMoralsError
        from ..missions import (
            MissionRunner,
            MissionStatus,
            MissionStore,
            MissionWatchers,
            StallCode,
            render_status_text,
        )

        store = MissionStore(self.context.db)

        def _resolve(ref: str) -> tuple[Any, str]:
            """_find_mission with ambiguity surfaced as a chat-ready reply.

            Returns ``(mission, "")`` on success, ``(None, "")`` when
            nothing matches, and ``(None, message)`` when the reference is
            ambiguous — the message lists the candidates instead of the
            resolver guessing one.
            """
            try:
                return self._find_mission(store, ref), ""
            except AmbiguousRef as exc:
                return None, str(exc)

        usage = ("usage: /mission status [id|name] | /mission list | "
                 "/mission stall <id> <code> <message> | /mission clear <id> | "
                 "/mission pause <id> | /mission resume <id> | "
                 "/mission cancel <id> [reason] | /mission retry <id> | "
                 "/mission watch <id> | /mission unwatch <id> | "
                 "/mission new <template> <args>\n"
                 f"stall codes: {', '.join(sorted(StallCode.ALL))}; "
                 "templates: research <topic> | build <what> | fix <target>")
        parts = (tail or "").strip().split(None, 1)
        verb = (parts[0] if parts else "status").lower()
        rest = parts[1] if len(parts) > 1 else ""

        if verb == "list":
            rows = store.resumable()
            if not rows:
                return "no active missions."
            lines = [f"missions ({len(rows)} active):"]
            for m in rows[:15]:
                # reconcile on read: a dead runner must not show as running
                store.reconcile(m.id)
                m = store.get(m.id)
                p = store.progress(m.id)
                stall = m.state.get("stall")
                line = (f"  · {m.name} [{m.status}] — "
                        f"{p['steps_done']}/{p['total_steps']} steps "
                        f"({p['percent']:.0f}%)")
                if stall:
                    code = stall.get("code")
                    label = StallCode.LABELS.get(code, code)
                    line += f" — ⚠️ stalled [{code}]: {label}"
                lines.append(line)
            return "\n".join(lines)

        if verb == "status":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return (f"no mission matching {rest!r} — "
                        "/mission list to see the active ones.")
            # reconcile on read: a dead runner must not report "running"
            rec = store.reconcile(mission.id)
            text = render_status_text(store.detail(mission.id))
            if rec["changed"]:
                text = (f"⚠️ reconciled: stale 'running' → {rec['status']}: "
                        f"{rec['reason']}\n{text}")
            return text

        if verb == "stall":
            sub = rest.split(None, 2)
            if len(sub) < 3:
                return ("usage: /mission stall <id|name> <code> <message>\n"
                        f"codes: {', '.join(sorted(StallCode.ALL))}")
            ref, code, message = sub
            mission, _amb = _resolve(ref)
            if _amb:
                return _amb
            if mission is None:
                return f"no mission matching {ref!r}."
            if code not in StallCode.ALL:
                return (f"unknown stall code {code!r} — one of: "
                        f"{', '.join(sorted(StallCode.ALL))}")
            try:
                out = MissionRunner(self.context, store=store).mark_stalled(
                    mission.id, code, message)
            except (NoMoralsError, ValueError) as exc:
                return f"couldn't mark stall: {exc}"
            stall = out["stall"] or {}
            code = stall.get("code")
            label = StallCode.LABELS.get(code, code)
            hint = StallCode.UNBLOCK_HINTS.get(code, "")
            reply = (f"⚠️ {mission.name} marked stalled [{code}]: "
                     f"{label} — {stall.get('message')}")
            if hint:
                reply += f"\nunblocks: {hint}"
            return reply

        if verb == "clear":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return f"no mission matching {rest!r}."
            cleared = MissionRunner(self.context, store=store).clear_stalled(mission.id)
            return (f"{mission.name}: stall cleared — back in play."
                    if cleared else f"{mission.name}: no stall recorded.")

        if verb == "pause":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return f"no mission matching {rest!r}."
            # idempotent: double-pause is a no-op success, and a finished
            # mission is already past pausing — never an error, never a
            # state rewrite.
            if mission.status == MissionStatus.PAUSED:
                return (f"⏸ {mission.name}: already paused — no change "
                        f"(/mission resume {mission.id} to continue).")
            if mission.terminal:
                return (f"{mission.name} is already {mission.status} — "
                        "nothing to pause.")
            try:
                store.set_status(mission.id, MissionStatus.PAUSED)
            except (NoMoralsError, ValueError) as exc:
                return f"couldn't pause: {exc}"
            return (f"⏸ {mission.name}: paused — "
                    f"/mission resume {mission.id} to continue.")

        if verb == "resume":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return f"no mission matching {rest!r}."
            # a "running" mission whose runner died is not running —
            # reconcile first so resume restarts it instead of no-op'ing
            rec = store.reconcile(mission.id)
            if rec["changed"]:
                mission = store.get(mission.id)
            if mission.terminal:
                return (f"{mission.name} is {mission.status} — terminal "
                        "missions can't resume; /mission retry starts a "
                        "fresh attempt.")
            # idempotent: resume of a live mission is a no-op success
            if mission.status == MissionStatus.RUNNING:
                return (f"▶ {mission.name}: already running — no change; "
                        f"/mission status {mission.id} for progress.")
            runner = MissionRunner(self.context, store=store)

            def _resume_job() -> None:
                try:
                    runner.resume(mission.id)
                except Exception:  # noqa: BLE001 - chat must stay alive
                    _log.exception("mission resume %s failed", mission.id)

            threading.Thread(target=_resume_job,
                             name=f"mission-resume-{mission.id[:8]}",
                             daemon=True).start()
            return (f"▶ {mission.name}: resumed in the background — "
                    f"/mission status {mission.id} for progress.")

        if verb == "cancel":
            sub = rest.split(None, 1)
            ref = sub[0] if sub else ""
            reason = sub[1].strip() if len(sub) > 1 else "cancelled by owner"
            mission, _amb = _resolve(ref)
            if _amb:
                return _amb
            if mission is None:
                return f"no mission matching {ref!r}."
            # idempotent: a finished mission stays finished — cancelling
            # must never rewrite done/failed into cancelled, and a second
            # cancel is a no-op success, not an error.
            if mission.status == MissionStatus.CANCELLED:
                return f"⏹ {mission.name}: already cancelled — no change."
            if mission.terminal:
                return (f"⏹ {mission.name}: already {mission.status} — "
                        "nothing to cancel.")
            try:
                store.set_status(mission.id, MissionStatus.CANCELLED, note=reason)
            except (NoMoralsError, ValueError) as exc:
                return f"couldn't cancel: {exc}"
            runner = MissionRunner(self.context, store=store)
            runner.cancel(reason)  # cooperative: any in-flight runner stops
            if runner.reporter is not None:
                try:
                    runner.reporter.on_terminal(
                        mission, MissionStatus.CANCELLED, error=reason)
                except Exception:  # noqa: BLE001 - telemetry, not chat
                    _log.debug("cancel milestone push failed", exc_info=True)
            return f"⏹ {mission.name}: cancelled ({reason})."

        if verb == "retry":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return f"no mission matching {rest!r}."
            if mission.status == MissionStatus.RUNNING:
                return (f"{mission.name} is still running — /mission cancel "
                        "it first if you want to start over.")
            # cancel + requeue: a fresh mission row keeps the goal, the plan
            # skeleton, the name and the budgets; spend, errors, stalls and
            # the milestone log start clean. The original row is kept
            # as-is when it is already terminal — its done/failed record
            # is history, not something retry may rewrite.
            fresh_state: dict[str, Any] = {}
            if mission.state.get("plan"):
                fresh_state["plan"] = mission.state["plan"]
            new = store.create_new(
                mission.goal,
                name=mission.name,
                budget_wall=mission.budget_wall,
                budget_tokens=mission.budget_tokens,
                state=fresh_state,
                metadata={**(mission.metadata or {}), "retry_of": mission.id},
            )
            if mission.terminal:
                old_note = (f"the {mission.status} original is kept as-is; "
                            "this is a brand-new attempt")
            else:
                try:
                    store.set_status(mission.id, MissionStatus.CANCELLED,
                                     note=f"superseded by retry {new.id}")
                except (NoMoralsError, ValueError) as exc:
                    _log.debug("retry: could not cancel old mission: %s", exc)
                old_note = "the previous attempt was cancelled"
            runner = MissionRunner(self.context, store=store)

            def _retry_job() -> None:
                try:
                    runner.run(new, max_iterations=8)
                except Exception:  # noqa: BLE001 - chat must stay alive
                    _log.exception("mission retry %s failed", new.id)

            threading.Thread(target=_retry_job,
                             name=f"mission-retry-{new.id[:8]}",
                             daemon=True).start()
            return (f"🔁 {mission.name} is {mission.status}: retry starts a "
                    f"NEW attempt [{new.id}] linked via metadata.retry_of "
                    f"({old_note}) — running in the background.")

        if verb == "watch":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return f"no mission matching {rest!r}."
            if not chat_key:
                return "watch needs a chat context — run this from a chat."
            # reconcile on read: report the true state, not stale "running"
            store.reconcile(mission.id)
            mission = store.get(mission.id)
            added = MissionWatchers(self.context.db).subscribe(
                mission.id, chat_key)
            state_note = f" (currently {mission.status})"
            if added:
                return (f"👀 watching {mission.name}: milestone updates "
                        f"will land in this chat.{state_note}")
            return f"already watching {mission.name} from this chat.{state_note}"

        if verb == "unwatch":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return f"no mission matching {rest!r}."
            if not chat_key:
                return "unwatch needs a chat context — run this from a chat."
            removed = MissionWatchers(self.context.db).unsubscribe(
                mission.id, chat_key)
            return (f"stopped watching {mission.name} from this chat."
                    if removed
                    else f"{mission.name} isn't watched from this chat.")

        if verb == "new":
            sub = rest.split(None, 1)
            template = sub[0] if sub else ""
            targs = sub[1].strip() if len(sub) > 1 else ""
            spec = self._mission_template_spec(template, targs)
            if spec is None:
                return ("usage: /mission new <template> <args>\n"
                        "templates: research <topic> | build <what> | "
                        "fix <target>")
            mission = store.create_new(**spec)
            return (f"🚀 mission created: {mission.name} [{mission.id}] — "
                    f"{mission.goal}\n"
                    f"/mission resume {mission.id} to start it.")

        return usage

    @staticmethod
    def _mission_template_spec(template: str, args: str) -> dict[str, Any] | None:
        """Goal + plan skeleton for ``/mission new <template> <args>``.

        Returns the kwargs for ``MissionStore.create_new``, or None when
        the template is unknown.
        """
        t = (template or "").lower()
        plans = {
            "research": ["gather sources", "synthesize findings", "write report"],
            "build": ["design", "implement", "verify with tests"],
            "fix": ["reproduce", "root-cause", "patch", "verify"],
        }
        if t not in plans or not args.strip():
            return None
        arg = args.strip()
        goals = {
            "research": (f"Research {arg}: gather sources, synthesize "
                         "findings, report with citations"),
            "build": f"Build {arg}: design, implement, verify with tests",
            "fix": (f"Fix {arg}: reproduce the issue, find the root cause, "
                    "patch it, verify"),
        }
        return {
            "goal": goals[t],
            "name": f"{t} — {arg[:48]}",
            "state": {
                "plan": [
                    {"name": s, "goal": s, "role": "execution",
                     "kind": "io", "depends_on": []}
                    for s in plans[t]
                ]
            },
            "metadata": {"template": t, "template_args": arg[:200]},
        }

    @staticmethod
    def _find_mission(store: Any, ref: str) -> Any | None:
        """Resolve an id, id prefix, or name/goal substring to a mission.

        Empty ref → the default active mission (the documented ``[id|name]``
        UX for ``/mission status``), or None when there is none — the
        default never goes through prefix matching, so it can never
        match-all.
        Exact id wins, even when it is also a prefix of another mission's
        id. A prefix/name matching 2+ missions raises
        :class:`~nomorals.core.errors.AmbiguousRef` — the resolver never
        guesses; the caller renders the candidates and asks the user.
        """
        from ..core.errors import AmbiguousRef, NotFound
        from ..core.ids import min_unique_prefix_len, resolve_id_prefix

        filt = (ref or "").strip()
        if not filt:
            active = store.resumable()
            if active:
                return active[0]
            rows = store.list(limit=1)
            return rows[0] if rows else None
        try:
            return store.get(filt)
        except NotFound:
            _log.debug("mission lookup: no exact id match for %r, "
                       "trying prefix/name", ref)
        rows = store.list(limit=100)
        by_id: dict[str, Any] = {}
        for m in rows:
            by_id.setdefault(m.id, m)
        res = resolve_id_prefix(filt, by_id)
        if res.outcome in ("exact", "unique"):
            # the id channel is authoritative; name/goal search is only a
            # fallback when the id channel finds nothing at all
            return by_id[res.matches[0]]
        low = filt.lower()
        name_hits = [m for m in rows
                     if low in (m.name or "").lower()
                     or low in (m.goal or "").lower()]
        if res.outcome == "none":
            if len(name_hits) == 1:
                return name_hits[0]
            if not name_hits:
                return None
            ordered = [m.id for m in name_hits]
        else:  # ambiguous id prefix — never guess; union with name hits
            ordered = list(res.matches)
            for m in name_hits:
                if m.id not in ordered:
                    ordered.append(m.id)
        if len(ordered) == 1:
            return by_id[ordered[0]]
        if not ordered:
            return None
        raise AmbiguousRef(
            ref=filt,
            entity="mission",
            candidates=[(mid, f"[{by_id[mid].status}] {by_id[mid].name}")
                        for mid in ordered],
            min_prefix_len=min_unique_prefix_len(ordered),
            hint="/mission list shows the active ones.",
        )

    # ── image tools ──────────────────────────────────────────────────────────
    @staticmethod
    def _tool_data(outcome: Any) -> Any:
        """registry.call wraps the tool's dict in Ok(...); the tools ALSO
        report failure inside their dict ({'ok': False, 'error': ...}), so a
        lookup needs both layers checked. Returns (data, error_text)."""
        if not outcome.ok:
            return None, (outcome.error.message if outcome.error else "unknown")
        data = outcome.value
        if isinstance(data, dict) and data.get("ok") is False:
            return None, str(data.get("error") or "unknown")
        return data, None

    def _control_image(self, tail: str) -> str:
        ref = (tail or "").strip()
        if not ref:
            return "usage: /image <path-or-url>"
        outcome = self.context.tools.call("image_lookup", path=ref)
        data, error = self._tool_data(outcome)
        if data is None:
            return f"image lookup failed: {error}"
        lines = [f"🖼 {data.get('format', '?')} · {data.get('width') or '?'}×{data.get('height') or '?'} · "
                 f"{data.get('bytes', 0) // 1024} KB"]
        if data.get("seen_before"):
            where = f" (last: {data['seen_where']})" if data.get("seen_where") else ""
            lines.append(f"seen before{data.get('seen_days', '')}{where}")
        near = data.get("near_duplicates") or []
        if near:
            lines.append("near duplicates: " + "; ".join(near[:3]))
        return "\n".join(lines)

    def _control_lens(self, tail: str) -> str:
        ref = (tail or "").strip()
        if not ref:
            return "usage: /lens <path-or-url>"
        outcome = self.context.tools.call("reverse_image_search", path=ref)
        data, error = self._tool_data(outcome)
        if data is None:
            return f"reverse search failed: {error}"
        lines = ["🔍 reverse image search:"]
        for name, link in (data.get("lookup_links") or {}).items():
            lines.append(f"  {name}: {link}")
        matches = data.get("unverified_matches") or []
        if matches:
            lines.append("possible related (unverified, from Bing):")
            lines.extend(f"  {m}" for m in matches[:5])
        if data.get("note"):
            lines.append(data["note"])
        return "\n".join(lines)

    def _model_status_line(self) -> str:
        """Which model is ACTUALLY answering, and whether it's failing over.

        This is the line that should have caught the silent mock fallback:
        a failing provider must be visible in /status, not buried in a warn
        log line that scrolls away.

        Honest by construction: a provider is only called "answering" if it
        has actually succeeded (``last_success``).  The old line showed the
        first fallback *by position* — "hf_serverless is answering for now"
        while every message went unanswered because that fallback was
        failing too (no token).  That lie is what made a dead chain look
        alive in /status.
        """
        router = getattr(self.context, "router", None)
        snapshot = getattr(router, "stats_snapshot", None)
        if snapshot is None:
            return "n/a"
        try:
            snap = snapshot()
        except Exception:  # noqa: BLE001 - status must never crash a chat
            return "n/a"
        active = str(snap.get("active") or "?")
        chain = [str(p) for p in (snap.get("chain") or [])]
        health = (snap.get("health") or {})
        active_health = health.get(active) or {}
        if not int(active_health.get("failures") or 0):
            return f"{active} (chain: {', '.join(chain)})" if chain else active
        last_error = str(active_health.get("last_error") or "unknown error")
        reason = self._plain_model_reason(last_error)
        # Who has ACTUALLY answered recently?  By last_success, not by
        # position in the chain.
        succeeded = [
            (float((health.get(p) or {}).get("last_success") or 0.0), p)
            for p in chain
            if p != active
        ]
        succeeded = [(ts, p) for ts, p in succeeded if ts > 0]
        if succeeded:
            top = max(succeeded)[1]
            return (f"{top} is answering for now — {active} is down ({reason}); "
                    f"{active} gets another try on the next message")
        # Nothing in the chain has ever answered.  Say so, with the backup's
        # own failure reason — that is the difference between "down and
        # recovering" and "silently dead".
        backups = [p for p in chain if p != active]
        if not backups:
            return f"no model is answering — {active} is down ({reason}) and " \
                   "there is no backup configured"
        caps = {str(n): set(v) for n, v in (snap.get("chain_caps") or {}).items()}
        backup_reasons = []
        for p in backups:
            known_caps = caps.get(p)
            if known_caps is not None \
                    and "chat" not in known_caps and "complete" not in known_caps:
                # this backup can never answer a chat message — saying
                # "has not been tried yet" for it is a soft lie
                # (unknown capability lists are treated as chat-capable)
                continue
            b_health = health.get(p) or {}
            if int(b_health.get("failures") or 0):
                backup_reasons.append(f"{p} failed too ({self._plain_model_reason(str(b_health.get('last_error') or 'unknown error'))})")
            else:
                backup_reasons.append(f"{p} has not been tried yet")
        if backup_reasons:
            return (f"no model is answering — {active} is down ({reason}); "
                    + "; ".join(backup_reasons)
                    + ". Messages get no reply until one of them works")
        # every backup is a non-chat provider (e.g. ocr) — only the local
        # model can answer
        return (f"no model is answering — {active} is down ({reason}) and the "
                "other providers in the chain cannot chat. Only fixing "
                f"{active} restores replies")

    @staticmethod
    def _plain_model_reason(error: str) -> str:
        """A raw exception string in chat reads like the system is broken.
        Translate the common ones into plain words."""
        low = error.lower()
        if "still loading" in low or "loading" in low:
            return "still loading the model — normal on a phone, takes a few minutes"
        if "400" in low or "not served by this hf" in low or "not in the inference" in low:
            return "the model name is not hosted there — it tries to swap to a hosted model"
        if "timed out" in low or "timeout" in low:
            return "took too long — if it's your phone's model, close other apps"
        if "connection refused" in low or "errno 111" in low:
            return "not running — start it with `nm models --start-local`"
        if "not responding" in low:
            return "frozen — kill and restart it with `nm models --start-local`"
        if "rate limit" in low or "429" in low:
            return "rate limited — it recovers on its own"
        if "401" in low or "unauthorized" in low or "invalid" in low:
            return "bad credentials"
        return error[:60]

    def _control_status(self) -> str:
        mood = self.brain.mood.current()
        rel = self.brain.relationship
        aut = self._autonomy.status() if self._autonomy else {"mode": "off"}
        from .power import power_mode_for

        power_state = "active" if power_mode_for(self.context).active else "locked"
        platforms = ", ".join(
            name for name, info in self.gateway.status().items()
            if name != "_stats" and info.get("running_in_session")
        ) or "none"
        return "\n".join(
            [
                f"mood: {mood.label} (affection {mood.values.get('affection', 0):.0f}, "
                f"energy {mood.values.get('energy', 0):.0f}, frustration {mood.values.get('frustration', 0):.0f})",
                f"relationship: {rel.stage}, trust {rel.trust:.0f}, "
                f"{len(rel.milestones)} milestones, {len(rel.fights)} fights logged",
                f"autonomy: {aut.get('mode', 'off')}, {aut.get('pending_proposals', 0)} pending",
                f"model: {self._model_status_line()}",
                f"platforms running: {platforms}",
                f"power mode: {power_state}",
                f"stats: {self.stats['messages']} msgs, {self.stats['replies']} replies, "
                f"{self.stats['errors']} errors, {self.stats.get('controls', 0)} commands",
            ]
        )

    def _control_mood(self, arg: str) -> str:
        engine = self.brain.mood
        try:
            if not arg:
                state = engine.current()
                dims = ", ".join(f"{k} {int(v)}" for k, v in sorted(state.values.items()))
                return f"mood: {state.label} — {dims}"
            if arg.strip().lower() == "reset":
                engine.reset()
                return "mood reset to baselines"
            if "=" in arg:
                pairs = dict(
                    (p.split("=", 1)[0].strip().lower(), float(p.split("=", 1)[1]))
                    for p in arg.split() if "=" in p
                )
                engine.set_dimensions(pairs)
                state = engine.current()
                return f"mood set: {state.label} ({', '.join(f'{k} {int(v)}' for k, v in pairs.items())})"
            engine.set_label(arg)
            return f"mood forced: {engine.current().label}"
        except ValueError as exc:
            return str(exc)

    def _control_mode(self, arg: str) -> str:
        mode = arg.strip().lower()
        if mode not in {"off", "suggest", "auto"}:
            return "usage: /mode off|suggest|auto"
        self.settings.partner.autonomy_mode = mode
        try:
            with self.context.db.transaction():
                self.context.db.execute(
                    "INSERT INTO kv_store (key, value, kind, updated_at) VALUES (?, ?, 'json', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                    ("partner.autonomy_mode", json.dumps({"mode": mode}), time.time()),
                )
        except Exception:  # noqa: BLE001 - a write failure must not block the switch
            pass
        try:
            from .autonomy import AutonomyAgent

            if self._autonomy is None and mode != "off":
                dm_cap, group_cap = self._tuned_autonomy_caps()
                self._autonomy = AutonomyAgent(
                    self.context, self.brain, self.gateway,
                    mode=mode,
                    owner_chats=self._owner_chats,
                    group_chats=_key_set(self.settings.partner.group_chats),
                    quiet_start=self.settings.partner.quiet_start,
                    quiet_end=self.settings.partner.quiet_end,
                    max_dm_per_day=dm_cap,
                    max_group_per_day=group_cap,
                )
                self._autonomy.start()
            elif self._autonomy is not None:
                self._autonomy.set_mode(mode)
        except Exception as exc:  # noqa: BLE001
            _log.warning("autonomy mode switch failed: %s", exc)
            return f"mode set to {mode} (agent switch failed: {exc})"
        return f"autonomy: {mode}"

    def _control_model(self, arg: str) -> str:
        """Live model switch — no restart.

        ``/model`` reports what is answering; ``/model <provider> [fallback …]``
        rebuilds the router in place. The choice is persisted to the kv_store
        (the same store ``nm models --set-provider`` writes), so it also
        survives a restart.
        """
        parts = [p for p in (arg or "").split() if p]
        if not parts:
            return f"model: {self._model_status_line()}"
        provider = parts[0].lower()
        chain = [p.lower() for p in parts[1:]]

        from ..llm.providers import build_provider

        try:
            build_provider(provider, model="probe")  # validates the name
        except Exception as exc:  # noqa: BLE001
            return f"can't use {provider!r}: {exc}"

        from .context import _build_router, persist_provider_override

        db = self.context.db
        try:
            persist_provider_override(db, provider, chain or None)
        except Exception as exc:  # noqa: BLE001
            return f"could not persist the switch: {exc}"
        try:
            new_router = _build_router(self.settings, self.context.bus, db=db, context=self.context)
        except Exception as exc:  # noqa: BLE001
            return f"could not build the {provider} router: {exc}"
        self.brain.responder.router = new_router
        self.context.router = new_router
        _log.info("model switched live: provider=%s chain=%s", provider, chain)
        return f"model switched — now: {self._model_status_line()}"

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


def _key_set(raw: str) -> set[str]:
    from ..social.chat.gateway import parse_chat_keys

    return parse_chat_keys(raw or "")


def _direct_approve(context: Any, gateway: Any, proposal_id: str) -> dict[str, Any]:
    """Approve a held proposal without a running autonomy agent (CLI path)."""
    row = context.db.query_one("SELECT * FROM proactive_log WHERE id = ?", (proposal_id,))
    if row is None:
        return {"ok": False, "error": f"no proposal {proposal_id!r}"}
    if row["status"] != "pending":
        return {"ok": False, "error": f"proposal is {row['status']}, not pending"}
    from ..social.chat.base import ChatRef

    chat = ChatRef(
        platform=row["platform"], chat_id=row["chat_id"],
        kind="group" if row["kind"] == "group" else "dm",
    )
    result = gateway.send(chat.platform, chat, row["content"])
    status = "sent" if result.ok else "failed"
    try:
        context.db.execute(
            "UPDATE proactive_log SET status = ?, acted_at = ? WHERE id = ?",
            (status, time.time(), proposal_id),
        )
    except Exception:  # noqa: BLE001
        pass
    return {"ok": result.ok, "status": status, "error": result.error}


def _direct_deny(context: Any, proposal_id: str) -> dict[str, Any]:
    row = context.db.query_one("SELECT status FROM proactive_log WHERE id = ?", (proposal_id,))
    if row is None:
        return {"ok": False, "error": f"no proposal {proposal_id!r}"}
    if row["status"] != "pending":
        return {"ok": False, "error": f"proposal is {row['status']}, not pending"}
    context.db.execute(
        "UPDATE proactive_log SET status = 'denied', acted_at = ? WHERE id = ?",
        (time.time(), proposal_id),
    )
    return {"ok": True, "status": "denied"}
