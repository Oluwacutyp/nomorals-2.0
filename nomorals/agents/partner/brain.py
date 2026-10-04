"""PartnerBrain: stateless-per-message cognition for the partner."""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from typing import Any, Callable
from ...core.config import Settings
from ...core.ids import ulid_now
from ...core.logging_setup import get_logger
from ...llm.base import Message, SamplingParams
from ...partner.background import BackgroundSelector
from ...partner.lexicon_feed import LexiconFeed, seed_partner_lexicon
from ...partner.mood import MoodEngine
from ...partner.persona import Persona, default_persona
from ...partner.gating import gate_decision, is_owner_chat, is_restricted
from ...partner.presence import Presence, decide_presence, human_typing_seconds
from ...partner.relationship import Relationship
from ...partner.responder import PartnerResponder, detect_signals
from ...social.chat.base import ChatKind, ChatMessage, ChatRef
from .outcome import PresenceOutcome
from .approvals import _key_set
_log = get_logger(__name__)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


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
            self._persist_outbound(
                message.chat, reply_parts, model,
                session_id=message.meta.get("os_session_id") or message.chat.key)

    def _persist_inbound(self, message: ChatMessage) -> None:
        """The user's message — written the moment she READS it, even if the
        reply comes minutes later or not at all."""
        chat = message.chat
        db = self.context.db
        # Session-scoped conversation id (falls back to chat.key).
        conv_id = message.meta.get("os_session_id") or chat.key
        title = chat.title or chat.peer or chat.chat_id
        try:
            with db.transaction():
                db.execute(
                    """INSERT INTO conversations (id, title, agent, channel, created_at, updated_at)
                       VALUES (?, ?, 'partner', ?, 0, ?)
                       ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at,
                                                    title = CASE WHEN excluded.title != '' THEN excluded.title ELSE conversations.title END""",
                    (conv_id, title, chat.platform, time.time()),
                )
                db.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, name, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ulid_now(), conv_id, "user", message.text, message.sender, "", time.time()),
                )
        except Exception as exc:  # noqa: BLE001 - persistence must never block a reply
            _log.warning("persist turn failed: %s", exc)

    def _persist_outbound(self, chat: ChatRef, reply_parts: list[str], model: str,
                          *, session_id: str = "") -> None:
        if not reply_parts:
            return
        conv_id = session_id or chat.key
        try:
            with self.context.db.transaction():
                self.context.db.execute(
                    "INSERT INTO messages (id, conversation_id, role, content, name, model, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ulid_now(), conv_id, "assistant", "\n".join(reply_parts), self.persona.name, model, time.time()),
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
        # OS Session: the gateway attaches message.meta["os_session_id"] when
        # a SessionBridge is injected at boot. It is the canonical
        # conversation scope — memory, history, and extraction key off it.
        # Falls back to chat.key (identical scoping) when no bridge is wired.
        session_id = message.meta.get("os_session_id") or chat.key
        message.meta["os_session_id"] = session_id

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
        # Canonical session scope (set by handle_message; deliver_reply path
        # ensures it too). Memory, history, and extraction key off this.
        session_id = message.meta.get("os_session_id") or chat.key
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
        _origin = f"chat:{message.chat.key}" if message.chat else ""
        memories = () if restricted else self.responder.recall(message.text, limit=5, origin=_origin)
        romantic = is_owner and self.relationship.is_romantic()
        background_lines = self.background.context(
            message.text,
            user_in_us=flags["in_us"],
            romantic=romantic,
        )

        # Generate.
        history = self._history(session_id, limit=self.settings.partner.history_window)
        continuity = () if restricted else self._continuity_lines(session_id)
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
        self._persist_outbound(chat, parts, bundle.model, session_id=session_id)
        self._note_reply(bundle, chat)
        self._log_training_pair(chat, message.text, "\n".join(parts), bundle.model,
                                self.mood.current().label)
        self.relationship.save(self.context.db)
        self._maybe_curate(session_id, is_owner)
        self._maybe_extract(message, session_id, is_owner)
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
            self._persist_outbound(
                chat, [reply_text], "fast-path",
                session_id=message.meta.get("os_session_id") or chat.key)
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
                    from ...memory.extract import MemoryExtractor

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
        from ..reasoning import ReasoningEngine, looks_complex
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

        def _safe_ts(loop: dict) -> float:
            try:
                return float(loop.get("ts", 0))
            except (TypeError, ValueError):
                return 0.0

        fresh = [loop for loop in existing if isinstance(loop, dict) and now - _safe_ts(loop) < 48 * 3600]
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
