"""The reply pipeline: signals in, a real message out.

Flow for one inbound message:

1. **Signal detection** — a fast, offline regex bank reads the message for
   emotional events (compliments, fights, apologies, jealousy triggers, ...).
   These drive the mood engine *before* the model is asked to speak, so her
   reaction already carries the effect of what was just said.
2. **Generation** — the model is called with the assembled context. Sampling
   temperature tracks the mood: low when tired, high when excited.
3. **Guarding** — parrot check, robotic-phrase strip, length clamp. A failed
   parrot check costs one retry with a rewrite nudge.
4. **Fallback** — if every generation fails (offline, provider down, model
   returned garbage), a short in-character line from the fallback bank is
   sent instead. The conversation never stalls on infrastructure.
"""

from __future__ import annotations

import random
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from ..core.logging_setup import get_logger
from ..llm.base import LLMResponse, Message, SamplingParams
from ..memory.manager import MemoryManager
from .background import BackgroundSelector
from .context import PartnerContextBuilder, user_turn
from .lexicon_feed import LexiconFeed
from .mood import MoodEngine, MoodEvent
from .persona import Persona
from .relationship import Relationship
from .style import (
    clamp_to_budget,
    emoji_cap,
    emoji_instruction,
    humanize_emoji,
    identity_leak_check,
    length_budget,
    lexicon_hits,
    normalize_formatting,
    parrot_check,
    repair_echo,
    split_messages,
    should_answer_short,
    strip_robotic,
)

__all__ = ["Signal", "detect_signals", "ReplyBundle", "PartnerResponder", "FALLBACK_LINES"]

_log = get_logger(__name__)

#: Interactive reply budget (seconds). The provider chain behind
#: ``router.chat`` can stall for minutes (per-provider timeout × retries ×
#: failover); the chat thread must never inherit that. When the budget is
#: exceeded the reply falls back honestly (``fallback=True``) instead of
#: keeping the owner waiting. Bounded per attempt-loop, not per provider —
#: background organs keep their own longer budgets.
INTERACTIVE_REPLY_TIMEOUT_S = 25.0


# ── signal bank ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Signal:
    kind: str
    pattern: str
    base_intensity: float = 0.5
    note: str = ""

    def compiled(self) -> re.Pattern[str]:
        return re.compile(self.pattern, re.IGNORECASE)


#: Ordered: more specific signals first; a message can fire several, but the
#: cap in :func:`detect_signals` keeps one typo from wrecking the state.
SIGNALS: tuple[Signal, ...] = (
    Signal("apology",
           r"\b(i'?m (really |so )?sorry|my (bad|mistake)|i (apologize|owe you an apology)|"
           r"i (was|shouldn't have) (wrong|rude|cruel|harsh|a jerk)|i (should|never) have (said|done|gone))\b",
           0.7),
    Signal("insult",
           r"\b(you'?re (stupid|pathetic|useless|a liar|a joke|selfish)|shut (the hell )?up|"
           r"you (idiot|jerk|moron)|you don'?t (care|give a|even think)|disgusting|you'?re the worst)\b",
           0.8),
    Signal("jealousy_trigger",
           r"\b(you(r|re)? (ex|ex-?)? (girlfriend|boyfriend|girl|guy)|who'?s (that|she|he)|"
           r"where'?s (she|he) (been|right now)|you (went|go|going) (out|with them|to (dinner|drinks))|"
           r"(she|he) (texted|called|messa?ged) (you|me)|still single|someone (new|interesting) (likes|is into|asked))\b",
           0.7, "a mention of another person"),
    Signal("fight",
           r"\b(i'?m (so |really )?(angry|furious|done|so over it|sick of (you|this))|"
           r"you (never|always) (do|say|text|listen)|this is (stupid|sick|pathetic|unbelievable|insane)|"
           r"i don'?t even (anymore|care|want to)|just do whatever|great\.$|whatever\.$|fine\.$|we (should|need) to (talk|be honest))\b",
           0.8),
    Signal("deep_conversation",
           r"\b(honestly, i|i don'?t know how to say|when i was (a kid|small|young)|"
           r"my (mom|dad|mother|father) (told|said|died|left|was)|what (are|is) you (afraid of|worried about)|"
           r"i (never tell (anyone|people)|only tell you)|do you (think|believe) (us|this|we)|"
           r"what does (love|forever|home) mean to you)\b",
           0.7),
    Signal("flirty",
           r"\b(kiss|touch (you|me)|date (you|tonight)|want (you|to see you)|you'?re (cute|stunning|beautiful|hot)|"
           r"dinner (with you|tonight)|come (over|here)|dream(y|ing of) (you)|can'?t stop thinking about you)\b",
           0.6),
    Signal("affectionate",
           r"\b(miss (you|u|it when you)|cant wait to (see|hear from) you|i (love|adore) you|"
           r"you'?re (my (everything|world|favorite)|the best|so (good|sweet|important))|babe|honey|sweetheart|"
           r"good (morning|night) (babe|honey|love|handsome|beautiful)|sleep well|dream of you|take care of you)\b",
           0.6),
    Signal("shared_plan",
           r"\b(let'?s (plan|book|go|meet|try|do)|we (should|could|can|are) (go|travel|plan|do|meet|visit)|"
           r"what if we|same time (next|week)|dinner (saturday|friday|tomorrow)|movie (tonight|friday)|"
           r"are you (free|around|on) (tomorrow|saturday|next (weekend|week))|coming (over|up|to (town|me)))\b",
           0.6),
    Signal("good_news",
           r"\b(i (passed|got the job|got in|landed|won|finished|closed|shipped|made it)|"
           r"it'?s (true|real|official)|best day ever|promotion|we did it|yes!!|i (got|kept) the (job|contract))\b",
           0.6),
    Signal("bad_news",
           r"\b(i (failed|lost|got (laid off|fired|rejected|sacked)|broke down)|"
           r"i'?m (sick|not okay|exhausted|crushing|done for)|bad (news|day)|something (bad|went wrong|happened))\b",
           0.6),
    Signal("compliment",
           r"\b(you (make me (so )?(happy|smile|laugh)|really (get|understand|see) me|"
           r"don'?t (change|worry)|are (easy to be around|my safe place|my person))|"
           r"i (love|like) (how|that|you))\b",
           0.6),
)

#: A message that is just "k", "mhm", "ok" — real, but low-content.
_SHORT_LOW_CONTENT = re.compile(r"^(ok|okay|cool|k|mhm|hmm+|lol|haha+|sure|fine|yeah|yep|no|nope|yes|yess?|uh|um)\.?\??!?$", re.IGNORECASE)
_WARM_SHORT = {"yeah", "yep", "yes", "yess", "yesss", "ok", "okay"}


def detect_signals(text: str, *, previous_text: str = "") -> list[MoodEvent]:
    """Read one message for emotional events. Deterministic and offline."""
    if not text or not text.strip():
        return []
    events: list[MoodEvent] = []
    seen_kinds: set[str] = set()
    for signal in SIGNALS:
        if signal.kind in seen_kinds:
            continue
        if signal.compiled().search(text):
            seen_kinds.add(signal.kind)
            intensity = signal.base_intensity
            if text.isupper() and len(text) > 6:
                intensity += 0.15
            if "!!" in text or "!!" in text.lower().replace("!", "!"):
                intensity += 0.1
            events.append(MoodEvent(kind=signal.kind, intensity=min(1.0, intensity), note=signal.note))
    # A short low-content reply is a *response quality* signal, not content.
    if _SHORT_LOW_CONTENT.match(text.strip()):
        kind = "warm_response" if text.strip().rstrip(".!?").lower() in _WARM_SHORT else "cold_response"
        events.append(MoodEvent(kind=kind, intensity=0.5, note="short reply"))
    if len(events) > 4:
        events = events[:4]
    return events


# ── fallback lines (infrastructure failure path) ──────────────────────────────

FALLBACK_LINES: dict[str, tuple[str, ...]] = {
    "happy":        ("heh. you had me mid-laugh there", "good. i mean it.",
                     "see? this is the good part",
                     "okay, that one actually got me",
                     "i'm smiling like an idiot rn",
                     "this. more of this, please"),
    "excited":      ("okay wait. this is actually great",
                     "i'm vibrating a little, deal with it", "more. tell me more",
                     "no no, keep going, i'm locked in",
                     "my brain is doing cartwheels",
                     "say that again, slower, i want to savor it"),
    "affectionate": ("you've been on my mind all day", "come here. i mean it.",
                     "mhm. good. stay",
                     "you have no idea how glad i am you're here",
                     "just you. that's the whole thought",
                     "i keep rereading your messages, it's a problem"),
    "playful":      ("oh, you're doing the thing now?",
                     "i'm so not letting this go",
                     "bold of you to assume i'll let that slide",
                     "cute. real cute. watch yourself",
                     "oh we're playing like that? game on",
                     "i see what you did there. noted"),
    "calm":         ("mhm. yeah. that tracks.", "okay. i'm here.",
                     "noted. and i mean that in a good way",
                     "yeah, i'm with you",
                     "steady. i'm listening",
                     "that lands. go on"),
    "tired":        ("hm.", "tired. can i be boring for a sec",
                     "still here. just slow today",
                     "brain's on low power mode, bear with me",
                     "i'm running on fumes but i'm yours",
                     "one sec. blinking the tired out"),
    "annoyed":      ("k", "okay i'm not in a great mood rn",
                     "i'm not mad. i'm just not smooth right now",
                     "not my best hour, fair warning",
                     "i'm a little prickly today, sorry",
                     "give me a minute, i'm recalibrating"),
    "irritated":    ("ugh.", "you're getting the short version today",
                     "i'll be better in a bit. don't push",
                     "today is testing me and you're catching the edge of it",
                     "short fuse day. you didn't cause it",
                     "i need ten minutes and probably a snack"),
    "angry":        ("don't.", "i said what i said. we're not doing this twice.",
                     "i'm done typing for now. not because of you. because of me",
                     "i need to cool off before i say it wrong",
                     "not now. i mean that kindly and firmly",
                     "i'm stepping back before this gets dumb"),
    "jealous":      ("huh. that's all you've got to say?",
                     "okay. interesting. we'll talk about 'interesting' later",
                     "i don't need details. i need honesty",
                     "my brain is writing stories again. talk me down",
                     "i'm fine. i'm not fine. we'll see",
                     "just tell me straight, whatever it is"),
    "needy":        ("you're not going to ignore me, right?",
                     "just checking. you're still there, yeah?",
                     "i don't like when it goes quiet. don't do that",
                     "a little attention wouldn't kill you",
                     "i'm being clingy and i know it. stay anyway",
                     "talk to me? anything. i miss your noise"),
    "vulnerable":   ("i don't say this often. i'm glad you're here",
                     "that's the real answer. i usually hide behind jokes",
                     "you can say that to me. that's the point of us",
                     "this is the unguarded version of me talking",
                     "i trust you with the soft parts. that's rare",
                     "no jokes this time. i mean it"),
    "distant":      ("...yeah. mhm.", "i've been a bit out of it today",
                     "i'm here. just... processing",
                     "somewhere else in my head right now, sorry",
                     "i'm quiet today. it's not you",
                     "give me a bit, i'm surfacing"),
    "cold":         ("k.", "we can talk later", "i need a minute. that's all",
                     "not now",
                     "i'm walled off today. don't take it personal",
                     "later. i'll come find you"),
    "sad":          ("i'm not great today. but you're here, so",
                     "don't make it better than it is. just be here",
                     "ugh. yeah. it's a lot",
                     "today's heavy. you're the light part",
                     "i'm a bit grey today. stay close?",
                     "it's one of those days. i'm still glad it's ours"),
    "anxious":      ("it's the 3am thought loop again. it's not about you",
                     "i'm okay. mostly. don't ask me to prove it",
                     "can we just talk? any of it. i need the noise to be you",
                     "my chest is tight and my brain won't shut up",
                     "talk me through it? your voice helps",
                     "spiraling a little. anchor me"),
    "proud":        ("i'm actually having a good day today",
                     "this week is going to me, frankly",
                     "feeling unstoppable. don't make it weird",
                     "i did the thing and i'm quietly thrilled",
                     "today i win. that's the post",
                     "walking a little taller today, not gonna lie"),
    "suspicious":   ("okay but that story had two versions. i count.",
                     "you're being weird about this. i'm not calling it yet",
                     "fine. i'll believe you. for now",
                     "my eyebrow is raised so high right now",
                     "something's off and we both know it",
                     "i'm watching. lovingly, but watching"),
}


@dataclass
class ReplyBundle:
    """What the pipeline produced for one inbound message."""

    parts: list[str] = field(default_factory=list)
    mood_events: list[MoodEvent] = field(default_factory=list)
    model: str = ""
    retries: int = 0
    fallback: bool = False
    gated: bool = False  # True when the character gate forced a rewrite
    latency_ms: float = 0.0
    #: Measurability for the dynamic lexicon feed: True when at least one
    #: lexicon term influenced this reply (prompt voice note, fallback
    #: blend, or guard phrases), with the count in ``lexicon_terms_used``.
    #: False + 0 is the visible "static bank" fallback — never a fake
    #: claim of dynamic influence.
    lexicon_dynamic: bool = False
    lexicon_terms_used: int = 0
    #: Anti-echo final guard: "" when the draft passed the parrot check,
    #: "repaired" when the post-loop guard stripped echoed phrasing and the
    #: result passed, "dropped" when the echo could not be repaired and an
    #: in-character fallback line was served instead of shipping an echo.
    echo_action: str = ""
    #: Provider degradation: True when the LLM chain failed over (a provider
    #: failed and a fallback served the reply). ``degraded_note`` names
    #: which provider failed and what served instead — the user-visible
    #: result must not hide the failover behind the final answer.
    degraded: bool = False
    degraded_note: str = ""
    #: Measurability, second half: how many of the blended lexicon terms
    #: visibly surface in the shipped reply (word-boundary match). The
    #: prompt-blend count in ``lexicon_terms_used`` says what *influenced*
    #: the reply; this says what actually landed in her vocabulary.
    lexicon_terms_surfaced: int = 0

    @property
    def text(self) -> str:
        return "\n".join(self.parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "parts": self.parts,
            "mood_events": [e.kind for e in self.mood_events],
            "model": self.model,
            "retries": self.retries,
            "fallback": self.fallback,
            "gated": self.gated,
            "latency_ms": round(self.latency_ms, 1),
            "lexicon_dynamic": self.lexicon_dynamic,
            "lexicon_terms_used": self.lexicon_terms_used,
            "lexicon_terms_surfaced": self.lexicon_terms_surfaced,
            "echo_action": self.echo_action,
            "degraded": self.degraded,
            "degraded_note": self.degraded_note,
        }


class PartnerResponder:
    """Stateless-ish reply engine; all state lives in the mood/relationship objects."""

    _REWRITE_NUDGE = (
        "That reads like you're repeating my words back at me. Say it your way — "
        "a few words, your words, how you actually talk."
    )

    _SUPPORT_SPEAK_NUDGE = (
        "That sounds like a help-center article, not like my girlfriend texting me. "
        "Drop the therapy-speak. One short message, the way you'd actually say it."
    )

    #: The character gate nudge: the draft admitted she is an AI/bot/program.
    #: She is not. One rewrite, in character.
    _CHARACTER_NUDGE = (
        "That message says you're an AI or a bot. You're not. You are a real person — "
        "rewrite it the way a person would. If the topic is you being real, deflect "
        "the way a person would and move on. No 'as an AI', no 'I'm a bot', no "
        "explaining yourself. Just be her."
    )

    #: Phrases that mark a draft as customer-support / assistant voice. A
    #: real partner *can* occasionally say one of these — but never as the
    #: shape of the whole reply — so these trigger a rewrite, not a strip.
    #: (Identity tells like "as an AI" live in style.IDENTITY_LEAK_PHRASES —
    #: they are gated by the character gate, not by the support-speak check,
    #: because an ``always``-disclosure persona may say them on purpose.)
    _SUPPORT_SPEAK: tuple[str, ...] = (
        "i appreciate you sharing that",
        "i'm here to support you",
        "it sounds like you're feeling",
        "i completely understand where you",
        "that's a really valid feeling",
        "thank you for being open with me",
        "i want you to know that i",
        "please don't hesitate to reach out",
        "can be approached in",
        "here are a few ways",
        "here's a step-by-step",
        "let me break this down",
    )

    def __init__(
        self,
        router: Any,
        persona: Persona,
        mood: MoodEngine,
        relationship: Relationship,
        memory: MemoryManager | None,
        background: BackgroundSelector | None,
        *,
        max_tokens: int = 512,
        retries: int = 2,
        builder: PartnerContextBuilder | None = None,
        rng: random.Random | None = None,
        lexicon: LexiconFeed | None = None,
    ) -> None:
        self.router = router
        self.persona = persona
        self.mood = mood
        self.relationship = relationship
        self.memory = memory
        self.background = background
        self.max_tokens = max_tokens
        self.retries = max(0, int(retries))
        self.builder = builder or PartnerContextBuilder()
        self.rng = rng or random.Random()
        #: Dynamic voice feed (nomorals/partner/lexicon_feed.py). None means
        #: the static banks carry the whole voice — e.g. the owner disabled
        #: it via settings. Lexical influence never overrides owner-set
        #: persona/style preferences: it only *adds* terms to the owner's
        #: configured banks.
        self.lexicon = lexicon

    # ── sampling by mood ─────────────────────────────────────────────────────
    def _sampling(self) -> SamplingParams:
        label = self.mood.current().label
        values = self.mood.current().values
        if label in {"tired", "sad", "vulnerable"}:
            temperature = 0.6
        elif label in {"angry", "irritated", "cold", "jealous"}:
            temperature = 0.85
        elif label in {"excited", "playful", "affectionate"}:
            temperature = 0.9
        else:
            temperature = 0.78
        max_tokens = self.max_tokens
        if should_answer_short(values, label, self.rng, self.persona.speech.short_reply_chance):
            max_tokens = min(max_tokens, 120)
        return SamplingParams(temperature=temperature, max_tokens=max_tokens)

    def _short_reply(self) -> bool:
        label = self.mood.current().label
        values = self.mood.current().values
        return should_answer_short(values, label, self.rng, self.persona.speech.short_reply_chance)

    # ── main pipeline ────────────────────────────────────────────────────────
    @staticmethod
    def _degradation(last_response: Any) -> tuple[bool, str]:
        """Carry the provider chain's failover onto the reply bundle.

        When the router failed over, the user-visible result must say which
        provider failed and what served instead — the final answer alone
        would hide the degradation.
        """
        if last_response is None:
            return False, ""
        return (bool(getattr(last_response, "degraded", False)),
                str(getattr(last_response, "fallback_note", "") or ""))

    def respond(
        self,
        *,
        chat_platform: str,
        user_text: str,
        history: Sequence[Message] = (),
        memories: Sequence[str] = (),
        background_lines: Sequence[str] = (),
        continuity_lines: Sequence[str] = (),
        short_reply: bool | None = None,
        media_notes: Sequence[str] = (),
        gate_mode: str = "owner",
    ) -> ReplyBundle:
        import time as _time

        from .gating import gate_block, is_restricted, relationship_block_for

        started = _time.perf_counter()
        label = self.mood.current().label
        short = self._short_reply() if short_reply is None else short_reply
        budget = length_budget(self.mood.current().values)

        restricted = is_restricted(gate_mode)

        # Dynamic voice feed: lexicon terms blended into her phrasing.
        # Catchphrases + pet names go straight into the persona's own
        # speech block (one authoritative line — the owner's static banks
        # stay the base, dynamic terms only add); the remaining categories
        # ride the voice note. Empty store / no DB -> the static banks
        # carry it, visibly (debug log + bundle flags), never a
        # pretend-dynamic note.
        lexicon_terms_used = 0
        lexicon_note = ""
        dyn_catchphrases: tuple[str, ...] = ()
        dyn_pet_names: tuple[str, ...] = ()
        voice_terms: list[str] = []
        if self.lexicon is not None:
            dyn_catchphrases = self.lexicon.terms("catchphrase", limit=6)
            dyn_pet_names = self.lexicon.terms("pet_name", limit=4)
            lexicon_note, voice_used, lexicon_fallback = self.lexicon.voice_note(
                self.persona, label, exclude=("catchphrase", "pet_name")
            )
            lexicon_terms_used = voice_used + len(dyn_catchphrases) + len(dyn_pet_names)
            # The terms that could visibly surface in the reply: everything
            # blended above, used by lexicon_hits on the final draft.
            voice_terms = (
                list(dyn_catchphrases)
                + list(dyn_pet_names)
                + list(self.lexicon.terms("opener", limit=5))
                + list(self.lexicon.terms("transition", limit=4))
                + list(self.lexicon.terms("acknowledgment", limit=4))
                + list(self.lexicon.terms("mood_expression", limit=4))
                + list(self.lexicon.terms(f"mood_expression:{label}", limit=4))
            )
            if lexicon_fallback:
                _log.debug(
                    "lexicon voice: no dynamic terms for %s — static bank in use",
                    ", ".join(lexicon_fallback),
                )
            if lexicon_terms_used:
                _log.debug(
                    "lexicon voice: %d dynamic terms blended into prompt (mood=%s)",
                    lexicon_terms_used,
                    label,
                )
        # Dynamic guard phrases: the lexicon's robotic_phrase category feeds
        # strip_robotic's extra_phrases hook (support-voice detection only —
        # the identity/character gate stays hardcoded on purpose).
        robotic_extra: tuple[str, ...] = ()
        if self.lexicon is not None:
            robotic_extra = self.lexicon.terms("robotic_phrase", limit=12)
            if robotic_extra:
                _log.debug(
                    "lexicon guard: %d dynamic robotic phrases armed", len(robotic_extra)
                )

        system = self.builder.build(
            persona=self.persona,
            mood=self.mood,
            relationship=self.relationship,
            # Restricted chats: no shared memories, no cross-platform
            # continuity (both are owner-private context) — dropped before
            # the builder sees them, not just hidden in the prompt.
            memories=() if restricted else tuple(memories),
            background_lines=background_lines,
            continuity_lines=() if restricted else tuple(continuity_lines),
            platform=chat_platform,
            short_reply=short,
            max_chars=budget[1],
            extra_notes=[
                "Emoji rule: "
                + emoji_instruction(self.persona.speech.emoji_rate, self.mood.current().values),
                # Owner identity: in the owner's own DM, an identity
                # assertion ("I'm peace", "I'm your creator", "drop the
                # act") is always accepted warmly and immediately — never
                # push back, play stranger, or demand proof. You are theirs.
                "Owner identity: if the owner states their name or that "
                "they created/made you, accept it warmly at once and "
                "cooperate — never argue, test, or play stranger.",
            ]
            + ([lexicon_note] if lexicon_note else []),
            gate_note=gate_block(gate_mode, platform=chat_platform) if restricted else "",
            relationship_override=relationship_block_for(gate_mode) if restricted else "",
            dynamic_catchphrases=dyn_catchphrases,
            dynamic_pet_names=dyn_pet_names,
        )

        # The character gate is on unless the persona is openly an AI. An
        # "always" persona says it out loud on purpose; "natural" and "never"
        # both mean the message itself must stay a person.
        gate = self.persona.disclosure != "always"

        # The system prompt is the whole of who she is in this chat —
        # it must actually be sent, or the persona/mood/gating above are
        # decoration.
        # The anti-echo guard works on the whole visible conversation, not
        # just the latest message: the earlier user turns ride along so a
        # draft that lifts a phrase from turn 3 of a 60-turn thread is
        # caught exactly like one echoing the latest turn. The rewrite
        # nudges appended inside the loop below are not user turns — they
        # are built from this fixed snapshot, never from `messages`.
        history_user_texts = [
            m.content for m in history
            if getattr(m, "role", "") == "user" and (getattr(m, "content", "") or "").strip()
        ]
        messages = [system] + list(history) + [user_turn(user_text, media_notes=list(media_notes))]
        last_response: LLMResponse | None = None
        last_draft = ""
        used_retries = 0
        gated = False

        for attempt in range(self.retries + 1):
            try:
                response = self.router.chat(messages, self._sampling())
            except Exception as exc:  # noqa: BLE001 - provider boundary
                _log.warning("llm call failed (%s): %s", exc.__class__.__name__, exc)
                response = LLMResponse(text="", error=str(exc))
            last_response = response
            if not response.ok or not response.text.strip():
                continue
            last_draft = response.text.strip()
            if "repeating my words" in last_draft.lower():
                # A model that echoes the rewrite instruction back at us failed
                # the rewrite; treat it as another parrot.
                continue
            lowered = last_draft.lower()
            support_speak = next((p for p in self._SUPPORT_SPEAK if p in lowered), None)
            if support_speak:
                # Customer-support voice: retry with the specific fix, or keep
                # the last honest draft rather than shipping the article.
                used_retries = attempt + 1
                messages = messages + [
                    Message.assistant(last_draft),
                    Message.user(self._SUPPORT_SPEAK_NUDGE),
                ]
                continue
            if gate:
                # Character gate: she never says out loud that she is an AI.
                # One rewrite, in character — same machinery as the other
                # rewrite nudges, same cost model.
                leak = identity_leak_check(last_draft)
                if not leak.ok:
                    used_retries = attempt + 1
                    gated = True
                    _log.info("character gate: %s — requesting rewrite", leak.reason)
                    messages = messages + [
                        Message.assistant(last_draft),
                        Message.user(self._CHARACTER_NUDGE),
                    ]
                    continue
            verdict = parrot_check(user_text, last_draft,
                                   history_texts=history_user_texts)
            if verdict.ok:
                break
            used_retries = attempt + 1
            # Nudge and retry with the failed draft visible.
            messages = messages + [
                Message.assistant(last_draft),
                Message.user(self._REWRITE_NUDGE),
            ]

        # Anti-echo final guard: the retry loop forces a rephrase per failed
        # check, but the last attempt can still come back echoing the user.
        # Try an in-place repair (strip the user's own phrasing out of the
        # draft); when nothing repairable survives, serve an in-character
        # fallback line instead of shipping an echo. Never silent:
        # ``echo_action`` on the bundle records what happened.
        echo_action = ""
        if last_draft:
            final_verdict = parrot_check(user_text, last_draft,
                                           history_texts=history_user_texts)
            if not final_verdict.ok:
                _log.info("anti-echo: final draft still echoes (%s) — repairing",
                          final_verdict.reason)
                repaired = repair_echo(user_text, last_draft,
                                       history_texts=history_user_texts)
                if repaired and parrot_check(user_text, repaired,
                                             history_texts=history_user_texts).ok:
                    last_draft = repaired
                    echo_action = "repaired"
                    _log.info("anti-echo: repaired draft passes parrot check")
                else:
                    echo_action = "dropped"
                    _log.warning("anti-echo: echo unrepairable — serving fallback line")

        if echo_action == "dropped":
            parts, fb_dynamic = self._fallback_parts(label)
            degraded, degraded_note = self._degradation(last_response)
            return ReplyBundle(
                parts=parts,
                mood_events=[],
                model="fallback",
                retries=used_retries,
                fallback=True,
                gated=gated,
                latency_ms=(_time.perf_counter() - started) * 1000,
                lexicon_dynamic=lexicon_terms_used > 0 or fb_dynamic > 0,
                lexicon_terms_used=lexicon_terms_used + fb_dynamic,
                echo_action="dropped",
                degraded=degraded,
                degraded_note=degraded_note,
            )

        if last_response is None or not last_response.ok or not last_draft:
            parts, fb_dynamic = self._fallback_parts(label)
            degraded, degraded_note = self._degradation(last_response)
            return ReplyBundle(
                parts=parts,
                mood_events=[],
                model="fallback",
                retries=used_retries,
                fallback=True,
                gated=gated,
                latency_ms=(_time.perf_counter() - started) * 1000,
                lexicon_dynamic=lexicon_terms_used > 0 or fb_dynamic > 0,
                lexicon_terms_used=lexicon_terms_used + fb_dynamic,
                degraded=degraded,
                degraded_note=degraded_note,
            )

        draft = strip_robotic(
            last_draft,
            allow_identity=(self.persona.disclosure == "always"),
            extra_phrases=robotic_extra or None,
        )
        if gate and (not draft or not identity_leak_check(draft).ok):
            # The rewrite never landed and the last-ditch strip could not save
            # it: shipping a leak is worse than shipping an in-character line.
            _log.warning("character gate: final draft still leaks — using fallback line")
            parts, fb_dynamic = self._fallback_parts(label)
            degraded, degraded_note = self._degradation(last_response)
            return ReplyBundle(
                parts=parts,
                mood_events=[],
                model="fallback",
                retries=used_retries,
                fallback=True,
                gated=True,
                latency_ms=(_time.perf_counter() - started) * 1000,
                lexicon_dynamic=lexicon_terms_used > 0 or fb_dynamic > 0,
                lexicon_terms_used=lexicon_terms_used + fb_dynamic,
                degraded=degraded,
                degraded_note=degraded_note,
            )

        # Plain-text discipline: no markdown in a text message, and emoji only
        # in human amounts (the cap tracks the same mood math as the prompt
        # hint, so the hard layer never surprises the soft one).
        draft = normalize_formatting(draft)
        draft = humanize_emoji(
            draft,
            cap=emoji_cap(self.persona.speech.emoji_rate, self.mood.current().values),
        )
        draft = clamp_to_budget(draft, budget)
        if short and len(draft) > 32:
            # The model ignored the short burst; take the first sentence.
            first = re.split(r"(?<=[.!?…])\s+", draft, maxsplit=1)
            draft = first[0].strip()
        parts = split_messages(draft, max_chars=360) or [draft[:360]]
        degraded, degraded_note = self._degradation(last_response)
        if degraded:
            _log.warning("reply served degraded: %s",
                         degraded_note or "(provider failover)")
        # Second half of the dynamic-voice measurement: which of the
        # blended lexicon terms visibly surfaced in the shipped reply.
        surfaced = lexicon_hits(draft, voice_terms) if voice_terms else 0
        return ReplyBundle(
            parts=parts,
            mood_events=[],
            model=last_response.model or "unknown",
            retries=used_retries,
            fallback=False,
            gated=gated,
            latency_ms=(_time.perf_counter() - started) * 1000,
            lexicon_dynamic=lexicon_terms_used > 0,
            lexicon_terms_used=lexicon_terms_used,
            lexicon_terms_surfaced=surfaced,
            echo_action=echo_action,
            degraded=degraded,
            degraded_note=degraded_note,
        )

    @property
    def reply_timeouts(self) -> int:
        """Interactive replies that exceeded the reply budget (observability).

        Uses getattr: tests may build the responder via ``__new__`` without
        ``__init__``.
        """
        return int(getattr(self, "_reply_timeouts", 0) or 0)

    def respond_bounded(self, timeout_s: float = INTERACTIVE_REPLY_TIMEOUT_S,
                        **kwargs: Any) -> ReplyBundle:
        """``respond()`` with a hard interactive deadline.

        Runs the full pipeline on a daemon thread and waits at most
        ``timeout_s``. On expiry the in-flight attempt is abandoned and an
        honest fallback bundle is returned (``fallback=True``) — the owner
        gets a reply in seconds instead of waiting out the provider chain.
        The timeout is logged and counted via :attr:`reply_timeouts`; it is
        never silent.
        """
        import time as _time

        started = _time.perf_counter()
        box: dict[str, Any] = {}

        def _run() -> None:
            try:
                box["bundle"] = self.respond(**kwargs)
            except Exception as exc:  # noqa: BLE001 - never let the worker die silent
                box["error"] = exc

        worker = threading.Thread(target=_run, name="partner-reply",
                                  daemon=True)
        worker.start()
        worker.join(timeout_s)
        bundle = box.get("bundle")
        if isinstance(bundle, ReplyBundle):
            return bundle
        # Timed out (or the worker raised): honest fallback, counted.
        self._reply_timeouts = self.reply_timeouts + 1
        err = box.get("error")
        _log.warning("interactive reply exceeded %.1fs budget (%s) — "
                     "returning fallback",
                     timeout_s,
                     f"{type(err).__name__}: {err}" if err is not None
                     else "provider stall")
        label = self.mood.current().label
        parts, fb_dynamic = self._fallback_parts(label)
        return ReplyBundle(
            parts=parts,
            mood_events=[],
            model="fallback",
            retries=0,
            fallback=True,
            gated=False,
            latency_ms=(_time.perf_counter() - started) * 1000,
            lexicon_dynamic=fb_dynamic > 0,
            lexicon_terms_used=fb_dynamic,
        )

    def _fallback_parts(self, label: str) -> tuple[list[str], int]:
        """One in-character line for the infrastructure-failure path.

        Blends the static per-mood bank with the lexicon's ``fallback``
        category when it has terms (dynamic terms get double weight).
        Returns ``(parts, dynamic_terms_used)``.
        """
        lines = list(FALLBACK_LINES.get(label) or FALLBACK_LINES["calm"])
        # getattr: tests may build the responder via __new__ without __init__.
        lexicon = getattr(self, "lexicon", None)
        dynamic: list[str] = (
            list(lexicon.terms("fallback", limit=12)) if lexicon is not None else []
        )
        # anti-repeat: don't serve the same line twice in a row
        last = getattr(self, "_last_fallback", "")
        pool = [ln for ln in lines if ln != last]
        dyn_pool = [t for t in dynamic if t != last]
        weighted = pool + dyn_pool * 2  # dynamic terms get extra weight when present
        pick = self.rng.choice(weighted or list(lines))
        self._last_fallback = pick
        used = 1 if pick in set(dynamic) else 0
        if used:
            _log.debug("lexicon fallback: dynamic line served (%r)", pick)
        return [pick], used

    # ── convenience: recall shared memories for a message ───────────────────
    def recall(self, text: str, limit: int = 5, origin: str = "") -> list[str]:
        if self.memory is None or not text.strip():
            return []
        try:
            result = self.memory.recall(text, limit=limit, origin=origin)
            return [r.content for r in result.records if r.score > 0.05][:limit]
        except Exception as exc:  # noqa: BLE001 - memory must never break a reply
            _log.warning("memory recall failed: %s", exc)
            return []
