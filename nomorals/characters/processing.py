"""Post-interaction processing: the background pass.

After a conversation, scene, or game, a headless pass reads what
happened and updates the slow-moving state — relationships, mood,
beliefs, memory salience. The talking agent stays focused; this does
the heavy bookkeeping (Musubi gold).

Call ``process_session`` after any significant character interaction.
It's cheap, synchronous, and never raises.
"""
from __future__ import annotations

import time
from typing import Any

from .arcs import grow_from_interaction
from .character import Character
from .relationships import RelationshipGraph
from .voice import fingerprint as _voice_fingerprint

__all__ = ["process_session", "SessionEvent"]


class SessionEvent:
    """One thing that happened, from one character's perspective."""
    def __init__(self, char_id: str, other_id: str, event: str,
                 salience: float = 0.5, note: str = "") -> None:
        self.char_id = char_id
        self.other_id = other_id
        self.event = event            # key in EVENT_DELTAS
        self.salience = max(0.0, min(1.0, salience))
        self.note = note[:300]
        self.ts = time.time()


# mood shifts per event (fast-moving, decays naturally)
MOOD_DELTAS: dict[str, dict[str, float]] = {
    "good_conversation": {"valence": 0.15, "arousal": 0.1},
    "deep_conversation": {"valence": 0.25, "arousal": 0.15, "trust": 0.1},
    "laughed_together": {"valence": 0.3, "arousal": 0.25},
    "helped": {"valence": 0.15, "trust": 0.1},
    "was_helped_by": {"valence": 0.2, "trust": 0.12},
    "disagreement": {"valence": -0.1, "arousal": 0.2},
    "argument": {"valence": -0.35, "arousal": 0.35, "trust": -0.1},
    "made_up": {"valence": 0.3, "arousal": -0.15},
    "betrayed": {"valence": -0.5, "arousal": 0.3, "trust": -0.3},
    "won_game": {"valence": 0.25, "arousal": 0.2},
    "lost_game": {"valence": -0.15, "arousal": 0.1},
    "praised": {"valence": 0.25},
    "insulted": {"valence": -0.3, "arousal": 0.25},
    # ── sweep additions ──
    "secret_shared": {"valence": 0.2, "trust": 0.15, "arousal": 0.1},
    "opened_up": {"valence": 0.2, "trust": 0.12},
    "defended": {"valence": 0.2, "trust": 0.1, "arousal": 0.15},
    "celebrated": {"valence": 0.35, "arousal": 0.3},
    "forgave": {"valence": 0.25, "arousal": -0.1},
    "abandoned": {"valence": -0.45, "trust": -0.25, "arousal": 0.2},
    "stood_up_to": {"valence": -0.05, "arousal": 0.25},
    "missed_them": {"valence": -0.1},
}


def _shift_mood(char: Character, event: str) -> None:
    d = MOOD_DELTAS.get(event)
    if not d:
        return
    m = char.mood
    m["valence"] = max(-1.0, min(1.0, m["valence"] + d.get("valence", 0)))
    m["arousal"] = max(0.0, min(1.0, m["arousal"] + d.get("arousal", 0)))
    m["trust"] = max(0.0, min(1.0, m["trust"] + d.get("trust", 0)))
    # mood drifts back toward neutral over time — handled by decay_moods


def decay_moods(chars: list[Character],
                now: float | None = None) -> None:
    """Moods are fast-moving: pull toward baseline. Call periodically."""
    now = now or time.time()
    for c in chars:
        idle_h = (now - c.last_active) / 3600.0
        if idle_h < 1:
            continue
        pull = min(0.6, idle_h / 24.0 * 0.3)
        m = c.mood
        m["valence"] *= (1 - pull)
        m["arousal"] = 0.3 + (m["arousal"] - 0.3) * (1 - pull)
        m["trust"] = 0.5 + (m["trust"] - 0.5) * (1 - pull * 0.5)


def process_session(events: list[SessionEvent],
                    chars: dict[str, Character],
                    graph: RelationshipGraph,
                    suggest: Any = None) -> dict[str, Any]:
    """Background pass over a session's events. Never raises.

    ``suggest`` (optional): model for the reflection step — Stanford's
    critical piece. Without it, reflection falls back to a heuristic.
    """
    processed = 0
    reflections = 0
    drift_flags: list[str] = []
    try:
        for ev in events:
            char = chars.get(ev.char_id)
            if char is None:
                continue
            # 1. relationships shift (slow)
            graph.interact(ev.char_id, ev.other_id, ev.event)
            # 2. mood shifts (fast)
            _shift_mood(char, ev.event)
            # 3. memory write for notable events
            if ev.salience >= 0.6 or ev.event in (
                    "betrayed", "deep_conversation", "argument", "made_up"):
                note = ev.note or f"{ev.event} with {ev.other_id}"
                char.remember(note, ev.salience)
            # 4. growth hook
            other_name = (chars[ev.other_id].name
                          if ev.other_id in chars else ev.other_id)
            try:
                grow_from_interaction(char, other_name, ev.event)
            except Exception:
                pass
            char.last_active = time.time()
            processed += 1
        # 5. relationship decay pass (cheap, idempotent)
        try:
            graph.decay_all()
        except Exception:
            pass
        # 6. voice fingerprint refresh — characters stay in voice
        try:
            _refresh_voice(chars, events)
        except Exception:
            pass
        # 7. voice drift check — fingerprint exists to be USED
        try:
            drift_flags = _check_drift(chars, events)
        except Exception:
            pass
        # 8. reflection — Stanford gold: salient sessions get synthesized
        # into insight, not just logged. This is what keeps characters
        # coherent over months instead of drifting into amnesiacs.
        try:
            reflections = _reflect_session(chars, events, suggest)
        except Exception:
            pass
    except Exception:
        pass
    return {"processed": processed, "reflections": reflections,
            "drift_flags": drift_flags}


def _refresh_voice(chars: dict[str, Character],
                   events: list[SessionEvent]) -> None:
    """Update each character's voice fingerprint from session utterances.

    Utterances ride on SessionEvent.note (last-3-turns join). Fingerprints
    accumulate — they get sharper with more data, never reset.
    """
    by_char: dict[str, list[str]] = {}
    for ev in events:
        note = getattr(ev, "note", "") or ""
        for part in note.split(" | "):
            part = part.strip()
            if part:
                by_char.setdefault(ev.char_id, []).append(part)
    for char_id, utterances in by_char.items():
        char = chars.get(char_id)
        if char is None or not utterances:
            continue
        try:
            catchphrases = list(
                (getattr(char, "expression", None) or {}).get(
                    "catchphrases", ()))
            fp = _voice_fingerprint(utterances, catchphrases)
            old = getattr(char, "voice_fingerprint", None)
            if isinstance(old, dict) and old.get("n_samples", 0) >= 5:
                from collections import Counter
                from .voice import VoiceFingerprint
                old_fp = VoiceFingerprint.from_dict(old)
                total = old_fp.n_samples + fp.n_samples
                w_old = old_fp.n_samples / total
                w_new = fp.n_samples / total
                merged: Counter[str] = Counter()
                for w, c in old_fp.top_words:
                    merged[w] += int(c * w_old)
                for w, c in fp.top_words:
                    merged[w] += int(c * w_new)
                # function-word rates merge as weighted averages
                func: dict[str, float] = {}
                for w, r in old_fp.function_words:
                    func[w] = func.get(w, 0.0) + r * w_old
                for w, r in fp.function_words:
                    func[w] = func.get(w, 0.0) + r * w_new
                char.voice_fingerprint = {
                    "top_words": [[w, c] for w, c in merged.most_common(25)],
                    "mean_sentence_len": (
                        old_fp.mean_sentence_len * w_old
                        + fp.mean_sentence_len * w_new),
                    "sentence_len_spread": max(
                        old_fp.sentence_len_spread, fp.sentence_len_spread),
                    "emoji_rate": (old_fp.emoji_rate * w_old
                                   + fp.emoji_rate * w_new),
                    "question_rate": (old_fp.question_rate * w_old
                                      + fp.question_rate * w_new),
                    "exclaim_rate": (old_fp.exclaim_rate * w_old
                                     + fp.exclaim_rate * w_new),
                    "catchphrase_hits": (old_fp.catchphrase_hits * w_old
                                         + fp.catchphrase_hits * w_new),
                    "n_samples": total,
                    "function_words": [[w, round(r, 2)]
                                       for w, r in sorted(
                                           func.items(), key=lambda kv: -kv[1])[:20]],
                    "ttr": old_fp.ttr * w_old + fp.ttr * w_new,
                    "mean_word_len": (old_fp.mean_word_len * w_old
                                      + fp.mean_word_len * w_new),
                }
                # preserve any active drift flag across refreshes
                if old.get("drift_flag"):
                    char.voice_fingerprint["drift_flag"] = old["drift_flag"]
            else:
                char.voice_fingerprint = fp.to_dict()
        except Exception:
            pass


def _check_drift(chars: dict[str, Character],
                 events: list[SessionEvent]) -> list[str]:
    """Score this session's utterances against each character's stored
    fingerprint. Flags drift — the fingerprint finally gets USED, not
    just accumulated. Sets ``drift_flag`` on the fingerprint dict so
    the context builder can nudge the character back into voice."""
    from .voice import VoiceFingerprint, consistency
    flags: list[str] = []
    by_char: dict[str, list[str]] = {}
    for ev in events:
        note = getattr(ev, "note", "") or ""
        for part in note.split(" | "):
            part = part.strip()
            if part:
                by_char.setdefault(ev.char_id, []).append(part)
    for char_id, utterances in by_char.items():
        char = chars.get(char_id)
        if char is None or len(utterances) < 2:
            continue
        old = getattr(char, "voice_fingerprint", None)
        if not isinstance(old, dict) or int(old.get("n_samples", 0)) < 5:
            continue
        try:
            fp = VoiceFingerprint.from_dict(old)
            catchphrases = list(
                (getattr(char, "expression", None) or {}).get(
                    "catchphrases", ()))
            scores = [consistency(fp, u, catchphrases) for u in utterances]
            avg = sum(scores) / len(scores)
            if avg < 0.45:
                note = (f"sounding off-session (consistency {avg:.2f} "
                        f"over {len(utterances)} lines)")
                char.voice_fingerprint["drift_flag"] = note
                flags.append(f"{char.name}: {note}")
            else:
                char.voice_fingerprint.pop("drift_flag", None)
        except Exception:
            continue
    return flags


def _reflect_session(chars: dict[str, Character],
                     events: list[SessionEvent],
                     suggest: Any) -> int:
    """Stanford gold: after a salient session, each involved character
    synthesizes what happened into insight. Only fires when the session
    was actually notable — reflection is for meaning, not logistics."""
    notable_kinds = {"deep_conversation", "betrayed", "argument", "made_up",
                     "secret_shared", "opened_up", "celebrated", "forgave",
                     "abandoned"}
    per_char: dict[str, float] = {}
    for ev in events:
        w = 1.5 if ev.event in notable_kinds else 1.0
        per_char[ev.char_id] = per_char.get(ev.char_id, 0.0) + \
            ev.salience * w
    made = 0
    for char_id, weight in per_char.items():
        if weight < 1.2:  # not notable enough to reflect on
            continue
        char = chars.get(char_id)
        if char is None:
            continue
        try:
            insights = char.reflect(suggest)
            made += len(insights)
        except Exception:
            continue
    return made


def events_from_dialogue(char_id: str, other_id: str,
                         turns: list[Any],
                         kind: str = "good_conversation") -> list[SessionEvent]:
    """Convenience: build events from a dialogue's turns."""
    if not turns:
        return []
    # one event per participant pair per dialogue — the relationship
    # moves once per session, not per message (slow by design)
    salience = 0.7 if kind == "deep_conversation" else 0.5
    note = " | ".join(str(getattr(t, "text", t))[:80] for t in turns[-3:])
    return [SessionEvent(char_id, other_id, kind,
                         salience=salience, note=note)]
