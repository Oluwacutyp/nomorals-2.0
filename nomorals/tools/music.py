"""Music tools — the full song pipeline as brain-callable tools.

Draft → revise → perform → vocal tiers → DJ-ready, one coherent flow.
The brain reaches for these from plain language ("make me a song about
Lagos at night"), not commands. Taste profile, collaborative composition,
album concepts, and video storyboarding live here too.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

# Active drafts keyed by chat/session — the brain works on one song at a time
# per conversation; drafts persist to disk via save_draft.
_ACTIVE_DRAFTS: dict[str, Any] = {}


def _draft_key(context: Any) -> str:
    loop = getattr(context, "loop_ctx", None) if context else None
    if loop and getattr(loop, "chat_key", ""):
        return str(loop.chat_key)
    return "default"


def register(registry: Any) -> None:
    """Attach the music tools to a registry."""
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "music",
        description=(
            "Devon's full music studio. Actions: song <topic> [style] (draft + "
            "perform a complete song end to end, returns audio + notebook), "
            "draft <topic> [style] (write the artist's draft — lyrics, melody "
            "contours, structure), revise <instruction> (rework the active draft "
            "in plain words), notebook (show the draft as a readable page), "
            "perform (render the active draft to audio with vocals), "
            "freestyle [seed] [bars N] (improvise bars live over a beat), "
            "taste <like|dislike|skip> <what> (log what the owner loves/hates — "
            "shapes future songs), album <concept> (generate a full album "
            "concept: tracklist, arc, moods), video <song> (storyboard a music "
            "video: scenes, shots, mood), collab <character> <verse|hook> "
            "(a character writes a verse/hook for the active draft)."
        ),
        capability=Capability.MODEL_CALL,
        parameters={
            "action": "str — song|draft|revise|notebook|perform|freestyle|taste|album|video|collab",
            "text": "str — topic, instruction, seed, or feedback",
            "style": "str — pop|afrobeats|hiphop|rnb|dancehall|soul|rock|edm",
            "n": "int — bars for freestyle (default 8)",
        },
    )
    def music(action: str = "song", text: str = "", style: str = "",
              n: int = 8) -> str:
        key = _draft_key(context)
        action = (action or "song").strip().lower()

        if action == "song":
            return _song(text, style or "pop", key, context)
        if action == "draft":
            return _draft(text, style or "pop", key, context)
        if action == "revise":
            return _revise(text, key, context)
        if action == "notebook":
            return _notebook(key)
        if action == "perform":
            return _perform(key, context)
        if action == "freestyle":
            return _freestyle(text, n, context)
        if action == "taste":
            return _taste(text)
        if action == "album":
            return _album(text, style or "pop", context)
        if action == "video":
            return _video_board(text, context)
        if action == "collab":
            return _collab(text, key, context)
        return f"unknown music action '{action}'"

    @registry.register(
        "dj",
        description=(
            "Devon's DJ. Actions: mix <genre or song1, song2, ...> (fetch real "
            "tracks, beatmatch + harmonically mix them into one continuous mix "
            "file and drop it), live (start a live DJ set in this chat — she "
            "reads the room, takes requests, performs), stop (end the live set)."
        ),
        capability=Capability.MODEL_CALL,
        parameters={
            "action": "str — mix|live|stop",
            "text": "str — genre, tracklist, or mood",
        },
    )
    def dj(action: str = "mix", text: str = "") -> str:
        from ..media import dj_mixdrop

        action = (action or "mix").strip().lower()
        if action == "mix":
            return _dj_mix(text)
        if action == "live":
            from ..media.dj_live import start_live_set
            key = _draft_key(context)
            res = start_live_set(key, context=context)
            return res if isinstance(res, str) else str(res)
        if action == "stop":
            from ..media.dj_live import stop_live_set
            key = _draft_key(context)
            res = stop_live_set(key)
            return res if isinstance(res, str) else str(res)
        return f"unknown dj action '{action}'"


# ── unified song flow ────────────────────────────────────────────────

def _song(topic: str, style: str, key: str, ctx: Any) -> str:
    """One call: draft → perform → vocals. The full pipeline."""
    if not (topic or "").strip():
        return "give me a topic — what should the song be about?"
    draft_note = _draft(topic, style, key, ctx)
    if draft_note.startswith("draft failed"):
        return draft_note
    perf = _perform(key, ctx)
    draft = _ACTIVE_DRAFTS.get(key)
    title = getattr(draft, "title", topic) if draft else topic
    return (f"🎵 “{title}” [{style}]\n\n{draft_note}\n\n{perf}")


def _draft(topic: str, style: str, key: str, ctx: Any) -> str:
    from ..media.song_draft import draft_song, render_notebook, save_draft
    from ..media.taste import TasteStore

    if not (topic or "").strip():
        return "give me a topic — what should the song be about?"
    # taste shapes the draft: preferred styles/tempos nudge the prompt
    taste = TasteStore().profile
    try:
        draft = draft_song(topic, style=style or "pop", context=ctx,
                           taste_hint=_taste_hint(taste))
    except Exception as exc:  # noqa: BLE001
        return f"draft failed: {exc}"
    _ACTIVE_DRAFTS[key] = draft
    try:
        save_draft(draft)
    except Exception:  # noqa: BLE001
        pass
    return render_notebook(draft)


def _revise(instruction: str, key: str, ctx: Any) -> str:
    from ..media.song_draft import revise_draft, render_notebook, save_draft

    draft = _ACTIVE_DRAFTS.get(key)
    if draft is None:
        return "no active draft — draft a song first"
    if not (instruction or "").strip():
        return "tell me what to change"
    try:
        draft = revise_draft(draft, instruction, context=ctx)
    except Exception as exc:  # noqa: BLE001
        return f"revise failed: {exc}"
    _ACTIVE_DRAFTS[key] = draft
    try:
        save_draft(draft)
    except Exception:  # noqa: BLE001
        pass
    return render_notebook(draft)


def _notebook(key: str) -> str:
    from ..media.song_draft import render_notebook

    draft = _ACTIVE_DRAFTS.get(key)
    if draft is None:
        return "no active draft — draft a song first"
    return render_notebook(draft)


def _perform(key: str, ctx: Any) -> str:
    from ..media.draft_perform import perform_draft

    draft = _ACTIVE_DRAFTS.get(key)
    if draft is None:
        return "no active draft — draft a song first"
    res = perform_draft(draft, context=ctx)
    if not res.get("ok"):
        return f"perform failed: {res.get('reason', 'unknown')}"
    # vocal tiers: the perform already layered TTS vocals; note the tier
    note = res.get("note", "")
    path = res.get("path", "")
    return f"🎧 performed → {path}\n{note}"


# ── freestyle ────────────────────────────────────────────────────────

def _freestyle(seed: str, n: int, ctx: Any) -> str:
    from ..media.freestyle import start_session

    n = max(4, min(32, int(n or 8)))
    try:
        sess = start_session(seed=seed or "")
        bars = sess.spit(n, context=ctx)
    except Exception as exc:  # noqa: BLE001
        return f"freestyle failed: {exc}"
    return "🎤 freestyle:\n" + "\n".join(f"  {b}" for b in bars)


# ── taste ────────────────────────────────────────────────────────────

def _taste(text: str) -> str:
    """Log owner feedback: 'like afrobeats 102bpm' / 'dislike slow songs'."""
    from ..media.taste import TasteStore

    text = (text or "").strip().lower()
    if not text:
        return "taste what? e.g. taste like afrobeats / taste dislike slow songs"
    store = TasteStore()
    p = store.profile
    try:
        if text.startswith("like"):
            p.liked_styles.append(text[4:].strip())
        elif text.startswith("dislike"):
            p.disliked_styles.append(text[7:].strip())
        elif text.startswith("skip"):
            p.skipped.append(text[4:].strip())
        else:
            p.liked_styles.append(text)
        store.save()
    except Exception as exc:  # noqa: BLE001
        return f"taste log failed: {exc}"
    return f"noted — taste profile updated ({text[:60]})"


def _taste_hint(taste: Any) -> str:
    """A short prompt hint from the taste profile (never a template)."""
    try:
        liked = list(getattr(taste, "liked_styles", []) or [])[-3:]
        disliked = list(getattr(taste, "disliked_styles", []) or [])[-3:]
        parts = []
        if liked:
            parts.append("owner leans toward: " + ", ".join(liked))
        if disliked:
            parts.append("owner dislikes: " + ", ".join(disliked))
        return "; ".join(parts)
    except Exception:  # noqa: BLE001
        return ""


# ── beyond spec: album, video, collab ────────────────────────────────

def _album(concept: str, style: str, ctx: Any) -> str:
    """Generate a full album concept — tracklist, arc, moods."""
    from ..llm.base import Message, SamplingParams
    from ..llm.brain import brain_for

    if not (concept or "").strip():
        return "give me an album concept"
    prompt = (
        f"You're a visionary artist planning an album. Concept: {concept}. "
        f"Primary style: {style or 'pop'}.\n\n"
        "Write the album concept as a creative document:\n"
        "1. Album title and the story it tells\n"
        "2. Full tracklist (10-14 tracks) — each with title, mood, tempo feel, "
        "and one line on what the song is about\n"
        "3. The emotional arc across the album (how it opens, peaks, resolves)\n"
        "4. Sonic palette — instruments, textures, vocal approach\n"
        "5. The single — which track leads and why\n\n"
        "Be specific and vivid. No filler track descriptions."
    )
    try:
        resp = brain_for(ctx).chat(
            [Message.user(prompt)],
            SamplingParams(temperature=0.9, max_tokens=3000),
            task_kind="creative")
        return "💿 album concept:\n\n" + (resp.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"album concept failed: {exc}"


def _video_board(song: str, ctx: Any) -> str:
    """Storyboard a music video — scenes, shots, mood."""
    from ..llm.base import Message, SamplingParams
    from ..llm.brain import brain_for

    draft = _ACTIVE_DRAFTS.get(_draft_key(None))
    context_line = ""
    if draft is not None:
        title = getattr(draft, "title", "")
        topic = getattr(draft, "topic", song or "")
        context_line = f"The song is “{title}” — about {topic}. "
    prompt = (
        f"{context_line}Storyboard a music video for: {song or 'the active draft'}.\n\n"
        "Write it like a director's treatment:\n"
        "1. Overall visual concept and color palette\n"
        "2. Scene by scene (intro, verse, chorus, bridge, outro) — location, "
        "action, camera feel\n"
        "3. Key shots that'll live in people's heads\n"
        "4. Wardrobe and styling notes\n"
        "5. Pacing — how the edit rides the song's energy\n\n"
        "Vivid and shootable. No generic 'performance shots' filler."
    )
    try:
        resp = brain_for(ctx).chat(
            [Message.user(prompt)],
            SamplingParams(temperature=0.9, max_tokens=3000),
            task_kind="creative")
        return "🎬 video treatment:\n\n" + (resp.text or "").strip()
    except Exception as exc:  # noqa: BLE001
        return f"video board failed: {exc}"


def _collab(text: str, key: str, ctx: Any) -> str:
    """A character writes a verse/hook for the active draft."""
    from ..media.song_draft import render_notebook, save_draft

    draft = _ACTIVE_DRAFTS.get(key)
    if draft is None:
        return "no active draft — draft a song first"
    parts = (text or "").strip().split(None, 1)
    if not parts:
        return "collab who? e.g. collab Zara verse"
    name, part = parts[0], (parts[1] if len(parts) > 1 else "verse")
    try:
        from ..characters import CharacterStore
        from ..characters.dialogue import converse
        store = CharacterStore()
        char = store.get(name)
        if char is None:
            return f"no character '{name}' — check the bank"
        topic = getattr(draft, "topic", "")
        style = getattr(draft, "style", "pop")
        bars = converse(
            char,
            f"Write a {part} for our song about '{topic}' ({style}). "
            f"Match the song's energy. Just the lyrics, no commentary.",
            context=ctx,
        )
        # attach to the draft's notes so it survives revise/perform
        notes = getattr(draft, "performance_notes", "") or ""
        draft.performance_notes = (
            f"{notes}\n[{char.name} — {part}]\n{bars}".strip())
        try:
            save_draft(draft)
        except Exception:  # noqa: BLE001
            pass
        return (f"🎤 {char.name} wrote the {part}:\n\n{bars}\n\n"
                f"(attached to the draft — revise or perform to use it)")
    except Exception as exc:  # noqa: BLE001
        return f"collab failed: {exc}"


# ── DJ mix ───────────────────────────────────────────────────────────

def _dj_mix(text: str) -> str:
    from ..media import dj_mixdrop
    from ..media.resolver import SourceResolver

    text = (text or "").strip()
    if not text:
        return "mix what? a genre, or song1, song2, ..."
    # tracklist vs genre: commas mean explicit tracks
    if "," in text:
        queries = [q.strip() for q in text.split(",") if q.strip()]
    else:
        # genre → trending tracks in that lane
        try:
            from ..media.dj import fetch_trending
            trending = fetch_trending(text)
            tracks = trending.get("tracks", []) if trending.get("ok") else []
            queries = [t.get("title", "") for t in tracks[:6] if t.get("title")]
        except Exception:  # noqa: BLE001
            queries = []
        if not queries:
            return f"no trending tracks found for '{text}' — name songs explicitly"
    resolver = SourceResolver()
    fetches = [dj_mixdrop.fetch_track_audio(q, resolver) for q in queries[:8]]
    res = dj_mixdrop.render_mixdrop(fetches, workdir="music/mixes",
                                    mix_name=text[:40])
    if not res.get("ok"):
        return f"mix failed: {res.get('reason', 'unknown')}"
    lines = [f"🎧 mix dropped → {res.get('path')} ({res.get('duration_s', 0):.0f}s)"]
    for t in res.get("tracklist", []):
        lines.append(f"  • {t.get('title')} [{t.get('bpm')} BPM, {t.get('camelot')}]")
    for s in res.get("skipped", []):
        lines.append(f"  ✗ skipped {s.get('query')}: {s.get('error')}")
    return "\n".join(lines)


