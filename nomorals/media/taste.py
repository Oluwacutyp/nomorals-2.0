"""Taste memory — a real producer knows their artist's taste.

``/produce`` used to need a Spotify link or a vibe description every
time.  This module remembers what Devon has produced, what the owner
liked and disliked, and builds a ``ReferenceProfile`` from that memory
when no reference is given — so "make me something" just works.

Everything is forgiving: corrupt or missing JSON starts fresh, feedback
parsing never raises, and an empty history yields ``None`` from
``suggest_profile()`` so the caller can say "no taste yet" honestly.
"""

from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "TasteProfile", "TasteStore", "load_taste", "modify_profile",
    "detect_produce_intent", "detect_feedback",
]

_FILENAME = "taste.json"

#: energy words we track with weights; anything else passes through.
_TRACKED_ENERGY = (
    "driving", "dark", "melancholic", "relentless", "euphoric", "chill",
    "aggressive", "dreamy", "menacing", "hard", "soft", "melodic",
    "uplifting", "heavy", "hypnotic", "raw",
)

#: iteration verbs → how they reshape a profile.
_ITERATE = (
    (("harder", "heavier", "more energy", "go harder"), {"bpm_delta": 8, "add_energy": ("aggressive", "driving"), "remove_energy": ("chill", "dreamy", "soft")}),
    (("faster", "speed it up", "quicker"), {"bpm_delta": 14, "add_energy": ("driving",), "remove_energy": ("chill",)}),
    (("slower", "slow it down", "chill out"), {"bpm_delta": -14, "add_energy": ("chill",), "remove_energy": ("aggressive", "driving")}),
    (("more melodic", "melodier", "prettier", "softer"), {"bpm_delta": 0, "add_energy": ("melodic", "dreamy"), "remove_energy": ("aggressive", "menacing")}),
    (("darker", "more dark"), {"bpm_delta": 0, "add_energy": ("dark", "menacing"), "remove_energy": ("euphoric", "uplifting")}),
    (("lighter", "happier", "more uplifting"), {"bpm_delta": 0, "add_energy": ("uplifting", "euphoric"), "remove_energy": ("dark", "menacing", "melancholic")}),
    (("softer", "gentler", "calmer"), {"bpm_delta": -8, "add_energy": ("soft", "chill"), "remove_energy": ("aggressive", "hard")}),
)

#: feedback hints → preference nudges.
_FEEDBACK_HINTS = (
    (("too slow", "too chill", "boring", "puts me to sleep"), {"bpm_bias": 12, "energy": {"driving": 1.0, "aggressive": 0.5}}),
    (("too fast", "too hectic", "too much"), {"bpm_bias": -12, "energy": {"chill": 1.0, "soft": 0.5}}),
    (("too dark", "too sad", "depressing"), {"bpm_bias": 0, "energy": {"uplifting": 1.0, "euphoric": 0.5}}),
    (("too happy", "too cheesy", "too pop"), {"bpm_bias": 0, "energy": {"dark": 1.0, "raw": 0.5}}),
    (("too hard", "too aggressive", "too loud"), {"bpm_bias": -8, "energy": {"soft": 1.0, "melodic": 0.5}}),
    (("more energy", "needs energy", "go harder"), {"bpm_bias": 10, "energy": {"aggressive": 1.0, "driving": 1.0}}),
    (("love the bass", "bass is fire", "heavy bass"), {"bpm_bias": 0, "energy": {"heavy": 1.5}}),
    (("too repetitive", "repetitive"), {"bpm_bias": 0, "energy": {"hypnotic": -0.5}}),
)


def _taste_path(context: Any = None) -> Path:
    """Where the taste file lives.  Never raises."""
    try:
        settings = getattr(context, "settings", None)
        root = (str(getattr(settings, "workspace_dir", "") or "")).strip()
        if root and os.path.isdir(root):
            return Path(root) / ".nomorals" / _FILENAME
    except Exception:  # noqa: BLE001
        pass
    return Path(os.path.expanduser("~")) / ".nomorals" / _FILENAME


@dataclass
class LikedTrack:
    description: str = ""
    bpm: float = 128.0
    key: str = "A"
    mode: str = "minor"
    energy_words: tuple[str, ...] = ()
    genre: str = ""
    liked: bool = True
    notes: str = ""


@dataclass
class TasteProfile:
    """Everything Devon knows about the owner's musical taste."""
    preferred_tempos: list[list[float]] = field(default_factory=list)
    preferred_keys: list[str] = field(default_factory=list)
    energy_weights: dict[str, float] = field(default_factory=dict)
    genre_weights: dict[str, float] = field(default_factory=dict)
    tracks: list[LikedTrack] = field(default_factory=list)
    production_count: int = 0
    bpm_bias: float = 0.0
    last_production: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["tracks"] = [asdict(t) for t in self.tracks]
        return d

    @classmethod
    def from_dict(cls, raw: Any) -> "TasteProfile":
        try:
            if not isinstance(raw, dict):
                return cls()
            tracks = []
            for t in (raw.get("tracks") or []):
                if isinstance(t, dict):
                    try:
                        tracks.append(LikedTrack(
                            description=str(t.get("description", ""))[:200],
                            bpm=float(t.get("bpm", 128.0)),
                            key=str(t.get("key", "A"))[:4],
                            mode=str(t.get("mode", "minor"))[:8],
                            energy_words=tuple(str(w) for w in (t.get("energy_words") or ()))[:12],
                            genre=str(t.get("genre", ""))[:40],
                            liked=bool(t.get("liked", True)),
                            notes=str(t.get("notes", ""))[:300],
                        ))
                    except Exception:  # noqa: BLE001 - one bad track can't kill the file
                        continue
            tempos = []
            for pair in (raw.get("preferred_tempos") or []):
                try:
                    lo, hi = float(pair[0]), float(pair[1])
                    tempos.append([lo, hi])
                except Exception:  # noqa: BLE001
                    continue
            return cls(
                preferred_tempos=tempos[:20],
                preferred_keys=[str(k)[:4] for k in (raw.get("preferred_keys") or [])][:20],
                energy_weights={str(k): float(v) for k, v in (raw.get("energy_weights") or {}).items() if isinstance(v, (int, float))},
                genre_weights={str(k): float(v) for k, v in (raw.get("genre_weights") or {}).items() if isinstance(v, (int, float))},
                tracks=tracks[-60:],  # keep the file small
                production_count=int(raw.get("production_count", 0) or 0),
                bpm_bias=float(raw.get("bpm_bias", 0.0) or 0.0),
                last_production=raw.get("last_production") if isinstance(raw.get("last_production"), dict) else {},
            )
        except Exception:  # noqa: BLE001 - corrupt file → fresh profile
            return cls()


class TasteStore:
    """Loads/saves the taste profile.  Never raises."""

    def __init__(self, context: Any = None) -> None:
        self.context = context
        self.path = _taste_path(context)
        self.profile = self._load()

    def _load(self) -> TasteProfile:
        try:
            if self.path.is_file():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                return TasteProfile.from_dict(raw)
        except Exception as exc:  # noqa: BLE001 - corrupt JSON starts fresh
            _log.warning("taste file unreadable (%s) — starting fresh", exc)
        return TasteProfile()

    def save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.profile.to_dict(), indent=1),
                           encoding="utf-8")
            tmp.replace(self.path)
            return True
        except Exception as exc:  # noqa: BLE001
            _log.warning("taste save failed: %s", exc)
            return False

    # ── recording ────────────────────────────────────────────────────

    def record_production(self, ref_profile: Any, *, source: str,
                          ref: str = "") -> None:
        """After each /produce: remember what was made."""
        try:
            p = self.profile
            p.production_count += 1
            energy = tuple(w for w in (getattr(ref_profile, "energy_words", ()) or ())
                           if isinstance(w, str))[:12]
            p.last_production = {
                "ts": time.time(),
                "source": source,
                "ref": (ref or "")[:200],
                "bpm": float(getattr(ref_profile, "bpm", 128.0) or 128.0),
                "key": str(getattr(ref_profile, "key", "A"))[:4],
                "mode": str(getattr(ref_profile, "mode", "minor"))[:8],
                "energy_words": list(energy),
                "genre": str(getattr(ref_profile, "genre", ""))[:40],
                "title": str(getattr(ref_profile, "title", ""))[:80],
            }
            # fold the production's characteristics into the running taste
            bpm = p.last_production["bpm"]
            p.preferred_tempos.append([bpm - 8, bpm + 8])
            p.preferred_tempos = p.preferred_tempos[-20:]
            key = p.last_production["key"]
            if key and key not in p.preferred_keys:
                p.preferred_keys.append(key)
                p.preferred_keys = p.preferred_keys[-20:]
            for w in energy:
                w = w.lower()
                if w in _TRACKED_ENERGY:
                    p.energy_weights[w] = p.energy_weights.get(w, 0.0) + 0.25
            genre = p.last_production["genre"]
            if genre:
                p.genre_weights[genre] = p.genre_weights.get(genre, 0.0) + 0.25
            self.save()
        except Exception:  # noqa: BLE001
            _log.exception("record_production failed")

    def record_feedback(self, liked: bool, notes: str = "") -> str:
        """User verdict on the last production.  Returns a short summary."""
        try:
            p = self.profile
            notes = (notes or "").strip()[:300]
            low = notes.lower()
            if not p.last_production:
                return "nothing to rate yet — produce something first."
            # attach to the track record
            lp = p.last_production
            track = LikedTrack(
                description=lp.get("title", "") or lp.get("ref", ""),
                bpm=float(lp.get("bpm", 128.0)),
                key=str(lp.get("key", "A")),
                mode=str(lp.get("mode", "minor")),
                energy_words=tuple(lp.get("energy_words", ())),
                genre=str(lp.get("genre", "")),
                liked=liked,
                notes=notes,
            )
            p.tracks.append(track)
            p.tracks = p.tracks[-60:]
            # weight liked characteristics harder than mere exposure
            delta = 1.0 if liked else -1.0
            for w in track.energy_words:
                w = w.lower()
                if w in _TRACKED_ENERGY:
                    p.energy_weights[w] = p.energy_weights.get(w, 0.0) + 0.75 * delta
            if track.genre:
                p.genre_weights[track.genre] = p.genre_weights.get(track.genre, 0.0) + 0.75 * delta
            # parse the notes for directional hints
            nudges = []
            for hints, nudge in _FEEDBACK_HINTS:
                if any(h in low for h in hints):
                    # directional hints ("too slow" -> faster) apply as
                    # written — they don't flip with like/dislike
                    p.bpm_bias += nudge["bpm_bias"]
                    for w, amt in nudge["energy"].items():
                        p.energy_weights[w] = p.energy_weights.get(w, 0.0) + amt
                    nudges.append(hints[0])
            # drop non-positive weights so taste only pulls toward likes
            p.energy_weights = {k: v for k, v in p.energy_weights.items() if v > 0.05}
            p.genre_weights = {k: v for k, v in p.genre_weights.items() if v > 0.05}
            p.bpm_bias = max(-30.0, min(30.0, p.bpm_bias))
            self.save()
            verdict = "liked" if liked else "noted"
            extra = f" — picked up: {', '.join(nudges)}" if nudges else ""
            return f"got it, {verdict}{extra}. taste updated."
        except Exception:  # noqa: BLE001
            _log.exception("record_feedback failed")
            return "feedback noted (taste save hiccup)."

    # ── suggesting ───────────────────────────────────────────────────

    def suggest_profile(self) -> Any | None:
        """Build a ReferenceProfile from taste.  None when there's no
        history worth building from."""
        try:
            from .producer import ReferenceProfile
        except Exception:  # noqa: BLE001
            return None
        try:
            p = self.profile
            liked = [t for t in p.tracks if t.liked]
            if not liked and p.production_count < 2 and not p.energy_weights:
                return None
            # tempo: median of liked, else of recent tempos
            bpms = [t.bpm for t in liked if t.bpm] or [
                (lo + hi) / 2 for lo, hi in p.preferred_tempos]
            if not bpms:
                return None
            bpms.sort()
            bpm = bpms[len(bpms) // 2] + p.bpm_bias
            bpm = max(70.0, min(180.0, bpm))
            # key: most common liked key
            key = "A"
            if liked:
                counts: dict[str, int] = {}
                for t in liked:
                    counts[t.key] = counts.get(t.key, 0) + 1
                key = max(counts, key=lambda k: counts[k])
            elif p.preferred_keys:
                key = p.preferred_keys[-1]
            # energy: top weighted words
            energy = [w for w, _ in sorted(
                p.energy_weights.items(), key=lambda kv: kv[1],
                reverse=True)[:5]]
            if not energy and liked:
                seen: dict[str, int] = {}
                for t in liked:
                    for w in t.energy_words:
                        seen[w] = seen.get(w, 0) + 1
                energy = [w for w, _ in sorted(
                    seen.items(), key=lambda kv: kv[1], reverse=True)[:5]]
            genre = ""
            if p.genre_weights:
                genre = max(p.genre_weights, key=lambda k: p.genre_weights[k])
            elif liked and liked[-1].genre:
                genre = liked[-1].genre
            return ReferenceProfile(
                title="from your taste", bpm=bpm, key=key, mode="minor",
                energy_words=tuple(energy) or ("driving",),
                genre=genre or "electronic",
                mood=", ".join(energy) or "driving",
            )
        except Exception:  # noqa: BLE001
            _log.exception("suggest_profile failed")
            return None


def load_taste(context: Any = None) -> TasteStore:
    return TasteStore(context)


def modify_profile(profile: Any, instruction: str) -> Any:
    """Iterative tweak: 'harder' / 'faster' / 'slower' / 'more melodic' …
    Returns a NEW profile; never mutates the input.  Never raises."""
    try:
        from .producer import ReferenceProfile
        low = (instruction or "").lower()
        bpm_delta = 0.0
        add: tuple[str, ...] = ()
        remove: tuple[str, ...] = ()
        matched = False
        for triggers, rule in _ITERATE:
            if any(t in low for t in triggers):
                bpm_delta += rule["bpm_delta"]
                add += rule["add_energy"]
                remove += rule["remove_energy"]
                matched = True
        if not matched:
            return profile
        energy = [w for w in (profile.energy_words or ())
                  if w.lower() not in {r.lower() for r in remove}]
        for w in add:
            if w not in {e.lower() for e in energy}:
                energy.append(w)
        return ReferenceProfile(
            title=profile.title, artist=profile.artist,
            bpm=max(70.0, min(180.0, profile.bpm + bpm_delta)),
            key=profile.key, mode=profile.mode,
            energy_words=tuple(energy) or ("driving",),
            genre=profile.genre, mood=", ".join(energy),
            ok=True,
        )
    except Exception:  # noqa: BLE001
        return profile


# ── natural-language detection ─────────────────────────────────────────

_PRODUCE_PATTERNS = (
    # strong triggers — explicit production verbs
    r"\b(make|cook|drop|produce|compose|write)\b.{0,20}\b(me\s+)?(something|a\s+(track|song|beat|banger|tune|jam)|some\s+(music|heat|fire))\b",
    r"\b(produce|compose)\b.{0,25}\b(track|song|beat|music)\b",
    # context triggers — vibe + occasion
    r"\bsomething\s+(dark|hard|chill|heavy|driving|melodic|hype|mellow|wild)\b",
    r"\bsomething\s+for\s+(the\s+)?(gym|workout|party|club|tonight|drive|run|morning|night)\b",
)

_MOOD_WORDS = {
    "good mood": ("uplifting", "euphoric"),
    "great mood": ("uplifting", "euphoric"),
    "happy": ("uplifting", "euphoric"),
    "hype": ("aggressive", "driving"),
    "hyped": ("aggressive", "driving"),
    "pumped": ("aggressive", "driving"),
    "sad": ("melancholic", "dark"),
    "down": ("melancholic", "dark"),
    "chill": ("chill", "dreamy"),
    "relaxed": ("chill", "dreamy"),
    "tired": ("chill", "dreamy"),
    "angry": ("aggressive", "hard"),
    "dark": ("dark", "menacing"),
    "tonight": ("dark", "driving"),
    "gym": ("aggressive", "driving"),
    "workout": ("aggressive", "driving"),
    "party": ("euphoric", "driving"),
    "club": ("driving", "heavy"),
    "drive": ("driving", "dark"),
    "morning": ("uplifting", "driving"),
    "focus": ("hypnotic", "chill"),
    "study": ("hypnotic", "chill"),
}


def detect_produce_intent(text: str) -> str | None:
    """NL → vibe tail for /produce, or None.  Never raises.

    Catches "make me something", "produce a track", "something dark for
    tonight", "make me something for the gym".  A bare mood statement
    ("I'm in a good mood") only fires when music/produce/make words are
    present — Devon shouldn't compose a track because you said you're
    happy.
    """
    try:
        low = (text or "").lower().strip()
        if not low or low.startswith("/"):
            return None
        if re.search(r"\bspotify\.com/track/", low):
            return text.strip()  # a pasted link is a produce request
        for pat in _PRODUCE_PATTERNS:
            if re.search(pat, low):
                return text.strip()
        # mood + music context
        has_music_ctx = bool(re.search(
            r"\b(music|song|track|beat|produce|make|something|sound)\b", low))
        for mood, _words in _MOOD_WORDS.items():
            if mood in low and has_music_ctx:
                return text.strip()
        return None
    except Exception:  # noqa: BLE001
        return None


def mood_hint(text: str) -> tuple[str, ...]:
    """Extract energy words from a mood/context sentence.  Never raises."""
    try:
        low = (text or "").lower()
        out: list[str] = []
        for mood, words in _MOOD_WORDS.items():
            if mood in low:
                for w in words:
                    if w not in out:
                        out.append(w)
        return tuple(out)
    except Exception:  # noqa: BLE001
        return ()


_FEEDBACK_POS = (
    r"\bi\s+(like|love)\s+(this|it|that)\b",
    r"\bthis\s+is\s+(fire|hard|sick|dope|amazing|great|perfect)\b",
    r"\b(love\s+it|fire\s*🔥*|hard\s*🔥*|goes\s+hard)\b",
)
_FEEDBACK_NEG = (
    r"\bnot\s+feeling\s+(this|it)\b",
    r"\btoo\s+(slow|fast|dark|happy|hard|soft|repetitive|chill|boring)\b",
    r"\b(nah|meh|skip)\b.{0,15}\b(track|song|beat|this)\b",
    r"\b(don't|dont)\s+like\s+(this|it|that)\b",
)


def detect_feedback(text: str) -> tuple[bool, str] | None:
    """NL feedback → (liked, notes), or None.  Never raises.

    Deliberately conservative: bare "nah" alone doesn't fire (too
    ambiguous in chat), but "nah, not feeling this track" does.
    """
    try:
        low = (text or "").lower().strip()
        if not low or low.startswith("/"):
            return None
        for pat in _FEEDBACK_POS:
            if re.search(pat, low):
                return (True, text.strip())
        for pat in _FEEDBACK_NEG:
            if re.search(pat, low):
                return (False, text.strip())
        return None
    except Exception:  # noqa: BLE001
        return None


def last_production_profile(store: TasteStore) -> Any | None:
    """Rebuild a ReferenceProfile from the last production record."""
    try:
        from .producer import ReferenceProfile
        lp = store.profile.last_production
        if not lp:
            return None
        return ReferenceProfile(
            title=str(lp.get("title", "")),
            bpm=float(lp.get("bpm", 128.0)),
            key=str(lp.get("key", "A")),
            mode=str(lp.get("mode", "minor")),
            energy_words=tuple(lp.get("energy_words", ())),
            genre=str(lp.get("genre", "")),
            mood=", ".join(lp.get("energy_words", ())),
            ok=True,
        )
    except Exception:  # noqa: BLE001
        return None
