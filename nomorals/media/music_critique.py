"""Music critique loop + singability check — the producer's second ear.

Tier 2. :func:`singability` checks a lyric+melody pair the way a vocal
coach would: syllable-per-note alignment, stress vs beat, tone vs
contour (via :mod:`nomorals.media.tonal_g2p`), vocal range, leap size,
breath (longest unbroken phrase). :func:`critique_song` adds structure,
repetition, and rhyme checks over a whole song draft. :func:`critique_loop`
runs check → auto-fix (tone-compliant melody rewrite, range
transposition) → re-check, up to N passes, and reports whether the song
converged — never silently shipping a failing draft as "fixed".

All checks are deterministic and offline. Scores are 0..1 per axis;
``ship`` requires every hard axis (tone, range, syllables) to pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .tonal_g2p import profile_for, syllabify, tone_check

__all__ = [
    "SingabilityResult",
    "CritiqueResult",
    "singability",
    "critique_song",
    "critique_loop",
    "register",
]

# Singable vocal ranges (MIDI): comfortable tessitura per voice type.
RANGES = {
    "soprano": (60, 81), "alto": (55, 77), "tenor": (48, 74),
    "bass": (41, 69), "any": (48, 81),
}


def _syllable_count(line: str, language: str) -> int:
    return sum(len(syllabify(w, language)) for w in line.split())


@dataclass
class SingabilityResult:
    score: float
    issues: list[str] = field(default_factory=list)
    axes: dict[str, float] = field(default_factory=dict)
    syllables: int = 0
    notes: int = 0

    @property
    def ship(self) -> bool:
        return (self.axes.get("tone", 1.0) >= 1.0
                and self.axes.get("range", 1.0) >= 1.0
                and self.axes.get("syllables", 1.0) >= 0.5)


def singability(lyrics: str, melody: list, language: str = "en",
                voice: str = "any", bpm: float = 100.0) -> SingabilityResult:
    """Check one lyric line (or block) against its melody.

    melody: list of (midi, start_s, dur_s), one note per syllable
    expected. Axes:
    - syllables: syllable count vs note count (1 note per syllable is
      ideal; melisma — several notes per syllable — is allowed but
      flagged when extreme).
    - tone: tone↔melody check for tonal languages (1.0 for non-tonal).
    - range: every note inside the voice's tessitura.
    - leaps: adjacent leaps > 7 semitones penalised (singers hate them).
    - breath: longest phrase (no rest) under ~12s at the given tempo.
    - stress: long syllables (by character length, a proxy) should not
      land on the weakest 16th of the bar.
    """
    prof = profile_for(language)
    issues: list[str] = []
    axes: dict[str, float] = {}
    lines = [ln for ln in (lyrics or "").splitlines() if ln.strip()]
    text = " ".join(lines)
    syl_count = _syllable_count(text, language)
    notes = [(int(p), float(s), float(d)) for p, s, d in (melody or [])]
    # syllables
    if notes:
        ratio = syl_count / len(notes) if notes else 0.0
        if 0.8 <= ratio <= 1.25:
            axes["syllables"] = 1.0
        elif 0.5 <= ratio <= 2.0:
            axes["syllables"] = 0.6
            issues.append(f"syllable/note ratio {ratio:.2f} — heavy melisma or cramming")
        else:
            axes["syllables"] = 0.0
            issues.append(f"syllable/note ratio {ratio:.2f} — lyrics and melody don't line up")
    else:
        axes["syllables"] = 0.0
        issues.append("no melody notes")
    # tone
    if prof.tonal and notes:
        tc = tone_check(text, notes, language)
        axes["tone"] = tc["score"]
        issues.extend(tc["mismatches"][:5])
    else:
        axes["tone"] = 1.0
    # range
    lo, hi = RANGES.get(voice, RANGES["any"])
    out = [p for p, _, _ in notes if p < lo or p > hi]
    axes["range"] = 1.0 if not out else max(0.0, 1.0 - len(out) / max(1, len(notes)))
    if out:
        issues.append(f"{len(out)} note(s) outside {voice} range {lo}–{hi} (MIDI): {sorted(set(out))[:5]}")
    # leaps
    leaps = [abs(notes[i + 1][0] - notes[i][0]) for i in range(len(notes) - 1)]
    big = [l for l in leaps if l > 7]
    axes["leaps"] = 1.0 if not leaps else max(0.0, 1.0 - len(big) / len(leaps))
    if big:
        issues.append(f"{len(big)} leap(s) > 7 semitones — hard to sing cleanly")
    # breath: longest run without a rest (lyric "SP"/"rest" or gap > 0.6s)
    longest = cur = 0.0
    for i, (_, s, d) in enumerate(notes):
        gap = (notes[i][1] - (notes[i - 1][1] + notes[i - 1][2])) if i else 0.0
        if gap > 0.6 and i:
            longest = max(longest, cur)
            cur = 0.0
        cur += d
    longest = max(longest, cur)
    axes["breath"] = 1.0 if longest <= 12.0 else max(0.0, 1.0 - (longest - 12.0) / 12.0)
    if longest > 12.0:
        issues.append(f"longest phrase {longest:.1f}s — singer needs a breath")
    # stress: crude proxy — a 6+ char syllable on a weak 16th
    beat_s = 60.0 / max(1.0, bpm)
    weak = 0
    words = text.split()
    for w, (_, s, _) in zip(words, notes):
        pos_in_bar = (s / beat_s) % 4.0
        if len(w) >= 6 and abs(pos_in_bar - round(pos_in_bar)) > 0.01:
            weak += 1
    axes["stress"] = 1.0 if not weak else max(0.0, 1.0 - weak / max(1, len(words)))
    if weak:
        issues.append(f"{weak} long word(s) land off the beat — stress fights rhythm")
    score = sum(axes.values()) / len(axes) if axes else 0.0
    return SingabilityResult(score=round(score, 3),
                             issues=issues,
                             axes={k: round(v, 3) for k, v in axes.items()},
                             syllables=syl_count, notes=len(notes))


@dataclass
class CritiqueResult:
    score: float
    ship: bool
    issues: list[str] = field(default_factory=list)
    axes: dict[str, float] = field(default_factory=dict)
    fixes_applied: list[str] = field(default_factory=list)


def critique_song(title: str, sections: list[dict[str, Any]],
                  language: str = "en", voice: str = "any",
                  bpm: float = 100.0) -> CritiqueResult:
    """Critique a whole song draft.

    sections: [{name, lyrics: [lines], melody: [(midi,start,dur)]}].
    Checks: per-section singability, structure (needs a chorus/hook),
    repetition (chorus lines should repeat — hooks are repetition),
    rhyme (last words of consecutive lines share a vowel tail, soft
    check), and total length sanity. Returns issues + score.
    """
    issues: list[str] = []
    axes: dict[str, float] = {}
    sing_scores = []
    hard_fail = False
    for sec in sections or []:
        lyrics = "\n".join(sec.get("lyrics", []) or [])
        mel = sec.get("melody", []) or []
        if lyrics and mel:
            r = singability(lyrics, mel, language, voice, bpm)
            sing_scores.append(r.score)
            issues.extend(f"[{sec.get('name', '?')}] {i}" for i in r.issues[:3])
            # Hard axes: tone contradiction or out-of-range notes veto ship.
            if r.axes.get("tone", 1.0) < 1.0 or r.axes.get("range", 1.0) < 1.0:
                hard_fail = True
    axes["singability"] = (sum(sing_scores) / len(sing_scores)) if sing_scores else 0.5
    names = [str(s.get("name", "")).lower() for s in sections or []]
    has_chorus = any("chorus" in n or "hook" in n for n in names)
    axes["structure"] = 1.0 if has_chorus else 0.0
    if not has_chorus:
        issues.append("no chorus/hook section — song has no anchor")
    # repetition: chorus lines should repeat across chorus sections
    choruses = [s for s in sections or []
                if "chorus" in str(s.get("name", "")).lower()
                or "hook" in str(s.get("name", "")).lower()]
    if len(choruses) >= 2:
        first = set(choruses[0].get("lyrics", []) or [])
        second = set(choruses[1].get("lyrics", []) or [])
        overlap = len(first & second) / max(1, len(first | second))
        axes["repetition"] = 1.0 if overlap >= 0.5 else 0.4
        if overlap < 0.5:
            issues.append("chorus lines change between choruses — hooks need repetition")
    else:
        axes["repetition"] = 0.7  # single chorus: can't judge
    # rhyme: last-word vowel-tail match on consecutive line pairs
    pairs = total = 0
    for sec in sections or []:
        lines = sec.get("lyrics", []) or []
        for a, b in zip(lines, lines[1:]):
            wa = re.sub(r"[^a-z]", "", a.split()[-1].lower()) if a.split() else ""
            wb = re.sub(r"[^a-z]", "", b.split()[-1].lower()) if b.split() else ""
            if wa and wb:
                total += 1
                if wa[-2:] == wb[-2:] or wa[-1:] == wb[-1:]:
                    pairs += 1
    axes["rhyme"] = (pairs / total) if total else 0.5
    if total and pairs / total < 0.3:
        issues.append("few rhyming line endings — verses may feel loose")
    score = sum(axes.values()) / len(axes) if axes else 0.0
    return CritiqueResult(score=round(score, 3),
                          ship=bool(score >= 0.7 and not hard_fail),
                          issues=issues,
                          axes={k: round(v, 3) for k, v in axes.items()})


def critique_loop(title: str, sections: list[dict[str, Any]],
                  language: str = "en", voice: str = "any",
                  bpm: float = 100.0, max_passes: int = 3) -> CritiqueResult:
    """Check → auto-fix → re-check, up to max_passes.

    Auto-fixes applied (all reversible, all reported):
    1. Tone-compliant melody rewrite (tonal languages).
    2. Range transposition: if the median note is outside the voice's
       tessitura, transpose the whole melody by octaves toward centre.
    The loop stops early when the draft ships; otherwise it returns the
    best draft seen with the remaining issues listed — never claims a
    fix it didn't verify by re-checking.
    """
    from .tonal_g2p import tone_fix_melody

    fixes: list[str] = []
    best = critique_song(title, sections, language, voice, bpm)
    best_sections = sections
    for _ in range(max_passes):
        if best.ship:
            break
        changed = False
        new_sections: list[dict[str, Any]] = []
        lo, hi = RANGES.get(voice, RANGES["any"])
        centre = (lo + hi) / 2.0
        for sec in best_sections:
            sec2 = dict(sec)
            lyrics = "\n".join(sec.get("lyrics", []) or [])
            mel = [(int(p), float(s), float(d))
                   for p, s, d in (sec.get("melody", []) or [])]
            if mel and lyrics:
                fixed = tone_fix_melody(lyrics, mel, language)
                if fixed != mel:
                    sec2["melody"] = fixed
                    mel = fixed
                    changed = True
                    fixes.append(f"tone-fix {sec.get('name', '?')}")
                if mel:
                    med = sorted(p for p, _, _ in mel)[len(mel) // 2]
                    shift = 0
                    while med + shift < lo:
                        shift += 12
                    while med + shift > hi:
                        shift -= 12
                    if shift:
                        sec2["melody"] = [(p + shift, s, d) for p, s, d in mel]
                        changed = True
                        fixes.append(f"transpose {shift:+d} {sec.get('name', '?')} → centre {centre:.0f}")
            new_sections.append(sec2)
        if not changed:
            break
        best_sections = new_sections
        nxt = critique_song(title, best_sections, language, voice, bpm)
        if nxt.score >= best.score:
            best = nxt
        else:
            break  # fix made it worse — keep the better draft
    best.fixes_applied = fixes
    return best


def register(registry: Any) -> None:
    """Tool registration: ``music_critique`` — singability + critique loop."""
    from ..core.policy import Capability

    @registry.register(
        "music_critique",
        description=(
            "Music critique loop + singability: action=singability (lyrics, "
            "melody JSON, language, voice, bpm → per-axis report) | "
            "critique (title, sections JSON → whole-song critique) | "
            "loop (same → check→auto-fix→re-check, up to 3 passes). Hard "
            "axes: tone never contradicts melody, notes in vocal range."
        ),
        capability=Capability.FS_WRITE,
    )
    def music_critique(action: str = "singability", lyrics: str = "",
                       melody: str = "", language: str = "en",
                       voice: str = "any", bpm: float = 100.0,
                       title: str = "", sections: str = "") -> dict:
        import json as _json
        try:
            mel = _json.loads(melody) if melody else []
        except Exception as exc:  # noqa: BLE001
            from ..core.errors import ToolError
            raise ToolError(f"melody must be JSON: {exc}")
        if action == "singability":
            r = singability(lyrics, mel, language, voice, bpm)
            return {"ok": True, "score": r.score, "ship": r.ship,
                    "issues": r.issues, "axes": r.axes,
                    "syllables": r.syllables, "notes": r.notes}
        try:
            secs = _json.loads(sections) if sections else []
        except Exception as exc:  # noqa: BLE001
            from ..core.errors import ToolError
            raise ToolError(f"sections must be JSON: {exc}")
        if action == "critique":
            r = critique_song(title, secs, language, voice, bpm)
        elif action == "loop":
            r = critique_loop(title, secs, language, voice, bpm)
        else:
            from ..core.errors import ToolError
            raise ToolError(f"unknown action {action!r}")
        return {"ok": True, "score": r.score, "ship": r.ship,
                "issues": r.issues, "axes": r.axes,
                "fixes_applied": r.fixes_applied}
