"""Character presentation: god-tier output, not functional strings.

Characters are personalities — how they LOOK on screen matters as much
as how they think. This module renders character cards, cast
announcements, styled scene transcripts, relationship webs, and voice
reports. Two themes: "rich" (box-drawing, emoji, full texture) and
"minimal" (clean, quiet).

Nothing here changes state. Pure presentation over the real classes.
"""
from __future__ import annotations

from typing import Any

from .character import Character

__all__ = ["render_card", "render_cast", "render_scene", "render_web",
           "render_arc", "THEMES"]

THEMES = ("rich", "minimal")


def _mood_emoji(char: Character) -> str:
    v = char.mood.get("valence", 0.0)
    a = char.mood.get("arousal", 0.3)
    if v > 0.4 and a > 0.6:
        return "🔥"
    if v > 0.4:
        return "😊"
    if v < -0.4:
        return "🌧️"
    if a > 0.7:
        return "⚡"
    return "🌤️"


def _trait_bar(v: float, width: int = 10) -> str:
    filled = int(round(max(0.0, min(1.0, v)) * width))
    return "█" * filled + "░" * (width - filled)


def render_card(char: Character, theme: str = "rich") -> str:
    """A character card — the whole person at a glance."""
    if theme == "minimal":
        lines = [f"{char.name} — {char.stage_of_life or 'no stage set'}"]
        if char.persona:
            top = sorted(char.persona.items(), key=lambda kv: -kv[1])[:4]
            lines.append("traits: " + ", ".join(f"{k} {v:.1f}" for k, v in top))
        if char.core_motive:
            lines.append(f"wants: {char.core_motive}")
        if char.roles:
            lines.append("roles: " + ", ".join(char.roles))
        return "\n".join(lines)

    w = 46
    top_edge = "╭" + "─" * w + "╮"
    bot_edge = "╰" + "─" * w + "╯"

    def row(text: str) -> str:
        t = text[:w]
        return "│" + t + " " * (w - len(t)) + "│"

    lines = [top_edge]
    lines.append(row(f"{_mood_emoji(char)}  {char.name}"))
    if char.stage_of_life:
        lines.append(row(f"   {char.stage_of_life}"))
    lines.append(row("─" * w))
    if char.core_motive:
        lines.append(row(f'❝ {char.core_motive[:w - 4]}'))
    # top traits with bars
    for trait, v in sorted(char.persona.items(), key=lambda kv: -kv[1])[:5]:
        lines.append(row(f"  {trait:<14} {_trait_bar(v)} {v:.1f}"))
    # OCEAN strip
    ocean = getattr(char, "ocean", None) or {}
    if ocean:
        obits = " ".join(
            f"{t[0].upper()}{float(ocean.get(t, 0.5)):.0%}" for t in
            ("openness", "conscientiousness", "extraversion",
             "agreeableness", "neuroticism"))
        lines.append(row(f"  OCEAN  {obits}"))
    if char.roles:
        lines.append(row("  🎭 " + ", ".join(char.roles[:4])))
    if char.skills:
        sk = sorted(char.skills.items(), key=lambda kv: -kv[1])[:3]
        lines.append(row("  ⭐ " + ", ".join(f"{k} {v:.1f}" for k, v in sk)))
    # live state
    agenda = ""
    try:
        agenda = char.goal_agenda()
    except Exception:
        pass
    if agenda:
        lines.append(row("─" * w))
        lines.append(row(f"  🎯 {agenda[:w - 5]}"))
    v, a = char.mood.get("valence", 0.0), char.mood.get("arousal", 0.3)
    mood_word = ("upbeat" if v > 0.3 else "down" if v < -0.3 else "neutral")
    lines.append(row(f"  mood: {mood_word}, "
                     f"{'energetic' if a > 0.6 else 'calm'}  "
                     f"│ {len(char.memory)} memories"))
    lines.append(bot_edge)
    return "\n".join(lines)


def render_cast(cast: list[tuple[Character, float, str]],
                title: str = "Cast",
                theme: str = "rich") -> str:
    """A casting announcement — who got picked and why."""
    if theme == "minimal":
        return "\n".join(f"{c.name} ({s:.0%}): {r}" for c, s, r in cast)
    lines = [f"🎬 {title}"]
    medals = ["🥇", "🥈", "🥉"]
    for i, (c, s, r) in enumerate(cast):
        medal = medals[i] if i < 3 else "  "
        lines.append(f"{medal} {c.name} — {s:.0%} — {r}")
    return "\n".join(lines)


def render_scene(scene: Any, theme: str = "rich") -> str:
    """A styled scene transcript — reads like a play, not a log."""
    title = getattr(scene, "title", "Scene")
    topic = getattr(scene, "topic", "")
    turns = getattr(scene, "turns", [])
    if theme == "minimal":
        head = f"# {title}" + (f" — {topic}" if topic else "")
        body = "\n".join(f"{t.speaker}: {t.text}" for t in turns)
        return head + "\n" + body
    lines = ["═" * 52, f"  🎭 {title}"]
    if topic:
        lines.append(f"  ❝ {topic}")
    lines.append("═" * 52)
    last_speaker = ""
    for t in turns:
        sp = str(getattr(t, "speaker", "?"))
        tx = str(getattr(t, "text", ""))
        if sp == "Devon (director)":
            lines.append(f"  ── {tx} ──")
            continue
        if sp != last_speaker:
            lines.append("")
            lines.append(f"  ┌─ {sp}")
            last_speaker = sp
        # wrap long lines gently
        while len(tx) > 60:
            cut = tx[:60].rsplit(" ", 1)[0]
            lines.append(f"  │ {cut}")
            tx = tx[len(cut):].lstrip()
        lines.append(f"  │ {tx}")
    lines.append("")
    lines.append("═" * 52)
    lines.append(f"  🎬 end scene — {len(turns)} turns")
    return "\n".join(lines)


def render_web(graph: Any, names: dict[str, str] | None = None,
               who: str | None = None, theme: str = "rich") -> str:
    """The relationship web as a diagram, not a dict dump."""
    names = names or {}
    edges = getattr(graph, "edges", {}) or {}
    items = []
    for (a, b), e in edges.items():
        if who and a != who and b != who:
            continue
        na, nb = names.get(a, a), names.get(b, b)
        kind = getattr(e, "kind", "?")
        d = getattr(e, "dims", None) or {}
        items.append((na, nb, kind, float(d.get("trust", 0.5)),
                      float(d.get("friction", 0.0))))
    if not items:
        return "🕸️ no relationships on record yet."
    if theme == "minimal":
        return "\n".join(f"{a} → {b}: {k}" for a, b, k, _, _ in items)
    kind_glyph = {"close": "💞", "friend": "🤝", "rival": "⚔️",
                  "familiar": "👋", "acquaintance": "👤", "mentor": "🧙"}
    lines = ["🕸️ relationship web"]
    for a, b, k, trust, fric in sorted(items):
        g = kind_glyph.get(k, "•")
        extra = ""
        if fric > 0.6:
            extra = "  🔥 tense"
        elif trust > 0.8:
            extra = "  💎 deep trust"
        lines.append(f"  {g} {a} → {b}: {k}{extra}")
    return "\n".join(lines)


def render_arc(char: Character, theme: str = "rich") -> str:
    """The character's arc, presentation-wrapped."""
    try:
        from .arcs import arc_story
        story = arc_story(char)
    except Exception:
        story = f"{char.name}: no arc yet."
    if theme == "minimal":
        return story.replace("📖 ", "").replace("💭 ", "- ") \
            .replace("🩹 ", "- ").replace("🏛️ ", "- ")
    return story
