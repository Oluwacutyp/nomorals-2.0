"""The ``characters`` tool: agent-callable character casting and direction.

This is how the brain reaches the character bank through the tool loop —
she casts podcast hosts, room guests, game players, and conversation
partners herself, in plain language, not via commands.

Actions: list | cast | talk | direct | scene | podcast | arc | create.
"""
from __future__ import annotations

from typing import Any


def _store() -> Any:
    from ..characters.store import CharacterStore
    return CharacterStore()


def _graph(store: Any) -> Any:
    from ..characters.relationships import RelationshipGraph
    import json
    g = RelationshipGraph()
    p = store.dir / "_relationships.json"
    if p.exists():
        try:
            g = RelationshipGraph.from_dict(json.loads(p.read_text("utf-8")))
        except Exception:
            pass
    return g


def _save_graph(store: Any, graph: Any) -> None:
    import json
    try:
        (store.dir / "_relationships.json").write_text(
            json.dumps(graph.to_dict(), indent=1), encoding="utf-8")
    except Exception:
        pass


def _suggest(registry: Any) -> Any:
    context = getattr(registry, "context", None)
    return getattr(context, "suggest", None) if context else None


def register(registry: Any) -> None:
    @registry.register(
        "characters",
        description=(
            "Character bank: persistent agent personas Devon can cast, talk "
            "to, and direct. Actions: list (who's in the bank), cast "
            "<purpose> (pick the right character(s) for a moment — podcast "
            "host, room guest, game player, deep talk — with reasons), talk "
            "<name> <message> (converse with a character), direct <name> "
            "<role> <brief> (give a character a role for a scene), scene "
            "<title> <topic> <names...> (run a multi-character scene), "
            "podcast <topic> [guest names...] (cast + run a podcast episode), "
            "arc <name> (character's growth story), create <name> "
            "<description> (bring a new character to life from a plain-"
            "words description)."
        ),
        capability="characters.manage",
        parameters={
            "action": "str — list|cast|talk|direct|scene|podcast|arc|create",
            "target": "str — character name(s), purpose, or topic",
            "text": "str — message, brief, or description",
            "n": "int — how many to cast (default 1)",
        },
    )
    def characters(action: str = "list", target: str = "",
                   text: str = "", n: int = 1) -> str:
        from ..characters import Character
        from ..characters.casting import cast_for, cast_preset, ROLE_PRESETS
        from ..characters.dialogue import converse
        from ..characters.ensemble import run_scene, podcast_episode
        from ..characters.arcs import arc_summary

        store = _store()
        graph = _graph(store)
        suggest = _suggest(registry)
        action = (action or "list").lower().strip()

        if action == "list":
            chars = store.all()
            if not chars:
                return "the character bank is empty."
            lines = []
            for c in chars:
                roles = f" [{', '.join(c.roles[:3])}]" if c.roles else ""
                lines.append(f"🎭 {c.name}{roles} — "
                             f"{', '.join(sorted(c.persona)[:3])}")
            return "\n".join(lines)

        if action == "cast":
            chars = store.all()
            if not chars:
                return "the character bank is empty — create someone first."
            purpose = (target or "conversation").lower()
            preset = ROLE_PRESETS.get(purpose, {})
            picks = cast_for(
                chars, role=preset.get("role", ""),
                traits=preset.get("traits"), skills=preset.get("skills"),
                topic=text, n=max(1, min(4, n)), graph=graph)
            if not picks:
                return "no one fits — the bank is empty."
            out = [f"🎬 casting for '{purpose}':"]
            for c, score, reason in picks:
                out.append(f"  • {c.name} ({score:.2f}) — {reason}")
            return "\n".join(out)

        if action == "talk":
            c = store.get_by_name(target)
            if c is None:
                return f"no character named '{target}'."
            dlg = converse("You are Devon, a warm AI companion.",
                           c, text or "hey", suggest, rounds=2)
            store.save(c)
            _save_graph(store, graph)
            last = dlg.turns[-1] if dlg.turns else None
            return f"🎭 {c.name}: {last.text}" if last else "…"

        if action == "direct":
            c = store.get_by_name(target)
            if c is None:
                return f"no character named '{target}'."
            # text = role + brief, e.g. "podcast_host interview about Afrobeats"
            bits = (text or "").split(None, 1)
            role = bits[0] if bits else "guest"
            brief = bits[1] if len(bits) > 1 else ""
            if role not in c.roles:
                c.roles.append(role)
            c.remember(f"Devon directed me as {role}: {brief[:200]}", 0.6)
            store.save(c)
            line = c.speak(
                f"Devon just cast you as {role} for: {brief}. "
                f"React in character — accept, riff on it, make it yours.",
                suggest)
            return f"🎬 {c.name} ({role}): {line}"

        if action == "scene":
            names = [w for w in target.split(",") if w.strip()]
            chars = [store.get_by_name(nm) for nm in names]
            chars = [c for c in chars if c is not None]
            if len(chars) < 2:
                return "a scene needs at least 2 characters from the bank."
            sc = run_scene(chars, title=text or "Scene",
                           topic=text or "anything", suggest=suggest,
                           rounds=4, graph=graph)
            for c in chars:
                store.save(c)
            _save_graph(store, graph)
            return (f"🎬 {sc.title} — "
                    + ", ".join(sc.participants) + "\n"
                    + sc.transcript(12))

        if action == "podcast":
            chars = store.all()
            if not chars:
                return "the character bank is empty."
            topic = target or text or "anything"
            # cast host + guests
            host_pick = cast_preset(chars, "podcast_host", n=1,
                                    topic=topic, graph=graph)
            if not host_pick:
                return "no one can host."
            host = host_pick[0][0]
            guest_names = [w.strip() for w in text.split(",") if w.strip()]
            guests: list[Any] = []
            for gn in guest_names:
                g = store.get_by_name(gn)
                if g is not None and g.id != host.id:
                    guests.append(g)
            if not guests:
                gpicks = cast_preset(
                    chars, "podcast_guest", n=2, topic=topic, graph=graph,
                    exclude={host.id})
                guests = [c for c, _, _ in gpicks if c.id != host.id][:2]
            if not guests:
                return f"{host.name} is ready to host, but there's no one to interview."
            ep = podcast_episode(host, guests, topic, suggest,
                                 graph=graph, rounds=6)
            for c in [host] + guests:
                store.save(c)
            _save_graph(store, graph)
            return (f"🎙️ {ep.title}\nHost: {host.name} · Guests: "
                    + ", ".join(g.name for g in guests) + "\n\n"
                    + ep.transcript(16))

        if action == "arc":
            c = store.get_by_name(target)
            if c is None:
                return f"no character named '{target}'."
            return arc_summary(c)

        if action == "create":
            from ..characters import Character as _C
            name = (target or "").strip()
            if not name:
                return "give the character a name."
            if store.get_by_name(name):
                return f"🎭 {name} already exists."
            # Brain-grade creation: the model fleshes out the description
            desc = text or "an interesting person"
            c = _C(name=name)
            if suggest is not None:
                try:
                    prompt = (
                        f"Create a deep, original character named {name}. "
                        f"Description: {desc}.\n"
                        f"Reply as JSON with keys: persona (object of "
                        f"trait: 0.0-1.0, 4-7 traits), backstory (2-3 "
                        f"sentences), core_motive (one line), knowledge "
                        f"(list of 4-6 things they know), goals (list of "
                        f"2-3), skills (object of skill: 0.0-1.0, 3-5), "
                        f"roles (list of 2-4 from: podcast_host, dj, gamer, "
                        f"sage, hype, interviewer, storyteller), beliefs "
                        f"(list of 2-3 strong opinions), secrets (list of "
                        f"1-2 hidden things), spine (0.0-1.0), expression "
                        f"(object with speech_patterns list and catchphrases "
                        f"list). JSON only.")
                    import json as _json
                    raw = (suggest(prompt) or "").strip()
                    # tolerate fences
                    if "```" in raw:
                        raw = raw.split("```")[1]
                        if raw.startswith("json"):
                            raw = raw[4:]
                    data = _json.loads(raw)
                    c.persona = {k: float(v) for k, v in
                                 data.get("persona", {}).items()}
                    c.backstory = str(data.get("backstory", ""))
                    c.core_motive = str(data.get("core_motive", ""))
                    c.knowledge = [str(x) for x in data.get("knowledge", [])]
                    c.goals = [str(x) for x in data.get("goals", [])]
                    c.skills = {k: float(v) for k, v in
                                data.get("skills", {}).items()}
                    c.roles = [str(x) for x in data.get("roles", [])]
                    for btxt in data.get("beliefs", []):
                        from ..characters.arcs import add_belief
                        add_belief(c, str(btxt), 0.7)
                    c.secrets = [str(x) for x in data.get("secrets", [])]
                    c.spine = float(data.get("spine", 0.6))
                    c.expression = dict(data.get("expression", {}))
                except Exception:
                    pass  # keep the shell; owner shapes them later
            if not c.persona:
                c.persona = {"curious": 0.7, "warm": 0.6}
            store.save(c)
            return (f"🎭 {name} is alive.\n"
                    f"{c.backstory[:200]}\n"
                    f"Motive: {c.core_motive or '—'}\n"
                    f"Roles: {', '.join(c.roles) or '—'}")

        return (f"unknown action '{action}'. "
                "list | cast | talk | direct | scene | podcast | arc | create")
    return characters
