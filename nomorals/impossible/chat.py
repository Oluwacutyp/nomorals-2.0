"""Chat control for /impossible — same control the CLI uses, so CLI and chat stay in sync."""

from __future__ import annotations

from .core import ImpossibleMadePossible
from .first_principles import worked_example_car, worked_example_software
from .charter import charter as _charter
from .lattice import lattice as _lattice

_HELP = ("Usage: /impossible solve <problem> [--software] [--fetch-art] | status <id> | "
         "attempt <id> | dossier <id> | queue | charter | lattice | examples | sources [domain] | cross <problem>")


def control_impossible(text: str, context=None, chat_key: str = "chat", **_kwargs) -> str:
    words = (text or "").strip()
    if words.startswith("/impossible"):
        words = words[len("/impossible"):].strip()
    if not words or words in ("help", "--help"):
        return _HELP
    parts = words.split(None, 1)
    cmd, rest = parts[0].lower(), (parts[1] if len(parts) > 1 else "")
    imp = ImpossibleMadePossible()
    if cmd == "solve" and rest:
        domain = "software" if "--software" in rest else "hardware"
        fetch = "--fetch-art" in rest
        problem = rest.replace("--software", "").replace("--fetch-art", "").strip()
        dossier = imp.submit(problem, domain=domain, fetch_prior_art=fetch)
        if dossier.get("status") == "needs_clarification":
            return ("Underspecified — not guessing. Answer these and resubmit:\n"
                    + "\n".join(f"- {c}" for c in dossier["clarifications"]))
        return dossier["report"]
    if cmd == "status" and rest:
        return str(imp.status(rest.strip()))
    if cmd == "attempt" and rest:
        return str(imp.attempt(rest.strip()))
    if cmd == "dossier" and rest:
        d = imp.dossier(rest.strip())
        return str(d) if d else f"No claim with id '{rest.strip()}'."
    if cmd == "queue":
        items = imp.queue_list()
        if not items:
            return "Queue is empty."
        return "\n".join(f"- {i['claim_id']} [{i['status']}] {i['problem']}" for i in items)
    if cmd == "charter":
        return str(_charter())
    if cmd == "lattice":
        lat = _lattice()
        return "Zones: " + ", ".join(lat["zones"].keys()) + ". Layers: " + ", ".join(lat["layers"])
    if cmd == "examples":
        car = worked_example_car()
        sw = worked_example_software()
        return ("CAR:\n" + car["first_principles_answer"] + "\n\nSOFTWARE:\n" + sw["first_principles_answer"])
    if cmd == "cross" and rest:
        from .bridges import cross_query as _cross
        out = _cross(rest.strip())
        lines = [f"Cross-knowledge for: {rest.strip()}",
                 f"Impossible: {out['impossible'].get('status')} "
                 f"{out['impossible'].get('gate_verdict', '')}"]
        if "wisdomkeeper" in out:
            w = out["wisdomkeeper"]
            lines.append(f"WisdomKeeper: {w.get('status')} — {w.get('reason', w.get('synthesis', ''))}")
        if "bookforge" in out:
            b = out["bookforge"]
            lines.append(f"BookForge: {b.get('status')} — {b.get('reason', '')}")
        return "\n".join(lines)
    if cmd == "sources":
        from .sources import sources as _sources, sources_by_domain as _by_domain, domains as _domains
        if rest:
            rows = _by_domain(rest.strip())
            header = f"Digestion sources for domain '{rest.strip()}':"
        else:
            rows = _sources()
            header = f"Digestion sources ({len(rows)} total, domains: {', '.join(_domains())}):"
        if not rows:
            return header + "\n(none — unknown domain)"
        lines = [header]
        for s in rows:
            lines.append(f"- {s['name']} [{s['access']}] domains={','.join(s['domains'])} "
                         f"yields={s['yields']} url={s['url']}")
        return "\n".join(lines)
    return _HELP
