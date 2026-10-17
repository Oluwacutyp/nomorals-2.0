"""Chat control for /impossible — same control the CLI uses, so CLI and chat stay in sync."""

from __future__ import annotations

from .core import ImpossibleMadePossible
from .first_principles import worked_example_car, worked_example_software
from .charter import charter as _charter
from .lattice import lattice as _lattice

_HELP = ("Usage: /impossible solve <problem> [--software] [--fetch-art] | status <id> | "
         "attempt <id> | dossier <id> | queue | charter | lattice | examples")


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
    return _HELP
