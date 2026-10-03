"""Repo orientation for LLM-driven agents.

Every agent that reasons *about Devon's own codebase* (the Devon agent,
the orchestrator, evolution, …) gets this block in its prompt so it
never guesses paths like ``src/integrations/spotify.py`` and never
claims a connector "doesn't exist" when it lives in
``nomorals/connectors/``.

The connector catalog is built live from the connector registry, so it
stays current as connectors are added.  The layout map is static —
update it when top-level directories move.
"""

from __future__ import annotations

__all__ = [
    "REPO_LAYOUT",
    "connector_catalog_block",
    "repo_orientation_block",
]

#: Where things live.  Agents must prefer these real paths over guesses.
REPO_LAYOUT = """\
NoMorals repo layout (repo-relative, always):
  nomorals/agents/        the organs: coremind.py (intent router), devon.py
                          (this agent), coding.py (builder), orchestrator.py,
                          research.py, directives.py (missions), partner/
                          (companion brain + chat runtime)
  nomorals/connectors/    ONE file per service integration, e.g.
                          spotify.py, github.py, gmail.py, discord.py,
                          binance.py, mono.py, flutterwave.py, proxypool.py.
                          This is the ONLY place service integrations live —
                          there is no src/integrations/ directory.
  nomorals/social/chat/   chat gateway + control.py (all /commands)
  nomorals/media/         music.py (composer), playback.py, library.py,
                          image/video editing + generation
  nomorals/books/         BookForge — writes real books, builds PDFs
  nomorals/games/         game engine (40 games)
  nomorals/partner/       companion persona, mood, relationship, responder
  nomorals/tools/         shared tool registry (tools.call names live here)
  nomorals/core/          config, database, http client, logging
  nomorals/voice/         TTS / voice cloning
  nomorals/wisdom/        WisdomKeeper knowledge base
  nomorals/media_edit/     image/video editing organs: cv_ops.py (35 ops),
                          layers.py (LayerStack), images.py, videos.py,
                          generate.py. NEVER write raw PIL/OpenCV scripts —
                          use the media_edit tools.
  tests/                  test suite (tests/test_*.py)
  docs/                   DEPENDENCIES.md, TERMUX_ENV_TEMPLATE.txt, …
  scripts/termux/         Termux one-shot installer
"""


def connector_catalog_block() -> str:
    """One line per registered connector: id, name, auth, what it does."""
    try:
        from ..connectors import list_connectors
    except Exception:  # noqa: BLE001 - orientation is best-effort
        return "connector catalog unavailable."
    try:
        infos = list_connectors()
    except Exception:  # noqa: BLE001
        return "connector catalog unavailable."
    if not infos:
        return "no connectors registered."
    lines = ["Connectors (nomorals/connectors/<id>.py — all real, all wired):"]
    for info in infos:
        auth = ",".join(info.get("auth_methods") or ["?"])
        desc = (info.get("description") or "").split(".")[0].strip()
        lines.append(f"  - {info['id']}: {desc} [auth: {auth}]")
    lines.append(
        "To link one: say which service — Devon drives the real OAuth/API "
        "flow via its connector (never ask for raw tokens in chat)."
    )
    return "\n".join(lines)


def repo_orientation_block() -> str:
    """The full orientation block for an agent system prompt."""
    return (
        "REPO ORIENTATION — you run INSIDE the NoMorals repo. "
        "Answer questions about Devon's own capabilities from this, "
        "not from guesses:\n"
        + REPO_LAYOUT
        + "\n"
        + connector_catalog_block()
    )
