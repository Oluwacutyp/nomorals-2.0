"""translate — the translation organ as an agent tool (doc 19 §4).

Agents get the TICKET, never a bare string: every call runs a
TranslationTicket through the organ and returns its full result —
the translation, the rung that produced it, the quality gate's
verdict, whether the result may be labelled exact, the durable
receipt reference(s), and, for document-shaped input, the section
map (per-section rung/gate/receipt, structure preserved: code
fences byte-identical, tables cell by cell, link destinations
untouched).

    translate(text="Bonjour le monde", target="en")
    translate(text="# Titre\n\nVoir [le guide](https://x.y/g).",
              target="en", exactness="fidelity", glossary_id="legal")

``glossary_id`` names a stored glossary whose terms are enforced
for the pair (protected before the rung, restored verified after —
see ``llm.glossary``); manage the terms themselves through the
glossary store.  ``local_only=True`` keeps the text on this machine:
the local rung alone answers, or the ticket honestly returns the
original marked untranslated.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import ToolError
from ..core.policy import Capability


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "translate",
        description=(
            "Translate text through Devon's translation organ and "
            "return the full ticket: translation + rung provenance "
            "+ quality-gate verdict + exactness label + durable "
            "receipt id(s). exactness: gist (default) | fidelity | "
            "legal | code — fidelity classes run the bidirectional "
            "quality gate and are labelled exact only when it "
            "passes. Document-shaped input (markdown/HTML) is "
            "translated section-wise with its structure preserved "
            "and a per-section map in the result. glossary_id "
            "enforces a stored glossary's terms for the pair. "
            "local_only=True never lets the text leave this "
            "machine."
        ),
        capability=Capability.MODEL_CALL,
        annotations={"destructive": False, "idempotent": True,
                     "read_only": True, "open_world": True},
        examples=[
            "translate(text=\"Bonjour le monde\", target=\"en\")",
            "translate(text=\"# Titre\\n\\nCorps.\", target=\"en\", "
            "exactness=\"fidelity\", glossary_id=\"legal\")",
        ],
    )
    def translate(text: str, target: str = "", source: str = "",
                  exactness: str = "gist", glossary_id: str = "",
                  preserve: str = "", local_only: bool = False,
                  requester: str = "agent") -> dict[str, Any]:
        if not (text or "").strip():
            raise ToolError("translate needs text=")
        from ..llm.ticket import TranslationTicket, run_ticket
        from ..llm.translate import translator_for

        rules = tuple(rule.strip() for rule in (preserve or "")
                      .split(",") if rule.strip())
        ticket = TranslationTicket(
            text=text, target_lang=target, source_lang=source,
            exactness=exactness, glossary_id=glossary_id,
            preserve=rules, local_only=bool(local_only),
            requester=requester or "agent")
        try:
            outcome = run_ticket(translator_for(context), ticket)
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        return outcome.to_dict()
