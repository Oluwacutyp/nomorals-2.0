"""decoder_agent — one tool over the wave-71 decoder stack.

Thin dispatch on ``action``:

    (default)   full chain analysis via :mod:`nomorals.agents.decoder`
    "hash"       known-plaintext lookup for a digest
    "decoders"   the decoder registry (names, count)
"""

from __future__ import annotations

from typing import Any


def decoder_agent(data: str = "", action: str = "", **kwargs: Any) -> dict[str, Any]:
    """Decode data using the layered decoder (magic, chain, hash table)."""
    from ..core import decoder as D

    action = (action or "").strip().lower()
    if action == "decoders":
        names = [d.name for d in D.DECODERS]
        return {"count": len(names), "names": names}
    if action == "hash":
        candidate = (data or "").strip()
        hit = D.known_hash_lookup(candidate)
        out: dict[str, Any] = {
            "input": candidate,
            "known": hit,
            "algorithms": D.identify_hash(candidate) if hit is None else [hit.get("algorithm", "")],
        }
        if hit is None:
            learned = getattr(D, "learned_hash_lookup", None)
            if learned is not None:
                try:
                    out["known"] = learned(None, candidate)
                except Exception:  # noqa: BLE001 — learned store needs a db
                    pass
        return out
    # default: the full agent chain (explanation, KG curation, saves)
    from ..agents.decoder import DecoderAgent

    class _NoCtx:  # context is injected by the registry when available
        pass

    context = kwargs.pop("context", None) or kwargs.pop("_context", None)
    if context is None:
        from ..agents.context import build_context
        from ..core.config import load_settings

        with build_context(load_settings(), with_router=False,
                           with_memory=False, with_tools=False) as ctx:
            result = DecoderAgent(context=ctx, name="tool").run(
                {"data": data, **kwargs})
    else:
        result = DecoderAgent(context=context, name="tool").run(
            {"data": data, **kwargs})
    if not getattr(result, "ok", False):
        raise RuntimeError(getattr(result, "error", "decoder agent failed"))
    return dict(result.output)


def register(registry: Any) -> None:
    """Register the decoder-agent tool."""
    registry.register(
        "decoder_agent",
        decoder_agent,
        description="Decode/identify encodings, chains, and known hashes",
        capability="decoder",
        parameters={
            "data": {"type": "string", "description": "Data (or file:path) to decode"},
            "action": {"type": "string",
                       "description": "'' (full chain) | hash | decoders"},
            "save": {"type": "boolean", "description": "Save decoded binaries to the workspace"},
            "explain": {"type": "boolean", "description": "Include a rule-based explanation"},
        },
    )
