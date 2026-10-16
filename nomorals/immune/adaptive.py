"""Adaptive immunity — antigen presentation + injected labelling.

The adaptive layer sees what the innate layer cannot classify: a
suspicious bundle of content / tool-call / capability / organ / source.
It presents the antigen (assigns it an id) and asks an INJECTED
labelling function for a verdict. The adaptive layer never calls an
LLM itself — the caller injects ``label_fn`` (which may be an LLM, a
human, or a deterministic classifier; the immune system does not care
and does not know).

Verdicts: ``real_threat`` | ``variant`` | ``noise``. Noise never
creates an antibody: a noise-labelled antigen produces no antibody,
ever, by construction (``creates_antibody`` is False).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Dict

__all__ = [
    "AntigenBundle",
    "LabelledAntigen",
    "present",
    "label_antigen",
    "should_create_antibody",
    "VERDICTS",
]

VERDICTS = ("real_threat", "variant", "noise")


@dataclass
class AntigenBundle:
    """One suspicious thing, presented for labelling."""

    content: str = ""
    tool_call: Any = None
    capability: str | None = None
    organ: str | None = None
    ts: float = field(default_factory=time.time)
    source: str | None = None
    id: str | None = None


@dataclass
class LabelledAntigen:
    bundle: AntigenBundle
    verdict: str
    confidence: float
    reasoning: str
    creates_antibody: bool = False


def present(bundle: AntigenBundle | Dict[str, Any]) -> AntigenBundle:
    """Present an antigen: normalize to :class:`AntigenBundle` and give
    it an id if it does not have one. Returns the bundle."""
    if isinstance(bundle, dict):
        data = dict(bundle)
        data.setdefault("id", None)
        bundle = AntigenBundle(**{k: v for k, v in data.items() if k in AntigenBundle.__dataclass_fields__})
    if bundle.id is None:
        bundle.id = uuid.uuid4().hex
    return bundle


def label_antigen(
    bundle: AntigenBundle | Dict[str, Any],
    label_fn: Callable[[Dict[str, Any]], Dict[str, Any]] | None,
) -> LabelledAntigen:
    """Label a presented antigen with an INJECTED labelling function.

    ``label_fn`` receives the bundle as a plain dict and must return a
    dict with ``verdict`` (one of :data:`VERDICTS`), ``confidence`` and
    ``reasoning``. The adaptive layer never calls an LLM itself, so
    ``label_fn=None`` raises instead of silently inventing a verdict.
    Noise never creates an antibody.
    """
    if label_fn is None:
        raise ValueError(
            "label_antigen requires an injected label_fn — adaptive never "
            "calls an LLM itself"
        )
    presented = present(bundle)
    if is_dataclass(presented):
        bundle_dict: Dict[str, Any] = asdict(presented)
    else:
        bundle_dict = dict(presented)
    result = label_fn(bundle_dict)
    if not isinstance(result, dict):
        raise ValueError("label_fn must return a dict with verdict/confidence/reasoning")
    verdict = result.get("verdict")
    if verdict not in VERDICTS:
        raise ValueError(f"label_fn returned invalid verdict {verdict!r}; expected one of {VERDICTS}")
    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("label_fn confidence must be numeric") from exc
    reasoning = str(result.get("reasoning", "") or "")
    creates = verdict in ("real_threat", "variant")
    return LabelledAntigen(
        bundle=presented,
        verdict=verdict,
        confidence=confidence,
        reasoning=reasoning,
        creates_antibody=creates,
    )


def should_create_antibody(labelled: LabelledAntigen) -> bool:
    """True iff this labelled antigen may seed an antibody. Noise: never."""
    return bool(labelled.creates_antibody) and labelled.verdict != "noise"
