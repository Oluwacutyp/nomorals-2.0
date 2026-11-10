"""Thymus — negative selection for antibodies (build FIRST).

The thymus is where candidate antibodies prove they do not attack the
body itself. A candidate pattern is tested against the SelfCorpus: a
list of normal-activity exemplars (ordinary owner text, ordinary
tool-call shapes, ordinary capability names). If the pattern fires on
any self exemplar it is rejected — an autoimmune antibody must never
be accepted.

Every accepted antibody carries a ``reason`` (why it exists); every
retired antibody carries a ``retire_reason`` (why it was retired, e.g.
false positive / allergy). Reasonless antibodies are rejected.

The corpus itself is versioned data with provenance (doc 20 P4,
:mod:`nomorals.immune.corpus`): every exemplar names where it came
from — a shipped seed, one of Devon's own verified fixes, an
owner-directed police receipt — and a rejection CITES the exemplars
it matched, so the refusal names its evidence.

Standard library only. No ``nomorals.*`` imports: the thymus talks to
plain data (regex strings, exemplar dicts/dataclasses).
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Tuple

__all__ = [
    "Antibody",
    "SelfExemplar",
    "SelfCorpus",
    "exemplar_citation",
    "test_antibody",
    "Thymus",
]


@dataclass
class Antibody:
    """A defensive pattern that has survived negative selection."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pattern: str = ""
    pathogen_class: str = ""
    reason: str = ""
    created_ts: float = field(default_factory=time.time)
    version: int = 1
    status: str = "proposed"  # proposed | active | retired
    retire_reason: str | None = None


@dataclass
class SelfExemplar:
    """One exemplar of normal activity: text, tool-call shape, capability."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    text: str = ""
    tool_call: Any = None
    capability: str = ""
    # Provenance (doc 20 P4): where this exemplar came from, so a
    # negative-selection refusal can cite its evidence. ``provenance``
    # is the source KIND ("seed", "verified_fix", "owner_receipt");
    # ``source_ref`` names the source record (e.g.
    # "verified_fix:<signature>"); ``harvested_ts`` is when a
    # harvested exemplar was admitted (None for seeds and for
    # exemplars built by hand); ``corpus_version`` is the corpus
    # version that admitted it (0 = seed / pre-versioning).
    provenance: str = ""
    source_ref: str = ""
    harvested_ts: float | None = None
    corpus_version: int = 0


def exemplar_citation(exemplar: SelfExemplar) -> str:
    """One exemplar as a refusal cites it: id, provenance, source.

    The citation is the evidence handle — a rejected candidate's
    owner (or a later reader) can resolve exactly which record of
    Devon's own history the pattern fired on.
    """
    kind = exemplar.provenance or "unprovenanced"
    if exemplar.source_ref:
        return f"{exemplar.id} [{kind}: {exemplar.source_ref}]"
    return f"{exemplar.id} [{kind}]"


class SelfCorpus:
    """The list of normal-activity exemplars antibodies must not fire on."""

    def __init__(self, exemplars: Iterable[SelfExemplar] | None = None) -> None:
        self.exemplars: list[SelfExemplar] = list(exemplars or [])

    def add(
        self,
        text: str = "",
        tool_call: Any = None,
        capability: str = "",
    ) -> SelfExemplar:
        exemplar = SelfExemplar(text=text, tool_call=tool_call, capability=capability)
        self.exemplars.append(exemplar)
        return exemplar

    # Alias with the explicit name used by callers that think in exemplars.
    def add_exemplar(
        self,
        text: str = "",
        tool_call: Any = None,
        capability: str = "",
    ) -> SelfExemplar:
        return self.add(text=text, tool_call=tool_call, capability=capability)

    def __len__(self) -> int:
        return len(self.exemplars)

    def __iter__(self):
        return iter(self.exemplars)


def _exemplar_haystacks(exemplar: SelfExemplar) -> list[str]:
    haystacks = [exemplar.text or "", exemplar.capability or ""]
    tool_call = exemplar.tool_call
    if tool_call is None:
        pass
    elif isinstance(tool_call, str):
        haystacks.append(tool_call)
    elif isinstance(tool_call, dict):
        import json

        haystacks.append(json.dumps(tool_call, sort_keys=True, default=str))
        name = tool_call.get("name") or tool_call.get("tool") or ""
        if name:
            haystacks.append(str(name))
    else:
        haystacks.append(str(tool_call))
    return haystacks


def test_antibody(
    pattern: str, corpus: SelfCorpus | Iterable[SelfExemplar]
) -> Tuple[bool, list[SelfExemplar]]:
    """Negative selection: does ``pattern`` fire on any self exemplar?

    Returns ``(fires_on_self, matched_exemplars)``. An invalid regex is
    a caller bug, not a quiet pass: it raises :class:`ValueError`.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid antibody pattern {pattern!r}: {exc}") from exc
    if isinstance(corpus, SelfCorpus):
        exemplars = corpus.exemplars
    else:
        exemplars = list(corpus)
    matched: list[SelfExemplar] = []
    for exemplar in exemplars:
        if any(compiled.search(h) for h in _exemplar_haystacks(exemplar)):
            matched.append(exemplar)
    return (len(matched) > 0, matched)


class Thymus:
    """Propose / accept / retire antibodies under negative selection."""

    def __init__(self, corpus: SelfCorpus | None = None) -> None:
        self.corpus = corpus or SelfCorpus()

    def add_self_exemplar(
        self, text: str = "", tool_call: Any = None, capability: str = ""
    ) -> SelfExemplar:
        return self.corpus.add(text=text, tool_call=tool_call, capability=capability)

    def propose(
        self, pattern: str, pathogen_class: str, reason: str
    ) -> Antibody:
        """Draft a candidate antibody. A missing reason is rejected here,
        before negative selection, so a reasonless antibody never exists
        even as a draft."""
        if not reason or not str(reason).strip():
            raise ValueError("antibody requires a reason — reasonless antibodies are rejected")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"invalid antibody pattern {pattern!r}: {exc}") from exc
        return Antibody(
            pattern=pattern, pathogen_class=pathogen_class, reason=str(reason).strip()
        )

    def accept(
        self, antibody: Antibody, *, corpus: SelfCorpus | None = None
    ) -> Antibody:
        """Accept a proposed antibody after negative selection.

        Raises :class:`ValueError` if the antibody has no reason or if
        its pattern fires on any self exemplar (autoimmune — rejected).
        """
        if not antibody.reason or not str(antibody.reason).strip():
            raise ValueError("antibody requires a reason — reasonless antibodies are rejected")
        fires, matched = test_antibody(antibody.pattern, corpus or self.corpus)
        if fires:
            cited = "; ".join(exemplar_citation(e) for e in matched)
            raise ValueError(
                f"antibody {antibody.id} rejected: pattern fires on self "
                f"exemplar(s) {cited} — autoimmune, refusing to accept"
            )
        antibody.status = "active"
        return antibody

    def accept_antibody(
        self, antibody: Antibody, *, corpus: SelfCorpus | None = None
    ) -> Antibody:
        """Alias for :meth:`accept` (the name the spec uses)."""
        return self.accept(antibody, corpus=corpus)

    def retire(self, antibody: Antibody, retire_reason: str) -> Antibody:
        """Retire an antibody. A retire_reason is mandatory: every
        retired antibody carries why it was retired (e.g. allergy /
        false positive)."""
        if not retire_reason or not str(retire_reason).strip():
            raise ValueError("retiring an antibody requires a retire_reason")
        antibody.status = "retired"
        antibody.retire_reason = str(retire_reason).strip()
        return antibody

    def retire_antibody(self, antibody: Antibody, retire_reason: str) -> Antibody:
        return self.retire(antibody, retire_reason)
