"""Counterfactual simulation for real decisions.

Generalizes ``books/branches.py`` what-if (story branches: fork a
chapter, see what follows) and ``planning/graph.py``
``WorldGraph.simulate_disruption`` (dry-run blast radius, never writes
state) to REAL decisions: "what happens if I change X?" — evaluated
against the live world graph before anything is committed.

A decision option is a node whose state would change.  Simulation
propagates along the graph's existing ``depends_on`` / ``blocks``
edges (the same propagation ``simulate_disruption`` uses), so the
blast radius is the graph's own structure, not a guess.  Nothing here
mutates the graph: ``mark_disrupted`` is never called.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

_log = logging.getLogger(__name__)

__all__ = [
    "DecisionOption",
    "CounterfactualResult",
    "simulate_counterfactual",
    "compare_options",
    "recommend_option",
    "whatif_for_options",
]


@dataclass
class DecisionOption:
    """One candidate decision: change ``node_id`` to ``new_state``."""

    node_id: str
    label: str = ""
    new_state: str = ""
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"node_id": self.node_id, "label": self.label,
                "new_state": self.new_state, "description": self.description}


@dataclass
class CounterfactualResult:
    """What-if outcome for one option (dry-run, graph untouched)."""

    option: DecisionOption
    ok: bool
    total_affected: int = 0
    max_depth: int = 0
    affected_labels: list[str] = field(default_factory=list)
    layers: dict[str, list[str]] = field(default_factory=dict)
    reason: str = ""

    @property
    def blast_radius(self) -> int:
        return self.total_affected

    def to_dict(self) -> dict[str, Any]:
        return {
            "option": self.option.to_dict(), "ok": self.ok,
            "total_affected": self.total_affected,
            "max_depth": self.max_depth,
            "affected_labels": list(self.affected_labels),
            "layers": {k: list(v) for k, v in self.layers.items()},
            "reason": self.reason,
        }


def simulate_counterfactual(
    graph: Any,
    option: DecisionOption,
) -> CounterfactualResult:
    """Simulate one decision against the world graph (never mutates it).

    Uses ``WorldGraph.simulate_disruption`` — the dry-run blast-radius
    tree — so the propagation rules are the graph's own, shared with
    every other disruption analysis in the system.
    """
    node = graph.get(option.node_id) if hasattr(graph, "get") else None
    if node is None:
        return CounterfactualResult(
            option=option, ok=False,
            reason=f"node not found: {option.node_id!r}")
    try:
        sim = graph.simulate_disruption(option.node_id)
    except Exception as exc:  # noqa: BLE001 — simulation failure is a result
        return CounterfactualResult(option=option, ok=False,
                                    reason=f"simulation failed: {exc}")
    if not sim.get("ok"):
        return CounterfactualResult(
            option=option, ok=False,
            reason=str(sim.get("reason") or "simulation failed"))
    affected: list[str] = []
    for _depth, labels in sorted((sim.get("layers") or {}).items()):
        affected.extend(labels)
    return CounterfactualResult(
        option=option, ok=True,
        total_affected=int(sim.get("total_affected") or 0),
        max_depth=int(sim.get("max_depth") or 0),
        affected_labels=sorted(set(affected)),
        layers={str(k): list(v) for k, v in (sim.get("layers") or {}).items()},
        reason=(f"changing {node.label!r} to {option.new_state!r} would "
                f"affect {sim.get('total_affected')} node(s)"),
    )


def compare_options(
    graph: Any,
    options: Sequence[DecisionOption],
) -> list[CounterfactualResult]:
    """Simulate every option; results sorted by blast radius (smallest first).

    Invalid options (unknown node) sort last with ``ok=False`` — they
    are reported, never silently dropped.
    """
    results = [simulate_counterfactual(graph, opt) for opt in options]
    results.sort(key=lambda r: (not r.ok, r.total_affected, r.max_depth))
    return results


def recommend_option(
    graph: Any,
    options: Sequence[DecisionOption],
    *,
    max_acceptable_blast: int | None = None,
) -> dict[str, Any]:
    """Recommend the least-disruptive viable option, with full evidence.

    The recommendation is a comparison, not a verdict: every option's
    blast radius is returned, so the owner can see what "least
    disruptive" cost.  When ``max_acceptable_blast`` is set, options
    above it are marked ``over_budget`` rather than recommended.
    """
    results = compare_options(graph, options)
    viable = [r for r in results if r.ok]
    recommendation: CounterfactualResult | None = None
    over_budget: list[str] = []
    if viable:
        if max_acceptable_blast is not None:
            within = [r for r in viable
                      if r.total_affected <= max_acceptable_blast]
            over_budget = [r.option.label or r.option.node_id
                           for r in viable if r not in within]
            recommendation = within[0] if within else None
        else:
            recommendation = viable[0]
    return {
        "recommendation": (recommendation.option.to_dict()
                           if recommendation else None),
        "recommendation_blast": (recommendation.total_affected
                                 if recommendation else None),
        "over_budget": over_budget,
        "options": [r.to_dict() for r in results],
    }


def whatif_for_options(
    graph: Any,
    options: Sequence[str],
    *,
    max_acceptable_blast: int | None = None,
) -> dict[str, Any]:
    """Match free-text decision options to world-graph nodes and compare them.

    An option names the thing that would change — a graph node id
    (``wg_...``) or a node label (``"Launch"``).  Matching is exact only
    (node id, then case-insensitive label): a loose match would simulate
    the wrong node and present its blast radius as the option's.  Options
    that match nothing — or a label shared by several nodes — are reported
    in ``unmatched`` with the reason, never dropped and never force-matched.
    The return is the :func:`recommend_option` payload plus ``unmatched``.

    Never raises and never mutates the graph: an unreadable graph or a
    failed lookup degrades to unmatched entries, not an exception.
    """
    seen: set[str] = set()
    clean: list[str] = []
    for raw in options or []:
        text = (raw or "").strip()
        if text and text not in seen:
            seen.add(text)
            clean.append(text)
    try:
        nodes = list(graph.list_nodes()) if hasattr(graph, "list_nodes") else []
    except Exception as exc:  # noqa: BLE001 — an unreadable graph simulates nothing
        return {
            "recommendation": None, "recommendation_blast": None,
            "over_budget": [], "options": [],
            "unmatched": [{"option": text,
                           "reason": f"world graph unreadable: {exc}"}
                          for text in clean],
        }
    by_id = {n.node_id: n for n in nodes}
    by_label: dict[str, list] = {}
    for node in nodes:
        by_label.setdefault((node.label or "").strip().lower(), []).append(node)
    unmatched: list[dict[str, str]] = []
    matched: list[DecisionOption] = []
    for text in clean:
        node = by_id.get(text)
        if node is None and hasattr(graph, "get"):
            try:
                node = graph.get(text)
            except Exception:  # noqa: BLE001 — a failed lookup is "no match"
                node = None
        if node is None:
            hits = by_label.get(text.lower(), [])
            if len(hits) == 1:
                node = hits[0]
            elif len(hits) > 1:
                unmatched.append({
                    "option": text,
                    "reason": (f"label matches {len(hits)} nodes — use a "
                               f"node id (e.g. {hits[0].node_id}) to pick one"),
                })
                continue
        if node is None:
            unmatched.append({
                "option": text,
                "reason": "no world-graph node matches this option",
            })
            continue
        matched.append(DecisionOption(node_id=node.node_id, label=text,
                                      new_state=text))
    result = recommend_option(graph, matched,
                              max_acceptable_blast=max_acceptable_blast)
    result["unmatched"] = unmatched
    return result
