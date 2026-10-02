"""Fan-out / fan-in patterns for the agent swarm (Prompt 02).

:func:`fan_out` spawns N parallel role agents with different angles over one
goal; :func:`fan_in` merges their Blackboard contributions with a chosen
strategy (``concat_dedupe``, ``vote``, ``judge``); :func:`map_reduce`
splits a work list across K workers and reduces the results.

Workers are injected callables, so the patterns are fully testable without
a model.  In production the default worker drives one model call with the
role's system prompt and angle brief.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .base import Budget
from .blackboard import Blackboard
from .role_specs import RoleRegistry, SwarmAgent, default_registry

__all__ = [
    "fan_out", "fan_in", "map_reduce",
    "FanOutResult", "FanInResult", "MapReduceResult",
    "MERGE_STRATEGIES",
]

_log = get_logger(__name__)

MERGE_STRATEGIES = ("concat_dedupe", "vote", "judge")

#: Hard cap on parallel role agents per fan-out (queue the rest).
DEFAULT_MAX_WORKERS = 5


@dataclass
class FanOutResult:
    run_id: str
    role: str
    angles: list[str]
    results: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.errors or bool(self.results)

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "role": self.role,
                "angles": self.angles, "n_results": len(self.results),
                "n_errors": len(self.errors), "errors": self.errors,
                "seconds": round(self.seconds, 3)}


@dataclass
class FanInResult:
    strategy: str
    merged: Any
    n_inputs: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "n_inputs": self.n_inputs,
                "notes": self.notes,
                "merged": self.merged if isinstance(
                    self.merged, (str, int, float, bool, type(None), list, dict))
                else str(self.merged)[:4000]}


@dataclass
class MapReduceResult:
    n_items: int
    n_workers: int
    reduced: Any
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


def _slug(text: str) -> str:
    from ..core.text import slugify

    # canonical: nomorals.core.text.slugify (underscore separator preserved)
    return slugify(text, limit=40, fallback="angle", separator="_")


def fan_out(
    goal: str,
    angles: list[str],
    *,
    role: str = "researcher",
    worker_fn: Callable[[str, SwarmAgent], Any] | None = None,
    blackboard: Blackboard | None = None,
    registry: RoleRegistry | None = None,
    budget: Budget | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    context: Any = None,
    topic: str = "",
) -> FanOutResult:
    """Spawn one role agent per angle in parallel; each publishes its result
    to the Blackboard under its angle.  Every agent gets its own slice of
    the budget so one runaway angle cannot eat the whole run."""
    board = blackboard if blackboard is not None else Blackboard()
    reg = registry or default_registry()
    run_id = f"fanout-{new_id()[-8:]}"
    topic = topic or run_id
    started = time.perf_counter()
    results: list[Any] = [None] * len(angles)
    errors: list[str] = []
    errors_lock = threading.Lock()

    parent_budget = budget or Budget(wall_seconds=900, tokens=600_000)
    # Split the budget across angles; the geometric-series bound in
    # child_budget keeps the total under the parent's ceiling.
    fraction = 1.0 / max(1, len(angles))

    def _one(index: int, angle: str) -> None:
        spec = reg.resolve(role)
        agent = SwarmAgent(
            context, spec,
            budget=parent_budget.child_budget(fraction=fraction),
            worker=(lambda payload, ag, _a=angle: worker_fn(_a, ag))
            if worker_fn else None,
        )
        brief = f"{goal}\n\nAngle: {angle}"
        try:
            res = agent.run({"goal": brief, "angle": angle})
            output = res.output
            if not res.ok:
                raise RuntimeError(str(res.error or "worker failed"))
        except Exception as exc:  # noqa: BLE001 - one bad angle ≠ failed fan-out
            with errors_lock:
                errors.append(f"{angle}: {exc}")
            output = {"error": str(exc), "angle": angle}
        results[index] = output
        board.post(f"{run_id}.{_slug(angle)}", output, author=agent.name,
                   topic=topic,
                   metadata={"angle": angle, "role": spec.name,
                             "run_id": run_id, "index": index})

    workers = min(max_workers, len(angles)) or 1
    with ThreadPoolExecutor(max_workers=workers,
                           thread_name_prefix="fanout") as pool:
        futures = {pool.submit(_one, i, angle): i
                   for i, angle in enumerate(angles)}
        for future in as_completed(futures):
            future.result()  # re-raise unexpected harness errors

    elapsed = time.perf_counter() - started
    if context is not None:
        try:
            context.emit("swarm.fan_out", run_id=run_id, role=role,
                         angles=len(angles), errors=len(errors))
        except Exception:  # noqa: BLE001 - telemetry never breaks fan-out
            pass
    _log.info("fan_out %s: %d angles, %d ok, %d errors (%.1fs)",
              run_id, len(angles), len([r for r in results if r is not None]),
              len(errors), elapsed)
    return FanOutResult(run_id=run_id, role=role, angles=list(angles),
                        results=results, errors=errors, seconds=elapsed)


def fan_in(
    run_id: str,
    strategy: str = "concat_dedupe",
    *,
    blackboard: Blackboard,
    judge_fn: Callable[[list[Any], tuple[str, ...]], dict[str, Any]] | None = None,
    rubric: tuple[str, ...] = (),
    topic: str = "",
) -> FanInResult:
    """Merge one fan-out run's Blackboard contributions."""
    if strategy not in MERGE_STRATEGIES:
        raise ValueError(f"unknown merge strategy {strategy!r}; "
                         f"expected one of {MERGE_STRATEGIES}")
    prefix = f"{run_id}."
    keys = sorted(k for k in blackboard.keys(f"{prefix}*"))
    inputs = [blackboard.get(k) for k in keys]
    inputs = [i for i in inputs if i is not None]

    if strategy == "concat_dedupe":
        merged, notes = _concat_dedupe(inputs)
    elif strategy == "vote":
        merged, notes = _vote(inputs)
    else:  # judge
        merged, notes = _judge(inputs, judge_fn, rubric)

    result = FanInResult(strategy=strategy, merged=merged,
                         n_inputs=len(inputs), notes=notes)
    board_topic = topic or run_id
    blackboard.post(f"{run_id}.merged", result.to_dict(),
                    author="fan_in", topic=board_topic,
                    metadata={"strategy": strategy})
    return result


def _concat_dedupe(inputs: list[Any]) -> tuple[dict[str, Any], str]:
    """Merge findings lists; drop near-duplicates; keep every source."""
    findings: list[dict[str, Any]] = []
    sources: list[str] = []
    seen: set[str] = set()
    for item in inputs:
        if not isinstance(item, dict):
            continue
        for finding in item.get("findings", []) or []:
            text = finding if isinstance(finding, str) else str(
                finding.get("text", finding))
            norm = re.sub(r"\s+", " ", text.lower()).strip()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            findings.append({"text": text, "angle": item.get("angle", "")})
        for source in item.get("sources", []) or []:
            if source not in sources:
                sources.append(source)
    merged = {"findings": findings, "sources": sources,
              "n_angles": len(inputs)}
    return merged, f"merged {len(inputs)} angles → {len(findings)} findings"


def _vote(inputs: list[Any]) -> tuple[dict[str, Any], str]:
    """Majority vote over (label, confidence) judgments, confidence-weighted."""
    weights: dict[str, float] = {}
    counts: dict[str, int] = {}
    for item in inputs:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label", item.get("verdict", "abstain")))
        conf = float(item.get("confidence", 0.5) or 0.5)
        weights[label] = weights.get(label, 0.0) + conf
        counts[label] = counts.get(label, 0) + 1
    if not weights:
        return {"winner": None, "weights": {}}, "no votable inputs"
    winner = max(weights, key=lambda k: weights[k])
    total = sum(weights.values()) or 1.0
    return ({"winner": winner,
             "weight": round(weights[winner] / total, 3),
             "counts": counts,
             "tied": len([w for w in weights.values()
                          if w == weights[winner]]) > 1},
            f"{len(inputs)} votes → {winner!r}")


def _judge(inputs: list[Any],
           judge_fn: Callable[[list[Any], tuple[str, ...]], dict[str, Any]] | None,
           rubric: tuple[str, ...]) -> tuple[Any, str]:
    """A critic-role judge picks the best candidate against a rubric."""
    if judge_fn is None:
        return inputs[0] if inputs else None, "no judge; kept first"
    ruling = judge_fn(inputs, rubric)
    if not isinstance(ruling, dict):
        return inputs[0] if inputs else None, "judge returned garbage; kept first"
    return ruling.get("pick", inputs[0] if inputs else None), str(
        ruling.get("rationale", "judge picked"))


def map_reduce(
    items: list[Any],
    worker_fn: Callable[[list[Any], int], Any],
    reduce_fn: Callable[[list[Any]], Any],
    *,
    k: int = DEFAULT_MAX_WORKERS,
    context: Any = None,
) -> MapReduceResult:
    """Split ``items`` across K workers, then reduce their outputs."""
    started = time.perf_counter()
    chunks = _chunks(items, max(1, k))
    outputs: list[Any] = [None] * len(chunks)
    errors: list[str] = []
    lock = threading.Lock()

    def _one(index: int, chunk: list[Any]) -> None:
        try:
            outputs[index] = worker_fn(chunk, index)
        except Exception as exc:  # noqa: BLE001 - one bad chunk ≠ failed job
            with lock:
                errors.append(f"chunk {index}: {exc}")

    with ThreadPoolExecutor(max_workers=min(k, len(chunks)) or 1,
                           thread_name_prefix="mapreduce") as pool:
        futures = [pool.submit(_one, i, chunk)
                   for i, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            future.result()

    good = [o for o in outputs if o is not None]
    try:
        reduced = reduce_fn(good)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"reduce: {exc}")
        reduced = None
    if context is not None:
        try:
            context.emit("swarm.map_reduce", items=len(items),
                         workers=len(chunks), errors=len(errors))
        except Exception:  # noqa: BLE001 - telemetry never breaks map-reduce
            pass
    return MapReduceResult(n_items=len(items), n_workers=len(chunks),
                           reduced=reduced, errors=errors,
                           seconds=time.perf_counter() - started)


def _chunks(items: list[Any], k: int) -> list[list[Any]]:
    if not items:
        return []
    size = max(1, -(-len(items) // k))  # ceil division
    return [items[i:i + size] for i in range(0, len(items), size)]
