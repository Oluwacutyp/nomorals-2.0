"""Fan-out / fan-in patterns for the agent swarm (Prompt 02).

:func:`fan_out` spawns N parallel role agents with different angles over one
goal; :func:`fan_in` merges their Blackboard contributions with a chosen
strategy (``concat_dedupe``, ``vote``, ``judge``); :func:`map_reduce`
splits a work list across K workers and reduces the results.

Extension #6 (Hark pattern): :func:`fan_out_compare` fans out one worker
per *source* for comparison tasks ("compare X across N sources") with
profile-gated parallelism (workstation 36 / laptop 12 / termux 4 —
:func:`fanout_cap`), and :func:`synthesize_comparison` / :func:`compare`
merge the per-source outputs into a comparison table + summary.

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
    "fan_out_compare", "synthesize_comparison", "compare", "fanout_cap",
    "FanOutResult", "FanInResult", "MapReduceResult", "CompareResult",
    "MERGE_STRATEGIES", "FANOUT_WORKERS_BY_PROFILE",
]

#: Profile-gated ceiling for parallel comparison workers (Hark pattern):
#: workstation fans out 36 parallel browsers/agents, laptop 12, termux 4.
FANOUT_WORKERS_BY_PROFILE = {"workstation": 36, "laptop": 12, "termux": 4}


def fanout_cap(profile: str | None = None) -> int:
    """Max parallel comparison workers for this machine. Never raises.

    Resolution: explicit ``profile`` arg → ``NM_PROFILE`` env →
    Termux ``PREFIX`` heuristic → :func:`nomorals.core.profile.detect_profile`.
    """
    try:
        if profile is None:
            from ..core.profiles import get_profile_kind
            profile = get_profile_kind()
        return FANOUT_WORKERS_BY_PROFILE.get(str(profile or "").lower(), 12)
    except Exception:  # noqa: BLE001 - profile detection is best-effort
        return 12


@dataclass
class CompareResult:
    run_id: str
    goal: str
    sources: list[str]
    results: list[Any] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0
    workers_used: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors or bool([r for r in self.results
                                        if not (isinstance(r, dict)
                                                and r.get("error"))])

    def to_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "goal": self.goal,
                "n_sources": len(self.sources),
                "n_results": len(self.results),
                "n_errors": len(self.errors), "errors": self.errors,
                "workers_used": self.workers_used,
                "seconds": round(self.seconds, 3)}

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
    #: Per-shard status records: {index, n_items, ok, retries, seconds}.
    #: Powers the shard table in render_mapreduce().
    shard_status: list[dict[str, Any]] = field(default_factory=list)


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
    retries: int = 1,
    on_progress: Callable[[int, int], None] | None = None,
) -> MapReduceResult:
    """Split ``items`` across K workers, then reduce their outputs.

    ``retries``: a failed chunk is retried that many extra times before
    it's recorded as an error (per-shard resilience — one flaky chunk
    no longer poisons the job). ``on_progress`` fires as shards finish:
    ``(completed, total)``. Per-shard status lands on
    ``result.shard_status`` for the rendered table.
    """
    started = time.perf_counter()
    chunks = _chunks(items, max(1, k))
    outputs: list[Any] = [None] * len(chunks)
    errors: list[str] = []
    shards: list[dict[str, Any]] = [{} for _ in chunks]
    lock = threading.Lock()
    completed = [0]

    def _emit() -> None:
        if on_progress is None:
            return
        try:
            on_progress(completed[0], len(chunks))
        except Exception:  # noqa: BLE001 — progress never breaks the job
            pass

    def _one(index: int, chunk: list[Any]) -> None:
        shard_started = time.perf_counter()
        attempts = 0
        last_error: str = ""
        while attempts <= max(0, int(retries)):
            attempts += 1
            try:
                outputs[index] = worker_fn(chunk, index)
                last_error = ""
                break
            except Exception as exc:  # noqa: BLE001 - retry, then record
                last_error = f"{type(exc).__name__}: {exc}"
                if attempts <= max(0, int(retries)):
                    _log.debug("map_reduce chunk %d failed (attempt %d), "
                               "retrying: %s", index, attempts, last_error)
        if last_error:
            with lock:
                errors.append(f"chunk {index}: {last_error}")
        with lock:
            shards[index] = {
                "index": index, "n_items": len(chunk),
                "ok": not last_error, "retries": attempts - 1,
                "seconds": round(time.perf_counter() - shard_started, 2),
            }
            completed[0] += 1
        _emit()

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
                           seconds=time.perf_counter() - started,
                           shard_status=shards)


def fan_out_compare(
    goal: str,
    sources: list[str],
    *,
    worker_fn: Callable[[str, int], Any] | None = None,
    max_workers: int | None = None,
    profile: str | None = None,
    context: Any = None,
    blackboard: Blackboard | None = None,
) -> CompareResult:
    """Hark-style N-way fan-out for comparison tasks.

    "Compare X across N sources" → one parallel worker per source, then
    :func:`synthesize_comparison` merges their outputs into a comparison
    table.  ``worker_fn(source, index)`` is injected (a browser-task
    worker in production, a mock in tests).  Parallelism is profile-gated:
    workstation 36 / laptop 12 / termux 4 unless ``max_workers`` overrides.

    One bad source never kills the run; results stay in source order.
    Never raises — failures are captured in ``errors`` / per-source
    ``{"error": ...}`` dicts.
    """
    started = time.perf_counter()
    run_id = f"compare-{new_id()[-8:]}"
    board = blackboard if blackboard is not None else Blackboard()
    errors: list[str] = []
    errors_lock = threading.Lock()
    try:
        srcs = [str(s) for s in (sources or [])]
    except Exception:  # noqa: BLE001
        srcs = []
    results: list[Any] = [None] * len(srcs)

    cap = fanout_cap(profile)
    try:
        if max_workers is not None:
            cap = max(1, int(max_workers))
    except Exception:  # noqa: BLE001
        pass

    def _default_worker(source: str, index: int) -> dict[str, Any]:
        return {"source": source, "index": index,
                "note": "no worker_fn supplied; nothing fetched"}

    worker = worker_fn or _default_worker

    def _one(index: int, source: str) -> None:
        try:
            out = worker(source, index)
        except Exception as exc:  # noqa: BLE001 - one bad source ≠ failed run
            with errors_lock:
                errors.append(f"{source}: {exc}")
            out = {"source": source, "error": str(exc)}
        results[index] = out
        try:
            board.post(f"{run_id}.{_slug(source)}", out, author="fan_out_compare",
                       topic=run_id,
                       metadata={"source": source, "run_id": run_id,
                                 "index": index})
        except Exception:  # noqa: BLE001 - telemetry never breaks fan-out
            pass

    workers_used = min(cap, len(srcs)) or (1 if srcs else 0)
    try:
        if srcs:
            with ThreadPoolExecutor(max_workers=workers_used,
                                   thread_name_prefix="compare") as pool:
                futures = {pool.submit(_one, i, s): i
                           for i, s in enumerate(srcs)}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001 - harness guard
                        with errors_lock:
                            errors.append(f"harness: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"fan-out failed: {exc}")

    elapsed = time.perf_counter() - started
    if context is not None:
        try:
            context.emit("swarm.fan_out_compare", run_id=run_id,
                         sources=len(srcs), errors=len(errors))
        except Exception:  # noqa: BLE001 - telemetry never breaks fan-out
            pass
    _log.info("fan_out_compare %s: %d sources, %d ok, %d errors (%.1fs)",
              run_id, len(srcs),
              len([r for r in results
                   if not (isinstance(r, dict) and r.get("error"))]),
              len(errors), elapsed)
    return CompareResult(run_id=run_id, goal=str(goal or ""),
                         sources=srcs, results=results, errors=errors,
                         seconds=elapsed, workers_used=workers_used)


def synthesize_comparison(results: list[Any],
                          sources: list[str] | None = None) -> dict[str, Any]:
    """Merge per-source worker outputs into a comparison table + summary.

    Each result is expected to be a dict of ``{aspect: value}`` pairs
    (``"source"`` / ``"error"`` keys are treated as metadata).  The union
    of aspects across all sources becomes the table rows, so sources that
    report different fields still line up.  Never raises.
    """
    try:
        items: list[dict[str, Any]] = []
        for i, raw in enumerate(results or []):
            src = (sources[i] if sources and i < len(sources)
                   else f"source_{i}")
            if isinstance(raw, dict):
                err = raw.get("error")
                fields = {k: v for k, v in raw.items()
                          if k not in ("source", "error")}
            else:
                err = f"non-dict result ({type(raw).__name__})"
                fields = {}
            items.append({"source": str(src), "fields": fields,
                          "error": err})
        aspects: list[str] = []
        for item in items:
            for key in item["fields"]:
                if key not in aspects:
                    aspects.append(key)
        table = [{"aspect": a,
                  "values": {it["source"]: it["fields"].get(a)
                             for it in items}}
                 for a in aspects]
        n_ok = len([it for it in items if not it["error"]])
        coverage = {a: sum(1 for it in items if a in it["fields"])
                    for a in aspects}
        failed = [it["source"] for it in items if it["error"]]
        summary = (f"{n_ok}/{len(items)} sources compared; "
                   f"{len(aspects)} aspects; "
                   + (f"failed: {', '.join(failed)}" if failed
                      else "no failures"))
        return {"table": table,
                "markdown": render_comparison_table(table),
                "summary": summary,
                "n_sources": len(items), "n_ok": n_ok,
                "n_aspects": len(aspects), "coverage": coverage,
                "failed_sources": failed}
    except Exception as exc:  # noqa: BLE001
        return {"table": [], "markdown": "", "summary": f"synthesis failed: {exc}",
                "n_sources": 0, "n_ok": 0, "n_aspects": 0,
                "coverage": {}, "failed_sources": []}


def render_comparison_table(table: list[dict[str, Any]]) -> str:
    """Render a synthesized comparison table as a Markdown table.

    Never raises; returns "" for an empty table.
    """
    try:
        if not table:
            return ""
        headers = ["aspect"]
        for row in table:
            for src in (row.get("values") or {}):
                if src not in headers:
                    headers.append(src)

        def cell(v: Any) -> str:
            if v is None:
                return "—"
            text = str(v)
            return text.replace("|", "\\|").replace("\n", " ").strip()[:120]

        lines = ["| " + " | ".join(headers) + " |",
                 "| " + " | ".join("---" for _ in headers) + " |"]
        for row in table:
            vals = row.get("values") or {}
            lines.append("| " + " | ".join(
                [cell(row.get("aspect"))] + [cell(vals.get(h))
                                            for h in headers[1:]]) + " |")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return ""


def compare(goal: str,
            sources: list[str],
            *,
            worker_fn: Callable[[str, int], Any] | None = None,
            max_workers: int | None = None,
            profile: str | None = None,
            context: Any = None,
            blackboard: Blackboard | None = None) -> dict[str, Any]:
    """One-call "compare X across N sources": fan out, then synthesize.

    Returns ``{"run": <CompareResult.to_dict()>, "table": [...],
    "markdown": ..., "summary": ...}``.  Never raises.
    """
    try:
        run = fan_out_compare(goal, sources, worker_fn=worker_fn,
                              max_workers=max_workers, profile=profile,
                              context=context, blackboard=blackboard)
        synth = synthesize_comparison(run.results, run.sources)
        out = {"run": run.to_dict()}
        out.update(synth)
        return out
    except Exception as exc:  # noqa: BLE001
        return {"run": {}, "table": [], "markdown": "",
                "summary": f"compare failed: {exc}", "n_sources": 0,
                "n_ok": 0, "n_aspects": 0, "coverage": {},
                "failed_sources": []}


def _chunks(items: list[Any], k: int) -> list[list[Any]]:
    if not items:
        return []
    size = max(1, -(-len(items) // k))  # ceil division
    return [items[i:i + size] for i in range(0, len(items), size)]


# ── presentation ──────────────────────────────────────────────────────

def render_mapreduce(result: MapReduceResult) -> str:
    """Render a map-reduce run: shard table + reduced output. Never raises."""
    from .render import ICONS, banner, bar, kv, section, table, truncate

    try:
        shards = result.shard_status or []
        ok = sum(1 for s in shards if s.get("ok"))
        lines = [banner("Map-reduce", ICONS["stats"]),
                 kv({"items": result.n_items, "workers": result.n_workers,
                     "shards ok": f"{ok}/{len(shards)}" if shards else "—",
                     "errors": len(result.errors),
                     "time": f"{result.seconds:.1f}s"}.items())]
        if shards:
            lines.append("")
            lines.append(bar(ok / len(shards)))
            rows = [[f"{'✅' if s.get('ok') else '❌'} {s.get('index')}",
                     s.get("n_items", 0),
                     s.get("retries", 0),
                     f"{s.get('seconds', 0)}s"]
                    for s in sorted(shards, key=lambda s: s.get("index", 0))]
            lines.append("")
            lines.append(table(["shard", "items", "retries", "time"], rows))
        if result.errors:
            lines.append("")
            lines.append(section("Errors",
                                 "\n".join(f"❌ {truncate(e, 120)}"
                                           for e in result.errors[:8]),
                                 ICONS["warn"]))
        lines.append("")
        lines.append(section("Reduced",
                             truncate(str(result.reduced), 600),
                             ICONS["star"]))
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — rendering never breaks callers
        return f"map-reduce ({result.n_items} items, render failed)"
