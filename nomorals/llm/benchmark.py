"""Benchmark harness: pick Devon's brain with evidence, not vibes.

Runs Devon's ACTUAL task set (code fix, summarization, NL command
parsing, multilingual reply) against each candidate model served by a
local llama-server, measuring latency, tokens/sec, and a keyword-based
quality score. The codebeast-3.8b vs qwen3.5-4b decision is made here.

Usage::

    python -m nomorals.llm.benchmark --models codebeast-3.8b,qwen3.5-4b
    python -m nomorals.llm.benchmark --models qwen3.5-4b --out /tmp/bench.json

``server_factory`` is duck-typed: ``factory(model_id)`` returns an
object with ``generate(prompt) -> str``. The CLI default builds a
:class:`GGUFServerManager` per model and talks to its OpenAI-compatible
``/v1/chat/completions`` endpoint with stdlib urllib (zero new deps).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

#: Devon's own task set — the workload the brain actually serves.
#: ``keywords`` are case-insensitive must-contains for a quality point;
#: ``expected`` (when present) is an exact-match shortcut to 1.0.
TASKS: list[dict[str, Any]] = [
    {
        "id": "code_fix",
        "prompt": (
            "Fix the bug in this Python function. Reply with ONLY the "
            "fixed function, no explanation:\n\n"
            "def add(a, b):\n    return a - b  # should add"
        ),
        "keywords": ["return a + b"],
    },
    {
        "id": "summarize",
        "prompt": (
            "Summarize in one sentence: The Nigerian naira strengthened "
            "against the dollar on Tuesday as the central bank intervened "
            "in the foreign exchange market, injecting liquidity to ease "
            "pressure on the local currency ahead of the holiday season."
        ),
        "keywords": ["naira", "dollar"],
    },
    {
        "id": "nl_parse",
        "prompt": (
            "The user said: 'remind me to call Ada tomorrow at 9am'. "
            "Reply with ONLY a JSON object with keys 'action', 'who', "
            "'when'. No other text."
        ),
        "keywords": ["remind", "Ada"],
    },
    {
        "id": "multilingual",
        "prompt": (
            "Translate to Yoruba, keeping it short and natural: "
            "'Good morning, how are you today?'"
        ),
        "keywords": ["káàárọ̀", "kaaro", "káàárọ"],
    },
]


@dataclass
class TaskResult:
    task_id: str
    model: str
    latency_s: float
    tokens_per_sec: float
    quality: float  # 0.0–1.0
    output_excerpt: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model": self.model,
            "latency_s": round(self.latency_s, 3),
            "tokens_per_sec": round(self.tokens_per_sec, 2),
            "quality": round(self.quality, 3),
            "output_excerpt": self.output_excerpt,
            "error": self.error,
        }


@dataclass
class BenchmarkReport:
    models: list[str]
    results: list[TaskResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-model aggregates: avg latency, avg tok/s, avg quality."""
        out: dict[str, dict[str, float]] = {}
        for model in self.models:
            rows = [r for r in self.results if r.model == model and not r.error]
            if not rows:
                out[model] = {"avg_latency_s": 0.0, "avg_tokens_per_sec": 0.0,
                              "avg_quality": 0.0, "tasks_ok": 0}
                continue
            out[model] = {
                "avg_latency_s": round(sum(r.latency_s for r in rows) / len(rows), 3),
                "avg_tokens_per_sec": round(sum(r.tokens_per_sec for r in rows) / len(rows), 2),
                "avg_quality": round(sum(r.quality for r in rows) / len(rows), 3),
                "tasks_ok": len(rows),
            }
        return out

    def winner(self) -> str:
        """Highest quality wins; ties break on tokens/sec. Empty when no data."""
        scored = [
            (m, s["avg_quality"], s["avg_tokens_per_sec"])
            for m, s in self.summary().items() if s["tasks_ok"]
        ]
        if not scored:
            return ""
        scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
        return scored[0][0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": self.models,
            "started_at": self.started_at,
            "summary": self.summary(),
            "winner": self.winner(),
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


def _score(task: dict[str, Any], output: str) -> float:
    """Quality 0–1: exact match shortcut, else keyword hit fraction."""
    expected = (task.get("expected") or "").strip()
    if expected and output.strip() == expected:
        return 1.0
    keywords: list[str] = [k for k in task.get("keywords", []) if k]
    if not keywords:
        return 1.0 if output.strip() else 0.0
    lowered = output.lower()
    hits = sum(1 for k in keywords if k.lower() in lowered)
    return hits / len(keywords)


def _rough_tokens_per_sec(text: str, latency_s: float) -> float:
    # Word-count proxy — honest approximation, documented as such.
    if latency_s <= 0:
        return 0.0
    return len(text.split()) / latency_s


def run_benchmark(
    server_factory: Callable[[str], Any],
    models: list[str],
    tasks: list[dict[str, Any]] | None = None,
    *,
    per_task_timeout: float = 300.0,
) -> BenchmarkReport:
    """Run every task against every model. Never raises per-task — a
    failed task records its error and the harness moves on."""
    tasks = tasks if tasks is not None else TASKS
    report = BenchmarkReport(models=list(models))
    for model in models:
        try:
            server = server_factory(model)
        except Exception as exc:  # noqa: BLE001
            _log.warning("benchmark: cannot build server for %s: %s", model, exc)
            for task in tasks:
                report.results.append(TaskResult(
                    task_id=task["id"], model=model, latency_s=0.0,
                    tokens_per_sec=0.0, quality=0.0, error=f"server_factory: {exc}"))
            continue
        for task in tasks:
            started = time.time()
            try:
                output = server.generate(task["prompt"])
                latency = time.time() - started
                if latency > per_task_timeout:
                    raise TimeoutError(f"task exceeded {per_task_timeout}s")
                report.results.append(TaskResult(
                    task_id=task["id"], model=model, latency_s=latency,
                    tokens_per_sec=_rough_tokens_per_sec(output, latency),
                    quality=_score(task, output),
                    output_excerpt=output[:200]))
            except Exception as exc:  # noqa: BLE001
                report.results.append(TaskResult(
                    task_id=task["id"], model=model,
                    latency_s=time.time() - started, tokens_per_sec=0.0,
                    quality=0.0, error=str(exc)[:300]))
    return report


# ── default server factory: llama-server via its OpenAI-compatible API ──

class _LlamaServerClient:
    """Thin stdlib client for a running llama-server (chat completions)."""

    def __init__(self, base_url: str, *, timeout: float = 300.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        body = json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 512,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"llama-server chat failed: {exc}") from exc
        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected chat response shape: {exc}") from exc


def default_server_factory(port: int = 8080, cache_dir: str = "models",
                           **manager_kwargs: Any) -> Callable[[str], Any]:
    """Build ``factory(model_id)``: starts a GGUFServerManager per model
    (KV-cache quantization applied per profile) and returns a chat client."""
    from .local_server import GGUFServerManager
    from .model_catalog import brain_spec

    managers: dict[str, Any] = {}

    def factory(model_id: str) -> _LlamaServerClient:
        if model_id not in managers:
            spec = brain_spec(model_id)
            if not spec:
                raise ValueError(f"unknown brain {model_id!r} — see model_catalog.BRAIN_MODELS")
            mgr = GGUFServerManager(port=port, cache_dir=cache_dir, **manager_kwargs)
            model_path = mgr.fetch_gguf(spec["hf_repo"], prefer_quant=spec.get("quant", "Q4_K_M"))
            if not model_path.ok:
                raise RuntimeError("; ".join(model_path.problems))
            diagnosis = mgr.start(model_path.model_path)
            if not diagnosis.ok:
                raise RuntimeError("; ".join(diagnosis.problems))
            managers[model_id] = mgr
        return _LlamaServerClient(managers[model_id].base_url)

    return factory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark Devon's brain candidates.")
    parser.add_argument("--models", default="codebeast-3.8b,qwen3.5-4b",
                        help="comma-separated brain ids from the catalog")
    parser.add_argument("--out", default="",
                        help="write the JSON report here (default: print summary)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    factory = default_server_factory(port=args.port)
    report = run_benchmark(factory, models)
    if args.out:
        path = report.save(args.out)
        print(f"report saved to {path}")
    print(json.dumps(report.summary(), indent=2))
    winner = report.winner()
    print(f"winner: {winner or '(no successful runs)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
