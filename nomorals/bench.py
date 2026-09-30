"""The system benchmark: `nm bench`.

Measures the real hot paths in the form the system actually uses them —
native C++ vs pure Python for the kernels (vector search, BPE encode,
training batch), plus platform baselines (ULID generation, JSON
round-trip) so a phone can see how its hardware compares.  A saved
baseline (``--save`` → ``$NM_HOME/bench.json``) is compared on later
runs, so drift in either direction is visible.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

__all__ = ["run_benchmarks", "baseline_path", "load_baseline", "save_baseline"]


def _time(fn: Callable[[], Any], reps: int = 3) -> float:
    """Best-of-N seconds, so one GC pause can't skew the number."""
    best = float("inf")
    for _ in range(reps):
        started = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - started)
    return best


def _corpus_words() -> list[str]:
    import random as _random

    rng = _random.Random(31337)
    words = ["the", "cat", "sat", "mat", "dog", "ran", "quick", "brown", "fox",
             "jump", "over", "lazy", "hello", "world", "again", "ab", "abc"]
    return [w for _ in range(200) for w in rng.choices(words, k=rng.randint(2, 8))]


def run_benchmarks(quick: bool = False) -> dict[str, Any]:
    """Run every benchmark.  Returns {name: {...ms, speedup, backend}}."""
    from . import native
    from .training.tokenize import BPETokenizer

    results: dict[str, Any] = {}
    bpe_built = native.bpe_available()
    mlp_built = native.mlp_available()

    # ── vector top-k (memory recall's inner loop) ───────────────────────────
    bench = native.benchmark(n=600 if quick else 2000, dim=64)
    results["vector_topk"] = {
        "python_ms": bench["python_ms"],
        "native_ms": bench["native_ms"],
        "speedup": bench["speedup"],
        "agreement": bench["match"],
        "backend": bench["backend"],
    }

    # ── BPE train + encode (training data pipeline) ─────────────────────────
    words = _corpus_words()[: 400 if quick else 1200]
    corpus = [" ".join(words[i:i + 10]) for i in range(0, len(words), 10)]
    with mock_off(native, "bpe_available"):
        t_train_py = _time(
            lambda: BPETokenizer.train(corpus, vocab_size=512, min_frequency=2))
        tok_py = BPETokenizer.train(corpus, vocab_size=512, min_frequency=2)
    t_train = _time(
        lambda: BPETokenizer.train(corpus, vocab_size=512, min_frequency=2))
    tok = BPETokenizer.train(corpus, vocab_size=512, min_frequency=2)
    t_py = _time(lambda: [tok_py._encode_word(w) for w in words])
    t_native = _time(lambda: [tok._encode_word(w) for w in words])
    results["bpe_train"] = {
        "python_ms": round(t_train_py * 1000, 3),
        "native_ms": round(t_train * 1000, 3),
        "speedup": round(t_train_py / t_train, 1) if t_train > 0 else 0,
        "backend": "native-cpp" if bpe_built else "pure-python",
    }
    results["bpe_encode"] = {
        "python_ms": round(t_py * 1000, 3),
        "native_ms": round(t_native * 1000, 3),
        "speedup": round(t_py / t_native, 1) if t_native > 0 else 0,
        "backend": "native-cpp" if bpe_built else "pure-python",
    }

    # ── training batch (the phone's model trainer) ──────────────────────────
    if mlp_built:
        job = _make_train_job(quick)
        t_train = _time(lambda: job(use_cpp=True))
        t_train_py = _time(lambda: job(use_cpp=False))
        results["train_batch"] = {
            "python_ms": round(t_train_py * 1000, 3),
            "native_ms": round(t_train * 1000, 3),
            "speedup": round(t_train_py / t_train, 1) if t_train > 0 else 0,
            "backend": "native-cpp",
        }
    else:
        results["train_batch"] = {
            "backend": "pure-python",
            "note": "build the kernel: nm native --build",
        }

    # ── memory extraction (the per-message heuristic pass) ──────────────────
    from .memory import extract as _extract

    mem_msgs = _mem_corpus()[: 60 if quick else 240]
    with mock_off(native, "mem_heuristic", return_value=None):
        t_mem_py = _time(lambda: [_extract._heuristic_pass_python(m) for m in mem_msgs])
    t_mem = _time(lambda: [_extract._heuristic_pass(m) for m in mem_msgs])
    results["mem_extract"] = {
        "per_msg_us_py": round(t_mem_py * 1e6 / len(mem_msgs), 2),
        "per_msg_us_native": round(t_mem * 1e6 / len(mem_msgs), 2),
        "speedup": round(t_mem_py / t_mem, 2) if t_mem > 0 else 0,
        "agreement": all(
            [(c.kind, c.importance, c.content, tuple(c.tags))
             for c in _extract._heuristic_pass_python(m)] ==
            [(c.kind, c.importance, c.content, tuple(c.tags))
             for c in _extract._heuristic_pass(m)]
            for m in mem_msgs[:12]),
        "backend": "native-cpp" if native.mem_available() else "pure-python",
    }

    # ── platform baselines ──────────────────────────────────────────────────
    from .core.ids import new_id

    n = 2000 if quick else 20000
    results["ulid"] = {
        "per_1k": round(
            _time(lambda: [new_id() for _ in range(n)]) * 1000 / (n / 1000), 2),
    }
    blob = json.dumps({"k": list(range(2000)), "pad": "x" * 4096})
    results["json"] = {
        "roundtrip_ms": round(
            _time(lambda: json.loads(json.dumps(json.loads(blob)))) * 1000, 3),
    }

    return results


def _make_train_job(quick: bool) -> Callable[[bool], None]:
    """A deterministic training job; each call re-runs it with a fresh
    trainer so the timed work is the training itself, not setup."""
    from .training import trainer as T
    from .training.dataset import Example, Turn
    from .training.tokenize import BPETokenizer

    words = _corpus_words()
    n = 30 if quick else 80
    examples = [
        Example(turns=[
            Turn(role="user", content=" ".join(words[i:i + 8])),
            Turn(role="assistant", content=" ".join(words[i + 8:i + 16])),
        ])
        for i in range(0, n * 16, 16)
    ]
    texts = [e.to_chatml() for e in examples]
    tok = BPETokenizer.train(texts, vocab_size=300, min_frequency=1)

    def job(use_cpp: bool) -> None:
        config = T.TrainConfig(context_window=8, hidden_size=16, epochs=1,
                               batch_size=8, learning_rate=0.05, seed=5,
                               max_examples=1000)
        trainer = T.NativeTrainer(tok, config)
        trainer._cpp_kernel = (lambda: True) if use_cpp else (lambda: False)
        trainer.fit(examples)

    return job


_MEM_SENTENCES = (
    "I live in Lagos with my mum.",
    "I prefer coffee to tea in the morning.",
    "Let's go to the beach this weekend.",
    "You're my best friend.",
    "I work at the bank downtown.",
    "I just bought a new laptop.",
    "Keep it short and casual for me.",
    "How do you feel about us right now?",
    "Do you like the new song?",
    "I'm glad you came back to me.",
    "We should start the project tomorrow.",
    "The game was so good tonight.",
    "I hate when people are late.",
    "My family is coming for dinner.",
    "I slept badly last night, I'm tired.",
    "You make me happy when you text.",
)


def _mem_corpus() -> list[str]:
    """A deterministic mix of owner-like turns for the extraction bench."""
    import random as _random

    rng = _random.Random(2024)
    out: list[str] = []
    for i in range(240):
        n = rng.randint(1, 3)
        out.append(" ".join(rng.choices(_MEM_SENTENCES, k=n)))
    return out


class mock_off:
    """Temporarily force a native kernel 'not available' (reference times)."""

    def __init__(self, module: Any, name: str,
                 return_value: Any = False) -> None:
        self._module = module
        self._name = name
        self._return_value = return_value
        self._patcher = None

    def __enter__(self) -> "mock_off":
        from unittest import mock

        value = self._return_value
        self._patcher = mock.patch.object(
            self._module, self._name, lambda *a, **k: value)
        self._patcher.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._patcher is not None:
            self._patcher.stop()


def baseline_path() -> Path:
    from .core.config import get_settings

    return Path(get_settings().home_path) / "bench.json"


def save_baseline(results: dict[str, Any], quick: bool = False) -> Path:
    path = baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"quick": quick, "results": results}, indent=2),
        encoding="utf-8",
    )
    return path


def load_baseline(quick: bool = False) -> dict[str, Any] | None:
    """The saved baseline, or None if absent — or if it was recorded with a
    different workload size (a quick run must not be judged against a full
    baseline)."""
    raw = None
    path = baseline_path()
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    if raw is None:
        return None
    if isinstance(raw, dict) and "results" in raw:
        return raw["results"] if bool(raw.get("quick", False)) == quick else None
    return raw  # legacy flat format
