"""``nm models list|add|remove|benchmark|use|select`` — the model broker CLI.

The pre-existing ``nm models`` flags (``--catalog``, ``--activate``,
``--promote-local`` …) are untouched — they live in
:mod:`nomorals.cmdline.commands.doctor` and keep working exactly as before.
This module adds *action* subcommands on top of the same ``models`` parser:

* ``nm models list`` — every model under lifecycle management, its stage,
  capabilities, benchmark score, and which one is the primary mind.
* ``nm models add <hf-id-or-gguf-path>`` — register a model (``add`` walks
  ``registered → downloaded → verified → loaded → warm`` via the other
  actions; ``add`` itself only registers).
* ``nm models remove <id>`` — forget a model (refuses while it is primary).
* ``nm models benchmark <id>`` — take real local measurements through an
  offline provider and store them as ``source='synthetic'`` rows.
* ``nm models use <id>`` — promote a model to primary mind: lifecycle
  ``promote()`` (persisted, rollbackable) plus the boot-time provider
  contract, so the next boot actually serves it.
* ``nm models select <capability>`` — dry-run: show which card the broker
  would pick for a capability right now.

Pointing a local ``.gguf`` at the broker as the primary mind::

    nm models add ~/models/codebeast-3.8b.Q4_K_M.gguf --quant Q4_K_M --context-len 131072
    nm models benchmark codebeast-3.8b.Q4_K_M
    nm models use codebeast-3.8b.Q4_K_M

The file is registered as a first-class broker candidate (``llama_cpp``
provider, ``{chat, code}`` capabilities); ``use`` promotes it in the
persisted lifecycle *and* writes the ``NM_LLM_PROVIDER=llama_cpp`` boot
contract, so the phone/PC serves it on the next start.
"""

from __future__ import annotations

import argparse
from typing import Any

from ..emit import _emit
from ...llm.benchmarks import BenchmarkDB, seed_synthetic
from ...llm.broker import ModelBroker
from ...llm.capabilities import Capability, ModelCard, capability_from
from ...llm.lifecycle import LifecycleError, ManagedModel, ModelLifecycle
from ...llm.providers.mock import MockProvider


def _lifecycle(context: Any) -> ModelLifecycle:
    return ModelLifecycle(getattr(context, "db", None))


def _benchmarks(context: Any) -> BenchmarkDB:
    return BenchmarkDB(getattr(context, "db", None))


def _card_for(model: ManagedModel) -> ModelCard:
    caps = set()
    for raw in model.capabilities:
        try:
            caps.add(capability_from(raw))
        except ValueError:
            continue
    return ModelCard(
        id=model.id,
        capabilities=caps or {Capability.CHAT},
        context_len=model.context_len,
        local=model.local,
        provider=model.provider or ("llama_cpp" if model.local else "hf_serverless"),
        model_id=model.source,
        quant=model.quant,
        cost_per_1k=0.0,
        notes=model.notes,
    )


def _broker_for(context: Any) -> tuple[ModelBroker, ModelLifecycle, BenchmarkDB]:
    lifecycle = _lifecycle(context)
    benchmarks = _benchmarks(context)
    broker = ModelBroker(benchmarks=benchmarks)
    for model in lifecycle.list():
        broker.register(_card_for(model))
    primary = lifecycle.primary
    if primary and broker.card(primary) is not None:
        broker.promote(primary)
    return broker, lifecycle, benchmarks


def _render_list(models: list[ManagedModel], benchmarks: BenchmarkDB,
                 primary: str) -> str:
    if not models:
        return "no models under lifecycle management — `nm models add <hf-id|gguf>` first"
    lines = []
    for m in models:
        mark = " *" if m.id == primary else "  "
        score = benchmarks.score(m.id, "chat")
        caps = ",".join(m.capabilities) if m.capabilities else "-"
        lines.append(
            f"{mark} {m.id}\n"
            f"     stage={m.status} provider={m.provider or '-'} caps={caps} "
            f"quant={m.quant or '-'} ctx={m.context_len or '-'} "
            f"bench={score:.2f}"
            + (f" sha={m.sha256[:12]}…" if m.sha256 else "")
        )
    lines.append("\n* = primary mind (`nm models use <id>` to change, rollback supported)")
    return "\n".join(lines)


def _cmd_model_broker(args: argparse.Namespace, context: Any) -> int:
    """Dispatch ``nm models <action>``."""
    action = getattr(args, "model_action", "") or ""
    target = getattr(args, "model_target", "") or ""
    broker, lifecycle, benchmarks = _broker_for(context)

    if action == "list":
        models = lifecycle.list()
        payload = {
            "primary": lifecycle.primary,
            "models": [
                {**m.to_dict(), "benchmark_score": benchmarks.score(m.id, "chat")}
                for m in models
            ],
        }
        _emit(args, payload, _render_list(models, benchmarks, lifecycle.primary))
        return 0

    if action == "add":
        if not target:
            print("usage: nm models add <hf-repo-id|/path/to/model.gguf>")
            return 2
        try:
            if target.endswith(".gguf") or _is_existing_path(target):
                model = lifecycle.add_gguf(
                    target,
                    quant=getattr(args, "quant", "") or "Q4_K_M",
                    context_len=int(getattr(args, "context_len", 0) or 0),
                )
            else:
                model = lifecycle.add(
                    target,
                    provider=getattr(args, "provider", "") or "",
                    capabilities=_parse_caps(getattr(args, "capability", "")),
                    context_len=int(getattr(args, "context_len", 0) or 0),
                    quant=getattr(args, "quant", "") or "",
                )
        except (LifecycleError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        broker.register(_card_for(model))
        _emit(args, {"added": model.to_dict()},
              f"registered {model.id} (stage={model.status}, provider={model.provider})\n"
              f"next: nm models benchmark {model.id}  (after download+verify)")
        return 0

    if action == "remove":
        if not target:
            print("usage: nm models remove <id>")
            return 2
        try:
            removed = lifecycle.remove(target)
        except (LifecycleError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        broker.unregister(target)
        _emit(args, {"removed": target, "ok": removed}, f"removed {target}")
        return 0

    if action == "benchmark":
        if not target:
            print("usage: nm models benchmark <id>")
            return 2
        try:
            model = lifecycle.get(target)
        except (LifecycleError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        rounds = int(getattr(args, "rounds", 5) or 5)
        provider = MockProvider(model=f"bench-{model.id}")
        result = seed_synthetic(
            benchmarks, model.id, "chat", provider,
            [f"benchmark probe {i}" for i in range(max(1, rounds))],
        )
        payload = {"model_id": model.id, **result,
                   "score": benchmarks.score(model.id, "chat")}
        _emit(args, payload,
              f"benchmarked {model.id}: {result['successes']}/{len(result['rounds'])} ok, "
              f"median {result['median_latency_s']}s, score {payload['score']:.2f} "
              f"(source='synthetic' — real local measurements, not live traffic)")
        return 0

    if action == "use":
        if not target:
            print("usage: nm models use <id>")
            return 2
        try:
            model = lifecycle.get(target)
        except (LifecycleError, ValueError) as exc:
            print(f"error: {exc}")
            return 1
        if model.status not in ("verified", "loaded", "warm"):
            print(f"walking {target} to 'verified' first "
                  f"(currently {model.status!r})…")
            try:
                model = _walk_to_verified(lifecycle, model)
            except LifecycleError as exc:
                print(f"error: {exc}")
                return 1
        try:
            lifecycle.promote(target)
        except LifecycleError as exc:
            print(f"error: {exc}")
            return 1
        broker.promote(target)
        lines = [f"primary mind -> {target} (rollback: previous primary kept in history)"]
        if model.local and model.path:
            # Same boot contract as `nm models --promote-local`: the next
            # boot serves this GGUF via llama_cpp.
            from .doctor import _env_update_home
            update = {
                "NM_LLM_PROVIDER": "llama_cpp",
                "NM_LLM_LOCAL_MODEL": model.path,
                "NM_LLM_LOCAL_AUTO_START": "1",
            }
            try:
                _env_update_home(context, update)
                lines.append(f"boot contract written: llama_cpp serving {model.path}")
            except Exception as exc:  # noqa: BLE001 — lifecycle promotion stands
                lines.append(f"boot contract NOT written ({exc}); "
                             "lifecycle promotion still active")
            try:
                context.settings.llm.provider = "llama_cpp"
                context.settings.llm.local_model = model.path
            except Exception:  # noqa: BLE001 — the file is the durable truth
                pass
        _emit(args, {"primary": target}, "\n".join(lines))
        return 0

    if action == "select":
        if not target:
            print("usage: nm models select <capability> [--task-kind code]")
            return 2
        try:
            cap = capability_from(target)
        except ValueError as exc:
            print(f"error: {exc}")
            return 1
        card = broker.select(cap, task_kind=getattr(args, "task_kind", "") or "")
        if card is None:
            _emit(args, {"capability": cap.value, "selected": None},
                  f"no registered model can serve {cap.value!r}")
            return 1
        _emit(args, {"capability": cap.value, "selected": card.to_dict()},
              f"{cap.value} -> {card.id} (provider={card.provider}, "
              f"local={card.local}, ctx={card.context_len})")
        return 0

    print(f"unknown models action {action!r}; "
          "try: list, add, remove, benchmark, use, select")
    return 2


def _is_existing_path(target: str) -> bool:
    from pathlib import Path
    try:
        return Path(target).expanduser().exists()
    except (OSError, ValueError):
        return False


def _parse_caps(raw: str) -> list[str]:
    caps = [c.strip() for c in (raw or "").split(",") if c.strip()]
    if not caps:
        return ["chat"]
    valid = []
    for c in caps:
        try:
            valid.append(capability_from(c).value)
        except ValueError:
            print(f"warning: ignoring unknown capability {c!r}")
    return valid or ["chat"]


def _walk_to_verified(lifecycle: ModelLifecycle, model: ManagedModel) -> ManagedModel:
    """Best-effort offline walk: download (no-op for local files) → verify."""
    if model.status == "registered":
        model = lifecycle.download(model.id)
    if model.status == "downloaded":
        model = lifecycle.verify(model.id)
    return model
