"""Training-data tools: the conversation→training agent's control surface.

`train_mine` runs the ConversationMiner over the partner's own chat
history and writes fine-tune-ready bundles (internal JSONL + Alpaca +
ShareGPT + ChatML + manifest) into the training data directory,
registering the result in the dataset registry so `nm train` can pick
it up by id.  `train_datasets` lists what is registered and what the
miner has produced.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..training.collect import ConversationMiner
from ..training.dataset import DatasetRegistry

_log = get_logger(__name__)

__all__ = ["register"]


def _data_dir(context: Any):
    from pathlib import Path

    settings = getattr(context, "settings", None)
    training = getattr(settings, "training", None) if settings else None
    rel = getattr(training, "data_dir", "") or ""
    if settings is not None and rel:
        return Path(settings.resolve(rel))
    if settings is not None:
        return settings.home_path / "training" / "datasets"
    return Path.cwd() / "workspace" / "training" / "datasets"


def mine(context: Any, *, name: str = "", min_score: str = "",
         limit: str = "", since_hours: str = "") -> dict[str, Any]:
    try:
        min_score_f = float(min_score) if str(min_score or "").strip() else 0.3
    except ValueError:
        raise ToolError(f"bad min_score {min_score!r}") from None
    try:
        limit_i = int(limit) if str(limit or "").strip() else 400
    except ValueError:
        raise ToolError(f"bad limit {limit!r}") from None
    since = 0.0
    if str(since_hours or "").strip():
        try:
            since = time.time() - float(since_hours) * 3600
        except ValueError:
            raise ToolError(f"bad since_hours {since_hours!r}") from None
    miner = ConversationMiner(context.db)
    return miner.export(
        output_dir=_data_dir(context),
        name=(name or "").strip(),
        since=since,
        min_score=min_score_f,
        limit=limit_i,
        register=True,
    )


def datasets(context: Any, *, limit: str = "") -> dict[str, Any]:
    limit_i = max(1, min(200, int(limit or 20)))
    registry = DatasetRegistry(context.db)
    rows = []
    for dataset in registry.list(limit=limit_i):
        rows.append({
            "id": dataset.id,
            "name": dataset.name,
            "kind": getattr(dataset, "kind", ""),
            "rows": getattr(dataset, "rows", 0),
            "bytes": getattr(dataset, "bytes", 0),
            "path": getattr(dataset, "path", ""),
        })
    stats = registry.stats()
    from ..training.free_datasets import catalog

    free_catalog = [
        {
            "name": e["name"],
            "id": e["id"],
            "kind": e["kind"],
            "license": e["license"],
            "use": e["use"],
            "verified": e.get("verified", False),
        }
        for e in catalog()
    ]
    # recently mined bundles (manifests live next to the jsonl)
    mined: list[dict[str, Any]] = []
    data_dir = _data_dir(context)
    if data_dir.is_dir():
        for manifest in sorted(data_dir.glob("*.manifest.json"),
                               key=lambda p: p.stat().st_mtime, reverse=True)[:10]:
            try:
                mined.append(json.loads(manifest.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return {
        "datasets": rows,
        "registry_stats": stats,
        "recently_mined": mined,
        "free_catalog": free_catalog,
        "fetch_hint": "dataset_fetch ref=<name or org/name> max_rows=<n>",
    }


def fetch(context: Any, *, ref: str = "", max_rows: str = "") -> dict[str, Any]:
    from ..training.free_datasets import fetch_dataset, make_registry_register

    ref = (ref or "").strip()
    if not ref:
        raise ToolError("dataset_fetch needs a ref (catalog name or org/name)")
    try:
        rows_i = int(max_rows or 500)
    except ValueError:
        raise ToolError(f"bad max_rows {max_rows!r}") from None
    out = fetch_dataset(ref, _data_dir(context), max_rows=rows_i)
    make_registry_register(context)(out)
    return out


def mix(context: Any, *, sources: str = "", persona: str = "",
        rows: str = "", colab: str = "true",
        base_model: str = "") -> dict[str, Any]:
    """Build the persona fine-tune mix from fetched datasets.

    ``sources`` = comma list of catalog names / fetched file stems
    (e.g. ``"open-hermes-25,ultra-data-agent,dolphin-2.9"``) — resolved
    to the training data dir.  ``persona`` = the persona text (empty →
    NM_HOME/persona.txt → built-in default).  Writes the bundle +
    manifest + Colab script into the training data dir and registers
    the messages-format JSONL.
    """
    from ..training.finetune import (DEFAULT_TARGET_ROWS, build_persona_mix,
                                     load_persona, write_colab_notebook,
                                     write_colab_script)

    names = [s.strip() for s in (sources or "").split(",") if s.strip()]
    if not names:
        raise ToolError(
            "train_mix needs sources — comma list of fetched datasets, "
            "e.g. sources=open-hermes-25,ultra-data-agent")
    data_dir = _data_dir(context)
    resolved = []
    missing = []
    for name in names:
        candidate = data_dir / f"{name}.jsonl"
        if candidate.is_file():
            resolved.append((name, str(candidate)))
        else:
            missing.append(name)
    if not resolved:
        raise ToolError(
            f"no fetched files found for {names} in {data_dir} — "
            "fetch first: /data fetch <name> [rows] (or nm data fetch)")
    if missing:
        _log.warning("mix sources not fetched (skipped): %s", missing)

    home = getattr(context.settings, "home", "") or ""
    persona_text = (persona or "").strip() or load_persona(
        str(Path(home).expanduser() / "persona.txt") if home else "")
    try:
        rows_i = max(100, min(int(rows or DEFAULT_TARGET_ROWS), 500_000))
    except ValueError:
        rows_i = DEFAULT_TARGET_ROWS

    from ..training.finetune import MixSource

    srcs = [MixSource(name=n, path=path) for n, path in resolved]
    out_base = data_dir / "persona-mix"
    manifest = build_persona_mix(srcs, persona_text, out_base,
                                 target_rows=rows_i)
    # register the messages-format bundle
    dataset_id = ""
    try:
        ds = DatasetRegistry(context.db).register(
            name="persona-mix",
            path=manifest["outputs"]["messages_jsonl"],
            kind="messages",
            metadata={"persona_chars": manifest["persona_chars"],
                      "sources": [s.name for s in srcs]},
        )
        dataset_id = ds.id
    except Exception as exc:  # noqa: BLE001 — registration is best-effort
        _log.warning("persona-mix registration failed: %s", exc)
    colab_path = notebook_path = ""
    if str(colab or "true").lower() not in {"0", "false", "no", "off"}:
        from ..training.finetune import DEFAULT_COLAB_BASE

        base = (base_model or "").strip() or DEFAULT_COLAB_BASE
        try:
            colab_path = write_colab_script(out_base, base_model=base)
        except Exception as exc:  # noqa: BLE001
            _log.warning("colab script write failed: %s", exc)
        # the zero-phone-data doc: Colab pulls the datasets itself,
        # streams them, mixes with the persona, trains, and exports a
        # small LoRA GGUF — the phone only uploads/downloads KBs..MBs
        try:
            notebook_path = write_colab_notebook(
                out_base, base_model=base, persona=manifest["persona"],
                sources=_colab_source_specs(resolved),
                target_rows=manifest["target_rows"])
        except Exception as exc:  # noqa: BLE001
            _log.warning("colab notebook write failed: %s", exc)
    return {
        "ok": True,
        "rows": manifest["rows"],
        "target": manifest["target_rows"],
        "per_source": manifest["per_source"],
        "outputs": manifest["outputs"],
        "colab_script": colab_path,
        "colab_notebook": notebook_path,
        "dataset_id": dataset_id,
        "missing_sources": missing,
    }


def _colab_source_specs(resolved: list[tuple[str, str]]) -> list[dict[str, Any]]:
    """Map each fetched file back to its HF source for the Colab notebook.

    The fetch manifest (`<name>.manifest.json`) records the HF id and
    config actually used; the catalog supplies the row normalizer.
    Weights/caps are borrowed from the default recipe when the source
    is a known one, so the in-Colab mix mirrors the phone-side mix.
    """
    from ..training.free_datasets import FREE_DATASET_CATALOG
    from ..training.finetune import DEFAULT_COLAB_SOURCES

    specs: list[dict[str, Any]] = []
    for name, path in resolved:
        manifest: dict[str, Any] = {}
        try:
            stem = Path(path).name
            mpath = Path(path).with_name(stem.replace(".jsonl", ".manifest.json"))
            manifest = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        src_id = str(manifest.get("source") or "")
        config = str(manifest.get("config") or "")
        entry = next((e for e in FREE_DATASET_CATALOG
                      if e["id"].lower() == src_id.lower()), None)
        if entry is None:
            # file stem may carry a `-{config}` suffix — match by prefix
            entry = next((e for e in FREE_DATASET_CATALOG
                          if name.startswith(e["name"] + "-")), None)
        normalize = str((entry or {}).get("normalize") or "generic")
        src = next(
            (s for s in DEFAULT_COLAB_SOURCES
             if (s["id"].lower() == src_id.lower()
                 and (s.get("config") or "") == config)
             or s["name"] == name), None)
        specs.append({
            "name": name,
            "id": src_id or str((entry or {}).get("id") or ""),
            "config": config,
            "normalize": normalize,
            "weight": float((src or {}).get("weight") or 1.0),
            "cap": int((src or {}).get("cap") or 20000),
        })
    return specs


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "train_mine",
        description=(
            "Run the conversation→training agent: mine the chat history into "
            "quality-scored Q&A pairs and write Alpaca/ShareGPT/ChatML bundles "
            "into the training data dir (registered for nm train)."
        ),
        capability=Capability.DB_READ,
        parameters={
            "name": "str (optional) — bundle name",
            "min_score": "float 0..1 (optional, 0.3) — quality floor",
            "limit": "int (optional, 400) — max pairs considered",
            "since_hours": "float (optional) — only this many hours back",
        },
    )
    def train_mine(*, name: str = "", min_score: str = "", limit: str = "",
                   since_hours: str = "") -> dict[str, Any]:
        return mine(context, name=name, min_score=min_score, limit=limit,
                    since_hours=since_hours)

    @registry.register(
        "train_datasets",
        description=(
            "List registered training datasets + registry stats + recently "
            "mined conversation bundles."
        ),
        capability=Capability.DB_READ,
        parameters={"limit": "int (optional, 20)"},
    )
    def train_datasets(*, limit: str = "") -> dict[str, Any]:
        return datasets(context, limit=limit)

    @registry.register(
        "dataset_fetch",
        description=(
            "Download a free HuggingFace dataset (catalog name from "
            "train_datasets.free_catalog, or any public org/name) and "
            "normalize it into the training data dir as JSONL; registered "
            "for nm train. Mobile-friendly default of 500 rows."
        ),
        capability=Capability.NET_OUT,
        parameters={
            "ref": "str — catalog name (oasst1, hh-rlhf, swe-bench-verified, …) or org/name",
            "max_rows": "int (optional, 500) — rows to pull",
        },
    )
    def dataset_fetch(*, ref: str = "", max_rows: str = "") -> dict[str, Any]:
        return fetch(context, ref=ref, max_rows=max_rows)

    @registry.register(
        "train_mix",
        description=(
            "Build the PERSONA fine-tune mix: merge fetched datasets "
            "(broad topics + agent + style layers) into one "
            "fine-tune-ready bundle with your persona as the system "
            "turn of every row; writes messages/alpaca/sharegpt/chatml "
            "formats + manifest + a Colab-free QLoRA training script. "
            "sources = comma list of fetched dataset names."
        ),
        capability=Capability.DB_READ,
        parameters={
            "sources": "str — comma list, e.g. 'open-hermes-25,ultra-data-agent,dolphin-2.9'",
            "persona": "str (optional) — the persona text (else NM_HOME/persona.txt, else built-in)",
            "rows": "int (optional, 500000) — target mix size (no technical cap)",
            "colab": "bool (optional, true) — also write the Colab training script",
            "base_model": "str (optional) — HF id for the Colab script header",
        },
    )
    def train_mix(*, sources: str = "", persona: str = "", rows: str = "",
                  colab: str = "true", base_model: str = "") -> dict[str, Any]:
        return mix(context, sources=sources, persona=persona, rows=rows,
                   colab=colab, base_model=base_model)

    @registry.register(
        "train_checkpoints",
        description=(
            "Checkpoint resume status for a free-tier training run: which "
            "checkpoint-* dirs are complete+valid (a kill mid-save can "
            "leave the newest one corrupt), which one the next session "
            "resumes from, and how many steps remain. run_dir = the "
            "training OUT dir (the one holding checkpoint-<n>/ folders); "
            "max_steps = the run's MAX_STEPS (0 = unknown)."
        ),
        capability=Capability.DB_READ,
        parameters={
            "run_dir": "str — the run's output dir (holds checkpoint-<n>/)",
            "max_steps": "int (optional, 0) — the run's MAX_STEPS bound",
        },
    )
    def train_checkpoints(*, run_dir: str = "",
                          max_steps: str = "0") -> dict[str, Any]:
        if not (run_dir or "").strip():
            return {"ok": False,
                    "error": "train_checkpoints needs run_dir="}
        from ..training.checkpoints import report

        try:
            n = int(max_steps or 0)
        except ValueError:
            n = 0
        return report(run_dir.strip(), max_step=(n or None))
