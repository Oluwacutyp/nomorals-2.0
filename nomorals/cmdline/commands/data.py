"""``nm data`` — fine-tune data catalog and persona mix."""

from __future__ import annotations

import argparse
import json
from typing import Any



def _cmd_data(args: argparse.Namespace, context: Any) -> int:
    """Fine-tune data: what's free, what to base on, and the persona mix."""
    from pathlib import Path as _P

    from ...training import finetune as ft
    from ...training.free_datasets import catalog as data_catalog

    action = getattr(args, "action", "") or "catalog"
    data_dir = _P(context.settings.resolve(context.settings.training.data_dir))

    if action == "catalog":
        entries = data_catalog()
        print("free fine-tune datasets "
              "([v] = format verified against the live repo, [?] = unverified):")
        for e in entries:
            mark = "[v]" if e.get("verified") else "[?]"
            print(f"  {mark} {e.get('name', '?'):26s} {e.get('kind', ''):12s} "
                  f"{e.get('license', ''):14s} {e.get('size', ''):24s} "
                  f"{e.get('id', '')}")
        return 0

    if action == "models":
        print("Colab-safe base models (free-tier VRAM, permissive licenses):")
        for m in ft.COLAB_BASE_MODELS:
            why = str(m.get('why', '')).replace(chr(10), ' ')
            print(f"  {m['id']}")
            print(f"      {m.get('license', '')} - {why[:160]}")
        return 0

    if action == "mix":
        names = [s.strip() for s in (getattr(args, "sources", "") or "").split(",")
                 if s.strip()]
        if not names:
            print("usage: nm data mix <dataset,dataset,...> [--rows N] "
                  "[--persona-file PATH] [--base MODEL] [--out BASE] [--json]")
            print(f"hint: `nm data catalog` lists fetchable datasets; mix reads "
                  f"fetched files from {data_dir} as <name>.jsonl")
            return 2
        found: list[str] = []
        missing: list[str] = []
        for n in names:
            f = data_dir / f"{n}.jsonl"
            (found if f.is_file() else missing).append(str(f) if f.is_file() else n)
        if not found:
            print(f"no fetched files for: {', '.join(missing)} - the datasets must "
                  f"be fetched into {data_dir} first")
            return 1
        if missing:
            print(f"warning: no fetched files for {', '.join(missing)} - "
                  f"mixing without them")
        persona = getattr(args, "persona", "") or ""
        if not persona:
            persona = ft.load_persona(getattr(args, "persona_file", "") or "")
        out_base = _P(args.out) if getattr(args, "out", "") else data_dir / "persona-mix"
        sources = [ft.MixSource(name=_P(fpath).stem, path=fpath) for fpath in found]
        # A fetched source carries <name>.manifest.json (rows, HF id,
        # config): recipe-complete.  For those, a tiny --rows is treated as
        # "at least the Colab floor" — 100 rows is the smallest sample that
        # moves a QLoRA — clamped down to what the sources actually hold.
        # Hand-made files with no manifest take the number literally.
        requested = int(getattr(args, "rows", 0) or 0)
        metas: list[dict] = []
        for fpath in found:
            mf = _P(fpath).with_suffix(".manifest.json")
            try:
                metas.append(json.loads(mf.read_text(encoding="utf-8"))
                             if mf.is_file() else {})
            except Exception:  # noqa: BLE001 — a corrupt manifest = no manifest
                metas.append({})
        recipe = bool(metas) and all(m and m.get("rows") for m in metas)
        kwargs: dict = {}
        notebook_target = 0
        if recipe:
            available = sum(int(m.get("rows") or 0) for m in metas)
            floor = min(ft.NOTEBOOK_MIN_TARGET_ROWS, available)
            eff = max(requested or floor, floor)
            kwargs["target_rows"] = eff
            notebook_target = max(requested, ft.NOTEBOOK_MIN_TARGET_ROWS)
        elif requested:
            kwargs["target_rows"] = requested
            notebook_target = requested
        manifest = ft.build_persona_mix(sources, persona, out_base, **kwargs)
        colab = ft.write_colab_script(
            out_base,
            base_model=getattr(args, 'base', '') or ft.DEFAULT_COLAB_BASE,
            train_file=str(manifest["outputs"]["messages_jsonl"]))
        manifest["colab_script"] = colab
        nb_sources = [
            {"name": _P(fpath).stem,
             "id": str((m or {}).get("source") or ""),
             "config": str((m or {}).get("config") or ""),
             "rows": int((m or {}).get("rows") or 0)}
            for fpath, m in zip(found, metas)
        ]
        notebook = ft.write_colab_notebook(
            out_base,
            base_model=getattr(args, "base", "") or ft.DEFAULT_COLAB_BASE,
            persona=persona,
            sources=nb_sources,
            target_rows=notebook_target or ft.DEFAULT_TARGET_ROWS)
        manifest["colab_notebook"] = notebook
        per_src = ", ".join(f"{k}={v['rows']}" for k, v in manifest["per_source"].items())
        payload = manifest
        if getattr(args, 'json', False):
            print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
            return 0
        print(f"persona mix ready — {manifest['rows']} rows (target "
              f"{manifest['target_rows']}) from {len(sources)} source(s)")
        print(f"  {out_base.with_suffix('.jsonl')}")
        print(f"  filtered: {manifest['filtered']}  dupes dropped: "
              f"{manifest['deduped']}  per source: {per_src}")
        print(f"  colab script: {colab}")
        print(f"  colab nb:   {notebook}")
        return 0

    print(f"unknown data action: {action}")
    return 2
