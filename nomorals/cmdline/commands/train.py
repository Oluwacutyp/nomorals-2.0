"""``nm train`` — fine-tune training surfaces."""

from __future__ import annotations

from ..emit import _emit



def _cmd_train(args, context):
    """Training control: honest backend listing, or one real pipeline run."""
    import dataclasses

    from ...training.backends import available_backends

    if getattr(args, "backends", False):
        rows = available_backends()
        configured = (context.settings.training.backend or "native").strip().lower()
        lines = [f"configured backend: {configured}"]
        for row in rows:
            mark = "ok   " if row["available"] else "miss "
            reason = "" if row["available"] else f" — {row['reason']}"
            lines.append(f"  {mark}{row['name']}{reason}")
        _emit(args, {"configured": configured, "backends": rows}, "\n".join(lines))
        return 0

    if getattr(args, "status", False):
        # The DM promise: `CognitiveLoop._tick_train` tells the owner
        # "`nm train status` any time". The status already exists —
        # SelfImprovementStatus computes policy + last run and already
        # feeds the brain-side `autonomy` tool — this is one branch over
        # that function, same _emit/--json pattern as --backends.
        from ...agents.cognition import SelfImprovementStatus

        status = SelfImprovementStatus(context)
        runs = status.get("runs", {}) or {}
        total = int(runs.get("total") or 0)
        lines = ["training due — retraining recommended"
                 if status.get("should_retrain") else "training not due"]
        reasons = status.get("reasons", []) or []
        if reasons:
            lines.append("  reasons: " + "; ".join(str(r) for r in reasons))
        if not total:
            lines.append("  no training run yet")
        else:
            lines.append(
                f"  {total} run{'s' if total != 1 else ''} — "
                f"{runs.get('promoted', 0)} promoted, "
                f"{runs.get('rejected', 0)} rejected, "
                f"{runs.get('gate_passed', 0)} passed the gate")
            # Per-run detail (the aggregates above are the summary; the
            # runs themselves carry the story). Omitted when there are none.
            recent = runs.get("recent", []) or []
            if recent:
                import time

                lines.append("  recent runs:")
                for run in recent:
                    name = str(run.get("name") or "unnamed")
                    status_text = str(run.get("status") or "pending")
                    gate = run.get("gate_passed")
                    gate_text = (" — gate passed" if gate is True
                                 else " — gate failed" if gate is False
                                 else "")
                    created = float(run.get("created_at") or 0.0)
                    when = (time.strftime("%Y-%m-%d %H:%M",
                                          time.localtime(created))
                            if created else "unknown time")
                    error = str(run.get("error") or "")
                    error_text = (f" — {error[:120]}" if error else "")
                    lines.append(f"    {when}  {name} — {status_text}"
                                 f"{gate_text}{error_text}")
        _emit(args, status, "\n".join(lines))
        return 0

    if getattr(args, "export_telegram", False):
        # Owner-history export (audit R3-71): the harvester itself
        # (nomorals/exporter.py) is complete and already reachable through
        # the brain tool registry, but the CLI had no surface — and the
        # module docstring's `nm train --run --dataset <id>` never existed
        # either (see --dataset below). One invocation does one job: the
        # export runs and returns; the text names the dataset id for the
        # follow-up `nm train --run --dataset <id>`.
        from ...exporter import export_telegram

        db = getattr(context, "db", None)
        register = not bool(getattr(args, "no_register", False))
        if register and db is None:
            _emit(args, {"ok": False, "error": "no database available"},
                  "telegram export failed — no database available to register the dataset")
            return 1
        try:
            result = export_telegram(
                context.settings,
                db=db,
                max_dialogs=int(getattr(args, "max_dialogs", 50) or 50),
                max_per_chat=int(getattr(args, "max_per_chat", 300) or 300),
                only_chats=getattr(args, "only_chats", "") or "",
                register=register,
            )
        except Exception as exc:  # noqa: BLE001 — honest CLI error, never a traceback
            _emit(args, {"ok": False, "error": f"{type(exc).__name__}: {exc}"},
                  f"telegram export failed — {type(exc).__name__}: {exc}")
            return 1
        stats = result.stats
        payload = result.to_dict()
        payload["ok"] = True

        def _n(count: int, word: str) -> str:
            return f"{count} {word}" + ("s" if count != 1 else "")

        # The skip is honest, never silent: when any dialog was refused
        # by the Tier-3 exclusion filter, the summary names the count.
        # Byte-identical to before when nothing was excluded.
        parts = [f"{stats.pii_scrubbed} PII scrubbed",
                 f"{stats.duplicates} duplicates"]
        if stats.excluded:
            parts.append(f"{_n(stats.excluded, 'dialog')} skipped as excluded")
        summary = (f"{_n(stats.dialogs, 'dialog')}, "
                   f"{_n(stats.messages, 'message')} -> "
                   f"{_n(stats.examples, 'example')} "
                   f"({', '.join(parts)})")
        if not stats.examples:
            _emit(args, payload,
                  f"telegram export: no examples harvested — {summary} "
                  "(a window needs both an owner reply and another side)")
            return 1
        if result.dataset_id:
            _emit(args, payload,
                  f"telegram export: {summary}; dataset {result.dataset_id}")
        else:
            _emit(args, payload,
                  f"telegram export: {summary}; file {result.path} (not registered)")
        return 0

    if getattr(args, "continual", False):
        from ...training.continual_driver import run_continual_update_once

        result = run_continual_update_once(
            context, force=bool(getattr(args, "force", False)))
        status = str(result.get("status") or "failed")
        if status == "accepted":
            before = result.get("before") or {}
            after = result.get("after") or {}
            text = (f"continual update accepted — fixed harness "
                    f"{before.get('accuracy')} → {after.get('accuracy')} "
                    f"({result.get('buffer')} replay pairs, "
                    f"adapter covers {result.get('pair_count')} pairs)")
        elif status == "rejected":
            text = (f"continual update rejected — "
                    f"{result.get('reason') or 'fixed-harness regression'} "
                    f"(previous adapter kept)")
        elif status == "not_due":
            text = "continual update not due yet — daily interval not elapsed"
        elif status == "deferred":
            text = f"continual update deferred — {result.get('reason') or ''}"
        else:
            text = (f"continual update failed — "
                    f"{result.get('error') or result.get('reason') or status}")
        _emit(args, result, text)
        return 0 if result.get("ok") else 1
    if getattr(args, "distill", False):
        from ...training.distill_driver import run_distill_once

        result = run_distill_once(
            context, force=bool(getattr(args, "force", False)))
        status = str(result.get("status") or "failed")
        if status == "trained" and result.get("promoted"):
            text = (f"distill trained and promoted — "
                    f"{result.get('pairs')} kept pairs, "
                    f"checkpoint {result.get('checkpoint')}")
        elif status == "trained":
            text = (f"distill trained but the gate refused it — "
                    f"candidate never lands "
                    f"({result.get('pairs')} kept pairs)")
        elif status == "deferred":
            text = f"distill deferred — {result.get('reason') or ''}"
        elif status == "not_due":
            text = "distill not due yet — daily interval not elapsed"
        else:
            text = (f"distill failed — "
                    f"{result.get('error') or result.get('reason') or status}")
        _emit(args, result, text)
        return 0 if result.get("ok") else 1
    if getattr(args, "quick", False):
        # The one-command demo: seed + collected pairs, 3 epochs, artifact
        # saved. Different job from --run (the full pipeline); --quick
        # wins when both are passed, so the two never run together.
        from ...training.quick import run_quick_training

        result = run_quick_training(context)
        _emit(args, result.to_dict(), result.render())
        return 0 if result.ok else 1

    if not getattr(args, "run", False):
        _emit(args, {"started": False},
              "nothing to do — pass --backends to see engines or --run to train")
        return 0

    # --dataset seeds the run with a specific registered corpus (e.g. an
    # account-history export) instead of collecting live-use rows. The
    # module docstring promised `nm train --run --dataset <id>`; the flag
    # never existed and _cmd_train never passed the id through, so an
    # exported dataset could only reach training via the newest-dataset
    # fallback in _step_collect. Validate up front: a bad id is refused
    # before any mission is created.
    dataset_id = getattr(args, "dataset", "") or ""
    if dataset_id:
        from ...core.errors import NotFound
        from ...training.dataset import DatasetRegistry

        db = getattr(context, "db", None)
        if db is None:
            _emit(args, {"ok": False, "error": "no database available"},
                  "training failed — no database available to look up the dataset")
            return 1
        try:
            DatasetRegistry(db).get(dataset_id)
        except NotFound:
            _emit(args, {"ok": False, "error": f"no dataset {dataset_id!r}"},
                  f"training failed — no dataset {dataset_id!r} "
                  "(export one with `nm train export-telegram`)")
            return 1

    from ...self_improvement import SelfImprovementJob

    job = SelfImprovementJob(context)
    result = job.run(
        force=True,
        dataset_id=dataset_id,
        backend=getattr(args, "backend", "") or "",
        base_model=getattr(args, "base_model", "") or "",
    )
    payload = dataclasses.asdict(result)
    if result.status in ("skipped",):
        _emit(args, payload, f"training skipped — {result.reason or 'policy'}")
        return 0
    if result.status in ("done", "completed", "succeeded"):
        _emit(args, payload,
              f"training {result.status} — run {result.run_id}"
              + (" PROMOTED" if result.promoted else " (gate did not promote)"))
        return 0
    _emit(args, payload, f"training {result.status} — {result.error or result.reason}")
    return 1
