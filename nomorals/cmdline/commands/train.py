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

    if not getattr(args, "run", False):
        _emit(args, {"started": False},
              "nothing to do — pass --backends to see engines or --run to train")
        return 0

    from ...self_improvement import SelfImprovementJob

    job = SelfImprovementJob(context)
    result = job.run(
        force=True,
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
