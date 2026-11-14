"""One general command that installs every missing model.

``/models`` — bare installs everything missing, ``status`` shows what
is installed or missing per model, ``learners`` shows the
journal-trained learners' status, a single backend name installs just
that one. Never raises.
"""

from __future__ import annotations


def control_models(tail: str = "") -> str:
    """Chat entry: /models …  Never raises."""
    try:
        from ..tools import models_provision as _mp

        args = (tail or "").strip().lower().split()
        if not args:
            args = ["all"]
        if args[0] == "status":
            return _mp.status_text()
        if args[0] == "learners":
            lines = ["📦 models — learners (status only, they train "
                     "from journals):"]
            for row in _mp.learner_status():
                lines.append(f"• {row['job']}: {row['status']} — "
                             f"{row['detail']}")
            lines.append("")
            lines.append(_mp.OUT_OF_SCOPE_NOTE)
            return "\n".join(lines)
        wanted = (list(_mp.BACKENDS) if args[0] == "all"
                  else [a for a in args if a in _mp.BACKENDS])
        if not wanted:
            return ("📦 which model? /models "
                    "[all|status|learners|<name>] — installs every "
                    "missing model on this box.\n"
                    f"unknown: {args[0]} — names are: "
                    + ", ".join(_mp.BACKENDS))
        result = _mp.provision_models(wanted)
        return _mp.models_report_text(result)
    except Exception as exc:  # noqa: BLE001 - the command never dies
        return f"📦 /models failed: {exc}"
