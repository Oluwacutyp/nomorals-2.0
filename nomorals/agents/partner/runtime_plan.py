"""RuntimePlanMixin — /plan chat control.

``/plan <task>`` surveys the real code first, then plans the task
and runs it: the Implementation Plan is saved durably, shown in
chat, and its delegation runs are spawned at once.

No review-before-run gate (owner order 2026-11-16, same as
/workflow): nothing blocks on approval. The saved plan IS the
review surface — it is shown in chat and stays on disk under
``<home>/plans/<id>.md``.

Owner-only: absent from PUBLIC_COMMANDS, like /workflow.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

from .failures import record_failure

__all__ = ["RuntimePlanMixin"]


class RuntimePlanMixin:
    """Chat control for saved Implementation Plans."""

    # ── store ────────────────────────────────────────────────────

    def _plan_store(self):  # type: ignore[no-untyped-def]
        """The durable plan store, cached per runtime."""
        cached = getattr(self, "_plan_store_cache", None)
        if cached is None:
            from ...agents.plan.store import (
                PlanStore,
                default_plans_dir,
                default_plans_path,
            )

            settings = getattr(self, "settings", None)
            home = getattr(settings, "home_path", None)
            if home is None:
                home = getattr(settings, "home", None)
            home = Path(home) if home is not None \
                else Path.home() / ".nomorals"
            cached = PlanStore(
                path=default_plans_path(home),
                plans_dir=default_plans_dir(home))
            self._plan_store_cache = cached
        return cached

    def _plan_manager(self):  # type: ignore[no-untyped-def]
        """The delegation manager, or None when the runtime has none."""
        resolver = getattr(self, "_delegation_manager", None)
        if not callable(resolver):
            return None
        try:
            return resolver()
        except Exception:  # noqa: BLE001 — no manager means no runs
            return None

    # ── verbs ────────────────────────────────────────────────────

    def _control_plan(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """``/plan`` — plan+run a task, or list/show/status saved plans.

        A bare task surveys the code, plans, saves, shows, and runs.
        ``list`` / ``show`` / ``status`` manage saved plans. ``cancel``
        stops a running plan (and its live runs) on the owner's word.
        A two-word tail whose first word is a verb is the verb; a longer
        tail starting with a verb word is a task (``show me the
        files`` plans, it does not look for a plan called ``me``).
        """
        from ...agents.plan import PlanError

        store = self._plan_store()
        if store.degraded_reason:
            return ("Plan store degraded: " + store.degraded_reason
                    + " — saved plans are unreadable right now.")
        text = " ".join(str(tail or "").split()).strip()
        parts = text.split()
        verb = parts[0].lower() if parts else "list"
        chat = getattr(message, "chat", None) if message is not None \
            else None
        chat_key = str(getattr(chat, "key", "") or "")

        try:
            if verb in ("show", "status") and len(parts) == 1:
                return ("usage: /plan " + verb + " <id> — "
                        "/plan list shows them.")
            if verb == "list" and len(parts) == 1:
                rows = store.list_plans()
                if not rows:
                    return "no plans yet — /plan <task> writes one."
                lines = ["plans:"]
                for row in rows:
                    lines.append(
                        f"• {row['id']} [{row['state']}] — "
                        f"{str(row.get('task', ''))[:60]}")
                return "\n".join(lines)
            if verb == "status" and len(parts) <= 2:
                if len(parts) == 1:
                    rows = store.list_plans()
                    if not rows:
                        return "no plans yet — /plan <task> writes one."
                    lines = ["plan status:"]
                    for row in rows:
                        line = (f"• {row['id']} [{row['state']}]")
                        if row.get("plan_error"):
                            line += f" — {row['plan_error'][:100]}"
                        lines.append(line)
                    return "\n".join(lines)
                row, error = self._resolve_plan(store, parts[1])
                if row is None:
                    return error
                return self._plan_status_text(store, row)
            if verb == "show" and len(parts) == 2:
                row, error = self._resolve_plan(store, parts[1])
                if row is None:
                    return error
                return self._plan_show_text(store, row)
            if verb == "cancel" and len(parts) == 1:
                return ("usage: /plan cancel <id> — "
                        "/plan list shows them.")
            if verb == "cancel" and len(parts) == 2:
                row, error = self._resolve_plan(store, parts[1])
                if row is None:
                    return error
                return self._plan_cancel_text(
                    store, row, manager=self._plan_manager())
            # Everything else is a task: survey, plan, save, show, run.
            return self._plan_run(store, text, chat_key)
        except PlanError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - control never raises
            record_failure("plan.control", exc)
            return f"the plan command failed: {exc}"

    def _resolve_plan(self, store, ref):  # type: ignore[no-untyped-def]
        row = store.resolve(ref)
        if row is None:
            return None, (f"no plan '{ref}' — "
                          "/plan list shows them.")
        return row, ""

    def _plan_status_text(self, store, row):  # type: ignore[no-untyped-def]
        """One plan's state, refreshed from its live runs."""
        manager = self._plan_manager()
        if manager is not None:
            with contextlib.suppress(Exception):
                row = store.refresh_state(row["id"], manager.get_run)
        lines = [f"plan {row['id']} [{row['state']}]",
                 f"task: {row.get('task', '')}"]
        if row.get("plan_error"):
            lines.append(f"note: {row['plan_error']}")
        for run in row.get("runs") or []:
            if isinstance(run, dict):
                lines.append(f"• {run.get('id')} ({run.get('role')}) — "
                             f"{run.get('status', 'unknown')}")
        for entry in row.get("unassigned") or []:
            if isinstance(entry, dict):
                lines.append(f"• unassigned {entry.get('step')}: "
                             f"{entry.get('reason', 'no roster worker')}")
        return "\n".join(lines)

    def _plan_show_text(self, store, row):  # type: ignore[no-untyped-def]
        """The full Implementation Plan artifact for one saved plan."""
        if store.plans_dir is not None:
            artifact = store.plans_dir / f"{row['id']}.md"
            try:
                return artifact.read_text(encoding="utf-8")
            except OSError:
                pass
        from ...agents.plan.render import render_plan_md

        return render_plan_md(row)

    def _plan_cancel_text(self, store, row, manager=None):  # type: ignore[no-untyped-def]
        """Cancel one saved plan and say what stopped."""
        outcome = store.cancel(row["id"], by="owner", manager=manager)
        record = outcome["record"]
        if outcome.get("already_cancelled"):
            when = record.get("cancelled_at")
            stamp = ""
            if when:
                stamp = (" (cancelled "
                         + time.strftime("%Y-%m-%d %H:%M",
                                         time.localtime(when)) + ")")
            return (f"plan {record['id']} was already cancelled{stamp} "
                    "— the plan stays saved.")
        lines = [f"plan {record['id']} cancelled — "
                 f"'{record.get('task', '')}' is stopped."]
        stopped = outcome.get("runs_cancelled") or []
        if stopped:
            lines.append(
                f"stopped {len(stopped)} live run"
                + ("s" if len(stopped) != 1 else "")
                + ": " + ", ".join(stopped))
        finished = outcome.get("runs_terminal") or []
        if finished:
            lines.append(
                "already finished, left alone: "
                + ", ".join(finished))
        untouched = outcome.get("runs_untouched") or []
        if untouched:
            lines.append(
                "no delegation manager to stop them through — "
                "these runs keep going: " + ", ".join(untouched))
        lines.append("the plan stays saved — /plan list shows it.")
        return "\n".join(lines)

    def _plan_run(self, store, task, chat_key):  # type: ignore[no-untyped-def]
        """Survey, plan, spawn, save, and show one task."""
        from ...agents.plan import plan_and_run

        result = plan_and_run(
            getattr(self, "context", None), task, chat_key=chat_key,
            store=store, manager=self._plan_manager())
        return result["display"]
