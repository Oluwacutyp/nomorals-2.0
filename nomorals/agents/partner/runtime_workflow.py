"""RuntimeWorkflowMixin — /workflow chat control.

``/workflow <task>`` drafts a workflow script from the task (the
brain + the dynamic-workflows skill), saves it durably, shows it
in chat, and queues it — the unified-scheduler tick runs it in the
background and delivers the summary into the origin chat.

No review-before-run gate (owner order 2026-11-16: "Not necessary
I review"): nothing blocks on approval. The saved script IS the
review surface — it is shown in chat and stays on disk, re-runnable
with ``/workflow run <name>``.

Owner-only: absent from PUBLIC_COMMANDS, like /poster.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .failures import record_failure

_SKILL_RELATIVE = Path("nomorals") / "skills" / "library" \
    / "dynamic-workflows.md"

_DRAFT_SYSTEM = (
    "You write Devon workflow scripts. Output the script ONLY — "
    "no prose around it, no explanation. A fenced ```python block "
    "is fine. Follow the format in the skill text exactly: "
    "synchronous Python, phase() / log() / agent() / parallel() / "
    "pipeline() / trajectory() / args only. No imports, no open / "
    "exec / eval, no dunder access, no while loops."
)

_DRAFT_FORMAT_FALLBACK = (
    "Workflow scripts are synchronous Python with these primitives: "
    "phase(name), log(*args), agent(prompt, schema=None, role=..., "
    "input=None, workspace='inherit'), parallel(items, fn), "
    "pipeline(items, fn), trajectory(limit=10, kind=None), args. "
    "Roles: planner, implementer, reviewer, tester, refactorer, "
    "dep_hunter, api_designer. implementer needs input={'path': ..., "
    "'instruction': ...}; refactorer needs input={'path': ..., "
    "'goal': ...}. No imports, no open/exec/eval, no dunders, no "
    "while loops."
)


class RuntimeWorkflowMixin:
    """Chat control for saved, reviewable workflow scripts."""

    # ── store ────────────────────────────────────────────────────

    def _workflow_store(self):  # type: ignore[no-untyped-def]
        """The durable workflow store, cached per runtime."""
        cached = getattr(self, "_workflow_store_cache", None)
        if cached is None:
            from ...agents.workflow.store import (
                WorkflowStore,
                default_workflows_dir,
                default_workflows_path,
            )

            settings = getattr(self, "settings", None)
            home = getattr(settings, "home_path", None)
            if home is None:
                home = getattr(settings, "home", None)
            home = Path(home) if home is not None \
                else Path.home() / ".nomorals"
            cached = WorkflowStore(
                path=default_workflows_path(home),
                scripts_dir=default_workflows_dir(home))
            self._workflow_store_cache = cached
        return cached

    # ── drafting ─────────────────────────────────────────────────

    def _workflow_llm(self, prompt: str) -> str:
        """One synchronous brain call, returning the draft text.

        Raises RuntimeError when the brain fails — the caller turns
        that into an honest reply, never a silent queue.
        """
        from ...llm.base import Message, SamplingParams
        from ...llm.brain import brain_for

        response = brain_for(self.context).chat(
            [Message.system(_DRAFT_SYSTEM), Message.user(prompt)],
            SamplingParams(temperature=0.2, max_tokens=1500),
            task_kind="chat")
        if not response.ok or not response.text:
            raise RuntimeError(response.error or "empty LLM response")
        return response.text

    def _workflow_skill_text(self) -> str:
        """The dynamic-workflows skill text, or the short fallback."""
        try:
            skill = Path(__file__).resolve().parents[3] / _SKILL_RELATIVE
            return skill.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001 - the fallback is the format
            return _DRAFT_FORMAT_FALLBACK

    @staticmethod
    def _extract_script(text: str) -> str:
        """Pull a script out of a draft (fenced block, else raw)."""
        raw = str(text or "")
        if "```python" in raw:
            body = raw.split("```python", 1)[1]
            return body.split("```", 1)[0].strip() + "\n"
        if "```" in raw:
            body = raw.split("```", 1)[1]
            return body.split("```", 1)[0].strip() + "\n"
        return raw.strip() + "\n" if raw.strip() else ""

    def _draft_workflow_script(self, task: str) -> tuple[str, str]:
        """Draft a script for the task.

        Returns (script, reason): script is "" when the draft never
        came out valid, and reason names why. Two attempts — the
        second carries the validation errors of the first.
        """
        from ...agents.workflow.script import validate_script

        skill = self._workflow_skill_text()
        prompt = (
            f"Write ONE synchronous Python workflow script for "
            f"this task:\n{task}\n\nFollow the workflow format "
            f"below exactly. Keep it short (under 40 lines), 2-4 "
            f"agent calls at most, concrete prompts, one phase per "
            f"stage.\n\n{skill}")
        errors: list[str] = []
        for attempt in range(2):
            if attempt:
                prompt = (
                    f"Your previous draft was rejected for these "
                    f"reasons: {'; '.join(errors)}. Fix ONLY those, "
                    f"keep everything else.\n\n{prompt}")
            draft = self._extract_script(self._workflow_llm(prompt))
            errors = validate_script(draft)
            if not errors:
                return draft, ""
        return "", ("the draft never came out valid: "
                    + "; ".join(errors))

    @staticmethod
    def _workflow_name_for(task: str, taken: set[str]) -> str:
        """A slug from the task, de-duplicated against taken names."""
        slug = re.sub(r"[^a-z0-9]+", "-", str(task or "").lower())
        slug = slug.strip("-")[:40] or "workflow"
        name = slug
        suffix = 2
        while name in taken:
            name = f"{slug}-{suffix}"
            suffix += 1
        return name

    # ── resolution ───────────────────────────────────────────────

    @staticmethod
    def _resolve_workflow(store, ref: str):  # type: ignore[no-untyped-def]
        """Resolve a workflow by exact name, unique name prefix, or id.

        Returns (record, error): error names the miss or the
        ambiguity, "" when it resolved.
        """
        ref = (ref or "").strip()
        if not ref:
            return None, "name a workflow — /workflow lists them."
        rows = store.list_workflows()
        for row in rows:
            if row.get("name") == ref or row.get("id") == ref:
                return row, ""
        prefix = [row for row in rows
                  if str(row.get("name", "")).startswith(ref)
                  or str(row.get("id", "")).startswith(ref)]
        if len(prefix) == 1:
            return prefix[0], ""
        if len(prefix) > 1:
            names = ", ".join(sorted(r["name"] for r in prefix))
            return None, (f"{ref!r} matches several workflows "
                          f"({names}) — name one exactly.")
        return None, f"no workflow {ref!r} — /workflow lists them."

    # ── control ──────────────────────────────────────────────────

    def _control_workflow(self, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """``/workflow`` — draft + queue, or list/show/status/run/cancel.

        A bare task drafts a script, saves it, shows it, and queues
        it; ``list`` / ``show`` / ``status`` / ``run`` / ``cancel`` /
        ``resume`` manage saved scripts. A two-word tail whose first
        word is a verb is the verb; a longer tail starting with a
        verb word is a task (``show me the files`` drafts, it does
        not look for a workflow called ``me``).
        """
        from ...agents.workflow.errors import WorkflowError

        store = self._workflow_store()
        text = " ".join(str(tail or "").split()).strip()
        parts = text.split()
        verb = parts[0].lower() if parts else "list"
        chat = getattr(message, "chat", None) if message is not None \
            else None
        chat_key = str(getattr(chat, "key", "") or "")

        try:
            if verb in ("show", "run", "cancel", "resume") \
                    and len(parts) == 1:
                return ("usage: /workflow " + verb + " <name> — "
                        "/workflow lists them.")
            if verb == "list" and len(parts) == 1:
                rows = store.list_workflows()
                if not rows:
                    return ("no workflows yet — /workflow <task> "
                            "writes one.")
                lines = ["workflows:"]
                for row in rows:
                    lines.append(
                        f"• {row['name']} [{row['state']}] — "
                        f"{str(row.get('task', ''))[:60]}")
                return "\n".join(lines)
            if verb == "status" and len(parts) <= 2:
                if len(parts) == 1:
                    rows = store.list_workflows()
                    if not rows:
                        return ("no workflows yet — /workflow <task> "
                                "writes one.")
                    lines = ["workflow status:"]
                    for row in rows:
                        line = (f"• {row['name']} [{row['state']}]")
                        if row.get("error"):
                            line += f" — {row['error'][:100]}"
                        lines.append(line)
                    return "\n".join(lines)
                row, error = self._resolve_workflow(store, parts[1])
                if row is None:
                    return error
                return self._workflow_status_text(store, row)
            if verb in ("show", "run", "cancel", "resume") \
                    and len(parts) == 2:
                row, error = self._resolve_workflow(store, parts[1])
                if row is None:
                    return error
                if verb == "show":
                    return self._workflow_show_text(store, row)
                if verb == "run":
                    try:
                        store.requeue(row["id"], by="owner")
                    except WorkflowError as exc:
                        return str(exc)
                    return (f"{row['name']} queued again — running "
                            f"in the background; I'll send the "
                            f"summary here when it's done.")
                if verb == "cancel":
                    try:
                        store.cancel(row["id"], by="owner")
                    except WorkflowError as exc:
                        return str(exc)
                    return (f"{row['name']} cancelled — the script "
                            f"stays saved; /workflow run {row['name']} "
                            f"starts it again.")
                try:
                    store.resume(row["id"], by="owner")
                except WorkflowError as exc:
                    return str(exc)
                return (f"{row['name']} queued again — running in "
                        f"the background; I'll send the summary here "
                        f"when it's done.")
            # Everything else is a task: draft, save, show, queue.
            return self._workflow_draft_and_queue(store, text, chat_key)
        except WorkflowError as exc:
            return str(exc)
        except Exception as exc:  # noqa: BLE001 - control never raises
            record_failure("workflow.control", exc)
            return f"the workflow command failed: {exc}"

    def _workflow_draft_and_queue(self, store, task: str,
                                  chat_key: str) -> str:
        """Draft a script for the task, save it, show it, queue it."""
        if not task:
            return ("usage: /workflow <task>  ·  /workflow list  ·  "
                    "/workflow show <name>  ·  /workflow status "
                    "[<name>]  ·  /workflow run <name>  ·  /workflow "
                    "cancel <name>")
        try:
            script, reason = self._draft_workflow_script(task)
        except Exception as exc:  # noqa: BLE001 - the brain failing is a reply
            record_failure("workflow.draft", exc)
            return (f"couldn't draft the workflow: {exc} — the "
                    f"task is still just words; try again and it "
                    f"drafts fresh.")
        if not script:
            return (f"couldn't draft a valid workflow script — "
                    f"{reason}")
        from ...agents.workflow.errors import WorkflowError

        taken = {row["name"] for row in store.list_workflows()}
        name = self._workflow_name_for(task, taken)
        try:
            row = store.save_workflow(name=name, task=task,
                                      script=script, chat_key=chat_key)
        except WorkflowError as exc:
            return str(exc)
        lines = [f"saved as {name} — queued, running in the "
                 f"background; I'll send the summary here when "
                 f"it's done.",
                 "",
                 "```python",
                 script.rstrip(),
                 "```"]
        if row.get("script_path"):
            lines.append(f"script: {row['script_path']}")
        return "\n".join(lines)

    @staticmethod
    def _workflow_show_text(store, row: dict[str, Any]) -> str:
        """A saved workflow: state + the script itself."""
        record = store.get_workflow(row["id"])
        script = record.get("script") or ""
        lines = [f"{record['name']} [{record['state']}]",
                 f"task: {record.get('task', '')}"]
        if record.get("error"):
            lines.append(f"error: {record['error']}")
        if record.get("summary"):
            lines.append(f"summary: {record['summary']}")
        lines += ["", "```python", script.rstrip(), "```"]
        return "\n".join(lines)

    @staticmethod
    def _workflow_status_text(store, row: dict[str, Any]) -> str:
        """One workflow's state, error, and phase journal tail."""
        record = store.get_workflow(row["id"])
        lines = [f"{record['name']} [{record['state']}]",
                 f"task: {record.get('task', '')}"]
        if record.get("error"):
            lines.append(f"error: {record['error']}")
        if record.get("summary"):
            lines.append(f"summary: {record['summary']}")
        phases = record.get("phases") or []
        if phases:
            lines.append("phases:")
            for phase in phases[-3:]:
                done = "done" if phase.get("finished_at") else "open"
                lines.append(f"• {phase.get('name', '')} ({done})")
                for line in (phase.get("log") or [])[-3:]:
                    lines.append(f"  {line}")
        return "\n".join(lines)
