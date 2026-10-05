"""RuntimeScheduleMixin: PartnerRuntime command group (schedule)."""

from __future__ import annotations

import json
import time
from typing import Any, Callable

class RuntimeScheduleMixin:
    """RuntimeScheduleMixin for :class:`PartnerRuntime`."""


    def _control_schedule(self, tail: str) -> str:
        from ...core.errors import AmbiguousRef
        scheduler = self._scheduler_or_build()
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "status"
        if verb in {"status", "list"}:
            jobs = scheduler.list_jobs()
            if not jobs:
                return "no scheduled jobs — /schedule add <name> <when> <action>"
            lines = ["scheduled jobs:"]
            for job in jobs[:15]:
                state = "on " if job["enabled"] else "off"
                nxt = job.get("next_run_iso") or ("past" if job["kind"] == "at" else "—")
                lines.append(f"  [{state}] {job['name']} — {job['kind']} {job['spec']} (next: {nxt})")
                extras = []
                if job.get("depends_on"):
                    extras.append(f"after:{job['depends_on'][:8]}")
                if job.get("max_retries"):
                    extras.append(f"retry×{job['max_retries']}")
                if job.get("retry_count"):
                    extras.append(f"retrying({job['retry_count']})")
                if extras:
                    lines[-1] += " [" + ", ".join(extras) + "]"
                if job.get("last_result"):
                    lines.append(f"        last: {job['last_result'][:80]}")
            return "\n".join(lines)
        if verb == "add":
            return self._schedule_add(parts[1:], scheduler)
        if verb in {"rm", "remove"}:
            ref = " ".join(parts[1:])
            if not ref:
                return "usage: /schedule rm <name or id>"
            try:
                return ("removed." if scheduler.remove(ref)
        else f"no job named {ref!r} — /schedule list to see the jobs")
            except AmbiguousRef as exc:
                return str(exc)
        if verb in {"enable", "disable"}:
            ref = " ".join(parts[1:])
            if not ref:
                return f"usage: /schedule {verb} <name or id>"
            try:
                job = scheduler.set_enabled(ref, enabled=(verb == "enable"))
            except (LookupError, AmbiguousRef) as exc:
                return str(exc)
            return f"{job['name']} {'enabled' if job['enabled'] else 'disabled'} (next {job.get('next_run_iso') or '—'})."
        if verb == "run":
            ref = " ".join(parts[1:])
            if not ref:
                return "usage: /schedule run <name or id>"
            try:
                outcome = scheduler.run_now(ref)
            except (LookupError, AmbiguousRef) as exc:
                return str(exc)
            return f"ran {outcome['name']}: {outcome['result'][:400]}"
        return ("usage: /schedule add <name> <when> <message|tool|command> <...> [options] | list | "
                "rm <name> | enable|disable <name> | run <name>\n"
                "  when: 'at 2026-12-25 09:00' | 'every 30m' | '22:00' | 'daily 22:00 America/New_York'\n"
                "        | 'cron 0 22 * * *' (minute hour day month weekday)\n"
                "  action: message goodnight | tool web_research {\"query\":\"ai news\"} | command python3 -V\n"
                "  options: tz <IANA> | after <job> | retry <n> [delay <secs>]")

    def _schedule_add(self, parts: list[str], scheduler: Any) -> str:
        """parts = the tail after 'add': [name, spec..., verb, payload...]"""
        if len(parts) < 3:
            return ("usage: /schedule add <name> <when> <message|tool|command> <...>\n"
                    "  e.g. /schedule add goodnight 22:00 message goodnight 🌙")
        name = parts[0]
        rest = parts[1:]
        payload_kind = ""
        split_at = -1
        for i, tok in enumerate(rest):
            if tok.lower() in {"message", "tool", "command"}:
                payload_kind = tok.lower()
                split_at = i
                break
        if split_at < 0:
            return "add needs an action: message <text> | tool <name> <json> | command <cmd>"
        spec = " ".join(rest[:split_at])
        payload_parts = rest[split_at + 1:]
        # ── trailing options: tz <IANA> | after <job> | retry <n> | delay <s>
        # stripped from the end of the payload so free-form message text
        # isn't polluted.
        timezone, depends_on = "", ""
        max_retries, retry_delay = 0, 60.0
        while len(payload_parts) >= 2:
            key = payload_parts[-2].lower()
            val = payload_parts[-1]
            if key == "tz":
                timezone = val
                payload_parts = payload_parts[:-2]
            elif key == "after":
                depends_on = val
                payload_parts = payload_parts[:-2]
            elif key == "retry":
                try:
                    max_retries = max(0, int(val))
                except ValueError:
                    break
                payload_parts = payload_parts[:-2]
            elif key == "delay":
                try:
                    retry_delay = max(10.0, float(val))
                except ValueError:
                    break
                payload_parts = payload_parts[:-2]
            else:
                break
        payload: dict[str, Any]
        if payload_kind == "message":
            payload = {"text": " ".join(payload_parts)}
        elif payload_kind == "tool":
            if not payload_parts:
                return "tool action needs: <tool name> <json args>"
            args: dict[str, Any] = {}
            if len(payload_parts) > 1:
                try:
                    parsed = json.loads(" ".join(payload_parts[1:]))
                    if isinstance(parsed, dict):
                        args = parsed
                except (ValueError, TypeError):
                    return "tool args must be a JSON object, e.g. {\"query\":\"ai news\"}"
            payload = {"tool": payload_parts[0], "args": args}
        else:
            if not payload_parts:
                return ("command action needs the command text — e.g. "
                        "/schedule add backup every 1h command \"python3 backup.py\"")
            payload = {"command": " ".join(payload_parts)}
        try:
            job = scheduler.add(name, spec, payload_kind, payload,
                                timezone=timezone, depends_on=depends_on,
                                max_retries=max_retries, retry_delay=retry_delay)
        except (ValueError, RuntimeError) as exc:
            return f"scheduling failed: {exc}"
        when = time.strftime("%m-%d %H:%M", time.localtime(job["next_run"]))
        extras = []
        if timezone:
            extras.append(f"tz={timezone}")
        if depends_on:
            extras.append(f"after={depends_on}")
        if max_retries:
            extras.append(f"retry×{max_retries}")
        extra_s = (" [" + ", ".join(extras) + "]") if extras else ""
        return f"⏰ scheduled {job['name']} — {job['kind']} ({spec}) — next {when}{extra_s}"

    # ── scheduler: /schedule ─────────────────────────────────────────────────
    def _scheduler_or_build(self) -> Any:
        scheduler = getattr(self, "_scheduler", None)
        if scheduler is None:
            from ..scheduler import Scheduler

            scheduler = Scheduler(self.context, gateway=self.gateway)
            self._scheduler = scheduler
        return scheduler

    # ── notifications ────────────────────────────────────────────────────────
    def _control_notify(self, arg: str) -> str:
        from ..notifier import Notifier

        notifier = Notifier(self.context, self.gateway)
        limit = int(arg) if arg.isdigit() else 10
        rows = notifier.recent(limit)
        if not rows:
            return "no notifications yet — arena builds, research, news and tasks land here."
        marks = {"sent": "✓", "failed": "✗", "pending": "…",
                 "held-quiet-hours": "⏸", "disabled": "⊘",
                 "muted": "⊘", "deduped": "⤺"}
        lines = [f"notifications ({len(rows)}):"]
        for row in rows:
            when = time.strftime("%m-%d %H:%M", time.localtime(row.get("created_at", 0)))
            state = row.get("delivery_state") or (
                "sent" if row.get("delivered") else "pending")
            mark = marks.get(state, "?")
            body = str(row.get("body", "")).replace("\n", " ")[:80]
            lines.append(f"  {when} [{row.get('kind')}] {mark} {state} "
                         f"{row.get('title', '')[:50]}")
            if body:
                lines.append(f"      {body}")
        lines.append("states: ✓sent ✗failed …pending ⏸held-quiet-hours "
                     "⊘disabled/muted ⤺deduped — see `nm briefing status`")
        return "\n".join(lines)

    def _control_proactive(self, arg: str) -> str:
        """Owner-only: show the proactive push-send switches and the
        delivery states of recent proactive sends (briefing + watcher
        alerts).  Proactive messages go to the owner's DMs only — this
        command just reports; toggles are env vars (see /help)."""
        from ..morning_briefing import proactive_status

        payload = proactive_status(self.context)
        s = payload["settings"]
        marks = {"sent": "✓", "failed": "✗", "pending": "…",
                 "held-quiet-hours": "⏸", "disabled": "⊘",
                 "muted": "⊘", "deduped": "⤺"}
        lines = [
            "she speaks first — proactive push sends (owner DMs only):",
            f"  master:   {'ON' if s['proactive_enabled'] else 'OFF'}"
            "  (NM_PARTNER_PROACTIVE_ENABLED=0 silences everything)",
            f"  briefing: {'ON' if s['proactive_briefing'] else 'OFF'}"
            f"  daily {s['briefing_time']} ({s['timezone']})",
            f"  watchers: {'ON' if s['proactive_watchers'] else 'OFF'}",
            f"  quiet hours: {s['quiet_hours']} — watcher alerts hold, "
            "the scheduled briefing still goes out",
        ]
        recent = payload["recent"]
        counts = payload.get("counts") or {}
        if counts:
            agg = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            lines.append(f"  last 24h: {agg}")
        health = payload.get("health") or {}
        for reason in health.get("degraded", []):
            lines.append(f"  ⚠️ degraded: {reason}")
        if not recent:
            lines.append("no proactive sends recorded yet")
        else:
            lines.append("recent sends:")
            for r in recent:
                when = time.strftime(
                    "%m-%d %H:%M", time.localtime(r.get("created_at", 0)))
                state = r.get("delivery_state", "?")
                lines.append(f"  {when} [{r.get('kind')}] "
                             f"{marks.get(state, '?')} {state} — "
                             f"{r.get('title', '')[:60]}")
        return "\n".join(lines)

    # ── directives (direct instructions to the core) ─────────────────────────
    def _control_task(self, tail: str, chat_key: str) -> str:
        from ..directives import DirectivesAgent
        from ..notifier import Notifier

        agent = DirectivesAgent(self.context, notifier=Notifier(self.context, self.gateway))
        parts = (tail or "").split()
        verb = parts[0].lower() if parts else "list"
        if verb == "add":
            text = " ".join(parts[1:]).strip()
            if not text:
                return "usage: /task add <instruction>"
            # wave 68 (F2): substantial task instructions pass through the
            # structuring sub-agent — the stored directive becomes the
            # god-tier spec (objective, steps, done-means), not the raw line.
            # wave 68 (F2): substantial task instructions pass through the
            # structuring sub-agent — the queued directive becomes the
            # god-tier spec (objective, steps, done-means), not the raw line.
            queued, brief_note = text, ""
            try:
                from ..brief import BriefAgent, should_brief
                if should_brief(text):
                    brief = BriefAgent(self.context).refine(text, kind="task")
                    if brief.by == "model" and brief.objective:
                        queued = brief.as_goal()
                        brief_note = f" (structured: {len(brief.steps)} steps)"
                    elif brief.by == "heuristic":
                        brief_note = " (structured heuristically)"
            except Exception:  # noqa: BLE001 - structuring must never block the queue
                pass
            result = agent.add(queued)
            if not result.get("ok"):
                return f"could not add: {result.get('error')}"
            return (f"queued {result['id'][:8]}: “{text[:70]}”{brief_note} — "
                    f"/task run to execute now")
        if verb == "run":
            did = parts[1] if len(parts) > 1 else ""
            chat = self._ref_from_key(chat_key)
            try:
                self.gateway.send(chat.platform, chat, "⏳ executing directive…")
            except Exception:  # noqa: BLE001
                pass
            result = agent.run(did)
            if not result.get("ok"):
                return f"task: {result.get('error')}"
            text = f"✅ {result['id'][:8]} done:\n{str(result.get('result', ''))[:3000]}"
            return self._send_long_checked(chat.platform, chat, text)
        if verb == "list":
            return agent.format_list(agent.list(15))
        return "usage: /task add <instruction> | /task run [id] | /task list"
