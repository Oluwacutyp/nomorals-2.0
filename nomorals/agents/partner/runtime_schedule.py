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
        if verb == "health":
            return self._schedule_health(scheduler)
        if verb == "runs":
            ref = " ".join(parts[1:])
            if not ref:
                return "usage: /schedule runs <name or id>"
            try:
                runs = scheduler.recent_runs(ref, limit=10)
            except AmbiguousRef as exc:
                return str(exc)
            if not runs:
                return f"no runs recorded for {ref!r}"
            lines = [f"recent runs for {ref}:"]
            for r in runs:
                mark = "✅" if r["ok"] else "❌"
                ts = time.strftime("%m-%d %H:%M",
                                   time.localtime(r["started_at"]))
                lines.append(f"  {mark} {ts} {r['seconds']}s "
                             f"[{r['trigger_source']}] {r['result'][:90]}")
            return "\n".join(lines)
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
                deps = job.get("depends_on") or []
                if deps:
                    deps = deps if isinstance(deps, list) else [deps]
                    extras.append("after:" + ",".join(d[:8] for d in deps))
                    if str(job.get("depends_policy") or "all_ok") != "all_ok":
                        extras.append(f"gate:{job['depends_policy']}")
                if job.get("max_retries"):
                    extras.append(f"retry×{job['max_retries']}")
                if job.get("retry_count"):
                    extras.append(f"retrying({job['retry_count']})")
                if job.get("heavy"):
                    extras.append("heavy")
                if str(job.get("missed_fire_policy") or "fire_now") != "fire_now":
                    extras.append(f"missed:{job['missed_fire_policy']}")
                if str(job.get("overlap_policy") or "concurrent") != "concurrent":
                    extras.append(f"overlap:{job['overlap_policy']}")
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
                "health | runs <name> | rm <name> | enable|disable <name> | run <name>\n"
                "  when: 'at 2026-12-25 09:00' | 'every 30m' | '22:00' | 'daily 22:00 America/New_York'\n"
                "        | 'cron 0 22 * * *' (minute hour day month weekday)\n"
                "  action: message goodnight | tool web_research {\"query\":\"ai news\"} | command python3 -V\n"
                "  options: tz <IANA> | after <job> | retry <n> [delay <secs>]")

    def _schedule_health(self, scheduler: Any) -> str:
        """Owner-visible scheduler health: ``/schedule health``.

        Answers "is my scheduler alive": tick loop state, last tick,
        next fire, last job outcome, and the delivery queue (retryable /
        dead-lettered) with the live owner channels.
        """
        try:
            h = scheduler.health()
        except Exception as exc:  # noqa: BLE001 - health must never raise
            return f"scheduler health check failed: {exc}"
        lines = ["scheduler health: "
                 + ("OK" if h.get("ok") else "DEGRADED")]
        loop_state = h.get("loop_state") or (
            "running" if h.get("running") else "not-started")
        lines.append(f"  tick loop: {loop_state}"
                     f" (every {h.get('tick_seconds', '?')}s)")
        started = h.get("started_at")
        if started:
            lines.append("  started: "
                         + time.strftime("%m-%d %H:%M", time.localtime(started)))
        age = h.get("last_tick_age_s")
        lines.append("  last tick: "
                     + (f"{age}s ago" if age is not None else "never"))
        if h.get("tick_errors"):
            lines.append(f"  tick errors: {h['tick_errors']}")
        jobs = h.get("jobs") or {}
        lines.append(f"  jobs: {jobs.get('total', 0)} total, "
                     f"{jobs.get('enabled', 0)} enabled, "
                     f"{jobs.get('due_now', 0)} due now")
        nxt = h.get("next_job")
        if nxt:
            lines.append(f"  next: {nxt['name']} at {nxt.get('next_run_iso')} "
                         f"(in {nxt.get('in_s', '?')}s)")
        last = h.get("last_job")
        if last:
            mark = "✅" if last.get("ok") else "❌"
            lines.append(f"  last ran: {mark} {last['name']} — "
                         f"{last.get('last_result', '')[:80]}")
        d = h.get("delivery") or {}
        q = d.get("queue") or {}
        lines.append(f"  delivery queue: {q.get('retryable', 0)} retrying, "
                     f"{q.get('held', 0)} held (quiet hours), "
                     f"{q.get('dead', 0)} dead-lettered")
        live = d.get("live_owner_channels") or []
        lines.append("  live owner channels: " + (", ".join(live) or "NONE"))
        if d.get("termux_fallback"):
            lines.append("  fallback: termux-notification available "
                         "(Android shade when no chat is live)")
        for reason in h.get("reasons") or []:
            lines.append(f"  ⚠️ {reason}")
        for note in h.get("notes") or []:
            lines.append(f"  ℹ️ {note}")
        return "\n".join(lines)

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
        # ── trailing options: tz <IANA> | after <job> | depends <id1,id2> |
        #   dependspolicy <all|any|latest> | retry <n> | delay <s> |
        #   backoff <constant|linear|exponential> | missed <fire|skip|next> |
        #   overlap <skip|queue|concurrent> | timeout <s> | heavy
        # stripped from the end of the payload so free-form message text
        # isn't polluted.
        timezone, depends_on = "", ""
        depends_policy = "all_ok"
        max_retries, retry_delay = 0, 60.0
        backoff = "exponential"
        missed_fire_policy = "fire_now"
        overlap_policy = "concurrent"
        run_timeout_s = 0.0
        heavy = False
        while payload_parts:
            if payload_parts[-1].lower() == "heavy":
                heavy = True
                payload_parts = payload_parts[:-1]
                continue
            if len(payload_parts) < 2:
                break
            key = payload_parts[-2].lower()
            val = payload_parts[-1]
            if key == "tz":
                timezone = val
                payload_parts = payload_parts[:-2]
            elif key == "after":
                depends_on = val
                payload_parts = payload_parts[:-2]
            elif key == "depends":
                depends_on = [d.strip() for d in val.split(",") if d.strip()]
                payload_parts = payload_parts[:-2]
            elif key == "dependspolicy":
                if val.lower() not in {"all", "any", "latest"}:
                    break
                depends_policy = {"all": "all_ok", "any": "any_ok",
                                  "latest": "latest_ok"}[val.lower()]
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
            elif key == "backoff":
                if val.lower() not in {"constant", "linear", "exponential"}:
                    break
                backoff = val.lower()
                payload_parts = payload_parts[:-2]
            elif key == "missed":
                mapping = {"fire": "fire_now", "skip": "skip",
                           "next": "next_only"}
                if val.lower() not in mapping:
                    break
                missed_fire_policy = mapping[val.lower()]
                payload_parts = payload_parts[:-2]
            elif key == "overlap":
                if val.lower() not in {"skip", "queue", "concurrent"}:
                    break
                overlap_policy = val.lower()
                payload_parts = payload_parts[:-2]
            elif key == "timeout":
                try:
                    run_timeout_s = max(0.0, float(val))
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
                                depends_policy=depends_policy,
                                max_retries=max_retries, retry_delay=retry_delay,
                                backoff=backoff,
                                missed_fire_policy=missed_fire_policy,
                                overlap_policy=overlap_policy,
                                run_timeout_s=run_timeout_s, heavy=heavy)
        except (ValueError, RuntimeError) as exc:
            return f"scheduling failed: {exc}"
        when = time.strftime("%m-%d %H:%M", time.localtime(job["next_run"]))
        extras = []
        if timezone:
            extras.append(f"tz={timezone}")
        if depends_on:
            deps = depends_on if isinstance(depends_on, list) else [depends_on]
            extras.append("after=" + ",".join(deps))
        if max_retries:
            extras.append(f"retry×{max_retries} ({backoff})")
        if missed_fire_policy != "fire_now":
            extras.append(f"missed={missed_fire_policy}")
        if overlap_policy != "concurrent":
            extras.append(f"overlap={overlap_policy}")
        if run_timeout_s:
            extras.append(f"timeout={run_timeout_s:g}s")
        if heavy:
            extras.append("heavy")
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
                 "muted": "⊘", "deduped": "⤺", "dead": "✖"}
        attempt_marks = {"sent": "✓", "failed": "✗", "skipped": "…"}
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
            # Honest delivery detail: per-channel attempts (sent via
            # which channel, or WHY each target didn't land) instead of
            # a bare ✗ failed.  Kept compact — at most 3 attempt lines.
            if state in ("failed", "dead", "sent"):
                try:
                    attempts = notifier.delivery_attempts(
                        str(row.get("id") or ""))[:3]
                except Exception:  # noqa: BLE001 - detail is best-effort
                    attempts = []
                for a in attempts:
                    amark = attempt_marks.get(str(a.get("state") or ""), "?")
                    where = (f"{a.get('platform')}:{a.get('chat_id')}"
                             if a.get("platform") else
                             f"entry {a.get('source')!r}")
                    detail = ""
                    if a.get("state") == "sent" and a.get("message_id"):
                        detail = f" (msg {a.get('message_id')})"
                    elif a.get("error"):
                        detail = f" — {str(a.get('error'))[:90]}"
                    lines.append(f"      ↳ {amark} {where}"
                                 f"{detail}")
        lines.append("states: ✓sent ✗failed …pending ⏸held-quiet-hours "
                     "⊘disabled/muted ⤺deduped ✖dead — see `nm briefing status`")
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
                 "muted": "⊘", "deduped": "⤺", "dead": "✖"}
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
