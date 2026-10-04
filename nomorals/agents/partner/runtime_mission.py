"""RuntimeMissionMixin: PartnerRuntime command group (mission)."""

from __future__ import annotations

import threading
from typing import Any, Callable
from ...core.logging_setup import get_logger
_log = get_logger(__name__)


class RuntimeMissionMixin:
    """RuntimeMissionMixin for :class:`PartnerRuntime`."""


    # ── missions: chat-visible progress ──────────────────────────────────────
    def _control_mission(self, tail: str, chat_key: str = "") -> str:
        """Mission progress and control, from persisted state.

        /mission status [id|name] — % complete, current step, ETA, stall reason
        /mission list             — active missions at a glance
        /mission stall <id> <code> <message> — record a concrete blocker
        /mission clear <id>       — drop the stall record (progress resumed)
        /mission pause <id>       — freeze it (status → paused)
        /mission resume <id>      — continue it in the background
        /mission cancel <id> [reason] — stop it for good (terminal)
        /mission retry <id>       — fresh attempt: cancel + requeue the goal
        /mission watch <id>       — this chat gets milestone updates
        /mission unwatch <id>     — stop milestone updates in this chat
        /mission new <template> <args> — create from a template
            (research <topic> | build <what> | fix <target>)
        """
        from ...core.errors import AmbiguousRef, NoMoralsError
        from ...missions import (
            MissionStatus,
            MissionStore,
            MissionWatchers,
            StallCode,
            render_status_text,
            wired_runner,
        )

        store = MissionStore(self.context.db)

        def _resolve(ref: str) -> tuple[Any, str]:
            """_find_mission with ambiguity surfaced as a chat-ready reply.

            Returns ``(mission, "")`` on success, ``(None, "")`` when
            nothing matches, and ``(None, message)`` when the reference is
            ambiguous — the message lists the candidates instead of the
            resolver guessing one.
            """
            try:
                return self._find_mission(store, ref), ""
            except AmbiguousRef as exc:
                return None, str(exc)

        usage = ("usage: /mission status [id|name] | /mission list | "
                 "/mission stall <id> <code> <message> | /mission clear <id> | "
                 "/mission pause <id> | /mission resume <id> | "
                 "/mission cancel <id> [reason] | /mission retry <id> | "
                 "/mission watch <id> | /mission unwatch <id> | "
                 "/mission new <template> <args>\n"
                 f"stall codes: {', '.join(sorted(StallCode.ALL))}; "
                 "templates: research <topic> | build <what> | fix <target>")
        parts = (tail or "").strip().split(None, 1)
        verb = (parts[0] if parts else "status").lower()
        rest = parts[1] if len(parts) > 1 else ""

        if verb == "list":
            rows = store.resumable()
            if not rows:
                return ("no active missions — start one with "
                        "/mission new <research|build|fix> …")
            lines = [f"missions ({len(rows)} active):"]
            for m in rows[:15]:
                # reconcile on read: a dead runner must not show as running
                store.reconcile(m.id)
                m = store.get(m.id)
                p = store.progress(m.id)
                stall = m.state.get("stall")
                line = (f"  · {m.name} [{m.status}] — "
                        f"{p['steps_done']}/{p['total_steps']} steps "
                        f"({p['percent']:.0f}%)")
                if stall:
                    code = stall.get("code")
                    label = StallCode.LABELS.get(code, code)
                    line += f" — ⚠️ stalled [{code}]: {label}"
                lines.append(line)
            return "\n".join(lines)

        if verb == "status":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return (f"no mission matching {rest!r} — "
                        "/mission list to see the active ones.")
            # reconcile on read: a dead runner must not report "running"
            rec = store.reconcile(mission.id)
            text = render_status_text(store.detail(mission.id))
            if rec["changed"]:
                text = (f"⚠️ reconciled: stale 'running' → {rec['status']}: "
                        f"{rec['reason']}\n{text}")
            return text

        if verb == "stall":
            sub = rest.split(None, 2)
            if len(sub) < 3:
                return ("usage: /mission stall <id|name> <code> <message>\n"
                        f"codes: {', '.join(sorted(StallCode.ALL))}")
            ref, code, message = sub
            mission, _amb = _resolve(ref)
            if _amb:
                return _amb
            if mission is None:
                return (f"no mission matching {ref!r} — "
                        "/mission list to see the active ones.")
            if code not in StallCode.ALL:
                return (f"unknown stall code {code!r} — one of: "
                        f"{', '.join(sorted(StallCode.ALL))}")
            try:
                out = wired_runner(self.context, store=store).mark_stalled(
                    mission.id, code, message)
            except (NoMoralsError, ValueError) as exc:
                return f"couldn't mark stall: {exc}"
            stall = out["stall"] or {}
            code = stall.get("code")
            label = StallCode.LABELS.get(code, code)
            hint = StallCode.UNBLOCK_HINTS.get(code, "")
            reply = (f"⚠️ {mission.name} marked stalled [{code}]: "
                     f"{label} — {stall.get('message')}")
            if hint:
                reply += f"\nunblocks: {hint}"
            return reply

        if verb == "clear":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return (f"no mission matching {rest!r} — "
                        "/mission list to see the active ones.")
            cleared = wired_runner(self.context, store=store).clear_stalled(mission.id)
            return (f"{mission.name}: stall cleared — back in play."
                    if cleared else f"{mission.name}: no stall recorded.")

        if verb == "pause":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return (f"no mission matching {rest!r} — "
                        "/mission list to see the active ones.")
            # idempotent: double-pause is a no-op success, and a finished
            # mission is already past pausing — never an error, never a
            # state rewrite.
            if mission.status == MissionStatus.PAUSED:
                return (f"⏸ {mission.name}: already paused — no change "
                        f"(/mission resume {mission.id} to continue).")
            if mission.terminal:
                return (f"{mission.name} is already {mission.status} — "
                        "nothing to pause.")
            try:
                store.set_status(mission.id, MissionStatus.PAUSED)
            except (NoMoralsError, ValueError) as exc:
                return f"couldn't pause: {exc}"
            return (f"⏸ {mission.name}: paused — "
                    f"/mission resume {mission.id} to continue.")

        if verb == "resume":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return (f"no mission matching {rest!r} — "
                        "/mission list to see the active ones.")
            # a "running" mission whose runner died is not running —
            # reconcile first so resume restarts it instead of no-op'ing
            rec = store.reconcile(mission.id)
            if rec["changed"]:
                mission = store.get(mission.id)
            if mission.terminal:
                return (f"{mission.name} is {mission.status} — terminal "
                        "missions can't resume; /mission retry starts a "
                        "fresh attempt.")
            # idempotent: resume of a live mission is a no-op success
            if mission.status == MissionStatus.RUNNING:
                return (f"▶ {mission.name}: already running — no change; "
                        f"/mission status {mission.id} for progress.")
            runner = wired_runner(self.context, store=store)

            def _resume_job() -> None:
                try:
                    runner.resume(mission.id)
                except Exception:  # noqa: BLE001 - chat must stay alive
                    _log.exception("mission resume %s failed", mission.id)
                finally:
                    # One-shot thread: don't leak its DB connection.
                    self._release_db_thread()

            threading.Thread(target=_resume_job,
                             name=f"mission-resume-{mission.id[:8]}",
                             daemon=True).start()
            return (f"▶ {mission.name}: resumed in the background — "
                    f"/mission status {mission.id} for progress.")

        if verb == "cancel":
            sub = rest.split(None, 1)
            ref = sub[0] if sub else ""
            reason = sub[1].strip() if len(sub) > 1 else "cancelled by owner"
            mission, _amb = _resolve(ref)
            if _amb:
                return _amb
            if mission is None:
                return (f"no mission matching {ref!r} — "
                        "/mission list to see the active ones.")
            # idempotent: a finished mission stays finished — cancelling
            # must never rewrite done/failed into cancelled, and a second
            # cancel is a no-op success, not an error.
            if mission.status == MissionStatus.CANCELLED:
                return f"⏹ {mission.name}: already cancelled — no change."
            if mission.terminal:
                return (f"⏹ {mission.name}: already {mission.status} — "
                        "nothing to cancel.")
            try:
                mission = store.set_status(mission.id, MissionStatus.CANCELLED,
                                           note=reason)
            except (NoMoralsError, ValueError) as exc:
                return f"couldn't cancel: {exc}"
            runner = wired_runner(self.context, store=store)
            runner.cancel(reason)  # cooperative: any in-flight runner stops
            if runner.reporter is not None:
                try:
                    # NOTE: mission is the fresh post-set_status row — passing
                    # the pre-cancel copy would let the milestone _mark save
                    # the stale status back over "cancelled".
                    runner.reporter.on_terminal(
                        mission, MissionStatus.CANCELLED, error=reason)
                except Exception:  # noqa: BLE001 - telemetry, not chat
                    _log.debug("cancel milestone push failed", exc_info=True)
            return f"⏹ {mission.name}: cancelled ({reason})."

        if verb == "retry":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return (f"no mission matching {rest!r} — "
                        "/mission list to see the active ones.")
            if mission.status == MissionStatus.RUNNING:
                return (f"{mission.name} is still running — /mission cancel "
                        "it first if you want to start over.")
            # cancel + requeue: a fresh mission row keeps the goal, the plan
            # skeleton, the name and the budgets; spend, errors, stalls and
            # the milestone log start clean. The original row is kept
            # as-is when it is already terminal — its done/failed record
            # is history, not something retry may rewrite.
            fresh_state: dict[str, Any] = {}
            if mission.state.get("plan"):
                fresh_state["plan"] = mission.state["plan"]
            new = store.create_new(
                mission.goal,
                name=mission.name,
                budget_wall=mission.budget_wall,
                budget_tokens=mission.budget_tokens,
                state=fresh_state,
                metadata={**(mission.metadata or {}), "retry_of": mission.id},
            )
            if mission.terminal:
                old_note = (f"the {mission.status} original is kept as-is; "
                            "this is a brand-new attempt")
            else:
                try:
                    store.set_status(mission.id, MissionStatus.CANCELLED,
                                     note=f"superseded by retry {new.id}")
                except (NoMoralsError, ValueError) as exc:
                    _log.debug("retry: could not cancel old mission: %s", exc)
                old_note = "the previous attempt was cancelled"
            runner = wired_runner(self.context, store=store)

            def _retry_job() -> None:
                try:
                    runner.run(new, max_iterations=8)
                except Exception:  # noqa: BLE001 - chat must stay alive
                    _log.exception("mission retry %s failed", new.id)
                finally:
                    # One-shot thread: don't leak its DB connection.
                    self._release_db_thread()

            threading.Thread(target=_retry_job,
                             name=f"mission-retry-{new.id[:8]}",
                             daemon=True).start()
            return (f"🔁 {mission.name} is {mission.status}: retry starts a "
                    f"NEW attempt [{new.id}] linked via metadata.retry_of "
                    f"({old_note}) — running in the background.")

        if verb == "watch":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return (f"no mission matching {rest!r} — "
                        "/mission list to see the active ones.")
            if not chat_key:
                return "watch needs a chat context — run this from a chat."
            # reconcile on read: report the true state, not stale "running"
            store.reconcile(mission.id)
            mission = store.get(mission.id)
            added = MissionWatchers(self.context.db).subscribe(
                mission.id, chat_key)
            state_note = f" (currently {mission.status})"
            if added:
                return (f"👀 watching {mission.name}: milestone updates "
                        f"will land in this chat.{state_note}")
            return f"already watching {mission.name} from this chat.{state_note}"

        if verb == "unwatch":
            mission, _amb = _resolve(rest)
            if _amb:
                return _amb
            if mission is None:
                return (f"no mission matching {rest!r} — "
                        "/mission list to see the active ones.")
            if not chat_key:
                return "unwatch needs a chat context — run this from a chat."
            removed = MissionWatchers(self.context.db).unsubscribe(
                mission.id, chat_key)
            return (f"stopped watching {mission.name} from this chat."
                    if removed
                    else f"{mission.name} isn't watched from this chat.")

        if verb == "new":
            sub = rest.split(None, 1)
            template = sub[0] if sub else ""
            targs = sub[1].strip() if len(sub) > 1 else ""
            spec = self._mission_template_spec(template, targs)
            if spec is None:
                return ("usage: /mission new <template> <args>\n"
                        "templates: research <topic> | build <what> | "
                        "fix <target>")
            mission = store.create_new(**spec)
            return (f"🚀 mission created: {mission.name} [{mission.id}] — "
                    f"{mission.goal}\n"
                    f"/mission resume {mission.id} to start it.")

        return usage

    @staticmethod
    def _mission_template_spec(template: str, args: str) -> dict[str, Any] | None:
        """Goal + plan skeleton for ``/mission new <template> <args>``.

        Returns the kwargs for ``MissionStore.create_new``, or None when
        the template is unknown.
        """
        t = (template or "").lower()
        plans = {
            "research": ["gather sources", "synthesize findings", "write report"],
            "build": ["design", "implement", "verify with tests"],
            "fix": ["reproduce", "root-cause", "patch", "verify"],
        }
        if t not in plans or not args.strip():
            return None
        arg = args.strip()
        goals = {
            "research": (f"Research {arg}: gather sources, synthesize "
                         "findings, report with citations"),
            "build": f"Build {arg}: design, implement, verify with tests",
            "fix": (f"Fix {arg}: reproduce the issue, find the root cause, "
                    "patch it, verify"),
        }
        return {
            "goal": goals[t],
            "name": f"{t} — {arg[:48]}",
            "state": {
                "plan": [
                    {"name": s, "goal": s, "role": "execution",
                     "kind": "io", "depends_on": []}
                    for s in plans[t]
                ]
            },
            "metadata": {"template": t, "template_args": arg[:200]},
        }

    @staticmethod
    def _find_mission(store: Any, ref: str) -> Any | None:
        """Resolve an id, id prefix, or name/goal substring to a mission.

        Empty ref → the default active mission (the documented ``[id|name]``
        UX for ``/mission status``), or None when there is none — the
        default never goes through prefix matching, so it can never
        match-all.
        Exact id wins, even when it is also a prefix of another mission's
        id. A prefix/name matching 2+ missions raises
        :class:`~nomorals.core.errors.AmbiguousRef` — the resolver never
        guesses; the caller renders the candidates and asks the user.
        """
        from ...core.errors import AmbiguousRef, NotFound
        from ...core.ids import min_unique_prefix_len, resolve_id_prefix

        filt = (ref or "").strip()
        if not filt:
            active = store.resumable()
            if active:
                return active[0]
            rows = store.list(limit=1)
            return rows[0] if rows else None
        try:
            return store.get(filt)
        except NotFound:
            _log.debug("mission lookup: no exact id match for %r, "
                       "trying prefix/name", ref)
        rows = store.list(limit=100)
        by_id: dict[str, Any] = {}
        for m in rows:
            by_id.setdefault(m.id, m)
        res = resolve_id_prefix(filt, by_id)
        if res.outcome in ("exact", "unique"):
            # the id channel is authoritative; name/goal search is only a
            # fallback when the id channel finds nothing at all
            return by_id[res.matches[0]]
        low = filt.lower()
        name_hits = [m for m in rows
                     if low in (m.name or "").lower()
                     or low in (m.goal or "").lower()]
        if res.outcome == "none":
            if len(name_hits) == 1:
                return name_hits[0]
            if not name_hits:
                return None
            ordered = [m.id for m in name_hits]
        else:  # ambiguous id prefix — never guess; union with name hits
            ordered = list(res.matches)
            for m in name_hits:
                if m.id not in ordered:
                    ordered.append(m.id)
        if len(ordered) == 1:
            return by_id[ordered[0]]
        if not ordered:
            return None
        raise AmbiguousRef(
            ref=filt,
            entity="mission",
            candidates=[(mid, f"[{by_id[mid].status}] {by_id[mid].name}")
                        for mid in ordered],
            min_prefix_len=min_unique_prefix_len(ordered),
            hint="/mission list shows the active ones.",
        )

    # ── image tools ──────────────────────────────────────────────────────────
    @staticmethod
    def _tool_data(outcome: Any) -> Any:
        """registry.call wraps the tool's dict in Ok(...); the tools ALSO
        report failure inside their dict ({'ok': False, 'error': ...}), so a
        lookup needs both layers checked. Returns (data, error_text)."""
        if not outcome.ok:
            return None, (outcome.error.message if outcome.error else "unknown")
        data = outcome.value
        if isinstance(data, dict) and data.get("ok") is False:
            return None, str(data.get("error") or "unknown")
        return data, None

    @staticmethod
    def _fake_message(chat_key: str) -> Any:
        """A minimal message shell for console-style /mind dispatch."""
        from dataclasses import dataclass, field

        @dataclass
        class _Msg:
            text: str = ""
            sender: str = "console"
            chat: Any = None

        @dataclass
        class _Chat:
            key: str
            platform: str = "local"
            kind: str = "dm"

        return _Msg(chat=_Chat(key=chat_key))
