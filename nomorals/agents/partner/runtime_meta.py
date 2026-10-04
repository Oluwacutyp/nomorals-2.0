"""RuntimeMetaMixin: PartnerRuntime command group (meta)."""

from __future__ import annotations

import json
import time
from ...core.logging_setup import get_logger
from .approvals import _key_set
_log = get_logger(__name__)


class RuntimeMetaMixin:
    """RuntimeMetaMixin for :class:`PartnerRuntime`."""


    def _model_status_line(self) -> str:
        """Which model is ACTUALLY answering, and whether it's failing over.

        This is the line that should have caught the silent mock fallback:
        a failing provider must be visible in /status, not buried in a warn
        log line that scrolls away.

        Honest by construction: a provider is only called "answering" if it
        has actually succeeded (``last_success``).  The old line showed the
        first fallback *by position* — "hf_serverless is answering for now"
        while every message went unanswered because that fallback was
        failing too (no token).  That lie is what made a dead chain look
        alive in /status.
        """
        router = getattr(self.context, "router", None)
        snapshot = getattr(router, "stats_snapshot", None)
        if snapshot is None:
            return "n/a"
        try:
            snap = snapshot()
        except Exception:  # noqa: BLE001 - status must never crash a chat
            return "n/a"
        active = str(snap.get("active") or "?")
        chain = [str(p) for p in (snap.get("chain") or [])]
        health = (snap.get("health") or {})
        active_health = health.get(active) or {}
        if not int(active_health.get("failures") or 0):
            return f"{active} (chain: {', '.join(chain)})" if chain else active
        last_error = str(active_health.get("last_error") or "unknown error")
        reason = self._plain_model_reason(last_error)
        # Who has ACTUALLY answered recently?  By last_success, not by
        # position in the chain.
        succeeded = [
            (float((health.get(p) or {}).get("last_success") or 0.0), p)
            for p in chain
            if p != active
        ]
        succeeded = [(ts, p) for ts, p in succeeded if ts > 0]
        if succeeded:
            top = max(succeeded)[1]
            return (f"{top} is answering for now — {active} is down ({reason}); "
                    f"{active} gets another try on the next message")
        # Nothing in the chain has ever answered.  Say so, with the backup's
        # own failure reason — that is the difference between "down and
        # recovering" and "silently dead".
        backups = [p for p in chain if p != active]
        if not backups:
            return f"no model is answering — {active} is down ({reason}) and " \
                   "there is no backup configured"
        caps = {str(n): set(v) for n, v in (snap.get("chain_caps") or {}).items()}
        backup_reasons = []
        for p in backups:
            known_caps = caps.get(p)
            if known_caps is not None \
                    and "chat" not in known_caps and "complete" not in known_caps:
                # this backup can never answer a chat message — saying
                # "has not been tried yet" for it is a soft lie
                # (unknown capability lists are treated as chat-capable)
                continue
            b_health = health.get(p) or {}
            if int(b_health.get("failures") or 0):
                backup_reasons.append(f"{p} failed too ({self._plain_model_reason(str(b_health.get('last_error') or 'unknown error'))})")
            else:
                backup_reasons.append(f"{p} has not been tried yet")
        if backup_reasons:
            return (f"no model is answering — {active} is down ({reason}); "
                    + "; ".join(backup_reasons)
                    + ". Messages get no reply until one of them works")
        # every backup is a non-chat provider (e.g. ocr) — only the local
        # model can answer
        return (f"no model is answering — {active} is down ({reason}) and the "
                "other providers in the chain cannot chat. Only fixing "
                f"{active} restores replies")

    @staticmethod
    def _plain_model_reason(error: str) -> str:
        """A raw exception string in chat reads like the system is broken.
        Translate the common ones into plain words."""
        low = error.lower()
        if "still loading" in low or "loading" in low:
            return "still loading the model — normal on a phone, takes a few minutes"
        if "400" in low or "not served by this hf" in low or "not in the inference" in low:
            return "the model name is not hosted there — it tries to swap to a hosted model"
        if "timed out" in low or "timeout" in low:
            return "took too long — if it's your phone's model, close other apps"
        if "connection refused" in low or "errno 111" in low:
            return "not running — start it with `nm models --start-local`"
        if "not responding" in low:
            return "frozen — kill and restart it with `nm models --start-local`"
        if "rate limit" in low or "429" in low:
            return "rate limited — it recovers on its own"
        if "401" in low or "unauthorized" in low or "invalid" in low:
            return "bad credentials"
        return error[:60]

    def _control_status(self) -> str:
        mood = self.brain.mood.current()
        rel = self.brain.relationship
        aut = self._autonomy.status() if self._autonomy else {"mode": "off"}
        from ..power import power_mode_for

        power_state = "active" if power_mode_for(self.context).active else "locked"
        platforms = ", ".join(
            name for name, info in self.gateway.status().items()
            if name != "_stats" and info.get("running_in_session")
        ) or "none"
        stats = self._stats_snapshot()
        return "\n".join(
            [
                f"mood: {mood.label} (affection {mood.values.get('affection', 0):.0f}, "
                f"energy {mood.values.get('energy', 0):.0f}, frustration {mood.values.get('frustration', 0):.0f})",
                f"relationship: {rel.stage}, trust {rel.trust:.0f}, "
                f"{len(rel.milestones)} milestones, {len(rel.fights)} fights logged",
                f"autonomy: {aut.get('mode', 'off')}, {aut.get('pending_proposals', 0)} pending",
                f"model: {self._model_status_line()}",
                f"platforms running: {platforms}",
                f"power mode: {power_state}",
                f"stats: {stats['messages']} msgs, {stats['replies']} replies, "
                f"{stats['errors']} errors, {stats.get('controls', 0)} commands",
            ]
        )

    def _control_mood(self, arg: str) -> str:
        engine = self.brain.mood
        try:
            if not arg:
                state = engine.current()
                dims = ", ".join(f"{k} {int(v)}" for k, v in sorted(state.values.items()))
                return f"mood: {state.label} — {dims}"
            if arg.strip().lower() == "reset":
                engine.reset()
                return "mood reset to baselines"
            if "=" in arg:
                pairs = dict(
                    (p.split("=", 1)[0].strip().lower(), float(p.split("=", 1)[1]))
                    for p in arg.split() if "=" in p
                )
                engine.set_dimensions(pairs)
                state = engine.current()
                return f"mood set: {state.label} ({', '.join(f'{k} {int(v)}' for k, v in pairs.items())})"
            engine.set_label(arg)
            return f"mood forced: {engine.current().label}"
        except ValueError as exc:
            return str(exc)

    def _control_mode(self, arg: str) -> str:
        mode = arg.strip().lower()
        if mode not in {"off", "suggest", "auto"}:
            return "usage: /mode off|suggest|auto"
        self.settings.partner.autonomy_mode = mode
        try:
            with self.context.db.transaction():
                self.context.db.execute(
                    "INSERT INTO kv_store (key, value, kind, updated_at) VALUES (?, ?, 'json', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                    ("partner.autonomy_mode", json.dumps({"mode": mode}), time.time()),
                )
        except Exception:  # noqa: BLE001 - a write failure must not block the switch
            pass
        try:
            from ..autonomy import AutonomyAgent

            if self._autonomy is None and mode != "off":
                dm_cap, group_cap = self._tuned_autonomy_caps()
                self._autonomy = AutonomyAgent(
                    self.context, self.brain, self.gateway,
                    mode=mode,
                    owner_chats=self._owner_chats,
                    group_chats=_key_set(self.settings.partner.group_chats),
                    quiet_start=self.settings.partner.quiet_start,
                    quiet_end=self.settings.partner.quiet_end,
                    max_dm_per_day=dm_cap,
                    max_group_per_day=group_cap,
                )
                self._autonomy.start()
            elif self._autonomy is not None:
                self._autonomy.set_mode(mode)
        except Exception as exc:  # noqa: BLE001
            _log.warning("autonomy mode switch failed: %s", exc)
            return f"mode set to {mode} (agent switch failed: {exc})"
        return f"autonomy: {mode}"

    def _control_model(self, arg: str) -> str:
        """Live model switch — no restart.

        ``/model`` reports what is answering; ``/model <provider> [fallback …]``
        rebuilds the router in place. The choice is persisted to the kv_store
        (the same store ``nm models --set-provider`` writes), so it also
        survives a restart.
        """
        parts = [p for p in (arg or "").split() if p]
        if not parts:
            return f"model: {self._model_status_line()}"
        provider = parts[0].lower()
        chain = [p.lower() for p in parts[1:]]

        from ...llm.providers import build_provider

        try:
            build_provider(provider, model="probe")  # validates the name
        except Exception as exc:  # noqa: BLE001
            return (f"can't use {provider!r}: {exc} — /model alone shows "
                    f"what's answering and the current fallbacks")

        from ..context import _build_router, persist_provider_override

        db = self.context.db
        try:
            persist_provider_override(db, provider, chain or None)
        except Exception as exc:  # noqa: BLE001
            return f"could not persist the switch: {exc}"
        try:
            new_router = _build_router(self.settings, self.context.bus, db=db, context=self.context)
        except Exception as exc:  # noqa: BLE001
            return f"could not build the {provider} router: {exc}"
        self.brain.responder.router = new_router
        self.context.router = new_router
        try:
            from ...llm.power import invalidate as _invalidate_power
            _invalidate_power(self.context)
        except Exception:  # noqa: BLE001 - power cache is best-effort
            pass
        _log.info("model switched live: provider=%s chain=%s", provider, chain)
        return f"model switched — now: {self._model_status_line()}"

    def _control_providers(self) -> str:
        """Probe every registered provider and report live/dead with reasons.

        ``/providers`` — the diagnostic for a silent brain.  Runs each
        provider's liveness probe in parallel (bounded), then joins it with
        the router's recorded health (calls, failures, last error, cooldown)
        and translates the last error into an actionable hint: bad key,
        dead model id, rate limit, or network.
        """
        from concurrent.futures import ThreadPoolExecutor
        from ...core.errors import (
            is_auth_error,
            is_not_found_error,
            is_rate_limited_error,
        )

        router = getattr(self.context, "router", None)
        if router is None:
            return "providers: no router on this runtime."
        names = list(router.providers())
        if not names:
            return "providers: chain is empty — nothing can answer. /model to configure."

        def _probe(name: str) -> tuple[str, bool, float]:
            provider = router.get(name)
            if provider is None:
                return name, False, 0.0
            started = time.perf_counter()
            try:
                ok = bool(provider.health())
            except Exception:  # noqa: BLE001 - a probe must never raise
                ok = False
            return name, ok, (time.perf_counter() - started)

        # Parallel: sequential probes would cost sum(health timeouts).
        live: dict[str, tuple[bool, float]] = {}
        with ThreadPoolExecutor(max_workers=max(1, len(names)),
                                thread_name_prefix="providers-probe") as pool:
            for name, ok, elapsed in pool.map(_probe, names):
                live[name] = (ok, elapsed)

        try:
            snap = router.stats_snapshot()
        except Exception:  # noqa: BLE001 - status must never crash a chat
            snap = {}
        health = snap.get("health") or {}
        active = str(snap.get("active") or "")

        def _hint(last_error: str) -> str:
            if not last_error:
                return ""
            probe_exc: Exception = Exception(last_error)
            # Rebuild enough signal for the classifiers: the recorded error
            # already embeds the HTTP status ("401 unauthorized for …").
            if is_auth_error(probe_exc):
                return "→ auth failed: check the key/token for this provider"
            if is_not_found_error(probe_exc):
                return "→ model id not hosted: fix the model name in config"
            if is_rate_limited_error(probe_exc):
                return "→ rate limited: wait for the quota window"
            lowered = last_error.lower()
            if "timed out" in lowered or "timeout" in lowered:
                return "→ network timeout: check connectivity"
            if "cool" in lowered:
                return ""
            return ""

        lines = [f"providers (active: {active or 'none'}):"]
        for name in names:
            provider = router.get(name)
            model_id = ""
            if provider is not None:
                try:
                    model_id = str(provider.model_id or "")
                except Exception:  # noqa: BLE001
                    model_id = ""
            h = health.get(name) or {}
            ok, elapsed = live.get(name, (False, 0.0))
            calls = int(h.get("calls") or 0)
            failures = int(h.get("failures") or 0)
            cooling = bool(h.get("cooling_down"))
            last_error = str(h.get("last_error") or "")
            marker = "★" if name == active else " "
            if cooling:
                state = "🧊 cooling down"
            elif ok:
                state = f"✅ live ({elapsed:.1f}s)"
            elif calls and failures >= calls:
                state = "❌ failing every call"
            elif calls:
                state = f"⚠️ probe failed ({failures}/{calls} failed)"
            else:
                state = "⚠️ probe failed (no calls yet)"
            detail = f" — {model_id}" if model_id else ""
            lines.append(f" {marker} {name}{detail}: {state}")
            if last_error:
                short = last_error[:160] + ("…" if len(last_error) > 160 else "")
                lines.append(f"    last error: {short}")
                hint = _hint(last_error)
                if hint:
                    lines.append(f"    {hint}")
            elif calls and not failures:
                lines.append(f"    {calls} calls, 0 failures")
        return "\n".join(lines)

    # ── devon: the autonomous dev & investigation agent ──────────────────────
    def _control_devon(self, tail: str, chat_key: str) -> str:
        """Owner types ``/devon <free text>``: Devon plans the tools that fit
        the question, runs them within a budget, journals every step into his
        own memory box, and digests the findings into a plain-English reply.

        With no task, it's a status line: what he's on and his last digests.
        """
        from ..devon import DevonAgent

        task = (tail or "").strip()
        agent = DevonAgent(self.context, brain=self.brain, gateway=self.gateway)
        if not task:
            digests = agent.recent_digests(3)
            if not digests:
                return (
                    "devon: give me a task — e.g. /devon check if the brain replied to the last messages\n"
                    "tools: read/grep code, git status+diff, live db, message flow, turn traces, "
                    "run tests, log tail, mood, chat+game stats, model chain health, config snapshot, "
                    "his own history, web research, scheduled watches, background missions.\n"
                    "every run is journaled in his memory box, so he remembers what he already checked."
                )
            lines = ["devon — recent digests:"]
            for d in digests:
                lines.append(f"  · {d['task'][:60]} → {d['digest'][:160]}")
            return "\n".join(lines)

        result = agent.run(task, chat_key=chat_key)
        tools = ", ".join(dict.fromkeys(s.tool for s in result.steps)) or "none"
        header = f"🔧 devon (planned by {result.planned_by}, {result.seconds:.0f}s · tools: {tools})"
        # wave 68: when the plan degraded, say WHY — an honest
        # "planned by heuristic (model down)" beats a silent fallback.
        if result.planned_by == "heuristic" and result.plan_error:
            header += f" — {result.plan_error[:90]}"
        if result.plan_error:
            # Persist EVERY plan degradation (heuristic AND reasoning-engine
            # fallback) so `nm mind` shows the last plan_error with a
            # timestamp (wave E router telemetry). Goes through the CoreMind
            # helper built for exactly this — never raises, never breaks
            # the reply.
            try:
                self.mind.record_plan_error(result.plan_error, route="devon")
            except Exception:  # noqa: BLE001 - telemetry never breaks a reply
                _log.debug("devon plan-error telemetry failed", exc_info=True)
        text = f"{header}\n{result.digest}"
        if len(text) > 1800:
            chat = self._ref_from_key(chat_key)
            return self._send_long_checked(chat.platform, chat, text)  # report already delivered in chunks
        return text

    # ── reasoning: /think — explicit, auditable multi-step thought ──────────
    def _control_think(self, tail: str, chat_key: str) -> str:
        """Owner types ``/think <question> [strategy]``: the reasoning
        engine works it out with an explicit trace — plan, subgoals,
        actions, observations, critiques, verdict — and answers with the
        full work shown, not just the conclusion."""
        from ..reasoning import ReasoningEngine, trace_text

        parts = (tail or "").strip().split(None, 1)
        if not parts or not parts[0].strip():
            return ("usage: /think <question> [strategy]\n"
                    "strategies: auto (default) · cot · decompose · "
                    "hypothesize · critique · tree\n"
                    "it shows the work — every step, then the answer.")
        question = parts[0].strip()
        strategy = parts[1].strip() if len(parts) > 1 and parts[1].strip() \
            else "auto"
        if strategy not in {"auto", "cot", "decompose", "hypothesize",
                            "critique", "tree"}:
            return (f"unknown strategy {strategy!r} — use auto, cot, "
                    "decompose, hypothesize, critique, or tree")
        engine = ReasoningEngine(self.context, max_llm_calls=14,
                                 max_seconds=120.0)
        try:
            result = engine.reason(question, strategy=strategy)
        except Exception as exc:  # noqa: BLE001 — never let a bad run kill chat
            return f"reasoning failed: {type(exc).__name__}: {exc}"
        text = trace_text(result)
        if len(text) > 3500:
            text = text[:3480] + "\n…(trace trimmed)"
        if len(text) > 1800:
            chat = self._ref_from_key(chat_key)
            return self._send_long_checked(chat.platform, chat, text)  # report already delivered in chunks
        return text

    # ── /benchmark — how sharp is the system right now ──────────────────────
    def _control_benchmark(self, tail: str) -> str:
        """Owner types ``/benchmark [dimension]``: runs the agent benchmark
        (1 task per dimension for speed in-chat) and reports the score."""
        from ..benchmark import run_benchmark

        dim = (tail or "").strip()
        dims = [dim] if dim in {"reasoning", "planning", "tool_use",
                                "self_correction"} else None
        if tail and not dims:
            return "dimensions: reasoning, planning, tool_use, self_correction"
        report = run_benchmark(self.context, dimensions=dims, limit=1)
        if not report.measurable:
            return (f"can't benchmark with the {report.provider!r} provider "
                    f"(mock/offline) — switch to a real model and ask again")
        overall = report.overall if report.overall is not None else 0.0
        lines = [f"benchmark: {overall:.2f} ({report.provider}, "
                 f"{report.seconds:.0f}s)"]
        for name, d in report.scores.items():
            score = "n/a" if d.score is None else f"{d.score:.0%}"
            lines.append(f"  {name}: {score} ({d.passed}/{d.total})")
        return "\n".join(lines)

    def _control_identity(self, tail: str) -> str:
        """Identity bank for signups: /identity show | set <field> <value> | clear.

        Stores the owner's real identity once (name, email, phone) so account
        creation flows can use it without asking every time. Persisted to DB.
        """
        from ...accounts.creator import AccountCreator
        from ...accounts.vault import CredentialVault

        parts = (tail or "").strip().split(None, 2)
        verb = parts[0].lower() if parts else "show"

        # Get or create the AccountCreator
        vault = CredentialVault(self.context.db if hasattr(self.context, "db") else None)
        creator = AccountCreator(vault, db=getattr(self.context, "db", None))

        if verb == "show":
            identity = creator.get_owner_identity()
            if not identity:
                return ("no identity bank set yet.\n"
                        "usage:\n"
                        "  /identity set name <your name>\n"
                        "  /identity set email <your email>\n"
                        "  /identity set phone <your phone>  (optional)")
            lines = ["identity bank:"]
            lines.append(f"  name: {identity.get('name', '')}")
            lines.append(f"  email: {identity.get('email', '')}")
            if identity.get("phone"):
                lines.append(f"  phone: {identity.get('phone', '')}")
            return "\n".join(lines)

        if verb == "set":
            if len(parts) < 3:
                return "usage: /identity set <name|email|phone> <value>"
            field, value = parts[1].lower(), parts[2].strip()
            if field not in ("name", "email", "phone"):
                return f"unknown field {field!r} — use name, email, or phone"
            if not value:
                return f"value for {field} cannot be empty"
            # Load existing (may be partial), update the field, persist.
            # Partial saves are allowed — signup flows validate completeness.
            current = creator.get_owner_identity() or {}
            current[field] = value
            try:
                import json, time
                if creator.db is not None:
                    creator.db.execute(
                        "INSERT OR REPLACE INTO kv_store (key, value, kind, updated_at)"
                        " VALUES (?, ?, 'json', ?)",
                        (creator.IDENTITY_KV_KEY, json.dumps(current), time.time()),
                    )
                creator._owner_identity = dict(current)
            except Exception as exc:  # noqa: BLE001
                return f"save failed: {exc}"
            missing = [f for f in ("name", "email") if not current.get(f)]
            if missing:
                return (f"set {field}. still need: {', '.join(missing)}\n"
                        f"  /identity set {missing[0]} <value>")
            return "identity bank complete."

        if verb == "clear":
            # Clear by setting empty (will fail validation, so do direct kv delete)
            try:
                if creator.db is not None:
                    creator.db.execute(
                        "DELETE FROM kv_store WHERE key = ?",
                        (creator.IDENTITY_KV_KEY,),
                    )
                creator._owner_identity = None
                return "identity bank cleared."
            except Exception as exc:  # noqa: BLE001
                return f"clear failed: {exc}"

        return "usage: /identity [show|set <field> <value>|clear]"
