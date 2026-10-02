"""RuntimeBuildMixin: PartnerRuntime command group (build)."""

from __future__ import annotations

import time
from ...partner.gating import gate_decision, is_owner_chat, is_restricted

class RuntimeBuildMixin:
    """RuntimeBuildMixin for :class:`PartnerRuntime`."""


    # ── swarm: /swarm ────────────────────────────────────────────────────────
    def _control_swarm(self, tail: str, *, chat_key: str) -> str:
        stripped = (tail or "").strip()
        # wave 86: /swarm research <topic> — the research swarm (parallel
        # researchers, conflict-aware synthesis) instead of devon builders.
        if stripped.lower().startswith("research"):
            topic = stripped[8:].strip()
            if not topic:
                return "usage: /swarm research <topic> — parallel research swarm"
            return self._control_swarm_research(topic, chat_key=chat_key)
        from ..swarm import SwarmAgent

        goal, workers = stripped, 3
        parts = stripped.rsplit(None, 1)
        if len(parts) == 2 and parts[1].isdigit() and 1 <= int(parts[1]) <= 5:
            goal, workers = parts[0].strip(), int(parts[1])
        if not goal:
            return ("usage: /swarm <goal> [workers 1-5] — parallel devon agents + fusion\n"
                    "        /swarm research <topic> — parallel research swarm")
        chat = self._ref_from_key(chat_key)
        try:
            self.gateway.send(chat.platform, chat,
                              f"🐝 swarm: {goal[:80]}\n{workers} worker(s) — this takes a few minutes.")
        except Exception:  # noqa: BLE001
            pass
        started = time.time()
        try:
            result = SwarmAgent(self.context, brain=self.brain, gateway=self.gateway).run(
                goal, workers=workers)
        except Exception as exc:  # noqa: BLE001
            return f"swarm crashed: {exc}"
        elapsed = time.time() - started
        ok_legs = sum(1 for leg in result.legs if leg.get("ok"))
        text = (f"🐝 swarm done: {ok_legs}/{len(result.legs)} legs ok, {elapsed:.0f}s\n\n"
                f"{result.synthesis}")
        return self._send_long_checked(chat.platform, chat, text)

    def _control_swarm_research(self, topic: str, *, chat_key: str) -> str:
        """wave 86: /swarm research <topic> — parallel research swarm."""
        from ..research_swarm import ResearchSwarm

        chat = self._ref_from_key(chat_key)
        try:
            self.gateway.send(chat.platform, chat,
                              f"🔬 research swarm: {topic[:80]}\n"
                              f"several angles at once — a minute or two.")
        except Exception:  # noqa: BLE001
            pass
        started = time.time()
        try:
            swarm = ResearchSwarm(self.context)
            report = swarm.run(topic, save_memory=True)
        except Exception as exc:  # noqa: BLE001
            return f"research swarm crashed: {exc}"
        elapsed = time.time() - started
        text = (f"🔬 swarm: {len(report.findings)} findings, "
                f"{len(report.angles)} angles, {len(report.sources)} sources, "
                f"{elapsed:.0f}s\n\n" + report.to_text(3200))
        return self._send_long_checked(chat.platform, chat, text)

    # ── the coding bot ───────────────────────────────────────────────────────
    def _control_code(self, tail: str, chat_key: str) -> str:
        from ..coding import CodingAgent

        task = (tail or "").strip()
        if not task:
            return "usage: /code <what to build> — e.g. /code write a script that renames all .JPG files to .jpg"
        chat = self._ref_from_key(chat_key)
        try:
            self.gateway.send(chat.platform, chat, f"⏳ coding bot: {task[:70]}\n(draft → run → fix, up to 5 tries)")
        except Exception:  # noqa: BLE001
            pass
        started = time.time()
        try:
            result = CodingAgent(self.context).run(task, max_iterations=5, timeout=120.0)
        except Exception as exc:  # noqa: BLE001
            return f"coding bot crashed: {exc}"
        elapsed = time.time() - started
        if result.ok:
            files = ", ".join(getattr(result, "files", []) or []) or "output"
            output = str(getattr(result, "output", ""))[:600]
            text = (f"✅ code done in {result.iterations} iteration(s), {elapsed:.0f}s — {files}\n"
                    f"run output:\n{output}")
        else:
            text = (f"❌ coding bot gave up after {result.iterations} iteration(s), {elapsed:.0f}s\n"
                    f"{str(getattr(result, 'error', ''))[:600]}\n"
                    f"it's in the workspace if you want to take over.")
        return self._send_long_checked(chat.platform, chat, text)

    # ── code interpreter (run python, keep a session) ───────────────────────
    def _control_py(self, tail: str, chat_key: str) -> str:
        """/py <python code> — run code in the sandbox and get the result.

        ``/py -s <name> <code>`` keeps variables in a named session;
        ``/py -r [name]`` resets a session. Runs inside the sandbox
        (resource-limited, no network) — same trust level as /code.
        """
        from ...tools.sandbox_code import CodeInterpreter

        raw = (tail or "").strip()
        if not raw:
            return ("usage: /py <python code>\n"
                    "  /py -s <name> <code>   keep variables in a session\n"
                    "  /py -r [name]          reset a session (default: console)\n"
                    "e.g. /py sum(range(100)) or /py -s work x = 10")
        session, reset = "console", False
        parts = raw.split(None, 2)
        if parts[0] == "-r":
            reset, session = True, (parts[1] if len(parts) > 1 else "console")
        elif parts[0] == "-s":
            if len(parts) < 3:
                return "usage: /py -s <name> <code>"
            session, raw = parts[1], parts[2].strip()
            if not raw:
                return "usage: /py -s <name> <code>"
        else:
            raw = raw.strip()

        try:
            if reset:
                CodeInterpreter().run("", session=session, reset=True)
                return f"session '{session}' reset."
            result = CodeInterpreter().run(raw, session=session, timeout=60.0)
        except Exception as exc:  # noqa: BLE001
            return f"code error: {exc}"

        if not result.get("ok") and result.get("error"):
            return f"code error: {result['error']}"
        lines = []
        if result.get("result") is not None:
            lines.append(f"→ {str(result['result'])[:400]}")
        if result.get("stdout"):
            lines.append(result["stdout"].strip()[:1200])
        if result.get("stderr"):
            lines.append("⚠ " + result["stderr"].strip()[-600:])
        if result.get("files"):
            lines.append("files: " + ", ".join(result["files"][:12]))
        if not lines:
            lines.append("(ran clean, no output)")
        head = f"[{session}] exit={result.get('exit_code')}, {result.get('seconds', 0):.2f}s"
        if result.get("timed_out"):
            head += " (timed out)"
        body = head + "\n" + "\n".join(l for l in lines if l)
        chat = self._ref_from_key(chat_key)
        return self._send_long_checked(chat.platform, chat, body)

    def _control_evolve(self, tail: str, chat_key: str) -> str:
        from ..evolution import EvolutionAgent
        from ..power import power_mode_for

        parts = (tail or "").split()
        verb = parts[0].lower() if parts else ""
        agent = EvolutionAgent(self.context)
        if verb == "audit":
            started = time.time()
            report = agent.audit()
            lines = [f"🔬 audit ({time.time() - started:.1f}s): {report['summary']}"]
            for f in report["findings"][:10]:
                lines.append(f"  [{f['severity']}] {f['where']} — {f['what']}")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), "\n".join(lines))
        if verb == "research" and len(parts) >= 2:
            topic = " ".join(parts[1:])
            started = time.time()
            outcome = self.context.tools.call("evolve_research", topic=topic)
            if not outcome.ok:
                return f"research failed: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            lines = [f"📚 research: {v['topic']} ({time.time() - started:.1f}s)",
                     f"evidence: {v['evidence_summary']}"]
            for r in v.get("recommendations", [])[:5]:
                targets = ", ".join(r.get("targets", [])[:3])
                lines.append(f"  • {r.get('title', '?')} → {targets or 'framework'} "
                             f"(risk: {r.get('risk', '?')})")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), "\n".join(lines))
        if verb == "revert" and len(parts) >= 2:
            started = time.time()
            try:
                out = agent.revert(parts[1])
            except Exception as exc:  # noqa: BLE001
                return f"revert failed: {exc}"
            if out.get("ok"):
                return (f"↩️ evolution {parts[1]} rolled back via "
                        f"{out['method']} ({time.time() - started:.0f}s, "
                        f"test gate re-verified: {out.get('verified')})")
            return f"revert problem: {out.get('report', 'see logs')[-400:]}"
        if verb == "git":
            st = agent.git_status()
            lines = [
                f"🌿 evolution git — on {st['branch'] or '(detached)'}",
                f"  remote: {st['remote'] or '(none — local only)'}",
                f"  work branch: {st['work_branch']}",
                f"  main branch: {st['main_branch']}",
                f"  push on publish: {st['push_on_publish']}",
                f"  tree clean: {st['clean']}",
            ]
            if "ahead" in st:
                lines.append(f"  vs upstream: +{st['ahead']} / -{st['behind']}")
            lines.append("  /evolve publish [branch] [--push] to make it permanent")
            return "\n".join(lines)
        if verb == "publish":
            target = parts[1] if len(parts) > 1 and not parts[1].startswith("--") else ""
            want_push = any(p.startswith("--push") for p in parts[1:])
            started = time.time()
            try:
                out = agent.git.publish(target,
                                        push=True if want_push else None)
            except Exception as exc:  # noqa: BLE001
                return f"publish failed: {exc}"
            elapsed = time.time() - started
            if not out.get("published"):
                return f"publish: {out.get('reason', 'nothing to do')} " \
                       f"({elapsed:.0f}s)"
            push_note = ""
            if want_push:
                if out.get("pushed"):
                    push_note = " — pushed to origin"
                else:
                    push_note = (f" — push skipped: "
                                 f"{out.get('reason', 'no origin remote')}")
            return (f"🚀 evolution published to {out['target']} "
                    f"({out['method']}, {elapsed:.0f}s)"
                    f"{push_note}: {out.get('commit', '')}")
        if verb == "auto":
            steps = parts[1] if len(parts) > 1 else "3"
            started = time.time()
            outcome = self.context.tools.call("evolve_auto", steps=steps)
            if not outcome.ok:
                return f"autopilot: {getattr(outcome.error, 'message', outcome.error)}"
            v = outcome.value
            lines = [f"🤖 autopilot ({time.time() - started:.0f}s): "
                     f"{len(v['applied'])} applied, {len(v['reverted'])} reverted, "
                     f"stopped: {v['stopped_reason']}"]
            for a in v["applied"]:
                lines.append(f"  ✅ {a['source']} → {a['instruction'][:60]} "
                             f"({a.get('tag') or a.get('commit') or 'on disk'})")
            for r in v["reverted"]:
                lines.append(f"  ↩️ {r['source']} → {r['instruction'][:60]} "
                             f"(gate failed, rolled back)")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), "\n".join(lines))
        if verb == "queue":
            sub = parts[1].lower() if len(parts) > 1 else "list"
            if sub == "add" and len(parts) >= 3:
                outcome = self.context.tools.call(
                    "evolve_queue", action="add", instruction=" ".join(parts[2:]))
                if not outcome.ok:
                    return f"queue failed: {getattr(outcome.error, 'message', outcome.error)}"
                return f"🎯 queued for autopilot ({outcome.value['queued']} total): " \
                       f"{' '.join(parts[2:])[:80]}"
            outcome = self.context.tools.call("evolve_queue", action=sub)
            if not outcome.ok:
                return f"queue failed: {getattr(outcome.error, 'message', outcome.error)}"
            goals = outcome.value["goals"]
            if not goals:
                return "autopilot queue is empty — /evolve queue add <goal>"
            lines = ["autopilot queue:"]
            lines += [f"  {i + 1}. {g[:80]}" for i, g in enumerate(goals[:10])]
            return "\n".join(lines)
        if verb in {"list", ""}:
            proposals = agent.list(8)
            if not proposals:
                return "no evolution proposals yet — /evolve <what to improve>"
            lines = ["evolution proposals:"]
            for p in proposals:
                files = ", ".join(e["path"] for e in p.edits[:4]) or "—"
                lines.append(f"  {p.id} [{p.status}] {p.instruction[:70]} → {files}")
            return "\n".join(lines)
        if verb == "apply" and len(parts) >= 2:
            proposal_id = parts[1]
            commit = len(parts) > 2 and parts[2].lower() in {"commit", "1", "true", "yes"}
            started = time.time()
            try:
                out = agent.apply(proposal_id, verify=True, commit=commit)
            except Exception as exc:  # noqa: BLE001
                return f"evolve failed: {exc}"
            elapsed = time.time() - started
            if out.get("applied"):
                return (f"🧬 evolution applied to the framework ({elapsed:.0f}s, "
                        f"full test suite passed): {', '.join(out['edits'])}\n"
                        f"commit: {out.get('commit') or 'on disk (not committed)'}")
            return (f"🧬 evolution REVERTED — verification failed, the bot is "
                    f"untouched ({elapsed:.0f}s):\n{out.get('report', '')[-1200:]}")
        if verb:
            instruction = " ".join(parts)
            plan_outcome = self.context.tools.call("evolve_plan", instruction=instruction)
            if not plan_outcome.ok:
                return f"evolve failed: {getattr(plan_outcome.error, 'message', plan_outcome.error)}"
            proposal = plan_outcome.value
            lines = [
                f"🧬 evolution plan {proposal['id']} ({len(proposal['edits'])} edits):",
            ]
            for edit in proposal["edits"][:6]:
                lines.append(f"  • {edit['path']}")
            if proposal.get("rationale"):
                lines.append(f"why: {proposal['rationale'][:200]}")
            power = power_mode_for(self.context).active
            if power:
                lines.append("power mode on — applying now (full test suite first)…")
                started = time.time()
                apply_outcome = self.context.tools.call(
                    "evolve_apply", proposal_id=proposal["id"], commit="")
                if apply_outcome.ok and apply_outcome.value.get("applied"):
                    lines.append(
                        f"✅ applied, all tests passed ({time.time() - started:.0f}s) — "
                        "restart to load it")
                elif apply_outcome.ok:
                    lines.append(
                        "↩️ verification failed — REVERTED, the framework is untouched:\n"
                        + apply_outcome.value.get("report", "")[-900:])
                else:
                    lines.append(
                        f"apply failed: {getattr(apply_outcome.error, 'message', apply_outcome.error)}")
            else:
                lines.append(
                    f"NOT applied — normal mode. Approve with: "
                    f"/evolve apply {proposal['id']}   (or turn power mode on)")
            return self._send_long_checked(self._ref_from_key(chat_key).platform,
                            self._ref_from_key(chat_key), "\n".join(lines))
        return ("usage: /evolve <instruction> | /evolve apply <id> [commit] "
                "| /evolve list")

    def _control_upgrade(self, tail: str, *,
                         _chat: Any | None = None,
                         _pipeline: Any | None = None) -> str:
        """The research → approve → evolve loop, from chat.

        /upgrade list            pending proposals, one line each
        /upgrade show <id>       full ticket: problem, patch plan, risk
        /upgrade diff <id>       preview the actual patch before approving
        /upgrade approve <id>    approve → test-gated apply → what-changed digest
        /upgrade deny <id> <reason>
        /upgrade applied         recently applied, with what-changed digests

        Owner-only, enforced at two layers: ``on_message`` only routes
        slash commands into ``handle_control`` for operator chats (the
        ``_is_operator`` gate), and this method re-checks
        :func:`is_owner_chat` itself before touching the queue — a direct
        call from a non-owner chat is denied, fail-closed.  ``_chat`` is
        the originating chat (``message.chat``); ``None`` means no chat
        was proven and is denied.

        ``_pipeline`` is a test seam (a mock pipeline); production always
        uses the real UpgradePipeline.
        """
        if not is_owner_chat(_chat,
                             owner_chats=getattr(self, "_owner_chats", ())):
            return "owner-only: /upgrade is not available in this chat."

        import time

        from ...core.errors import NoMoralsError
        from ..upgrade_chat import (
            UPGRADE_USAGE,
            render_applied_digest,
            render_upgrade_diff,
            render_upgrade_list,
            render_upgrade_show,
            resolve_proposal,
        )
        from ..upgrade_queue import UpgradePipeline, UpgradeQueue

        parts = (tail or "").strip().split(None, 1)
        verb = (parts[0] if parts else "list").lower()
        rest = parts[1] if len(parts) > 1 else ""
        queue = UpgradeQueue(self.context)
        pipeline = (_pipeline if _pipeline is not None
                    else UpgradePipeline(self.context))

        if verb == "list":
            return render_upgrade_list(queue.list(status="proposed"))

        if verb in ("show", "diff"):
            proposal, err = resolve_proposal(queue, rest)
            if err:
                return err
            if verb == "show":
                return render_upgrade_show(proposal)
            evo = None
            plan = proposal.get("patch_plan") or {}
            evo_id = (plan.get("evolution_proposal_id")
                      if isinstance(plan, dict) else "")
            if evo_id:
                # the ticket references a planned evolution proposal — load
                # its real edits so the preview shows actual hunks. Any
                # load problem falls back to the ticket text.
                try:
                    agent = getattr(pipeline, "evolution", None)
                    evo = agent._load(str(evo_id)) if agent is not None else None
                except Exception:  # noqa: BLE001 — fallback is fine
                    evo = None
            return render_upgrade_diff(proposal, evo_proposal=evo)

        if verb == "approve":
            proposal, err = resolve_proposal(queue, rest)
            if err:
                return err
            if (proposal.get("status") or "") != "proposed":
                return (f"{proposal.get('id')} is "
                        f"'{proposal.get('status')}' — only 'proposed' "
                        "tickets can be approved.")
            started = time.time()
            try:
                done = pipeline.approve_and_implement(
                    str(proposal.get("id")), by="owner")
            except Exception as exc:  # noqa: BLE001 — already recorded failed
                return (f"❌ apply failed and was recorded as failed: "
                        f"{type(exc).__name__}: {exc}")
            digest = render_applied_digest(done)
            return f"{digest}\n  ⏱️ took {time.time() - started:.0f}s"

        if verb == "deny":
            sub = rest.split(None, 1)
            if len(sub) < 2:
                return "usage: /upgrade deny <id> <reason>"
            ref, reason = sub
            proposal, err = resolve_proposal(queue, ref)
            if err:
                return err
            try:
                denied = pipeline.deny_with_reason(
                    str(proposal.get("id")), reason, by="owner")
            except NoMoralsError as exc:
                return f"deny failed: {exc}"
            return (f"🚫 upgrade denied: {denied.get('title') or denied.get('id')}\n"
                    f"reason: {reason.strip()}")

        if verb == "applied":
            rows = queue.list(status="implemented", limit=10)
            if not rows:
                return "no upgrades applied yet."
            lines = [f"applied upgrades ({len(rows)}):"]
            lines.extend(render_applied_digest(p) for p in rows)
            return "\n\n".join(lines)

        return UPGRADE_USAGE

    def _control_deliver(self, tail: str, chat_key: str) -> str:
        """`/deliver report <topic> [--section "T::body"] [--to p:c] [--no-pdf]`.

        Create-and-deliver: styled HTML report + real PDF → zip → the live
        gateway's file-send path.  Default target is the chat the command
        came from; `--to platform:chat` overrides.
        """
        import shlex

        usage = ('usage: /deliver report <topic> --section "Title::body" '
                 "[--to platform:chat] [--no-pdf]")
        try:
            tokens = shlex.split(tail or "")
        except ValueError as exc:
            return f"{usage} (could not parse: {exc})"
        if not tokens or tokens[0].lower() != "report":
            return usage
        topic_parts: list[str] = []
        sections: list[tuple[str, str]] = []
        target = ""
        include_pdf = True
        i = 1
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--section" and i + 1 < len(tokens):
                raw = tokens[i + 1]
                sec_title, sep, sec_body = raw.partition("::")
                if not sep or not sec_title.strip():
                    return f"{usage} (bad --section {raw!r}; use 'Title::body')"
                sections.append((sec_title.strip(), sec_body.strip()))
                i += 2
            elif tok == "--to" and i + 1 < len(tokens):
                target = tokens[i + 1].strip()
                i += 2
            elif tok == "--no-pdf":
                include_pdf = False
                i += 1
            else:
                topic_parts.append(tok)
                i += 1
        topic = " ".join(topic_parts).strip()
        if not topic:
            return usage + " — a topic is required"
        if not sections:
            return (f"{usage} — at least one section is required, e.g.\n"
                    f'/deliver report "{topic}" '
                    '--section "Overview::The key points…"')
        ref = self._ref_from_key(chat_key)
        platform, chat_id = ref.platform, ref.chat_id
        if target:
            if ":" in target:
                platform, chat_id = (p.strip() for p in target.split(":", 1))
            else:
                chat_id = target
        try:
            from ...tools.deliver_report import deliver_report

            out = deliver_report(self.context, topic, sections, platform,
                                 chat_id, include_pdf=include_pdf)
        except Exception as exc:  # noqa: BLE001 - the reply carries the failure
            return f"deliver failed: {exc}"
        name = out["zip_path"].rsplit("/", 1)[-1]
        return (f"📄 delivered {name} → {platform}:{chat_id} "
                f"({out['zip_bytes']} B, {len(out['sections'])} sections, "
                f"message {out['message_id']})")
