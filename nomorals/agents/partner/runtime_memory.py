"""RuntimeMemoryMixin: PartnerRuntime command group (memory)."""

from __future__ import annotations

import re
import time

class RuntimeMemoryMixin:
    """RuntimeMemoryMixin for :class:`PartnerRuntime`."""


    # ── long-term memory: /remember /recall /forget ──────────────────────────
    def _control_remember(self, tail: str, *, chat_key: str = "") -> str:
        """Explicitly store something: /remember <text> [kind] [tags:a,b]."""
        from ...memory.base import ALL_KINDS

        raw = (tail or "").strip()
        if not raw:
            return ("usage: /remember <what to remember> [kind] [tags:a,b]\n"
                    "kinds: fact preference decision relationship episode skill lesson")
        tokens = raw.split()
        kind = "episode"
        for i, tok in enumerate(tokens):
            if tok.lower() in ALL_KINDS:
                kind = tok.lower()
                tokens = tokens[:i] + tokens[i + 1:]
                break
        content = " ".join(tokens)
        tags = ""
        if " " in content and content.rsplit(" ", 1)[-1].startswith("tags:"):
            content, tagpart = content.rsplit(" ", 1)
            tags = tagpart[len("tags:"):]
        content = content.strip()
        if not content:
            return "usage: /remember <what to remember> [kind] [tags:a,b]"
        record_id = self.context.memory.remember(
            content, kind=kind, importance=0.9,
            source="user:command", origin=f"chat:{chat_key or 'command'}",
            tags=tags,
        )
        return f"remembered it. [{kind}] {content[:120]} (id {record_id[:8]})"

    def _control_recall(self, tail: str) -> str:
        """Show the top memories matching the query (or the freshest, if none)."""
        query = (tail or "").strip()
        if self.context.memory is None:
            return "memory is off in this session."
        result = self.context.memory.recall(query, limit=5)
        if not result.records:
            return "nothing in memory matches that yet — try /remember."
        lines = ["from memory:"]
        now = time.time()
        for r in result.records:
            days = (now - r.created_at) / 86400.0
            age = f"today" if days < 0.05 else f"{days:.0f}d ago" if days < 60 else f"{days / 30.0:.0f}mo ago"
            score = f" {r.score:.2f}" if r.score else ""
            tags = f" #{r.tags}" if r.tags else ""
            lines.append(f"  [{r.kind}]{tags} {r.content[:130]}  ({age}, id {r.id[:6]})")
        return "\n".join(lines)

    # ── representation review: /review /how did I do ─────────────────────
    def _control_review(self, tail: str, *, chat_key: str = "") -> str:
        """/review [days] — how well did I represent you?  Owner-only."""
        from ...cognition.representation import _ledger_for

        days = 30.0
        raw = (tail or "").strip()
        if raw:
            try:
                days = float(raw.split()[0])
            except (ValueError, IndexError):
                return "usage: /review [days] — e.g. /review 7"
            if not 0 < days <= 365:
                return "/review takes 1–365 days"
        try:
            ledger = _ledger_for(
                getattr(self, "context", None)
                and getattr(self.context, "settings", None))
            return ledger.summary(limit=10, since_days=days)
        except Exception as exc:  # noqa: BLE001 — review never breaks chat
            return f"couldn't pull the review: {exc}"

    def _control_tutor(self, tail: str, *, chat_key: str = "") -> str:
        """/tutor — Socratic tutoring. Owner-only.

        /tutor <topic>             start in direct ("just tell me") mode
        /tutor guide me <topic>    start in socratic ("guide me") mode
        /tutor answer <text>       answer the current question
        /tutor hint                progressive hint (never the answer, socratic)
        /tutor status              mastery snapshot + mistake count
        /tutor stop                end the session
        """
        from ...learn.tutor import (
            TutorSession, end_session, get_notebook, get_session, set_session,
        )
        raw = (tail or "").strip()
        if not raw:
            return ("usage: /tutor [guide me] <topic> — e.g. /tutor guide me "
                    "fractions\n       /tutor answer <text> | /tutor hint | "
                    "/tutor status | /tutor stop")
        verb, _, rest = raw.partition(" ")
        verb = verb.lower()

        if verb == "stop":
            return ("session ended." if end_session(chat_key)
                    else "no tutoring session running.")
        if verb == "status":
            sess = get_session(chat_key)
            if sess is None:
                return "no tutoring session running."
            snap = sess.mastery.snapshot()
            nb = get_notebook(chat_key)
            lines = [f"📚 tutoring '{sess.topic}' ({sess.mode} mode, "
                     f"{sess.turns} turns)"]
            for skill, val in sorted(snap.items(), key=lambda kv: kv[1]):
                bar = "█" * int(val * 10) + "░" * (10 - int(val * 10))
                lines.append(f"  {skill}: {bar} {val:.0%}")
            lines.append(f"  mistakes notebooked: {nb.count()}")
            return "\n".join(lines)
        if verb == "hint":
            sess = get_session(chat_key)
            if sess is None:
                return "no tutoring session running — /tutor <topic> to start."
            try:
                turn = sess.hint()
            except Exception as exc:  # noqa: BLE001
                return f"hint failed: {exc}"
            return f"{turn.feedback}\n{turn.prompt}"
        if verb == "answer":
            sess = get_session(chat_key)
            if sess is None:
                return "no tutoring session running — /tutor <topic> to start."
            if not rest.strip():
                return "usage: /tutor answer <your answer>"
            try:
                turn = sess.respond(rest.strip())
            except Exception as exc:  # noqa: BLE001
                return f"couldn't process that: {exc}"
            out = f"{turn.feedback}\n\n{turn.prompt}" if turn.feedback else turn.prompt
            # Wrong answers become flashcards automatically.
            if turn.diagnosis == "wrong" and sess.mode == "socratic":
                try:
                    nb = get_notebook(chat_key)
                    nb.record(sess._current_question or sess.topic,
                              rest.strip(), sess.expected_answer,
                              topic=sess.topic)
                    out += "\n\n📝 noted in your mistake notebook for review."
                except Exception:  # noqa: BLE001
                    pass
            if turn.done:
                out += "\n\n🎓 you've got this one — nice work."
            return out

        # Start a new session: "guide me <topic>" -> socratic, else direct.
        if verb == "guide" and rest.lower().startswith("me "):
            mode, topic = "socratic", rest[3:].strip()
        elif verb == "guide":
            return "did you mean: /tutor guide me <topic>?"
        else:
            mode, topic = "direct", raw
        if not topic:
            return "usage: /tutor [guide me] <topic>"
        try:
            sess = TutorSession(topic=topic, mode=mode)
            set_session(chat_key, sess)
            turn = sess.start()
        except Exception as exc:  # noqa: BLE001
            return f"couldn't start tutoring: {exc}"
        opener = ("🔍 socratic mode — I'll guide you, not tell you. "
                  "Answer with /tutor answer <text>."
                  if mode == "socratic" else
                  "📖 direct mode — straight teaching. "
                  "Ask with /tutor answer <text> to check yourself.")
        return f"{opener}\n\n{turn.prompt}"

    def _control_ekiti(self, tail: str, *, chat_key: str = "") -> str:
        """/ekiti — Ekiti/Ilawe Ekiti dialect tutoring. Owner-only.

        /ekiti converse <text>   practice conversation in Ekiti dialect
        /ekiti debate <topic>     argue with me in Ekiti (opposing side)
        /ekiti say <text>         native-speaker reference audio (XTTS, private)
        /ekiti check "<expected>" arm a pronunciation check, then send a voice note
        /ekiti tones              the three Yoruba tones + minimal pairs
        """
        from ...learn.dialect import (
            TONE_GUIDE,
            YORUBA_MINIMAL_PAIRS,
            arm_check,
            get_tutor,
        )
        raw = (tail or "").strip()
        if not raw:
            return ("usage: /ekiti converse <text> | /ekiti debate <topic> | "
                    '/ekiti say <text> | /ekiti check "<expected>" | /ekiti tones')
        verb, _, rest = raw.partition(" ")
        verb = verb.lower()
        rest = rest.strip()
        try:
            if verb == "tones":
                pairs = "\n".join(
                    f"• {form} = {gloss} ({tone})"
                    for form, gloss, tone in YORUBA_MINIMAL_PAIRS)
                return f"{TONE_GUIDE}\n\nMinimal pairs:\n{pairs}"
            if verb == "converse":
                if not rest:
                    return "usage: /ekiti converse <text in Yoruba/Ekiti>"
                tutor = get_tutor(chat_key)
                turn = tutor.converse(rest)
                out = turn.response
                if turn.correction:
                    out += f"\n\n💡 {turn.correction}"
                return out
            if verb == "debate":
                if not rest:
                    return "usage: /ekiti debate <topic>"
                tutor = get_tutor(chat_key)
                turn = tutor.debate(rest)
                return (f"🥊 round {turn.debate_round} — mi lòdì sí ọ̀!\n\n"
                        f"{turn.response}")
            if verb == "say":
                if not rest:
                    return "usage: /ekiti say <text>"
                tutor = get_tutor(chat_key)
                try:
                    result = tutor.reference_audio(rest[:500])
                except Exception as exc:  # noqa: BLE001
                    return f"reference audio failed: {exc}"
                path = result.get("path", "")
                chat = self._ref_from_key(chat_key)
                try:
                    sent = self.gateway.send_file(
                        chat.platform, chat, path,
                        caption="🎙️ Ekiti reference — shadow this")
                    if getattr(sent, "ok", False):
                        return f"🎙️ reference audio sent ({result.get('backend', '?')})"
                except Exception:  # noqa: BLE001
                    pass
                return f"🎙️ reference audio saved at {path}"
            if verb == "check":
                try:
                    return arm_check(chat_key, rest)
                except Exception as exc:  # noqa: BLE001
                    return str(exc)
            return ("usage: /ekiti converse <text> | /ekiti debate <topic> | "
                    '/ekiti say <text> | /ekiti check "<expected>" | /ekiti tones')
        except Exception as exc:  # noqa: BLE001
            return f"ekiti tutor failed: {exc}"

    def _control_course(self, tail: str, *, chat_key: str = "") -> str:
        """/course — WAEC/JAMB course builder. Owner-only.

        /course <topic>          start scoping (3 questions first — never
                                 builds from a bare prompt)
        /course waec <subject>   browse the syllabus topics for a subject
        /course set <field> <v>  answer a scoping question (level/depth/start)
        /course build            storyboard + build the course
        /course list             saved courses
        /course teach <id>       start the embedded course tutor
        /course status           pending scoping state
        """
        from ...learn.curriculum import (
            Course, CourseScope, build_course, clear_pending_scope,
            courses_dir, find_topic, pending_scope, scoping_prompt,
            set_pending_scope, subject_topics, syllabus_code,
        )
        raw = (tail or "").strip()
        if not raw:
            return ("usage: /course <topic> — e.g. /course quadratic equations\n"
                    "       /course waec physics | /course build | /course list")
        verb, _, rest = raw.partition(" ")
        verb = verb.lower()

        if verb == "waec":
            subject = rest.strip()
            topics = subject_topics(subject)
            if not topics:
                return ("unknown subject — try: physics, chemistry, biology, "
                        "mathematics, english, economics")
            lines = [f"📖 WAEC {topics[0].subject} syllabus "
                     f"({len(topics)} topics):"]
            for t in topics:
                jamb = " ·JAMB" if t.jamb else ""
                lines.append(f"  {t.code.split()[-1]} {t.title}{jamb}")
            return "\n".join(lines)

        if verb == "list":
            files = sorted(courses_dir().glob("*.json"))
            if not files:
                return "no courses yet — /course <topic> to start one."
            lines = ["📚 saved courses:"]
            for f in files:
                c = Course.load(f.stem)
                if c is not None:
                    lines.append(f"  {c.id} — {c.title} "
                                 f"({len(c.lessons)} lessons)")
            return "\n".join(lines)

        if verb == "status":
            scope = pending_scope(chat_key)
            if scope is None:
                return "no course being scoped — /course <topic> to start."
            return (f"📚 scoping **{scope.topic}**: level={scope.level}, "
                    f"depth={scope.depth}, start={scope.start}\n"
                    f"{scoping_prompt(scope).splitlines()[1]}")

        if verb == "set":
            scope = pending_scope(chat_key)
            if scope is None:
                return "nothing to set — /course <topic> first."
            field, _, value = rest.partition(" ")
            field, value = field.strip().lower(), value.strip().lower()
            if field == "level" and value in ("waec", "jamb", "both"):
                scope.level = value
            elif field == "depth" and value in ("quick", "full"):
                scope.depth = value
            elif field == "start" and value:
                scope.start = value
            elif field == "weeks" and value.isdigit():
                scope.weeks = max(1, int(value))
            else:
                return ("usage: /course set level <waec|jamb|both> | "
                        "/course set depth <quick|full> | "
                        "/course set start <code|topic|syllabus order> | "
                        "/course set weeks <n>")
            set_pending_scope(chat_key, scope)
            return f"set {field}={value} for **{scope.topic}**."

        if verb == "build":
            scope = pending_scope(chat_key)
            if scope is None:
                return "nothing to build — /course <topic> first."
            try:
                course = build_course(scope)
            except ValueError as exc:
                return str(exc)
            except Exception as exc:  # noqa: BLE001
                return f"course build failed: {exc}"
            course.save()
            clear_pending_scope(chat_key)
            lines = [f"✅ **{course.title}** — {len(course.lessons)} lessons:"]
            for ls in course.lessons[:10]:
                lines.append(f"  {ls.n}. {ls.title} ({ls.syllabus_code})")
            if len(course.lessons) > 10:
                lines.append(f"  …and {len(course.lessons) - 10} more")
            lines.append(f"\n/course teach {course.id} — start the tutor")
            return "\n".join(lines)

        if verb == "teach":
            cid = rest.strip().split()[0] if rest.strip() else ""
            course = Course.load(cid) if cid else None
            if course is None:
                return "unknown course id — /course list to see them."
            try:
                from ...learn.curriculum import course_tutor
                from ...learn.tutor import set_session
                session = course_tutor(course)
                set_session(chat_key, session)
                turn = session.start()
            except Exception as exc:  # noqa: BLE001
                return f"couldn't start the course tutor: {exc}"
            return (f"📚 tutoring from **{course.title}** "
                    f"(course content only)\n\n{turn.feedback}\n{turn.prompt}")

        # Otherwise: the tail is a topic -> start scoping (never build bare).
        scope = CourseScope(topic=raw)
        hit = find_topic(raw)
        if hit is not None:
            scope.subject = hit.subject
        set_pending_scope(chat_key, scope)
        code_hint = f"\n🔎 matched syllabus: {syllabus_code(hit)}" \
            if hit is not None else ""
        return scoping_prompt(scope) + code_hint

    def _control_forget(self, tail: str) -> str:
        """Forget by id, or by the best matching description."""
        target = (tail or "").strip()
        if not target:
            return "usage: /forget <id or description>"
        if self.context.memory is None:
            return "memory is off in this session."
        if re.fullmatch(r"[0-9a-f]{12}", target):
            removed = self.context.memory.forget(target)
            return ("forgotten." if removed
        else f"no memory with id {target} — /recall <query> to search instead")
        record = self.context.memory.find_one(target)
        if record is None:
            return "didn't find that in memory — give me the id from /recall, or better words."
        self.context.memory.forget(record.id)
        return f"forgotten: [{record.kind}] {record.content[:110]}"

    # ── core mind (wave 87) ──────────────────────────────────────────────────
    def _control_mind(self, tail: str, chat_key: str) -> str:
        """/mind — the manual override over the Core Mind."""
        tail = (tail or "").strip()
        if tail in ("", "status"):
            return self.mind.status()
        if tail == "clear":
            n = self.mind.clear(chat_key)
            return f"mind: {n} pending clarification(s) cleared."
        if tail == "pending":
            pend = self.mind.pending()
            if not pend:
                return "no open clarifications."
            lines = []
            for key, p in pend.items():
                lines.append(f"  {key}: “{p.get('question', '')[:80]}”")
            return "open clarifications:\n" + "\n".join(lines)
        # /mind <goal> — force the routing and show the decision
        intent = self.mind.decide(tail, live_game=self.mind._live_game(chat_key))
        if intent.kind == "chat":
            return (f"the mind reads no goal in “{tail[:60]}” — “{tail[:30]}” "
                    f"stays conversation. Try a verb: research/build/browse/"
                    f"download/mission: …")
        if intent.action == "ask":
            question = self.mind._question_for(intent)
            self.mind._set_pending(chat_key, intent, question)
            return f"decision: {intent.kind} (ask) — {intent.why}\n{question}"
        reply = self.mind._dispatch(intent, chat_key, self._fake_message(chat_key))
        return f"decision: {intent.kind} (route {intent.route}) — {intent.why}\n" + \
            (reply or "(no reply)")

    def _control_health(self, tail: str, *, chat_key: str = "") -> str:
        """Patient-side health timeline. Owner-only. TRACKING ONLY.

        /health log <text>     log an event ("headache, 3/5, since morning")
        /health timeline       chronological entries
        /health summary [days] human-readable recap (default 30 days)
        /health ask <question> ask your biometrics (sleep/recovery/activity)
        /health readiness      recovery score from sleep + HRV + strain
        /health week           this week's movement/sleep/recovery recap
        /health patterns       mood↔sleep correlations from your logs
        /health route <symptoms>  conservative care routing (navigation only)
        /health costs          typical Nigerian private-hospital costs (approximate)
        /health prep [symptoms] pre-visit packet: timeline + questions + what to bring
        /health visited <notes> post-visit recap: what the doctor said, structured

        Devon is not a doctor. This records what the user reports;
        it never interprets medically.
        """
        from ...health.previsit import (
            format_costs,
            format_recap,
            format_route,
            prepare_visit,
            summarize_visit,
            triage_route,
        )
        from ...health.timeline import HealthTimeline, parse_health_note
        raw = (tail or "").strip()
        if not raw:
            return ("usage: /health log <text> — e.g. /health log headache, "
                    "3/5, since morning\n"
                    "       /health timeline | /health summary [days]\n"
                    "       /health ask <question> — e.g. /health ask how "
                    "did I sleep this week?\n"
                    "       /health readiness | /health week\n"
                    "       /health route <symptoms> — where to go next\n"
                    "       /health costs — typical private-hospital costs\n"
                    "       /health prep [symptoms] — pre-visit packet\n"
                    "       /health visited <notes> — post-visit recap\n"
                    "       /health meal — photo meal logging (2 questions)\n"
                    "tracking only — I'm not a doctor; show the log to "
                    "yours for medical guidance.")
        verb, _, rest = raw.partition(" ")
        verb = verb.lower()
        try:
            tl = HealthTimeline()
        except Exception as exc:  # noqa: BLE001
            return f"health log unavailable: {exc}"
        try:
            if verb == "log":
                if not rest.strip():
                    return "usage: /health log <text>"
                draft = parse_health_note(rest)
                ev = tl.log(draft["event_type"], draft["text"],
                            severity=draft["severity"], source="chat")
                sev = f" [{ev.severity}/5]" if ev.severity else ""
                return (f"logged 🩺 {ev.event_type}{sev}: "
                        f"{ev.text[:120]}")
            if verb == "timeline":
                events = tl.timeline(limit=50)
                if not events:
                    return "health timeline is empty — /health log <text> to start."
                lines = [f"🩺 health timeline ({len(events)} entries):"]
                for ev in events[-20:]:
                    sev = f" [{ev.severity}/5]" if ev.severity else ""
                    lines.append(f"• {ev.when_str()} · {ev.event_type}{sev}: "
                                 f"{ev.text[:100]}")
                return "\n".join(lines)
            if verb == "summary":
                days = 30
                if rest.strip().isdigit():
                    days = max(1, min(365, int(rest.strip())))
                return tl.summary(days=days)
            if verb == "route":
                if not rest.strip():
                    return ("usage: /health route <symptoms> — e.g. "
                            "/health route headache and fever since morning")
                symptoms = [s.strip() for s in rest.split(",") if s.strip()]
                route = triage_route(
                    symptoms or [rest.strip()],
                    history=tl.timeline(event_type="symptom", limit=20))
                return format_route(route)
            if verb == "costs":
                return format_costs()
            if verb == "prep":
                symptoms = [s.strip() for s in rest.split(",")
                            if s.strip()] or None
                return prepare_visit(symptoms, timeline=tl)
            if verb == "visited":
                if not rest.strip():
                    return "usage: /health visited <what the doctor said>"
                return format_recap(
                    summarize_visit(rest.strip(), timeline=tl))
            if verb == "meal":
                # Photo meal logging (owner-only, tracking only).
                # /health meal — arms the flow: send a photo of your meal,
                # Devon identifies it and asks at most 2 questions the photo
                # can't answer (hidden fats, portion, drinks).
                from ...health import nutrition as _nut
                _nut.arm_meal_flow(chat_key)
                return ("send a photo of your meal — I'll identify the foods "
                        "and ask at most 2 questions the photo can't answer "
                        "(oil, portion size, drinks). estimates are always "
                        "approximate ranges, never exact numbers.")
            if verb == "ask":
                if not rest.strip():
                    return ("usage: /health ask <question> — e.g. /health ask "
                            "how did I sleep this week?")
                from ...health.coach import HealthCoach
                coach = HealthCoach(timeline=tl)
                return coach.ask(rest.strip()).text
            if verb == "readiness":
                from ...health.coach import HealthCoach
                coach = HealthCoach(timeline=tl)
                return coach.readiness().format()
            if verb == "week":
                from ...health.coach import HealthCoach
                coach = HealthCoach(timeline=tl)
                return coach.weekly_recap().text
            if verb == "patterns":
                from ...health.patterns import detect_patterns, format_patterns
                return format_patterns(detect_patterns(tl, days=30))
            return ("usage: /health log <text> | /health timeline | "
                    "/health summary [days] | /health ask <question> | "
                    "/health readiness | /health week | "
                    "/health route <symptoms> | "
                    "/health costs | /health prep [symptoms] | "
                    "/health visited <notes>")
        finally:
            tl.close()

    def _control_routine(self, tail: str, *, chat_key: str = "") -> str:
        """Natural-language smart-home routines. Owner-only.

        /routine <natural language>  parse → describe → pending confirm
        /routine confirm <id>        validate → activate via Home Assistant
        /routine list                pending drafts
        """
        from ...integrations.routines import (
            activate, build_routine, confirm, describe, load_aliases,
            pending_draft, validate,
        )
        rest = (tail or "").strip()
        if not rest:
            return ("usage: /routine <natural language> — e.g. "
                    "\"/routine every morning at 7, turn on the kitchen "
                    "lights\"\n"
                    "       /routine confirm <id> — activate a parsed routine\n"
                    "       /routine list — show pending drafts")
        if rest == "list":
            from ...integrations import routines as _r
            drafts = list(_r._DRAFTS.values())
            if not drafts:
                return "no pending routine drafts."
            lines = ["pending routines:"]
            for d in drafts:
                lines.append(f"• {d.id}: {d.raw[:60]}")
            return "\n".join(lines)
        if rest.startswith("confirm "):
            draft_id = rest[len("confirm "):].strip()
            draft = pending_draft(draft_id)
            if draft is None:
                return f"no pending draft '{draft_id}'. Use /routine list."
            try:
                routine = confirm(draft)
            except Exception as exc:  # noqa: BLE001 — show the problem
                return f"can't activate yet:\n{exc}"
            # activate via Home Assistant (best-effort wiring)
            try:
                import asyncio
                from ...integrations.smarthome_integration import (
                    SmartHomeIntegration)
                integration = getattr(self, "_smarthome", None)
                if integration is None:
                    return (f"✅ routine validated: {describe(draft)}\n"
                            "⚠️ smart-home backend not connected here — "
                            "connect Home Assistant to activate it.")
                result = asyncio.run(activate(routine, integration))
                return (f"✅ routine active: {result.name}\n"
                        f"(Home Assistant automation {result.ha_automation_id})")
            except Exception as exc:  # noqa: BLE001 — never break chat
                return (f"✅ routine validated: {describe(draft)}\n"
                        f"⚠️ activation failed: {exc}")
        # parse a new routine
        memory = getattr(self, "_memory", None)
        devices: list[dict] = []
        try:
            aliases = load_aliases(memory)
        except Exception:  # noqa: BLE001
            aliases = {}
        draft = build_routine(rest, devices=devices, aliases=aliases,
                              memory=memory)
        errors = validate(draft)
        out = describe(draft)
        if errors:
            out += "\n\n⚠️ needs fixing:\n" + "\n".join(
                f"• {e.message}" for e in errors)
        else:
            out += (f"\n\nReply `/routine confirm {draft.id}` to activate it.")
        return out
