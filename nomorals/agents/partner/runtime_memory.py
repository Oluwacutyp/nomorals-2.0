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

    def _control_train(self, tail: str, *, chat_key: str = "") -> str:
        """Recovery-gated training plans + voice coaching. Owner-only.

        /train plan <goal> [equipment] [weeks] — periodized plan
        /train today — today's workout, gated by recovery
        /train readiness — recovery score from sleep + HRV + strain
        /train log [completed|skipped] [rpe] [notes]
        /train list — your plans
        """
        from ...health.training import TrainingCoach, control_train
        coach = getattr(self, "_training_coach", None)
        if coach is None:
            coach = TrainingCoach()
            self._training_coach = coach
        return control_train(tail or "", coach=coach)

    def _control_form(self, tail: str, *, chat_key: str = "") -> str:
        """Seer-powered form coach — video movement analysis. Owner-only.

        /form analyze <video-or-photo> <exercise> — form read + fixes
        /form gait <video> — running gait risk flags
        /form movements — the 5 coached movements
        /form apply <analysis-id> — mobility work into today's session
        """
        from ...health.form import FormStore, control_form
        from ...health.training import TrainingCoach
        coach = getattr(self, "_training_coach", None)
        if coach is None:
            coach = TrainingCoach()
            self._training_coach = coach
        store = getattr(self, "_form_store", None)
        if store is None:
            store = FormStore()
            self._form_store = store
        return control_form(tail or "", coach=coach, store=store)

    def _control_film(self, tail: str, *, chat_key: str = "") -> str:
        """VOD film study — mistakes + highlights + fingerprints. Owner-only.

        /film analyze <video> <game> [exercise] — VOD breakdown
        /film games — supported games
        /film fingerprint [game] — recurring playstyle patterns
        /film list [game] — past breakdowns
        /film export <id> — shareable summary
        """
        from ...games.film import (FingerprintStore, FilmStore,
                                   control_film)
        store = getattr(self, "_film_store", None)
        if store is None:
            store = FilmStore()
            self._film_store = store
        fps = getattr(self, "_fingerprint_store", None)
        if fps is None:
            fps = FingerprintStore()
            self._fingerprint_store = fps
        return control_film(tail or "", store=store, fingerprints=fps)

    def _control_audio(self, tail: str, *, chat_key: str = "") -> str:
        """Transcript-as-timeline audio editing. Owner-only.

        /audio edit <file> [lang] — transcribe with word timings
        /audio fillers <file> [lang] — cut filler words
        /audio enhance <file> — denoise before transcribing
        """
        from ...audio.edit import control_audio
        return control_audio(tail or "")

    def _control_overview(self, tail: str, *, chat_key: str = "") -> str:
        """Interactive Audio Overviews — the podcast you can talk to. Owner-only.

        /overview make <deep-dive|debate|brief> [lang] <Title :: text ;; ...>
        /overview list — saved overviews
        /overview ask <question> — interrupt, get a grounded answer
        /overview voices — the two hosts
        """
        from ...audio.overview import control_overview
        return control_overview(tail or "")

    def _control_character(self, tail: str, *, chat_key: str = "") -> str:
        """Conversational story characters — talk to them. Owner-only.

        /character list — talkable characters
        /character talk <name> <question> — interview them
        /character add <name> [book] [--cutoff N] — add a character
        /character cutoff <name> <chapter> — move their knowledge cutoff
        /character forget <name> — remove a character
        """
        from ...audio.characters import control_character
        return control_character(tail or "")

    def _control_audiobook(self, tail: str, *, chat_key: str = "") -> str:
        """EPUB → audiobook, one click + store disclosure. Owner-only.

        /audiobook make <epub> [narrator=<ref>] [stores...] — build it
        /audiobook status — produced audiobooks
        """
        from ...audio.audiobook import control_audiobook
        return control_audiobook(tail or "")

    def _control_graph(self, tail: str, *, chat_key: str = "") -> str:
        """Live world-graph planning substrate. Owner-only.

        /graph add <type> <label> — node (person|project|commitment|asset|schedule|deadline)
        /graph link <from-id> <to-id> <type> — edge (depends_on|blocks|owned_by|due)
        /graph disrupt <id> [note] — mark disrupted, propagate to dependents
        /graph clear <id> — clear a disruption
        /graph show [id] — summary, or one node + dependents
        /graph breaks <id> — what breaks if this node fails
        /graph path — critical path (longest dependency chain)
        /graph sync — project memory into the graph (read-only)
        """
        from ...planning.graph import WorldGraph, control_graph
        g = getattr(self, "_world_graph", None)
        if g is None:
            g = WorldGraph()
            self._world_graph = g
        return control_graph(tail or "", graph=g)

    def _control_route(self, tail: str, *, chat_key: str = "") -> str:
        """Predict → Build → Solve routing pipeline. Owner-only.

        /route plan <stop1>; <stop2>; ... — optimal order + honest times
        /route add <label> [lat,lng] — save a stop
        /route stops — list saved stops
        /route record <from> > <to> <minutes> [cost_kobo] — teach the model
        /route stats — what the cost model has learned
        """
        from ...planning.route import RoutePlanner, control_route
        p = getattr(self, "_route_planner", None)
        if p is None:
            p = RoutePlanner()
            self._route_planner = p
        return control_route(tail or "", planner=p)

    def _control_eta(self, tail: str, *, chat_key: str = "") -> str:
        """Honest time estimates — bands, not points. Owner-only.

        /eta <task-type> [seg=min] ... — banded estimate
        /eta record <task> <predicted> <actual> — feed a real outcome
        /eta risk <task> <elapsed> — miss probability right now
        /eta stats [task] — history per task type
        """
        from ...planning.estimates import control_eta
        return control_eta(tail or "")

    def _control_congestion(self, tail: str, *, chat_key: str = "") -> str:
        """Multi-agent congestion prediction — owner-only.

        /congestion status | register <name> [kind] [capacity] |
        advise <resource> [agents=N] | predict <resource> |
        alternate <resource> <alternate>
        """
        from ...planning.congestion import control_congestion
        return control_congestion(tail or "")

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

    def _control_home(self, tail: str, *, chat_key: str = "") -> str:
        """Home digital twin. Owner-only.

        /home status            current snapshot
        /home what changed [h]  changes in the last h hours (default 12)
        /home unusual?          statistical anomalies
        """
        from ...integrations.digital_twin import HomeTwin
        twin = getattr(self, "_home_twin", None)
        if twin is None:
            twin = HomeTwin()
            self._home_twin = twin
        rest = (tail or "").strip().lower()
        if not rest or rest == "status":
            return twin.summary()
        if rest.startswith("what changed"):
            hours = 12.0
            parts = rest.split()
            if len(parts) > 2:
                try:
                    hours = float(parts[2])
                except ValueError:
                    pass
            import time as _t
            return twin.what_changed(_t.time() - hours * 3600)
        if rest in ("unusual?", "unusual", "anomalies"):
            anomalies = twin.anomalies()
            if not anomalies:
                return "nothing unusual — everything matches your home's normal patterns."
            lines = ["⚠️ unusual:"]
            for a in anomalies[:8]:
                lines.append(f"• {a.message}")
            return "\n".join(lines)
        return ("usage: /home status | /home what changed [hours] | "
                "/home unusual?")

    def _control_store(self, tail: str, *, chat_key: str = "") -> str:
        """Self-hosted storefronts (Medusa / WooCommerce). Owner-only.

        /store provision <business>   provision a Medusa store
        /store woo <business> <url>    provision a WooCommerce store
        /store list                   all stores + status
        /store catalog <id> <desc>    AI-generate a product catalog
        /store <id> <instruction>     conversational management
        """
        from ...commerce.medusa import StoreManager
        mgr = getattr(self, "_store_manager", None)
        if mgr is None:
            mgr = StoreManager()
            self._store_manager = mgr
        rest = (tail or "").strip()
        if not rest:
            return ("usage: /store provision <business> | /store woo "
                    "<business> <url> | /store list | /store catalog <id> "
                    "<description> | /store <id> <instruction>")
        low = rest.lower()
        if low == "list":
            stores = mgr.list()
            if not stores:
                return "no stores yet — /store provision <business>."
            lines = ["🏪 stores:"]
            for s in stores:
                lines.append(f"• {s.id}: {s.name} [{s.engine}] — {s.status}")
            return "\n".join(lines)
        if low.startswith("provision "):
            business = rest[len("provision "):].strip()
            try:
                store = mgr.provision_store(business)
            except Exception as exc:  # noqa: BLE001 — show the problem
                return f"provisioning failed: {exc}"
            return (f"🏪 store '{store.name}' → {store.status}\n"
                    f"API: {store.api_url}\nAdmin: {store.admin_url}")
        if low.startswith("woo "):
            parts = rest[4:].strip().split(None, 1)
            if len(parts) < 2:
                return "usage: /store woo <business> <site-url>"
            try:
                store = mgr.provision_woocommerce(parts[0], site_url=parts[1])
            except Exception as exc:  # noqa: BLE001
                return f"provisioning failed: {exc}"
            return (f"🏪 WooCommerce store '{store.name}' → {store.status}\n"
                    f"API: {store.api_url}")
        if low.startswith("catalog "):
            parts = rest[8:].strip().split(None, 1)
            if len(parts) < 2:
                return "usage: /store catalog <id> <business description>"
            store = mgr.get(parts[0])
            if store is None:
                return f"no store '{parts[0]}'. Use /store list."
            llm_fn = getattr(self, "_llm_fn", None)
            try:
                created = mgr.generate_catalog(store, parts[1], llm_fn=llm_fn)
            except Exception as exc:  # noqa: BLE001
                return f"catalog failed: {exc}"
            lines = [f"✅ {len(created)} products created:"]
            for p in created[:10]:
                lines.append(f"• {p['name']} — ₦{p['price_naira']:,}")
            return "\n".join(lines)
        # /store <id> <instruction>
        parts = rest.split(None, 1)
        if len(parts) < 2:
            return ("usage: /store <id> <instruction> — e.g. "
                    "'/store store_abc123 add 10% discount this weekend'")
        store = mgr.get(parts[0])
        if store is None:
            return f"no store '{parts[0]}'. Use /store list."
        result = mgr.manage(store, parts[1])
        return result.get("message", str(result))

    def _control_track(self, tail: str, *, chat_key: str = "") -> str:
        """Flight price watchers — persistent monitoring as a sentence.

        Owner-only.

        /track LOS LHR 2026-12-01 [under 400k]   watch a route
        /track list                              active watchers
        /untrack <watch_id>                      stop watching
        """
        from ...finance.ledger import format_naira
        from ...travel.watchers import PriceWatcher, parse_watch_request
        watcher = getattr(self, "_price_watcher", None)
        if watcher is None:
            watcher = PriceWatcher()
            self._price_watcher = watcher
        rest = (tail or "").strip()
        if not rest:
            return ("usage: /track LOS LHR 2026-12-01 [under 400k] | "
                    "/track list | /untrack <watch_id>")
        low = rest.lower()
        if low == "list":
            watches = watcher.list_watches()
            if not watches:
                return ("no active watchers — /track LOS LHR 2026-12-01 "
                        "to start one.")
            lines = ["✈️ watching:"]
            for w in watches:
                target = (f" (target {format_naira(w.target_kobo)})"
                          if w.target_kobo else "")
                last = (f" — last {format_naira(w.last_kobo)}"
                        if w.last_kobo else "")
                lines.append(f"• {w.id}: {w.route} {w.departure_date}"
                             f"{target}{last}")
            return "\n".join(lines)
        parsed = parse_watch_request(rest)
        if parsed is None or not parsed.get("departure_date"):
            return ("I need a route and date — e.g. "
                    "'/track LOS LHR 2026-12-01 under 400k'.")
        try:
            w = watcher.watch(
                parsed["origin"], parsed["destination"],
                parsed["departure_date"],
                target_kobo=parsed["target_kobo"])
        except ValueError as exc:
            return f"couldn't start the watch: {exc}"
        # immediate first check so the owner sees a price right away
        alert = None
        try:
            alert = watcher.check(w)
        except Exception:  # noqa: BLE001 - first check is best-effort
            pass
        msg = (f"👀 watching {w.route} on {w.departure_date}")
        if w.target_kobo:
            msg += f" — I'll ping you under {format_naira(w.target_kobo)}"
        msg += f" (id {w.id})."
        if alert is not None:
            msg += "\n" + alert.text
        elif w.last_kobo:
            msg += f" Current: {format_naira(w.last_kobo)}."
        return msg

    def _control_untrack(self, tail: str, *, chat_key: str = "") -> str:
        """Stop a price watcher. Owner-only. /untrack <watch_id>"""
        from ...travel.watchers import PriceWatcher
        watcher = getattr(self, "_price_watcher", None)
        if watcher is None:
            watcher = PriceWatcher()
            self._price_watcher = watcher
        wid = (tail or "").strip()
        if not wid:
            return "usage: /untrack <watch_id> — see /track list."
        if watcher.unwatch(wid):
            return f"stopped watching {wid}."
        return f"no watcher '{wid}'. Use /track list."

    def _control_trip(self, tail: str, *, chat_key: str = "") -> str:
        """Auto itineraries from forwarded booking confirmations. Owner-only.

        /trip                  list trips
        /trip <id>             itinerary summary
        /trip add <text>       parse a pasted confirmation
        /trip calendar <id>    sync to Google Calendar (confirmation-gated)
        /trip docs <id>        vault docs for a trip
        """
        from ...travel.itinerary import ItineraryBuilder, added_message
        builder = getattr(self, "_trip_builder", None)
        if builder is None:
            builder = ItineraryBuilder()
            self._trip_builder = builder
        rest = (tail or "").strip()
        if not rest:
            trips = builder.list_trips()
            if not trips:
                return ("no trips yet — forward a booking confirmation "
                        "(email, PDF, screenshot) and I'll build the "
                        "itinerary.\nusage: /trip <id> | /trip add <text> | "
                        "/trip calendar <id> | /trip docs <id>")
            lines = ["🧳 trips:"]
            for t in trips[:10]:
                n = (len(t.flights) + len(t.hotels) + len(t.cars))
                lines.append(f"  • {t.id} — {t.name or 'untitled'} "
                             f"({n} segment(s))")
            return "\n".join(lines)
        low = rest.lower()
        if low.startswith("add "):
            trip = builder.ingest_email(rest[4:].strip())
            if trip.is_empty():
                return ("couldn't parse a booking from that — I need a "
                        "flight number, PNR, or hotel name. Try forwarding "
                        "the full confirmation.")
            return added_message(trip) + f"\n(trip id: {trip.id})"
        if low.startswith("calendar "):
            tid = rest[9:].strip()
            gcal = self._resolve_gcal()
            if gcal is None:
                return ("Google Calendar isn't connected — connect it first, "
                        "then /trip calendar <id>.")
            created = builder.to_calendar(tid, gcal, confirmed=False)
            if not created:
                return (f"nothing to sync for {tid} — no dated segments, or "
                        "the trip doesn't exist.")
            return (f"📅 {len(created)} event(s) prepared for {tid} — "
                    "approve the calendar checkpoint to create them.")
        if low.startswith("docs "):
            tid = rest[5:].strip()
            docs = builder.vault_docs(tid)
            if not docs:
                return f"no docs for {tid}."
            return f"📎 {tid} vault:\n" + "\n".join(f"  • {d}" for d in docs)
        trip = builder.get_trip(rest.split()[0])
        if trip is None:
            return ("usage: /trip <id> | /trip add <text> | "
                    "/trip calendar <id> | /trip docs <id>")
        return builder.summary(trip.id)

    def _resolve_gcal(self) -> Any:
        """Connected GCalendarConnector or None.

        The instance is injected by the runtime setup (``self._gcal_instance``).
        Never raises; returns None when calendar isn't wired up so the
        handler reports honestly instead of faking events.
        """
        try:
            return getattr(self, "_gcal_instance", None)
        except Exception:  # noqa: BLE001
            return None

    def _control_mandate(self, tail: str, *, chat_key: str = "") -> str:
        """Payment mandates — the agent's standing authority to move money.

        Owner-only.

        /mandate issue <scope> <per-txn-₦> [per-day-₦] [days-valid]
        /mandate list
        /mandate revoke <id>
        /mandate stop all
        """
        from ...finance.ledger import format_naira, parse_amount
        from ...finance.mandate import MandateError, MandateStore
        store = getattr(self, "_mandate_store", None)
        if store is None:
            store = MandateStore()
            self._mandate_store = store
        rest = (tail or "").strip()
        if not rest:
            return ("usage: /mandate issue <scope> <per-txn-₦> [per-day-₦] "
                    "[days-valid] — e.g. /mandate issue transfer 50000 200000 30\n"
                    "       /mandate list | /mandate revoke <id> | "
                    "/mandate stop all")
        low = rest.lower()
        if low == "list":
            mandates = store.list("owner")
            if not mandates:
                return ("no mandates — money cannot move until you issue one:\n"
                        "/mandate issue transfer <per-txn-₦> [per-day-₦] [days]")
            lines = ["💳 payment mandates:"]
            for m in mandates:
                state = "revoked" if m.revoked else (
                    "expired" if m.expired else "active")
                lines.append(
                    f"• {m.id} [{m.scope}] {state} — "
                    f"{format_naira(m.cap_per_txn)}/txn, "
                    f"{format_naira(m.cap_per_day)}/day")
            return "\n".join(lines)
        if low == "stop all":
            n = store.revoke_all("owner")
            return (f"🛑 revoked {n} mandate(s) — all agent spending is "
                    f"stopped until you issue a new mandate.")
        if low.startswith("revoke "):
            mid = rest[len("revoke "):].strip()
            try:
                if store.revoke(mid):
                    return f"mandate {mid} revoked — effective immediately."
                return f"no mandate '{mid}'. Use /mandate list."
            except MandateError as exc:
                return f"cannot revoke: {exc}"
        if low.startswith("issue "):
            parts = rest[len("issue "):].split()
            if len(parts) < 2:
                return ("usage: /mandate issue <scope> <per-txn-₦> "
                        "[per-day-₦] [days-valid]")
            scope = parts[0].lower()
            per_txn = parse_amount(parts[1])
            per_day = parse_amount(parts[2]) if len(parts) > 2 else None
            try:
                days = float(parts[3]) if len(parts) > 3 else 30.0
            except ValueError:
                return f"bad days value: {parts[3]!r}"
            if per_txn is None or per_txn <= 0:
                return f"bad per-transaction cap: {parts[1]!r}"
            if per_day is None:
                per_day = per_txn * 4  # sensible default: 4× per-txn
            try:
                m = store.issue(principal="owner", scope=scope,
                                cap_per_txn=per_txn, cap_per_day=per_day,
                                ttl_days=days)
            except MandateError as exc:
                return f"cannot issue: {exc}"
            return (f"✅ mandate {m.id} issued: {m.scope} scope, "
                    f"{format_naira(m.cap_per_txn)}/txn, "
                    f"{format_naira(m.cap_per_day)}/day, "
                    f"valid {days:g} days.")
        return ("usage: /mandate issue <scope> <per-txn-₦> [per-day-₦] "
                "[days-valid] | /mandate list | /mandate revoke <id> | "
                "/mandate stop all")
