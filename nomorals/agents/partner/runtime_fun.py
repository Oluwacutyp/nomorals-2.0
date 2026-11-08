"""RuntimeFunMixin: the fun one-shots (FEI gap wave D) in the
live event path.

The organ itself (``social/chat/fun.py``) is transport-free;
this mixin owns the live wiring:

- **Commands**: /joke /dadjoke /progjoke /chuck /fact /fact2
  /catfact /numfact /advice /quote /activity /history (text
  from health-checked source chains), /dog /cat /fox /meme
  /nasa /pokemon (the picture ATTACHED through the expression
  send path, caption riding the media message — one message,
  so success returns "" and the kinds ride
  expression.MEDIA_DELIVERED_KINDS), /riddle (question posts,
  answer held per room by the service, ``/riddle answer``
  reveals), /8ball /choose (the local code rung), and /calc.
- **/calc reuses the games organ's evaluator**
  (``games.trivia_forge.safe_eval_arith`` — the same checked
  arithmetic path /math verifies with).  It is never a raw
  ``eval``: names, calls and attributes are not computable
  there, and its None is this command's honest error.
- **Gate**: in groups these are bot-system output, so they
  pass the same §33 gate as the expression commands beside
  them (``_group_gate``); DMs are never gated.
- The mixin never breaks its host: the service never raises,
  and the calc path converts every failure into its honest
  line.
"""

from __future__ import annotations

import logging
import os

from .failures import record_failure

_log = logging.getLogger(__name__)


class RuntimeFunMixin:
    """Fun one-shot commands for PartnerRuntime."""

    # ---- the service -------------------------------------------
    def _fun_service(self):  # type: ignore[no-untyped-def]
        service = getattr(self, "_fun_service_obj", None)
        if service is not None:
            return service
        from ...social.chat.fun import FunService

        settings = getattr(self.context, "settings", None)
        nasa_key = str(getattr(settings, "nasa_api_key", "")
                       or os.environ.get("NM_NASA_API_KEY", "") or "")
        service = FunService(
            media_dir=self._expression_root() / "fun_media",
            nasa_api_key=nasa_key)
        self._fun_service_obj = service
        return service

    # ---- the one handler ----------------------------------------
    def _control_fun(self, kind: str, tail: str, message=None) -> str:  # type: ignore[no-untyped-def]
        """One fun one-shot.  Text kinds return their text;
        image kinds return "" after a confirmed send (the
        media WAS the reply); everything else is one honest
        line — a dead source chain, an unreadable date, an
        expression arithmetic can't compute."""
        kind = str(kind or "").strip().lower()
        tail = str(tail or "").strip()
        if message is not None and not self._group_gate(message, kind):
            return ("this room isn't claimed for my bot systems "
                    "(§33) — my fun one-shots only play where my "
                    "bot runs.")
        if kind == "calc":
            return self._fun_calc(tail)
        if kind == "8ball":
            from ...social.chat.fun import eight_ball_answer

            service = self._fun_service()
            return f"🎱 {eight_ball_answer(service.rng)}"
        if kind == "choose":
            from ...social.chat.fun import choose_pick, split_choices

            service = self._fun_service()
            pick = choose_pick(tail, service.rng)
            if pick is None:
                if len(split_choices(tail)) == 1:
                    return (f"only one option there — "
                            f"{split_choices(tail)[0]} it is.")
                return ("usage: /choose <this> or <that> — "
                        "give me at least two options.")
            return f"🎯 {pick}"
        if kind == "riddle":
            return self._fun_riddle(tail, message)
        service = self._fun_service()
        if kind == "numfact" and tail and not tail.lstrip("-").isdigit():
            return (f"usage: /numfact [number] — “{tail}” isn't "
                    f"a number I can look up.")
        from ...social.chat.fun import FUN_IMAGE_KINDS

        if kind in FUN_IMAGE_KINDS:
            if message is None:
                return (f"usage: /{kind} in a chat — the picture "
                        f"needs somewhere to land.")
            out = service.image(
                kind, tail,
                send=lambda ref, caption: self._expression_send(
                    message, ref, caption=caption))
            if out.get("sent"):
                return ""
            if out.get("text"):
                return str(out["text"])
            return str(out.get("reason") or f"no /{kind} right now")
        out = service.text(kind, tail)
        if out.get("ok"):
            return str(out.get("text") or "")
        return str(out.get("reason") or f"no /{kind} right now")

    # ---- /riddle --------------------------------------------------
    def _fun_riddle(self, tail: str, message=None):  # type: ignore[no-untyped-def]
        """The held-answer game, rendered as ONE live session
        message (presentation architecture): bare /riddle posts
        the question WITH its Reveal/Another buttons declared —
        no printed "(/riddle answer …)" hint — and the reveal
        render EDITS that same message in place (session
        "riddle", session_edit) instead of posting a second
        "the answer was:" message. A fresh typed /riddle posts
        a new live message; a tapped Another edits the tapped
        one. The service still holds the answer per room;
        /riddle answer typed or tapped reveals (and clears)
        it exactly as before."""
        from ...social.chat.tgbot_buttons import actionable

        if message is None:
            return ("usage: /riddle in a chat — the answer is "
                    "held per room.")
        room = str(message.chat.key or "")
        service = self._fun_service()
        if tail.lower() in ("answer", "reveal", "solve"):
            out = service.riddle_answer(room)
            if not out.get("ok"):
                return actionable(
                    "no riddle is open in this chat.",
                    [[("🧩 New riddle", "riddle")]])
            return actionable(
                f"🧩 {out['question']}\n"
                f"💡 the answer: {out['answer']}",
                [[("🔀 Another riddle", "riddle")]],
                session="riddle", session_edit=True)
        try:
            out = service.new_riddle(room)
        except Exception as exc:  # noqa: BLE001 - the game never breaks the pipeline
            record_failure("fun.riddle", exc)
            return "my riddle sources stumbled — try again in a bit."
        if not out.get("ok"):
            return str(out.get("reason") or "no riddle right now")
        return actionable(
            f"🧩 {out['question']}",
            [[("💡 Reveal answer", "riddle answer"),
              ("🔀 Another riddle", "riddle")]],
            session="riddle")

    # ---- /calc ------------------------------------------------------
    def _fun_calc(self, tail: str) -> str:  # type: ignore[no-untyped-def]
        """Arithmetic through the games organ's own evaluator —
        the independent computation path /math verifies with.
        Its refusals are this command's errors, verbatim in
        spirit: what it can't compute, /calc can't either."""
        expr = str(tail or "").strip()
        if not expr:
            return ("usage: /calc expression — e.g. /calc "
                    "12 * (3 + 4) / 2")
        from ...games.trivia_forge import safe_eval_arith

        try:
            value = safe_eval_arith(expr)
        except Exception as exc:  # noqa: BLE001 - the evaluator is total, but never trust
            record_failure("fun.calc", exc)
            value = None
        if value is None:
            return (f"I can't compute “{expr}” — I do arithmetic: "
                    f"numbers with + − × ÷, brackets, % and powers.")
        if value == int(value) and abs(value) < 1e16:
            shown = str(int(value))
        else:
            shown = f"{value:.10f}".rstrip("0").rstrip(".")
        return f"{expr} = {shown}"
