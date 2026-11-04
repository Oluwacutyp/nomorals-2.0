"""``/math`` command handlers — pure dicts, never sends.

Every handler returns a plain dict (``problem`` / ``result`` /
``reward`` / ``leaderboard`` / …). The chat layer decides how to render
and send it. Handlers take their stores injected (``MathClass``,
``MathEconomy``), so tests can drive them with fresh in-memory stores
and the bot can wire in the real ones.

Commands:
    /math                        — start a problem at your adaptive tier
    /math easy|medium|hard|expert|master|impossible
                                 — start a problem at that tier
    /math answer <text>         — grade your active problem
    /math leaderboard            — top solvers
    /math streak                 — your streak + adaptive tier
    /math daily                  — today's challenge
    /math duel <opponent>        — head-to-head duel with someone here
    /math duel                   — matchmaking: queue for a duel with
                                   the first compatible player waiting,
                                   on any platform
    /math duel open              — open a duel anyone can join with
                                   the short code it hands you
    /math duel cancel            — leave the queue / cancel your open duel
    /math join <code>            — take seat 2 of an open duel by code
    /math duel_answer <answer>   — answer the current round of the
                                   duel in front of you (held state —
                                   no ids are ever quoted back)
    /math set                    — timed set: 10 in 5 min
    /math teach <explanation>    — explain the problem in front of you
    /math upvote                 — upvote the explanation in front of you
    /math balance                — your math ledger balance + statement

Matching is state-held throughout (games-engine pass): the handler
remembers each member's open duel, current problem, teachable
problem and held explanation, so no member ever quotes an internal
id back. The old id-carrying grammars still parse, but they resolve
through the held state — an id that is not the member's held one
scores/teaches/votes nothing. Timed rounds settle at the first
graded answer or at expiry (§34): late answers get the scoreless
close with the answer and its working.

Wire points — hook functions at the bottom expose Math Class to the
rest of the system: quests (daily math quest), tournaments (math
season), pulses (daily math challenge), parties (math duel between
tracks), digest (top math solvers).
"""
from __future__ import annotations

import time
from typing import Any

from ..core.logging_setup import get_logger
from .math_class import TIERS, MathClass
from .math_economy import MATH_COINS_DISCLAIMER, MathEconomy
from .reward_ladder import describe_reward, pool_split, reward_for

_log = get_logger(__name__)

__all__ = [
    "MathCommandHandler", "parse_math_command",
    "daily_math_quest", "math_season", "daily_math_pulse",
    "math_party_duel", "digest_top_solvers", "math_tournament_payouts",
]

DIFFICULTY_ALIASES: dict[str, str] = {
    "e": "easy", "ez": "easy", "m": "medium", "med": "medium",
    "h": "hard", "x": "expert", "exp": "expert",
    "master": "master", "mast": "master",
    "impossible": "impossible", "imp": "impossible",
}

#: Matchmaking queue entries expire after this long unfilled (§13:
#: nobody waits forever — a stale spot is dropped, with a note).
QUEUE_TTL_S = 600.0
#: A waiting entry older than this pairs with ANY tier — the
#: tolerance widening the engine matchmaker uses, in math clothes.
QUEUE_WIDEN_S = 120.0
#: Failed /math join attempts tolerated per member per window —
#: join codes are short by design, so guessing must stay pointless.
JOIN_FAIL_LIMIT = 5
JOIN_FAIL_WINDOW_S = 600.0


def parse_math_command(text: str) -> tuple[str, list[str]]:
    """Split ``/math ...`` into (subcommand, args). Bare ``/math`` is
    ``("start", [])``; a lone difficulty is ``("start", [tier])``."""
    parts = (text or "").strip().split()
    if not parts:
        return "start", []
    if parts[0].lstrip("/").lower() == "math":
        parts = parts[1:]
    if not parts:
        return "start", []
    head = parts[0].lower()
    if head in TIERS or head in DIFFICULTY_ALIASES:
        return "start", [DIFFICULTY_ALIASES.get(head, head)]
    return head, parts[1:]


class MathCommandHandler:
    """Stateless-over-stores handler: all state lives in the injected
    ``MathClass`` / ``MathEconomy``."""

    def __init__(self, math_class: MathClass | None = None,
                 economy: MathEconomy | None = None) -> None:
        self.math = math_class if math_class is not None else MathClass()
        self.economy = economy if economy is not None else MathEconomy()
        self._active: dict[str, str] = {}  # player -> active problem_id
        self._tier_clears: dict[str, set[str]] = {}  # player -> tiers first-cleared
        # Timed-set tracking: player -> {"problems": [ids], "idx": int}.
        # One active track at a time — starting a fresh problem or the
        # daily challenge abandons the set (see _cmd_start / _cmd_daily).
        self._sets: dict[str, dict[str, Any]] = {}
        # ── state-held matching (games-engine pass) ──
        # The duel in front of each member: player -> duel_id. Set
        # when a duel starts (direct, queued, or by join code) for
        # BOTH seats; cleared when the duel settles.
        self._open_duels: dict[str, str] = {}
        # The problem in front of each member for /math teach: the
        # one they just started or just answered.
        self._last_problem: dict[str, str] = {}
        # The explanation in front of each member for /math upvote:
        # the newest explanation (by someone else) of the problem
        # they just answered. One-shot — consumed by the upvote.
        self._held_expl: dict[str, str] = {}
        # Matchmaking queue (§13): [{player, tier, at}] in FIFO
        # order, global to the handler — every platform it serves.
        self._duel_queue: list[dict[str, Any]] = []
        # /math join rate limiting: player -> [failure timestamps].
        self._join_fails: dict[str, list[float]] = {}
        # Held notices for a member's next interaction: round/duel
        # closes swept by the clock, a queue match that filled
        # while they were away. Delivered (once) by handle().
        self._pending_notes: dict[str, list[dict[str, Any]]] = {}

    # ── router ──
    def handle(self, text: str, player: str,
               now: float | None = None) -> dict[str, Any]:
        sub, args = parse_math_command(text)
        now = now if now is not None else time.time()
        fn = getattr(self, f"_cmd_{sub}", None)
        if fn is None:
            return {"ok": False, "error": "unknown_command",
                    "message": f"Unknown /math command {sub!r}. Try /math, "
                               f"/math easy|medium|hard|expert|master|impossible, "
                               f"/math leaderboard, /math streak, /math daily, "
                               f"/math duel <opponent>."}
        notes = self._pending_notes.pop(player, [])
        try:
            result = fn(player, args, now)
        except Exception as exc:  # noqa: BLE001
            _log.warning("math command %s failed: %s", sub, exc)
            result = {"ok": False, "error": "internal",
                      "message": str(exc)}
        if notes and isinstance(result, dict):
            result["notes"] = notes
        return result

    # ── held-state helpers ──
    def _note(self, player: str, note: dict[str, Any]) -> None:
        """Queue a held notice for a member's next interaction."""
        self._pending_notes.setdefault(player, []).append(note)

    def _held_duel(self, player: str):
        """The member's open duel object, or None (stale holds are
        cleared as they're found)."""
        duel_id = self._open_duels.get(player)
        if not duel_id:
            return None
        duel = self.math.store["duels"].get(duel_id)
        if duel is None or duel.status == "finished":
            self._open_duels.pop(player, None)
            return None
        return duel

    def _clear_duel_holds(self, duel: Any) -> None:
        for p in duel.players:
            if self._open_duels.get(p) == duel.duel_id:
                self._open_duels.pop(p, None)

    # ── §34 sweep: post round/duel closes as held notices ──
    def sweep_rounds(self, now: float | None = None) -> list[dict[str, Any]]:
        """Settle every expired timed round/duel (MathClass sweep)
        and queue the closes as held notices for the members they
        belong to. Also expires stale matchmaking queue spots.
        Returns the close records."""
        now = now if now is not None else time.time()
        alive = []
        for entry in self._duel_queue:
            if now - float(entry.get("at") or now) > QUEUE_TTL_S:
                self._note(entry["player"], {"kind": "queue_expired"})
            else:
                alive.append(entry)
        self._duel_queue = alive
        closes = self.math.sweep_expired(now)
        for record in closes:
            kind = record.get("kind")
            if kind in ("round_close", "set_close"):
                self._note(record["player"], record)
            elif kind in ("duel_close", "duel_invite_expired"):
                for p in record.get("players") or []:
                    self._note(p, record)
                    self._open_duels.pop(p, None)
        return closes

    # ── commands ──
    def _cmd_start(self, player: str, args: list[str],
                   now: float) -> dict[str, Any]:
        tier = args[0] if args else ""
        allowed, reason, wait = self.math.attempt_allowed(
            player, tier or self.math.adaptive_tier(player), now)
        if not allowed:
            return {"ok": False, "error": reason, "wait_s": round(wait, 1),
                    "message": ("Cool down first." if reason == "cooldown"
                                else "Daily attempt cap for this tier reached.")}
        problem = self.math.generate(player, tier, "quickfire", now=now)
        self._active[player] = problem.problem_id
        self._last_problem[player] = problem.problem_id
        # A fresh problem abandons any in-progress timed set.
        self._sets.pop(player, None)
        return {"ok": True, "kind": "problem",
                "problem": problem.public(),
                "adaptive_tier": self.math.adaptive_tier(player),
                "hint": "Answer with /math answer <your answer>."}

    def _cmd_answer(self, player: str, args: list[str],
                    now: float) -> dict[str, Any]:
        if not args:
            return {"ok": False, "error": "no_answer",
                    "message": "Usage: /math answer <your answer>"}
        problem_id = self._active.get(player)
        if problem_id is None:
            return {"ok": False, "error": "no_active_problem",
                    "message": "No active problem — start one with /math."}
        submitted = " ".join(args)
        res = self.math.submit(player, problem_id, submitted, now=now)
        if not res.get("ok"):
            return res
        if res.get("closed"):
            # §34 close: the round settled (expiry, or an earlier
            # answer) — this answer scores nothing. The reply IS
            # the close: answer + the generator's own working.
            if self._active.get(player) == problem_id:
                self._active.pop(player, None)
            set_state = self._sets.get(player)
            if set_state and problem_id in set_state.get("problems", []):
                self._sets.pop(player, None)
            return {"ok": True, "kind": "result", "closed": True,
                    "expired": res.get("close_reason") == "expired",
                    "correct": False, "tier": res["tier"],
                    "answer": res["answer"],
                    "solution": res["solution"],
                    "solution_steps": res.get("solution_steps", []),
                    "reward": {"coins": 0, "xp": 0, "paid": False},
                    "reward_text": ""}
        # The problem just answered is the one in front of the
        # member now — for /math teach, and for the explanation it
        # may carry (held for /math upvote).
        self._last_problem[player] = problem_id
        out: dict[str, Any] = {"ok": True, "kind": "result",
                               "correct": res["correct"],
                               "tier": res["tier"],
                               "answer": res["answer"],
                               "solution": res["solution"],
                               "solution_steps": res.get("solution_steps", []),
                               "adaptive": res.get("adaptive", {}),
                               "streak": res.get("streak", 0)}
        expl = self.economy.latest_explanation_for(problem_id,
                                                   exclude=player)
        if expl is not None:
            self._held_expl[player] = str(expl.get("id") or "")
            out["held_explanation"] = {
                "author": expl.get("author", ""),
                "text": str(expl.get("text") or "")[:200]}
        else:
            self._held_expl.pop(player, None)
        # Timed-set advance: answering the current set problem moves the
        # active pointer to the next one (or closes the set).
        set_state = self._sets.get(player)
        if set_state and problem_id in set_state.get("problems", []):
            problems = list(set_state["problems"])
            idx = problems.index(problem_id) + 1
            if idx < len(problems):
                set_state["idx"] = idx
                next_id = problems[idx]
                self._active[player] = next_id
                self._last_problem[player] = next_id
                next_problem = self.math.store["active"].get(next_id)
                out["set_progress"] = f"{idx + 1}/{len(problems)}"
                if next_problem is not None:
                    out["next_problem"] = next_problem.public()
            else:
                self._sets.pop(player, None)
                self._active.pop(player, None)
                out["set_complete"] = True
        if res["correct"]:
            reward = self._pay(player, res["tier"], now)
            out["reward"] = reward
            out["reward_text"] = (describe_reward(reward)
                                  if reward.get("paid") else
                                  "Reward blocked (unverified member or earn cap).")
        else:
            out["reward"] = {"coins": 0, "xp": 0, "paid": False}
            out["reward_text"] = "Wrong answer — no reward. Read the worked solution and try again."
        return out

    def _cmd_leaderboard(self, player: str, args: list[str],
                         now: float) -> dict[str, Any]:
        limit = 10
        try:
            limit = max(1, min(50, int(args[0]))) if args else 10
        except ValueError:
            limit = 10
        return {"ok": True, "kind": "leaderboard",
                "leaders": self.math.leaderboard(limit),
                "you": next((r for r in self.math.leaderboard(1000)
                             if r["player"] == player), None)}

    def _cmd_streak(self, player: str, args: list[str],
                    now: float) -> dict[str, Any]:
        state = self.math._player_state(player)
        return {"ok": True, "kind": "streak",
                "player": player, "streak": self.math.store["streaks"].get(player, 0),
                "best_streak": self.math.store["scores"].get(player, {}).get("best_streak", 0),
                "adaptive_tier": state["tier"], "wins": state["wins"],
                "losses": state["losses"],
                "next": ("Win 3 in a row to tier up; lose 2 in a row to tier down.")}

    def _cmd_daily(self, player: str, args: list[str],
                   now: float) -> dict[str, Any]:
        date_iso = args[0] if args else time.strftime("%Y-%m-%d", time.gmtime(now))
        problem = self.math.daily_challenge(date_iso)
        self._active[player] = problem.problem_id
        self._last_problem[player] = problem.problem_id
        # The daily challenge is its own track — it abandons a set.
        self._sets.pop(player, None)
        # Daily problems are not in the active store (same for everyone),
        # so register this instance for grading.
        self.math.store["active"][problem.problem_id] = problem
        return {"ok": True, "kind": "problem", "daily": True,
                "problem": problem.public()}

    def _duel_started(self, duel: Any, *, matched: bool = False,
                      joined: bool = False) -> dict[str, Any]:
        """The duel-start result (internal duel_id included for
        state/journal — renders never print it)."""
        return {"ok": True, "kind": "duel", "duel_id": duel.duel_id,
                "players": list(duel.players),
                "matched": matched, "joined": joined,
                "problems": [p.public() for p in duel.problems],
                "message": "Same problems for both players — first "
                           "correct answer wins each round."}

    def _cmd_duel(self, player: str, args: list[str],
                  now: float) -> dict[str, Any]:
        head = args[0].lower() if args else ""
        if head in ("open", "code"):
            return self._duel_open(player,
                                   args[1] if len(args) > 1 else "", now)
        if head == "cancel":
            return self._duel_cancel(player, now)
        if head == "queue":
            return self._duel_enqueue(player,
                                    args[1] if len(args) > 1 else "", now)
        if not args:
            return self._duel_enqueue(player, "", now)
        opponent = args[0].lstrip("@")
        if opponent == player:
            return {"ok": False, "error": "self_duel",
                    "message": "You cannot duel yourself."}
        if self._held_duel(player) is not None:
            return {"ok": False, "error": "already_in_duel",
                    "message": "You're already in a duel — answer it "
                               "with /math duel_answer <answer>."}
        tier = args[1] if len(args) > 1 else ""
        duel = self.math.create_duel(player, opponent, tier, now=now)
        self._open_duels[player] = duel.duel_id
        self._open_duels[opponent] = duel.duel_id
        return self._duel_started(duel)

    def _duel_open(self, player: str, tier: str,
                   now: float) -> dict[str, Any]:
        """Open a coded duel: anyone, on any platform, takes seat 2
        with /math join <code>."""
        held = self._held_duel(player)
        if held is not None:
            if held.status == "waiting":
                return {"ok": True, "kind": "duel_open",
                        "duel_id": held.duel_id,
                        "join_code": held.join_code,
                        "players": list(held.players),
                        "problems": [p.public() for p in held.problems],
                        "message": "Your open duel is still waiting — "
                                   "share the code."}
            return {"ok": False, "error": "already_in_duel",
                    "message": "You're already in a duel — answer it "
                               "with /math duel_answer <answer>."}
        duel = self.math.create_open_duel(player, tier, now=now)
        self._open_duels[player] = duel.duel_id
        return {"ok": True, "kind": "duel_open",
                "duel_id": duel.duel_id, "join_code": duel.join_code,
                "players": list(duel.players),
                "problems": [p.public() for p in duel.problems],
                "message": "Open duel — share the code; the clock "
                           "starts when someone joins."}

    def _duel_cancel(self, player: str, now: float) -> dict[str, Any]:
        before = len(self._duel_queue)
        self._duel_queue = [e for e in self._duel_queue
                            if e["player"] != player]
        did = []
        if len(self._duel_queue) < before:
            did.append("left the duel queue")
        held = self._held_duel(player)
        if held is not None and held.status == "waiting":
            held.status = "finished"
            held.join_code = ""
            self._clear_duel_holds(held)
            did.append("open duel cancelled")
        return {"ok": True, "kind": "duel_cancel",
                "message": (" · ".join(did) + ".") if did
                else "Nothing to cancel — no queue spot, no open duel."}

    def _duel_enqueue(self, player: str, tier: str,
                    now: float) -> dict[str, Any]:
        """§13 matchmaking for math duels: the global queue pairs
        the first compatible waiters — same tier, or any tier once
        a waiter has been patient (QUEUE_WIDEN_S). Player keys are
        platform-scoped, so the queue pairs strangers across every
        platform the handler serves."""
        held = self._held_duel(player)
        if held is not None:
            if held.status == "waiting":
                return {"ok": True, "kind": "duel_open",
                        "duel_id": held.duel_id,
                        "join_code": held.join_code,
                        "players": list(held.players),
                        "problems": [p.public() for p in held.problems],
                        "message": "Your open duel is still waiting — "
                                   "share the code."}
            return {"ok": False, "error": "already_in_duel",
                    "message": "You're already in a duel — answer it "
                               "with /math duel_answer <answer>."}
        tier = tier.strip().lower()
        if tier in DIFFICULTY_ALIASES:
            tier = DIFFICULTY_ALIASES[tier]
        if tier and tier not in TIERS:
            tier = ""
        mine = tier or self.math.adaptive_tier(player)
        my_entry = next((e for e in self._duel_queue
                         if e["player"] == player), None)
        my_wait = (now - float(my_entry.get("at") or now)
                   if my_entry else 0.0)
        for entry in list(self._duel_queue):
            waiter = entry["player"]
            if waiter == player:
                continue
            waited = now - float(entry.get("at") or now)
            if entry["tier"] == mine or waited > QUEUE_WIDEN_S \
                    or my_wait > QUEUE_WIDEN_S:
                self._duel_queue.remove(entry)
                if my_entry is not None:
                    self._duel_queue.remove(my_entry)
                duel = self.math.create_duel(waiter, player,
                                             entry["tier"], now=now)
                self._open_duels[waiter] = duel.duel_id
                self._open_duels[player] = duel.duel_id
                self._note(waiter, {
                    "kind": "duel_filled", "opponent": player,
                    "problems": [p.public() for p in duel.problems]})
                return self._duel_started(duel, matched=True)
        if my_entry is not None:
            return {"ok": True, "kind": "duel_queue",
                    "tier": my_entry["tier"],
                    "message": "You're in the duel queue — first "
                               "compatible player pairs with you."}
        self._duel_queue.append({"player": player, "tier": mine,
                                 "at": now})
        return {"ok": True, "kind": "duel_queue", "tier": mine,
                "message": "You're in the duel queue — first "
                           "compatible player pairs with you, on "
                           "any platform."}

    def _cmd_join(self, player: str, args: list[str],
                  now: float) -> dict[str, Any]:
        """Take seat 2 of an open duel by its short join code."""
        if not args:
            return {"ok": False, "error": "usage",
                    "message": "Usage: /math join <code> — the code "
                               "from an open duel post."}
        fails = [t for t in self._join_fails.get(player, [])
                 if now - t < JOIN_FAIL_WINDOW_S]
        self._join_fails[player] = fails
        if len(fails) >= JOIN_FAIL_LIMIT:
            return {"ok": False, "error": "too_many_attempts",
                    "message": "Too many wrong codes — give it a few "
                               "minutes, then try the real one."}
        res = self.math.join_open_duel(args[0], player, now=now)
        if not res.get("ok"):
            fails.append(now)
            self._join_fails[player] = fails
            msg = {"bad_code": "That code doesn't match an open duel.",
                   "code_expired": "That duel's code has expired — "
                                   "ask for a fresh one.",
                   "own_duel": "That's your own duel — share the "
                               "code with someone else."}.get(
                        str(res.get("error")), "Couldn't join that duel.")
            return {"ok": False, "error": res.get("error"), "message": msg}
        self._join_fails.pop(player, None)
        duel = res["duel"]
        creator = duel.players[0]
        self._open_duels[player] = duel.duel_id
        self._open_duels[creator] = duel.duel_id
        self._note(creator, {
            "kind": "duel_filled", "opponent": player,
            "problems": [p.public() for p in duel.problems]})
        return self._duel_started(duel, joined=True)

    def _cmd_duel_answer(self, player: str, args: list[str],
                         now: float) -> dict[str, Any]:
        """Answer the current round of the duel in front of you.

        State-held: the duel and the round come from the handler's
        held state, never from the message. The legacy grammar
        (``duel_answer <duel_id> <round> <answer>``) still parses
        but resolves through the held duel — a quoted id that is
        not the member's held duel scores nothing."""
        if not args:
            return {"ok": False, "error": "usage",
                    "message": "Usage: /math duel_answer <your answer>"}
        round_idx: int | None = None
        if args[0].startswith("duel_"):
            claimed = args[0]
            rest = args[1:]
            if len(rest) >= 2:
                try:
                    round_idx = int(rest[0])
                    rest = rest[1:]
                except ValueError:
                    round_idx = None
            held = self._held_duel(player)
            if held is None or claimed != held.duel_id:
                return {"ok": False, "error": "stale_duel",
                        "message": "That duel isn't the one in front "
                                   "of you — your open duel answers "
                                   "with /math duel_answer <answer>."}
            if not rest:
                return {"ok": False, "error": "usage",
                        "message": "Usage: /math duel_answer <your answer>"}
            args = rest
        duel = self._held_duel(player)
        if duel is None:
            return {"ok": False, "error": "no_open_duel",
                    "message": "No duel in front of you — start one "
                               "with /math duel, /math duel "
                               "<opponent>, or /math duel open."}
        if round_idx is None:
            round_idx = duel.current_round()
        res = self.math.duel_answer(duel.duel_id, player, round_idx,
                                    " ".join(args), now=now)
        if res.get("status") == "finished" or res.get("closed"):
            self._clear_duel_holds(duel)
        if res.get("ok") and res.get("correct"):
            reward = self._pay(player, res.get("tier", "medium"), now,
                               reason="duel round win")
            res["reward"] = reward
        return res

    def _cmd_set(self, player: str, args: list[str],
                 now: float) -> dict[str, Any]:
        tier = args[0] if args and args[0] in TIERS else ""
        result = self.math.timed_set(player, tier, now=now)
        problems = result.get("problems") or []
        if problems:
            # Track the set so /math answer advances through it instead
            # of grading only the first problem and stranding the rest.
            self._sets[player] = {
                "problems": [p.get("problem_id", "") for p in problems],
                "idx": 0,
            }
            self._active[player] = problems[0].get("problem_id", "")
            self._last_problem[player] = problems[0].get("problem_id", "")
            result["set_progress"] = f"1/{len(problems)}"
        return {"ok": True, "kind": "timed_set", **result}

    def _teach_target(self, player: str) -> str:
        """The problem in front of the member: the one they just
        started or just answered."""
        return self._last_problem.get(player) \
            or self._active.get(player) or ""

    def _cmd_teach(self, player: str, args: list[str],
                   now: float) -> dict[str, Any]:
        """Teach the problem in front of you. The legacy grammar
        (``teach <problem_id> <text>``) still parses but resolves
        through the held problem — a quoted id that is not the
        member's held problem teaches nothing."""
        if not args:
            return {"ok": False, "error": "usage",
                    "message": "Usage: /math teach <your explanation> "
                               "— it attaches to the problem in "
                               "front of you."}
        target = self._teach_target(player)
        if args[0] in self.math.store["active"] \
                or args[0].startswith("math_"):
            if args[0] != target:
                return {"ok": False, "error": "stale_problem",
                        "message": "That problem isn't the one in "
                                   "front of you — explanations "
                                   "attach to your current problem."}
            args = args[1:]
            if not args:
                return {"ok": False, "error": "usage",
                        "message": "Usage: /math teach <your explanation>"}
        if not target:
            return {"ok": False, "error": "no_problem",
                    "message": "No problem in front of you — start "
                               "one with /math, then teach it."}
        return {"ok": True, "kind": "teaching",
                **self.economy.explain_solution(player, target,
                                                " ".join(args))}

    def _cmd_upvote(self, player: str, args: list[str],
                    now: float) -> dict[str, Any]:
        """Upvote the explanation in front of you (held when you
        answered its problem). The legacy grammar (``upvote
        <explanation_id>``) still parses but resolves through the
        held explanation — a quoted id that is not the member's
        held one votes nothing. One-shot: the hold is consumed."""
        held = self._held_expl.get(player) or ""
        if args:
            if not held or args[0] != held:
                return {"ok": False, "error": "stale_explanation",
                        "message": "That explanation isn't the one "
                                   "in front of you — answer a "
                                   "problem someone explained, then "
                                   "/math upvote."}
        if not held:
            return {"ok": False, "error": "no_held_explanation",
                    "message": "No explanation in front of you — "
                               "answer a problem someone has "
                               "explained, then /math upvote."}
        res = self.economy.upvote_explanation(held, player, now=now)
        if res.get("ok") or res.get("error") == "unknown_explanation":
            self._held_expl.pop(player, None)
        return {"ok": True, "kind": "upvote", **res}

    def _cmd_balance(self, player: str, args: list[str],
                     now: float) -> dict[str, Any]:
        return {"ok": True, "kind": "balance",
                **self.economy.statement(player)}

    # ── payout: reward is computed from the ladder, then credited ──
    def _pay(self, player: str, tier: str, now: float,
             reason: str = "") -> dict[str, Any]:
        """Compute the ladder reward and credit it. First-clear is
        tracked per player per tier via the economy's lesson clears
        is NOT reused here — tier first-clear is a separate one-off,
        tracked on the handler's own set."""
        first_clear = tier not in self._tier_clears.get(player, set())
        reward = reward_for(tier, first_clear=first_clear)
        entry = self.economy.credit(
            player, reward["coins"], reward["xp"],
            reason or f"math: solved {tier} problem",
            ref=f"math:{tier}", difficulty=tier, now=now)
        if entry is None:
            reward["paid"] = False
            reward["ledger_id"] = None
            return reward
        reward["paid"] = True
        reward["ledger_id"] = entry.id
        if first_clear:
            self._tier_clears.setdefault(player, set()).add(tier)
        return reward


# ── wire points: Math Class into the rest of the system ───────────────────

def daily_math_quest(handler: MathCommandHandler, player: str,
                     now: float | None = None) -> dict[str, Any]:
    """Quest hook: today's daily math quest — solve the daily challenge
    at your adaptive tier. Returns the quest spec (the quest system
    claims/completes it; this module only defines it)."""
    now = now if now is not None else time.time()
    date_iso = time.strftime("%Y-%m-%d", time.gmtime(now))
    problem = handler.math.daily_challenge(date_iso)
    reward = reward_for(problem.tier, first_clear=False)
    return {"quest_id": f"math_daily_{date_iso}", "name": "Daily Math",
            "blurb": f"Solve today's {problem.tier} challenge.",
            "period": "daily", "tier": problem.tier,
            "problem": problem.public(),
            "reward": {"coins": reward["coins"], "xp": reward["xp"]},
            "disclaimer": MATH_COINS_DISCLAIMER}


def math_season(handler: MathCommandHandler, name: str = "Math Season",
                weeks: int = 4) -> dict[str, Any]:
    """Tournament hook: a math season — weekly boards, season-end pool
    split by placement (see :func:`pool_split`)."""
    return {"season": name, "weeks": weeks,
            "format": "weekly leaderboard, season-end pool split",
            "tiers": list(TIERS),
            "pool_split_note": ("Season pool splits by placement via "
                                "reward_ladder.pool_split — 40/25/15/12/8 for 1st–5th."),
            "leaders": handler.math.leaderboard(10)}


def daily_math_pulse(handler: MathCommandHandler,
                     now: float | None = None) -> dict[str, Any]:
    """Pulse hook: the daily math challenge for the morning pulse —
    one problem, one reward, same for everyone that day."""
    now = now if now is not None else time.time()
    date_iso = time.strftime("%Y-%m-%d", time.gmtime(now))
    problem = handler.math.daily_challenge(date_iso)
    return {"kind": "daily_math_pulse", "date": date_iso,
            "problem": problem.public(),
            "cta": "Solve it with /math daily — first correct answers climb the board."}


def math_party_duel(handler: MathCommandHandler, player_a: str,
                    player_b: str, tier: str = "") -> dict[str, Any]:
    """Party hook: a math duel between two party tracks/players.
    Same problems, first correct wins each round."""
    duel = handler.math.create_duel(player_a, player_b, tier)
    return {"kind": "party_math_duel", "duel_id": duel.duel_id,
            "players": duel.players,
            "problems": [p.public() for p in duel.problems]}


def digest_top_solvers(handler: MathCommandHandler,
                       limit: int = 5) -> dict[str, Any]:
    """Digest hook: top math solvers for the group digest."""
    leaders = handler.math.leaderboard(limit)
    lines = [f"{i + 1}. {r['player']} — {r['wins']} wins, "
             f"best streak {r['best_streak']}"
             for i, r in enumerate(leaders)]
    return {"kind": "math_digest", "title": "🧮 Top Math Solvers",
            "leaders": leaders,
            "text": "\n".join(lines) if lines else "No solvers yet — be the first with /math."}


def math_tournament_payouts(pool: int, placements: int) -> list[dict[str, Any]]:
    """Tournament hook: split a math tournament pool by placement."""
    amounts = pool_split(pool, placements)
    return [{"placement": i + 1, "coins": amt,
             "note": "Game currency only — " + MATH_COINS_DISCLAIMER}
            for i, amt in enumerate(amounts)]
