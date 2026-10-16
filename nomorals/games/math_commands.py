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
    /math duel <opponent>        — head-to-head duel
    /math duel_answer <id> <round> <answer>
                                 — answer one duel round
    /math set                    — timed set: 10 in 5 min
    /math teach <problem_id> <explanation>
                                 — post a solution explanation
    /math upvote <explanation_id>
                                 — upvote an explanation
    /math balance                — your math ledger balance + statement

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
        try:
            return fn(player, args, now)
        except Exception as exc:  # noqa: BLE001
            _log.warning("math command %s failed: %s", sub, exc)
            return {"ok": False, "error": "internal", "message": str(exc)}

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
        out: dict[str, Any] = {"ok": True, "kind": "result",
                               "correct": res["correct"],
                               "tier": res["tier"],
                               "answer": res["answer"],
                               "solution": res["solution"],
                               "solution_steps": res.get("solution_steps", []),
                               "adaptive": res.get("adaptive", {}),
                               "streak": res.get("streak", 0)}
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
        # Daily problems are not in the active store (same for everyone),
        # so register this instance for grading.
        self.math.store["active"][problem.problem_id] = problem
        return {"ok": True, "kind": "problem", "daily": True,
                "problem": problem.public()}

    def _cmd_duel(self, player: str, args: list[str],
                  now: float) -> dict[str, Any]:
        if not args:
            return {"ok": False, "error": "no_opponent",
                    "message": "Usage: /math duel <opponent>"}
        opponent = args[0].lstrip("@")
        if opponent == player:
            return {"ok": False, "error": "self_duel",
                    "message": "You cannot duel yourself."}
        tier = args[1] if len(args) > 1 else ""
        duel = self.math.create_duel(player, opponent, tier, now=now)
        return {"ok": True, "kind": "duel", "duel_id": duel.duel_id,
                "players": duel.players,
                "problems": [p.public() for p in duel.problems],
                "message": "Same problems for both players — first correct answer wins each round."}

    def _cmd_duel_answer(self, player: str, args: list[str],
                         now: float) -> dict[str, Any]:
        if len(args) < 3:
            return {"ok": False, "error": "usage",
                    "message": "Usage: /math duel_answer <duel_id> <round> <answer>"}
        try:
            round_idx = int(args[1])
        except ValueError:
            return {"ok": False, "error": "bad_round"}
        res = self.math.duel_answer(args[0], player, round_idx, " ".join(args[2:]), now=now)
        if res.get("ok") and res.get("correct"):
            reward = self._pay(player, res.get("tier", "medium"), now, reason="duel round win")
            res["reward"] = reward
        return res

    def _cmd_set(self, player: str, args: list[str],
                 now: float) -> dict[str, Any]:
        tier = args[0] if args and args[0] in TIERS else ""
        return {"ok": True, "kind": "timed_set",
                **self.math.timed_set(player, tier, now=now)}

    def _cmd_teach(self, player: str, args: list[str],
                   now: float) -> dict[str, Any]:
        if len(args) < 2:
            return {"ok": False, "error": "usage",
                    "message": "Usage: /math teach <problem_id> <explanation>"}
        return {"ok": True, "kind": "teaching",
                **self.economy.explain_solution(player, args[0], " ".join(args[1:]))}

    def _cmd_upvote(self, player: str, args: list[str],
                   now: float) -> dict[str, Any]:
        if not args:
            return {"ok": False, "error": "usage",
                    "message": "Usage: /math upvote <explanation_id>"}
        return {"ok": True, "kind": "upvote",
                **self.economy.upvote_explanation(args[0], player, now=now)}

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
