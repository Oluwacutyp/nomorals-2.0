"""Multi-agent swarm: parallel sub-investigators with a synthesis step.

A swarm splits one goal into N subtasks, runs each on its own Devon agent in
a thread (real parallelism — the devon loop is I/O- and model-bound), then
fuses the digests into one answer.

Decomposition is model-driven when a real model is answering (the same
decompose contract the search engine uses); otherwise it falls back to
heuristic splits ("A and B", "A, then B") and, failing that, to
perspective workers — evidence / risks / plan — so a swarm is still useful
offline.

Coordination:
* each worker gets its own DevonAgent (own journal, own run id — no shared
  mutable state to lock);
* every run gets a swarm board (public briefs, an append-only feed,
  @mentions, a verifier per 2–4 workers, a monitor that wakes stalled
  boards) — the team talks mid-run, not just at the fusion step;
* a wall-clock budget caps the whole swarm and each worker;
* a crashed worker is reported as a failed leg, never fatal to the rest;
* the synthesis is a single model call over the digests, with a plain
  concatenation fallback.

``/swarm <goal> [workers]`` from chat, and Devon's own ``swarm`` tool for
recursive delegation.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from ..llm.brain import brain_for
from ..core.ids import new_short_id
from ..core.tool_render import describe_exc, emit_progress
from ..core.logging_setup import get_logger
from .devon import DevonAgent, DevonResult
from .swarm_board import (
    SwarmBoard,
    parse_verifier_verdict,
    worker_name_for,
)

_log = get_logger(__name__)

__all__ = ["SwarmAgent", "SwarmResult"]

_PERSPECTIVES = (
    "evidence and facts",
    "risks, unknowns, and failure modes",
    "concrete step-by-step plan",
    "costs, trade-offs, and cheaper alternatives",
    "who this affects and second-order effects",
    "prior art and what already exists",
    "the skeptical counter-case",
    "timeline, sequencing, and dependencies",
    "how to verify it worked",
)


def _rotate_perspectives(goal: str, workers: int) -> list[str]:
    """Seeded rotation over the perspective pool.

    Same goal → same angles (reproducible); different goals → different
    angles, so repeated heuristic swarms don't always investigate the
    same three things.
    """
    import hashlib
    import random as _random

    seed = int(hashlib.sha256(goal.encode("utf-8")).hexdigest(), 16)
    rng = _random.Random(seed)
    pool = list(_PERSPECTIVES)
    rng.shuffle(pool)
    return pool[:max(1, workers)]


@dataclass
class SwarmResult:
    goal: str
    subtasks: list[str]
    legs: list[dict[str, Any]] = field(default_factory=list)
    synthesis: str = ""
    seconds: float = 0.0
    run_id: str = field(default_factory=lambda: new_short_id("swarm"))

    @property
    def ok(self) -> bool:
        return any(leg.get("ok") for leg in self.legs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "subtasks": self.subtasks,
            "legs": self.legs,
            "synthesis": self.synthesis,
            "seconds": round(self.seconds, 2),
            "run_id": self.run_id,
            "ok": self.ok,
        }


class SwarmAgent:
    """Plan → parallel devon legs → fuse.

    When a ``blackboard`` is shared, legs stop being isolated: each leg
    posts its digest to the board as it finishes, and legs that start
    later read the digests already posted — cross-leg awareness without
    point-to-point wiring. Without a board the legs run isolated, exactly
    as before.
    """

    #: Schema version for :meth:`save_state` / :meth:`resume` payloads.
    STATE_VERSION = 1

    def __init__(
        self,
        context: Any,
        *,
        brain: Any = None,
        gateway: Any = None,
        wall_seconds: float = 420.0,
        blackboard: Any = None,
        board: SwarmBoard | None = None,
        chat_key: str = "",
        on_leg: Callable[[int, int, dict[str, Any]], None] | None = None,
    ) -> None:
        self.context = context
        self.brain = brain
        self.gateway = gateway
        self.wall_seconds = float(wall_seconds)
        self.blackboard = blackboard
        #: The run's shared board (public briefs, feed, @mentions,
        #: verifiers). When None, :meth:`run` builds one on the
        #: blackboard — a board that cannot be built degrades to
        #: isolated legs, exactly as before.
        self.board = board
        #: Where the monitor tick sends its wake-up notice.
        self.chat_key = str(chat_key or "")
        #: Progress hook: called as legs complete — (index, total, leg).
        #: Long swarms are otherwise silent; this feeds live UIs.
        self.on_leg = on_leg
        self._last_result: SwarmResult | None = None
        #: Deterministic worker names for the current run.
        self._leg_names: list[str] = []

    def _emit_leg(self, index: int, total: int, leg: dict[str, Any]) -> None:
        """Fire the leg-progress hook. Never raises."""
        if self.on_leg is None:
            return
        try:
            self.on_leg(index, total, leg)
        except Exception:  # noqa: BLE001 — telemetry never breaks the swarm
            _log.debug("swarm on_leg hook failed", exc_info=True)

    # ── the swarm board ──────────────────────────────────────────────────
    def _board_store(self) -> Any:
        """The durable board store under the runtime home, or None.

        Best-effort: no home / a broken path means the board lives
        in memory only, and the swarm still runs.
        """
        try:
            home = getattr(getattr(self.context, "settings", None),
                           "home", None)
            if home is None:
                home = getattr(self.context, "home", None)
            if home is None:
                return None
            from .swarm_board import store_for_context

            return store_for_context(self.context)
        except Exception:  # noqa: BLE001 - durability is a bonus
            _log.debug("swarm board store unavailable", exc_info=True)
            return None

    def _save_board(self, board: SwarmBoard) -> None:
        """Persist a board (best-effort — never breaks the swarm)."""
        store = self._board_store()
        if store is None:
            return
        try:
            store.save_board(board)
        except Exception:  # noqa: BLE001 - durability is a bonus
            _log.debug("swarm board save failed", exc_info=True)

    def _ensure_board(self, run_id: str, goal: str) -> SwarmBoard | None:
        """The run's board, built on the blackboard (never raises)."""
        if self.board is not None:
            return self.board
        try:
            board = SwarmBoard(run_id, goal=goal, board=self.blackboard)
            board.chat_key = self.chat_key
            self.board = board
            return board
        except Exception:  # noqa: BLE001 - isolated legs are the fallback
            _log.debug("swarm board unavailable", exc_info=True)
            return None

    def _setup_board(self, board: SwarmBoard, subtasks: list[str]) -> list[str]:
        """Roster + public briefs for a run. Returns worker names."""
        taken: set[str] = set()
        names: list[str] = []
        for i, sub in enumerate(subtasks):
            name = worker_name_for(sub, i, taken)
            taken.add(name)
            names.append(name)
        # The coordinator posts the header and the briefs, but is
        # NOT a roster member — the roster is workers + verifiers.
        # (Members[0] is the first worker: the verifier flags it,
        # the monitor wakes it, the owner reads it.)
        workers_line = ", ".join("@" + n for n in names)
        board.post(
            "coordinator",
            f"🐝 swarm {board.run_id}\ngoal: {board.goal[:200]}\n"
            f"workers: {workers_line}",
            kind="brief")
        for i, (name, sub) in enumerate(zip(names, subtasks, strict=True)):
            others = "; ".join(
                s[:160] for j, s in enumerate(subtasks) if j != i)
            board.brief(
                name,
                owns=sub[:400],
                must_not_touch=others or "(nothing else)",
                done="a digest posted to the board")
        self._leg_names = names
        self._save_board(board)
        return names

    # ── decomposition ────────────────────────────────────────────────────────
    def _model_available(self) -> bool:
        router = getattr(self.context, "router", None)
        snapshot = getattr(router, "stats_snapshot", None)
        if snapshot is None:
            return False
        try:
            snap = snapshot()
        except Exception:  # noqa: BLE001
            return False
        active = str(snap.get("active") or "")
        return bool(active) and active not in {"mock", "offline", "test", ""}

    def decompose(self, goal: str, workers: int) -> tuple[list[str], str]:
        """→ (subtasks, planner name)."""
        if self._model_available():
            try:
                from ..llm.base import Message, SamplingParams

                response = brain_for(self.context).chat(
                    [
                        Message.system(
                            f"Split this goal into at most {workers} parallel subtasks that "
                            "independent investigators can each complete without talking to "
                            "each other. Complementary, non-overlapping, each self-contained. "
                            "Reply with ONLY a JSON array of strings."
                        ),
                        Message.user(goal),
                    ],
                    SamplingParams(temperature=0.2, max_tokens=300),
                task_kind="chat")
                text = (response.text or "").strip()
                start, end = text.find("["), text.rfind("]")
                if start != -1 and end > start:
                    subs = [
                        str(x).strip()
                        for x in json.loads(text[start : end + 1])
                        if str(x).strip()
                    ]
                    if subs:
                        return subs[:workers], "llm"
            except Exception as exc:  # noqa: BLE001 - fall through to heuristics
                _log.debug("swarm decomposition failed: %s", exc)

        return self._heuristic_decompose(goal, workers), "heuristic"

    def _heuristic_decompose(self, goal: str, workers: int) -> list[str]:
        """Explicit conjunctions first; else numbered items; else perspectives."""
        parts = [p.strip() for p in re.split(r"\s+(?:and|then|,)\s+(?=[a-z])", goal, flags=re.IGNORECASE) if p.strip()]
        if len(parts) >= 2:
            return parts[:workers]
        numbered = re.split(r"(?:^|\n)\s*\d+[.)]\s*", goal)
        parts = [p.strip() for p in numbered if p.strip()]
        # a preamble ("do these:") survives as part 0 — drop it
        if len(parts) >= 2 and not re.search(r"\d+[.)]", goal.split("\n")[0].strip()[:3]):
            parts = parts[1:]
        if len(parts) >= 2:
            return parts[:workers]
        return [f"investigate the {pers} for: {goal}"
                for pers in _rotate_perspectives(goal, workers)]

    # ── workers ──────────────────────────────────────────────────────────────
    def _prior_digests(self, run_id: str) -> str:
        """Digests already posted by finished legs (cross-leg awareness).

        Best-effort: whatever legs have posted so far. Each digest is
        truncated so a large swarm can't blow the next leg's context.
        """
        board = self.blackboard
        if board is None:
            return ""
        try:
            entries = board.topic(run_id) if hasattr(board, "topic") else {}
        except Exception:  # noqa: BLE001 - awareness is a bonus
            return ""
        legs: list[tuple[int, str, str]] = []
        for key, value in entries.items():
            if not key.startswith(f"swarm.{run_id}.leg."):
                continue
            try:
                index = int(key.rsplit(".leg.", 1)[1])
            except ValueError:
                continue
            digest = value.get("digest", "") if isinstance(value, dict) else str(value)
            subtask = value.get("subtask", "") if isinstance(value, dict) else ""
            legs.append((index, subtask, digest))
        if not legs:
            return ""
        parts = []
        for index, subtask, digest in sorted(legs):
            snippet = str(digest)[:600].strip()
            if snippet:
                parts.append(f"[leg {index} — {subtask}]\n{snippet}")
        return "\n\n".join(parts)

    def _post_leg(self, run_id: str, index: int, leg: dict[str, Any]) -> None:
        board = self.blackboard
        if board is None:
            return
        try:
            board.post(
                f"swarm.{run_id}.leg.{index}",
                {"subtask": leg.get("subtask", ""), "ok": leg.get("ok", False),
                 "digest": str(leg.get("digest", ""))[:4000],
                 "seconds": leg.get("seconds", 0)},
                author=f"swarm-leg-{index}",
                topic=run_id,
                metadata={"run_id": run_id, "index": index,
                          "ok": bool(leg.get("ok"))},
            )
        except Exception:  # noqa: BLE001 - posting never breaks a leg
            _log.debug("swarm leg post failed", exc_info=True)

    def _post_board_result(self, board: SwarmBoard, name: str,
                           leg: dict[str, Any], *,
                           kind: str = "result") -> None:
        """Post a leg's digest to the board and mark the member
        waiting. The coordinator posts it — the run's narrator —
        so a worker's unread feed is the whole board, not the board
        minus its own posts. Never raises."""
        try:
            board.post("coordinator",
                       str(leg.get("digest") or "")[:4000], kind=kind)
        except Exception:  # noqa: BLE001 - posting never breaks a leg
            _log.debug("swarm board result post failed", exc_info=True)
        try:
            board.set_state(name, "waiting")
        except Exception:  # noqa: BLE001 - state is bookkeeping
            _log.debug("swarm board set_state failed", exc_info=True)

    def _run_leg(self, subtask: str, deadline: float, *,
                 index: int = 0, run_id: str = "",
                 on_progress: Callable[[int, int, str, str], None] | None = None,
                 total_legs: int = 0, member: str = "",
                 verifier: bool = False,
                 verifier_for: tuple[str, ...] = ()) -> dict[str, Any]:
        started = time.time()
        remaining = max(20.0, deadline - started)
        brief = subtask
        if self.blackboard is not None and run_id:
            prior = self._prior_digests(run_id)
            if prior:
                brief = (
                    f"{subtask}\n\nFindings already reported by fellow "
                    f"investigators — do not repeat them, build on them:\n{prior}"
                )
        board = self.board
        if board is not None and member:
            # A leg that starts later reads the messages addressed to
            # it — briefs are public, but a mention is a direct ask.
            # Only real messages count (not briefs, not "started"
            # posts): verdicts, wake-ups, other workers' words.
            try:
                board.set_state(member, "active")
            except Exception:  # noqa: BLE001 - a ghost member is not fatal
                _log.debug("swarm board set_state failed", exc_info=True)
            try:
                addressed = [p for p in board.mentions_for(member)
                             if p.get("kind")
                             in ("message", "verdict", "wake")]
            except Exception:  # noqa: BLE001 - mentions are a bonus
                addressed = []
            if addressed:
                brief = (
                    f"{brief}\n\nMessages addressed to you on the swarm "
                    f"board — answer them in your digest:\n"
                    + "\n".join(
                        f"@{p['author']}: {str(p.get('text') or '')[:400]}"
                        for p in addressed))
                board.mark_read(member, max(p["seq"] for p in addressed))
            if verifier:
                progress = ("started: verifying "
                            + ", ".join(verifier_for))
            else:
                progress = f"started: {subtask[:160]}"
            board.post(member, progress, kind="progress")
        try:
            # step_timeout doubles as the per-leg wall budget (× max_steps
            # inside devon), so scale it to the swarm's remaining time.
            agent = DevonAgent(self.context, brain=self.brain, gateway=self.gateway,
                               max_steps=8, step_timeout=remaining / 8.0)
            emit_progress(on_progress, index + 1, total_legs or 1,
                          f"swarm leg {index + 1}", "start")
            result: DevonResult = agent.run(brief, chat_key="",
                                            on_progress=lambda step, total, tool, status, _i=index: emit_progress(
                                                on_progress, _i + 1, total_legs or 1,
                                                f"leg {_i + 1} · {tool}", status))
            emit_progress(on_progress, index + 1, total_legs or 1,
                          f"swarm leg {index + 1}", "done")
            leg = {
                "subtask": subtask,
                "ok": True,
                "digest": result.digest,
                "planned_by": result.planned_by,
                "steps": len(result.steps),
                "seconds": round(result.seconds, 1),
            }
        except Exception as exc:  # noqa: BLE001 - a dead leg is a result
            _log.exception("swarm leg failed: %r", subtask)
            leg = {
                "subtask": subtask,
                "ok": False,
                "digest": f"leg failed: {describe_exc(exc)}",
                "planned_by": "error",
                "steps": 0,
                "seconds": round(time.time() - started, 1),
            }
        # The result post + the waiting state live in the
        # orchestration (run / _verify_and_rerun), not here: a leg
        # that timed out at the future level still gets its result
        # posted, and the verdict phase posts verdicts the same way.
        self._post_leg(run_id, index, leg)
        return leg

    # ── synthesis ────────────────────────────────────────────────────────────
    def _synthesize(self, goal: str, legs: list[dict[str, Any]]) -> tuple[str, bool]:
        if self._model_available():
            try:
                from ..llm.base import Message, SamplingParams

                leg_text = "\n\n".join(
                    f"--- worker {i + 1} ({leg['subtask']}) ---\n{leg['digest']}"
                    for i, leg in enumerate(legs)
                    if leg.get("ok")
                )
                response = brain_for(self.context).chat(
                    [
                        Message.system(
                            "You are fusing the results of a parallel multi-agent swarm "
                            "into one coherent answer. Merge overlaps, keep conflicts "
                            "visible, and lead with the direct answer."
                        ),
                        Message.user(f"Goal: {goal}\n\nWorker results:\n{leg_text[:12000]}"),
                    ],
                    SamplingParams(temperature=0.3, max_tokens=900),
                task_kind="chat")
                text = (response.text or "").strip()
                if text:
                    return text, True
            except Exception as exc:  # noqa: BLE001
                _log.debug("swarm synthesis failed; concatenating: %s", exc)
        merged = "\n\n".join(f"• {leg['subtask']}\n  {leg['digest'][:800]}"
                             for leg in legs if leg.get("ok"))
        return (merged or "all swarm legs failed"), False

    def _verify_and_rerun(self, board: SwarmBoard, goal: str,
                          subtasks: list[str], leg_names: list[str],
                          result: SwarmResult, deadline: float,
                          on_progress: Callable[[int, int, str, str], None]
                          | None = None) -> None:
        """Verifier phase, then exactly one re-run per flagged / failed
        worker. A worker that fails twice is killed, not retried again.

        The verifier phase only runs when at least one worker leg
        succeeded — there is nothing to verify when everything failed.
        Verifier legs are sequential (they read the same board, and a
        swarm's verifiers are few), re-runs too: the second attempt
        must see the first attempt's digest and the verifier's notes.
        """
        if not any(leg.get("ok") for leg in result.legs):
            return
        flags: dict[str, list[str]] = {}
        for v_index, verifier in enumerate(board.assign_verifiers()):
            group = board.verifier_group(verifier)
            v_subtask = (
                f"Verify the swarm work of "
                f"{', '.join(group)}: read their digests on the board, "
                f"check the work, and report one line per worker — "
                f"'OK name: ...' or 'FLAG name: <problem>'.")
            v_leg = self._run_leg(
                v_subtask, deadline, index=len(subtasks) + v_index,
                run_id=result.run_id, on_progress=on_progress,
                total_legs=len(subtasks), member=verifier,
                verifier=True, verifier_for=tuple(group))
            self._post_board_result(board, verifier, v_leg,
                                    kind="verdict")
            for worker, notes in parse_verifier_verdict(
                    str(v_leg.get("digest") or "")).items():
                flags.setdefault(worker, []).extend(notes)
        for i, leg in enumerate(list(result.legs)):
            name = leg_names[i] if i < len(leg_names) else ""
            notes = flags.get(name, [])
            if leg.get("ok") and not notes:
                continue
            retry_brief = subtasks[i]
            if notes:
                retry_brief = (
                    f"{subtasks[i]}\n\nVerifier notes — fix these, "
                    f"do not repeat them:\n"
                    + "\n".join(f"- {n}" for n in notes))
            elif not leg.get("ok"):
                retry_brief = (
                    f"{subtasks[i]}\n\nYour first attempt failed: "
                    f"{str(leg.get('digest') or '')[:400]}. Try again.")
            new_leg = self._run_leg(
                retry_brief, deadline, index=i, run_id=result.run_id,
                on_progress=on_progress, total_legs=len(subtasks),
                member=name)
            result.legs[i] = new_leg
            self._post_board_result(board, name, new_leg)
            if not new_leg.get("ok") and name:
                try:
                    board.set_state(name, "killed")
                except Exception:  # noqa: BLE001 - state is bookkeeping
                    _log.debug("swarm board set_state failed", exc_info=True)

    # ── the run ──────────────────────────────────────────────────────────────
    def run(self, goal: str, workers: int = 3,
            on_progress: Callable[[int, int, str, str], None] | None = None) -> SwarmResult:
        """Run the swarm. ``on_progress(leg, total_legs, label, status)``
        fires per leg and per Devon step inside each leg, so a chat
        caller can show progress on this long-running path."""
        started = time.time()
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("swarm needs a goal")
        workers = max(1, min(int(workers), 5))
        deadline = started + self.wall_seconds

        subtasks, planner = self.decompose(goal, workers)
        result = SwarmResult(goal=goal, subtasks=subtasks)

        # ── the board: roster + public briefs before any leg starts ──
        board = self._ensure_board(result.run_id, goal)
        leg_names: list[str] = []
        if board is not None:
            leg_names = self._setup_board(board, subtasks)

        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="swarm")
        futures = [pool.submit(self._run_leg, sub, deadline,
                               index=i, run_id=result.run_id,
                               on_progress=on_progress,
                               total_legs=len(subtasks),
                               member=leg_names[i] if i < len(leg_names)
                               else "")
                   for i, sub in enumerate(subtasks)]
        for i, future in enumerate(futures):
            try:
                leg = future.result(timeout=max(10.0, deadline - time.time()))
            except Exception as exc:  # noqa: BLE001 - a late leg is a result
                leg = {
                    "subtask": subtasks[i], "ok": False,
                    "digest": f"leg timed out: {describe_exc(exc)}", "planned_by": "timeout",
                    "steps": 0, "seconds": 0,
                }
                self._post_leg(result.run_id, i, leg)
            result.legs.append(leg)
            if board is not None and i < len(leg_names):
                self._post_board_result(board, leg_names[i], leg)
            self._emit_leg(i, len(subtasks), leg)
        # Don't wait on late legs past the swarm's own deadline: they are
        # bounded by devon's internal step budgets and finish on their own.
        pool.shutdown(wait=False, cancel_futures=True)

        # ── verifier phase + one re-run per flagged / failed worker ──
        if board is not None:
            self._verify_and_rerun(
                board, goal, subtasks, leg_names, result, deadline,
                on_progress=on_progress)
            # The run is over: everyone not killed is waiting. (A
            # killed member stays killed — never revived by the sweep.)
            for member in board.members:
                if member.get("state") != "killed":
                    try:
                        board.set_state(member["name"], "waiting")
                    except Exception:  # noqa: BLE001 - state is bookkeeping
                        _log.debug("swarm board set_state failed",
                                   exc_info=True)
            board.state = "done"
            self._save_board(board)

        result.synthesis, model_ok = self._synthesize(goal, result.legs)
        result.seconds = time.time() - started
        self._last_result = result
        _log.info(
            "swarm %s done: %d/%d legs ok, %.0fs (%s planner%s)",
            result.run_id, sum(1 for l in result.legs if l["ok"]), len(result.legs),
            result.seconds, planner, ", model-synthesized" if model_ok else "",
        )
        return result

    # ── checkpoint / resume ────────────────────────────────────────────
    def save_state(self, path: str) -> dict[str, Any]:
        """Persist the last run's goal, subtasks, and legs as JSON.

        A crashed swarm can be continued with :meth:`resume` — finished
        legs are not re-run. Best-effort: returns ``{"ok": False,
        "error": ...}`` instead of raising.
        """
        result = self._last_result
        if result is None:
            return {"ok": False, "error": "no swarm run to save yet"}
        payload = {
            "version": self.STATE_VERSION,
            "goal": result.goal,
            "subtasks": result.subtasks,
            "legs": result.legs,
            "run_id": result.run_id,
            "saved_at": time.time(),
        }
        try:
            from pathlib import Path as _Path
            target = _Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False),
                              encoding="utf-8")
            return {"ok": True, "path": str(target),
                    "legs": len(result.legs)}
        except OSError as exc:
            return {"ok": False, "error": str(exc), "path": str(path)}

    def resume(self, path: str, *, workers: int = 3) -> SwarmResult:
        """Continue an interrupted swarm from a :meth:`save_state` file.

        Subtasks with a successful recorded leg keep their digest; every
        other subtask is re-run in parallel. Raises ``ValueError`` on a
        corrupt/foreign state file, ``FileNotFoundError`` when missing.
        """
        from pathlib import Path as _Path
        raw = json.loads(_Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != self.STATE_VERSION:
            raise ValueError(f"not a swarm state file: {path}")
        goal = str(raw.get("goal") or "")
        subtasks = [str(s) for s in (raw.get("subtasks") or [])]
        if not goal or not subtasks:
            raise ValueError(f"swarm state file has no goal/subtasks: {path}")
        kept = {leg.get("subtask"): leg for leg in (raw.get("legs") or [])
                if isinstance(leg, dict) and leg.get("ok")}

        started = time.time()
        result = SwarmResult(goal=goal, subtasks=subtasks)
        # Keep the original run id so the blackboard topic stays continuous.
        if raw.get("run_id"):
            result.run_id = str(raw["run_id"])
        pending = [(i, s) for i, s in enumerate(subtasks) if s not in kept]
        deadline = started + self.wall_seconds
        fresh: dict[int, dict[str, Any]] = {}
        if pending:
            pool = ThreadPoolExecutor(max_workers=max(1, min(int(workers), 5)),
                                      thread_name_prefix="swarm-resume")
            futures = {pool.submit(self._run_leg, sub, deadline,
                                   index=i, run_id=result.run_id): i
                       for i, sub in pending}
            for future, i in futures.items():
                try:
                    fresh[i] = future.result(
                        timeout=max(10.0, deadline - time.time()))
                except Exception as exc:  # noqa: BLE001 - a late leg is a result
                    leg = {"subtask": subtasks[i], "ok": False,
                           "digest": f"leg timed out: {exc}",
                           "planned_by": "timeout", "steps": 0, "seconds": 0}
                    self._post_leg(result.run_id, i, leg)
                    fresh[i] = leg
            pool.shutdown(wait=False, cancel_futures=True)

        result.legs = [kept.get(s, fresh.get(i, {
            "subtask": s, "ok": False, "digest": "leg missing from state",
            "planned_by": "resume", "steps": 0, "seconds": 0}))
            for i, s in enumerate(subtasks)]
        result.synthesis, _ = self._synthesize(goal, result.legs)
        result.seconds = time.time() - started
        self._last_result = result
        _log.info("swarm %s resumed: %d kept, %d re-run, %d/%d legs ok",
                  result.run_id, len(kept), len(pending),
                  sum(1 for l in result.legs if l["ok"]), len(result.legs))
        return result


# ── presentation ──────────────────────────────────────────────────────
# A swarm run is a mission report: lead with the fused answer, then show
# every leg's status. Legs are evidence, not a log tail.

def render_swarm(result: "SwarmResult") -> str:
    """Render a swarm run as a human-readable mission report. Never raises."""
    from .render import ICONS, banner, bar, kv, section, table, truncate

    try:
        legs = result.legs or []
        done = sum(1 for leg in legs if leg.get("ok"))
        head = {
            "goal": truncate(result.goal, 80),
            "legs": f"{done}/{len(legs)} ok",
            "time": f"{result.seconds:.1f}s",
            "run": result.run_id,
        }
        lines = [banner("Swarm", ICONS["thinking"]), kv(head.items()),
                 "", bar(done / len(legs) if legs else 0.0)]
        if legs:
            rows = []
            for i, leg in enumerate(legs, 1):
                icon = ICONS["ok"] if leg.get("ok") else ICONS["fail"]
                rows.append([f"{icon} {i}",
                             truncate(str(leg.get("subtask", "")), 36),
                             truncate(str(leg.get("digest", "")), 60),
                             f"{leg.get('seconds', 0)}s"])
            lines.append("")
            lines.append(table(["#", "subtask", "digest", "time"], rows))
        lines.append("")
        lines.append(section("Fused answer",
                             truncate(result.synthesis or "(none)", 1200),
                             ICONS["star"]))
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — rendering never breaks callers
        return f"swarm {getattr(result, 'run_id', '?')} (render failed)"
