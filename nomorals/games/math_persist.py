"""Durable Math Class state — JSON persistence for :class:`MathCommandHandler`.

Math Class state lives in plain dicts (``MathClass.store``) plus the
``MathEconomy`` ledger and the handler's own ``_active`` / ``_tier_clears``
/ ``_sets``. None of it is durable by itself: a fresh handler per call
(or a bot restart) wipes problems, streaks, leaderboards and the ledger.
This module is the durable layer — the runtime holds ONE handler per
process and saves it after every command.

Encoding notes (all JSON-safe):
* ``Fraction`` / ``complex`` answer values get a tagged dict
  (``{"__math__": "fraction", ...}``) — grading needs the exact value,
  and JSON has no fraction/complex.
* Tuple-keyed maps (attempts, cooldowns, counters, daily_best,
  lesson clears) become ``[{"k": [...], "v": ...}]`` lists — no
  separator collisions, keys round-trip exactly.
* ``Problem`` / ``Duel`` / ``MathLedgerEntry`` round-trip via
  ``asdict``-shaped dicts (``Problem`` via explicit fields so a missing
  field degrades to a skipped problem, never a crash).

Load is tolerant: a missing file → ``None`` (caller makes a fresh
handler); a corrupt section → that section resets to empty with a
warning, the rest loads. Save is atomic (tmp + replace) and never
raises — the runtime wraps it in try/except anyway.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from fractions import Fraction
from typing import Any

from .math_class import Duel, MathClass, Problem
from .math_commands import MathCommandHandler
from .math_economy import MathEconomy, MathLedgerEntry

_log = logging.getLogger("nomorals.games.math_persist")

STATE_VERSION = 1


def _store_path(path: str = "") -> str:
    if path:
        return os.path.expanduser(path)
    override = os.environ.get("NOMORALS_MATH_STATE_PATH", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".config",
                        "nomorals", "math_state.json")


# ── value encoding ──────────────────────────────────────────────────────────

def _enc_value(value: Any) -> Any:
    """Encode one answer value (or submitted answer) for JSON."""
    if isinstance(value, Fraction):
        return {"__math__": "fraction", "n": value.numerator,
                "d": value.denominator}
    if isinstance(value, complex):
        return {"__math__": "complex", "re": value.real, "im": value.imag}
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    raise ValueError(
        f"cannot persist math value of type {type(value).__name__}")


def _dec_value(value: Any) -> Any:
    if isinstance(value, dict) and "__math__" in value:
        tag = value.get("__math__")
        if tag == "fraction":
            return Fraction(int(value["n"]), int(value["d"]))
        if tag == "complex":
            return complex(float(value["re"]), float(value["im"]))
        raise ValueError(f"unknown math value tag {tag!r}")
    return value


# ── Problem / Duel ──────────────────────────────────────────────────────────

_PROBLEM_FIELDS = (
    "problem_id", "seed", "tier", "topic", "format", "question",
    "answer_value", "answer_text", "solution", "solution_steps",
    "tolerance", "time_limit_s", "created_at",
)


def _enc_problem(problem: Problem) -> dict[str, Any]:
    data = asdict(problem)
    data["answer_value"] = _enc_value(problem.answer_value)
    return data


def _dec_problem(data: Any) -> Problem | None:
    if not isinstance(data, dict):
        return None
    try:
        fields = {k: data.get(k) for k in _PROBLEM_FIELDS}
        fields["answer_value"] = _dec_value(data.get("answer_value"))
        fields["solution_steps"] = list(fields.get("solution_steps") or [])
        return Problem(**fields)  # type: ignore[arg-type]
    except (TypeError, ValueError, KeyError) as exc:
        _log.warning("math state: skipping undecodable problem: %s", exc)
        return None


def _enc_duel(duel: Duel) -> dict[str, Any]:
    return {
        "duel_id": duel.duel_id,
        "players": list(duel.players),
        "problems": [_enc_problem(p) for p in duel.problems],
        "round_winners": list(duel.round_winners),
        "answers": {p: {str(r): _enc_value(a) for r, a in rounds.items()}
                    for p, rounds in duel.answers.items()},
        "status": duel.status,
        "winner": duel.winner,
        "created_at": duel.created_at,
    }


def _dec_duel(data: Any) -> Duel | None:
    if not isinstance(data, dict):
        return None
    try:
        problems = []
        for raw in data.get("problems") or []:
            problem = _dec_problem(raw)
            if problem is not None:
                problems.append(problem)
        answers = {}
        for player, rounds in (data.get("answers") or {}).items():
            answers[str(player)] = {int(r): _dec_value(a)
                                   for r, a in (rounds or {}).items()}
        return Duel(
            duel_id=str(data.get("duel_id", "")),
            players=[str(p) for p in data.get("players") or []],
            problems=problems,
            round_winners=list(data.get("round_winners") or []),
            answers=answers,
            status=str(data.get("status", "open")),
            winner=data.get("winner"),
            created_at=float(data.get("created_at") or 0.0),
        )
    except (TypeError, ValueError, KeyError) as exc:
        _log.warning("math state: skipping undecodable duel: %s", exc)
        return None


# ── tuple-keyed maps ────────────────────────────────────────────────────────

def _enc_tuple_map(mapping: Any) -> list[dict[str, Any]]:
    if not isinstance(mapping, dict):
        return []
    return [{"k": list(k), "v": v} for k, v in mapping.items()]


def _dec_tuple_map(data: Any) -> dict[tuple[Any, ...], Any]:
    out: dict[tuple[Any, ...], Any] = {}
    if not isinstance(data, list):
        return out
    for entry in data:
        if isinstance(entry, dict) and isinstance(entry.get("k"), list):
            out[tuple(entry["k"])] = entry.get("v")
    return out


# ── dump / load ─────────────────────────────────────────────────────────────

def dump_state(handler: MathCommandHandler) -> dict[str, Any]:
    """Serialize a handler's full state to a JSON-safe dict."""
    store = handler.math.store
    economy = handler.economy
    return {
        "version": STATE_VERSION,
        "math_store": {
            "seen": store.get("seen", {}),
            "players": store.get("players", {}),
            "attempts": _enc_tuple_map(store.get("attempts", {})),
            "cooldowns": _enc_tuple_map(store.get("cooldowns", {})),
            "active": {pid: _enc_problem(p)
                       for pid, p in store.get("active", {}).items()},
            "scores": store.get("scores", {}),
            "duels": {did: _enc_duel(d)
                       for did, d in store.get("duels", {}).items()},
            "counters": _enc_tuple_map(store.get("counters", {})),
            "streaks": store.get("streaks", {}),
            "daily_best": _enc_tuple_map(store.get("daily_best", {})),
        },
        "economy": {
            "entries": [asdict(e) for e in economy._entries],
            "verified": economy._verified,
            "fingerprints": economy._fingerprints,
            "flags": economy._flags,
            "lessons": economy._lessons,
            "lesson_clears": _enc_tuple_map(economy._lesson_clears),
            "explanations": economy._explanations,
            "problems": economy._problems,
            "seq": economy._seq,
        },
        "handler": {
            "active": dict(getattr(handler, "_active", {})),
            "tier_clears": {p: sorted(t)
                            for p, t in getattr(handler, "_tier_clears", {}).items()},
            "sets": dict(getattr(handler, "_sets", {})),
        },
    }


def load_state(data: Any) -> MathCommandHandler:
    """Rebuild a handler from :func:`dump_state` output.

    Malformed sections reset to empty (with a warning) — a corrupt
    file never crashes the load; the worst case is a fresh Math
    Class, not a dead bot.
    """
    if not isinstance(data, dict):
        raise ValueError("math state is not an object")
    math_class = MathClass(store={})
    economy = MathEconomy()
    handler = MathCommandHandler(math_class=math_class, economy=economy)

    raw_store = data.get("math_store")
    if isinstance(raw_store, dict):
        store = math_class.store
        store["seen"] = raw_store.get("seen") or {}
        store["players"] = raw_store.get("players") or {}
        store["attempts"] = _dec_tuple_map(raw_store.get("attempts"))
        store["cooldowns"] = _dec_tuple_map(raw_store.get("cooldowns"))
        active = {}
        for pid, raw in (raw_store.get("active") or {}).items():
            problem = _dec_problem(raw)
            if problem is not None:
                active[str(pid)] = problem
        store["active"] = active
        store["scores"] = raw_store.get("scores") or {}
        duels = {}
        for did, raw in (raw_store.get("duels") or {}).items():
            duel = _dec_duel(raw)
            if duel is not None:
                duels[str(did)] = duel
        store["duels"] = duels
        store["counters"] = _dec_tuple_map(raw_store.get("counters"))
        store["streaks"] = raw_store.get("streaks") or {}
        store["daily_best"] = _dec_tuple_map(raw_store.get("daily_best"))

    raw_econ = data.get("economy")
    if isinstance(raw_econ, dict):
        entries = []
        for raw in raw_econ.get("entries") or []:
            try:
                entries.append(MathLedgerEntry.from_dict(raw))
            except (TypeError, ValueError, KeyError):
                continue
        economy._entries = entries
        economy._verified = raw_econ.get("verified") or {}
        economy._fingerprints = raw_econ.get("fingerprints") or {}
        economy._flags = raw_econ.get("flags") or []
        economy._lessons = raw_econ.get("lessons") or {}
        economy._lesson_clears = _dec_tuple_map(raw_econ.get("lesson_clears"))
        economy._explanations = raw_econ.get("explanations") or {}
        economy._problems = raw_econ.get("problems") or {}
        try:
            economy._seq = int(raw_econ.get("seq") or 0)
        except (TypeError, ValueError):
            economy._seq = len(entries)

    raw_handler = data.get("handler")
    if isinstance(raw_handler, dict):
        handler._active = {str(k): str(v)
                           for k, v in (raw_handler.get("active") or {}).items()}
        handler._tier_clears = {str(p): set(t or [])
                                for p, t in (raw_handler.get("tier_clears") or {}).items()}
        sets = raw_handler.get("sets")
        handler._sets = sets if isinstance(sets, dict) else {}
    return handler


def save_math_state(handler: MathCommandHandler, path: str = "") -> bool:
    """Atomic save. Returns False (never raises) on any failure."""
    try:
        target = _store_path(path)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(dump_state(handler), fh)
        os.replace(tmp, target)
        return True
    except (OSError, ValueError, TypeError) as exc:
        _log.warning("math state save failed: %s", exc)
        return False


def load_math_state(path: str = "") -> MathCommandHandler | None:
    """Load a persisted handler. ``None`` = no usable file (missing or
    corrupt) — the caller makes a fresh handler."""
    try:
        with open(_store_path(path), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    try:
        return load_state(data)
    except (ValueError, TypeError, KeyError) as exc:
        _log.warning("math state load failed (%s) — starting fresh", exc)
        return None
