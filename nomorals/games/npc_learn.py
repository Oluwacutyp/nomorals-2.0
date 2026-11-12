"""NPC learner — the NPC's own scorer and combat policy (Phase 2).

The owner's doctrine (2026-10-24/25): NPCs are IN-GAME characters —
they do NOT need faces; they need dynamic behavior / dialogue /
tactics, and the small-ML family (a) is their own scorer/learner:
learn the player, pick the most in-character line, opinion shifts.

Two learners live here, both journal → grade → retrain:

* **Line scorer** — generated candidate lines, the scorer picks the
  most in-character (``score_npc_candidates``), outcomes journaled
  and graded (``NPCJournal``), retrained from the journal
  (``fit_npc_scorer``). The hand-weight scorer is the code rung; the
  batch fit promotes only when it beats the hand weights on a held-out
  slice.
* **Combat policy** — journaled rounds (state -> action -> fight
  outcome, ``TacticsJournal``), retrained into a policy
  (``fit_tactics_policy``) that picks among LEGAL actions by predicted
  win probability. The math (``GameMind._combat_math``) is the code
  rung; the policy promotes only when it predicts fight outcomes
  better than the math's action agreement on a held-out slice. The
  label is the FIGHT outcome (attached at fight end), never the
  immediate payoff — defend/focus are investments, and an immediate-
  payoff label would teach "always attack".

Storage mirrors the characters/learn.py durability pattern: one JSON
doc per (game, npc) / per game, atomic tmp+replace writes, lazy load,
no import-time I/O. A learn failure never eats a line or a move.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "NPC_FEATURES", "NPC_WEIGHTS", "NPC_MIN_ROWS",
    "TACTICS_FEATURES", "TACTICS_ACTIONS", "TACTICS_MIN_ROWS",
    "npc_line_features", "score_npc_candidates",
    "NPCJournal", "fit_npc_scorer", "load_npc_scorer",
    "TacticsJournal", "TacticsPolicy", "fit_tactics_policy",
    "load_tactics_policy", "record_combat_round", "grade_fight",
    "score_npc_candidates_full",
]

_log = get_logger(__name__)

# ── line scorer ─────────────────────────────────────────────────────────

NPC_FEATURES = ("trait_overlap", "mood_fit", "situation_fit", "novelty",
                "length_fit")
NPC_WEIGHTS = {"trait_overlap": 0.35, "mood_fit": 0.20,
               "situation_fit": 0.25, "novelty": 0.10, "length_fit": 0.10}
#: Floor for the NPC line scorer (graded lines accumulate slowly).
NPC_MIN_ROWS = 30

_POSITIVE_WORDS = frozenset(
    ["love", "lovely", "great", "wonderful", "brilliant", "happy", "glad", "joy", "delight", "win", "won", "triumph", "cheer", "laugh", "smile", "kind", "warm", "friend"])
_NEGATIVE_WORDS = frozenset(
    ["hate", "awful", "terrible", "grim", "dark", "dead", "die", "doom", "curse", "fool", "stupid", "boring", "dull", "grim"])
_WORD_RE = re.compile(r"[a-z][a-z']+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower())
            if len(w) > 2}


def _line_valence(text: str) -> float:
    words = _content_words(text)
    if not words:
        return 0.0
    pos = len(words & _POSITIVE_WORDS)
    neg = len(words & _NEGATIVE_WORDS)
    return max(-1.0, min(1.0, (pos - neg) / max(len(words), 1) * 2.0))


def npc_line_features(line: str, npc: Any, situation: str = "",
                      recent_lines: list[str] | tuple[str, ...] = ()
                      ) -> dict[str, float]:
    """Feature vector for one candidate NPC line, all 0–1.

    ``npc`` needs ``personality`` (trait -> weight), ``mood``
    (valence -1..1) and ``voice_style``. The features are the NPC's
    own voice, not the prompt's vocabulary: trait words surfacing in
    the line, mood fit, situation fit, novelty vs recent lines, and
    length fit.
    """
    text = line or ""
    low = text.lower()
    personality = dict(getattr(npc, "personality", None) or {})
    top_traits = [t for t, w in personality.items()
                  if float(w or 0.0) >= 0.5]
    if top_traits:
        hits = sum(1 for t in top_traits
                   if str(t).lower() in low)
        trait_overlap = hits / len(top_traits)
    else:
        trait_overlap = 0.5
    mood_val = 0.0
    try:
        mood_val = float((getattr(npc, "mood", None) or {}).get(
            "valence", 0.0))
    except (TypeError, ValueError):
        mood_val = 0.0
    mood_fit = 1.0 - abs(_line_valence(text) - mood_val) / 2.0
    sit_words = {w for w in _WORD_RE.findall((situation or "").lower())
                 if len(w) > 3}
    if sit_words:
        line_words = _content_words(text)
        situation_fit = len(sit_words & line_words) / len(sit_words)
    else:
        situation_fit = 0.5
    if recent_lines:
        line_words = _content_words(text)
        overlaps = []
        for old in recent_lines:
            old_words = _content_words(old)
            if line_words and old_words:
                overlaps.append(len(line_words & old_words)
                                / len(line_words | old_words))
        novelty = 1.0 - max(overlaps, default=0.0)
    else:
        novelty = 1.0
    ideal = 60.0
    length_fit = max(0.0, 1.0 - abs(len(text) - ideal) / ideal)
    return {"trait_overlap": round(trait_overlap, 4),
            "mood_fit": round(mood_fit, 4),
            "situation_fit": round(situation_fit, 4),
            "novelty": round(novelty, 4),
            "length_fit": round(length_fit, 4)}


def score_npc_candidates(lines: list[str], npc: Any, situation: str = "",
                         recent_lines: list[str] | tuple[str, ...] = (),
                         model: Any = None
                         ) -> list[tuple[str, float]]:
    """Score candidate NPC lines: (text, score 0..1), input order.

    ``model`` is a fitted :class:`llm.specialists.LearnerModel` from
    :func:`fit_npc_scorer` — when given, the LEARNED scorer scores
    (its -1..1 grade prediction rescaled to 0..1); otherwise the hand
    weights score. The learned scorer is only handed here after it
    beat the hand weights on a held-out slice.
    """
    return [(c["text"], c["score"]) for c in
            score_npc_candidates_full(lines, npc, situation,
                                      recent_lines, model)]


def score_npc_candidates_full(lines: list[str], npc: Any,
                              situation: str = "",
                              recent_lines: list[str] | tuple[str, ...] = (),
                              model: Any = None
                              ) -> list[dict[str, Any]]:
    """Score candidates with their feature vectors attached — the
    journaling form: the journal stores the real features, so the
    batch fit trains on real rows, never placeholders."""
    out = []
    for line in lines:
        feats = npc_line_features(line, npc, situation, recent_lines)
        if model is not None:
            try:
                pred = float(model.predict_one(feats))
            except Exception:  # noqa: BLE001 - a bad model never kills a line
                pred = 0.0
            score = max(0.0, min(1.0, (pred + 1.0) / 2.0))
        else:
            score = sum(NPC_WEIGHTS[k] * feats[k] for k in NPC_FEATURES)
        out.append({"text": line, "features": feats,
                    "score": round(score, 4)})
    return out


# ── NPC journal ─────────────────────────────────────────────────────────

def _default_npc_dir() -> Path:
    override = (os.environ.get("NM_NPC_LEARN_DIR") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path(os.path.expanduser("~")) / ".nomorals" / "npc_learn"


def _safe_id(value: str) -> str:
    safe = "".join(c for c in (value or "")
                   if c.isalnum() or c in "-_.")[:48]
    return safe or "npc"


class NPCJournal:
    """One JSON doc per (game, npc): journaled lines with their
    candidates, features, chosen index, and (later) grades."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = (Path(data_dir) if data_dir is not None
                         else _default_npc_dir())
        self._lock = threading.Lock()

    def _path(self, game_id: str, npc_id: str) -> Path:
        return (self.data_dir / f"{_safe_id(game_id)}__{_safe_id(npc_id)}.json")

    def _load(self, game_id: str, npc_id: str) -> dict[str, Any]:
        try:
            raw = self._path(game_id, npc_id).read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - a corrupt doc is an empty doc
            return {}

    def _save(self, game_id: str, npc_id: str, doc: dict[str, Any]) -> None:
        path = self._path(game_id, npc_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        tmp.replace(path)

    def journal(self, game_id: str, npc_id: str, *, line: str,
                candidates: list[dict[str, Any]],
                chosen: int, situation: str = "") -> str:
        """Journal one spoken NPC line with its scored candidates.
        ``candidates`` are the ``score_npc_candidates_full`` dicts
        (text + score + REAL feature vector) — the batch fit trains on
        these, so placeholders are never stored. Returns the entry id.
        Never raises."""
        try:
            with self._lock:
                doc = self._load(game_id, npc_id)
                entries = doc.setdefault("entries", [])
                entry = {
                    "id": f"{int(time.time() * 1000)}-{len(entries)}",
                    "ts": time.time(), "line": line,
                    "situation": (situation or "")[:200],
                    "candidates": [{"text": str(c.get("text", "")),
                                    "score": float(c.get("score", 0.0)),
                                    "features": dict(c.get("features") or {})}
                                   for c in (candidates or [])],
                    "chosen": int(chosen), "grade": None, "grades": [],
                }
                entries.append(entry)
                if len(entries) > 500:
                    entries[:] = entries[-500:]
                self._save(game_id, npc_id, doc)
                return str(entry["id"])
        except Exception:  # noqa: BLE001 - journaling never kills a line
            _log.debug("npc journal failed", exc_info=True)
            return ""

    def grade(self, game_id: str, npc_id: str, text: str,
              value: float) -> bool:
        """Grade the most recent ungraded entry with this exact line.
        Returns True when a grade landed."""
        try:
            with self._lock:
                doc = self._load(game_id, npc_id)
                for entry in reversed(doc.get("entries") or []):
                    if entry.get("line") == text and \
                            entry.get("grade") is None:
                        entry["grade"] = max(-1.0, min(1.0, float(value)))
                        entry.setdefault("grades", []).append(
                            {"ts": time.time(), "value": entry["grade"],
                             "source": "owner"})
                        self._save(game_id, npc_id, doc)
                        return True
                return False
        except Exception:  # noqa: BLE001 - grading never kills a line
            _log.debug("npc grade failed", exc_info=True)
            return False

    def entries(self, game_id: str, npc_id: str) -> list[dict[str, Any]]:
        """All journaled entries (graded and ungraded)."""
        return list(self._load(game_id, npc_id).get("entries") or [])

    def rows(self, game_id: str, npc_id: str) -> list[dict[str, Any]]:
        """Graded rows as (features, label) — the batch-fit input.
        Only entries with a grade; the chosen candidate's REAL stored
        feature vector carries the label."""
        doc = self._load(game_id, npc_id)
        rows = []
        for e in doc.get("entries") or []:
            if e.get("grade") is None:
                continue
            cands = e.get("candidates") or []
            chosen = cands[e.get("chosen", 0)] if cands else None
            if not chosen or not chosen.get("features"):
                continue
            rows.append({"features": dict(chosen["features"]),
                         "label": float(e["grade"]),
                         "line": e.get("line", ""),
                         "entry_id": e.get("id", "")})
        return rows


def _hand_npc_prediction(feats: dict[str, float]) -> float:
    """The hand-weight scorer's grade prediction, -1..1 scale."""
    score = sum(NPC_WEIGHTS[k] * float(feats.get(k, 0.0))
                for k in NPC_FEATURES)
    return max(-1.0, min(1.0, 2.0 * score - 1.0))


def fit_npc_scorer(npc_id: str, rows: list[dict[str, Any]], *,
                   store: Any = None,
                   backend: str = "gradboost") -> Any:
    """Batch-fit an NPC's line scorer on graded journal rows.

    Returns the toolkit's :class:`llm.specialists.FitOutcome` —
    "trained" (saved under ``npc-line:<npc_id>``), "held", "insufficient"
    (below ``NPC_MIN_ROWS``), or "unavailable" (backend missing). The
    code rung is the hand-weight scorer on the same held-out slice.
    """
    from ..llm.specialists import LearnerSpec, ModelStore, fit_if_ready

    def _code_score(feats: list[dict[str, float]],
                    labels: list[float]) -> float:
        if not labels:
            return 0.0
        preds = [_hand_npc_prediction(f) for f in feats]
        mae = sum(abs(p - y) for p, y in zip(preds, labels, strict=True)) / len(labels)
        span = (max(labels) - min(labels)) or 1.0
        return max(0.0, 1.0 - mae / span)

    spec = LearnerSpec(job=f"npc-line:{npc_id}", features=NPC_FEATURES,
                       kind="regressor", backend=backend,
                       min_examples=NPC_MIN_ROWS, label_name="grade")
    return fit_if_ready(rows, spec, code_score_fn=_code_score,
                        store=(store or ModelStore()))


def load_npc_scorer(npc_id: str, store: Any = None) -> Any | None:
    """The fitted NPC line scorer, if one was ever promoted."""
    try:
        from ..llm.specialists import ModelStore

        return (store or ModelStore()).load(f"npc-line:{npc_id}")
    except Exception:  # noqa: BLE001 - a bad store never kills a line
        return None


# ── combat policy ───────────────────────────────────────────────────────

TACTICS_FEATURES = ("self_hp", "foe_hp", "foe_hp_low", "potions",
                    "fury_ready", "focused")
TACTICS_ACTIONS = ("attack", "focus", "fury", "defend", "potion")
#: Floor for the tactics policy (graded fight rounds accumulate slowly).
TACTICS_MIN_ROWS = 40


def _default_tactics_dir() -> Path:
    override = (os.environ.get("NM_TACTICS_DIR") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path(os.path.expanduser("~")) / ".nomorals" / "tactics"


class TacticsJournal:
    """Journaled combat rounds: state -> action -> fight outcome.

    One JSON doc per game. A round is recorded at decision time with
    its state and action; the fight outcome attaches later via
    ``grade_fight`` (the delayed label — defend/focus are investments,
    and only the fight outcome grades them honestly)."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = (Path(data_dir) if data_dir is not None
                         else _default_tactics_dir())
        self._lock = threading.Lock()

    def _path(self, game_id: str) -> Path:
        return self.data_dir / f"{_safe_id(game_id)}.json"

    def _load(self, game_id: str) -> dict[str, Any]:
        try:
            raw = self._path(game_id).read_text(encoding="utf-8")
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except Exception:  # noqa: BLE001 - a corrupt doc is an empty doc
            return {}

    def _save(self, game_id: str, doc: dict[str, Any]) -> None:
        path = self._path(game_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
        tmp.replace(path)

    def record_round(self, game_id: str, state: dict[str, float],
                     action: str, *, payoff: float = 0.0,
                     won: bool | None = None, fight_id: str = "",
                     stats: dict[str, Any] | None = None) -> str:
        """Record one combat round. ``stats`` is the raw
        (self_stats, foe_stats) pair when the caller has it — the
        code-rung comparison uses the real math on it. Never raises."""
        try:
            with self._lock:
                doc = self._load(game_id)
                rounds = doc.setdefault("rounds", [])
                entry = {
                    "id": f"{int(time.time() * 1000)}-{len(rounds)}",
                    "ts": time.time(),
                    "fight_id": fight_id or "",
                    "state": {k: float(state.get(k, 0.0) or 0.0)
                              for k in TACTICS_FEATURES},
                    "action": str(action or "attack"),
                    "payoff": float(payoff or 0.0),
                    "won": (None if won is None else bool(won)),
                }
                if stats is not None:
                    entry["stats"] = stats
                rounds.append(entry)
                if len(rounds) > 2000:
                    rounds[:] = rounds[-2000:]
                self._save(game_id, doc)
                return str(entry["id"])
        except Exception:  # noqa: BLE001 - journaling never kills a move
            _log.debug("tactics journal failed", exc_info=True)
            return ""

    def grade_fight(self, game_id: str, fight_id: str,
                    won: bool) -> int:
        """Attach the fight outcome to every ungraded round of one
        fight. Returns how many rounds were graded."""
        try:
            with self._lock:
                doc = self._load(game_id)
                n = 0
                for entry in doc.get("rounds") or []:
                    if entry.get("fight_id") == fight_id and \
                            entry.get("won") is None:
                        entry["won"] = bool(won)
                        n += 1
                if n:
                    self._save(game_id, doc)
                return n
        except Exception:  # noqa: BLE001 - grading never kills a move
            _log.debug("tactics grade failed", exc_info=True)
            return 0

    def graded_rows(self, game_id: str) -> list[dict[str, Any]]:
        """Rounds with a fight outcome, as (features, label) rows.
        Features = state + action one-hot (+ raw stats when present,
        for the code-rung comparison)."""
        doc = self._load(game_id)
        rows = []
        for e in doc.get("rounds") or []:
            if e.get("won") is None:
                continue
            feats = dict(e.get("state") or {})
            for act in TACTICS_ACTIONS:
                feats[f"action_{act}"] = (
                    1.0 if e.get("action") == act else 0.0)
            stats = e.get("stats") or {}
            if stats:
                feats["_self"] = stats.get("self") or {}
                feats["_foe"] = stats.get("foe") or {}
            rows.append({"features": feats,
                         "label": 1 if e.get("won") else 0})
        return rows


def _math_action_from_state(state: dict[str, float]) -> str:
    """The math's action from ratio-form state — the same thresholds
    as ``GameMind._combat_math`` (healthy + fury ready -> fury, low
    hp + potions -> potion, otherwise attack). Used for the held-out
    comparison when the journaled round carries no raw stats."""
    self_hp = float(state.get("self_hp", 1.0) or 1.0)
    potions = float(state.get("potions", 0.0) or 0.0)
    fury_ready = float(state.get("fury_ready", 0.0) or 0.0)
    if self_hp < 0.3 and potions > 0:
        return "potion"
    if fury_ready > 0.5:
        return "fury"
    return "attack"


class TacticsPolicy:
    """A fitted combat policy: picks the legal action with the best
    predicted fight-win probability. Never raises; None means the
    math carries.

    The policy only ever picks an action it has EVIDENCE for — one
    the journal actually took. An untaken action's P(win) is the
    classifier extrapolating into the void (a never-taken "focus"
    once scored higher than a proven "fury"), so unseen actions are
    not candidates at all; the math carries those.
    """

    def __init__(self, model: Any,
                 actions: tuple[str, ...] | None = None) -> None:
        self.model = model
        seen: tuple[str, ...] | None = actions
        if seen is None:
            # provenance rides in the saved report's detail
            seen = ()
            report = getattr(model, "report", None)
            if report is not None:
                detail = getattr(report, "detail", None) or {}
                seen = tuple(detail.get("actions_seen") or ())
        self.actions = tuple(seen) if seen else tuple(TACTICS_ACTIONS)

    def pick_action(self, state: dict[str, float],
                    legal: tuple[str, ...] | None = None) -> str | None:
        """Best legal action for ``state``, or None (math carries)."""
        try:
            actions = [a for a in (legal or TACTICS_ACTIONS)
                       if a in self.actions]
            best: str | None = None
            best_p = -1.0
            for act in actions:
                if act == "potion" and float(state.get("potions", 0) or 0) <= 0:
                    continue
                if act == "fury" and float(state.get("fury_ready", 0) or 0) <= 0:
                    continue
                feats = dict(state)
                for a in TACTICS_ACTIONS:
                    feats[f"action_{a}"] = 1.0 if a == act else 0.0
                p = float(self.model.predict_one(feats))
                if p > best_p:
                    best_p, best = p, act
            return best
        except Exception:  # noqa: BLE001 - a bad policy never kills a move
            _log.debug("tactics policy failed", exc_info=True)
            return None


def fit_tactics_policy(game_id: str, journal: TacticsJournal, *,
                       store: Any = None,
                       backend: str = "gradboost") -> Any:
    """Batch-fit the combat policy on graded fight rounds.

    Returns the toolkit's :class:`llm.specialists.FitOutcome` —
    "trained" (saved under ``tactics:<game_id>``), "held",
    "insufficient" (below ``TACTICS_MIN_ROWS``), or "unavailable".
    The code rung is the math's action agreement on the same held-out
    slice: the math "predicts" a win exactly when it would have taken
    the journaled action.
    """
    from ..llm.specialists import LearnerSpec, ModelStore, fit_if_ready

    rows = journal.graded_rows(game_id)
    features = TACTICS_FEATURES + tuple(f"action_{a}"
                                        for a in TACTICS_ACTIONS)

    def _code_score(feats: list[dict[str, Any]],
                    labels: list[float]) -> float:
        if not labels:
            return 0.0
        correct = 0
        for f, y in zip(feats, labels, strict=True):
            stats = f.get("_self") or {}
            foe = f.get("_foe") or {}
            if stats and foe:
                try:
                    from .ai import GameMind

                    math_action = GameMind._combat_math(stats, foe, 0).get(
                        "action", "attack")
                except Exception:  # noqa: BLE001 - fall back to ratio math
                    math_action = _math_action_from_state(f)
            else:
                math_action = _math_action_from_state(f)
            pred = 1 if math_action == max(
                (a for a in TACTICS_ACTIONS
                 if float(f.get(f"action_{a}", 0.0) or 0.0) > 0.5),
                default="attack") else 0
            if pred == int(y):
                correct += 1
        return correct / len(labels)

    seen = sorted({a for r in rows for a in TACTICS_ACTIONS
                   if float(r["features"].get(f"action_{a}", 0.0) or 0.0)
                   > 0.5})
    spec = LearnerSpec(job=f"tactics:{game_id}", features=features,
                       kind="classifier", backend=backend,
                       min_examples=TACTICS_MIN_ROWS, label_name="won")
    outcome = fit_if_ready(rows, spec, code_score_fn=_code_score,
                           store=(store or ModelStore()),
                           detail={"actions_seen": seen})
    if outcome.status == "trained" and outcome.model is not None:
        outcome.model = TacticsPolicy(outcome.model, actions=tuple(seen))
    return outcome


def load_tactics_policy(game_id: str, store: Any = None
                        ) -> TacticsPolicy | None:
    """The fitted combat policy for a game, if one was ever promoted."""
    try:
        from ..llm.specialists import ModelStore

        model = (store or ModelStore()).load(f"tactics:{game_id}")
        return TacticsPolicy(model) if model is not None else None
    except Exception:  # noqa: BLE001 - a bad store never kills a move
        return None


def record_combat_round(game_id: str, state: dict[str, float],
                        action: str, *, payoff: float = 0.0,
                        fight_id: str = "",
                        stats: dict[str, Any] | None = None) -> str:
    """Module-level convenience: record one round in the default
    tactics journal."""
    return TacticsJournal().record_round(
        game_id, state, action, payoff=payoff, fight_id=fight_id,
        stats=stats)


def grade_fight(game_id: str, fight_id: str, won: bool) -> int:
    """Module-level convenience: grade one fight's rounds."""
    return TacticsJournal().grade_fight(game_id, fight_id, won)
