"""Narration specialist — the arena narrator's learned line scorer.

Journal → grade → retrain, the Phase 2 shape (see
``llm/specialists.py``): graded narration lines train a small
regressor that scores candidate narration lines; the GameMaster asks
the narration bridge for candidates and the scorer picks the line.
The brain GENERATES candidates; the scorer DECIDES. The brain never
narrates as a rung.

The code rung is the hand-weight scorer over the same five features;
a fitted model is promoted only when it beats the hand weights on a
held-out slice (``llm.specialists.fit_if_ready``). Below the floor
(``NARRATION_MIN_ROWS``) the hand weights carry.

Storage mirrors the NPC learner (``games/npc_learn.py``): one JSON doc
per game, atomic tmp+replace writes, lazy load, no import-time I/O. A
learn failure never eats a line.
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
    "NARRATION_FEATURES", "NARRATION_WEIGHTS", "NARRATION_MIN_ROWS",
    "narration_line_features", "score_narration_candidates",
    "score_narration_candidates_full",
    "NarrationJournal", "fit_narration_scorer", "load_narration_scorer",
]

_log = get_logger(__name__)

# ── line scorer ─────────────────────────────────────────────────────────

NARRATION_FEATURES = ("event_fit", "mood_fit", "vividness", "novelty",
                      "length_fit")
NARRATION_WEIGHTS = {"event_fit": 0.35, "mood_fit": 0.20,
                     "vividness": 0.20, "novelty": 0.10,
                     "length_fit": 0.15}
#: Floor for the narration scorer (graded lines accumulate slowly).
NARRATION_MIN_ROWS = 30

#: The DM moods' valence, for mood_fit (see games.gamemaster.DM_MOODS).
_MOOD_VALENCE = {"neutral": 0.0, "grim": -0.6, "whimsical": 0.4,
                 "epic": 0.6, "deadpan": -0.2}

#: Vivid/action words — the code rung's vividness signal. A narration
#: line that moves, crashes and burns reads as a scene; one that
#: merely reports reads as a log entry.
_VIVID_WORDS = frozenset([
    "blaze", "blazing", "roar", "roars", "roaring", "crash", "crashes",
    "crashing", "shatter", "shatters", "shattering", "thunder",
    "thunders", "thundering", "slash", "slashes", "slashing", "surge",
    "surges", "surging", "erupt", "erupts", "erupting", "howl", "howls",
    "howling", "gleam", "gleams", "gleaming", "tremble", "trembles",
    "trembling", "burn", "burns", "burning", "storm", "storms",
    "storming", "strike", "strikes", "striking", "rend", "rends",
    "rending", "clash", "clashes", "clashing", "flare", "flares",
    "flaring", "snap", "snaps", "snapping", "whirl", "whirls",
    "whirling", "sear", "sears", "searing", "pulse", "pulses",
    "pulsing", "quake", "quakes", "quaking", "rip", "rips", "ripping",
    "tear", "tears", "tearing",
])

_POSITIVE_WORDS = frozenset(
    ["love", "lovely", "great", "wonderful", "brilliant", "happy", "glad",
     "joy", "delight", "win", "won", "triumph", "cheer", "laugh", "smile",
     "kind", "warm", "friend", "glorious", "victory", "heroic"])
_NEGATIVE_WORDS = frozenset(
    ["hate", "terrible", "awful", "sad", "angry", "fear", "dead", "death",
     "blood", "pain", "doom", "dark", "grim", "cruel", "fall", "falls",
     "fallen", "defeat", "lost", "broken"])

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


def narration_line_features(line: str, event: str, mood: str = "neutral",
                            recent_lines: list[str] | tuple[str, ...] = ()
                            ) -> dict[str, float]:
    """Feature vector for one candidate narration line, all 0–1.

    ``event`` is the bare game event the line narrates; ``mood`` is the
    DM mood it must stay in. The features are the narration's own fit,
    not the prompt's vocabulary: the event's words surfacing in the
    line, mood fit, vividness, novelty vs recent lines, and length fit.
    """
    text = line or ""
    line_words = _content_words(text)
    event_words = {w for w in _WORD_RE.findall((event or "").lower())
                   if len(w) > 3}
    event_fit = (len(event_words & line_words) / len(event_words)
                 if event_words else 0.5)
    mood_val = _MOOD_VALENCE.get((mood or "neutral").lower(), 0.0)
    mood_fit = 1.0 - abs(_line_valence(text) - mood_val) / 2.0
    vivid_hits = len(line_words & _VIVID_WORDS)
    vividness = min(1.0, vivid_hits / 3.0)
    if recent_lines:
        overlaps = []
        for old in recent_lines:
            old_words = _content_words(old)
            if line_words and old_words:
                overlaps.append(len(line_words & old_words)
                                / len(line_words | old_words))
        novelty = 1.0 - max(overlaps, default=0.0)
    else:
        novelty = 1.0
    ideal = 120.0
    length_fit = max(0.0, 1.0 - abs(len(text) - ideal) / ideal)
    return {"event_fit": round(event_fit, 4),
            "mood_fit": round(mood_fit, 4),
            "vividness": round(vividness, 4),
            "novelty": round(novelty, 4),
            "length_fit": round(length_fit, 4)}


def _hand_narration_prediction(feats: dict[str, float]) -> float:
    """The hand-weight scorer's grade prediction, -1..1 scale."""
    score = sum(NARRATION_WEIGHTS[k] * float(feats.get(k, 0.0))
                for k in NARRATION_FEATURES)
    return max(-1.0, min(1.0, 2.0 * score - 1.0))


def score_narration_candidates(lines: list[str], event: str,
                               mood: str = "neutral",
                               recent_lines: list[str] | tuple[str, ...] = (),
                               model: Any = None
                               ) -> list[tuple[str, float]]:
    """Score candidate narration lines: (text, score 0..1), input order.

    ``model`` is a fitted :class:`llm.specialists.LearnerModel` from
    :func:`fit_narration_scorer` — when given, the LEARNED scorer scores
    (its -1..1 grade prediction rescaled to 0..1); otherwise the hand
    weights score. The learned scorer is only handed here after it
    beat the hand weights on a held-out slice.
    """
    return [(c["text"], c["score"]) for c in
            score_narration_candidates_full(lines, event, mood,
                                            recent_lines, model)]


def score_narration_candidates_full(lines: list[str], event: str,
                                    mood: str = "neutral",
                                    recent_lines: list[str] | tuple[str, ...] = (),
                                    model: Any = None
                                    ) -> list[dict[str, Any]]:
    """Score candidates with their feature vectors attached — the
    journaling form: the journal stores the real features, so the
    batch fit trains on real rows, never placeholders."""
    out = []
    for line in lines:
        feats = narration_line_features(line, event, mood, recent_lines)
        if model is not None:
            try:
                pred = float(model.predict_one(feats))
            except Exception:  # noqa: BLE001 - a bad model never kills a line
                pred = 0.0
            score = max(0.0, min(1.0, (pred + 1.0) / 2.0))
        else:
            score = sum(NARRATION_WEIGHTS[k] * feats[k]
                        for k in NARRATION_FEATURES)
        out.append({"text": line, "features": feats,
                    "score": round(score, 4)})
    return out


# ── narration journal ───────────────────────────────────────────────────

def _default_narration_dir() -> Path:
    override = (os.environ.get("NM_NARRATION_LEARN_DIR") or "").strip()
    if override:
        return Path(os.path.expanduser(override))
    return Path(os.path.expanduser("~")) / ".nomorals" / "narration_learn"


def _safe_id(value: str) -> str:
    safe = "".join(c for c in (value or "")
                   if c.isalnum() or c in "-_.")[:48]
    return safe or "game"


class NarrationJournal:
    """One JSON doc per game: journaled narration lines with their
    candidates, features, chosen index, and (later) grades."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = (Path(data_dir) if data_dir is not None
                         else _default_narration_dir())
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

    def journal(self, game_id: str, *, line: str,
                candidates: list[dict[str, Any]],
                chosen: int, event: str = "", mood: str = "",
                prompt_version: str = "base") -> str:
        """Journal one narration line with its scored candidates.
        ``candidates`` are the ``score_narration_candidates_full``
        dicts (text + score + REAL feature vector) — the batch fit
        trains on these, so placeholders are never stored. Returns the
        entry id. Never raises."""
        try:
            with self._lock:
                doc = self._load(game_id)
                entries = doc.setdefault("entries", [])
                entry = {
                    "id": f"{int(time.time() * 1000)}-{len(entries)}",
                    "ts": time.time(), "line": line,
                    "event": (event or "")[:200],
                    "mood": str(mood or ""),
                    "prompt_version": str(prompt_version or "base"),
                    "candidates": [{"text": str(c.get("text", "")),
                                    "score": float(c.get("score", 0.0)),
                                    "features": dict(c.get("features") or {})}
                                   for c in (candidates or [])],
                    "chosen": int(chosen), "grade": None, "grades": [],
                }
                entries.append(entry)
                if len(entries) > 500:
                    entries[:] = entries[-500:]
                self._save(game_id, doc)
                return str(entry["id"])
        except Exception:  # noqa: BLE001 - journaling never kills a line
            _log.debug("narration journal failed", exc_info=True)
            return ""

    def grade(self, game_id: str, text: str, value: float) -> bool:
        """Grade the most recent ungraded entry with this exact line.
        Returns True when a grade landed."""
        try:
            with self._lock:
                doc = self._load(game_id)
                for entry in reversed(doc.get("entries") or []):
                    if entry.get("line") == text and \
                            entry.get("grade") is None:
                        entry["grade"] = max(-1.0, min(1.0, float(value)))
                        entry.setdefault("grades", []).append(
                            {"ts": time.time(), "value": entry["grade"],
                             "source": "owner"})
                        self._save(game_id, doc)
                        return True
                return False
        except Exception:  # noqa: BLE001 - grading never kills a line
            _log.debug("narration grade failed", exc_info=True)
            return False

    def entries(self, game_id: str) -> list[dict[str, Any]]:
        """All journaled entries (graded and ungraded)."""
        return list(self._load(game_id).get("entries") or [])

    def rows(self, game_id: str) -> list[dict[str, Any]]:
        """Graded rows as (features, label) — the batch-fit input.
        Only entries with a grade; the chosen candidate's REAL stored
        feature vector carries the label."""
        doc = self._load(game_id)
        rows = []
        for e in doc.get("entries") or []:
            if e.get("grade") is None:
                continue
            cands = e.get("candidates") or []
            idx = e.get("chosen", 0)
            chosen = cands[idx] if isinstance(idx, int) and 0 <= idx < len(cands) else None
            if not chosen or not chosen.get("features"):
                continue
            rows.append({"features": dict(chosen["features"]),
                         "label": float(e["grade"]),
                         "line": e.get("line", ""),
                         "event": e.get("event", ""),
                         "entry_id": e.get("id", "")})
        return rows


def fit_narration_scorer(rows: list[dict[str, Any]], *,
                         store: Any = None,
                         backend: str = "gradboost") -> Any:
    """Batch-fit the narration line scorer on graded journal rows.

    Returns the toolkit's :class:`llm.specialists.FitOutcome` —
    "trained" (saved under ``narration-line``), "held", "insufficient"
    (below ``NARRATION_MIN_ROWS``), or "unavailable" (backend missing).
    The code rung is the hand-weight scorer on the same held-out slice.
    """
    from ..llm.specialists import LearnerSpec, ModelStore, fit_if_ready

    def _code_score(feats: list[dict[str, float]],
                    labels: list[float]) -> float:
        if not labels:
            return 0.0
        preds = [_hand_narration_prediction(f) for f in feats]
        mae = sum(abs(p - y) for p, y in zip(preds, labels, strict=True)) / len(labels)
        span = (max(labels) - min(labels)) or 1.0
        return max(0.0, 1.0 - mae / span)

    spec = LearnerSpec(job="narration-line", features=NARRATION_FEATURES,
                       kind="regressor", backend=backend,
                       min_examples=NARRATION_MIN_ROWS, label_name="grade")
    return fit_if_ready(rows, spec, code_score_fn=_code_score,
                        store=(store or ModelStore()))


def load_narration_scorer(store: Any = None) -> Any | None:
    """The fitted narration line scorer, if one was ever promoted."""
    try:
        from ..llm.specialists import ModelStore

        return (store or ModelStore()).load("narration-line")
    except Exception:  # noqa: BLE001 - a bad store never kills a line
        return None
