"""Character learn loop: journal + grade + adjust.

A character that LEARNS and JUDGES for itself, not just rides the
brain's output. The pattern, ported from the repo's own precedents
(never copied from another repo):

- journal: every spoken line is recorded with its candidates, their
  feature vectors, and the chosen index — the decisions/journal
  record-before-outcome shape.
- grade: outcomes arrive later (owner grades, conversation outcomes)
  and attach to entries by id; grades are append-only revisions.
- adjust: grades move scorer weights (bounded), style preferences,
  and — only under an explicit opt-in flag — persona traits under a
  hard leash. The character's taste is learned, its identity is not
  rewritten by accident.
- candidate grading: when a caller asks for N > 1 candidates, the
  character's OWN scorer picks the most in-character line. The
  generator stays the brain (SuggestFn); the character judges and
  picks. No fake "own generator" claims.

Storage mirrors the failure_store durability pattern: one JSON doc per
character, atomic tmp+replace writes, lazy load, no import-time I/O,
path resolution inside try, save/load never raise out of the caller.
A learn failure must never eat a line — speak() journals best-effort
and always returns the line.

ML note: the journal stores (features, label) rows, and
``export_training_data()`` emits them, so a future sklearn/CatBoost
batch fit trains directly on this doc. No batch fit ships in v1:
fitting on a handful of graded rows would be fake learning, and
CatBoost is not installed in this environment (sklearn is, but the
volume gate comes first). Any future fit path must lazy-import and
return a clear "unavailable" reason, never fake success.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from .failures import record_failure
from .voice import VoiceFingerprint, consistency, fingerprint

__all__ = [
    "LearnStore", "LearnHandle", "ScoredCandidate",
    "attach", "detach", "get_handle",
    "score_candidates", "extract_features",
    "DEFAULT_WEIGHTS", "MAX_ENTRIES", "SESSION_GRADES",
    "GRADE_SOURCES", "FEATURES",
]

# ── constants ──────────────────────────────────────────────────────────
MAX_ENTRIES = 500           # journal cap per character; oldest pruned
FEATURES = ("consistency", "catchphrase", "length_fit", "punct_fit",
            "novelty")
DEFAULT_WEIGHTS = {"consistency": 0.5, "catchphrase": 0.1,
                   "length_fit": 0.15, "punct_fit": 0.15,
                   "novelty": 0.1}
WEIGHT_MIN = 0.0
WEIGHT_MAX = 2.0
LR_EXPLICIT = 0.05          # weight step per explicit/system grade unit
LR_SESSION = 0.0125         # session grades move 1/4 as far (noisy attribution)
PREFS_MIN_POSITIVE = 10     # graded-positive lines before prefs become targets
TRAIT_LEASH = 0.10          # max persona drift from seed, per trait
TRAIT_STEP = 0.02           # per-grade persona nudge (opt-in only)
TRAIT_STRONG = 0.7          # only strong traits move (identity anchors)
PENDING_CAP = 200           # session-pending entry ids kept per handle

GRADE_SOURCES = ("owner", "session", "system")
_SOURCE_WEIGHT = {"owner": 1.0, "session": 0.25, "system": 1.0}
_SOURCE_LR = {"owner": LR_EXPLICIT, "session": LR_SESSION,
              "system": LR_EXPLICIT}

# Session-kind -> grade (converse outcome mapping, named and small —
# not a sentiment model pretending to understand the dialogue).
SESSION_GRADES = {
    "deep_conversation": 0.5, "laughed_together": 0.5, "praised": 0.5,
    "good_conversation": 0.25, "won_game": 0.25, "disagreement": -0.25,
    "lost_game": -0.1, "argument": -0.5, "insulted": -0.5,
    "betrayed": -1.0,
}

_WORD_RE = re.compile(r"[a-z']+")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _content_words(text: str) -> set[str]:
    return set(_WORD_RE.findall((text or "").lower()))


def _word_overlap(a: str, b: str) -> float:
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _sent_stats(text: str) -> tuple[float, float, float, float]:
    """(mean sentence len, emoji rate, question rate, exclaim rate)."""
    fp = fingerprint([text])
    return (fp.mean_sentence_len, fp.emoji_rate, fp.question_rate,
            fp.exclaim_rate)


def _fit(value: float, target: float, spread: float) -> float:
    """0–1 fit of a value against a target within a spread."""
    if target <= 0:
        return 0.5
    z = abs(value - target) / max(spread, 2.0)
    return max(0.0, 1.0 - z / 3.0)


def _prefs_targets(doc: dict[str, Any]) -> dict[str, Any] | None:
    """Revealed-taste targets, once enough positive grades exist."""
    prefs = doc.get("prefs") or {}
    if int(prefs.get("n_graded_positive", 0)) < PREFS_MIN_POSITIVE:
        return None
    if prefs.get("mean_len") is None:
        return None
    return prefs


def extract_features(text: str, char: Any,
                     recent_lines: list[str] | tuple[str, ...] = (),
                     prefs: dict[str, Any] | None = None) -> dict[str, float]:
    """Feature vector for one candidate line, all 0–1.

    ``char`` needs ``voice_fingerprint`` (dict) and ``expression``.
    ``prefs`` (learned taste) replaces fingerprint targets once active;
    without enough fingerprint data the structural fits are 0.5
    (not enough data to judge — the voice.py rule, honored here).
    Persona trait words are deliberately NOT a feature: trait names
    ("witty", "curious") rarely surface literally in lines, and persona
    already conditions the prompt — the scorer judges voice, not the
    prompt's vocabulary.
    """
    text = text or ""
    cps = ((char.expression or {}).get("catchphrases") or [])
    fp = VoiceFingerprint.from_dict(char.voice_fingerprint or {})
    feats: dict[str, float] = {}
    feats["consistency"] = consistency(fp, text, cps)
    low = text.lower()
    feats["catchphrase"] = 1.0 if any(
        (c or "").lower() in low for c in cps if c) else 0.0
    mean_len, emo, que, exc = _sent_stats(text)
    if prefs is not None:
        feats["length_fit"] = _fit(mean_len, float(prefs["mean_len"]), 4.0)
        feats["punct_fit"] = sum(
            1.0 - min(1.0, abs(val - float(prefs[key])) * 2.0)
            for val, key in ((emo, "emoji_rate"), (que, "question_rate"),
                             (exc, "exclaim_rate"))) / 3.0
    elif fp.n_samples >= 5:
        feats["length_fit"] = _fit(mean_len, fp.mean_sentence_len,
                                   fp.sentence_len_spread)
        feats["punct_fit"] = sum(
            1.0 - min(1.0, abs(val - getattr(fp, key)) * 2.0)
            for val, key in ((emo, "emoji_rate"), (que, "question_rate"),
                             (exc, "exclaim_rate"))) / 3.0
    else:
        feats["length_fit"] = 0.5
        feats["punct_fit"] = 0.5
    if recent_lines:
        feats["novelty"] = 1.0 - max(
            (_word_overlap(text, old) for old in recent_lines), default=0.0)
    else:
        feats["novelty"] = 1.0
    return feats


class ScoredCandidate:
    """One candidate with its features, score, and reasons."""

    def __init__(self, text: str, features: dict[str, float],
                 score: float) -> None:
        self.text = text
        self.features = features
        self.score = score

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "features": self.features,
                "score": round(self.score, 4)}

    def reasons(self) -> str:
        best = sorted(self.features.items(), key=lambda kv: -kv[1])[:2]
        return ", ".join(f"{k} {v:.2f}" for k, v in best)


def score_candidates(texts: list[str], char: Any,
                     weights: dict[str, float] | None = None,
                     recent_lines: list[str] | tuple[str, ...] = (),
                     prefs: dict[str, Any] | None = None,
                     ) -> list[ScoredCandidate]:
    """Score candidates with the character's own scorer.

    Pure function (no I/O, no journaling) — usable from speak() even
    without an attached handle. Returns candidates in input order with
    scores attached; the caller picks the best (ties: earliest wins).
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        for k in FEATURES:
            if k in weights:
                try:
                    w[k] = _clamp(float(weights[k]), WEIGHT_MIN, WEIGHT_MAX)
                except (TypeError, ValueError) as e:
                    # bad caller weight: keep the default, but say so
                    record_failure("learn.score_weights", e)
    out = []
    for t in texts:
        feats = extract_features(t, char, recent_lines, prefs)
        score = sum(w[k] * feats[k] for k in FEATURES)
        out.append(ScoredCandidate(t, feats, score))
    return out


def pick_best(scored: list[ScoredCandidate]) -> int:
    """Index of the best candidate; ties go to the EARLIEST candidate
    (deterministic, documented, tested)."""
    best_i, best_s = 0, None
    for i, c in enumerate(scored):
        if best_s is None or c.score > best_s:
            best_i, best_s = i, c.score
    return best_i


# ── storage ────────────────────────────────────────────────────────────
def _empty_doc(char_id: str) -> dict[str, Any]:
    return {
        "version": 1, "char_id": char_id, "total": 0, "entries": [],
        "weights": dict(DEFAULT_WEIGHTS),
        "prefs": {"mean_len": None, "emoji_rate": None,
                  "question_rate": None, "exclaim_rate": None,
                  "n_graded_positive": 0},
        "trait_offsets": {}, "seed_persona": {},
        "stats": {"graded": 0, "mean_grade": None},
    }


def _safe_id(char_id: str) -> str:
    return "".join(c for c in (char_id or "") if c.isalnum() or c in "-_")[:32]


class LearnStore:
    """Per-character learn docs under ``<data_dir>/learn/``.

    Mirrors the failure_store durability pattern: atomic tmp+replace,
    lazy load, no import-time I/O, path resolution inside try, save
    and load never raise out of the caller. In-memory cache per
    char_id (one store instance = one cache) so handles sharing a
    store see each other's writes.
    """

    def __init__(self, data_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else None
        self._lock = threading.RLock()
        self._cache: dict[str, dict[str, Any]] = {}

    def _resolve_dir(self) -> Path | None:
        """Learn dir, resolved fresh every call (never cached at
        import or construction — the #141 home-resolution lesson)."""
        try:
            from .store import default_character_dir
            base = self.data_dir or default_character_dir()
            return Path(base) / "learn"
        except Exception as e:
            record_failure("learn.resolve_dir", e)
            return None

    def _path(self, char_id: str) -> Path | None:
        d = self._resolve_dir()
        return (d / f"{_safe_id(char_id)}.json") if d else None

    def doc(self, char_id: str) -> dict[str, Any]:
        """Cached doc, loading from disk on first miss. Corrupt or
        unreadable -> empty doc (the file is overwritten on next save:
        recovery, not silent loss — the failure is recorded)."""
        with self._lock:
            if char_id in self._cache:
                return self._cache[char_id]
            d = _empty_doc(char_id)
            p = self._path(char_id)
            if p is not None and p.exists():
                try:
                    raw = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(raw, dict) and raw.get("version") == 1:
                        d = raw
                        d.setdefault("weights", dict(DEFAULT_WEIGHTS))
                        d.setdefault("prefs", _empty_doc(char_id)["prefs"])
                        d.setdefault("trait_offsets", {})
                        d.setdefault("seed_persona", {})
                        d.setdefault("stats", {"graded": 0,
                                               "mean_grade": None})
                        d.setdefault("entries", [])
                        d.setdefault("total", len(d["entries"]))
                    else:
                        record_failure(
                            "learn.doc:bad_version",
                            ValueError(f"unexpected learn doc for {char_id}"))
                except Exception as e:
                    record_failure("learn.doc:load", e)
            self._cache[char_id] = d
            return d

    def save(self, char_id: str) -> bool:
        """Atomic persist of the cached doc. Never raises; False on
        any failure (callers keep the in-memory doc)."""
        with self._lock:
            d = self._cache.get(char_id)
            if d is None:
                return True
            try:
                p = self._path(char_id)
                if p is None:
                    return False
                p.parent.mkdir(parents=True, exist_ok=True)
                tmp = p.with_suffix(".tmp")
                tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False),
                               encoding="utf-8")
                try:
                    with open(tmp, "r+b") as fh:
                        os.fsync(fh.fileno())
                except OSError as e:
                    # fsync is best-effort; the atomic replace still holds
                    record_failure("learn.save:fsync", e)
                tmp.replace(p)
                return True
            except Exception as e:
                record_failure("learn.save", e)
                return False

    def journal(self, char_id: str, entry: dict[str, Any]) -> str:
        """Append one entry; prunes oldest past MAX_ENTRIES; total
        counts everything ever recorded. Returns the entry id."""
        with self._lock:
            d = self.doc(char_id)
            d["total"] = int(d.get("total", 0)) + 1
            entry["id"] = f"L-{d['total']:06d}"
            d["entries"].append(entry)
            while len(d["entries"]) > MAX_ENTRIES:
                d["entries"].pop(0)
            self.save(char_id)
            return entry["id"]

    def grade(self, char_id: str, entry_id: str, value: float,
              source: str, note: str = "") -> dict[str, Any]:
        """Attach a grade to an entry and adjust. Fail-fast on bad
        input (unknown entry, bad source, out-of-range value) — the
        grading path is interactive and must say so, never silently
        clamp or swallow. Returns the updated entry."""
        if source not in GRADE_SOURCES:
            raise ValueError(f"source must be one of {GRADE_SOURCES}")
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError("grade value must be a number")
        if not (-1.0 <= value <= 1.0):
            raise ValueError("grade value must be in [-1, 1]")
        if math.isnan(value) or math.isinf(value):
            raise ValueError("grade value must be finite")
        with self._lock:
            d = self.doc(char_id)
            entry = next((e for e in d["entries"]
                          if e.get("id") == entry_id), None)
            if entry is None:
                raise ValueError(f"no journal entry {entry_id!r} "
                                 f"for character {char_id!r}")
            entry.setdefault("grades", []).append({
                "value": value, "source": source, "ts": time.time(),
                "note": note or ""})
            # aggregate: source-weighted mean (session grades count 1/4)
            tot = sum(_SOURCE_WEIGHT[g["source"]] for g in entry["grades"])
            entry["grade"] = (
                sum(_SOURCE_WEIGHT[g["source"]] * g["value"]
                    for g in entry["grades"]) / tot) if tot else None
            self._adjust(d, entry, value, source)
            stats = d.setdefault("stats", {"graded": 0, "mean_grade": None})
            stats["graded"] = int(stats.get("graded", 0)) + 1
            prev = stats.get("mean_grade")
            n = stats["graded"]
            stats["mean_grade"] = (value if prev is None
                                   else prev + (value - prev) / n)
            self.save(char_id)
            return entry

    def _adjust(self, d: dict[str, Any], entry: dict[str, Any],
                value: float, source: str) -> None:
        """Bounded adjustment from one grade. Weights move by
        lr * grade * feature (chosen line's features); prefs track
        positive-grade line shapes; persona moves ONLY under the
        handle's opt-in flag (see LearnHandle) — the store itself
        never touches persona."""
        lr = _SOURCE_LR[source]
        weights = d.setdefault("weights", dict(DEFAULT_WEIGHTS))
        cands = entry.get("candidates") or []
        chosen = cands[entry.get("chosen", 0)] if cands else None
        feats = (chosen or {}).get("features") or {}
        for k in FEATURES:
            if k in feats:
                weights[k] = _clamp(
                    float(weights.get(k, DEFAULT_WEIGHTS[k]))
                    + lr * value * float(feats[k]),
                    WEIGHT_MIN, WEIGHT_MAX)
        if value > 0:
            prefs = d.setdefault("prefs", _empty_doc("")["prefs"])
            n = int(prefs.get("n_graded_positive", 0))
            mean_len, emo, que, exc = _sent_stats(entry.get("line", ""))
            for key, val in (("mean_len", mean_len), ("emoji_rate", emo),
                             ("question_rate", que), ("exclaim_rate", exc)):
                old = prefs.get(key)
                prefs[key] = val if old is None else (old * n + val) / (n + 1)
            prefs["n_graded_positive"] = n + 1

    def report(self, char_id: str) -> str:
        """Human-readable learned state for the `learned` tool action."""
        with self._lock:
            d = self.doc(char_id)
            entries = d.get("entries") or []
            stats = d.get("stats") or {}
            graded = [e for e in entries if e.get("grade") is not None]
            lines = [f"📓 learn journal for {char_id}: "
                     f"{len(entries)} entries "
                     f"({int(d.get('total', 0))} ever recorded), "
                     f"{len(graded)} graded"]
            mg = stats.get("mean_grade")
            lines.append(f"  mean grade: "
                         f"{mg:.2f}" if mg is not None else "  mean grade: —")
            w = d.get("weights") or {}
            lines.append("  weights: " + ", ".join(
                f"{k} {float(w.get(k, DEFAULT_WEIGHTS[k])):.2f}"
                for k in FEATURES))
            prefs = d.get("prefs") or {}
            if int(prefs.get("n_graded_positive", 0)) >= PREFS_MIN_POSITIVE:
                lines.append(
                    f"  learned taste: ~{float(prefs['mean_len']):.0f} "
                    f"words/sentence, emoji {float(prefs['emoji_rate']):.2f}, "
                    f"questions {float(prefs['question_rate']):.2f}, "
                    f"exclaims {float(prefs['exclaim_rate']):.2f}")
            else:
                lines.append(
                    f"  learned taste: not yet "
                    f"({int(prefs.get('n_graded_positive', 0))}/"
                    f"{PREFS_MIN_POSITIVE} positive grades)")
            if graded:
                top = sorted(graded, key=lambda e: -e["grade"])[:3]
                bot = sorted(graded, key=lambda e: e["grade"])[:3]
                lines.append("  top lines:")
                lines.extend(f"    + {e['line'][:80]}" for e in top)
                lines.append("  bottom lines:")
                lines.extend(f"    − {e['line'][:80]}" for e in bot)
            offs = d.get("trait_offsets") or {}
            moved = {k: v for k, v in offs.items() if v}
            if moved:
                lines.append("  persona offsets: " + ", ".join(
                    f"{k} {v:+.2f}" for k, v in sorted(moved.items())))
            return "\n".join(lines)

    def export_training_data(self, char_id: str) -> list[dict[str, Any]]:
        """Graded rows as (features, label) — the future batch-fit
        input. Only entries with a grade; ungraded rows are not
        training data, they're waiting data."""
        with self._lock:
            d = self.doc(char_id)
            rows = []
            for e in d.get("entries") or []:
                if e.get("grade") is None:
                    continue
                cands = e.get("candidates") or []
                chosen = cands[e.get("chosen", 0)] if cands else None
                if not chosen:
                    continue
                rows.append({"features": dict(chosen.get("features") or {}),
                             "label": float(e["grade"]),
                             "line": e.get("line", ""),
                             "entry_id": e.get("id", "")})
            return rows


# ── handle: the runtime object attached to a Character ────────────────
class LearnHandle:
    """Runtime binding between a Character and its learn doc.

    NOT serialized in Character.to_dict — the doc lives beside the
    character (LearnStore), and the handle is re-attached on load.
    ``adjust_persona`` is a runtime caller decision, deliberately not
    persisted: learning config is not character state.
    """

    def __init__(self, char: Any, store: LearnStore | None = None,
                 adjust_persona: bool = False) -> None:
        self.char = char
        self.store = store or LearnStore()
        self.adjust_persona = adjust_persona
        self._pending: list[str] = []

    @property
    def char_id(self) -> str:
        return str(getattr(self.char, "id", ""))

    def scorer_state(self) -> tuple[dict[str, float], dict[str, Any] | None,
                                    list[str]]:
        """(weights, prefs targets or None, recent lines) for the
        candidate scorer — the handle's view of learned state."""
        d = self.store.doc(self.char_id)
        return (d.get("weights") or dict(DEFAULT_WEIGHTS),
                _prefs_targets(d), self._recent_lines())

    def _recent_lines(self, n: int = 8) -> list[str]:
        d = self.store.doc(self.char_id)
        return [e.get("line", "") for e in d.get("entries", [])[-n:]
                if e.get("line")]

    def journal_line(self, line: str, *, context: str = "",
                     kind: str = "utterance",
                     candidates: list[str] | None = None,
                     chosen: int = 0) -> str:
        """Journal one spoken line. ``candidates`` (when given) are the
        scored candidate texts with the chosen index; otherwise the
        line is journaled alone with its own feature vector."""
        try:
            d = self.store.doc(self.char_id)
            weights = d.get("weights") or dict(DEFAULT_WEIGHTS)
            prefs = _prefs_targets(d)
            recent = self._recent_lines()
            if candidates:
                scored = score_candidates(
                    list(candidates), self.char, weights, recent, prefs)
                cand_dicts = [c.to_dict() for c in scored]
                chosen = max(0, min(int(chosen), len(cand_dicts) - 1))
            else:
                feats = extract_features(line, self.char, recent, prefs)
                cand_dicts = [{"text": line, "features": feats,
                              "score": sum(weights.get(k, DEFAULT_WEIGHTS[k])
                                           * feats[k] for k in FEATURES)}]
                chosen = 0
            entry = {
                "ts": time.time(), "kind": kind,
                "context": (context or "")[:200], "line": line,
                "candidates": cand_dicts, "chosen": chosen,
                "grade": None, "grades": [],
            }
            entry_id = self.store.journal(self.char_id, entry)
            self._pending.append(entry_id)
            if len(self._pending) > PENDING_CAP:
                self._pending = self._pending[-PENDING_CAP:]
            return entry_id
        except Exception as e:
            # Journaling is a side effect: record, never raise into speak().
            record_failure("learn.journal_line", e)
            return ""

    def grade(self, entry_id: str, value: float, source: str = "owner",
              note: str = "") -> dict[str, Any]:
        """Grade an entry (fail-fast on bad input — see LearnStore.grade).
        Persona nudges fire here, only when this handle opted in."""
        entry = self.store.grade(self.char_id, entry_id, value, source, note)
        if self.adjust_persona:
            self._nudge_persona(value)
        return entry

    def _nudge_persona(self, value: float) -> None:
        """Reinforce/weaken the character's STRONG traits (≥0.7) by one
        step, each trait leashed to ±TRAIT_LEASH from its seed snapshot.
        Seed is snapshotted on first nudge; explicit ocean is never
        re-derived (ocean stays authorial)."""
        try:
            d = self.store.doc(self.char_id)
            seed = d.get("seed_persona") or {}
            persona = getattr(self.char, "persona", None)
            if not isinstance(persona, dict):
                return
            if not seed:
                seed = dict(persona)
                d["seed_persona"] = seed
            offsets = d.setdefault("trait_offsets", {})
            delta = TRAIT_STEP if value > 0 else -TRAIT_STEP
            changed = False
            for trait, strength in list(persona.items()):
                if float(strength) < TRAIT_STRONG and trait not in offsets:
                    continue
                base = float(seed.get(trait, strength))
                new_off = _clamp(float(offsets.get(trait, 0.0)) + delta,
                                 -TRAIT_LEASH, TRAIT_LEASH)
                if new_off != float(offsets.get(trait, 0.0)):
                    offsets[trait] = new_off
                    persona[trait] = round(base + new_off, 4)
                    changed = True
            if changed:
                self.store.save(self.char_id)
        except Exception as e:
            record_failure("learn.nudge_persona", e)

    def begin_session(self) -> None:
        """Clear pending entry ids before a dialogue (stale ids from
        earlier dialogues must not inherit this session's grade)."""
        self._pending = []

    def drain_session_ids(self) -> list[str]:
        """Entry ids journaled since begin_session (or since attach),
        cleared on drain."""
        ids, self._pending = self._pending, []
        return ids

    def grade_session(self, kind: str) -> list[str]:
        """Grade every pending entry with a session-kind value.
        Returns the graded entry ids (empty when the kind is unknown
        or nothing is pending — never an error)."""
        value = SESSION_GRADES.get(kind)
        if value is None:
            return []
        out = []
        for eid in self.drain_session_ids():
            try:
                self.store.grade(self.char_id, eid, value, "session",
                                 note=f"session:{kind}")
                out.append(eid)
            except Exception as e:
                record_failure("learn.grade_session", e)
        return out

    def report(self) -> str:
        return self.store.report(self.char_id)

    def export_training_data(self) -> list[dict[str, Any]]:
        return self.store.export_training_data(self.char_id)


def attach(char: Any, data_dir: str | Path | None = None,
           adjust_persona: bool = False,
           store: LearnStore | None = None) -> LearnHandle:
    """Bind a learn handle to a character (stored as ``char._learn``,
    deliberately outside to_dict). Re-attaching replaces the handle."""
    handle = LearnHandle(char, store=store or
                         (LearnStore(data_dir) if data_dir is not None
                          else LearnStore()), adjust_persona=adjust_persona)
    try:
        char._learn = handle  # noqa: SLF001 - runtime binding, not state
    except Exception as e:
        record_failure("learn.attach", e)
    return handle


def detach(char: Any) -> None:
    """Remove the learn binding (the journal doc stays on disk)."""
    try:
        if hasattr(char, "_learn"):
            delattr(char, "_learn")
    except Exception as e:
        record_failure("learn.detach", e)


def get_handle(char: Any) -> LearnHandle | None:
    """The attached handle, if any."""
    h = getattr(char, "_learn", None)
    return h if isinstance(h, LearnHandle) else None
