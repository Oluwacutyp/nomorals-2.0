"""Video four (EDIT_GEN_UPGRADE 2.9).

* **Reframe / aspect** — :func:`reframe` maps any clip to 16:9, 9:16,
  1:1, 4:5, or 21:9 as an effect descriptor (``{"name": "reframe",
  "params": {...}}``) the engine's effect chain consumes; with subject
  tracking data (from :mod:`scene_intel.track`) the crop window
  follows the subject instead of sitting centered.
* **Match cuts / breath** — :func:`match_cut_score` + :func:`match_cuts`
  score cut pairs on colour/structure continuity (via
  :func:`videogen.consistency.boundary_metric`) and motion continuity;
  "breath" = the cut lands on a motion lull, not mid-gesture.
* **Auto B-roll from transcript** — :func:`auto_broll` queries the
  shot index (:class:`media.library.ClipIndex`) for each transcript
  topic gap and returns covering B-roll clips — no new search engine.
* **Pacing as a parameter** — :func:`tighten` / :func:`tighten_keeps`
  turn "tighten by a third" into measurable cut changes: fillers and
  pauses go first, then the longest pauses, until the target fraction
  is removed. Everything removed is recorded with its words in
  ``removed`` — recoverable from the transcript, never destroyed.
"""

from __future__ import annotations

import math
from typing import Any

from ...media_edit.videos import MediaEditError
from .timeline import Clip, Effect
from .failures import record_failure

__all__ = [
    "ASPECTS",
    "reframe",
    "reframe_dims",
    "match_cut_score",
    "match_cuts",
    "auto_broll",
    "tighten_keeps",
    "tighten",
]

ASPECTS: dict[str, tuple[int, int]] = {
    "16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1),
    "4:5": (4, 5), "5:4": (5, 4), "21:9": (21, 9),
    "4:3": (4, 3), "3:2": (3, 2),
}


def reframe_dims(aspect: str, *, height: int = 1080) -> tuple[int, int]:
    """Aspect → (width, height) at the given height (even numbers)."""
    key = str(aspect or "").strip().replace("x", ":")
    if key not in ASPECTS:
        raise MediaEditError(
            f"unknown aspect {aspect!r}; use: {sorted(ASPECTS)}")
    w_ratio, h_ratio = ASPECTS[key]
    w = int(round(height * w_ratio / h_ratio))
    w -= w % 2
    return w, height


def reframe(clip: Clip | dict[str, Any], aspect: str, *,
            subject_track: list[dict[str, Any]] | None = None,
            height: int = 1080) -> Clip:
    """Reframe a clip to ``aspect`` as an effect descriptor.

    ``subject_track``: optional ``[{"t": s, "bbox": (x1,y1,x2,y2)}]``
    (0..1 normalized, from :mod:`scene_intel.track`) — the crop window
    follows the subject centre instead of the frame centre. Without
    it the crop is centred (honest default, not fake tracking).
    """
    c = clip if isinstance(clip, Clip) else Clip.from_dict(dict(clip))
    w, h = reframe_dims(aspect, height=height)
    params: dict[str, Any] = {"aspect": aspect, "width": w, "height": h,
                              "subject_tracking": bool(subject_track)}
    if subject_track:
        # mean subject centre across the track — the crop anchor
        cxs = [(float(d["bbox"][0]) + float(d["bbox"][2])) / 2.0
               for d in subject_track if d.get("bbox")]
        cys = [(float(d["bbox"][1]) + float(d["bbox"][3])) / 2.0
               for d in subject_track if d.get("bbox")]
        if cxs:
            params["anchor_x"] = round(sum(cxs) / len(cxs), 3)
            params["anchor_y"] = round(sum(cys) / len(cys), 3)
    effects = tuple(c.effects) + (Effect(name="reframe", params=params),)
    return Clip(path=c.path, start=c.start, end=c.end, at=c.at,
                duration=c.duration, speed=c.speed, effects=effects,
                gain=c.gain)


def match_cut_score(shot_a: dict[str, Any], shot_b: dict[str, Any]) -> dict[str, Any]:
    """Score a cut pair: colour/structure continuity + motion breath.

    ``shot_a`` / ``shot_b`` carry optional ``"image"`` (PIL) for the
    boundary metric, plus ``"motion"`` (0..1) and ``"audio_energy"``
    (0..1) for the breath check. Without images the colour term is
    ``None`` (honest — not invented) and the score uses motion only.
    """
    motion_a = float(shot_a.get("motion", 0.5) or 0.5)
    motion_b = float(shot_b.get("motion", 0.5) or 0.5)
    # breath: both sides near a motion lull (low motion) = clean cut
    breath = 1.0 - min(1.0, (motion_a + motion_b) / 2.0)
    colour: float | None = None
    if shot_a.get("image") is not None and shot_b.get("image") is not None:
        from ..videogen.consistency import boundary_metric
        colour = float(boundary_metric(shot_a["image"],
                                       shot_b["image"])["score"])
        score = 0.6 * colour + 0.4 * breath
    else:
        score = breath
    return {"score": round(score, 3),
            "colour_continuity": (round(colour, 3) if colour is not None else None),
            "breath": round(breath, 3),
            "verdict": "match" if score >= 0.6 else "weak" if score >= 0.4 else "clash"}


def match_cuts(shots: list[dict[str, Any]], *,
               threshold: float = 0.6) -> list[dict[str, Any]]:
    """Score every consecutive cut pair; return the weak/clash ones."""
    out: list[dict[str, Any]] = []
    for i in range(len(shots) - 1):
        res = match_cut_score(shots[i], shots[i + 1])
        if res["score"] < threshold:
            out.append({"cut": i, "between": [shots[i].get("id", i),
                                              shots[i + 1].get("id", i + 1)],
                        **res})
    return out


def auto_broll(transcript: list[dict[str, Any]], index: Any, *,
               topics: list[str] | None = None,
               per_topic: int = 2) -> list[dict[str, Any]]:
    """Auto B-roll from the transcript via the indexed library.

    ``index`` is a :class:`media.library.ClipIndex` (or anything with
    a ``query(text)`` returning shot dicts). ``topics`` defaults to the
    transcript's own distinctive words (longest words, deduped). Each
    topic gets up to ``per_topic`` covering shots.
    """
    if topics is None:
        seen: list[str] = []
        for w in transcript:
            for token in str(w.get("text", "")).split():
                clean = token.strip(".,!?\"'").lower()
                if len(clean) >= 5 and clean not in seen:
                    seen.append(clean)
        topics = seen[:8]
    picks: list[dict[str, Any]] = []
    used: set[str] = set()
    for topic in topics:
        try:
            hits = index.query(topic) if hasattr(index, "query") else []
        except Exception as e:  # index miss is not fatal
            record_failure("video_four.auto_broll:index_query", e)
            hits = []
        for hit in (hits or [])[:per_topic]:
            key = str(hit.get("clip_id", "")) + ":" + str(hit.get("start", ""))
            if key in used:
                continue
            used.add(key)
            picks.append({"topic": topic, **dict(hit)})
    return picks


def tighten_keeps(keeps: list[tuple[float, float]],
                  words: list[dict[str, Any]], *,
                  fraction: float = 1.0 / 3.0,
                  filler_words: tuple[str, ...] = ("um", "uh", "like",
                                                   "you", "know")) -> dict[str, Any]:
    """Pacing as a parameter: remove ``fraction`` of keep-duration.

    Removal order (measurable, recoverable): filler-word ranges first,
    then the longest word gaps (pauses), then — only if still short —
    the longest keeps trimmed from their tail. Every removed range is
    returned with its words, so the cut is fully recoverable.

    Returns ``{"keeps", "removed", "duration_before", "duration_after",
    "removed_fraction", "cut_count"}``.
    """
    if not 0.0 < fraction < 1.0:
        raise MediaEditError("tighten fraction must be in (0, 1)")
    keeps = sorted((float(s), float(e)) for s, e in keeps)
    before = sum(e - s for s, e in keeps)
    target_remove = before * fraction
    removed: list[dict[str, Any]] = []

    def _words_in(s: float, e: float) -> list[dict[str, Any]]:
        return [w for w in words
                if float(w.get("start", 0)) >= s
                and float(w.get("end", 0)) <= e]

    def _cut(s: float, e: float, reason: str) -> None:
        nonlocal keeps
        if e - s <= 1e-6:
            return
        out: list[tuple[float, float]] = []
        for ks, ke in keeps:
            if e <= ks or s >= ke:
                out.append((ks, ke))
                continue
            if ks < s:
                out.append((ks, s))
            if ke > e:
                out.append((e, ke))
        keeps = out
        removed.append({"start": s, "end": e, "reason": reason,
                        "words": _words_in(s, e)})

    # 1. filler-word ranges
    fillers = {w.strip().lower().rstrip(",.") for w in filler_words}
    for w in words:
        if str(w.get("text", "")).strip().lower().rstrip(",.") in fillers:
            _cut(float(w["start"]), float(w["end"]), "filler")
    removed_so_far = sum(r["end"] - r["start"] for r in removed)
    # 2. longest pauses (gaps between consecutive words inside keeps)
    if removed_so_far < target_remove:
        gaps: list[tuple[float, float, float]] = []
        ws = sorted(words, key=lambda w: float(w.get("start", 0)))
        for a, b in zip(ws, ws[1:]):
            gap_s, gap_e = float(a.get("end", 0)), float(b.get("start", 0))
            if gap_e - gap_s > 0.4:
                gaps.append((gap_e - gap_s, gap_s, gap_e))
        for _dur, gs, ge in sorted(gaps, reverse=True):
            if removed_so_far >= target_remove:
                break
            # only the part inside current keeps, capped to what we need
            need = target_remove - removed_so_far
            take = min(ge - gs, need)
            _cut(gs, gs + take, "pause")
            removed_so_far = sum(r["end"] - r["start"] for r in removed)
    # 3. tail-trim longest keeps (last resort)
    while removed_so_far < target_remove - 1e-6 and keeps:
        longest = max(keeps, key=lambda k: k[1] - k[0])
        need = target_remove - removed_so_far
        take = min((longest[1] - longest[0]) * 0.5, need)
        if take <= 1e-3:
            break
        _cut(longest[1] - take, longest[1], "tail_trim")
        removed_so_far = sum(r["end"] - r["start"] for r in removed)
    after = sum(e - s for s, e in keeps)
    return {"keeps": keeps, "removed": removed,
            "duration_before": round(before, 3),
            "duration_after": round(after, 3),
            "removed_fraction": round((before - after) / before, 3)
            if before else 0.0,
            "cut_count": len(removed)}


def tighten(document: Any, *, fraction: float = 1.0 / 3.0) -> Any:
    """Tighten a :class:`Document` (or EditSpec) by ``fraction``.

    Rebuilds the video track from the tightened keeps and appends the
    removed ranges to ``document.removed`` (recoverable). Returns the
    document (sealed).
    """
    from .document import Document
    from .transcript import keeps_for
    if isinstance(document, Document):
        words = document.transcript
        dur = document.duration
        if not words or dur <= 0:
            raise MediaEditError("tighten needs a transcript + durations")
        base_keeps = keeps_for(words, duration=dur, ops=[])
        res = tighten_keeps(base_keeps, words, fraction=fraction)
        video = document.track("video")
        if video is None or not video.clips:
            raise MediaEditError("tighten needs a video track")
        src = video.clips[0].path
        video.clips = [Clip(path=src, start=s, end=e)
                       for s, e in res["keeps"]]
        document.removed.extend(res["removed"])
        return document.seal()
    raise MediaEditError("tighten takes a Document")
