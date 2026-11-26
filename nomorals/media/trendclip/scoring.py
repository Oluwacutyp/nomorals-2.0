"""The deterministic scoring rung — ported from the source's
``trendclip/highlight_strategies.py``, ``trendclip/media_pipeline.py``
(``score_moments``) and ``trendclip/drafts.py``.

Pure standard library, no numpy: the math is z-scores, weighted sums and
greedy window selection, and it stays that way so the code rung runs
everywhere the profile runtime can run Python, with no model and no heavy
dependency. Every number it returns is explainable: the receipt carries
the per-second signals, the strategy weights, the quality components and
the warnings, so a pick can be audited after the fact.

The shape of a moment::

    {
      "n": 0,                       # candidate number, 1-based in display
      "start": 12.4, "end": 72.4,   # seconds in the source
      "duration": 60.0,
      "score": 0.83,                # weighted z-score, -1..1ish
      "quality_score": 8.2,         # 0..10 editorial quality
      "components": {"clean_start": 1.0, "clean_end": 1.0, "hook": 0.9,
                     "payoff": 0.8, "gaming_event": 0.0},
      "why": ["strong speech density", "loudness peak at 41s"],
      "warnings": ["starts mid-sentence"],
      "strategy": "conversation",
      "words": [...],               # the transcript words inside the window
    }
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "STRATEGY_WEIGHTS",
    "assemble_drafts",
    "editorial_issues",
    "editorial_summary",
    "score_feed_item",
    "select_moments",
    "zscore",
]

#: Per-signal weights per strategy, ported from the source. The four
#: signals are loudness (dB RMS), hype (loudness variation), heat (visual
#: motion) and speech (word density). Weights sum to 1.0 per strategy.
STRATEGY_WEIGHTS: dict[str, dict[str, float]] = {
    "gaming": {"loudness": 0.30, "hype": 0.35, "heat": 0.20,
               "speech": 0.15},
    "conversation": {"loudness": 0.15, "hype": 0.15, "heat": 0.10,
                     "speech": 0.60},
    "educational": {"loudness": 0.10, "hype": 0.10, "heat": 0.10,
                    "speech": 0.70},
    "comedy": {"loudness": 0.25, "hype": 0.40, "heat": 0.15,
               "speech": 0.20},
    "craft": {"loudness": 0.20, "hype": 0.10, "heat": 0.30,
              "speech": 0.40},
}

#: Draft tiers by editorial quality, from the source's ``drafts.py``.
TOP_PICK_MIN = 7.5
HIGH_ENERGY_MIN = 6.5
QUICK_HIT_MIN = 5.5


def zscore(series: list[float]) -> list[float]:
    """Z-normalize a per-second signal series (mean 0, std 1)."""
    if not series:
        return []
    mean = sum(series) / len(series)
    var = sum((v - mean) ** 2 for v in series) / len(series)
    std = math.sqrt(var)
    if std < 1e-9:
        return [0.0] * len(series)
    return [(v - mean) / std for v in series]


def _speech_series(words: list[dict[str, Any]], seconds: int) -> list[float]:
    """Words per second, from word timestamps."""
    series = [0.0] * max(seconds, 0)
    for word in words:
        try:
            start = float(word.get("s", word.get("start", 0.0)))
        except (TypeError, ValueError):
            continue
        idx = int(start)
        if 0 <= idx < len(series):
            series[idx] += 1.0
    return series


def _hype_series(loudness: list[float]) -> list[float]:
    """Loudness variation: |dB(t) - dB(t-1)|, the source's hype signal."""
    out = [0.0] * len(loudness)
    for i in range(1, len(loudness)):
        out[i] = abs(loudness[i] - loudness[i - 1])
    return out


def _snap_to_speech(start: float, end: float,
                    words: list[dict[str, Any]]) -> tuple[float, float]:
    """Snap a window's edges to the nearest word boundary so a clip does
    not start or end mid-word. Falls back to the raw edges when the window
    holds no words."""
    if not words:
        return start, end
    starts = sorted(float(w.get("s", w.get("start", 0.0))) for w in words)
    ends = sorted(float(w.get("e", w.get("end", 0.0))) for w in words)
    new_start = start
    for s in starts:
        if s >= start - 2.0:
            new_start = max(start, min(s, start + 2.0))
            break
    new_end = end
    for e in reversed(ends):
        if e <= end + 2.0:
            new_end = min(end, max(e, end - 2.0))
            break
    if new_end - new_start < 5.0:
        return start, end
    return new_start, new_end


def _quality_components(start: float, end: float,
                        words: list[dict[str, Any]],
                        loudness: list[float],
                        strategy: str) -> dict[str, float]:
    """The editorial quality components, each 0..1.

    clean_start / clean_end: the window does not cut a word at its edge.
    hook: the first 3 seconds carry speech or energy. payoff: the loudest
    second sits inside the window, not at its edge. gaming_event: for the
    gaming strategy, a loudness spike sits inside the window.
    """
    window_words = [w for w in words
                    if float(w.get("s", w.get("start", 0.0))) >= start
                    and float(w.get("s", w.get("start", 0.0))) < end]
    if window_words:
        first = float(window_words[0].get("s",
                                          window_words[0].get("start", 0.0)))
        last = float(window_words[-1].get("e",
                                          window_words[-1].get("end", 0.0)))
        clean_start = 1.0 if first - start >= 0.15 else 0.5
        clean_end = 1.0 if end - last >= 0.15 else 0.5
        first3 = [w for w in window_words
                  if float(w.get("s", w.get("start", 0.0))) < start + 3.0]
        hook = min(1.0, len(first3) / 3.0)
    else:
        clean_start = 0.5
        clean_end = 0.5
        hook = 0.0
    span = loudness[max(0, int(start)):max(0, int(end))] or [0.0]
    peak = max(span)
    peak_idx = span.index(peak) if span else 0
    payoff = 1.0 if 2 <= peak_idx <= len(span) - 3 else 0.5
    gaming_event = 0.0
    if strategy == "gaming" and loudness:
        mean = sum(loudness) / len(loudness)
        gaming_event = 1.0 if peak > mean + 6.0 else 0.0
    return {
        "clean_start": clean_start,
        "clean_end": clean_end,
        "hook": hook,
        "payoff": payoff,
        "gaming_event": gaming_event,
    }


def _quality_score(components: dict[str, float],
                   strategy: str) -> float:
    """Editorial quality 0..10 from the components."""
    base = (components["clean_start"] * 0.2
            + components["clean_end"] * 0.2
            + components["hook"] * 0.3
            + components["payoff"] * 0.3)
    if strategy == "gaming":
        base = base * 0.85 + components["gaming_event"] * 0.15
    return round(base * 10.0, 1)


def select_moments(*, loudness: list[float],
                   heat: list[float] | None = None,
                   words: list[dict[str, Any]] | None = None,
                   strategy: str = "conversation",
                   target_seconds: float = 60.0,
                   min_seconds: float = 15.0,
                   max_seconds: float = 120.0,
                   max_moments: int = 5) -> list[dict[str, Any]]:
    """Score and select the best moments in a source.

    ``loudness`` is dB RMS per second, ``heat`` is visual motion per
    second (zeros when OpenCV is unavailable — the rung still runs on
    audio + speech), ``words`` are the transcript words with ``s``/``e``
    timestamps. Returns up to ``max_moments`` non-overlapping candidates,
    best first, each with its score, quality, components, why and warnings.
    """
    words = list(words or [])
    seconds = len(loudness)
    if seconds == 0:
        return []
    heat = list(heat or [0.0] * seconds)
    if len(heat) < seconds:
        heat = heat + [0.0] * (seconds - len(heat))
    weights = STRATEGY_WEIGHTS.get(strategy,
                                   STRATEGY_WEIGHTS["conversation"])
    z_loud = zscore(loudness)
    z_hype = zscore(_hype_series(loudness))
    z_heat = zscore(heat[:seconds])
    z_speech = zscore(_speech_series(words, seconds))
    per_second = [
        (weights["loudness"] * z_loud[i]
         + weights["hype"] * z_hype[i]
         + weights["heat"] * z_heat[i]
         + weights["speech"] * z_speech[i])
        for i in range(seconds)
    ]
    window = int(max(min_seconds, min(target_seconds, max_seconds,
                                      seconds)))
    if window < 5:
        window = min(seconds, 5)
    step = max(1, window // 4)
    candidates: list[dict[str, Any]] = []
    for start in range(0, max(1, seconds - window + 1), step):
        end = start + window
        score = sum(per_second[start:end]) / max(1, window)
        candidates.append({"start": float(start), "end": float(end),
                           "score": score})
    candidates.sort(key=lambda c: c["score"], reverse=True)
    picked: list[dict[str, Any]] = []
    for cand in candidates:
        if len(picked) >= max_moments:
            break
        if any(not (cand["end"] <= p["start"] or cand["start"] >= p["end"])
               for p in picked):
            continue
        start, end = _snap_to_speech(cand["start"], cand["end"], words)
        components = _quality_components(start, end, words, loudness,
                                         strategy)
        quality = _quality_score(components, strategy)
        why: list[str] = []
        warnings: list[str] = []
        span_words = [w for w in words
                      if start <= float(w.get("s", w.get("start", 0.0))) < end]
        if span_words:
            why.append(f"{len(span_words)} spoken words in the window")
        peak = max(loudness[int(start):int(end)] or [0.0])
        mean_loud = sum(loudness) / len(loudness)
        if peak > mean_loud + 3.0:
            why.append(f"loudness peak {peak:.1f} dB at "
                       f"{loudness.index(peak)}s")
        if components["clean_start"] < 1.0:
            warnings.append("starts mid-sentence")
        if components["clean_end"] < 1.0:
            warnings.append("ends mid-sentence")
        if components["hook"] < 0.5:
            warnings.append("no speech in the first 3 seconds")
        if quality < QUICK_HIT_MIN:
            warnings.append("below the quick-hit quality floor")
        picked.append({
            "n": len(picked) + 1,
            "start": start,
            "end": end,
            "duration": round(end - start, 1),
            "score": round(cand["score"], 3),
            "quality_score": quality,
            "components": components,
            "why": why,
            "warnings": warnings,
            "strategy": strategy,
            "words": span_words,
        })
    return picked


def assemble_drafts(moments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The editorial gate, from the source's ``drafts.py``: quality >= 7.5
    is a Top pick, >= 6.5 High energy, >= 5.5 a Quick hit, below that the
    moment is rejected with its reasons named."""
    drafts = []
    for moment in moments:
        quality = float(moment.get("quality_score", 0.0))
        if quality >= TOP_PICK_MIN:
            tier, verdict = "top_pick", "Top pick"
        elif quality >= HIGH_ENERGY_MIN:
            tier, verdict = "high_energy", "High energy"
        elif quality >= QUICK_HIT_MIN:
            tier, verdict = "quick_hit", "Quick hit"
        else:
            tier, verdict = "rejected", "Rejected"
        drafts.append({**moment, "tier": tier, "verdict": verdict})
    return drafts


def editorial_summary(moments: list[dict[str, Any]]) -> str:
    """One line per moment: its verdict, quality and why."""
    if not moments:
        return "No moments scored above the floor."
    lines = []
    for moment in moments:
        lines.append(
            f"#{moment.get('n', '?')} {moment.get('verdict', '?')} — "
            f"quality {moment.get('quality_score', '?')}/10, "
            f"{moment.get('start', 0):.0f}s–{moment.get('end', 0):.0f}s: "
            + "; ".join(moment.get("why") or ["no signal"]))
    return "\n".join(lines)


def editorial_issues(moments: list[dict[str, Any]]) -> list[str]:
    """The warnings across all moments, deduplicated, in first-seen order."""
    seen: list[str] = []
    for moment in moments:
        for warning in moment.get("warnings") or []:
            if warning not in seen:
                seen.append(warning)
    return seen


def score_feed_item(*, relevance: float, freshness: float,
                    reliability: float) -> float:
    """Rank a discovered video: relevance (keyword match), freshness
    (recency) and reliability (channel trust), each 0..1. Ported from the
    source's ``discovery.score_item``."""
    relevance = max(0.0, min(1.0, relevance))
    freshness = max(0.0, min(1.0, freshness))
    reliability = max(0.0, min(1.0, reliability))
    return round(0.5 * relevance + 0.3 * freshness + 0.2 * reliability, 3)
