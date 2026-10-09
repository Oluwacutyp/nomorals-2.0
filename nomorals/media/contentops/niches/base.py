"""Niche plugin architecture for Devon's short-form content automation.

A niche is a repeatable content format (motivation, horror stories, …)
that drives Devon's REAL systems:

  script   → Brain.complete(script_prompt(topic))          (nomorals/llm/brain.py)
  visuals  → Studio().generate(imggen_prompt, save_to=…)  (nomorals/media/imggen/studio.py)
  voice    → VoiceCatalogue.speak_as(text, voice_name, out_path=…)
             (nomorals/voice/catalogue.py)
  captions → burn_captions(video, words, style="hormozi")
             (nomorals/media_edit/captions.py)

The plugin never renders video itself — it produces the *plan* (a
retention-engineered script prompt and a scene-by-scene visual plan)
that the contentops pipeline feeds into those systems. New niches are
added by dropping a module in this package (or anywhere importable) that
subclasses :class:`NichePlugin` and calls ``register()`` — core code is
never touched.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence


# ── plan primitives ──────────────────────────────────────────────────────

@dataclass
class VoiceSpec:
    """Which Devon voice a niche speaks with, and how it performs.

    Catalogue voices are user-created, so ``preferred_voice`` may not
    exist on a given machine. :meth:`resolve` picks the first name that
    actually exists in the catalogue, falling back to ``fallback_voice``
    (the engine's ``default`` profile), and finally to ``""`` which lets
    the TTS engine pick its own default.
    """

    preferred_voice: str = ""
    fallback_voice: str = "default"
    mood: str = "neutral"
    intensity: int = 3
    pace_wpm: int = 150
    # director knobs passed through to the TTS performance layer
    effect: str = ""
    pitch_shift: float = 0.0
    pause_style: str = "natural"

    def candidates(self) -> list[str]:
        out = [c for c in (self.preferred_voice, self.fallback_voice) if c]
        return out

    def __str__(self) -> str:
        """Human/compat label — the preferred voice, else 'narrator'."""
        return self.preferred_voice or "narrator"

    def resolve(self, catalogue: Any) -> str:
        """Return the voice name to actually use. Never raises."""
        try:
            names = set(getattr(catalogue, "voices", {}) or {})
        except Exception:
            names = set()
        for name in self.candidates():
            if name in names:
                return name
        return ""


@dataclass
class SceneVisual:
    """One scene of the visual plan — one image + one motion directive."""

    index: int
    imggen_prompt: str
    motion: str = "zoom-in-slow"   # ken-burns directive
    duration_s: float = 4.0
    transition: str = "cut"
    caption_hook: str = ""         # word/phrase to emphasise in captions


@dataclass
class VisualPlan:
    """Scene-by-scene visual plan for a script."""

    style_lock: str                        # immutable style anchor
    scenes: list[SceneVisual] = field(default_factory=list)
    aspect: str = "9:16"
    notes: str = ""

    def render_prompts(self) -> list[str]:
        """Full imggen prompts for :meth:`Studio.generate`, style-locked."""
        return [s.imggen_prompt for s in self.scenes]

    def total_seconds(self) -> float:
        return sum(s.duration_s for s in self.scenes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "style_lock": self.style_lock,
            "aspect": self.aspect,
            "notes": self.notes,
            "scenes": [
                {
                    "index": s.index,
                    "imggen_prompt": s.imggen_prompt,
                    "motion": s.motion,
                    "duration_s": s.duration_s,
                    "transition": s.transition,
                    "caption_hook": s.caption_hook,
                }
                for s in self.scenes
            ],
        }


@dataclass
class ScriptResult:
    """The output of :meth:`NichePlugin.generate_script`."""

    text: str
    word_count: int
    est_seconds: float
    topic: str = ""
    niche: str = ""


# ── the plugin contract ──────────────────────────────────────────────────

_REQUIRED_ATTRS = (
    "name", "thesis", "cadence", "title_template",
    "description_template", "hashtags", "voice_spec", "ypp_rationale",
)


class NichePlugin(ABC):
    """Base class for every content niche.

    Required class attributes (validated on :func:`register`):

    * ``name``            — slug, e.g. ``"motivation"``
    * ``thesis``          — one-line content thesis
    * ``cadence``         — recommended posts per day (float > 0)
    * ``title_template``  — str with a ``{topic}`` placeholder
    * ``description_template`` — str with ``{topic}`` and optional ``{script}``
    * ``hashtags``        — dict mapping platform → list of tags, or a plain
      list used for every platform
    * ``voice_spec``      — :class:`VoiceSpec`
    * ``ypp_rationale``   — why this niche is monetization-safe
    """

    name: str = ""
    thesis: str = ""
    cadence: float = 1.0
    target_seconds: tuple[int, int] = (25, 55)
    title_template: str = ""
    description_template: str = ""
    hashtags: dict[str, list[str]] | list[str] = {}
    voice_spec: VoiceSpec = VoiceSpec()
    ypp_rationale: str = ""

    # ── the two core methods ──

    @abstractmethod
    def script_prompt(self, topic: str) -> str:
        """Return the full LLM prompt that produces a spoken script for
        ``topic``. The prompt must enforce: 25–55 s spoken length, a hook
        in the first 2 seconds, open loops, and a payoff — written for
        voice, not for reading."""

    @abstractmethod
    def visual_strategy(self, script: str) -> VisualPlan:
        """Return a scene-by-scene :class:`VisualPlan` for ``script`` —
        original/AI visuals first-class, style-locked per niche."""

    # ── shared helpers (not overridden by niches) ──

    def word_budget(self) -> tuple[int, int]:
        """(min, max) spoken words for the target duration at the niche's pace."""
        lo, hi = self.target_seconds
        wpm = max(60, self.voice_spec.pace_wpm)
        return (int(lo * wpm / 60), int(hi * wpm / 60))

    def generate_script(self, brain: Any, topic: str) -> ScriptResult:
        """Run the niche's script prompt through a brain.

        ``brain`` is anything with ``.complete(prompt) -> LLMResponse``
        (real :class:`Brain`, or a mock in tests). Never raises — a brain
        failure returns an empty ScriptResult with the reason attached.
        """
        prompt = self.script_prompt(topic)
        try:
            resp = brain.complete(prompt)
        except Exception as exc:  # noqa: BLE001 — honest failure, not a raise
            return ScriptResult(text="", word_count=0, est_seconds=0.0,
                                topic=topic, niche=self.name)
        text = (getattr(resp, "text", "") or "").strip()
        text = _strip_fence(text)
        words = len(text.split())
        est = words / max(60, self.voice_spec.pace_wpm) * 60.0
        return ScriptResult(text=text, word_count=words, est_seconds=est,
                            topic=topic, niche=self.name)

    def title_for(self, topic: str) -> str:
        return self.title_template.format(topic=topic).strip()

    def description_for(self, topic: str, script: str = "") -> str:
        desc = self.description_template.format(topic=topic, script=script)
        return desc.strip()

    def tags_for(self, platform: str = "tiktok") -> list[str]:
        """Platform-adaptable hashtags. Falls back to a generic set."""
        tags = self.hashtags
        if isinstance(tags, dict):
            return list(tags.get(platform, tags.get("default", [])))
        return list(tags)

    def imggen_prompts(self, script: str) -> list[str]:
        """Style-locked imggen prompt strings — one per scene.

        Compatibility surface: feeds straight into
        ``Studio().generate(prompt, save_to=...)``.
        """
        return self.visual_strategy(script).render_prompts()

    def cadence_spec(self) -> dict[str, float]:
        """Pipeline-friendly cadence: ``{"posts_per_day": …}``."""
        return {"posts_per_day": float(self.cadence)}

    # ── script → scenes helper ──

    def _beats_to_scenes(
        self,
        script: str,
        scene_briefs: Sequence[str],
        *,
        style_lock: str,
        motions: Sequence[str] | None = None,
        transitions: Sequence[str] | None = None,
        scene_seconds: float = 4.0,
        notes: str = "",
    ) -> VisualPlan:
        """Turn a script into scenes: one brief per script beat, zipped.

        ``scene_briefs`` are per-niche visual descriptions for each beat of
        the format (hook → escalation → payoff). If the script has more
        beats than briefs, briefs cycle; if fewer, extra briefs are dropped.
        Every prompt gets the niche's ``style_lock`` appended so the whole
        short looks like one piece.
        """
        beats = [b.strip() for b in re.split(r"\n\s*\n", script.strip()) if b.strip()]
        if not beats:
            beats = [script.strip() or "atmospheric establishing shot"]
        scenes: list[SceneVisual] = []
        motions = list(motions) if motions else ["zoom-in-slow"]
        transitions = list(transitions) if transitions else ["cut"]
        n = len(beats)
        per = max(1.0, (self.target_seconds[1] / max(1, n))) if n else scene_seconds
        for i, beat in enumerate(beats):
            brief = scene_briefs[i % len(scene_briefs)]
            prompt = f"{brief}. {style_lock}"
            scenes.append(SceneVisual(
                index=i,
                imggen_prompt=prompt,
                motion=motions[i % len(motions)],
                duration_s=round(scene_seconds if scene_seconds != 4.0 else per, 1),
                transition=transitions[i % len(transitions)],
                caption_hook=_first_stressed_words(beat),
            ))
        return VisualPlan(style_lock=style_lock, scenes=scenes, notes=notes)


def _strip_fence(text: str) -> str:
    m = re.match(r"^```(?:\w+)?\s*\n(.*?)```\s*$", text, re.S)
    return m.group(1).strip() if m else text


def _first_stressed_words(beat: str, limit: int = 3) -> str:
    words = [w.strip(".,!?;:\"'()").upper() for w in beat.split()]
    strong = [w for w in words if len(w) > 4][:limit]
    return " ".join(strong)
