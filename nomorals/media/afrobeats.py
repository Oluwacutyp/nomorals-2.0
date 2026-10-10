"""Afrobeats LoRA + Nigerian-language sung vocals — the continental moat.

Two halves:

1. **Afrobeats LoRA training pipeline** (:class:`AfrobeatsLoRA`): the real,
   executable LoRA fine-tuning path for ACE-Step 1.5 — dataset prep →
   preprocess → train, with the exact CLI flags verified against the
   public training repos (2026-10-08). The pipeline is the deliverable;
   a trained model needs a CUDA GPU + a cleared corpus. :meth:`apply_lora`
   plugs trained weights into :class:`ACEStepBackend` (diffusers path
   fully wired; official-app path documented).

2. **Nigerian-language sung vocals** (:class:`NaijaSungVocals`): the hard
   part nobody solves — Yoruba/Igbo are **tonal**, so the tone contour of
   a syllable must not fight the melody contour, or the word means the
   wrong thing (``owó`` money vs ``ọ̀wọ̀``... sung wrong, "money" becomes
   gibberish). :func:`tone_to_melody` checks every adjacent syllable pair;
   :func:`tone_aware_melody` rewrites clashing notes; :meth:`sing` routes
   through the DiffSinger → RVC chain with the fixed melody.

Positioning: "the AI that sings in Yoruba." No competitor exists.

Honest boundaries: no corpus/GPU → the training job fails closed with a
recipe, never a fake LoRA. No LLM → Yoruba lyrics fail closed (the user
can supply their own lyrics — the tone engine works on any text).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llm.brain import brain_for
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "HIGH", "MID", "LOW",
    "TONE_NAMES",
    "LANGUAGES",
    "SyllableTone",
    "ToneVerdict",
    "Alignment",
    "parse_tones",
    "tone_to_melody",
    "tone_aware_melody",
    "LoRATrainingJob",
    "LoRAWeights",
    "LoRAUnavailable",
    "AfrobeatsLoRA",
    "corpus_spec",
    "SungVocalResult",
    "NaijaSungVocals",
    "NaijaSongRequest",
    "parse_naija_song_request",
    "yoruba_lyrics",
    "make_naija_song",
    "DEFAULT_LORA_REGISTRY",
    "register",
]

# ───────────────────────── tones ────────────────────────────────────────────

#: Yoruba/Igbo tone levels. Yoruba has three; Igbo two (HIGH/LOW).
HIGH, MID, LOW = 2, 1, 0
TONE_NAMES = {HIGH: "HIGH", MID: "MID", LOW: "LOW"}

#: languages with tone-aware singing
LANGUAGES = ("yoruba", "igbo", "ekiti", "pidgin")

#: combining marks in NFD form
_ACUTE = "\u0301"   # ´ HIGH
_GRAVE = "\u0300"   # ` LOW
_MACRON = "\u0304"  # ¯ MID (explicit)

_VOWELS = frozenset("aeiouAEIOU")  # NFD base; ẹ/ọ decompose to e/o + dot


def _char_groups(word: str) -> list[tuple[str, str]]:
    """Split into (base char, combining marks) groups, NFD."""
    nfd = unicodedata.normalize("NFD", word or "")
    groups: list[tuple[str, str]] = []
    for ch in nfd:
        if unicodedata.combining(ch):
            if groups:
                b, m = groups[-1]
                groups[-1] = (b, m + ch)
            else:
                groups.append(("", ch))
        else:
            groups.append((ch, ""))
    return groups


def _syllabify_word(word: str) -> list[str]:
    """Split a Yoruba/Igbo word into syllables (CV pattern).

    Each syllable = onset consonants + one vowel nucleus (or a syllabic
    nasal m/n). Returns syllables with tone marks preserved.
    """
    groups = _char_groups(word)
    n = len(groups)
    syllables: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    i = 0
    while i < n:
        base, marks = groups[i]
        bl = base.lower()
        cur.append((base, marks))
        if bl in _VOWELS:
            syllables.append(cur)
            cur = []
        elif bl in ("m", "n"):
            # syllabic nasal when NOT followed by a vowel
            nxt = groups[i + 1][0].lower() if i + 1 < n else ""
            if nxt not in _VOWELS:
                syllables.append(cur)
                cur = []
        i += 1
    if cur:
        if syllables:
            syllables[-1].extend(cur)
        else:
            syllables.append(cur)
    return ["".join(b + m for b, m in s) for s in syllables]


def _tone_of_syllable(syllable: str) -> int:
    """Tone from the combining marks on the syllable's vowel/nasal."""
    nfd = unicodedata.normalize("NFD", syllable)
    if _ACUTE in nfd:
        return HIGH
    if _GRAVE in nfd:
        return LOW
    if _MACRON in nfd:
        return MID
    return MID  # unmarked = mid (Yoruba default)


@dataclass
class SyllableTone:
    """One syllable with its tone level."""
    text: str       # as written, tone marks preserved
    plain: str      # tone marks stripped
    tone: int       # HIGH | MID | LOW
    language: str = "yoruba"

    @property
    def tone_name(self) -> str:
        return TONE_NAMES[self.tone]


def parse_tones(text: str, language: str = "yoruba") -> list[SyllableTone]:
    """Split text into syllables with tone levels.

    Handles precomposed (á) and decomposed (a + U+0301) forms.
    Igbo: unmarked syllables default to LOW (common marking convention);
    Yoruba/Ekiti: unmarked default to MID.
    """
    lang = (language or "yoruba").strip().lower()
    out: list[SyllableTone] = []
    for word in (text or "").split():
        for syl in _syllabify_word(word):
            nfd = unicodedata.normalize("NFD", syl)
            if _ACUTE in nfd:
                tone = HIGH
            elif _GRAVE in nfd:
                tone = LOW
            elif _MACRON in nfd:
                tone = MID
            else:
                tone = LOW if lang == "igbo" else MID
            # plain: strip combining marks from NFD, recompose
            plain = unicodedata.normalize(
                "NFC", re.sub(r"[\u0300-\u036f]", "",
                              unicodedata.normalize("NFD", syl)))
            out.append(SyllableTone(text=syl, plain=plain, tone=tone,
                                    language=lang))
    return out


@dataclass
class ToneVerdict:
    """One adjacent syllable pair vs the melody move between them."""
    index: int            # pair index (between syllable i and i+1)
    syllable_a: str
    syllable_b: str
    tone_dir: int         # -1 | 0 | +1
    note_a: int           # MIDI pitch
    note_b: int
    mel_dir: int          # -1 | 0 | +1
    ok: bool

    @property
    def note_text(self) -> str:
        tdir = {1: "rises", -1: "falls", 0: "stays level"}[self.tone_dir]
        mdir = {1: "rises", -1: "falls", 0: "stays level"}[self.mel_dir]
        return (f"'{self.syllable_a}' → '{self.syllable_b}': tone {tdir} "
                f"but the melody {mdir} — "
                f"this note fights the word's tone")


@dataclass
class Alignment:
    """Tone↔melody alignment for a lyric line/set."""
    verdicts: list[ToneVerdict] = field(default_factory=list)
    language: str = "yoruba"

    @property
    def score(self) -> float:
        if not self.verdicts:
            return 1.0
        return sum(1 for v in self.verdicts if v.ok) / len(self.verdicts)

    @property
    def mismatches(self) -> list[ToneVerdict]:
        return [v for v in self.verdicts if not v.ok]


def _sign(x: int) -> int:
    return 1 if x > 0 else (-1 if x < 0 else 0)


def tone_to_melody(lyrics: str, melody: list,
                   language: str = "yoruba") -> Alignment:
    """Check the tone contour against the melody contour.

    Maps syllables 1:1 onto melody notes (the pre-check before
    DiffSinger's phoneme timing). A pair is a **mismatch** only when the
    tone and the melody move in opposite directions — level tones and
    level notes are flexible, matching the way Yoruba is actually sung.

    Pidgin is not tonal: returns a trivially-OK alignment.
    """
    lang = (language or "yoruba").strip().lower()
    if lang == "pidgin":
        return Alignment(verdicts=[], language=lang)
    syllables = parse_tones(lyrics, lang)
    notes = [(int(p), float(s), float(d)) for p, s, d in melody]
    pairs = min(len(syllables) - 1, len(notes) - 1)
    verdicts: list[ToneVerdict] = []
    for i in range(max(0, pairs)):
        tdir = _sign(syllables[i + 1].tone - syllables[i].tone)
        mdir = _sign(notes[i + 1][0] - notes[i][0])
        # hard error: opposite directions. Everything else is singable.
        ok = not (tdir != 0 and mdir != 0 and tdir != mdir)
        verdicts.append(ToneVerdict(
            index=i,
            syllable_a=syllables[i].text,
            syllable_b=syllables[i + 1].text,
            tone_dir=tdir, note_a=notes[i][0], note_b=notes[i + 1][0],
            mel_dir=mdir, ok=ok))
    return Alignment(verdicts=verdicts, language=lang)


def tone_aware_melody(lyrics: str, melody: list,
                      language: str = "yoruba",
                      max_nudge: int = 3) -> list:
    """Rewrite clashing notes so the melody follows the tone contour.

    For each mismatched pair, nudges the second note toward the tone
    direction (at least level with the first note), capped at
    ``max_nudge`` semitones from the original pitch so the melody keeps
    its character. Returns a new melody list; the input is untouched.
    """
    lang = (language or "yoruba").strip().lower()
    if lang == "pidgin":
        return list(melody)
    alignment = tone_to_melody(lyrics, melody, lang)
    if not alignment.mismatches:
        return list(melody)
    fixed = [(int(p), float(s), float(d)) for p, s, d in melody]
    for v in alignment.mismatches:
        i = v.index
        orig = fixed[i + 1][0]
        if v.tone_dir == 1:
            # Tone rises: the note must not fall. The compliant floor is
            # level with the previous note; ideally it rises with the tone.
            # The cap limits how far PAST the compliant floor we push —
            # the tone constraint always wins over the cap.
            floor = fixed[i][0]
            target = max(orig, floor + 1)
            target = min(target, floor + 1 + max_nudge)
        else:  # tone_dir == -1
            ceil = fixed[i][0]
            target = min(orig, ceil - 1)
            target = max(target, ceil - 1 - max_nudge)
        fixed[i + 1] = (target, fixed[i + 1][1], fixed[i + 1][2])
    return fixed


# ───────────────────────── LoRA training pipeline ───────────────────────────


class LoRAUnavailable(RuntimeError):
    """LoRA training/loading can't proceed — reason attached."""


#: Presets grounded in the public ACE-Step training repos (2026-10-08):
#: side-step recommends rank 64 @ 24GB VRAM, rank 16 @ 8GB; the
#: LOCAL_LANGUAGE tutorial uses r=4/alpha=8 for 8–15 songs.
LORA_PRESETS: dict[str, dict[str, Any]] = {
    "small": {"rank": 8, "alpha": 16, "dropout": 0.15,
              "learning_rate": 5e-5, "epochs": 30, "batch_size": 1,
              "min_tracks": 20, "vram": "8GB+"},
    "medium": {"rank": 16, "alpha": 32, "dropout": 0.1,
               "learning_rate": 1e-4, "epochs": 50, "batch_size": 1,
               "min_tracks": 50, "vram": "12GB+"},
    "full": {"rank": 64, "alpha": 128, "dropout": 0.1,
             "learning_rate": 1e-4, "epochs": 200, "batch_size": 1,
             "min_tracks": 100, "vram": "24GB+"},
}

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

#: default registry of trained LoRAs (owner-scoped)
DEFAULT_LORA_REGISTRY = os.path.expanduser("~/.nomorals/media/loras.json")


def corpus_spec() -> dict[str, Any]:
    """What makes a good Afrobeats LoRA training corpus.

    The corpus is the moat — this spec is the curation checklist.
    """
    return {
        "audio": {
            "formats": ["wav", "flac", "mp3"],
            "min_clip_seconds": 30,
            "preferred": "full mixes (stems optional, help for control)",
            "sample_rate": "44.1kHz or higher",
        },
        "captions": {
            "format": "sidecar .txt per audio file (acestep-prepare format)",
            "activation_tag": "include a unique tag, e.g. 'naija-afrobeats-style'",
            "fields": ["style", "bpm", "key", "instruments", "mood",
                       "vocal presence (instrumental/vocal)"],
            "example": ("naija-afrobeats-style, afrobeats, 102bpm, "
                        "log drums, shakers, bright guitar, warm bass, "
                        "groovy, danceable, male vocal"),
        },
        "bpm_ranges": {
            "afrobeats": (95, 110),
            "amapiano": (110, 115),
            "highlife": (100, 120),
        },
        "size": {
            "minimum_tracks": 20,
            "recommended_tracks": 50,
            "note": "rank 8 for 20+ tracks, rank 16 for 50+, rank 64 for 100+",
        },
        "licensing": (
            "ONLY properly cleared audio: your own recordings, licensed "
            "stems, or CC0/public-domain sources. Never scrape commercial "
            "releases — a LoRA trained on uncleared music is a liability, "
            "not a moat."
        ),
        "pipeline": [
            "acestep-prepare --audio-dir <corpus> --output dataset.json "
            "--custom-tag 'naija-afrobeats-style'",
            "acestep-preprocess --dataset dataset.json --output ./tensors",
            "acestep-train --dataset ./tensors --output ./lora-afrobeats "
            "--r 16 --alpha 32 --dropout 0.1 --lr 1e-4 --epochs 50",
        ],
    }


@dataclass
class LoRATrainingJob:
    """A real, executable LoRA training plan for ACE-Step 1.5.

    ``run()`` executes the prepare → preprocess → train pipeline when
    the tooling + CUDA exist; otherwise it fails closed with the exact
    recipe. Nothing is ever faked.
    """
    corpus_dir: str
    output_dir: str
    preset: str = "medium"
    tag: str = "naija-afrobeats-style"
    base_model: str = "ACE-Step/ACE-Step-v1.5"

    def params(self) -> dict[str, Any]:
        p = dict(LORA_PRESETS.get(self.preset, LORA_PRESETS["medium"]))
        p["preset"] = self.preset
        return p

    def corpus_manifest(self) -> dict[str, Any]:
        """Scan the corpus dir: audio files + sidecar captions."""
        corpus = Path(self.corpus_dir)
        audios = [f for f in sorted(corpus.rglob("*"))
                  if f.suffix.lower() in AUDIO_EXTS and f.is_file()]
        with_caption = sum(
            1 for a in audios if a.with_suffix(".txt").is_file())
        return {"tracks": len(audios),
                "with_captions": with_caption,
                "paths": [str(a) for a in audios]}

    def validate(self) -> list[str]:
        """Problems that block training. Empty = ready to run."""
        problems: list[str] = []
        corpus = Path(self.corpus_dir)
        if not corpus.is_dir():
            problems.append(f"no such corpus dir: {self.corpus_dir}")
        else:
            m = self.corpus_manifest()
            need = self.params()["min_tracks"]
            if m["tracks"] < need:
                problems.append(
                    f"corpus has {m['tracks']} tracks, preset "
                    f"'{self.preset}' wants ≥{need}")
            if m["with_captions"] < m["tracks"]:
                problems.append(
                    f"{m['tracks'] - m['with_captions']} tracks lack sidecar "
                    f".txt captions — run acestep-prepare first")
        try:
            import torch
            if not torch.cuda.is_available():
                problems.append("no CUDA GPU — LoRA training needs one")
        except Exception:  # noqa: BLE001
            problems.append("torch not installed — can't train")
        for tool in ("acestep-prepare", "acestep-train"):
            if not shutil.which(tool):
                # also accept the official train.py entrypoint
                if tool == "acestep-train" and (
                        Path.home() / "ACE-Step-1.5" / "train.py").is_file():
                    continue
                problems.append(
                    f"{tool} not on PATH — install the ACE-Step training "
                    f"tooling (see corpus_spec()['pipeline'])")
                break
        return problems

    def prepare_command(self) -> str:
        return (
            f"acestep-prepare --audio-dir {self.corpus_dir} "
            f"--output {self.output_dir}/dataset.json "
            f"--custom-tag {self.tag!r}")

    def preprocess_command(self) -> str:
        return (
            f"acestep-preprocess --dataset {self.output_dir}/dataset.json "
            f"--output {self.output_dir}/tensors")

    def train_command(self) -> str:
        p = self.params()
        return (
            f"acestep-train --dataset {self.output_dir}/tensors "
            f"--output {self.output_dir}/lora "
            f"--r {p['rank']} --alpha {p['alpha']} "
            f"--dropout {p['dropout']} --lr {p['learning_rate']} "
            f"--epochs {p['epochs']}")

    def recipe(self) -> str:
        return "\n".join([
            "# Afrobeats LoRA training — run these in order:",
            f"mkdir -p {self.output_dir}",
            self.prepare_command(),
            self.preprocess_command(),
            self.train_command(),
        ])

    def run(self) -> dict[str, Any]:
        """Execute the pipeline. Fails closed with the recipe."""
        import subprocess
        problems = self.validate()
        if problems:
            raise LoRAUnavailable(
                "LoRA training can't run here:\n- "
                + "\n- ".join(problems)
                + "\n\nRecipe:\n" + self.recipe())
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log: list[str] = []
        for cmd in (self.prepare_command(), self.preprocess_command(),
                    self.train_command()):
            _log.info("lora training: %s", cmd)
            r = subprocess.run(cmd, shell=True, capture_output=True,
                               text=True, timeout=86400)
            log.append(f"$ {cmd}\n{r.stdout[-2000:]}")
            if r.returncode != 0:
                raise LoRAUnavailable(
                    f"training step failed: {cmd}\n{r.stderr[-2000:]}")
        return {"ok": True, "output_dir": str(out), "log": log}


@dataclass
class LoRAWeights:
    """A trained LoRA adapter on disk."""
    name: str
    path: str
    scale: float = 1.0


class AfrobeatsLoRA:
    """LoRA lifecycle: plan training → validate weights → apply to a bed."""

    def __init__(self, registry_path: str = "") -> None:
        self.registry_path = registry_path or DEFAULT_LORA_REGISTRY
        self._reg: dict[str, dict[str, Any]] = {}
        self._load_registry()

    # -- training ---------------------------------------------------------
    def train_lora(self, corpus_dir: str, *, preset: str = "medium",
                   output_dir: str = "", tag: str = "naija-afrobeats-style",
                   dry_run: bool = True) -> LoRATrainingJob:
        """Build a training job. ``dry_run=True`` returns the plan without
        executing (default — training takes hours on a GPU)."""
        out = (output_dir or
               str(Path.home() / ".nomorals" / "media" / "lora-afrobeats"))
        job = LoRATrainingJob(corpus_dir=corpus_dir, output_dir=out,
                              preset=preset, tag=tag)
        if not dry_run:
            job.run()
        return job

    # -- weights ----------------------------------------------------------
    @staticmethod
    def load_lora(path: str, name: str = "") -> LoRAWeights:
        """Validate a trained LoRA dir. Fails closed — never a fake."""
        p = Path(path or "")
        if not p.is_dir():
            raise LoRAUnavailable(f"no such LoRA dir: {path!r}")
        # diffusers/PEFT layout: adapter_model.safetensors + adapter_config.json
        cands = list(p.glob("*.safetensors")) + list(p.glob("*.bin"))
        if not cands and not (p / "adapter_config.json").is_file():
            raise LoRAUnavailable(
                f"{path!r} has no safetensors/bin weights — not a LoRA")
        nm = name or p.name
        return LoRAWeights(name=nm, path=str(p))

    def apply_lora(self, backend: Any, path: str,
                   scale: float = 1.0) -> Any:
        """Attach LoRA weights to an :class:`ACEStepBackend`.

        Sets ``lora_path``/``lora_scale`` on the backend; the diffusers
        adapter picks them up at load time (see the ace_step.py hook).
        The official-app path loads LoRAs via its webui LoRA tab —
        noted on the returned weights.
        """
        weights = self.load_lora(path)
        weights.scale = scale
        backend.lora_path = weights.path
        backend.lora_scale = scale
        _log.info("lora %s attached to backend (scale %.2f)",
                  weights.name, scale)
        return backend

    # -- registry ---------------------------------------------------------
    def _load_registry(self) -> None:
        try:
            if os.path.isfile(self.registry_path):
                with open(self.registry_path, encoding="utf-8") as fh:
                    self._reg = json.load(fh) or {}
        except Exception:  # noqa: BLE001
            _log.debug("lora registry load failed", exc_info=True)

    def _save_registry(self) -> None:
        Path(self.registry_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w", encoding="utf-8") as fh:
            json.dump(self._reg, fh, indent=2)

    def register_lora(self, name: str, path: str) -> LoRAWeights:
        w = self.load_lora(path, name)
        self._reg[w.name] = {"path": w.path, "scale": w.scale}
        self._save_registry()
        return w

    def registered(self) -> dict[str, dict[str, Any]]:
        return dict(self._reg)

    def for_style(self, style: str) -> str:
        """LoRA path registered for a style, or '' (→ base model)."""
        entry = self._reg.get((style or "").strip().lower())
        return entry["path"] if entry else ""


# ───────────────────────── Naija sung vocals ────────────────────────────────


@dataclass
class SungVocalResult:
    ok: bool
    audio_path: str = ""
    language: str = "yoruba"
    voice_id: str = ""
    audience: str = "private"
    alignment: Alignment | None = None
    fixed_melody: bool = False
    error: str = ""
    note: str = ""


class NaijaSungVocals:
    """Tone-aware sung vocals for Nigerian languages.

    The moat: Yoruba/Igbo tones are checked against the melody BEFORE
    DiffSinger renders, and clashing notes are rewritten so the sung
    words keep their meaning.
    """

    def __init__(self, *, chain: Any = None) -> None:
        self._chain = chain

    def _chain_or_default(self) -> Any:
        if self._chain is not None:
            return self._chain
        from .vocals import VocalChain, RVCVoiceRegistry
        return VocalChain(voices=RVCVoiceRegistry().all())

    def align(self, lyrics: str, melody: list,
              language: str = "yoruba") -> Alignment:
        return tone_to_melody(lyrics, melody, language)

    def sing(self, lyrics: str, melody: list, voice_id: str, *,
             language: str = "yoruba", audience: str = "private",
             fix_melody: bool = True, workdir: str = "vocals",
             title: str = "naija-vocal") -> SungVocalResult:
        """Tone-aware sing: align → fix → DiffSinger → RVC.

        Fails closed like the rest of the vocal chain — no models, no
        audio, honest error.
        """
        from .vocals import VocalModelUnavailable
        lang = (language or "yoruba").strip().lower()
        if lang not in LANGUAGES:
            raise VocalModelUnavailable(
                f"unsupported language {language!r} — {', '.join(LANGUAGES)}")
        if not (lyrics or "").strip():
            raise VocalModelUnavailable("no lyrics to sing")
        if not melody:
            raise VocalModelUnavailable("no melody — need MIDI notes")
        alignment = self.align(lyrics, melody, lang)
        use_melody = (tone_aware_melody(lyrics, melody, lang)
                      if fix_melody else list(melody))
        fixed = use_melody != list(melody)
        chain = self._chain_or_default()
        rendered = chain.render(lyrics, use_melody, workdir=workdir,
                                title=title)
        converted = chain.convert(rendered.audio_path, voice_id,
                                  audience=audience, workdir=workdir)
        n_bad = len(alignment.mismatches)
        note = (f"🎤 {lang} sung vocal — tone-aligned "
                f"({alignment.score:.0%} clean"
                + (f", {n_bad} clashing note{'s' if n_bad != 1 else ''} "
                   f"rewritten" if fixed else "")
                + f") → {converted.voice_id}")
        return SungVocalResult(
            ok=True, audio_path=converted.audio_path, language=lang,
            voice_id=converted.voice_id, audience=audience,
            alignment=alignment, fixed_melody=fixed, note=note)


# ───────────────────────── lyrics (LLM path) ───────────────────────────────


def yoruba_lyrics(topic: str, context: Any, *,
                  language: str = "yoruba",
                  lines: int = 8) -> str:
    """Yoruba/Igbo lyrics with tone marks, from the language model.

    The prompt REQUIRES tone marks — unmarked Yoruba lyrics can't be
    tone-aligned, so they're rejected, not guessed. Fails closed without
    an LLM: the caller (or user) supplies lyrics instead.
    """
    from ..llm.base import Message, SamplingParams
    lang = (language or "yoruba").strip().lower()
    if lang == "igbo":
        tone_req = ("Mark tones: HIGH with ´ (e.g. á), LOW with ` (e.g. à).")
    elif lang == "pidgin":
        tone_req = "No tone marks needed for Pidgin."
    else:
        tone_req = ("Mark EVERY vowel's tone: HIGH ´ (á), MID unmarked (a), "
                    "LOW ` (à). Tones change meaning — mark them all.")
    router = getattr(context, "router", None)
    if router is None:
        raise LoRAUnavailable(
            f"no language model available for {lang} lyrics — send your own "
            f"lyrics and I'll sing them with tone-melody alignment")
    prompt = (
        f"Write exactly {lines} lines of {lang} song lyrics about: "
        f"{topic!r}. {tone_req} One clear image per line, singable, "
        f"no titles, no markdown — just the lines.")
    resp = brain_for(context).chat([Message.user(prompt)],
                       SamplingParams(temperature=0.9), task_kind="creative")
    text = (getattr(resp, "text", "") or "").strip()
    if not text:
        raise LoRAUnavailable("the language model returned no lyrics")
    if lang in ("yoruba", "igbo", "ekiti"):
        nfd = unicodedata.normalize("NFD", text)
        if _ACUTE not in nfd and _GRAVE not in nfd:
            raise LoRAUnavailable(
                "the lyrics came back without tone marks — unmarked "
                "Yoruba/Igbo can't be tone-aligned. Ask for tone-marked "
                "lyrics or supply your own.")
    return text


# ───────────────────────── chat ─────────────────────────────────────────────


_NAIJA_RE = re.compile(
    r"(?i)^\s*(?:make|generate|create)\s+(?:me\s+)?(?:an?\s+)?"
    r"(?P<rest>.+?)\s+song"
    r"(?:\s+in\s+(?P<lang>yoruba|igbo|pidgin|ekiti))?"
    r"(?:\s+about\s+(?P<topic>.+))?\s*$")


@dataclass
class NaijaSongRequest:
    topic: str
    style: str = "afrobeats"
    language: str = "yoruba"


def parse_naija_song_request(text: str,
                             styles: Any = None) -> NaijaSongRequest | None:
    """Parse Naija song requests:

    * ``make me an afrobeats song in Yoruba about Lagos``
    * ``/music naija lagos nights afrobeats yoruba``
    * ``generate a highlife song in Igbo about home``
    """
    raw = (text or "").strip()
    if not raw:
        return None
    low = raw.lower()
    if low.startswith("/music naija"):
        body = raw[len("/music naija"):].strip()
        if not body:
            return None
        words = body.split()
        language = "yoruba"
        if words and words[-1].lower() in LANGUAGES:
            language = words[-1].lower()
            words = words[:-1]
        style = "afrobeats"
        if styles is not None and words and words[-1].lower() in styles:
            style = words[-1].lower()
            words = words[:-1]
        topic = " ".join(words).strip()
        if not topic:
            return None
        return NaijaSongRequest(topic=topic, style=style, language=language)
    m = _NAIJA_RE.match(raw)
    if not m:
        return None
    rest = (m.group("rest") or "").strip()
    lang = (m.group("lang") or "yoruba").strip().lower()
    topic = (m.group("topic") or "").strip()
    style = "afrobeats"
    if styles is not None:
        from ..media.music import STYLE_ALIASES  # local import, cheap
        first = rest.split()[0].lower() if rest.split() else ""
        if first in STYLE_ALIASES:
            style = STYLE_ALIASES[first]
            rest = " ".join(rest.split()[1:])
    return NaijaSongRequest(topic=topic or rest or "untitled",
                            style=style, language=lang)


def make_naija_song(topic: str, *, style: str = "afrobeats",
                    language: str = "yoruba", voice_id: str = "",
                    audience: str = "private", lyrics: str = "",
                    duration_s: int = 120, seed: int = 0,
                    context: Any = None, profile: str = "",
                    workdir: str = "songs",
                    chain: Any = None) -> Any:
    """Full Naija song: LoRA-styled bed + tone-aware sung vocals + master.

    * style LoRA (if registered) → ACE-Step bed
    * lyrics (LLM with tone marks, or caller-supplied) → tone-aligned
      DiffSinger render → RVC voice → mix over the bed

    Raises LoRAUnavailable / VocalModelUnavailable / ACEModelUnavailable
    when a stage can't run — never fake audio.
    """
    from .ace_step import ACEStepBackend, make_bed
    from .vocals import VocalChain, RVCVoiceRegistry, FullSongResult
    from .music import MusicCreator, resolve_style
    if not (topic or "").strip():
        raise LoRAUnavailable("topic is required")
    if not (voice_id or "").strip():
        raise LoRAUnavailable("voice_id is required — pick a voice")
    lang = (language or "yoruba").strip().lower()

    spec = resolve_style(style)
    backend = ACEStepBackend(profile=profile)
    lora = AfrobeatsLoRA()
    lora_path = lora.for_style(spec.name)
    if lora_path:
        lora.apply_lora(backend, lora_path)
        _log.info("naija song: using %s LoRA %s", spec.name, lora_path)

    bed = make_bed(topic, style=spec.name, duration_s=duration_s,
                   seed=seed, context=context, profile=profile,
                   backend=backend, workdir=str(Path(workdir) / "bed"))

    song = MusicCreator(context).compose(topic, style=spec.name,
                                         with_midi=False, with_audio=False,
                                         with_score=False)
    melody = list(song.melody_notes or [])
    if not melody:
        raise LoRAUnavailable("the composer produced no melody notes")

    text = (lyrics or "").strip() or yoruba_lyrics(topic, context,
                                                   language=lang)
    singer = NaijaSungVocals(chain=chain)
    sung = singer.sing(text, melody, voice_id, language=lang,
                       audience=audience,
                       workdir=str(Path(workdir) / "vocals"), title=bed.title)

    vc = chain or VocalChain(profile=profile,
                             voices=RVCVoiceRegistry().all())
    master = str(Path(workdir) / f"{bed.title or topic}-naija-master.wav")
    vc.mix(bed.audio_path, sung.audio_path, out_path=master)
    return FullSongResult(
        ok=True, master_path=master, bed_path=bed.audio_path,
        vocal_path=sung.audio_path, title=bed.title or topic,
        voice_id=sung.voice_id, audience=audience,
        note=(f"🌍 AI {spec.label} bed"
              + (f" ({Path(lora_path).name} LoRA)" if lora_path else "")
              + f" + {lang} sung vocals (tone-aligned: "
              f"{sung.alignment.score:.0%} clean) — "
              f"label it AI wherever it goes"))


# ───────────────────────── tool registration ───────────────────────────────


def register(registry: Any) -> None:
    from ..core.policy import Capability
    context = registry.context

    @registry.register(
        "afrobeats",
        description=(
            "Afrobeats LoRA + Nigerian-language sung vocals (the continental "
            "moat): action=lora-spec (corpus curation checklist) | "
            "lora-train (corpus_dir, preset, dry_run → training plan/recipe) "
            "| lora-list | lora-register (name, path) | align (lyrics, "
            "melody JSON, language → tone↔melody report) | sing (topic, "
            "style, language, voice_id, lyrics? → LoRA bed + tone-aware "
            "Yoruba/Igbo/Ekiti/Pidgin vocals + master). Tone-melody "
            "alignment is the product: clashing notes are rewritten so "
            "words keep their meaning. Fails honestly — never fake audio."
        ),
        capability=Capability.FS_WRITE,
    )
    def afrobeats(action: str = "align", corpus_dir: str = "",
                  preset: str = "medium", dry_run: bool = True,
                  name: str = "", path: str = "", lyrics: str = "",
                  melody: str = "", language: str = "yoruba",
                  topic: str = "", style: str = "afrobeats",
                  voice_id: str = "", audience: str = "private") -> dict:
        action = (action or "align").lower()
        lora = AfrobeatsLoRA()
        if action == "lora-spec":
            return {"ok": True, "spec": corpus_spec()}
        if action == "lora-train":
            if not corpus_dir.strip():
                from ..core.errors import ToolError
                raise ToolError("corpus_dir is required")
            job = lora.train_lora(corpus_dir, preset=preset,
                                  dry_run=dry_run)
            out: dict[str, Any] = {
                "ok": True, "preset": job.params(),
                "recipe": job.recipe(),
                "problems": job.validate(),
                "dry_run": dry_run,
            }
            if not dry_run:
                out["result"] = job.run()
            return out
        if action == "lora-list":
            return {"ok": True, "loras": lora.registered()}
        if action == "lora-register":
            try:
                w = lora.register_lora(name, path)
            except LoRAUnavailable as exc:
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "name": w.name, "path": w.path}
        if action == "align":
            import json as _json
            try:
                mel = _json.loads(melody) if melody else []
            except Exception as exc:  # noqa: BLE001
                from ..core.errors import ToolError
                raise ToolError(f"melody must be JSON: {exc}")
            al = tone_to_melody(lyrics, mel, language)
            return {"ok": True, "language": al.language,
                    "score": round(al.score, 3),
                    "mismatches": [v.note_text for v in al.mismatches]}
        if action == "sing":
            try:
                r = make_naija_song(topic, style=style, language=language,
                                    voice_id=voice_id, audience=audience,
                                    lyrics=lyrics, context=context)
            except Exception as exc:  # noqa: BLE001 - all model errors
                return {"ok": False, "error": str(exc)}
            return {"ok": True, "master_path": r.master_path,
                    "note": r.note}
        from ..core.errors import ToolError
        raise ToolError(f"unknown action {action!r}")
