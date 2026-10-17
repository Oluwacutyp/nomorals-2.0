"""Global genre LoRA training — the Tier-3 outclass, for every genre.

Tier 3. Generalises :mod:`nomorals.media.afrobeats`' Afrobeats LoRA
pipeline (which stays untouched) to ANY genre: afrobeats, amapiano,
highlife, K-pop, Bollywood, Latin, EDM, ... Each genre gets a
:class:`GenreLoRAJob` — corpus spec (legally clean sources only),
activation tag, preset (rank/alpha/lr/epochs), and the exact
prepare → preprocess → train recipe. Plus the Kaggle free-GPU runbook
(~30h/week free GPU: split training into resumable chunks, checkpoint
every epoch, never assume a session survives).

SCOPE — universal, Nigeria as DEFAULT, not boundary. N genre packs;
afrobeats/amapiano are the default pack, not the boundary.

Legally clean corpus — the hard rule, repeated because it is the
moat's legal floor: own recordings, licensed stems, CC0 / public
domain, or explicitly licensed datasets. NEVER scraped commercial
releases. A LoRA trained on uncleared music is a liability, not a
moat. :func:`corpus_licence_check` scans a corpus dir's sidecar
licences and refuses to bless a corpus with unlicensed tracks.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .afrobeats import LORA_PRESETS

__all__ = [
    "GENRE_LORA_PRESETS",
    "GenreLoRAJob",
    "genre_corpus_spec",
    "corpus_licence_check",
    "kaggle_runbook",
    "register",
]

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

#: Per-genre activation tags + BPM windows for the caption sidecars.
GENRE_LORA_PRESETS: dict[str, dict[str, Any]] = {
    "afrobeats": {"tag": "naija-afrobeats-style", "bpm": (95, 110)},
    "amapiano": {"tag": "amapiano-style", "bpm": (110, 115)},
    "highlife": {"tag": "highlife-style", "bpm": (100, 120)},
    "k-pop": {"tag": "kpop-style", "bpm": (95, 130)},
    "bollywood": {"tag": "bollywood-film-style", "bpm": (90, 130)},
    "reggaeton": {"tag": "reggaeton-dembow-style", "bpm": (90, 100)},
    "house": {"tag": "house-style", "bpm": (120, 130)},
    "edm": {"tag": "edm-style", "bpm": (128, 150)},
    "hip-hop": {"tag": "hiphop-boombap-style", "bpm": (80, 100)},
    "trap": {"tag": "trap-style", "bpm": (130, 150)},
    "pop": {"tag": "pop-style", "bpm": (90, 120)},
    "rock": {"tag": "rock-style", "bpm": (110, 150)},
    "jazz": {"tag": "jazz-style", "bpm": (120, 200)},
    "gospel": {"tag": "gospel-style", "bpm": (70, 130)},
}

CLEAN_LICENCES = ("cc0", "public-domain", "public domain", "own-recording",
                  "licensed", "cc-by", "cc-by-sa")


def genre_corpus_spec(genre: str) -> dict[str, Any]:
    """Curation checklist for a legally clean genre LoRA corpus."""
    g = (genre or "").strip().lower()
    preset = GENRE_LORA_PRESETS.get(g, {"tag": f"{g}-style",
                                        "bpm": (80, 140)})
    return {
        "genre": g,
        "activation_tag": preset["tag"],
        "bpm_window": preset["bpm"],
        "audio": {"formats": ["wav", "flac", "mp3"],
                  "min_clip_seconds": 30,
                  "preferred": "full mixes (stems optional)",
                  "sample_rate": "44.1kHz or higher"},
        "captions": {"format": "sidecar .txt per audio file",
                     "fields": ["style", "bpm", "key", "instruments",
                                "mood", "vocal presence"],
                     "example": (f"{preset['tag']}, {g}, "
                                 f"{preset['bpm'][0]}bpm, ...")},
        "licence": {"rule": "ONLY cleared audio: own recordings, licensed "
                            "stems, CC0/public-domain, explicitly licensed "
                            "datasets. NEVER scraped commercial releases.",
                    "sidecar": "each track needs a .licence file naming "
                               "its licence (cc0 / public-domain / "
                               "own-recording / licensed / cc-by / cc-by-sa)",
                    "accepted": list(CLEAN_LICENCES)},
        "size": {"minimum_tracks": 20, "recommended_tracks": 50,
                 "note": "rank 8 for 20+ tracks, rank 16 for 50+, rank 64 for 100+"},
        "pipeline": ["acestep-prepare --audio-dir <corpus> --output dataset.json "
                     f"--custom-tag '{preset['tag']}'",
                     "acestep-preprocess --dataset dataset.json --output ./tensors",
                     "acestep-train --dataset ./tensors --output ./lora-<genre> "
                     "--r 16 --alpha 32 --dropout 0.1 --lr 1e-4 --epochs 50"],
    }


def corpus_licence_check(corpus_dir: str) -> dict[str, Any]:
    """Scan a corpus dir: every track needs a .licence sidecar.

    Returns {ok, tracks, licensed, unlicensed: [...]}. ``ok`` is False
    when ANY track lacks a clean-licence sidecar — the training job
    must not run on that corpus.
    """
    corpus = Path(corpus_dir)
    if not corpus.is_dir():
        return {"ok": False, "tracks": 0, "licensed": 0,
                "unlicensed": [f"no such corpus dir: {corpus_dir}"]}
    audios = [f for f in sorted(corpus.rglob("*"))
              if f.suffix.lower() in AUDIO_EXTS and f.is_file()]
    unlicensed: list[str] = []
    licensed = 0
    for a in audios:
        lic = a.with_suffix(".licence")
        if not lic.is_file():
            unlicensed.append(f"{a.name}: missing .licence sidecar")
            continue
        text = lic.read_text(errors="replace").strip().lower()
        if any(c in text for c in CLEAN_LICENCES):
            licensed += 1
        else:
            unlicensed.append(f"{a.name}: licence {text!r} not in clean list")
    return {"ok": not unlicensed and bool(audios), "tracks": len(audios),
            "licensed": licensed, "unlicensed": unlicensed}


def kaggle_runbook(genre: str, preset: str = "medium") -> dict[str, Any]:
    """Kaggle free-GPU runbook for a genre LoRA (~30h/week free GPU).

    Kaggle gives ~30h/week of free GPU (T4/P100 class). The runbook
    splits training into resumable chunks: checkpoint every epoch,
    sync checkpoints to a Kaggle dataset (persistent across sessions),
    resume from the latest checkpoint. Never assume a session survives
    — the 11h Kaggle session cap is real.
    """
    params = dict(LORA_PRESETS.get(preset, LORA_PRESETS["medium"]))
    return {
        "genre": genre,
        "preset": preset,
        "params": params,
        "platform": "Kaggle (free GPU ~30h/week, ~11h/session cap)",
        "steps": [
            "1. Upload the cleared corpus as a Kaggle dataset (private).",
            "2. Notebook: install ACE-Step training tooling (uv/pip), mount dataset.",
            "3. acestep-prepare → acestep-preprocess (CPU-safe, cache tensors).",
            "4. acestep-train with --save-every-epoch; sync ./lora-<genre> to a Kaggle dataset after every epoch.",
            "5. Session ends? Start a new session, pull the checkpoint dataset, resume with --resume <ckpt>.",
            "6. Download the final adapter; register it: AfrobeatsLoRA-style registry under ~/.nomorals/media/loras.json.",
            "7. Validate: generate 5 beds with the LoRA, critique-loop them, keep only if they beat base ACE-Step on taste + critique.",
        ],
        "budget_note": ("~30h/week free GPU: a rank-16 / 50-epoch run fits "
                        "in 2–3 sessions. Do not train rank-64 on Kaggle "
                        "free tier — it will not finish."),
    }


@dataclass
class GenreLoRAJob:
    """A real, executable LoRA training plan for one genre."""

    genre: str
    corpus_dir: str
    output_dir: str
    preset: str = "medium"
    base_model: str = "ACE-Step/ACE-Step-v1.5"

    @property
    def tag(self) -> str:
        return GENRE_LORA_PRESETS.get(
            self.genre, {"tag": f"{self.genre}-style"})["tag"]

    def params(self) -> dict[str, Any]:
        p = dict(LORA_PRESETS.get(self.preset, LORA_PRESETS["medium"]))
        p["preset"] = self.preset
        return p

    def validate(self) -> list[str]:
        problems: list[str] = []
        lic = corpus_licence_check(self.corpus_dir)
        if not lic["ok"]:
            problems.append(
                f"corpus licence check failed: {lic['unlicensed'][:3]}")
        if lic["tracks"] < self.params()["min_tracks"]:
            problems.append(
                f"corpus has {lic['tracks']} tracks, preset "
                f"'{self.preset}' wants ≥{self.params()['min_tracks']}")
        try:
            import torch
            if not torch.cuda.is_available():
                problems.append("no CUDA GPU — LoRA training needs one")
        except Exception:  # noqa: BLE001
            problems.append("torch not installed — can't train")
        for tool in ("acestep-prepare", "acestep-train"):
            if not shutil.which(tool):
                if tool == "acestep-train" and (
                        Path.home() / "ACE-Step-1.5" / "train.py").is_file():
                    continue
                problems.append(f"{tool} not on PATH — install ACE-Step training tooling")
                break
        return problems

    def recipe(self) -> str:
        return "\n".join([
            f"# {self.genre} LoRA training — run in order:",
            f"mkdir -p {self.output_dir}",
            f"acestep-prepare --audio-dir {self.corpus_dir} "
            f"--output {self.output_dir}/dataset.json "
            f"--custom-tag {self.tag!r}",
            f"acestep-preprocess --dataset {self.output_dir}/dataset.json "
            f"--output {self.output_dir}/tensors",
            f"acestep-train --dataset {self.output_dir}/tensors "
            f"--output {self.output_dir}/lora "
            f"--r {self.params()['rank']} --alpha {self.params()['alpha']} "
            f"--dropout {self.params()['dropout']} "
            f"--lr {self.params()['learning_rate']} "
            f"--epochs {self.params()['epochs']}",
        ])

    def _argv(self) -> list[list[str]]:
        """List-form commands for :meth:`run` (same steps as :meth:`recipe`).

        List form + ``shell=False`` is required on Windows: the string
        form goes through ``cmd.exe`` (no POSIX quoting, ``!r`` repr
        quoting is not shell quoting), and a corpus/output path
        containing a space or ``&`` would split or inject a command.
        ``recipe()`` stays the human-readable string form.
        """
        p = self.params()
        out = Path(self.output_dir)
        return [
            ["acestep-prepare", "--audio-dir", str(self.corpus_dir),
             "--output", str(out / "dataset.json"),
             "--custom-tag", str(self.tag)],
            ["acestep-preprocess", "--dataset", str(out / "dataset.json"),
             "--output", str(out / "tensors")],
            ["acestep-train", "--dataset", str(out / "tensors"),
             "--output", str(out / "lora"),
             "--r", str(p["rank"]), "--alpha", str(p["alpha"]),
             "--dropout", str(p["dropout"]),
             "--lr", str(p["learning_rate"]),
             "--epochs", str(p["epochs"])],
        ]

    def run(self) -> dict[str, Any]:
        import shlex
        import subprocess
        problems = self.validate()
        if problems:
            raise RuntimeError("Genre LoRA training can't run here:\n- "
                               + "\n- ".join(problems)
                               + "\n\nRecipe:\n" + self.recipe())
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        log: list[str] = []
        for argv in self._argv():
            cmd = shlex.join(argv)
            r = subprocess.run(argv, shell=False, capture_output=True,
                               text=True, encoding="utf-8",
                               errors="replace", timeout=86400)
            log.append(f"$ {cmd}\n{r.stdout[-2000:]}")
            if r.returncode != 0:
                raise RuntimeError(f"training step failed: {cmd}\n{r.stderr[-2000:]}")
        return {"ok": True, "output_dir": str(out), "log": log}


def register(registry: Any) -> None:
    """Tool registration: ``genre_lora`` — global LoRA training."""
    from ..core.policy import Capability

    @registry.register(
        "genre_lora",
        description=(
            "Global genre LoRA training: action=spec (genre → legally-clean "
            "corpus checklist) | licence-check (corpus_dir → every track "
            "needs a clean-licence sidecar) | kaggle (genre, preset → "
            "free-GPU runbook) | train (corpus_dir, genre, preset, "
            "dry_run → plan/recipe, or execute). NEVER trains on uncleared "
            "music — the licence check gates the job."
        ),
        capability=Capability.FS_WRITE,
    )
    def genre_lora(action: str = "spec", genre: str = "afrobeats",
                   corpus_dir: str = "", preset: str = "medium",
                   dry_run: bool = True,
                   output_dir: str = "") -> dict:
        action = (action or "spec").lower()
        if action == "spec":
            return {"ok": True, "spec": genre_corpus_spec(genre)}
        if action == "licence-check":
            return {"ok": True, **corpus_licence_check(corpus_dir)}
        if action == "kaggle":
            return {"ok": True, **kaggle_runbook(genre, preset)}
        if action == "train":
            job = GenreLoRAJob(genre=genre, corpus_dir=corpus_dir,
                               output_dir=output_dir or f"./lora-{genre}",
                               preset=preset)
            out: dict[str, Any] = {"ok": True, "recipe": job.recipe(),
                                   "problems": job.validate(),
                                   "dry_run": dry_run}
            if not dry_run:
                out["result"] = job.run()
            return out
        from ..core.errors import ToolError
        raise ToolError(f"unknown action {action!r}")
