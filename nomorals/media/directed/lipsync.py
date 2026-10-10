"""Lip sync — audio-driven lip movement on video/photo subjects.

Backends (mined, see LIPSYNC_MINING.md):
- Wav2Lip (zdh6090/Wav2Lip): video -> lip-synced video. ~2GB VRAM, 8GB GPU.
  Research license (non-commercial) — noted in status.
- LatentSync 1.5 (ByteDance, Apache 2.0): higher quality, 8GB VRAM.
  v1.6 needs 18GB — wired as alt.
- SadTalker: single photo + audio -> talking head with head motion.
- envelope_warp (CPU): audio RMS envelope -> jaw-region warp. REAL
  audio-driven motion (mouth opens on energy, closes on silence),
  honestly labeled. Needs a face box (or pose-derived).

lip_sync() routes to the best available. dub_video() orchestrates
translate-text -> TTS voice -> lip sync -> mux for dubbing.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MODELS_DIR = Path.home() / ".devon-models"


class ModelUnavailable(RuntimeError):
    """Neural lip-sync backend not installed. Carries the fix."""


@dataclass
class LipSyncResult:
    path: str
    backend: str          # "wav2lip" | "latentsync" | "sadtaker" | "warp"
    note: str = ""


def _ffmpeg() -> str | None:
    from shutil import which
    return which("ffmpeg")


def _has_torch_cuda() -> tuple[bool, bool]:
    try:
        import torch
        return True, torch.cuda.is_available()
    except Exception:
        return False, False


# ── backend status checks ────────────────────────────────────────────
def wav2lip_status() -> dict:
    has_torch, cuda = _has_torch_cuda()
    repo = MODELS_DIR / "wav2lip" / "repo"
    weights = MODELS_DIR / "wav2lip" / "wav2lip.pth"
    sfd = MODELS_DIR / "wav2lip" / "s3fd.pth"
    ok = has_torch and cuda and repo.is_dir() and weights.exists()
    return {
        "available": bool(ok), "torch": has_torch, "cuda": cuda,
        "license": "research/non-commercial (Wav2Lip upstream)",
        "reason": (
            "Wav2Lip needs torch+CUDA, the repo, and weights "
            "(wav2lip.pth ~350MB + s3fd.pth ~100MB). "
            "Install: git clone https://github.com/zdh6090/Wav2Lip "
            f"{repo} && download weights into {MODELS_DIR / 'wav2lip'}. "
            "See LIPSYNC_MINING.md."
        ),
    }


def latentsync_status() -> dict:
    has_torch, cuda = _has_torch_cuda()
    repo = MODELS_DIR / "latentsync" / "repo"
    unet = MODELS_DIR / "latentsync" / "latentsync_unet.pt"
    ok = has_torch and cuda and repo.is_dir() and unet.exists()
    vram = 0
    if cuda:
        try:
            import torch
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            pass
    return {
        "available": bool(ok and vram >= 7.5), "torch": has_torch,
        "cuda": cuda, "vram_gb": round(vram, 1),
        "license": "Apache 2.0 (commercial OK)",
        "reason": (
            "LatentSync 1.5 needs torch+CUDA (8GB+ VRAM), the repo, and "
            "checkpoints (latentsync_unet.pt + whisper tiny.pt). "
            "Install: git clone https://github.com/bytedance/LatentSync "
            f"{repo} && download checkpoints into "
            f"{MODELS_DIR / 'latentsync'}. v1.6 needs 18GB VRAM."
        ),
    }


def sadtaker_status() -> dict:
    has_torch, cuda = _has_torch_cuda()
    repo = MODELS_DIR / "sadtaker" / "repo"
    ok = has_torch and cuda and repo.is_dir()
    return {
        "available": bool(ok), "torch": has_torch, "cuda": cuda,
        "reason": (
            "SadTalker needs torch+CUDA and the repo + checkpoints. "
            "Install: git clone https://github.com/OpenTalker/SadTalker "
            f"{repo} && follow its checkpoint download script."
        ),
    }


# ── neural backends (subprocess to the repos) ────────────────────────
def wav2lip_sync(video: str, audio: str, *, out_path: str | None = None,
                 workdir: str | None = None) -> LipSyncResult:
    st = wav2lip_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="w2l_"))
    wd.mkdir(parents=True, exist_ok=True)
    out = out_path or str(wd / "lipsync.mp4")
    repo = MODELS_DIR / "wav2lip" / "repo"
    ckpt = MODELS_DIR / "wav2lip" / "wav2lip.pth"
    proc = subprocess.run(
        ["python", "inference.py",
         "--checkpoint_path", str(ckpt),
         "--face", video, "--audio", audio,
         "--outfile", out],
        cwd=str(repo), capture_output=True, timeout=3600)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("Wav2Lip failed: "
                           + proc.stderr.decode()[-2000:])
    return LipSyncResult(path=out, backend="wav2lip",
                         note="neural lip sync (Wav2Lip, 96px mouth region)")


def latentsync_sync(video: str, audio: str, *, out_path: str | None = None,
                    workdir: str | None = None) -> LipSyncResult:
    st = latentsync_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ls_"))
    wd.mkdir(parents=True, exist_ok=True)
    out = out_path or str(wd / "lipsync.mp4")
    repo = MODELS_DIR / "latentsync" / "repo"
    proc = subprocess.run(
        ["bash", "inference.sh", video, audio, out],
        cwd=str(repo), capture_output=True, timeout=3600)
    if proc.returncode != 0 or not os.path.exists(out):
        raise RuntimeError("LatentSync failed: "
                           + proc.stderr.decode()[-2000:])
    return LipSyncResult(path=out, backend="latentsync",
                         note="neural lip sync (LatentSync, diffusion)")


def sadtaker_animate(image: str, audio: str, *,
                     out_path: str | None = None,
                     workdir: str | None = None) -> LipSyncResult:
    """Single photo + audio -> talking head with head motion."""
    st = sadtaker_status()
    if not st["available"]:
        raise ModelUnavailable(st["reason"])
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="st_"))
    wd.mkdir(parents=True, exist_ok=True)
    out = out_path or str(wd / "talking_head.mp4")
    repo = MODELS_DIR / "sadtaker" / "repo"
    proc = subprocess.run(
        ["python", "inference.py",
         "--source_image", image, "--driven_audio", audio,
         "--result_dir", str(wd)],
        cwd=str(repo), capture_output=True, timeout=3600)
    # SadTalker names output itself; find the mp4
    mp4s = sorted(wd.glob("*.mp4"))
    if proc.returncode != 0 or not mp4s:
        raise RuntimeError("SadTalker failed: "
                           + proc.stderr.decode()[-2000:])
    if str(mp4s[0]) != out:
        import shutil
        shutil.move(str(mp4s[0]), out)
    return LipSyncResult(path=out, backend="sadtaker",
                         note="talking head from photo (SadTalker, 3DMM)")


# ── CPU visemes: phoneme -> mouth-shape map ──────────────────────────
# The CPU fallback used to be jaw-only (mouth opens on energy). Now it
# renders real viseme shapes — jaw openness, lip width, lip rounding —
# driven either by phonemes (from text, uniform timing — honest, not
# forced alignment) or by an acoustic heuristic (energy + spectral
# centroid) when no text is available. Neural backends still win on
# quality; this is the honest CPU tier.

#: viseme -> (jaw_open 0..1, lip_width ~0.5..1.2, lip_round 0..1)
VISEME_SHAPES: dict[str, tuple[float, float, float]] = {
    "sil": (0.00, 1.00, 0.00),   # closed / silence
    "MBP": (0.00, 0.95, 0.10),   # m, b, p — lips pressed
    "FV":  (0.15, 1.00, 0.00),   # f, v — teeth on lower lip
    "TDN": (0.30, 1.00, 0.00),   # t, d, n — tongue to ridge
    "SZ":  (0.20, 1.05, 0.00),   # s, z — narrow hiss
    "TH":  (0.35, 0.95, 0.00),   # th — tongue between teeth
    "L":   (0.35, 1.00, 0.00),   # l
    "SH":  (0.45, 0.70, 0.70),   # sh, ch — rounded
    "IY":  (0.25, 1.20, 0.00),   # beat — wide smile
    "IH":  (0.30, 1.05, 0.00),   # bit
    "EH":  (0.45, 1.10, 0.00),   # bed
    "AE":  (0.75, 1.15, 0.00),   # cat — wide open
    "AH":  (0.90, 1.00, 0.00),   # father — tall open
    "AA":  (0.80, 1.05, 0.00),   # hot
    "AO":  (0.70, 0.90, 0.40),   # bought — open rounded
    "ER":  (0.40, 0.85, 0.30),   # bird — tight mid
    "OW":  (0.55, 0.65, 0.80),   # boat — rounded mid
    "UW":  (0.35, 0.60, 0.90),   # boot — small round
    "W":   (0.30, 0.55, 1.00),   # we — puckered
    "R":   (0.40, 0.80, 0.40),   # red — slight round
    "KG":  (0.60, 0.95, 0.10),   # k, g — back open
}

#: ARPAbet phoneme -> viseme
PHONEME_TO_VISEME: dict[str, str] = {
    "AA": "AA", "AE": "AE", "AH": "AH", "AO": "AO", "AW": "OW",
    "AY": "AH", "EH": "EH", "ER": "ER", "EY": "EH", "IH": "IH",
    "IY": "IY", "OW": "OW", "OY": "OW", "UH": "UW", "UW": "UW",
    "B": "MBP", "CH": "SH", "D": "TDN", "DH": "TH", "F": "FV",
    "G": "KG", "HH": "sil", "JH": "SH", "K": "KG", "L": "L",
    "M": "MBP", "N": "TDN", "NG": "KG", "P": "MBP", "R": "R",
    "S": "SZ", "SH": "SH", "T": "TDN", "TH": "TH", "V": "FV",
    "W": "W", "Y": "IY", "Z": "SZ", "ZH": "SH",
}


def phoneme_to_viseme(phoneme: str) -> str:
    """ARPAbet phoneme -> viseme name. Unknown -> silence (closed)."""
    return PHONEME_TO_VISEME.get((phoneme or "").strip().upper(), "sil")


def mouth_shape_for_viseme(viseme: str) -> tuple[float, float, float]:
    """Viseme name -> (jaw_open, lip_width, lip_round)."""
    return VISEME_SHAPES.get((viseme or "").strip().upper(), VISEME_SHAPES["sil"])


# ── compact English G2P ──────────────────────────────────────────────
# Common-word ARPAbet dictionary + rule-based fallback. Covers everyday
# speech well; long-tail words get a reasonable approximation. This is a
# heuristic front-end — for broadcast quality plug in a real G2P
# (e.g. g2p-en / phonemizer) via text_to_phonemes()'s interface.
_G2P_WORDS: dict[str, tuple[str, ...]] = {
    "the": ("DH", "AH"), "a": ("AH",), "an": ("AE", "N"),
    "and": ("AE", "N", "D"), "or": ("AO", "R"), "but": ("B", "AH", "T"),
    "to": ("T", "UW"), "of": ("AH", "V"), "in": ("IH", "N"),
    "on": ("AA", "N"), "for": ("F", "AO", "R"), "with": ("W", "IH", "DH"),
    "is": ("IH", "Z"), "are": ("AA", "R"), "was": ("W", "AH", "Z"),
    "were": ("W", "ER"), "be": ("B", "IY"), "been": ("B", "IH", "N"),
    "have": ("HH", "AE", "V"), "has": ("HH", "AE", "Z"),
    "had": ("HH", "AE", "D"), "do": ("D", "UW"), "does": ("D", "AH", "Z"),
    "did": ("D", "IH", "D"), "will": ("W", "IH", "L"),
    "would": ("W", "UH", "D"), "can": ("K", "AE", "N"),
    "could": ("K", "UH", "D"), "should": ("SH", "UH", "D"),
    "i": ("AY",), "you": ("Y", "UW"), "he": ("HH", "IY"),
    "she": ("SH", "IY"), "we": ("W", "IY"), "they": ("DH", "EY"),
    "it": ("IH", "T"), "this": ("DH", "IH", "S"), "that": ("DH", "AE", "T"),
    "these": ("DH", "IY", "Z"), "those": ("DH", "OW", "Z"),
    "my": ("M", "AY"), "your": ("Y", "AO", "R"), "his": ("HH", "IH", "Z"),
    "her": ("HH", "ER"), "our": ("AW", "ER"), "their": ("DH", "EH", "R"),
    "me": ("M", "IY"), "him": ("HH", "IH", "M"), "us": ("AH", "S"),
    "them": ("DH", "EH", "M"), "what": ("W", "AH", "T"),
    "when": ("W", "EH", "N"), "where": ("W", "EH", "R"),
    "who": ("HH", "UW"), "why": ("W", "AY"), "how": ("HH", "AW"),
    "not": ("N", "AA", "T"), "no": ("N", "OW"), "yes": ("Y", "EH", "S"),
    "hello": ("HH", "EH", "L", "OW"), "hi": ("HH", "AY"),
    "hey": ("HH", "EY"), "thanks": ("TH", "AE", "NG", "K", "S"),
    "thank": ("TH", "AE", "NG", "K"), "please": ("P", "L", "IY", "Z"),
    "good": ("G", "UH", "D"), "bad": ("B", "AE", "D"),
    "new": ("N", "UW"), "old": ("OW", "L", "D"), "big": ("B", "IH", "G"),
    "small": ("S", "M", "AO", "L"), "great": ("G", "R", "EY", "T"),
    "love": ("L", "AH", "V"), "like": ("L", "AY", "K"),
    "want": ("W", "AA", "N", "T"), "need": ("N", "IY", "D"),
    "know": ("N", "OW"), "think": ("TH", "IH", "NG", "K"),
    "see": ("S", "IY"), "look": ("L", "UH", "K"), "come": ("K", "AH", "M"),
    "go": ("G", "OW"), "going": ("G", "OW", "IH", "NG"),
    "make": ("M", "EY", "K"), "take": ("T", "EY", "K"),
    "give": ("G", "IH", "V"), "get": ("G", "EH", "T"),
    "got": ("G", "AA", "T"), "say": ("S", "EY"), "said": ("S", "EH", "D"),
    "tell": ("T", "EH", "L"), "talk": ("T", "AO", "K"),
    "speak": ("S", "P", "IY", "K"), "listen": ("L", "IH", "S", "AH", "N"),
    "watch": ("W", "AA", "CH"), "play": ("P", "L", "EY"),
    "work": ("W", "ER", "K"), "time": ("T", "AY", "M"),
    "day": ("D", "EY"), "night": ("N", "AY", "T"), "today": ("T", "UW", "D", "EY"),
    "now": ("N", "AW"), "here": ("HH", "IY", "R"), "there": ("DH", "EH", "R"),
    "very": ("V", "EH", "R", "IY"), "really": ("R", "IY", "L", "IY"),
    "just": ("JH", "AH", "S", "T"), "also": ("AO", "L", "S", "OW"),
    "well": ("W", "EH", "L"), "so": ("S", "OW"), "too": ("T", "UW"),
    "people": ("P", "IY", "P", "AH", "L"), "man": ("M", "AE", "N"),
    "woman": ("W", "UH", "M", "AH", "N"), "child": ("CH", "AY", "L", "D"),
    "world": ("W", "ER", "L", "D"), "life": ("L", "AY", "F"),
    "house": ("HH", "AW", "S"), "home": ("HH", "OW", "M"),
    "water": ("W", "AO", "T", "ER"), "food": ("F", "UW", "D"),
    "money": ("M", "AH", "N", "IY"), "car": ("K", "AA", "R"),
    "phone": ("F", "OW", "N"), "video": ("V", "IH", "D", "IY", "OW"),
    "music": ("M", "Y", "UW", "Z", "IH", "K"), "song": ("S", "AO", "NG"),
    "dance": ("D", "AE", "N", "S"), "party": ("P", "AA", "R", "T", "IY"),
}

_G2P_DIGRAPHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("sh", ("SH",)), ("ch", ("CH",)), ("th", ("TH",)), ("ph", ("F",)),
    ("wh", ("W",)), ("ng", ("NG",)), ("ck", ("K",)), ("dge", ("JH",)),
    ("tch", ("CH",)), ("ee", ("IY",)), ("oo", ("UW",)), ("ea", ("IY",)),
    ("ai", ("EY",)), ("ay", ("EY",)), ("ie", ("IY",)), ("oa", ("OW",)),
    ("ow", ("AW",)), ("ou", ("AW",)), ("oi", ("OY",)), ("oy", ("OY",)),
    ("au", ("AO",)), ("aw", ("AO",)), ("ew", ("UW",)), ("qu", ("K", "W")),
)

_G2P_VOWELS: dict[str, tuple[str, ...]] = {
    "a": ("AE",), "e": ("EH",), "i": ("IH",), "o": ("AA",), "u": ("AH",),
    "y": ("IY",),
}


def _g2p_rules(word: str) -> list[str]:
    """Rule-based fallback: digraphs, magic-e, c/g softening."""
    w = word.lower()
    out: list[str] = []
    i = 0
    magic_e = w.endswith("e") and len(w) > 2
    core = w[:-1] if magic_e else w
    while i < len(core):
        matched = False
        for dg, ph in _G2P_DIGRAPHS:
            if core.startswith(dg, i):
                out.extend(ph)
                i += len(dg)
                matched = True
                break
        if matched:
            continue
        ch = core[i]
        nxt = core[i + 1] if i + 1 < len(core) else ""
        if ch == "c":
            out.append("S" if nxt in "eiy" else "K")
        elif ch == "g":
            out.append("JH" if nxt in "eiy" else "G")
        elif ch == "x":
            out.extend(("K", "S"))
        elif ch in _G2P_VOWELS:
            v = _G2P_VOWELS[ch]
            if magic_e and ch == "a":
                v = ("EY",)
            elif magic_e and ch == "i":
                v = ("AY",)
            elif magic_e and ch == "o":
                v = ("OW",)
            out.extend(v)
        elif ch == "r":
            # r colors the previous vowel; keep simple: append R
            out.append("R")
        elif ch.isalpha():
            out.append({"b": "B", "d": "D", "f": "F", "h": "HH",
                        "j": "JH", "k": "K", "l": "L", "m": "M",
                        "n": "N", "p": "P", "s": "S", "t": "T",
                        "v": "V", "w": "W", "z": "Z"}.get(ch, "HH"))
        i += 1
    # collapse doubles
    return [p for j, p in enumerate(out) if j == 0 or p != out[j - 1]]


def text_to_phonemes(text: str) -> list[str]:
    """English text -> ARPAbet phoneme list.

    Dictionary for common words, rule-based fallback otherwise. Word
    boundaries emit nothing (timing is uniform downstream); sentence
    punctuation emits a short silence.
    """
    phonemes: list[str] = []
    for tok in (text or "").lower().split():
        word = "".join(c for c in tok if c.isalpha() or c == "'")
        if not word:
            continue
        seq = _G2P_WORDS.get(word)
        phonemes.extend(seq if seq else _g2p_rules(word))
        if tok and tok[-1] in ".!?":
            phonemes.append("sil")
    return phonemes


def phonemes_to_visemes(phonemes: list[str]) -> list[str]:
    """Phoneme list -> viseme name list."""
    return [phoneme_to_viseme(p) if p != "sil" else "sil" for p in phonemes]


def viseme_track(n_frames: int, phonemes: list[str], duration_s: float,
                 fps: float) -> list[tuple[float, float, float]]:
    """Per-frame mouth shapes from a phoneme list.

    Uniform timing (each phoneme gets an equal slice — honest, NOT
    forced alignment) with a 3-frame moving-average smooth so shapes
    flow into each other instead of snapping.
    """
    shapes = [mouth_shape_for_viseme(v)
              for v in phonemes_to_visemes(phonemes)]
    if not shapes:
        return [VISEME_SHAPES["sil"]] * n_frames
    per = max(1e-6, duration_s / len(shapes))
    raw = []
    for i in range(n_frames):
        t = i / max(1e-6, fps)
        idx = min(len(shapes) - 1, int(t / per))
        raw.append(shapes[idx])
    # coarticulation smoothing: moving average over shapes
    sm = []
    for i in range(n_frames):
        lo, hi = max(0, i - 1), min(n_frames, i + 2)
        win = raw[lo:hi]
        sm.append(tuple(sum(s[j] for s in win) / len(win) for j in range(3)))
    return sm


def acoustic_viseme(openness: float, centroid: float) -> str:
    """Heuristic viseme from audio features when no text is available.

    openness: 0..1 energy envelope; centroid: 0..1 spectral centroid.
    A rough guess — bright spectra read as front vowels, dark as
    rounded back vowels, high energy as open vowels. NOT phoneme
    recognition; labeled as such wherever it is used.
    """
    if openness < 0.12:
        return "sil"
    if openness > 0.62:
        return "AH" if centroid < 0.5 else "AE"
    if centroid > 0.62:
        return "IY" if openness < 0.40 else "EH"
    if centroid < 0.30:
        return "OW" if openness > 0.35 else "UW"
    return "EH" if openness < 0.45 else "AH"


# ── CPU envelope warp ────────────────────────────────────────────────
def audio_features(audio: str, fps: float,
                   n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-video-frame (mouth openness, spectral centroid) from audio.

    openness: 0..1 RMS-energy envelope (fast attack, slow release,
    silence-gated) — REAL audio-driven signal. centroid: 0..1 spectral
    centroid (brightness) for the acoustic-viseme heuristic.
    """
    ff = _ffmpeg()
    if not ff:
        raise RuntimeError("ffmpeg not found")
    wd = Path(tempfile.mkdtemp(prefix="env_"))
    wav = str(wd / "a.wav")
    subprocess.run(
        [ff, "-hide_banner", "-loglevel", "error", "-y", "-i", audio,
         "-ac", "1", "-ar", "16000", "-f", "f32le", wav],
        check=True, capture_output=True, timeout=300)
    raw = np.fromfile(wav, dtype=np.float32)
    sr = 16000
    win = max(1, int(sr / fps))
    n = min(n_frames, len(raw) // win)
    rms = np.zeros(n_frames)
    cent = np.zeros(n_frames)
    freqs = np.fft.rfftfreq(win, 1.0 / sr) / (sr / 2)
    hann = np.hanning(win)
    for i in range(n):
        seg = raw[i * win:(i + 1) * win]
        rms[i] = np.sqrt(np.mean(seg ** 2) + 1e-9)
        mag = np.abs(np.fft.rfft(seg * hann)) + 1e-9
        cent[i] = float(np.sum(freqs * mag) / np.sum(mag))
    # normalize energy by 95th percentile (robust to spikes)
    p95 = np.percentile(rms, 95) + 1e-9
    norm = np.clip(rms / p95, 0, 1)
    # envelope follower: fast attack, slow release
    env = np.zeros_like(norm)
    for i, v in enumerate(norm):
        prev = env[i - 1] if i else 0.0
        env[i] = v if v > prev else prev * 0.82 + v * 0.18
    # silence gate: below 8% energy -> closed
    env[env < 0.08] = 0.0
    return np.clip(env, 0, 1), np.clip(cent, 0, 1)


def audio_envelope(audio: str, fps: float, n_frames: int) -> np.ndarray:
    """Per-video-frame mouth openness 0..1 from audio energy.

    Kept for backwards compatibility; see audio_features() for the
    (openness, centroid) pair the viseme warp uses.
    """
    return audio_features(audio, fps, n_frames)[0]


def _warp_mouth(frame_arr: np.ndarray,
                box: tuple[int, int, int, int],
                shape: tuple[float, float, float]) -> np.ndarray:
    """Reshape the mouth region to a viseme mouth shape.

    shape: (jaw_open, lip_width, lip_round). Jaw drops the region
    downward, width spreads/squeezes it horizontally, rounding puckers
    it (narrower + slight vertical pinch). Anchored at the top of the
    mouth so the upper lip stays put — the jaw does the moving.
    """
    x0, y0, x1, y1 = box
    jaw_open, lip_width, lip_round = shape
    H, W = frame_arr.shape[:2]
    x0, x1 = max(0, x0), min(W, x1)
    y0, y1 = max(0, y0), min(H, y1)
    if x1 <= x0 or y1 <= y0:
        return frame_arr
    region = frame_arr[y0:y1, x0:x1]
    rh, rw = region.shape[:2]
    new_w = max(1, int(rw * lip_width * (1.0 - 0.25 * lip_round)))
    new_h = max(1, int(rh * (1.0 + 1.1 * jaw_open)
                       * (1.0 - 0.15 * lip_round)))
    if new_w == rw and new_h == rh:
        return frame_arr
    warped = np.array(
        Image.fromarray(region).resize((new_w, new_h), Image.BICUBIC))
    canvas = frame_arr.copy()
    cx = (x0 + x1) // 2
    px0 = cx - new_w // 2
    ye = min(y0 + new_h, H)
    xs0, xs1 = max(0, px0), min(W, px0 + new_w)
    if xs1 > xs0 and ye > y0:
        canvas[y0:ye, xs0:xs1] = warped[:ye - y0, xs0 - px0:xs0 - px0 + (xs1 - xs0)]
    return canvas


def envelope_warp_sync(video: str, audio: str,
                       face_box: tuple[float, float, float, float],
                       *, out_path: str | None = None,
                       workdir: str | None = None,
                       fps: int = 24,
                       phonemes: list[str] | None = None,
                       text: str | None = None) -> LipSyncResult:
    """CPU lip sync: audio -> viseme mouth shapes -> warp.

    face_box: (x0, y0, x1, y1) normalized. The mouth region (lower
    third of the face box) is reshaped per frame.

    Drive, best first:
    1. ``phonemes`` (ARPAbet) or ``text`` (compact G2P): real viseme
       shapes with uniform timing — honest, not forced alignment.
    2. neither: acoustic heuristic (energy + spectral centroid) picks a
       plausible viseme per frame — labeled as heuristic.

    Honest label: viseme warp, not neural.
    """
    from .camera import read_frames, write_frames
    frames, _ = read_frames(video, fps=fps)
    if not frames:
        raise RuntimeError("no frames extracted")
    if text and not phonemes:
        phonemes = text_to_phonemes(text)
    openness, centroid = audio_features(audio, float(fps), len(frames))
    if phonemes:
        shapes = viseme_track(len(frames), phonemes,
                              len(frames) / float(fps), float(fps))
        drive = f"text-driven visemes ({len(phonemes)} phonemes, uniform timing)"
    else:
        shapes = [mouth_shape_for_viseme(acoustic_viseme(op, ce))
                  for op, ce in zip(openness, centroid)]
        drive = "acoustic-heuristic visemes (energy + spectral centroid)"
    W, H = frames[0].size
    fx0, fy0, fx1, fy1 = face_box
    x0, y0, x1, y1 = int(fx0 * W), int(fy0 * H), int(fx1 * W), int(fy1 * H)
    # mouth region: lower third of face box
    my0 = y0 + int((y1 - y0) * 0.55)
    my1 = y1
    out_frames = []
    for img, op, shape in zip(frames, openness, shapes):
        if op < 0.02:
            out_frames.append(img)
            continue
        a = np.array(img)
        out_frames.append(Image.fromarray(
            _warp_mouth(a, (x0, my0, x1, my1), shape)))
    out = out_path or str(
        Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="ew_"))
        / "warp_sync.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    write_frames(out_frames, out, fps=fps, audio_src=audio)
    return LipSyncResult(
        path=out, backend="warp",
        note=("CPU viseme warp (" + drive + "): mouth shapes follow the "
              "audio. Honest 2D warp — for neural quality use Wav2Lip / "
              "LatentSync on a GPU machine."))


# ── pose-derived face box (directed animation integration) ───────────
def face_box_from_pose(track) -> tuple[float, float, float, float]:
    """Estimate a face box from a pose track's head keypoints.

    Uses nose/eyes/ears (indices 0, 14-17). Returns normalized
    (x0, y0, x1, y1). This is how lip sync plugs into directed
    animation: we KNOW where the face is from the pose track.
    """
    kp = track.frames[len(track.frames) // 2]  # mid-action frame
    head = kp[[0, 14, 15, 16, 17]]
    x0, y0 = head.min(axis=0)
    x1, y1 = head.max(axis=0)
    # pad: face is bigger than the keypoint spread
    padx, pady = (x1 - x0) * 0.9 + 0.02, (y1 - y0) * 1.6 + 0.03
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 - 0.01
    return (max(0.0, cx - padx), max(0.0, cy - pady),
            min(1.0, cx + padx), min(1.0, cy + pady))


# ── router ───────────────────────────────────────────────────────────
def lip_sync(video: str, audio: str, *,
             face_box: tuple[float, float, float, float] | None = None,
             out_path: str | None = None,
             workdir: str | None = None,
             prefer: str = "auto",
             fps: int = 24,
             text: str | None = None,
             phonemes: list[str] | None = None) -> LipSyncResult:
    """Best-available lip sync. Never fakes it.

    prefer: "auto" | "wav2lip" | "latentsync" | "warp".
    Neural choices raise ModelUnavailable (with install instructions)
    instead of silently degrading.
    ``text``/``phonemes`` drive viseme shapes in the CPU warp backend.
    """
    order = {"auto": ["latentsync", "wav2lip", "warp"],
             "wav2lip": ["wav2lip"], "latentsync": ["latentsync"],
             "warp": ["warp"]}.get(prefer, ["latentsync", "wav2lip", "warp"])
    last_err = ""
    for b in order:
        try:
            if b == "latentsync":
                return latentsync_sync(video, audio, out_path=out_path,
                                       workdir=workdir)
            if b == "wav2lip":
                return wav2lip_sync(video, audio, out_path=out_path,
                                    workdir=workdir)
            if b == "warp":
                if face_box is None:
                    raise ModelUnavailable(
                        "warp lip sync needs a face_box (x0,y0,x1,y1 "
                        "normalized) — no face detector on this machine. "
                        "Pass the face region explicitly.")
                return envelope_warp_sync(video, audio, face_box,
                                          out_path=out_path,
                                          workdir=workdir, fps=fps,
                                          text=text, phonemes=phonemes)
        except ModelUnavailable as exc:
            last_err = str(exc)
            continue
    raise ModelUnavailable(
        "no lip-sync backend available. " + last_err)


# ── dubbing ──────────────────────────────────────────────────────────
def dub_video(video: str, translated_text: str, voice_name: str, *,
              face_box: tuple[float, float, float, float] | None = None,
              out_path: str | None = None,
              workdir: str | None = None,
              prefer: str = "auto", fps: int = 24) -> LipSyncResult:
    """Dub: translated text -> TTS in a catalogue voice -> lip sync -> mux.

    translated_text: the translation (the brain translates via LLM; no
    offline translator is wired — passing untranslated text dubs it
    as-is, honestly).
    """
    from ...voice.catalogue import default_catalogue
    wd = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="dub_"))
    wd.mkdir(parents=True, exist_ok=True)
    cat = default_catalogue()
    tts_out = str(wd / "dub_audio.wav")
    res = cat.speak_as(translated_text, voice_name, out_path=tts_out)
    audio_path = res.get("path") or tts_out
    if not os.path.exists(audio_path):
        raise RuntimeError(f"TTS produced no audio: {res}")
    synced = lip_sync(video, audio_path, face_box=face_box,
                      workdir=str(wd), prefer=prefer, fps=fps,
                      text=translated_text)
    out = out_path or str(wd / "dubbed.mp4")
    if synced.path != out:
        import shutil
        shutil.move(synced.path, out)
    synced.path = out
    synced.note += f" | dubbed with voice '{voice_name}'"
    return synced
