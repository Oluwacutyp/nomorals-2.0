"""MoneyPrinterTurbo factory — topic in, finished narrated short out.

The one-call front door over the existing :class:`ShortPipeline`:
the brain writes the script, Edge TTS (free, no key) speaks it,
stock footage fills the scenes when Pexels/Pixabay keys are set,
Whisper re-times the captions against the real voiceover when
faster-whisper is installed. Publishing is NOT the factory's job —
the publishers are L5 and the factory is L4 — the caller posts the
returned path through ``agents.shorts_publisher.publish_video``,
the factory's own publisher organ. Every rung reports honestly — a
rung that could not run says so in ``rungs`` / ``notes`` and the
short still ships with the rungs that did.

This is the MoneyPrinterTurbo shape, ported fresh into Devon's own
organs: no parallel video system, no LLM API key (the brain writes
the script), no paid video-gen backend, no vendored fonts, no
YouTube-sourced music (the pipeline's composed bed + DJ material).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ...core.logging_setup import get_logger
from .pipeline import ShortPipeline
from .stock import stock_status

_log = get_logger(__name__)

PUBLISH_PLATFORMS = ("tiktok", "youtube", "x")


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _rungs_from_run(run_dir: Path) -> dict[str, Any]:
    """Name every rung from the run's own artifacts (never assumed)."""
    vo = _read_json(run_dir / "voiceover_meta.json")
    frames = _read_json(run_dir / "frames.json")
    captions = _read_json(run_dir / "captions_meta.json")
    music = _read_json(run_dir / "music_meta.json")
    rungs: dict[str, Any] = {
        "script": "brain" if (run_dir / "script.json").exists() else "missing",
        "voiceover": vo.get("voice", "missing"),
        "visuals": "missing",
        "captions": captions.get("timings_source", "missing"),
        "music": music.get("source", "composed"),
    }
    if isinstance(frames, list) and frames:
        sources = [f.get("source", "stills") for f in frames]
        n_stock = sum(1 for s in sources if s == "stock")
        if n_stock == len(sources):
            rungs["visuals"] = "stock"
        elif n_stock == 0:
            rungs["visuals"] = "stills"
        else:
            rungs["visuals"] = (f"mixed ({n_stock} stock / "
                                f"{len(sources) - n_stock} stills)")
    return rungs


def factory_status() -> dict[str, Any]:
    """What the factory's optional rungs can do right now (honest)."""
    from ...tools.audio import detect_tts_engines

    try:
        from faster_whisper import WhisperModel  # noqa: F401
        whisper = True
    except ImportError:
        whisper = False
    stock = stock_status()
    return {
        "edge_tts": "edge_tts" in detect_tts_engines(),
        "whisper": whisper,
        "stock": stock,
        "publish_platforms": list(PUBLISH_PLATFORMS),
        "notes": [
            "edge_tts: free, no key — falls back to the catalogue voice",
            "whisper: faster-whisper when installed, else estimated timings",
            ("stock: " + ("configured: " + ", ".join(stock["configured"]))
             if stock["configured"]
             else "stock: no Pexels/Pixabay key — generated stills"),
            "publish: official APIs, OAuth-configured only",
        ],
    }


def retime_captions_with_whisper(run_dir: str | Path, *,
                                transcriber: Any = None,
                                edit: Any = None) -> dict[str, Any]:
    """Re-time a run's captions against the real voiceover audio.

    faster-whisper measures the spoken words; when the alignment is
    trustworthy the SRT is rebuilt from measured words and re-burned
    onto the draft, then the final is re-muxed. When faster-whisper
    is absent or the alignment is junk, the pipeline's captions stand
    and the result says so — the short is never blocked on this step.
    """
    run_dir = Path(run_dir)
    script_p = run_dir / "script.json"
    vo_p = run_dir / "voiceover_meta.json"
    if not script_p.exists() or not vo_p.exists():
        return {"ok": False,
                "reason": "script/voiceover artifacts missing"}
    script = _read_json(script_p)
    vo = _read_json(vo_p)
    scenes = script.get("scenes") or []
    text = vo.get("text", "")
    total = float(vo.get("duration_s") or 0)
    if not scenes or not text or total <= 0:
        return {"ok": False, "reason": "no narration to re-time"}
    draft = run_dir / "draft.mp4"
    mixed = run_dir / "mixed_audio.wav"
    if not draft.exists() or not mixed.exists():
        return {"ok": False,
                "reason": "draft/mix missing — run did not complete"}
    # Word-count-proportional windows per segment (same weighting as
    # the edit stage's scene bounds). The hook is segment 0 when
    # present — it is spoken first, so it needs its own window or the
    # re-timed captions would miss it.
    texts = [t for t in ([script.get("hook", "")]
                         + [s.get("say", "") for s in scenes]) if t.strip()]
    weights = [max(len(t.split()), 1) for t in texts]
    wsum = sum(weights)
    bounds = [0.0]
    acc = 0.0
    for w in weights:
        acc += total * w / wsum
        bounds.append(round(acc, 3))
    bounds[-1] = round(total, 3)
    segments = [{"text": txt, "spoken_start": bounds[i],
                 "spoken_end": bounds[i + 1]}
                for i, txt in enumerate(texts)]
    narration = {"path": vo.get("path", str(run_dir / "voiceover.wav")),
                 "duration": total, "segments": segments}
    from ...media.videogen.captions import caption_narration

    cap = caption_narration(narration, transcriber=transcriber)
    if cap.get("status") not in ("measured", "mixed"):
        return {"ok": False, "status": cap.get("status", "?"),
                "reason": f"whisper {cap.get('status')}: "
                          f"{cap.get('note', '')}"}
    words = cap.get("words") or []
    if not words:
        return {"ok": False, "reason": "whisper measured no words"}
    from .pipeline import CAPTION_BURN_STYLE, _ffmpeg, phrase_cues

    wt = [(str(w.get("word", "")), float(w.get("start", 0)),
           float(w.get("end", 0))) for w in words]
    cues = phrase_cues(wt)
    if not cues:
        return {"ok": False, "reason": "no caption cues from whisper words"}
    if edit is None:
        from .pipeline import _load_edit_contract
        edit = _load_edit_contract()
    srt_path = str(run_dir / "captions.srt")
    build = getattr(edit, "build_captions", None)
    if build is None:
        from .edit import build_captions as build
    build(cues, srt_path)
    (run_dir / "captions_meta.json").write_text(json.dumps(
        {"cues": len(cues), "timings_source": "whisper",
         "words_measured": len(words)}, indent=2), encoding="utf-8")
    captioned = str(run_dir / "captioned.mp4")
    srt_esc = srt_path.replace("'", r"'\''")
    try:
        _ffmpeg("-i", str(draft), "-vf",
                f"subtitles='{srt_esc}':force_style='{CAPTION_BURN_STYLE}'",
                "-c:a", "copy", captioned)
        final = str(run_dir / "final.mp4")
        _ffmpeg("-i", captioned, "-i", str(mixed),
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                "-shortest", "-movflags", "+faststart", final)
    except RuntimeError as exc:
        return {"ok": False, "reason": f"whisper re-burn/mux failed: {exc}",
                "status": "whisper", "words_measured": len(words)}
    return {"ok": True, "status": "whisper",
            "words_measured": len(words), "cues": len(cues)}


def make_short(topic: str, *, niche: str = "did_you_know", style: str = "",
               footage: str = "auto", voice: str = "edge",
               runs_root: str | os.PathLike | None = None,
               pipeline: ShortPipeline | None = None,
               transcriber: Any = None) -> dict[str, Any]:
    """Topic in, finished narrated vertical short out.

    Plans + runs a :class:`ShortPipeline` job with the factory
    defaults (Edge TTS voiceover, stock footage when configured)
    and re-times the captions with Whisper when it can. Returns
    ``{"ok", "path", "run_id", "job_id", "rungs", "notes"}`` — every
    rung named honestly. Publishing is NOT the factory's job (the
    publishers are L5; the factory is L4) — the caller posts the
    returned ``path`` through :func:`agents.shorts_publisher.
    publish_video`, which is the factory's own publisher organ.
    """
    topic = (topic or "").strip()
    if not topic:
        return {"ok": False, "error": "a topic is required",
                "rungs": {}, "notes": []}
    pipe = pipeline or ShortPipeline(runs_root=runs_root or "")
    job = pipe.plan(niche, topic, style=style, footage=footage,
                    voice=voice)
    result = pipe.run(job)
    run_dir = pipe.runs_root / result.run_id
    rungs = _rungs_from_run(run_dir)
    notes: list[str] = []
    if not result.ok:
        err = result.error.to_dict() if result.error else {}
        return {"ok": False, "run_id": result.run_id, "job_id": job.id,
                "error": err, "rungs": rungs,
                "notes": [f"failed at {err.get('stage', '?')}: "
                          f"{err.get('message', '')}"]}
    if rungs.get("visuals") == "stills" and footage in ("stock", "auto"):
        notes.append("stock footage unavailable — generated stills used "
                     "(set NM_PEXELS_API_KEY / NM_PIXABAY_API_KEY)")
    if rungs.get("voiceover") != "edge-tts" and voice == "edge":
        notes.append("edge-tts unavailable — catalogue voice used")
    retime = retime_captions_with_whisper(run_dir, transcriber=transcriber)
    if retime.get("ok"):
        rungs["captions"] = "whisper"
        notes.append(f"captions re-timed with Whisper "
                     f"({retime['words_measured']} words measured)")
    else:
        notes.append("captions kept the pipeline timings "
                     f"({rungs.get('captions', 'estimated')}) — "
                     f"{retime.get('reason', '')}")
    return {"ok": True, "path": str(run_dir / "final.mp4"),
            "run_id": result.run_id, "job_id": job.id,
            "rungs": rungs, "notes": notes}
