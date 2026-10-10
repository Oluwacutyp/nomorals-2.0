"""`nm audio` — Devon's own audio toolkit on the command line.

Native-first: enhancement via Devon's own DSP (no ffmpeg needed),
Devon-owned effect chains, native acoustic analysis, and the local
fingerprint recognition memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _cmd_audio(args: argparse.Namespace, context: Any) -> int:
    """Route `nm audio` subcommands."""
    action = args.audio_action
    if action == "enhance":
        return _audio_enhance(args)
    if action == "fx":
        return _audio_fx(args)
    if action == "fx-list":
        return _audio_fx_list(args)
    if action == "analyze":
        return _audio_analyze(args)
    if action == "fingerprint":
        return _audio_fingerprint(args)
    if action == "match":
        return _audio_match(args)
    print(f"unknown audio action: {action}", file=sys.stderr)
    return 2


def _audio_enhance(args: argparse.Namespace) -> int:
    from ...audio.edit import enhance_audio

    res = enhance_audio(args.file, profile=args.profile or "voice",
                        engine=args.engine or "auto")
    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("ok") else 1
    if not res.get("ok"):
        print(f"enhance failed: {res.get('reason')}", file=sys.stderr)
        return 1
    print(f"enhanced [{res.get('engine')}] via {res.get('filter')}")
    print(f"  → {res.get('output')}")
    if res.get("note"):
        print(f"  {res['note']}")
    return 0


def _audio_fx(args: argparse.Namespace) -> int:
    from ...audio.edit import apply_fx

    res = apply_fx(args.file, ", ".join(args.effects),
                   out_dir=args.out_dir or None)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("ok") else 1
    if not res.get("ok"):
        print(f"fx failed: {res.get('reason')}", file=sys.stderr)
        return 1
    print(f"🎛️ {res.get('note')}")
    print(f"  → {res.get('output')}")
    return 0


def _audio_fx_list(args: argparse.Namespace) -> int:
    from ...audio.edit import list_fx

    print(list_fx())
    return 0


def _audio_analyze(args: argparse.Namespace) -> int:
    from ...audio.fingerprint import analyze

    an = analyze(args.file)
    if args.json:
        print(json.dumps(an.to_dict(), indent=2))
        return 0 if an.ok else 1
    if not an.ok:
        print(f"analyze failed: {an.reason}", file=sys.stderr)
        return 1
    print(f"👂 {args.file}")
    print(f"  duration: {an.seconds:.1f}s @ {an.sample_rate} Hz")
    print(f"  loudness: {an.rms_db:.1f} dBFS (peak {an.peak_db:.1f})")
    print(f"  centroid: {an.spectral_centroid_hz:.0f} Hz · "
          f"zcr {an.zero_crossing_rate:.4f}")
    if an.tempo_bpm:
        print(f"  tempo: ~{an.tempo_bpm:.0f} BPM "
              f"(confidence {an.tempo_confidence:.2f})")
    if an.key:
        print(f"  key: {an.key} (confidence {an.key_confidence:.2f})")
    kind = ("music" if an.music_score >= 0.6 else
            "speech" if an.music_score <= 0.4 else "mixed")
    print(f"  content: {kind} (music score {an.music_score:.2f})")
    if an.clipped_ratio > 0.001:
        print("  ⚠️ clipping detected")
    if an.silence_ratio > 0.5:
        print("  ⚠️ mostly silence")
    return 0


def _audio_fingerprint(args: argparse.Namespace) -> int:
    from ...audio.fingerprint import default_fingerprint_db

    db = default_fingerprint_db()
    try:
        res = db.add_track(args.file, title=args.title or "",
                           artist=args.artist or "local",
                           source=args.source or "library")
    finally:
        db.close()
    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("ok") else 1
    if not res.get("ok"):
        print(f"fingerprint failed: {res.get('reason')}", file=sys.stderr)
        return 1
    print(f"🎛️ fingerprinted '{res.get('title')}' — "
          f"{res.get('hashes')} hashes in local memory")
    return 0


def _audio_match(args: argparse.Namespace) -> int:
    from ...audio.fingerprint import match_local_db

    res = match_local_db(args.file)
    if args.json:
        print(json.dumps(res, indent=2, default=str))
        return 0 if res.get("ok") else 1
    if not res.get("ok"):
        print(f"no match: {res.get('reason')}", file=sys.stderr)
        return 1
    print(f"🎵 {res.get('artist')} — {res.get('title')} "
          f"(local match, confidence {res.get('score')})")
    return 0
