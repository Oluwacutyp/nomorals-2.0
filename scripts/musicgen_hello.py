#!/usr/bin/env python3
"""Music generation hello-world: draft -> bed -> vocal -> RVC -> master.

Usage:
    python scripts/musicgen_hello.py [--fake] [--out DIR] [--real]

* Default (no flags): tries the REAL chain (ACE-Step + DiffSinger +
  RVC if installed). If models are absent it prints exactly what is
  missing and exits 2 — it never fakes a pass.
* --fake: runs the full chain with labelled test doubles (real sine
  WAVs, manifest says "fake"). Proves the wiring end to end without
  models. Exit 0 only when every stage produced output.
* --out DIR: output directory (default: musicgen_out/hello-world).

Setup for the real chain (Tier 0):
    1. Drop a DiffSinger voicebank into ~/.nomorals/svs/
    2. pip install onnxruntime
    3. Install RVC + one voice model (see voice/rvc_bridge.py)
    4. ACE-Step for the bed (see media/ace_step.py INSTALL_HINT)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nomorals.media.musicgen_chain import (  # noqa: E402
    ChainRequest, fake_bed, fake_vocal, run_chain)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fake", action="store_true",
                    help="run with labelled test doubles (no models needed)")
    ap.add_argument("--real", action="store_true",
                    help="force the real chain (same as default)")
    ap.add_argument("--out", default="musicgen_out/hello-world")
    ap.add_argument("--lyrics", default="Mo dé, ayọ dé")
    ap.add_argument("--language", default="yo")
    ap.add_argument("--genre", default="afrobeats")
    args = ap.parse_args()

    req = ChainRequest(title="hello-world", lyrics=args.lyrics,
                       language=args.language, genre=args.genre,
                       workdir=args.out,
                       rvc_model="fake-voice" if args.fake else "")
    if args.fake:
        def fake_rvc(vocal_path: str, model: str, workdir: str) -> str:
            import shutil
            dst = os.path.join(workdir, "vocal-rvc-fake.wav")
            shutil.copyfile(vocal_path, dst)
            return dst

        result = run_chain(req, bed_fn=fake_bed, vocal_fn=fake_vocal,
                           rvc_fn=fake_rvc)
        # fake RVC only runs when an rvc_model is set; force it here
        print("FAKE RUN — test doubles, not real models")
    else:
        result = run_chain(req)

    for stage in result.stages:
        mark = {"real": "OK ", "fake": "FAKE", "skipped": "SKIP"}.get(
            stage.status, "????")
        print(f"[{mark}] {stage.name:8s} {stage.path or stage.detail}")
    print(f"master: {result.master_path or '(none)'}")
    print(f"midi:   {result.midi_path}")
    print(f"spec:   {result.spec_path}")
    print(f"manifest: {result.manifest_path}")
    if result.ok:
        print("HELLO-WORLD: chain complete")
        return 0
    print(f"HELLO-WORLD FAILED: {result.error}")
    print("Tier-0 setup: voicebank in ~/.nomorals/svs/ + "
          "pip install onnxruntime (+ RVC, + ACE-Step for beds).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
