#!/usr/bin/env python3
"""Prepare video-model conditions — precompute (conditions, latents) for fine-tuning.

A runnable port of upstream FineTrainers' precompute step
(``training/prepare_dataset.sh --save_latents_and_embeddings`` +
``Trainer.prepare_precomputations``):

    python training/prepare_conditions.py \
        --data-root tests/fixtures/finetrainers-conditioning \
        --caption-column prompts_multi.txt \
        --video-column videos_multi.txt \
        --model-name ltx_video --precompute-conditions --demo-encoders

That writes ``<data-root>/ltx_video_Lightricks-LTX-Video_precomputed/``
with ``conditions/`` + ``latents/`` pairs and a ``manifest.json`` —
training input for an LTX-Video LoRA run (the model Devon's videogen
organ runs; its LoRAs live under the models root).

``--demo-encoders`` runs the deterministic stand-in encoders (no GPU,
no weights) so the whole path proves end to end on any box. Without
it, the real T5/Llama+CLIP text encoder and VAE load from Hugging Face
— which fails honestly naming the missing package or checkpoint when
torch / diffusers / transformers are absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from the repo root: training/ scripts live beside the
# nomorals/ package, not inside it.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nomorals.training.conditioning.configs import parse_args  # noqa: E402
from nomorals.training.conditioning.precompute import (  # noqa: E402
    run_precompute_from_config,
)


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    demo = "--demo-encoders" in raw
    raw = [arg for arg in raw if arg != "--demo-encoders"]
    config = parse_args(raw)
    if not config.precompute_conditions:
        print(
            "Nothing to do: pass --precompute-conditions to run the "
            "precompute pass.",
            file=sys.stderr,
        )
        return 2
    report = run_precompute_from_config(config, demo=demo)
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
