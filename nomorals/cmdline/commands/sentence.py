"""``nm sentence`` — sentence tokenization + segmentation, from the terminal.

Thin CLI over the sentence organ (``nomorals/media/sentence.py``):
the same tokenizer + segmenter the voice stack shares (the
``PerformanceTuner``, sentence-chunked TTS streaming and the live
call publisher all route through it). ``split`` shows the sentences
with offsets, terminator and the ambiguous flag; ``tokens`` shows
the token stream the segmenter decided on, so a surprising split
can be traced to the exact token that caused it.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ...media.sentence import segment, tokenize


def _out(args: argparse.Namespace, obj: Any) -> int:
    if getattr(args, "json", False):
        print(json.dumps(obj, indent=2))
        return 0
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(f"{key}: {value}")
    else:
        print(obj)
    return 0


def _cmd_sentence(args: argparse.Namespace, context: Any) -> int:
    """Route `nm sentence` subcommands."""
    text = " ".join(args.text)
    if args.sentence_action == "split":
        sentences = segment(text)
        if getattr(args, "json", False):
            return _out(args, [
                {
                    "n": n + 1,
                    "text": s.text,
                    "start": s.start,
                    "end": s.end,
                    "terminator": s.terminator.value if s.terminator
                    else None,
                    "ambiguous": s.ambiguous,
                }
                for n, s in enumerate(sentences)
            ])
        lines = [
            f"{n + 1}. [{s.start}:{s.end}] {s.text}"
            + ("  (ambiguous)" if s.ambiguous else "")
            for n, s in enumerate(sentences)
        ]
        return _out(args, "\n".join(lines) or "(no sentences)")
    if args.sentence_action == "tokens":
        tokens = tokenize(text)
        if getattr(args, "json", False):
            return _out(args, [
                {
                    "kind": t.kind,
                    "text": t.text,
                    "start": t.start,
                    "end": t.end,
                    "punct": t.punct.value if t.punct else None,
                }
                for t in tokens
            ])
        lines = [
            f"{t.kind}: {t.text!r} [{t.start}:{t.end}]"
            for t in tokens
        ]
        return _out(args, "\n".join(lines) or "(no tokens)")
    return 2
