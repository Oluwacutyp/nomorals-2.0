"""``nm g2p`` — pronounce English words in Arpabet, from the terminal.

Thin CLI over the English G2P rung: the same lexicon → learned-GRU
→ rules path the ``tonal_g2p`` brain tool answers, from the
terminal. An out-of-lexicon word is predicted by the ported
grapheme-to-phoneme GRU (``nomorals/media/g2p_learned``); the
source of every answer is named, so a dictionary pronunciation is
never confused with a model prediction or a rule guess.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ...media.g2p_en import g2p_en, learned_status, lexicon_stats


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


def _cmd_g2p(args: argparse.Namespace, context: Any) -> int:
    """Route `nm g2p` subcommands."""
    action = args.g2p_action
    if action == "status":
        learned = learned_status()
        stats = lexicon_stats()
        return _out(args, {
            "learned_available": learned["available"],
            "learned_reason": learned["reason"],
            "learned_arrays": learned["arrays"],
            "lexicon_bundled": stats["bundled"],
            "lexicon_user": stats["user"],
            "lexicon_files": stats["user_files"],
        })
    if action == "pronounce":
        words = list(getattr(args, "words", []) or [])
        if not words:
            print("pronounce needs at least one word",
                  file=sys.stderr)
            return 2
        results = []
        for word in words:
            result = g2p_en(word)
            results.append({
                "word": result.word,
                "phonemes": result.phonemes,
                "source": result.source,
                "rung": result.rung,
            })
        if getattr(args, "json", False):
            return _out(args, results)
        for item in results:
            phones = " ".join(item["phonemes"]) or "(unpronounceable)"
            print(f"{item['word']} → {phones} ({item['source']})")
        return 0
    print(f"unknown g2p action {action!r}", file=sys.stderr)
    return 2
