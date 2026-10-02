"""``nm music`` — music surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from ..emit import _emit



def _cmd_music(args: argparse.Namespace, context: Any) -> int:
    """Compose songs / list styles through the music_writer tool."""
    tools = context.tools
    action = getattr(args, "action", "styles") or "styles"
    if action == "styles":
        out = tools.call("music_writer", action="styles")
        if not out.ok:
            print(f"music: {out.error}", file=sys.stderr)
            return 1
        styles = out.value.get("styles", {})
        _emit(args, out.value,
              "\n".join(f"  {k:<12} {v.get('label', k)} · {v.get('tempo', '')} "
                         f"bpm · {v.get('mode', '')}" for k, v in styles.items())
              or "no styles")
        return 0
    if action == "songs":
        out = tools.call("music_writer", action="song",
                         topic=getattr(args, "topic", "") or "")
        if not out.ok:
            print(f"music: {out.error}", file=sys.stderr)
            return 1
        _emit(args, out.value, json.dumps(out.value, indent=2, default=str))
        return 0
    topic = getattr(args, "topic", "") or ""
    if not topic:
        print("music compose needs a topic — nm music compose \"about what\"",
              file=sys.stderr)
        return 2
    out = tools.call("music_writer", action="compose", topic=topic,
                     style=getattr(args, "style", "pop") or "pop",
                     title=getattr(args, "title", "") or "",
                     key=getattr(args, "key", "") or "",
                     seed=int(getattr(args, "seed", "0") or 0))
    if not out.ok:
        print(f"music: {out.error}", file=sys.stderr)
        return 1
    song = out.value
    _emit(args, song,
          f"composed: {song.get('title', topic)} [{song.get('style', '')}]\n"
          f"  midi: {song.get('midi_path', '')}\n"
          f"  melody: {str(song.get('melody_description', ''))[:160]}")
    return 0
