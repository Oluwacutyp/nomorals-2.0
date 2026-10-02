"""``nm hub`` — hub surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from ..emit import _emit



def _cmd_hub(args: argparse.Namespace, context: Any) -> int:
    """`nm hub <song|video|podcast|status|styles>` — the MediaHub console.

    Same orchestrator the ``/hub`` chat command uses. Podcast transcripts
    stay local in the CLI (``send_transcript=False``) — the chat path is
    the one that delivers to live channels.
    """
    from ...media import MediaHub

    mode = (getattr(args, "mode", "status") or "status").strip().lower()

    query = (getattr(args, "query", "") or "").strip()
    if mode in ("song", "video", "podcast") and not query:
        print(f"hub {mode} needs a topic/query — nm hub {mode} \"<topic>\"",
              file=sys.stderr)
        return 2

    hub = MediaHub(context)

    if mode == "status":
        st = hub.status()
        _emit(args, st, json.dumps(st, indent=2, default=str))
        return 0

    if mode == "styles":
        styles = hub.styles()
        lines = [f"  {k:<14} {v['label']} (tempo {v['tempo']}, {v['mode']}, energy {v['energy']})"
                 for k, v in sorted(styles.items())]
        _emit(args, {"styles": styles}, "song styles:\n" + "\n".join(lines))
        return 0

    try:
        result = hub.run(
            mode,
            topic=query, query=query,
            style=getattr(args, "style", "pop") or "pop",
            platform=getattr(args, "platform", "") or "",
            play=not getattr(args, "no_play", False),
            send_transcript=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"hub {mode}: {exc}", file=sys.stderr)
        return 1
    if result.get("error"):
        print(f"hub {mode}: {result['error']}", file=sys.stderr)
        return 1
    if mode == "song":
        song = result.get("song") or {}
        text = (f"composed “{song.get('title', query)}” "
                f"({song.get('style')}, {len(song.get('sections', []))} sections) → "
                f"{song.get('midi_path', '')}")
    elif mode == "podcast":
        text = (f"podcast saved: {result.get('transcript_path', '')} "
                f"({len(result.get('chapters', []) or [])} chapters)")
    else:
        pick = result.get("pick") or {}
        text = f"video: {pick.get('title', '')} → {result.get('download', {}).get('path', '')}"
    _emit(args, result, text)
    return 0
