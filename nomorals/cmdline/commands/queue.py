"""``nm queue`` — work queue surfaces."""

from __future__ import annotations

import argparse
import json
from typing import Any
from ..emit import _emit



def _cmd_queue(args: argparse.Namespace, context: Any) -> int:
    from ...storage.queue import WorkQueue

    queue = WorkQueue(context.db)
    payload = {"pending": queue.pending(args.topic or None), "topics": queue.topics()}
    _emit(args, payload, json.dumps(payload, indent=2, default=str))
    return 0
