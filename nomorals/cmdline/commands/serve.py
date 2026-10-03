"""``nm serve`` — HTTP API server launcher."""

from __future__ import annotations

import argparse
from typing import Any



def _cmd_serve(args: argparse.Namespace, context: Any) -> int:
    from ...api.server import serve

    host = getattr(args, "host", None) or context.settings.api.host
    port = getattr(args, "port", None) or context.settings.api.port
    return serve(
        context,
        host=host,
        port=port,
    )
