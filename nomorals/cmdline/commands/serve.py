"""``nm serve`` — HTTP API server launcher."""

from __future__ import annotations

import argparse
from typing import Any



def _cmd_serve(args: argparse.Namespace, context: Any) -> int:
    from ...api.server import serve

    return serve(
        context,
        host=context.settings.api.host,
        port=context.settings.api.port,
    )
