"""Tiny output helper shared by every ``nm`` command module."""

from __future__ import annotations

import argparse
import json
from typing import Any



def _emit(args: argparse.Namespace, payload: Any, text: str) -> None:
    # bare Namespaces (tests, embedders) may not carry --json at all
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str, ensure_ascii=False))
    else:
        print(text)
