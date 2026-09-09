"""Console entry point: ``python -m nomorals``."""

from __future__ import annotations

import sys


def main() -> int:
    from .cli import main as _cli_main

    return _cli_main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
