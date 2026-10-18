"""Console entrypoint for $PROJECT_NAME.  No credentials needed."""

from __future__ import annotations

import argparse

from bot import Bot, ConsoleRunner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="$PROJECT_NAME",
        description="$PROJECT_NAME -- a chat bot with a console runner.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.parse_args(argv)  # --help / --version handled here; anything else runs the console
    return ConsoleRunner(Bot()).run()


if __name__ == "__main__":
    raise SystemExit(main())
