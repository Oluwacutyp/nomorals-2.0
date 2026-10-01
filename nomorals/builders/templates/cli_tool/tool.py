"""$PROJECT_NAME -- an argparse-based CLI tool scaffold.

Usage:
    python run.py --help
    python run.py greet --name Ada
    python run.py greet --name Ada --shout
"""

from __future__ import annotations

import argparse

VERSION = "0.1.0"


def cmd_greet(args: argparse.Namespace) -> int:
    """Greet someone by name."""
    message = f"Hello, {args.name}!"
    if args.shout:
        message = message.upper()
    print(message)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="$PROJECT_NAME",
        description="$PROJECT_NAME -- a small command-line tool.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    greet = sub.add_parser("greet", help="greet someone by name")
    greet.add_argument("--name", default="world", help="who to greet")
    greet.add_argument("--shout", action="store_true", help="use ALL CAPS")
    greet.set_defaults(func=cmd_greet)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
