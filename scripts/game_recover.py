#!/usr/bin/env python3
"""Phone-side game-profile recovery for Devon (nomorals-2.0).

The stable-ID identity change keys profiles by ``telegram:<numeric_id>``
instead of the display name.  Legacy display-name-keyed profiles
(``telegram:Mary``, ``telegram:Vrede peace``) are folded in lazily on
the next sighting under each name — but if you want it done NOW, or a
profile was orphaned under a name you no longer use, this tool does it
explicitly.

    cd ~/nomorals-2.0
    python3 scripts/game_recover.py list
    python3 scripts/game_recover.py merge --into telegram:5478650254 \\
        --name "Vrede peace" --name "Mary"

Merge rules (same as the in-bot merge): xp keeps the HIGHER value,
coins/points/wins are summed, gear instances are all preserved,
skills union by slug, attributes take the max.  Nothing is deleted
except the folded legacy rows.

Stop the bot before running ``merge`` (``list`` is read-only and safe
any time).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def find_db() -> Path:
    candidates = []
    env = os.environ.get("NM_DB_PATH", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.home() / ".nomorals" / "data" / "nomorals.db")
    candidates.append(REPO_ROOT / "data" / "nomorals.db")
    for c in candidates:
        if c.is_file():
            return c
    raise SystemExit(
        "no game database found; tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nSet NM_DB_PATH to the right file."
    )


def open_store(db_path: Path):
    from nomorals.storage.db import Database
    from nomorals.games.players import PlayerStore

    db = Database(str(db_path))
    db.migrate()
    return PlayerStore(db), db


def cmd_list(args: argparse.Namespace) -> int:
    db_path = Path(args.db) if args.db else find_db()
    store, _db = open_store(db_path)
    db = store.db
    try:
        rows = db.query(
            "SELECT player_key, display, xp, coins, points, wins, losses, "
            "draws, games_played FROM game_players ORDER BY xp DESC"
        ) or []
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"could not read game_players: {exc}")
    print(f"db: {db_path}")
    print(f"{len(rows)} profile(s):")
    for r in rows:
        key = r["player_key"]
        try:
            gear = db.query(
                "SELECT COUNT(*) AS n FROM game_gear WHERE player_key = ?",
                (key,)) or []
            skills = db.query(
                "SELECT COUNT(*) AS n FROM game_skills WHERE player_key = ?",
                (key,)) or []
            gear_n = gear[0]["n"] if gear else 0
            skill_n = skills[0]["n"] if skills else 0
        except Exception:  # noqa: BLE001
            gear_n = skill_n = "?"
        print(
            f"  {key}  display={r['display']!r}  xp={r['xp']}  "
            f"coins={r['coins']}  pts={r['points']}  "
            f"W{r['wins']}/L{r['losses']}/D{r['draws']}  "
            f"gear={gear_n}  skills={skill_n}"
        )
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    db_path = Path(args.db) if args.db else find_db()
    store, _db = open_store(db_path)
    result = store.merge_legacy_names(args.into, args.name)
    merged = result["merged"]
    print(f"db: {db_path}")
    if merged:
        print(f"merged into {args.into}:")
        for k in merged:
            print(f"  {k}")
    else:
        print(f"nothing to merge into {args.into} "
              f"(no matching legacy rows).")
    prof = store.get(args.into)
    print(f"result: xp={prof.xp} coins={prof.coins} "
          f"display={prof.name!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="",
                    help="path to nomorals.db (default: auto-detect)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list all game profiles")
    m = sub.add_parser("merge", help="fold legacy name profiles into an ID key")
    m.add_argument("--into", required=True,
                   help="ID key, e.g. telegram:5478650254")
    m.add_argument("--name", action="append", default=[],
                   help="legacy display name to fold in (repeatable)")
    args = ap.parse_args(argv)
    if args.cmd == "list":
        return cmd_list(args)
    if args.cmd == "merge":
        if not args.name:
            raise SystemExit("merge needs at least one --name")
        return cmd_merge(args)
    raise SystemExit("unknown command")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
