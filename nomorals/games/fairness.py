"""Provably-fair RNG for the house games.

The problem: casino games (blackjack, roulette, slots, poker, craps)
draw from the room's RNG stream. The player has to take the house's word
that the deck wasn't stacked mid-game. This module fixes that with a
commit-reveal scheme — the same pattern real provably-fair casinos use,
implemented natively (stdlib only):

* **Commit** — at table open the house generates a random 256-bit server
  seed, publishes ``sha256(server_seed)`` as the *commitment*, and binds
  the player's client seed into every draw. The commitment is shown
  BEFORE any card is dealt.
* **Draw** — every random outcome (shuffles, spins, dice, reels) derives
  from ``HMAC-SHA256(server_seed, client_seed || counter || tag)``. The
  counter increments per draw and the tag names the draw ("deck",
  "spin:3", …), so the full sequence is deterministic given the seeds.
* **Reveal** — when the game ends the house reveals the server seed and
  posts the verification recipe. Anyone can check
  ``sha256(seed) == commitment`` and re-derive every draw.

Honest scope: this proves the house *committed to the full random
sequence before play and can't change it mid-game* — it does not defend
against a house that simply picks a bad seed up front (your bot is not
your adversary; the guarantee that matters is tamper-evidence and
auditability). The client seed is per-player, persistent, and
player-settable (``/game seed <word>``) so the sequence isn't
house-chosen alone.

Games opt in with ``fair = True``; the engine creates the table in
:func:`init_table`, games draw with :func:`draw_int` /
:func:`draw_choice` / :func:`shuffle`, and the engine posts the reveal
block at finish.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from typing import Any, Sequence

_log = logging.getLogger("nomorals.games.fairness")

__all__ = [
    "FAIR_GAMES",
    "commit_line",
    "draw_choice",
    "draw_int",
    "draw_weighted",
    "get_client_seed",
    "init_table",
    "reveal_block",
    "set_client_seed",
    "shuffle",
    "verify",
]

#: games whose randomness is provably fair (opt-in on the game class via
#: ``fair = True``; listed here for /game fair help text).
FAIR_GAMES = ("blackjack", "roulette", "slots", "poker", "craps")


# ── client seeds ───────────────────────────────────────────────────────────

def _ensure_table(db: Any) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS fair_seeds ("
            "player_key TEXT PRIMARY KEY, "
            "seed TEXT NOT NULL, "
            "updated_at REAL NOT NULL DEFAULT 0)")
    except Exception:  # noqa: BLE001
        _log.debug("fair_seeds ensure failed", exc_info=True)


def get_client_seed(db: Any, player_key: str) -> str:
    """This player's persistent client seed, minted on first use."""
    _ensure_table(db)
    try:
        row = db.query_one(
            "SELECT seed FROM fair_seeds WHERE player_key = ?",
            (player_key,))
        if row and row.get("seed"):
            return str(row["seed"])
        seed = secrets.token_hex(6)
        db.execute(
            "INSERT OR REPLACE INTO fair_seeds "
            "(player_key, seed, updated_at) VALUES (?, ?, ?)",
            (player_key, seed, time.time()))
        return seed
    except Exception:  # noqa: BLE001
        _log.debug("fair client seed read failed", exc_info=True)
        return "house"


def set_client_seed(db: Any, player_key: str, seed: str) -> str:
    """Set a player's client seed (their own entropy in every draw)."""
    _ensure_table(db)
    seed = "".join(c for c in (seed or "").strip() if c.isprintable())
    if not seed:
        raise ValueError("give me a word — /game seed <word>")
    if len(seed) > 64:
        seed = seed[:64]
    try:
        db.execute(
            "INSERT OR REPLACE INTO fair_seeds "
            "(player_key, seed, updated_at) VALUES (?, ?, ?)",
            (player_key, seed, time.time()))
    except Exception:  # noqa: BLE001
        _log.debug("fair client seed write failed", exc_info=True)
    return seed


# ── the commit-reveal table ────────────────────────────────────────────────

def init_table(client_seed: str = "") -> dict[str, Any]:
    """Open a fair table: fresh server seed, published commitment.

    Returns the ``state["fair"]`` fragment. The server seed lives in
    state (revealed at finish); the commitment is what the player sees.
    """
    server_seed = secrets.token_bytes(32)
    commitment = hashlib.sha256(server_seed).hexdigest()
    return {
        "commit": commitment,
        "seed": server_seed.hex(),      # revealed at finish
        "client": (client_seed or "house")[:64],
        "counter": 0,
        "revealed": False,
        "draws": 0,
    }


def verify(commitment: str, server_seed_hex: str) -> bool:
    """True iff ``sha256(seed) == commitment``."""
    try:
        raw = bytes.fromhex(server_seed_hex)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(), (commitment or "").lower())


def _hmac_block(server_seed: bytes, client: str, counter: int,
                tag: str, index: int) -> bytes:
    msg = f"{client}:{counter}:{tag}:{index}".encode("utf-8")
    return hmac.new(server_seed, msg, hashlib.sha256).digest()


def _stream(fair: dict[str, Any], tag: str, nbytes: int) -> bytes:
    """Draw ``nbytes`` of HMAC stream for one draw; bumps the counter."""
    seed = bytes.fromhex(fair["seed"])
    client = str(fair.get("client") or "house")
    counter = int(fair.get("counter") or 0)
    out = b""
    i = 0
    while len(out) < nbytes:
        out += _hmac_block(seed, client, counter, tag, i)
        i += 1
    fair["counter"] = counter + 1
    fair["draws"] = int(fair.get("draws") or 0) + 1
    return out[:nbytes]


def _uint32(fair: dict[str, Any], tag: str) -> int:
    return int.from_bytes(_stream(fair, tag, 4), "big")


def draw_int(fair: dict[str, Any], tag: str, lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi] (rejection sampling — no modulo bias)."""
    if hi < lo:
        raise ValueError("hi < lo")
    span = hi - lo + 1
    # rejection sampling over 32-bit words
    limit = (2 ** 32 // span) * span
    while True:
        v = _uint32(fair, tag)
        if v < limit:
            return lo + (v % span)


def draw_choice(fair: dict[str, Any], tag: str,
                seq: Sequence[Any]) -> Any:
    """Uniform pick from a sequence."""
    if not seq:
        raise ValueError("empty sequence")
    return seq[draw_int(fair, tag, 0, len(seq) - 1)]


def draw_weighted(fair: dict[str, Any], tag: str,
                  weights: Sequence[float]) -> int:
    """Weighted index draw (slots reels)."""
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("weights sum to 0")
    # 53 bits of precision from two 32-bit words
    hi = _uint32(fair, tag + ":hi")
    lo = _uint32(fair, tag + ":lo")
    r = ((hi * 2 ** 32 + lo) / 2 ** 64) * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r < acc:
            return i
    return len(weights) - 1


def shuffle(fair: dict[str, Any], tag: str, items: list[Any]) -> list[Any]:
    """Fisher–Yates shuffle driven by the fair stream (in place)."""
    for i in range(len(items) - 1, 0, -1):
        j = draw_int(fair, f"{tag}:{i}", 0, i)
        items[i], items[j] = items[j], items[i]
    return items


# ── player-facing text ─────────────────────────────────────────────────────

def commit_line(fair: dict[str, Any]) -> str:
    commit = str(fair.get("commit") or "")
    return (
        f"🔒 provably fair — commitment `{commit[:12]}…`\n"
        f"   every draw is locked in. the seed is revealed at the end — "
        f"/game fair to inspect.")


def reveal_block(fair: dict[str, Any]) -> str:
    """The end-of-game reveal: seed + verification recipe."""
    fair["revealed"] = True
    commit = str(fair.get("commit") or "")
    seed = str(fair.get("seed") or "")
    client = str(fair.get("client") or "house")
    draws = int(fair.get("draws") or 0)
    ok = "✓" if verify(commit, seed) else "✗"
    return (
        f"🔓 fairness reveal {ok}\n"
        f"   commitment: {commit}\n"
        f"   server seed: {seed}\n"
        f"   your seed: {client} · {draws} draws\n"
        f"   verify: sha256(server seed) == commitment, then re-derive "
        f"each draw as HMAC-SHA256(seed, \"<your seed>:<n>:<tag>:<i>\").")
