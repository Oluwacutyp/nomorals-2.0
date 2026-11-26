"""The pump desk's board — what the owner looks at and acts from.

The desk already computes a verdict for every dossier (the watch
cycle writes it) and grades every call against the curve — but no
surface ever read those verdicts BACK as a ranked view.  ``status``
is counts, ``calls`` is the ledger, ``dossier`` is one mint at a
time.  The board closes that gap:

* ``board`` — every dossiered mint ranked by its CURRENT verdict
  (re-evaluated fresh on every read from the stored dossier, so the
  board never shows a stale verdict from the last watch cycle),
  with trader filters (min composite, max rug risk, min momentum,
  candidates-only, watched-only) and watched mints pinned first.
* ``open`` — every open (unscored, unexpired) call with live
  progress: current SOL mcap from the desk's own observations
  ledger, progress toward target, distance to invalidation, time
  to expiry.  No USD anywhere (design §7.3): the ledger's SOL
  curve snapshots are the truth; a call with no curve read since
  entry says so instead of inventing a number.

Both are pure on-demand reads over the store — no network, no new
scheduler job, no Notifier publish.  The stopped PumpVet monitor
stays stopped: nothing here pages, pings, or re-enables anything.
"""

from __future__ import annotations

import time
from typing import Any

from . import chain
from . import learn as _learn
from . import score as _score

__all__ = [
    "parse_board_filters", "valid_mint", "board_rows", "open_rows",
    "render_board", "render_open",
]

#: Default / maximum rows the board shows.
DEFAULT_LIMIT = 10
MAX_LIMIT = 50

#: Verdict rank for board ordering (candidates first).
_VERDICT_RANK = {"CALL_CANDIDATE": 0, "NO_CALL": 1,
                 "INSUFFICIENT_DATA": 2}


def valid_mint(mint: str) -> bool:
    """True when *mint* parses as a 32-byte Solana address."""
    try:
        return len(chain.b58decode((mint or "").strip())) == 32
    except Exception:  # noqa: BLE001 - any decode failure is invalid
        return False


def parse_board_filters(text: str) -> tuple[dict[str, Any], list[str]]:
    """Parse board filter tokens into a filter dict + bad tokens.

    Accepted: ``min=0.6`` / ``min_composite=0.6``,
    ``rug=0.3`` / ``max_rug=0.3``, ``mom=0.45`` /
    ``min_momentum=0.45``, ``candidates`` / ``only_candidates``,
    ``watched`` / ``only_watched``, ``limit=10``.  Bad tokens are
    reported, never fatal — the board still renders with the good
    ones.  Floats must parse and sit in [0, 1]; limit clamps to
    [1, 50].
    """
    filters: dict[str, Any] = {
        "min_composite": 0.0, "max_rug": 1.0, "min_momentum": 0.0,
        "only_candidates": False, "only_watched": False,
        "limit": DEFAULT_LIMIT,
    }
    bad: list[str] = []
    for token in (text or "").split():
        low = token.strip().lower()
        if low in ("candidates", "only_candidates"):
            filters["only_candidates"] = True
            continue
        if low in ("watched", "only_watched"):
            filters["only_watched"] = True
            continue
        if "=" not in low:
            bad.append(token)
            continue
        key, _, raw = low.partition("=")
        try:
            value = float(raw)
        except ValueError:
            bad.append(token)
            continue
        if key in ("min", "min_composite"):
            if 0.0 <= value <= 1.0:
                filters["min_composite"] = value
            else:
                bad.append(token)
        elif key in ("rug", "max_rug"):
            if 0.0 <= value <= 1.0:
                filters["max_rug"] = value
            else:
                bad.append(token)
        elif key in ("mom", "min_momentum"):
            if 0.0 <= value <= 1.0:
                filters["min_momentum"] = value
            else:
                bad.append(token)
        elif key == "limit":
            filters["limit"] = max(1, min(MAX_LIMIT, int(value)))
        else:
            bad.append(token)
    return filters, bad


def _age_str(age_s: float | None) -> str:
    if age_s is None or age_s < 0:
        return "?"
    if age_s < 60:
        return f"{int(age_s)}s"
    if age_s < 3600:
        return f"{int(age_s // 60)}m"
    if age_s < 86400:
        return f"{int(age_s // 3600)}h"
    return f"{int(age_s // 86400)}d"


def board_rows(store: Any, *,
               min_composite: float = 0.0,
               max_rug: float = 1.0,
               min_momentum: float = 0.0,
               only_candidates: bool = False,
               only_watched: bool = False,
               limit: int = DEFAULT_LIMIT,
               now: float | None = None) -> tuple[list[dict[str, Any]], int]:
    """One row per dossiered mint, ranked and filtered.

    Returns (rows, total_dossiered).  The verdict is evaluated
    FRESH from the stored dossier with the desk's current scoring
    parameters — the board shows what the gate says NOW, not what
    the last watch cycle wrote.  Watched mints pin first, then
    CALL_CANDIDATE by composite, then NO_CALL, then
    INSUFFICIENT_DATA.
    """
    at = time.time() if now is None else now
    params = _learn.current_params(store)
    try:
        smart_wallets = store.watched_smart()
    except Exception:  # noqa: BLE001 - a store without watchlists
        smart_wallets = set()
    try:
        watched = store.watched_mints()
    except Exception:  # noqa: BLE001 - older watchlists lack "mints"
        watched = set()
    dossiers = store.dossiers()
    total = len(dossiers)
    rows: list[dict[str, Any]] = []
    for mint, d in dossiers.items():
        verdict = _score.evaluate(d, smart_wallets=smart_wallets,
                                  params=params)
        if verdict.composite < min_composite:
            continue
        if verdict.rug_risk > max_rug:
            continue
        if verdict.momentum < min_momentum:
            continue
        if only_candidates and verdict.verdict != "CALL_CANDIDATE":
            continue
        is_watched = mint in watched
        if only_watched and not is_watched:
            continue
        curve = d.get("curve") or {}
        att = d.get("attention") or {}
        created_at = float(d.get("created_at") or 0)
        rows.append({
            "mint": mint,
            "symbol": d.get("symbol") or "",
            "name": d.get("name") or "",
            "verdict": verdict.verdict,
            "composite": verdict.composite,
            "rug_risk": verdict.rug_risk,
            "manipulation": verdict.manipulation,
            "momentum": verdict.momentum,
            "smart_money": verdict.smart_money,
            "attention": str(att.get("verdict") or ""),
            "dev_serial_rugger": bool((d.get("dev") or {}).get(
                "serial_rugger")),
            "curve_pct": float(curve.get("curve_pct") or 0.0),
            "mcap_sol": float(curve.get("market_cap_sol") or 0.0),
            "watched": is_watched,
            "age": _age_str(at - created_at if created_at else None),
            "evidence": (verdict.evidence or "")[:160],
            "gates": list(verdict.gates),
        })
    rows.sort(key=lambda r: (
        0 if r["watched"] else 1,
        _VERDICT_RANK.get(r["verdict"], 3),
        -r["composite"]))
    return rows[:max(1, min(MAX_LIMIT, int(limit or DEFAULT_LIMIT)))], total


def open_rows(store: Any, calls: list[Any], *,
              now: float | None = None) -> list[dict[str, Any]]:
    """One row per OPEN call (unscored, unexpired) with live progress.

    Current SOL mcap comes from the desk's own observations ledger
    (curve snapshots since the call published) — the same series
    the grader scores first-touch from.  A call with no curve read
    since entry reports ``current_mc_sol`` None and flag
    ``no-read`` instead of inventing a number.  Sorted by time to
    expiry, most urgent first.
    """
    at = time.time() if now is None else now
    rows: list[dict[str, Any]] = []
    for c in calls:
        if c.scored is not None or c.expires_at <= at:
            continue
        obs = store.observations(c.mint, since=c.published_at)
        current = None
        peak = None
        if obs:
            mcs = [float(o.get("mc") or 0) / 1e9 for o in obs]
            current = mcs[-1]
            peak = max(mcs)
        entry = float(c.entry_mc_sol or 0)
        target = float(c.target_mc_sol or 0)
        invalidation = float(c.invalidation_mc_sol or 0)
        if current is None:
            progress = None
            flag = "no-read"
        elif target > entry:
            progress = max(0.0, min(1.0,
                                     (current - entry) / (target - entry)))
            if progress >= 0.8:
                flag = "near-target"
            elif current <= invalidation * 1.15:
                flag = "near-invalidation"
            elif current < entry:
                flag = "underwater"
            else:
                flag = "on-track"
        else:
            progress = None
            flag = "no-read"
        rows.append({
            "mint": c.mint,
            "symbol": c.symbol or "",
            "call_id": c.call_id or "",
            "call_class": c.call_class or "",
            "entry_mc_sol": entry,
            "target_mc_sol": target,
            "invalidation_mc_sol": invalidation,
            "current_mc_sol": current,
            "peak_mc_sol": peak,
            "composite": float(c.composite or 0),
            "progress": progress,
            "flag": flag,
            "seconds_to_expiry": max(0, int(c.expires_at - at)),
            "published_at": float(c.published_at or 0),
        })
    rows.sort(key=lambda r: r["seconds_to_expiry"])
    return rows


def _fmt_mc(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


def render_board(rows: list[dict[str, Any]], total: int, *,
                 filters_desc: str = "") -> str:
    """Render the board as plain chat/CLI text (SOL only)."""
    if total == 0:
        return ("pump board: no dossiers yet — run /pump watch once "
                "and the board fills with ranked candidates")
    if not rows:
        return (f"pump board: {total} dossier(s), none pass the "
                f"filters{filters_desc} — loosen them")
    lines = [f"🎰 pump board — {len(rows)} of {total} dossier(s)"
             + (filters_desc or "")]
    for i, r in enumerate(rows, 1):
        mark = "★ " if r["watched"] else ""
        rug_flag = " ⚠️ RUGGER-DEV" if r["dev_serial_rugger"] else ""
        att = f" att:{r['attention']}" if r["attention"] else ""
        lines.append(
            f"{i}. {mark}{r['symbol'] or r['mint'][:8]} "
            f"[{r['verdict']}] comp {r['composite']:.2f} "
            f"rug {r['rug_risk']:.2f} manip {r['manipulation']:.2f} "
            f"mom {r['momentum']:.2f} smart {r['smart_money']:.2f} — "
            f"curve {r['curve_pct']:.0%} mcap "
            f"{_fmt_mc(r['mcap_sol'])} SOL, {r['age']} old"
            f"{att}{rug_flag}")
        if r["evidence"]:
            lines.append(f"   {r['evidence']}")
    lines.append("mcap is last seen at dossier refresh, SOL — "
                 "not live; /pump dossier <mint> for the full record")
    return "\n".join(lines)


def render_open(rows: list[dict[str, Any]]) -> str:
    """Render open calls with live progress as plain text."""
    if not rows:
        return ("no open calls — nothing unscored and unexpired in "
                "the ledger; /pump calls shows the graded history")
    lines = [f"🎰 open calls — {len(rows)} live"]
    for r in rows:
        mins = r["seconds_to_expiry"] // 60
        expiry = f"{mins}m left" if mins < 60 else \
            f"{mins // 60}h{mins % 60:02d}m left"
        if r["progress"] is None:
            prog = "no curve read since entry"
        else:
            prog = (f"{r['progress']:.0%} to target "
                    f"({ _fmt_mc(r['current_mc_sol'])} SOL now vs "
                    f"{r['entry_mc_sol']:.1f} entry, "
                    f"{r['invalidation_mc_sol']:.1f} invalidation)")
        lines.append(
            f"• {r['symbol'] or r['mint'][:8]} [{r['flag']}] "
            f"entry {r['entry_mc_sol']:.1f} → target "
            f"{r['target_mc_sol']:.1f} SOL — {prog} — {expiry}")
        if r["peak_mc_sol"] is not None:
            lines.append(f"  peak since entry: "
                         f"{r['peak_mc_sol']:.1f} SOL")
    return "\n".join(lines)
