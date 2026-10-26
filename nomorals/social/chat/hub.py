"""Devon Hub: the tool hub card, per-screen inline keyboards,
per-tool cooldowns, and BIN lookup.

The hub is the button-driven tool surface on the Devon bots
(owner-only via the existing `caller_may_run` grant — this module
adds no gate of its own). Navigation follows the BotFather screen
STACK (owner refinement, 2026-10-26 — the model, not the look):

* hub screen → tool screen → result screen;
* every screen carries its OWN buttons: that screen's actions plus
  a « Back button to its PARENT (exactly one level up);
* tapping a button NAVIGATES (the adapter edits the tapped message
  into the new screen where the platform allows) instead of
  printing new command text;
* buttons are never a fixed set and never a static bottom keyboard.

Screens and their Back targets:
* hub card (root, no Back) → tools + help;
* proxy tool screen → Back to Hub; its Refresh runs the flow;
* proxy result / cooldown screens → Back to Proxy (tool screen);
* BIN tool screen → Back to Hub; a result → Back to BIN;
* help screen → Back to Hub.

Style is Devon's own (🥷/🧭 headers, section lines) — only the
navigation MODEL comes from BotFather.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .tgbot_buttons import Keyboard, nav_row

__all__ = [
    "BIN_HEADER",
    "HUB_HEADER",
    "PROXY_HEADER",
    "CooldownStore",
    "bin_usage_text",
    "format_bin_result",
    "hub_help_text",
    "hub_keyboard",
    "hub_text",
    "is_hub_screen",
    "lookup_bin",
    "proxy_intro_text",
]


def is_hub_screen(text: str) -> bool:
    """True when *text* is a hub screen (any level of the stack).

    The adapter's navigation-edit branch uses this: only hub screens
    are eligible to replace the tapped message. Header-anchored
    (startswith), so a hub card that MENTIONS the tools never
    misclassifies, and non-hub text never qualifies.
    """
    t = text or ""
    return (t.startswith(HUB_HEADER) or t.startswith(PROXY_HEADER)
            or t.startswith(BIN_HEADER))

#: Stable screen headers — the button-derivation hooks in
#: :mod:`nomorals.social.chat.tgbot_buttons` match on these, so a hub
#: screen's buttons follow its own text (gated bot path). Keep them
#: stable: renaming one silently drops that screen's keyboard.
HUB_HEADER = "🧭 Devon Hub"
PROXY_HEADER = "🌐 Proxy lab"
BIN_HEADER = "💳 BIN lookup"

#: Proxy generation cooldown (mirrors the hourly rhythm of the
#: reference tool bot): one generation per user per hour.
PROXY_COOLDOWN_S = 3600.0


def hub_text(*, proxy_line: str | None = None) -> str:
    """The hub card. ``proxy_line`` is live pool state from the
    caller (e.g. "12 working, refreshed 20m ago"); without it the
    card says so honestly instead of inventing a number."""
    lines = [
        f"{HUB_HEADER} — tap a tool below:",
        "",
        "🌐 Proxy — fresh tested proxies as a .txt file",
        f"  pool: {proxy_line or 'unknown — tap Proxy to refresh'}",
        "💳 BIN lookup — issuer / scheme / country for a card BIN",
        "  ready — paste a BIN, or tap BIN lookup",
        "",
        "Tap a button below to run it.",
    ]
    return "\n".join(lines)


def hub_help_text() -> str:
    """Hub help: what each tool does + the honest limits."""
    return (
        f"{HUB_HEADER} — help\n"
        "\n"
        "🌐 Proxy — scrapes free proxy lists, live-tests every\n"
        "candidate (liveness, latency, country), and sends the\n"
        "working ones as a .txt file. Free proxies rot fast\n"
        "(tens of minutes is normal) — refresh when they die.\n"
        "One generation per hour per chat.\n"
        "💳 BIN lookup — given a card BIN (first 6–8 digits),\n"
        "returns scheme, type, brand, country and issuing bank\n"
        "from public BIN data. Unknown BINs say so; the free\n"
        "lookup service is rate-limited (5/hour), so a 429 means\n"
        "try later, not a broken tool.\n"
        "\n"
        "Buttons on each screen are that screen's actions — Back\n"
        "always returns here."
    )


def proxy_intro_text(*, cooldown_note: str = "") -> str:
    """Proxy screen opener (before a run starts)."""
    lines = [
        f"{PROXY_HEADER} — fresh tested proxies",
        "",
        "Tap Refresh to scrape + live-test candidates and get a",
        ".txt file of the working ones. Free proxies rot fast —",
        "a file from an hour ago is usually dead.",
    ]
    if cooldown_note:
        lines += ["", cooldown_note]
    return "\n".join(lines)


def bin_usage_text() -> str:
    """BIN screen opener: what to paste."""
    return (
        f"{BIN_HEADER} — issuer / scheme / country\n"
        "\n"
        "Paste a BIN (the first 6–8 digits of a card):\n"
        "  /hub bin 457173\n"
        "\n"
        "Unknown BINs say so honestly; the free lookup service\n"
        "is rate-limited (5/hour) — a 429 means try later."
    )


def _row(*pairs: tuple[str, str]) -> list[tuple[str, str]]:
    return nav_row(*pairs)


def hub_keyboard() -> Keyboard:
    """Hub card buttons: one row per tool + help. Dynamic per
    screen — this is the hub screen's set, not a global menu."""
    return [
        _row(("🌐 Proxy", "toolhub proxy"), ("💳 BIN lookup", "toolhub bin")),
        _row(("❓ Help", "toolhub help")),
    ]


def proxy_keyboard() -> Keyboard:
    """Proxy TOOL screen buttons: run the flow + Back to Hub."""
    return [
        _row(("🔄 Refresh pool", "toolhub proxy run")),
        _row(("« Back to Hub", "toolhub")),
    ]


def proxy_result_keyboard() -> Keyboard:
    """Proxy RESULT screen buttons: re-run + Back to Proxy (one
    level up — the tool screen, not the hub)."""
    return [
        _row(("🔄 Refresh pool", "toolhub proxy run")),
        _row(("« Back to Proxy", "toolhub proxy")),
    ]


def proxy_cooldown_keyboard() -> Keyboard:
    """Cooldown notice buttons: re-check (real: the cooldown may
    have expired) + Back to Proxy (one level up). No dead buttons."""
    return [
        _row(("🔄 Try again", "toolhub proxy run")),
        _row(("« Back to Proxy", "toolhub proxy")),
    ]


def bin_keyboard() -> Keyboard:
    """BIN TOOL screen buttons: Back to Hub (the lookup itself is
    typed, so the tool screen has no action button)."""
    return [_row(("« Back to Hub", "toolhub"))]


def bin_result_keyboard() -> Keyboard:
    """BIN RESULT screen buttons: Back to BIN (one level up)."""
    return [_row(("« Back to BIN", "toolhub bin"))]


def help_keyboard() -> Keyboard:
    """Help screen buttons: Back to Hub."""
    return [_row(("« Back to Hub", "toolhub"))]


def keyboard_for_screen(screen: str) -> Keyboard:
    """Screen name → that screen's keyboard (single source of truth
    for both button-derivation hooks)."""
    return {
        "hub": hub_keyboard(),
        "proxy": proxy_keyboard(),
        "proxy_result": proxy_result_keyboard(),
        "proxy_cooldown": proxy_cooldown_keyboard(),
        "bin": bin_keyboard(),
        "bin_result": bin_result_keyboard(),
        "help": help_keyboard(),
    }.get(screen, hub_keyboard())


# ── cooldowns ──────────────────────────────────────────────────────────

class CooldownStore:
    """Per-user, per-tool cooldown timestamps, JSON-persisted.

    Same shape as proxylab's DomainAffinityManager: a lock, a JSON
    dict on disk, corrupt file → start clean, best-effort save.
    Survives restarts (the file is the state). Keys are
    ``f"{user_key}|{tool}"`` — the caller supplies the user key
    (the runtime passes the chat key, which is per-chat).
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._data: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    self._data = {str(k): float(v) for k, v in raw.items()
                                  if isinstance(v, (int, float))}
        except (OSError, ValueError):
            self._data = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._data, indent=1),
                                 encoding="utf-8")
        except OSError:
            pass  # cooldowns are best-effort; a failed save never blocks a tool

    def remaining(self, user_key: str, tool: str, cooldown_s: float,
                  *, now: float | None = None) -> float:
        """Seconds left on this cooldown, 0 when clear."""
        now = time.time() if now is None else now
        last = self._data.get(f"{user_key}|{tool}")
        if last is None:
            return 0.0
        return max(0.0, float(last) + float(cooldown_s) - now)

    def mark(self, user_key: str, tool: str,
             *, now: float | None = None) -> None:
        """Record a use now (call AFTER the cooldown check passes,
        BEFORE the work starts — a started run counts)."""
        with self._lock:
            self._data[f"{user_key}|{tool}"] = \
                time.time() if now is None else now
            self._save()

    @staticmethod
    def remaining_text(remaining_s: float) -> str:
        """Human wait time, rounded up (never "0m" for a live cooldown)."""
        total = int(remaining_s)
        if total < 60:
            return f"{max(1, total)}s"
        minutes = (total + 59) // 60
        if minutes < 60:
            return f"{minutes}m"
        return f"{minutes // 60}h {minutes % 60}m"


# ── BIN lookup ─────────────────────────────────────────────────────────

#: Public BIN data (binlist.net): free tier, no key. Documented
#: response: scheme / type / brand / prepaid / country / bank;
#: 404 = unknown BIN; 429 = free-tier rate limit (5/hour).
BIN_API_URL = "https://lookup.binlist.net/{bin}"
BIN_TIMEOUT_S = 10.0


def _default_fetch(url: str, headers: dict[str, str],
                   timeout: float) -> tuple[int, dict | None]:
    """urllib fetch returning (status, parsed JSON or None)."""
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return int(exc.code), None
    except (urllib.error.URLError, OSError, ValueError):
        return 0, None


def lookup_bin(bin_digits: str, *,
               fetcher: Callable[..., tuple[int, dict | None]] | None = None,
               timeout: float = BIN_TIMEOUT_S) -> dict[str, Any]:
    """Look up a BIN (first 6–8 digits) in public BIN data.

    Returns ``{"ok": True, ...}`` with scheme/type/brand/prepaid/
    country/bank, or ``{"ok": False, "error": ...}`` where error is
    one of: ``usage`` (not 6–8 digits — no call made), ``unknown``
    (404), ``rate_limited`` (429), ``unavailable`` (network/API
    failure). Unknown/unavailable are NEVER filled with invented
    data. BIN LOOKUP ONLY — this function generates nothing.
    """
    digits = "".join(ch for ch in (bin_digits or "") if ch.isdigit())
    if len(digits) < 6 or len(digits) > 8:
        return {"ok": False, "error": "usage",
                "detail": "a BIN is the first 6–8 digits of a card"}
    fetch = fetcher or _default_fetch
    try:
        status, data = fetch(BIN_API_URL.format(bin=digits),
                             {"Accept-Version": "3"}, timeout)
    except (OSError, ValueError):
        return {"ok": False, "error": "unavailable"}
    if status == 404:
        return {"ok": False, "error": "unknown"}
    if status == 429:
        return {"ok": False, "error": "rate_limited"}
    if status != 200 or not isinstance(data, dict):
        return {"ok": False, "error": "unavailable"}
    country = data.get("country") or {}
    bank = data.get("bank") or {}
    return {
        "ok": True,
        "bin": digits,
        "scheme": data.get("scheme"),
        "type": data.get("type"),
        "brand": data.get("brand"),
        "prepaid": data.get("prepaid"),
        "country": {
            "name": country.get("name"),
            "alpha2": country.get("alpha2"),
            "emoji": country.get("emoji"),
            "currency": country.get("currency"),
        },
        "bank": {
            "name": bank.get("name"),
            "city": bank.get("city"),
            "url": bank.get("url"),
            "phone": bank.get("phone"),
        },
    }


def format_bin_result(result: dict[str, Any]) -> str:
    """Render a BIN lookup result (or its honest failure)."""
    if not result.get("ok"):
        err = result.get("error")
        if err == "usage":
            return bin_usage_text()
        if err == "unknown":
            return (f"{BIN_HEADER}\n\n"
                    "Unknown BIN — no issuer on record for it.")
        if err == "rate_limited":
            return (f"{BIN_HEADER}\n\n"
                    "Rate-limited by the free lookup service "
                    "(5/hour) — try again later.")
        return (f"{BIN_HEADER}\n\n"
                "BIN lookup unavailable right now "
                "(network or service error) — try again later.")
    country = result.get("country") or {}
    bank = result.get("bank") or {}
    lines = [
        f"{BIN_HEADER} — {result.get('bin')}",
        "",
        f"scheme: {result.get('scheme') or '—'}",
        f"type: {result.get('type') or '—'}",
        f"brand: {result.get('brand') or '—'}",
        f"prepaid: {result.get('prepaid')}",
    ]
    cname = country.get("name")
    if cname:
        emoji = country.get("emoji") or ""
        lines.append(f"country: {emoji} {cname} "
                     f"({country.get('alpha2') or '—'})")
    bname = bank.get("name")
    if bname:
        lines.append(f"bank: {bname}"
                     + (f", {bank['city']}" if bank.get("city") else ""))
    return "\n".join(lines)
