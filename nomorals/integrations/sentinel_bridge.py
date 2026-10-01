"""Bridge between Devon and the Sentinel.py trading engine (Prompt 07).

Sentinel.py is vendored as a **git submodule** at ``vendor/sentinel`` — it is
NOT reimplemented here. To check it out::

    git submodule update --init vendor/sentinel

and to pin it to the branch the bridge was tested against::

    cd vendor/sentinel && git checkout arena/01a0a420-sentinel-py

Every Sentinel import in this module is LAZY (inside functions) so Devon
starts fast and works fine when the submodule is absent — callers get
:exc:`SentinelUnavailable` with the exact recovery command instead of an
``ImportError`` traceback. ``import nomorals.integrations.sentinel_bridge``
must never import pandas/numpy.

Market data is **keyless-first**: :func:`load_data` defaults to
``source="auto"``, which pulls OHLCV from the free adapters in
:mod:`nomorals.integrations.market_data` (Binance/Kraken/Coinbase/CoinGecko
for crypto, Yahoo/Stooq for stocks, Frankfurter/Yahoo for fiat FX) with no API keys and
no extra packages. ``source="ccxt"`` / ``source="yfinance"`` keep the legacy
package-backed paths for callers that want them.

Tested Sentinel commit: see :data:`TESTED_COMMIT`. ``doctor()`` warns when the
checked-out submodule differs.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..core.logging_setup import get_logger

__all__ = [
    "TESTED_COMMIT",
    "VENDOR_ROOT",
    "SentinelUnavailable",
    "SentinelError",
    "LiveTradingDisabled",
    "sentinel_available",
    "checked_commit",
    "get_engine",
    "load_data",
    "make_synthetic_bars",
    "list_strategies",
    "doctor",
    "MarketDataProvider",
]

_log = get_logger(__name__)

# The commit this bridge was developed and tested against.
TESTED_COMMIT = "3bb7e22efa3e00842df2e0a84d4364e8eaa742f6"

VENDOR_ROOT = Path(__file__).resolve().parents[2] / "vendor" / "sentinel"

CONFIGS = {
    "crypto": "crypto.yaml",
    "forex": "forex.yaml",
    "stocks": "stocks.yaml",
}
PROFILES = ("default", "aggressive", "conservative")

_MISSING_DEPS_HINT = (
    "pip install numpy pandas scipy scikit-learn pyyaml joblib"
)


class SentinelUnavailable(Exception):
    """Raised when the Sentinel.py submodule is not checked out."""


class SentinelError(Exception):
    """A Sentinel call failed (bad data, engine error, missing dep)."""


class LiveTradingDisabled(Exception):
    """A live order was attempted while live trading is not unlocked."""


def sentinel_available() -> bool:
    """True when the submodule directory exists and looks like Sentinel."""
    return (VENDOR_ROOT / "sentinel" / "core" / "engine.py").is_file()


def checked_commit() -> str | None:
    """Short commit hash of the checked-out submodule, or None."""
    if not sentinel_available():
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(VENDOR_ROOT), capture_output=True, text=True,
            timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _ensure_path() -> None:
    if not sentinel_available():
        raise SentinelUnavailable(
            "Sentinel.py is not checked out. Run:\n"
            "  git submodule update --init vendor/sentinel\n"
            "then retry. Paper trading, analysis and backtests need it."
        )
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _require(dep: str, install_hint: str) -> Any:
    try:
        return __import__(dep)
    except ImportError as exc:
        raise SentinelError(
            f"missing dependency {dep!r}: {install_hint}") from exc


def get_engine(market: str = "crypto", profile: str = "default") -> Any:
    """Build a Sentinel ``QuantumEngine`` for a market + risk profile.

    ``market`` is crypto|forex|stocks, ``profile`` is default|aggressive|
    conservative. The profile picks the matching YAML preset from the
    submodule's ``configs/``.
    """
    _ensure_path()
    market = (market or "crypto").strip().lower()
    profile = (profile or "default").strip().lower()
    if market not in CONFIGS:
        raise SentinelError(
            f"unknown market {market!r}; expected one of {sorted(CONFIGS)}")
    if profile not in PROFILES:
        raise SentinelError(
            f"unknown profile {profile!r}; expected one of {list(PROFILES)}")
    config_name = CONFIGS[market] if profile == "default" else f"{profile}.yaml"
    config_path = VENDOR_ROOT / "configs" / config_name
    if not config_path.is_file():
        raise SentinelError(f"missing Sentinel config: {config_path}")
    try:
        from sentinel.core.engine import QuantumEngine
    except ImportError as exc:
        raise SentinelError(
            f"could not import the Sentinel engine ({exc}); "
            f"{_MISSING_DEPS_HINT}") from exc
    started = time.time()
    engine = QuantumEngine(str(config_path))
    _log.info("sentinel engine ready: market=%s profile=%s (%.1fs)",
              market, profile, time.time() - started)
    return engine


def load_data(symbol: str, market: str = "crypto", timeframe: str = "1h",
              bars: int = 2000, source: str = "auto") -> Any:
    """Load OHLCV bars for a symbol, routed by market.

    ``source="auto"`` (default) uses the keyless free adapters in
    :mod:`nomorals.integrations.market_data` — no API keys, no extra
    packages. ``source="ccxt"`` forces the legacy ccxt path (crypto,
    needs the ``ccxt`` package); ``source="yfinance"`` forces the legacy
    yfinance path (forex/stocks, needs ``yfinance``). A symbol ending in
    ``.csv`` always loads a local CSV.
    """
    _ensure_path()
    market = (market or "crypto").strip().lower()
    symbol = (symbol or "").strip()
    source = (source or "auto").strip().lower()
    if not symbol:
        raise SentinelError("symbol is required")
    if symbol.lower().endswith(".csv"):
        path = Path(symbol).expanduser()
        if not path.is_file():
            raise SentinelError(f"CSV not found: {path}")
        try:
            from sentinel.data import feed
        except ImportError as exc:
            raise SentinelError(
                f"could not import sentinel.data.feed ({exc}); "
                f"{_MISSING_DEPS_HINT}") from exc
        return feed.load_csv(str(path))
    if source == "auto":
        try:
            from . import market_data
        except ImportError as exc:
            raise SentinelError(
                f"could not import market_data ({exc})") from exc
        try:
            return market_data.get_ohlcv(symbol, market=market,
                                         timeframe=timeframe, bars=bars)
        except market_data.MarketDataError as exc:
            raise SentinelError(f"free market-data feed failed: {exc}") from exc
    try:
        from sentinel.data import feed
    except ImportError as exc:
        raise SentinelError(
            f"could not import sentinel.data.feed ({exc}); "
            f"{_MISSING_DEPS_HINT}") from exc
    if source == "ccxt" or (source == "auto" and market == "crypto"):
        _require("ccxt", "pip install ccxt   (crypto feeds need it)")
        try:
            return feed.load_ccxt(symbol, timeframe=timeframe, limit=bars)
        except Exception as exc:  # noqa: BLE001 - network/provider errors
            raise SentinelError(f"ccxt feed failed for {symbol}: {exc}") from exc
    if source == "yfinance" or (source == "auto" and market in ("forex", "stocks")):
        _require("yfinance", "pip install yfinance   (forex/stock feeds need it)")
        try:
            return feed.load_yfinance(symbol, interval=timeframe)
        except Exception as exc:  # noqa: BLE001
            raise SentinelError(
                f"yfinance feed failed for {symbol}: {exc}") from exc
    raise SentinelError(f"unknown market {market!r}; expected crypto|forex|stocks")


def make_synthetic_bars(n: int = 600, seed: int = 42) -> Any:
    """Synthetic OHLCV bars for doctor checks and tests (no network)."""
    _ensure_path()
    try:
        from sentinel.data.feed import make_synthetic
    except ImportError as exc:
        raise SentinelError(
            f"could not import sentinel.data.feed ({exc}); "
            f"{_MISSING_DEPS_HINT}") from exc
    return make_synthetic(max(300, n), seed=seed)


def list_strategies() -> list[str]:
    """Names of the strategies in Sentinel's registry."""
    _ensure_path()
    try:
        from sentinel.core.engine import list_strategies as _ls
    except ImportError:
        try:
            from sentinel.strategies import list_strategies as _ls
        except ImportError as exc:
            raise SentinelError(
                f"strategy registry unavailable ({exc})") from exc
    return list(_ls())


@dataclass
class DoctorCheck:
    name: str
    ok: bool
    detail: str = ""


@dataclass
class DoctorReport:
    ok: bool
    commit: str | None
    commit_matches: bool
    checks: list[DoctorCheck] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "commit": self.commit,
            "commit_matches": self.commit_matches,
            "tested_commit": TESTED_COMMIT,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail}
                       for c in self.checks],
        }

    def summary_text(self) -> str:
        lines = [f"sentinel doctor: {'OK' if self.ok else 'FAIL'}"]
        for c in self.checks:
            mark = "ok" if c.ok else "FAIL"
            lines.append(f"  [{mark}] {c.name}"
                         + (f" — {c.detail}" if c.detail else ""))
        return "\n".join(lines)


def doctor() -> DoctorReport:
    """Check the whole Sentinel integration: submodule, commit pin, deps,
    configs, and an engine smoke test on synthetic data."""
    checks: list[DoctorCheck] = []

    if not sentinel_available():
        checks.append(DoctorCheck(
            "submodule", False,
            "vendor/sentinel is not checked out — run: "
            "git submodule update --init vendor/sentinel"))
        return DoctorReport(ok=False, commit=None, commit_matches=False,
                            checks=checks)
    checks.append(DoctorCheck("submodule", True, str(VENDOR_ROOT)))

    commit = checked_commit()
    matches = bool(commit) and TESTED_COMMIT.startswith(commit)
    checks.append(DoctorCheck(
        "commit-pin", matches,
        f"checked out {commit}, tested {TESTED_COMMIT[:7]}"
        if not matches else f"matches tested {TESTED_COMMIT[:7]}"))

    missing: list[str] = []
    for dep in ("numpy", "pandas", "scipy", "yaml"):
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    optional_missing: list[str] = []
    for dep, label in (("sklearn", "scikit-learn"), ("joblib", "joblib"),
                       ("ccxt", "ccxt"), ("yfinance", "yfinance")):
        try:
            __import__(dep)
        except ImportError:
            optional_missing.append(f"{label} (optional)")
    checks.append(DoctorCheck(
        "dependencies", not missing,
        "missing: " + ", ".join(missing + optional_missing)
        if (missing or optional_missing) else "all deps present"))

    for market, cfg in CONFIGS.items():
        exists = (VENDOR_ROOT / "configs" / cfg).is_file()
        checks.append(DoctorCheck(f"config:{market}", exists, cfg))

    try:
        from . import market_data
        status = market_data.source_status()
        keyed = [src for src, on in status["keyed"].items() if on]
        checks.append(DoctorCheck(
            "market-data", True,
            f"keyless: {', '.join(status['keyless'])}"
            + (f" | keyed upgrades active: {', '.join(keyed)}"
               if keyed else " | no keyed upgrades (optional)")))
    except Exception as exc:  # noqa: BLE001 - never fail doctor on this
        checks.append(DoctorCheck("market-data", False, str(exc)[:160]))

    try:
        engine = get_engine("crypto", "default")
        df = make_synthetic_bars(600, seed=7)
        report = engine.scan(df, symbol="BTC/USDT")
        checks.append(DoctorCheck(
            "engine-smoke", True,
            f"scan ok: regime={report.regime_label.strip()} "
            f"position={report.position_now:+.0f} "
            f"strategies={report.n_strategies}"))
        smoke_ok = True
    except Exception as exc:  # noqa: BLE001 - report, don't raise
        checks.append(DoctorCheck("engine-smoke", False, str(exc)[:200]))
        smoke_ok = False

    ok = all(c.ok for c in checks if c.name != "commit-pin") and smoke_ok
    # a commit mismatch is a warning, not a failure
    return DoctorReport(ok=ok, commit=commit, commit_matches=matches,
                        checks=checks)


class MarketDataProvider(Protocol):
    """Prompt 04 (morning briefing) market-data protocol.

    Defined here so the briefing implementer can depend on it before
    Prompt 04 lands; backed by the Sentinel bridge.
    """

    def quote(self, symbol: str, market: str = "crypto") -> dict[str, Any]:
        """Latest price + overnight change for a symbol."""
        ...

    def overnight_movers(self, symbols: list[str],
                         market: str = "crypto") -> list[dict[str, Any]]:
        """Quotes sorted by absolute overnight change, descending."""
        ...
