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
    "BacktestReport",
    "BacktestTrade",
    "backtest",
    "sma_cross_strategy",
    "rsi_mean_reversion_strategy",
    "PaperTrader",
    "RiskGuard",
    "register_strategy",
    "get_strategy",
    "list_registered_strategies",
    "monte_carlo",
    "position_size",
    "graduated_capital",
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


# ── validation layer (works WITHOUT the Sentinel submodule) ────────────
# freqtrade's honest pipeline: backtest → walk-forward → Monte Carlo →
# dry-run → graduated capital. This is the validation half; the real
# Sentinel engine stays the execution path.


def _bar_lists(bars: Any) -> tuple[list[float], list[float], list[float],
                                   list[float], list[float]]:
    """Normalize bars → (ts, open, high, low, close). Accepts DataFrame,
    list-of-lists [ts,o,h,l,c,v], or list-of-dicts."""
    ts, o, h, l, c = [], [], [], [], []
    try:
        import pandas as pd  # type: ignore[import]
        if isinstance(bars, pd.DataFrame):
            idx = bars.index
            try:
                ts = [float(t.value // 10**9) if hasattr(t, "value") else
                      float(t) for t in idx]
            except Exception:  # noqa: BLE001
                ts = list(range(len(bars)))
            for col, dst in (("open", o), ("high", h), ("low", l),
                             ("close", c)):
                dst.extend(float(x) for x in bars[col].tolist())
            return ts, o, h, l, c
    except ImportError:
        pass
    for i, row in enumerate(bars or []):
        if isinstance(row, dict):
            ts.append(float(row.get("timestamp", row.get("ts", i))))
            o.append(float(row.get("open", 0)))
            h.append(float(row.get("high", 0)))
            l.append(float(row.get("low", 0)))
            c.append(float(row.get("close", 0)))
        elif isinstance(row, (list, tuple)) and len(row) >= 5:
            ts.append(float(row[0]) / 1000 if row[0] > 10**12 else
                      float(row[0]))
            o.append(float(row[1])); h.append(float(row[2]))
            l.append(float(row[3])); c.append(float(row[4]))
    return ts, o, h, l, c


def _sma(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if period < 1:
        return out
    run = 0.0
    for i, v in enumerate(values):
        run += v
        if i >= period:
            run -= values[i - period]
        if i >= period - 1:
            out[i] = run / period
    return out


def sma_cross_strategy(fast: int = 10, slow: int = 30) -> Callable:
    """Reference strategy: long when fast SMA crosses above slow SMA,
    flat when it crosses back. Returns signals_fn(bars_dict)."""
    def _fn(data: dict[str, list[float]]) -> list[int]:
        closes = data["close"]
        f, s = _sma(closes, fast), _sma(closes, slow)
        sig = [0] * len(closes)
        pos = 0
        for i in range(1, len(closes)):
            if f[i] is None or s[i] is None or f[i - 1] is None \
                    or s[i - 1] is None:
                continue
            if pos == 0 and f[i] > s[i] and f[i - 1] <= s[i - 1]:
                pos = 1
            elif pos == 1 and f[i] < s[i] and f[i - 1] >= s[i - 1]:
                pos = 0
            sig[i] = pos
        return sig
    _fn.__name__ = f"sma_cross({fast},{slow})"
    return _fn


def rsi_mean_reversion_strategy(period: int = 14, oversold: float = 30,
                                overbought: float = 70) -> Callable:
    """Reference strategy: long when RSI < oversold, exit when RSI >
    overbought. Returns signals_fn(bars_dict)."""
    def _fn(data: dict[str, list[float]]) -> list[int]:
        closes = data["close"]
        n = len(closes)
        sig = [0] * n
        pos = 0
        gains, losses = [], []
        for i in range(1, n):
            d = closes[i] - closes[i - 1]
            gains.append(max(d, 0)); losses.append(max(-d, 0))
        ag = al = None
        for i in range(period, n):
            if ag is None:
                ag = sum(gains[:period]) / period
                al = sum(losses[:period]) / period
            else:
                ag = (ag * (period - 1) + gains[i - 1]) / period
                al = (al * (period - 1) + losses[i - 1]) / period
            rsi = 100 - 100 / (1 + ag / al) if al else 100.0
            if pos == 0 and rsi < oversold:
                pos = 1
            elif pos == 1 and rsi > overbought:
                pos = 0
            sig[i] = pos
        return sig
    _fn.__name__ = f"rsi_mr({period},{oversold},{overbought})"
    return _fn


@dataclass
class BacktestTrade:
    entry_i: int
    exit_i: int
    entry_price: float
    exit_price: float
    pnl_pct: float
    pnl_abs: float


@dataclass
class BacktestReport:
    strategy: str
    symbol: str
    timeframe: str
    n_bars: int
    n_trades: int = 0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    fee_pct: float = 0.0
    slippage_pct: float = 0.0
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy, "symbol": self.symbol,
            "timeframe": self.timeframe, "n_bars": self.n_bars,
            "n_trades": self.n_trades, "win_rate": self.win_rate,
            "total_return_pct": self.total_return_pct,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe": self.sharpe,
            "avg_win_pct": self.avg_win_pct,
            "avg_loss_pct": self.avg_loss_pct,
            "fee_pct": self.fee_pct, "slippage_pct": self.slippage_pct,
        }

    def format_report(self) -> str:
        """God-tier backtest card."""
        verdict = ("✅ tradable" if self.profit_factor > 1.2
                   and self.max_drawdown_pct > -15
                   and self.n_trades >= 20 else "⚠️ not yet")
        spark = _sparkline(self.equity_curve)
        lines = [
            f"📊 **backtest: {self.strategy}** on {self.symbol} "
            f"({self.timeframe}, {self.n_bars} bars)",
            f"{verdict}",
            f"trades {self.n_trades} · win {self.win_rate:.0%} · "
            f"return {self.total_return_pct:+.1f}%",
            f"profit factor {self.profit_factor:.2f} · max DD "
            f"{self.max_drawdown_pct:.1f}% · sharpe {self.sharpe:.2f}",
            f"avg win {self.avg_win_pct:+.2f}% / avg loss "
            f"{self.avg_loss_pct:+.2f}%",
            f"_fee {self.fee_pct:.3%} + slippage {self.slippage_pct:.3%}"
            f" modeled_",
        ]
        if spark:
            lines.append(f"`{spark}` equity")
        return "\n".join(lines)


def _sparkline(values: list[float], width: int = 40) -> str:
    if len(values) < 2:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    step = max(1, len(values) // width)
    sample = values[::step][:width]
    lo, hi = min(sample), max(sample)
    span = hi - lo or 1e-9
    return "".join(blocks[min(7, int((v - lo) / span * 7))]
                   for v in sample)


def backtest(signals_fn: Callable[[dict[str, list[float]]], list[int]],
             bars: Any, *, symbol: str = "?", timeframe: str = "1h",
             fee_pct: float = 0.001, slippage_pct: float = 0.0005,
             stake: float = 1000.0) -> BacktestReport:
    """Vectorized long/flat backtester with honest costs.

    ``signals_fn`` maps {open,high,low,close,timestamp} → position list
    in {-1,0,1} (shorts treated as flat — spot only). Every round trip
    pays ``fee_pct`` twice + ``slippage_pct`` twice. No lookahead:
    signals at bar i execute at bar i+1's open.
    """
    ts, o, h, l, c = _bar_lists(bars)
    n = len(c)
    if n < 30:
        raise SentinelError("backtest needs ≥30 bars")
    data = {"timestamp": ts, "open": o, "high": h, "low": l, "close": c}
    sig = signals_fn(data)
    if len(sig) != n:
        raise SentinelError(
            f"strategy returned {len(sig)} signals for {n} bars")

    name = getattr(signals_fn, "__name__", "strategy")
    report = BacktestReport(strategy=name, symbol=symbol,
                            timeframe=timeframe, n_bars=n,
                            fee_pct=fee_pct, slippage_pct=slippage_pct)
    equity = [stake]
    pos = 0
    entry_price = 0.0
    entry_i = 0
    entry_equity = stake
    peak = stake
    max_dd = 0.0
    rets: list[float] = []

    for i in range(n - 1):
        want = 1 if sig[i] > 0 else 0  # spot: long/flat only
        px = o[i + 1]  # execute next bar's open — no lookahead
        if want != pos:
            if pos == 1:  # exit
                exit_px = px * (1 - slippage_pct)
                gross = (exit_px - entry_price) / entry_price
                net = gross - 2 * fee_pct - 2 * slippage_pct
                pnl_abs = entry_equity * net
                report.trades.append(BacktestTrade(
                    entry_i=entry_i, exit_i=i + 1,
                    entry_price=entry_price, exit_price=exit_px,
                    pnl_pct=net * 100, pnl_abs=pnl_abs))
                equity.append(equity[-1] + pnl_abs)
            else:  # enter
                entry_price = px * (1 + slippage_pct)
                entry_i = i + 1
                entry_equity = equity[-1]
            pos = want
        else:
            equity.append(equity[-1])
        rets.append((equity[-1] - equity[-2]) / equity[-2]
                    if equity[-2] else 0.0)
        peak = max(peak, equity[-1])
        dd = (equity[-1] - peak) / peak * 100
        max_dd = min(max_dd, dd)

    # close any open position at the last close
    if pos == 1:
        exit_px = c[-1] * (1 - slippage_pct)
        net = (exit_px - entry_price) / entry_price - 2 * fee_pct \
            - 2 * slippage_pct
        report.trades.append(BacktestTrade(
            entry_i=entry_i, exit_i=n - 1, entry_price=entry_price,
            exit_price=exit_px, pnl_pct=net * 100, pnl_abs=stake * net))
        equity.append(equity[-1] + stake * net)

    report.equity_curve = equity
    report.n_trades = len(report.trades)
    report.max_drawdown_pct = max_dd
    if report.trades:
        wins = [t for t in report.trades if t.pnl_pct > 0]
        losses = [t for t in report.trades if t.pnl_pct <= 0]
        report.win_rate = len(wins) / len(report.trades)
        report.total_return_pct = (equity[-1] - stake) / stake * 100
        gp = sum(t.pnl_pct for t in wins)
        gl = abs(sum(t.pnl_pct for t in losses)) or 1e-9
        report.profit_factor = gp / gl
        report.avg_win_pct = gp / len(wins) if wins else 0.0
        report.avg_loss_pct = (sum(t.pnl_pct for t in losses)
                               / len(losses)) if losses else 0.0
    if len(rets) > 1:
        mu = sum(rets) / len(rets)
        sd = (sum((r - mu) ** 2 for r in rets) / len(rets)) ** 0.5
        report.sharpe = (mu / sd * (252 ** 0.5)) if sd else 0.0
    return report


class RiskGuard:
    """freqtrade-style protections: the kill-switch layer.

    ``check()`` returns (allowed, reason). ``halt()`` is the emergency
    brake — call it from Telegram /stop and everything stops.
    """

    def __init__(self, *, daily_stop_loss_pct: float = 3.0,
                 max_drawdown_pct: float = 15.0,
                 max_consecutive_losses: int = 5,
                 max_open_trades: int = 5,
                 cooldown_after_halt_s: float = 3600.0):
        self.daily_stop_loss_pct = daily_stop_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.max_open_trades = max_open_trades
        self.cooldown_after_halt_s = cooldown_after_halt_s
        self._halted_until = 0.0
        self._consec_losses = 0
        self._day = ""
        self._day_pnl_pct = 0.0

    def halt(self, reason: str = "manual") -> None:
        """Emergency brake — no new trades until cooldown expires."""
        import time as _t
        self._halted_until = _t.time() + self.cooldown_after_halt_s
        _log.warning("RiskGuard HALT: %s (cooldown %.0fs)", reason,
                     self.cooldown_after_halt_s)

    @property
    def halted(self) -> bool:
        import time as _t
        return _t.time() < self._halted_until

    def record_trade(self, pnl_pct: float) -> None:
        """Feed closed-trade PnL (drives consecutive-loss + daily PnL)."""
        import time as _t, datetime as _dt
        today = _dt.datetime.now().strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._day_pnl_pct = 0.0
        self._day_pnl_pct += pnl_pct
        if pnl_pct > 0:
            self._consec_losses = 0
        else:
            self._consec_losses += 1
        if self._consec_losses >= self.max_consecutive_losses:
            self.halt(f"{self._consec_losses} consecutive losses")

    def check(self, *, drawdown_pct: float = 0.0,
              open_trades: int = 0) -> tuple[bool, str]:
        """(allowed, reason) for opening a new trade right now."""
        if self.halted:
            return False, "RiskGuard halted (cooldown)"
        if self._day_pnl_pct <= -abs(self.daily_stop_loss_pct):
            self.halt(f"daily stop-loss {self._day_pnl_pct:.1f}%")
            return False, "daily stop-loss hit"
        if drawdown_pct <= -abs(self.max_drawdown_pct):
            self.halt(f"max drawdown {drawdown_pct:.1f}%")
            return False, "max drawdown hit"
        if open_trades >= self.max_open_trades:
            return False, f"max open trades ({self.max_open_trades})"
        return True, "ok"


class PaperTrader:
    """Dry-run engine: real market data, fake money. freqtrade's
    ``--dry-run`` as a class. Feed bars via on_bar(); inspect
    positions/equity any time. Never touches real orders."""

    def __init__(self, signals_fn: Callable, *,
                 stake: float = 1000.0, fee_pct: float = 0.001,
                 slippage_pct: float = 0.0005,
                 risk: RiskGuard | None = None):
        self.signals_fn = signals_fn
        self.stake = stake
        self.fee_pct = fee_pct
        self.slippage_pct = slippage_pct
        self.risk = risk or RiskGuard()
        self.equity = stake
        self.position: dict[str, Any] | None = None
        self.trades: list[BacktestTrade] = []
        self._bars: list[list[float]] = []
        self._i = 0

    def on_bar(self, bar: list[float] | dict[str, float]) -> dict[str, Any]:
        """Feed one bar [ts,o,h,l,c,v] or dict. Returns status dict."""
        if isinstance(bar, dict):
            row = [bar.get("timestamp", self._i), bar.get("open", 0),
                   bar.get("high", 0), bar.get("low", 0),
                   bar.get("close", 0), bar.get("volume", 0)]
        else:
            row = list(bar)
        self._bars.append(row)
        self._i += 1
        if len(self._bars) < 30:
            return {"status": "warming_up", "bars": len(self._bars)}
        ts = [r[0] for r in self._bars]
        data = {"timestamp": ts, "open": [r[1] for r in self._bars],
                "high": [r[2] for r in self._bars],
                "low": [r[3] for r in self._bars],
                "close": [r[4] for r in self._bars]}
        sig = self.signals_fn(data)
        want = 1 if sig[-1] > 0 else 0
        px = row[1]  # execute at this bar's open
        events = []
        if self.position and want == 0:
            entry = self.position
            exit_px = px * (1 - self.slippage_pct)
            net = (exit_px - entry["price"]) / entry["price"] \
                - 2 * self.fee_pct - 2 * self.slippage_pct
            pnl_abs = self.stake * net
            self.equity += pnl_abs
            self.trades.append(BacktestTrade(
                entry_i=entry["i"], exit_i=self._i,
                entry_price=entry["price"], exit_price=exit_px,
                pnl_pct=net * 100, pnl_abs=pnl_abs))
            self.risk.record_trade(net * 100)
            self.position = None
            events.append(f"exit {net * 100:+.2f}%")
        elif not self.position and want == 1:
            allowed, reason = self.risk.check(open_trades=0)
            if allowed:
                self.position = {"price": px * (1 + self.slippage_pct),
                                 "i": self._i}
                events.append(f"enter @ {px:.4f}")
            else:
                events.append(f"blocked: {reason}")
        return {"status": "live", "equity": round(self.equity, 2),
                "position": self.position is not None,
                "trades": len(self.trades), "events": events,
                "halted": self.risk.halted}


# ── strategy registry ────────────────────────────────────────────────
_STRATEGIES: dict[str, dict[str, Any]] = {}


def register_strategy(name: str, signals_fn: Callable, *,
                      description: str = "",
                      params: dict[str, Any] | None = None) -> None:
    """Register a named strategy (metadata + backtest summary live here)."""
    _STRATEGIES[name] = {"fn": signals_fn, "description": description,
                         "params": params or {},
                         "backtests": []}


def get_strategy(name: str) -> Callable:
    try:
        return _STRATEGIES[name]["fn"]
    except KeyError:
        raise SentinelError(
            f"unknown strategy {name!r}; registered: "
            f"{sorted(_STRATEGIES)}") from None


def list_registered_strategies() -> list[dict[str, Any]]:
    return [{"name": n, "description": s["description"],
             "params": s["params"], "backtests": len(s["backtests"])}
            for n, s in _STRATEGIES.items()]


def monte_carlo(trades: list[BacktestTrade] | list[float],
                n: int = 2000, seed: int = 42) -> dict[str, float]:
    """Reshuffle trade PnLs n times → P(profit>0). Voltra's 95% bar:
    don't go live below it."""
    import random
    rng = random.Random(seed)
    pnls = [t.pnl_pct if isinstance(t, BacktestTrade) else float(t)
            for t in trades]
    if len(pnls) < 10:
        return {"p_profit": 0.0, "n": n, "note": "need ≥10 trades"}
    wins = sum(1 for _ in range(n)
               if sum(rng.sample(pnls, len(pnls))) > 0)
    return {"p_profit": wins / n, "n": n,
            "mean_pct": sum(pnls) / len(pnls)}


def position_size(balance: float, risk_pct: float, entry: float,
                 stop: float) -> float:
    """Size so a stop-out loses exactly risk_pct of balance."""
    risk_amt = balance * risk_pct / 100
    per_unit = abs(entry - stop)
    if per_unit <= 0:
        raise SentinelError("stop must differ from entry")
    return risk_amt / per_unit


def graduated_capital(total: float) -> list[tuple[str, float]]:
    """freqtrade graduated deployment: 10% → 25% → 50% → 100%."""
    stages = [("stage-1 (week 1)", 0.10), ("stage-2 (weeks 2-4)", 0.25),
              ("stage-3 (month 2)", 0.50), ("stage-4 (month 3+)", 1.00)]
    return [(label, round(total * frac, 2)) for label, frac in stages]


# seed the registry with the reference strategies
register_strategy("sma_cross_10_30", sma_cross_strategy(),
                  description="SMA(10/30) cross — trend reference",
                  params={"fast": 10, "slow": 30})
register_strategy("rsi_mean_reversion", rsi_mean_reversion_strategy(),
                  description="RSI(14) mean reversion — range reference",
                  params={"period": 14, "oversold": 30,
                          "overbought": 70})
