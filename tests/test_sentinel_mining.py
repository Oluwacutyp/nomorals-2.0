"""Tests for Sentinel.py mining integration: adaptive, macro, lessons."""

import pytest


class TestAdaptive:
    def test_volatility_breath(self):
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")
        from nomorals.ta.adaptive import VolatilityBreath
        import pandas as pd
        import numpy as np

        # Synthetic: calm then volatile
        np.random.seed(42)
        calm = 100 + np.cumsum(np.random.randn(200) * 0.1)
        vol = 100 + np.cumsum(np.random.randn(200) * 1.0)
        close = pd.Series(list(calm) + list(vol))

        vb = VolatilityBreath()
        scale = vb.scale(close)
        # Volatile period should have higher scale than calm
        assert scale.iloc[-1] > scale.iloc[100]
        assert (scale >= 0.5).all() and (scale <= 2.0).all()

    def test_dynamic_period(self):
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")
        from nomorals.ta.adaptive import VolatilityBreath
        import pandas as pd
        import numpy as np

        np.random.seed(1)
        close = pd.Series(100 + np.cumsum(np.random.randn(400)))
        vb = VolatilityBreath()
        periods = vb.dynamic_period(close, base=14)
        assert (periods >= 2).all() and (periods <= 500).all()

    def test_quantile_gate(self):
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")
        from nomorals.ta.adaptive import QuantileGate
        import pandas as pd
        import numpy as np

        np.random.seed(2)
        s = pd.Series(np.random.randn(400).cumsum() + 100)
        qg = QuantileGate()
        pos = qg.position(s)
        assert (pos >= -1).all() and (pos <= 1).all()
        # Extreme high should be near +1
        assert pos.iloc[-1] > -1  # sanity

    def test_signal_rate_tuner(self):
        pytest.importorskip("numpy")
        from nomorals.ta.adaptive import SignalRateTuner

        tuner = SignalRateTuner(target_rate=0.05)
        # Over-trading: rapid flips
        flips = [1, -1] * 50
        m1 = tuner.update(flips)
        assert m1 > 1.0  # gate rises to reduce rate

        tuner2 = SignalRateTuner(target_rate=0.05)
        # Starving: no flips
        calm = [1] * 100
        m2 = tuner2.update(calm)
        assert m2 < 1.0  # gate falls to allow more

    def test_drawdown_throttle(self):
        pytest.importorskip("numpy")
        from nomorals.ta.adaptive import DrawdownThrottle

        dt = DrawdownThrottle(soft_dd=0.05, hard_dd=0.15)
        # No drawdown → full size
        assert dt.factor([100, 101, 102]) == 1.0
        # Deep drawdown → zero
        assert dt.factor([100, 90, 85]) == 0.0
        # Between → partial
        mid = dt.factor([100, 95, 93])
        assert 0.0 < mid < 1.0


class TestMacro:
    def test_trinity_bias_bullish(self):
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")
        from nomorals.ta.macro import trinity_bias
        import numpy as np

        # DXY falling, yields falling → bullish gold
        n = 60
        xau = [2000 + i * 2 for i in range(n)]
        dxy = [105 - i * 0.1 for i in range(n)]  # falling
        tny = [4.5 - i * 0.01 for i in range(n)]  # falling
        result = trinity_bias(xau, dxy, tny)
        assert result is not None
        assert result["bias"] == "bullish"
        assert result["dxy_trend"] < -0.2

    def test_trinity_bias_bearish(self):
        pytest.importorskip("numpy")
        pytest.importorskip("pandas")
        from nomorals.ta.macro import trinity_bias

        n = 60
        xau = [2000 - i * 2 for i in range(n)]
        dxy = [100 + i * 0.1 for i in range(n)]  # rising
        tny = [4.0 + i * 0.01 for i in range(n)]  # rising
        result = trinity_bias(xau, dxy, tny)
        assert result is not None
        assert result["bias"] == "bearish"

    def test_trinity_insufficient_data(self):
        from nomorals.ta.macro import trinity_bias

        assert trinity_bias([1, 2, 3], [1, 2, 3], [1, 2, 3]) is None


class TestLessons:
    def test_xau_blocked(self):
        from nomorals.ta.lessons import check_lesson

        result = check_lesson("XAUUSD", "1h")
        assert result["verdict"] == "DO NOT TRADE"
        assert "lesson_id" in result

    def test_xau_15m_blocked(self):
        from nomorals.ta.lessons import check_lesson

        result = check_lesson("XAUUSD", "15m")
        assert result["verdict"] == "DO NOT TRADE"

    def test_btc_1h_caution(self):
        from nomorals.ta.lessons import check_lesson

        result = check_lesson("BTCUSD", "1h")
        assert "CAUTION" in result["verdict"]

    def test_unknown_symbol(self):
        from nomorals.ta.lessons import check_lesson

        result = check_lesson("EURUSD", "1h")
        # Universal RULE lessons (sizing, no-tuning) apply to all symbols
        assert result["verdict"] in ("NO DATA", "RULE")

    def test_executor_refuses_xau(self):
        from nomorals.ta.analyst import TradeIdea, executor_check

        idea = TradeIdea(
            symbol="XAUUSD",
            timeframe="1h",
            side=1,
            confidence=75.0,
            entry=2000.0,
            invalidation=1990.0,
            targets=[(2020.0, 1.0)],
        )
        result = executor_check(idea)
        assert result["ok"] is False
        assert "backtest lesson" in result["reason"]

    def test_executor_allows_unknown(self):
        from nomorals.ta.analyst import TradeIdea, executor_check

        idea = TradeIdea(
            symbol="EURUSD",
            timeframe="1h",
            side=1,
            confidence=75.0,
            entry=1.10,
            invalidation=1.09,
            targets=[(1.12, 1.0)],
        )
        result = executor_check(idea)
        assert result["ok"] is True
