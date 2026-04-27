"""Walk-forward allocation-based backtester for regime-driven strategies."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import RegimeStrategy, StrategyConfig, TechnicalSetup
from data.feature_engineering import compute_observable_features

LOGGER = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtester."""

    is_window: int = 504
    oos_window: int = 126
    step_size: int = 126
    slippage_pct: float = 0.0005
    rebalance_threshold: float = 0.10
    initial_capital: float = 100000.0
    commission_pct: float = 0.0
    fill_delay_bars: int = 1
    min_train_bars: int = 630


@dataclass
class AllocationBar:
    """Single bar allocation snapshot."""

    timestamp: pd.Timestamp
    price: float
    target_allocation: float
    current_allocation: float
    shares: float
    cash: float
    equity: float
    leverage: float
    regime_id: int
    regime_name: str
    regime_probability: float
    rebalanced: bool


@dataclass
class Trade:
    """Rebalancing event recorded as a "trade"."""

    entry_bar: int
    exit_bar: int
    entry_price: float
    exit_price: float
    entry_allocation: float
    exit_allocation: float
    shares_delta: float
    pnl: float
    pnl_pct: float
    holding_bars: int


class BacktestResult:
    """Results container for a single walk-forward window."""

    def __init__(
        self,
        window_index: int,
        is_start: int,
        is_end: int,
        oos_start: int,
        oos_end: int,
        config: BacktestConfig,
    ) -> None:
        """Initialize result container for walk-forward window."""
        self.window_index = window_index
        self.is_start = is_start
        self.is_end = is_end
        self.oos_start = oos_start
        self.oos_end = oos_end
        self.config = config

        self.bars: list[AllocationBar] = []
        self.trades: list[Trade] = []
        self.hmm_metadata: dict[str, Any] = {}
        self.equity_curve: list[float] = []
        self.returns: list[float] = []

    def to_dataframe(self) -> pd.DataFrame:
        """Convert bar snapshots to DataFrame."""
        if not self.bars:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "timestamp": [bar.timestamp for bar in self.bars],
                "price": [bar.price for bar in self.bars],
                "target_allocation": [bar.target_allocation for bar in self.bars],
                "current_allocation": [bar.current_allocation for bar in self.bars],
                "shares": [bar.shares for bar in self.bars],
                "cash": [bar.cash for bar in self.bars],
                "equity": [bar.equity for bar in self.bars],
                "leverage": [bar.leverage for bar in self.bars],
                "regime_id": [bar.regime_id for bar in self.bars],
                "regime_name": [bar.regime_name for bar in self.bars],
                "regime_probability": [bar.regime_probability for bar in self.bars],
                "rebalanced": [bar.rebalanced for bar in self.bars],
            }
        )


class WalkForwardBacktester:
    """Allocation-based walk-forward backtester for regime strategies."""

    def __init__(
        self,
        backtest_config: BacktestConfig | None = None,
        hmm_config: HMMConfig | None = None,
        strategy_config: StrategyConfig | None = None,
    ) -> None:
        """Initialize backtester with configurations."""
        self.backtest_config = backtest_config or BacktestConfig()
        self.hmm_config = hmm_config or HMMConfig(
            n_components=[2, 3, 4],
            cv_tol=1e-3,
            cv_max_iter=100,
            train_bars=126,
            stability_bars=5,
            flicker_window=40,
            flicker_threshold=2,
            min_confidence=0.95,
            min_train_bars=60,
        )
        self.strategy_config = strategy_config or StrategyConfig()

        self.results: list[BacktestResult] = []

    def run(self, price_data: pd.DataFrame) -> list[BacktestResult]:
        """Execute walk-forward backtesting on price data."""
        if price_data.empty or len(price_data) < self.backtest_config.min_train_bars:
            raise ValueError(
                f"Insufficient data. Need at least {self.backtest_config.min_train_bars} bars."
            )

        n_bars = len(price_data)
        window_index = 0
        is_start = 0

        while is_start + self.backtest_config.is_window + self.backtest_config.oos_window <= n_bars:
            is_end = is_start + self.backtest_config.is_window
            oos_start = is_end
            oos_end = oos_start + self.backtest_config.oos_window

            result = self._backtest_window(
                price_data=price_data,
                window_index=window_index,
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
            self.results.append(result)

            window_index += 1
            is_start += self.backtest_config.step_size

            LOGGER.info(
                "Completed walk-forward window %d: IS [%d:%d], OOS [%d:%d]",
                window_index - 1,
                is_start,
                is_end,
                oos_start,
                oos_end,
            )

        return self.results

    def _backtest_window(
        self,
        price_data: pd.DataFrame,
        window_index: int,
        is_start: int,
        is_end: int,
        oos_start: int,
        oos_end: int,
    ) -> BacktestResult:
        """Backtest a single walk-forward window."""
        result = BacktestResult(
            window_index=window_index,
            is_start=is_start,
            is_end=is_end,
            oos_start=oos_start,
            oos_end=oos_end,
            config=self.backtest_config,
        )

        is_data = price_data.iloc[is_start:is_end]
        oos_data = price_data.iloc[oos_start:oos_end]

        try:
            features = compute_observable_features(is_data)
            hmm = HMMEngine(config=self.hmm_config)
            hmm.fit(features)
            result.hmm_metadata = hmm.to_dict()
        except Exception as e:
            LOGGER.error("HMM training failed for window %d: %s", window_index, e)
            return result

        strategy = RegimeStrategy(config=self.strategy_config)
        self._simulate_oos(
            oos_data=oos_data,
            price_data=price_data,
            oos_start_idx=oos_start,
            hmm=hmm,
            strategy=strategy,
            result=result,
        )

        return result

    def _simulate_oos(
        self,
        oos_data: pd.DataFrame,
        price_data: pd.DataFrame,
        oos_start_idx: int,
        hmm: HMMEngine,
        strategy: RegimeStrategy,
        result: BacktestResult,
    ) -> None:
        """Simulate OOS period bar by bar with allocation tracking.

        PERFORMANCE FIX: Features and indicators are computed ONCE for the
        entire price_data range up to oos_end, then sliced per bar.
        This reduces complexity from O(n^2) to O(n).
        """
        equity = self.backtest_config.initial_capital
        shares = 0.0
        cash = equity
        pending_rebalance: dict[str, Any] | None = None

        oos_end_idx = oos_start_idx + len(oos_data)

        # ── PRECOMPUTE ALL FEATURES AND INDICATORS ONCE ───────
        full_slice = price_data.iloc[:oos_end_idx]
        all_features = compute_observable_features(full_slice)

        # Precompute technical indicators on full slice
        from data.market_data import MarketDataClient
        _mdc = MarketDataClient(data_source="csv")
        full_with_indicators = _mdc.compute_technical_indicators(full_slice.copy())

        if all_features.empty:
            LOGGER.warning("Feature precomputation returned empty for OOS window")
            return

        for i, (bar_idx, row) in enumerate(oos_data.iterrows()):
            price = float(row["close"])
            full_idx = oos_start_idx + i

            if full_idx < 200:
                continue

            # Slice precomputed features up to current bar (O(1) view)
            features_up_to_now = all_features.iloc[:full_idx + 1 - (len(full_slice) - len(all_features))]
            if features_up_to_now.empty:
                continue

            try:
                regime_state = hmm.predict_regime_filtered(features_up_to_now)
                regime_info = hmm.regime_info[regime_state.current_regime_id]

                # Read precomputed indicators at current bar
                bar_data = full_with_indicators.iloc[full_idx]
                ema_9 = float(bar_data.get("ema_9", price))
                ema_200 = float(bar_data.get("ema_200", price))
                vwap = float(bar_data.get("vwap", price))
                atr_14 = float(bar_data.get("atr", abs(float(row["high"]) - float(row["low"]))))
                swing_high = float(full_with_indicators["high"].iloc[max(0, full_idx - 20):full_idx + 1].max())
                swing_low = float(full_with_indicators["low"].iloc[max(0, full_idx - 20):full_idx + 1].min())
                distance_from_ema200 = price - ema_200

                # Check pullback from precomputed data
                from core.regime_strategies import detect_pullback, Direction
                lookback_df = full_with_indicators.iloc[max(0, full_idx - 6):full_idx + 1]
                had_pullback = detect_pullback(
                    lookback_df,
                    Direction.LONG if price > ema_200 else Direction.SHORT,
                    lookback=5,
                ) if len(lookback_df) > 5 else False

                setup = TechnicalSetup(
                    price=price,
                    ema_9=ema_9,
                    ema_200=ema_200,
                    vwap=vwap,
                    atr_14=atr_14,
                    swing_high=swing_high,
                    swing_low=swing_low,
                    distance_from_ema200=distance_from_ema200,
                    had_pullback=had_pullback,
                )

                signal = strategy.evaluate_signal(
                    symbol="ASSET",
                    regime_state=regime_state,
                    regime_info_map=hmm.regime_info,
                    setup=setup,
                    account_equity=equity,
                )

                target_allocation = 1.0 if signal.direction == "LONG" else (0.0 if signal.direction == "SHORT" else 0.0)
                current_allocation = (shares * price) / (equity + 1e-12)

                rebalanced = abs(target_allocation - current_allocation) > self.backtest_config.rebalance_threshold

                if rebalanced and pending_rebalance is None:
                    pending_rebalance = {
                        "price": price,
                        "target_allocation": target_allocation,
                        "regime_id": regime_state.current_regime_id,
                        "regime_name": regime_info.label,
                        "regime_probability": regime_state.state_probability,
                    }

            except Exception as e:
                LOGGER.debug("Feature/signal computation failed at bar %d: %s", full_idx, e)
                rebalanced = False
                regime_state = None
                regime_info = None
                signal = None

            if pending_rebalance is not None and i >= self.backtest_config.fill_delay_bars:
                target_shares = (equity * pending_rebalance["target_allocation"]) / (price + 1e-12)
                delta = target_shares - shares
                slippage_cost = abs(delta * price * self.backtest_config.slippage_pct)
                cash -= delta * price + slippage_cost
                shares = target_shares

                if result.trades:
                    last_trade = result.trades[-1]
                    last_trade.exit_price = pending_rebalance["price"]
                    last_trade.exit_allocation = pending_rebalance["target_allocation"]
                    last_trade.exit_bar = len(result.bars) - 1
                    last_trade.holding_bars = last_trade.exit_bar - last_trade.entry_bar
                    last_trade.pnl = (shares * price - last_trade.shares_delta * pending_rebalance["price"])
                    last_trade.pnl_pct = (
                        (last_trade.pnl / (abs(last_trade.shares_delta * last_trade.entry_price) + 1e-12))
                        if last_trade.shares_delta != 0
                        else 0.0
                    )

                result.trades.append(
                    Trade(
                        entry_bar=len(result.bars),
                        exit_bar=-1,
                        entry_price=pending_rebalance["price"],
                        exit_price=0.0,
                        entry_allocation=pending_rebalance["target_allocation"],
                        exit_allocation=0.0,
                        shares_delta=delta,
                        pnl=0.0,
                        pnl_pct=0.0,
                        holding_bars=0,
                    )
                )
                pending_rebalance = None

            equity = cash + shares * price
            current_alloc = (shares * price) / (equity + 1e-12)

            bar = AllocationBar(
                timestamp=row.name if hasattr(row, "name") else pd.Timestamp(full_idx),
                price=price,
                target_allocation=pending_rebalance["target_allocation"]
                if pending_rebalance
                else current_alloc,
                current_allocation=current_alloc,
                shares=shares,
                cash=cash,
                equity=equity,
                leverage=current_alloc if current_alloc <= 1.0 else 1.0,
                regime_id=regime_state.current_regime_id if regime_state else -1,
                regime_name=regime_info.label if regime_info else "UNKNOWN",
                regime_probability=regime_state.state_probability if regime_state else 0.0,
                rebalanced=rebalanced,
            )
            result.bars.append(bar)
            result.equity_curve.append(equity)

            if len(result.equity_curve) > 1:
                result.returns.append(
                    (result.equity_curve[-1] - result.equity_curve[-2]) / (result.equity_curve[-2] + 1e-12)
                )

    @staticmethod
    def _compute_ema(prices: pd.Series, period: int) -> float:
        """Compute exponential moving average."""
        if len(prices) < period:
            return float(prices.iloc[-1])
        return float(prices.ewm(span=period, adjust=False).mean().iloc[-1])

    @staticmethod
    def _compute_vwap(ohlcv: pd.DataFrame) -> float:
        """Compute volume-weighted average price."""
        if len(ohlcv) < 1:
            return float(ohlcv["close"].iloc[-1])
        typical_price = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0
        cum_pv = (typical_price * ohlcv["volume"]).sum()
        cum_v = ohlcv["volume"].sum()
        return float(cum_pv / (cum_v + 1e-12))

    @staticmethod
    def _compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
        """Compute average true range."""
        if len(ohlcv) < period:
            return float(ohlcv["high"].iloc[-1] - ohlcv["low"].iloc[-1])
        
        high_low = ohlcv["high"] - ohlcv["low"]
        high_close = abs(ohlcv["high"] - ohlcv["close"].shift())
        low_close = abs(ohlcv["low"] - ohlcv["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return float(atr.iloc[-1]) if not atr.isna().all() else float(tr.iloc[-1])


# Backward-compatible aliases.
Backtester = WalkForwardBacktester
