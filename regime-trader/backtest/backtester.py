"""Walk-forward allocation-based backtester for regime-driven strategies."""

from __future__ import annotations

import logging
from bisect import bisect_right, insort
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
    """Configuration for walk-forward backtester.

    Default windows are calibrated for M5 bars:
      - 1 trading day  = 288 M5 bars  (24h * 60min / 5min)
      - is_window      = 14400 bars   (~50 trading days)
      - oos_window     = 2880 bars    (~10 trading days)
      - step_size      = 2880 bars    (~10 trading days)
      - min_train_bars = 8640 bars    (~30 trading days)
    """

    is_window: int = 14400
    oos_window: int = 2880
    step_size: int = 2880
    slippage_pct: float = 0.0005
    rebalance_threshold: float = 0.10
    initial_capital: float = 100000.0
    commission_pct: float = 0.0
    fill_delay_bars: int = 1
    min_train_bars: int = 8640


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
    exit_reason: str = ""
    direction: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0


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
            cv_max_iter=200,
            train_bars=14400,        # Use the full IS window
            stability_bars=12,       # 12 consecutive M5 bars = 1 hour
            flicker_window=120,      # ~10 hours lookback
            flicker_threshold=3,
            min_confidence=0.90,
            retrain_interval_bars=288,
            min_train_bars=8640,     # ~30 trading days of M5
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

        PERFORMANCE FIX v2: Features are computed ONCE. Regime states are
        computed in a SINGLE forward-filter pass for the entire OOS window.
        Per-bar work is O(1) lookups into precomputed arrays.
        Total complexity: O(n) instead of O(n^2).
        """
        equity = self.backtest_config.initial_capital
        shares = 0.0
        cash = equity

        oos_end_idx = oos_start_idx + len(oos_data)

        # ── STEP 1: PRECOMPUTE ALL FEATURES ONCE ──────────────
        full_slice = price_data.iloc[:oos_end_idx]
        all_features = compute_observable_features(full_slice)

        if all_features.empty:
            LOGGER.warning("Feature precomputation returned empty for OOS window")
            return

        # ── STEP 2: PRECOMPUTE TECHNICAL INDICATORS ONCE ──────
        from data.market_data import MarketDataClient
        _mdc = MarketDataClient(data_source="csv")
        full_with_indicators = _mdc.compute_technical_indicators(full_slice.copy())

        required_cols = ["ema_9", "ema_200", "vwap", "atr", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in full_with_indicators.columns]
        if missing_cols:
            LOGGER.error(
                "CRITICAL: Missing indicator columns: %s. Available: %s",
                missing_cols,
                full_with_indicators.columns.tolist(),
            )
            return

        # ── STEP 2b: PRECOMPUTE ALL ROLLING VOLATILITY STDS ONCE ──
        # This avoids O(n²) recomputation in the loop
        all_rolling_stds = full_with_indicators["close"].rolling(20).std()
        all_vol_percentiles = self._compute_causal_percentiles(all_rolling_stds)

        # ── STEP 3: BATCH REGIME INFERENCE (SINGLE FORWARD PASS) ──
        # Run the forward filter ONCE over ALL features and get every
        # bar's regime state.  This replaces the per-bar call that was
        # causing O(n^2) behaviour.
        all_regime_states = hmm.predict_regime_series_filtered(all_features)
        n_features = len(all_features)

        # Map from full_slice index -> feature index offset
        feat_offset = len(full_slice) - n_features

        from core.regime_strategies import detect_pullback, Direction

        signal_stats = {
            "total_bars": 0,
            "skipped_no_regime": 0,
            "chop_filtered": 0,
            "no_macro": 0,
            "exhausted": 0,
            "no_pullback": 0,
            "long_signals": 0,
            "short_signals": 0,
            "blocked_active_trade": 0,
        }

        active_trade: dict[str, Any] | None = None

        LOGGER.info(
            "OOS simulation: %d bars, %d features, %d regime states precomputed",
            len(oos_data), n_features, len(all_regime_states),
        )

        # CONFIG CHECK: Ensure strategy config values are flowing into simulator
        try:
            LOGGER.info(
                "CONFIG CHECK: swing_lookback=%d, rr_ratio=%.1f, risk_pct=%.3f",
                strategy.config.swing_lookback,
                strategy.config.rr_ratio,
                strategy.config.risk_percent,
            )
        except Exception:
            LOGGER.warning("CONFIG CHECK: unable to read strategy.config values")

        first_trade_logged = False

        for i, (bar_idx, row) in enumerate(oos_data.iterrows()):
            price = float(row["close"])
            full_idx = oos_start_idx + i
            signal_stats["total_bars"] += 1

            if full_idx < 200:
                continue

            # Index into precomputed regime states
            feat_idx = full_idx - feat_offset
            if feat_idx < 0 or feat_idx >= len(all_regime_states):
                continue

            try:
                regime_state = all_regime_states[feat_idx]
                regime_info = hmm.regime_info[regime_state.current_regime_id]

                exit_triggered = False
                exit_reason = ""
                exit_price = price

                if active_trade is not None:
                    if active_trade["direction"] == "LONG":
                        if price <= active_trade["stop_loss"]:
                            exit_triggered = True
                            exit_reason = "SL_HIT"
                            exit_price = active_trade["stop_loss"]
                        elif price >= active_trade["take_profit"]:
                            exit_triggered = True
                            exit_reason = "TP_HIT"
                            exit_price = active_trade["take_profit"]
                    else:
                        if price >= active_trade["stop_loss"]:
                            exit_triggered = True
                            exit_reason = "SL_HIT"
                            exit_price = active_trade["stop_loss"]
                        elif price <= active_trade["take_profit"]:
                            exit_triggered = True
                            exit_reason = "TP_HIT"
                            exit_price = active_trade["take_profit"]

                    if exit_triggered:
                        qty = float(active_trade["lots"])
                        trade_direction = active_trade["direction"]
                        entry_slippage = float(active_trade.get("entry_slippage", 0.0))
                        exit_slippage = abs(qty * exit_price * self.backtest_config.slippage_pct)
                        total_slippage = entry_slippage + exit_slippage

                        if trade_direction == "LONG":
                            cash += qty * exit_price - exit_slippage
                            shares -= qty
                            pnl = (exit_price - float(active_trade["entry_price"])) * qty - total_slippage
                        else:
                            cash -= qty * exit_price + exit_slippage
                            shares += qty
                            pnl = (float(active_trade["entry_price"]) - exit_price) * qty - total_slippage

                        equity = cash + shares * price

                        if result.trades:
                            last_trade = result.trades[-1]
                            last_trade.exit_price = exit_price
                            last_trade.exit_bar = len(result.bars)
                            last_trade.holding_bars = last_trade.exit_bar - last_trade.entry_bar
                            last_trade.pnl = pnl
                            last_trade.pnl_pct = pnl / (
                                abs(float(active_trade["entry_price"]) * qty) + 1e-12
                            )
                            last_trade.exit_reason = exit_reason

                        active_trade = None
                        rebalanced = True
                    else:
                        rebalanced = False
                else:
                    rebalanced = False

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
                lookback_df = full_with_indicators.iloc[max(0, full_idx - 6):full_idx + 1]
                had_pullback = detect_pullback(
                    lookback_df,
                    Direction.LONG if price > ema_200 else Direction.SHORT,
                    lookback=5,
                ) if len(lookback_df) > 5 else False

                volatility_percentile = float(all_vol_percentiles.iloc[full_idx]) if full_idx < len(all_vol_percentiles) else 0.5

                setup = TechnicalSetup(
                    price=price,
                    ema_9=ema_9,
                    ema_200=ema_200,
                    vwap=vwap,
                    atr_14=atr_14,
                    swing_high=swing_high,
                    swing_low=swing_low,
                    distance_from_ema200=distance_from_ema200,
                    volatility_percentile=volatility_percentile,
                    had_pullback=had_pullback,
                )

                signal = strategy.evaluate_signal(
                    symbol="ASSET",
                    regime_state=regime_state,
                    regime_info_map=hmm.regime_info,
                    setup=setup,
                    account_equity=equity,
                )



                reasoning = signal.reasoning.lower() if signal else ""
                if signal is None:
                    signal_stats["skipped_no_regime"] += 1
                elif signal.direction == "LONG":
                    signal_stats["long_signals"] += 1
                elif signal.direction == "SHORT":
                    signal_stats["short_signals"] += 1
                elif "chop" in reasoning:
                    signal_stats["chop_filtered"] += 1
                elif "no clear macro" in reasoning or "macro" in reasoning:
                    signal_stats["no_macro"] += 1
                elif "exhaust" in reasoning:
                    signal_stats["exhausted"] += 1
                elif "pullback" in reasoning or "no trigger" in reasoning:
                    signal_stats["no_pullback"] += 1
                else:
                    # Catch any signal that doesn't match above patterns
                    LOGGER.warning(f"Unclassified signal at bar {full_idx}: dir={signal.direction if signal else 'None'}, reason={reasoning[:60]}")

                if active_trade is not None and not exit_triggered and signal.direction not in ("FLAT", ""):
                    signal_stats["blocked_active_trade"] += 1

                if active_trade is None and not exit_triggered and signal.direction in ("LONG", "SHORT"):
                    qty = float(signal.lots if hasattr(signal, "lots") else 1.0)
                    if qty > 0:
                        next_bar_idx = full_idx + 1
                        if next_bar_idx < len(full_with_indicators):
                            fill_price = float(full_with_indicators["open"].iloc[next_bar_idx])
                        else:
                            fill_price = price

                        entry_slippage = abs(qty * fill_price * self.backtest_config.slippage_pct)
                        
                        # Fix 7: Log SL/TP distances for validation
                        sl_distance = abs(fill_price - float(signal.stop_loss))
                        tp_distance = abs(float(signal.take_profit) - fill_price)
                        actual_rr = tp_distance / (sl_distance + 1e-12)
                        # Log first trade details (explicit) for debugging config propagation
                        try:
                            if not first_trade_logged:
                                LOGGER.info(
                                    "FIRST TRADE: price=%.2f SL=%.2f TP=%.2f swing_high=%.2f swing_low=%.2f",
                                    price,
                                    float(signal.stop_loss),
                                    float(signal.take_profit),
                                    setup.swing_high,
                                    setup.swing_low,
                                )
                                first_trade_logged = True
                        except Exception:
                            LOGGER.warning("FIRST TRADE: unable to log full setup details")

                        LOGGER.info(
                            "TRADE OPEN: %s @ %.2f | SL=%.2f (dist=%.1f pts) | TP=%.2f (dist=%.1f pts) | ATR=%.1f | RR=%.2f",
                            signal.direction, fill_price,
                            float(signal.stop_loss), sl_distance,
                            float(signal.take_profit), tp_distance,
                            setup.atr_14,
                            actual_rr,
                        )
                        
                        if signal.direction == "LONG":
                            cash -= qty * fill_price + entry_slippage
                            shares += qty
                        else:
                            cash += qty * fill_price - entry_slippage
                            shares -= qty

                        active_trade = {
                            "direction": signal.direction,
                            "entry_price": fill_price,
                            "stop_loss": float(signal.stop_loss),
                            "take_profit": float(signal.take_profit),
                            "entry_bar": len(result.bars),
                            "lots": qty,
                            "entry_slippage": entry_slippage,
                        }
                        result.trades.append(
                            Trade(
                                entry_bar=len(result.bars),
                                exit_bar=-1,
                                entry_price=fill_price,
                                exit_price=0.0,
                                entry_allocation=1.0 if signal.direction == "LONG" else -1.0,
                                exit_allocation=0.0,
                                shares_delta=qty if signal.direction == "LONG" else -qty,
                                pnl=0.0,
                                pnl_pct=0.0,
                                holding_bars=0,
                                exit_reason="OPEN",
                                direction=signal.direction,
                                stop_loss=float(signal.stop_loss),
                                take_profit=float(signal.take_profit),
                            )
                        )
                        rebalanced = True

            except Exception as e:
                LOGGER.warning("Feature/signal computation failed at bar %d: %s", full_idx, e)
                rebalanced = False
                regime_state = None
                regime_info = None

            equity = cash + shares * price
            current_alloc = (shares * price) / (equity + 1e-12)
            target_alloc = 0.0
            if active_trade is not None:
                target_alloc = 1.0 if active_trade["direction"] == "LONG" else -1.0

            bar = AllocationBar(
                timestamp=row.name if hasattr(row, "name") else pd.Timestamp(full_idx),
                price=price,
                target_allocation=target_alloc,
                current_allocation=current_alloc,
                shares=shares,
                cash=cash,
                equity=equity,
                leverage=min(abs(current_alloc), 1.0),
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

        if active_trade is not None:
            final_price = float(oos_data.iloc[-1]["close"])
            qty = float(active_trade["lots"])
            trade_direction = active_trade["direction"]
            entry_slippage = float(active_trade.get("entry_slippage", 0.0))
            exit_slippage = abs(qty * final_price * self.backtest_config.slippage_pct)
            total_slippage = entry_slippage + exit_slippage

            if trade_direction == "LONG":
                cash += qty * final_price - exit_slippage
                shares -= qty
                pnl = (final_price - float(active_trade["entry_price"])) * qty - total_slippage
            else:
                cash -= qty * final_price + exit_slippage
                shares += qty
                pnl = (float(active_trade["entry_price"]) - final_price) * qty - total_slippage

            equity = cash + shares * final_price

            if result.trades:
                last_trade = result.trades[-1]
                last_trade.exit_price = final_price
                last_trade.exit_bar = len(result.bars) - 1
                last_trade.holding_bars = last_trade.exit_bar - last_trade.entry_bar
                last_trade.pnl = pnl
                last_trade.pnl_pct = pnl / (abs(float(active_trade["entry_price"]) * qty) + 1e-12)
                last_trade.exit_reason = "END_OF_WINDOW"

            active_trade = None

        LOGGER.info("Signal breakdown for OOS window:")
        LOGGER.info("  Total bars evaluated: %d", signal_stats["total_bars"])
        LOGGER.info(
            "  Skipped/no regime:    %d (%.1f%%)",
            signal_stats["skipped_no_regime"],
            100 * signal_stats["skipped_no_regime"] / max(signal_stats["total_bars"], 1),
        )
        LOGGER.info(
            "  Blocked (in trade):    %d",
            signal_stats["blocked_active_trade"],
        )
        LOGGER.info(
            "  CHOP filtered:        %d (%.1f%%)",
            signal_stats["chop_filtered"],
            100 * signal_stats["chop_filtered"] / max(signal_stats["total_bars"], 1),
        )
        LOGGER.info("  No macro alignment:   %d", signal_stats["no_macro"])
        LOGGER.info("  Exhaustion filtered:  %d", signal_stats["exhausted"])
        LOGGER.info("  No pullback:          %d", signal_stats["no_pullback"])
        LOGGER.info("  LONG signals:         %d", signal_stats["long_signals"])
        LOGGER.info("  SHORT signals:        %d", signal_stats["short_signals"])
        LOGGER.info("  Blocked (in trade):    %d", signal_stats["blocked_active_trade"])

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

    @staticmethod
    def _compute_causal_percentiles(series: pd.Series) -> pd.Series:
        """Compute causal percentiles of a series using only past observations."""
        values = series.to_numpy(dtype=float)
        percentiles = np.full(len(values), 0.5, dtype=float)
        history: list[float] = []

        for idx, value in enumerate(values):
            if np.isnan(value):
                continue
            if history:
                rank = bisect_right(history, value)
                percentiles[idx] = rank / len(history)
            history.append(float(value))
            history.sort()

        return pd.Series(percentiles, index=series.index)


# Backward-compatible aliases.
Backtester = WalkForwardBacktester
