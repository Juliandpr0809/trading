"""Walk-forward session-based backtester for the Liquidity Sweep strategy.

Processes day-by-day (not bar-by-bar). Each day is one opportunity.
Supports trailing SL at 1:1 RR and forced exit at session end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import time as dt_time
from typing import Any

import numpy as np
import pandas as pd

from core.liquidity_strategy import (
    LiquidityStrategy,
    LiquidityConfig,
    Signal,
)
from monitoring.paper_trading_logger import PaperTradingLogger

LOGGER = logging.getLogger(__name__)


@dataclass
class SessionBacktestConfig:
    """Walk-forward windows measured in trading DAYS (not bars)."""
    is_window_days: int = 500
    oos_window_days: int = 60
    step_size_days: int = 60
    min_train_days: int = 250
    initial_capital: float = 100_000.0
    slippage_points: float = 2.0
    enable_paper_logging: bool = False  # Log each session to paper_trading_log.csv


@dataclass
class SessionTrade:
    """Single trade record from a backtested session."""
    date: pd.Timestamp
    direction: str
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    lots: float
    pnl: float
    pnl_pct: float
    exit_reason: str
    entry_candle_high: float = 0.0
    entry_candle_low: float = 0.0
    trailing_triggered: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionBacktestResult:
    """Results for one walk-forward window."""
    window_index: int
    is_start_date: pd.Timestamp
    is_end_date: pd.Timestamp
    oos_start_date: pd.Timestamp
    oos_end_date: pd.Timestamp
    trades: list[SessionTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    daily_log: list[dict[str, Any]] = field(default_factory=list)


class WalkForwardSessionBacktester:
    """Walk-forward backtester for daily-session-based strategies."""

    def __init__(
        self,
        backtest_config: SessionBacktestConfig | None = None,
        strategy_config: LiquidityConfig | None = None,
    ) -> None:
        self.bt_config = backtest_config or SessionBacktestConfig()
        self.strat_config = strategy_config or LiquidityConfig()
        self.paper_logger = (
            PaperTradingLogger() if self.bt_config.enable_paper_logging else None
        )

    def run(
        self,
        m30_data: pd.DataFrame,
        m5_data: pd.DataFrame,
    ) -> list[SessionBacktestResult]:
        """Execute walk-forward backtesting on M30+M5 data."""
        # Get unique trading days from M5 data
        trading_days = sorted(m5_data.index.normalize().unique())
        n_days = len(trading_days)

        LOGGER.info("BACKTESTER: %d trading days from %s to %s",
                     n_days, trading_days[0].date(), trading_days[-1].date())

        min_required = self.bt_config.is_window_days + self.bt_config.oos_window_days
        if n_days < min_required:
            raise ValueError(
                f"Insufficient data: {n_days} days, need {min_required}. "
                f"IS={self.bt_config.is_window_days}, OOS={self.bt_config.oos_window_days}"
            )

        results: list[SessionBacktestResult] = []
        window_index = 0
        is_start_idx = 0

        while is_start_idx + self.bt_config.is_window_days + self.bt_config.oos_window_days <= n_days:
            is_end_idx = is_start_idx + self.bt_config.is_window_days
            oos_start_idx = is_end_idx
            oos_end_idx = min(oos_start_idx + self.bt_config.oos_window_days, n_days)

            is_start = trading_days[is_start_idx]
            is_end = trading_days[is_end_idx - 1]
            oos_start = trading_days[oos_start_idx]
            oos_end = trading_days[oos_end_idx - 1]

            LOGGER.info(
                "WINDOW %d: IS [%s to %s] (%d days) | OOS [%s to %s] (%d days)",
                window_index,
                is_start.date(), is_end.date(), is_end_idx - is_start_idx,
                oos_start.date(), oos_end.date(), oos_end_idx - oos_start_idx,
            )

            result = self._backtest_window(
                m30_data=m30_data,
                m5_data=m5_data,
                window_index=window_index,
                oos_days=trading_days[oos_start_idx:oos_end_idx],
                is_start=is_start,
                is_end=is_end,
                oos_start=oos_start,
                oos_end=oos_end,
            )
            results.append(result)

            window_index += 1
            is_start_idx += self.bt_config.step_size_days

        LOGGER.info("BACKTESTER: completed %d walk-forward windows", len(results))
        return results

    def _backtest_window(
        self,
        m30_data: pd.DataFrame,
        m5_data: pd.DataFrame,
        window_index: int,
        oos_days: list[pd.Timestamp],
        is_start: pd.Timestamp,
        is_end: pd.Timestamp,
        oos_start: pd.Timestamp,
        oos_end: pd.Timestamp,
    ) -> SessionBacktestResult:
        """Backtest a single OOS window day by day."""
        result = SessionBacktestResult(
            window_index=window_index,
            is_start_date=is_start,
            is_end_date=is_end,
            oos_start_date=oos_start,
            oos_end_date=oos_end,
        )

        strategy = LiquidityStrategy(config=self.strat_config)
        equity = self.bt_config.initial_capital

        for day in oos_days:
            signal = strategy.evaluate_session(
                m30_data=m30_data,
                m5_data=m5_data,
                session_date=day,
                account_equity=equity,
            )

            if signal is None:
                result.daily_log.append({
                    "date": day, "action": "NO_SIGNAL", "equity": equity,
                })
                result.equity_curve.append(equity)
                
                # Log no setup to paper trading log
                if self.paper_logger:
                    self.paper_logger.log_no_setup_today(
                        session_date=day,
                        reason="No valid signal from strategy",
                    )
                continue

            # Simulate the trade using M5 bars
            trade = self._simulate_trade(m5_data, signal, day, equity)
            if trade is not None:
                equity += trade.pnl
                result.trades.append(trade)
                result.daily_log.append({
                    "date": day,
                    "action": f"{trade.direction} -> {trade.exit_reason}",
                    "pnl": trade.pnl,
                    "equity": equity,
                })
                
                # Log setup to paper trading log
                if self.paper_logger:
                    sl_distance = abs(signal.entry_price - signal.stop_loss)
                    tp_distance = abs(signal.take_profit - signal.entry_price)
                    poi_type = "BEARISH" if signal.direction == "SHORT" else "BULLISH"
                    
                    self.paper_logger.log_setup_detected(
                        session_date=day,
                        signal=signal,
                        poi_type=poi_type,
                        sweep_level=signal.metadata.get("sweep_level", 0.0),
                        pattern_candle_a=signal.metadata.get("candle_a_time", ""),
                        pattern_candle_b=signal.metadata.get("candle_b_time", ""),
                        sl_distance=sl_distance,
                        tp_distance=tp_distance,
                        notes=f"{trade.exit_reason}: PnL=${trade.pnl:,.2f}",
                    )

            result.equity_curve.append(equity)

        # Log window summary
        n_trades = len(result.trades)
        if n_trades > 0:
            wins = sum(1 for t in result.trades if t.pnl > 0)
            total_pnl = sum(t.pnl for t in result.trades)
            LOGGER.info(
                "WINDOW %d DONE: %d trades | %d wins (%.1f%%) | PnL=$%.2f | "
                "Final equity=$%.2f",
                window_index, n_trades, wins,
                100 * wins / n_trades, total_pnl, equity,
            )
        else:
            LOGGER.info("WINDOW %d DONE: 0 trades", window_index)

        return result

    def _simulate_trade(
        self,
        m5_data: pd.DataFrame,
        signal: Signal,
        session_date: pd.Timestamp,
        equity: float,
    ) -> SessionTrade | None:
        """Simulate a single trade bar-by-bar on M5 data."""
        session_end = pd.Timestamp.combine(
            session_date.date(), self.strat_config.session_end_time
        )

        # Get M5 bars from entry candle onward within session
        entry_time = signal.timestamp
        if isinstance(entry_time, pd.Timestamp):
            entry_ts = entry_time
        else:
            entry_ts = pd.Timestamp(entry_time)

        sim_bars = m5_data[
            (m5_data.index > entry_ts) & (m5_data.index <= session_end)
        ]

        if sim_bars.empty:
            LOGGER.info("SIMULATE: no bars after entry for simulation")
            return None

        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        direction = signal.direction
        lots = signal.lots
        slippage = self.bt_config.slippage_points

        # Apply slippage to entry
        if direction == "LONG":
            entry_price += slippage
        else:
            entry_price -= slippage

        # Breakout candle info for trailing SL
        brk_high = signal.metadata.get("breakout_candle_high", entry_price)
        brk_low = signal.metadata.get("breakout_candle_low", entry_price)

        # Calculate 1:1 RR level for trailing
        sl_distance = abs(entry_price - stop_loss)
        if direction == "LONG":
            trailing_level = entry_price + sl_distance * self.strat_config.trailing_at_rr
        else:
            trailing_level = entry_price - sl_distance * self.strat_config.trailing_at_rr

        trailing_triggered = False
        current_sl = stop_loss
        exit_price = 0.0
        exit_reason = ""

        for bar_time, bar in sim_bars.iterrows():
            bar_high = float(bar["high"])
            bar_low = float(bar["low"])
            bar_close = float(bar["close"])

            if direction == "LONG":
                # Check SL hit (use bar low for intrabar)
                if bar_low <= current_sl:
                    exit_price = current_sl
                    exit_reason = "TRAILING_EXIT" if trailing_triggered else "SL_HIT"
                    break
                # Check TP hit (use bar high for intrabar)
                if bar_high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP_HIT"
                    break
                # Check trailing trigger at 1:1
                if not trailing_triggered and bar_high >= trailing_level:
                    trailing_triggered = True
                    new_sl = brk_low  # SL to breakout candle low
                    if new_sl > current_sl:  # Never move SL backward
                        current_sl = new_sl
                        LOGGER.debug("TRAILING: SL moved to %.2f (breakout candle low) at %s",
                                     current_sl, bar_time)
            else:  # SHORT
                # Check SL hit
                if bar_high >= current_sl:
                    exit_price = current_sl
                    exit_reason = "TRAILING_EXIT" if trailing_triggered else "SL_HIT"
                    break
                # Check TP hit
                if bar_low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP_HIT"
                    break
                # Check trailing trigger at 1:1
                if not trailing_triggered and bar_low <= trailing_level:
                    trailing_triggered = True
                    new_sl = brk_high  # SL to breakout candle high
                    if new_sl < current_sl:  # Never move SL backward
                        current_sl = new_sl
                        LOGGER.debug("TRAILING: SL moved to %.2f (breakout candle high) at %s",
                                     current_sl, bar_time)

        # If we exit the loop without hitting SL/TP → timeout at session end
        if not exit_reason:
            exit_price = float(sim_bars.iloc[-1]["close"])
            exit_reason = "TIMEOUT_2030"

        # Calculate PnL
        if direction == "LONG":
            pnl_points = exit_price - entry_price
        else:
            pnl_points = entry_price - exit_price

        pnl = pnl_points * lots * self.strat_config.contract_size
        pnl_pct = pnl / (equity + 1e-12)

        LOGGER.info(
            "TRADE: %s | entry=%.2f exit=%.2f | SL=%.2f TP=%.2f | "
            "PnL=%.2f pts ($%.2f) | %s%s",
            direction, entry_price, exit_price,
            signal.stop_loss, take_profit,
            pnl_points, pnl, exit_reason,
            " [TRAILING]" if trailing_triggered else "",
        )

        LOGGER.info(
            "TRADE DETAIL: date=%s | direction=%s | entry=%.2f | SL=%.2f | TP=%.2f | "
            "exit_reason=%s | poi_type=%s | sweep_level=%.2f | pattern_candle_A=%s | "
            "pattern_candle_B=%s",
            session_date.strftime("%Y-%m-%d"),
            direction,
            entry_price,
            signal.stop_loss,
            take_profit,
            exit_reason,
            signal.metadata.get("poi_type", "N/A"),
            float(signal.metadata.get("sweep_level", 0.0)),
            signal.metadata.get("pattern_candle_a", "N/A"),
            signal.metadata.get("pattern_candle_b", "N/A"),
        )

        return SessionTrade(
            date=session_date,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=signal.stop_loss,
            take_profit=take_profit,
            lots=lots,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=exit_reason,
            entry_candle_high=brk_high,
            entry_candle_low=brk_low,
            trailing_triggered=trailing_triggered,
            metadata=signal.metadata,
        )
