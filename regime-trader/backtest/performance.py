"""Performance analytics for backtest and live-trading results."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)
TRADING_DAYS_PER_YEAR = 252


@dataclass
class PerformanceMetrics:
    """Summary performance metrics."""

    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown_pct: float
    max_drawdown_duration_days: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    total_trades: int
    avg_holding_period: float


class PerformanceAnalyzer:
    """Calculates risk-adjusted and regime-aware performance metrics."""

    def __init__(self, risk_free_rate: float = 0.045) -> None:
        """Initialize analyzer with annualized risk-free rate."""
        self.risk_free_rate = risk_free_rate

    def compute_metrics(self, equity_curve: list[float] | pd.Series) -> PerformanceMetrics:
        """Return summary metrics including Sharpe, Sortino, Calmar, drawdown."""
        if isinstance(equity_curve, pd.Series):
            equity_curve = equity_curve.values.tolist()

        if len(equity_curve) < 2:
            raise ValueError("Equity curve must have at least 2 values.")

        equity_array = np.array(equity_curve, dtype=float)
        returns = np.diff(equity_array) / equity_array[:-1]

        start_equity = equity_array[0]
        end_equity = equity_array[-1]
        total_return = (end_equity - start_equity) / start_equity
        days = len(equity_array) - 1
        years = days / TRADING_DAYS_PER_YEAR
        cagr = (end_equity / start_equity) ** (1 / max(years, 0.01)) - 1

        annual_returns = returns * TRADING_DAYS_PER_YEAR
        sharpe = annual_returns.mean() / (annual_returns.std() + 1e-12) if years > 0 else 0.0
        downside_returns = returns[returns < 0]
        sortino = (annual_returns.mean() / (downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) + 1e-12)) if len(
            downside_returns
        ) > 0 else sharpe

        cumulative_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - cumulative_max) / (cumulative_max + 1e-12)
        max_dd = np.min(drawdowns)
        max_dd_pct = abs(max_dd) * 100

        dd_duration = 0
        current_dd_duration = 0
        for dd in drawdowns:
            if dd < 0:
                current_dd_duration += 1
                dd_duration = max(dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0

        calmar = (cagr / (abs(max_dd) + 1e-12)) if max_dd < 0 else cagr

        return PerformanceMetrics(
            total_return_pct=total_return * 100,
            cagr_pct=cagr * 100,
            sharpe_ratio=float(sharpe),
            sortino_ratio=float(sortino),
            calmar_ratio=float(calmar),
            max_drawdown_pct=max_dd_pct,
            max_drawdown_duration_days=int(dd_duration),
            win_rate=float(np.mean(returns > 0)) if len(returns) > 0 else 0.0,
            avg_win=float(returns[returns > 0].mean()) if len(returns[returns > 0]) > 0 else 0.0,
            avg_loss=float(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 0.0,
            profit_factor=self._compute_profit_factor(returns),
            total_trades=len(returns),
            avg_holding_period=days / max(len(returns), 1),
        )

    def compute_trade_metrics(self, trades: list[Any]) -> dict[str, float]:
        """Compute trade-level metrics instead of bar-level metrics."""
        if not trades:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_rr_achieved": 0.0,
                "expectancy_per_trade": 0.0,
                "tp_hit_count": 0,
                "sl_hit_count": 0,
                "flat_exit_count": 0,
            }

        closed = [
            trade
            for trade in trades
            if getattr(trade, "exit_price", 0.0) > 0
            and getattr(trade, "pnl", 0.0) != 0.0
            and getattr(trade, "exit_reason", "") not in ("", "OPEN")
        ]
        if not closed:
            return {
                "total_trades": 0,
                "winners": 0,
                "losers": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_rr_achieved": 0.0,
                "expectancy_per_trade": 0.0,
                "tp_hit_count": 0,
                "sl_hit_count": 0,
                "flat_exit_count": 0,
            }

        winners = [trade for trade in closed if getattr(trade, "pnl", 0.0) > 0]
        losers = [trade for trade in closed if getattr(trade, "pnl", 0.0) <= 0]
        tp_hits = [trade for trade in closed if getattr(trade, "exit_reason", "") == "TP_HIT"]
        sl_hits = [trade for trade in closed if getattr(trade, "exit_reason", "") == "SL_HIT"]
        flat_exits = [trade for trade in closed if getattr(trade, "exit_reason", "") == "REGIME_FLAT"]

        gross_profit = sum(float(getattr(trade, "pnl", 0.0)) for trade in winners)
        gross_loss = abs(sum(float(getattr(trade, "pnl", 0.0)) for trade in losers))
        avg_win = gross_profit / max(len(winners), 1)
        avg_loss = gross_loss / max(len(losers), 1)

        avg_hold_bars = sum(float(getattr(trade, "holding_bars", 0.0)) for trade in closed) / max(len(closed), 1)

        LOGGER.info("EXIT REASON BREAKDOWN:")
        LOGGER.info(
            "  TP hits:       %d  (%.1f%%)",
            len(tp_hits),
            100 * len(tp_hits) / max(len(closed), 1),
        )
        LOGGER.info(
            "  SL hits:       %d  (%.1f%%)",
            len(sl_hits),
            100 * len(sl_hits) / max(len(closed), 1),
        )
        LOGGER.info(
            "  Regime FLAT:   %d  (%.1f%%)",
            len(flat_exits),
            100 * len(flat_exits) / max(len(closed), 1),
        )
        LOGGER.info("  Avg hold bars: %.1f", avg_hold_bars)
        LOGGER.info("  Avg Win  $:    %.2f", avg_win)
        LOGGER.info("  Avg Loss $:    %.2f", avg_loss)
        LOGGER.info("  RR achieved:   %.2f", avg_win / max(avg_loss, 1e-12))

        # Diagnostic sample to verify trade fields before TRUE RR calculation.
        for i, trade in enumerate(closed[:5]):
            LOGGER.info(
                "TRADE DEBUG [%d]: dir=%s entry=%.2f exit=%.2f sl=%.2f tp=%.2f reason=%s pnl=%.2f",
                i,
                getattr(trade, "direction", "N/A"),
                float(getattr(trade, "entry_price", 0.0)),
                float(getattr(trade, "exit_price", 0.0)),
                float(getattr(trade, "stop_loss", 0.0)),
                float(getattr(trade, "take_profit", 0.0)),
                getattr(trade, "exit_reason", "N/A"),
                float(getattr(trade, "pnl", 0.0)),
            )

        # Compute TRUE RR in points (not dollars)
        rr_per_trade = []
        for trade in closed:
            entry = float(getattr(trade, "entry_price", 0.0))
            exit_p = float(getattr(trade, "exit_price", 0.0))
            sl = float(getattr(trade, "stop_loss", 0.0))
            direction = getattr(trade, "direction", "")

            if entry <= 0 or sl <= 0:
                continue

            sl_distance = abs(entry - sl)
            if sl_distance < 1e-6:
                continue

            if direction == "LONG":
                points_captured = exit_p - entry
            else:
                points_captured = entry - exit_p

            rr_this_trade = points_captured / sl_distance
            rr_per_trade.append(rr_this_trade)

        true_avg_rr = float(np.mean(rr_per_trade)) if rr_per_trade else 0.0
        LOGGER.info("  TRUE RR (puntos): %.2f  (n=%d trades)", true_avg_rr, len(rr_per_trade))

        return {
            "total_trades": len(closed),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": float(len(winners) / len(closed)),
            "profit_factor": float(gross_profit / (gross_loss + 1e-12)) if gross_loss > 0 else 0.0,
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "avg_rr_achieved": float(avg_win / (avg_loss + 1e-12)) if avg_loss > 0 else 0.0,
            "expectancy_per_trade": float((gross_profit - gross_loss) / len(closed)),
            "tp_hit_count": len(tp_hits),
            "sl_hit_count": len(sl_hits),
            "flat_exit_count": len(flat_exits),
        }

    def _compute_profit_factor(self, returns: np.ndarray) -> float:
        """Compute profit factor: sum(wins) / abs(sum(losses))."""
        wins = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        return float(wins / (losses + 1e-12)) if losses > 0 else 1.0

    def regime_breakdown(
        self,
        equity_curve: list[float],
        regime_history: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Return per-regime performance table."""
        if not regime_history:
            return pd.DataFrame()

        regime_data = []
        for regime_record in regime_history:
            regime_id = regime_record.get("regime_id")
            regime_name = regime_record.get("regime_name")
            bar_returns = regime_record.get("return", 0.0)

            regime_data.append(
                {
                    "Regime": regime_name,
                    "Time %": 0.0,
                    "Return %": bar_returns * 100,
                    "Sharpe": 0.0,
                    "Win Rate": 0.0,
                }
            )

        return pd.DataFrame(regime_data).drop_duplicates(subset=["Regime"])

    def confidence_bucketing(
        self,
        trades: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Return trades bucketed by regime confidence."""
        if not trades:
            return pd.DataFrame()

        buckets = {
            "<50%": [],
            "50-60%": [],
            "60-70%": [],
            "70%+": [],
        }

        for trade in trades:
            confidence = trade.get("confidence", 0.0)
            pnl_pct = trade.get("pnl_pct", 0.0)

            if confidence < 0.5:
                buckets["<50%"].append(pnl_pct)
            elif confidence < 0.6:
                buckets["50-60%"].append(pnl_pct)
            elif confidence < 0.7:
                buckets["60-70%"].append(pnl_pct)
            else:
                buckets["70%+"].append(pnl_pct)

        results = []
        for label, pnls in buckets.items():
            if pnls:
                results.append(
                    {
                        "Confidence": label,
                        "Trades": len(pnls),
                        "Sharpe": 0.0,
                        "Win Rate": float(np.mean(np.array(pnls) > 0)),
                        "Avg P&L %": float(np.mean(pnls)) * 100,
                    }
                )

        return pd.DataFrame(results)

    @staticmethod
    def buyhold_benchmark(price_data: list[float]) -> dict[str, float]:
        """Compute buy-and-hold benchmark metrics."""
        if len(price_data) < 2:
            return {}
        start_price = price_data[0]
        end_price = price_data[-1]
        total_return = (end_price - start_price) / start_price
        returns = np.diff(price_data) / np.array(price_data[:-1])
        sharpe = (returns.mean() * TRADING_DAYS_PER_YEAR) / (returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) + 1e-12)
        return {"total_return_pct": total_return * 100, "sharpe": float(sharpe)}

    @staticmethod
    def sma200_benchmark(price_data: list[float]) -> dict[str, float]:
        """Compute 200-SMA trend-following benchmark."""
        if len(price_data) < 200:
            return {}
        prices = np.array(price_data, dtype=float)
        sma_200 = np.convolve(prices, np.ones(200) / 200, mode="valid")
        prices_aligned = prices[199:]

        allocations = np.where(prices_aligned > sma_200, 1.0, 0.0)
        returns = np.diff(prices_aligned) / prices_aligned[:-1] * allocations[:-1]

        total_return = np.sum(returns) if len(returns) > 0 else 0.0
        sharpe = (
            (np.mean(returns) * TRADING_DAYS_PER_YEAR) / (np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR) + 1e-12)
            if len(returns) > 0
            else 0.0
        )
        return {"total_return_pct": total_return * 100, "sharpe": float(sharpe)}

    @staticmethod
    def random_benchmark(price_data: list[float], n_simulations: int = 100, seed: int = 42) -> dict[str, float]:
        """Compute random allocation baseline with Monte Carlo."""
        rng = np.random.default_rng(seed)
        results_sharpe = []
        results_return = []

        for _ in range(n_simulations):
            allocations = rng.uniform(0, 1, len(price_data) - 1)
            prices = np.array(price_data, dtype=float)
            returns = np.diff(prices) / prices[:-1] * allocations

            total_ret = np.sum(returns) if len(returns) > 0 else 0.0
            sharpe = (
                (np.mean(returns) * TRADING_DAYS_PER_YEAR) / (np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR) + 1e-12)
                if len(returns) > 0 and np.std(returns) > 0
                else 0.0
            )
            results_return.append(total_ret)
            results_sharpe.append(sharpe)

        return {
            "mean_return_pct": float(np.mean(results_return)) * 100,
            "std_return_pct": float(np.std(results_return)) * 100,
            "mean_sharpe": float(np.mean(results_sharpe)),
            "std_sharpe": float(np.std(results_sharpe)),
        }

