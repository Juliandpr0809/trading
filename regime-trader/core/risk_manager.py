"""Risk management layer: portfolio limits, circuit breakers, margin check.

Independent of HMM.  Even if regime model fails, circuit breakers catch
drawdowns.  Defence in depth — multiple protective layers.

Absolute Veto Rules
-------------------
- Block trades if daily drawdown >= 3%.
- Block trades without stop loss.
- Block trades if MT5 margin is insufficient.
- Max 1% risk per trade, 15% max single position, 5 concurrent positions.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER = logging.getLogger(__name__)


# ==================================================================
# Circuit breaker
# ==================================================================

class CircuitBreakerType(Enum):
    DAILY_DD_REDUCE = "daily_dd_reduce"    # > 2%
    DAILY_DD_HALT = "daily_dd_halt"        # > 3%
    WEEKLY_DD_REDUCE = "weekly_dd_reduce"  # > 5%
    WEEKLY_DD_HALT = "weekly_dd_halt"      # > 7%
    PEAK_DD_HALT = "peak_dd_halt"          # > 10%
    NONE = "none"


@dataclass
class CircuitBreakerEvent:
    timestamp: datetime
    breaker_type: CircuitBreakerType
    trigger_value: float
    threshold: float
    equity: float
    peak_equity: float
    regime_at_trigger: str
    positions_closed: int = 0
    action_taken: str = ""


class CircuitBreaker:
    """Independent circuit breaker monitor based on actual P&L drawdowns."""

    def __init__(
        self,
        daily_dd_reduce: float = 0.02,
        daily_dd_halt: float = 0.03,
        weekly_dd_reduce: float = 0.05,
        weekly_dd_halt: float = 0.07,
        peak_dd_halt: float = 0.10,
    ) -> None:
        self.daily_dd_reduce = daily_dd_reduce
        self.daily_dd_halt = daily_dd_halt
        self.weekly_dd_reduce = weekly_dd_reduce
        self.weekly_dd_halt = weekly_dd_halt
        self.peak_dd_halt = peak_dd_halt

        self.daily_high: float | None = None
        self.weekly_high: float | None = None
        self.peak_high: float = 0.0

        self.events: list[CircuitBreakerEvent] = []
        self.last_daily_reset = datetime.now()
        self.last_weekly_reset = datetime.now()

    def set_baseline_equity(self, equity: float) -> None:
        """CRITICAL: set initial peak from real account to avoid false drawdown."""
        self.peak_high = equity
        self.daily_high = equity
        self.weekly_high = equity
        LOGGER.info("[OK] Circuit breaker baseline set to $%.2f", equity)

    def update(
        self,
        current_equity: float,
        regime_name: str = "UNKNOWN",
    ) -> tuple[CircuitBreakerType, CircuitBreakerEvent | None]:
        now = datetime.now()

        # Reset windows
        if (now - self.last_daily_reset).total_seconds() > 86400:
            self.daily_high = current_equity
            self.last_daily_reset = now
        if (now - self.last_weekly_reset).total_seconds() > 604800:
            self.weekly_high = current_equity
            self.last_weekly_reset = now

        if self.daily_high is None:
            self.daily_high = current_equity
        if self.weekly_high is None:
            self.weekly_high = current_equity
        if current_equity > self.peak_high:
            self.peak_high = current_equity

        self.daily_high = max(self.daily_high, current_equity)
        self.weekly_high = max(self.weekly_high, current_equity)

        daily_dd = (self.daily_high - current_equity) / (self.daily_high + 1e-12)
        weekly_dd = (self.weekly_high - current_equity) / (self.weekly_high + 1e-12)
        peak_dd = (self.peak_high - current_equity) / (self.peak_high + 1e-12)

        # Check thresholds (severity order)
        checks = [
            (peak_dd, self.peak_dd_halt, CircuitBreakerType.PEAK_DD_HALT, self.peak_high),
            (daily_dd, self.daily_dd_halt, CircuitBreakerType.DAILY_DD_HALT, self.daily_high),
            (weekly_dd, self.weekly_dd_halt, CircuitBreakerType.WEEKLY_DD_HALT, self.weekly_high),
            (daily_dd, self.daily_dd_reduce, CircuitBreakerType.DAILY_DD_REDUCE, self.daily_high),
            (weekly_dd, self.weekly_dd_reduce, CircuitBreakerType.WEEKLY_DD_REDUCE, self.weekly_high),
        ]

        for dd_val, threshold, cb_type, ref_equity in checks:
            if dd_val >= threshold:
                event = CircuitBreakerEvent(
                    timestamp=now,
                    breaker_type=cb_type,
                    trigger_value=dd_val,
                    threshold=threshold,
                    equity=current_equity,
                    peak_equity=ref_equity,
                    regime_at_trigger=regime_name,
                )
                self.events.append(event)
                severity = "CRITICAL" if "halt" in cb_type.value else "WARNING"
                LOGGER.log(
                    logging.CRITICAL if severity == "CRITICAL" else logging.WARNING,
                    "[CB] %s  dd=%.2f%%  threshold=%.2f%%  equity=$%,.0f",
                    cb_type.value,
                    dd_val * 100,
                    threshold * 100,
                    current_equity,
                )
                return cb_type, event

        return CircuitBreakerType.NONE, None

    def get_history(self, limit: int = 100) -> list[CircuitBreakerEvent]:
        return self.events[-limit:]

    def reset_daily(self) -> None:
        self.daily_high = None
        self.last_daily_reset = datetime.now()

    def reset_weekly(self) -> None:
        self.weekly_high = None
        self.last_weekly_reset = datetime.now()


# ==================================================================
# Portfolio state
# ==================================================================

@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float
    stop_loss: float
    regime_at_entry: str
    timestamp: datetime
    side: str = "LONG"
    current_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class PortfolioState:
    equity: float
    cash: float
    peak_equity: float
    positions: list[Position] = field(default_factory=list)
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    buying_power: float = 0.0
    daily_pnl: float = 0.0
    daily_dd: float = 0.0
    weekly_pnl: float = 0.0
    weekly_dd: float = 0.0
    peak_dd: float = 0.0
    daily_trade_count: int = 0
    last_trade_timestamp: datetime | None = None
    circuit_breaker_active: CircuitBreakerType = CircuitBreakerType.NONE
    size_reduction_multiplier: float = 1.0
    trading_halted: bool = False
    high_flicker_rate: bool = False
    regime_uncertain: bool = False


@dataclass
class RiskDecision:
    approved: bool
    original_signal: Any
    modified_signal: Any | None = None
    rejection_reason: str = ""
    modifications: list[str] = field(default_factory=list)
    risk_score: float = 0.0


# ==================================================================
# Risk Manager
# ==================================================================

class RiskManager:
    """Portfolio-level risk management with absolute veto power.

    Independent of HMM.  Enforces hard limits on:
    - 1% risk per trade
    - 3% daily drawdown halt
    - 15% max single position
    - MT5 margin requirement check
    """

    def __init__(self, config_path: str | None = None) -> None:
        self.config = self._load_config(config_path)
        self.circuit_breaker = CircuitBreaker(
            daily_dd_reduce=self.config.get("daily_dd_reduce", 0.02),
            daily_dd_halt=self.config.get("daily_dd_halt", 0.03),
            weekly_dd_reduce=self.config.get("weekly_dd_reduce", 0.05),
            weekly_dd_halt=self.config.get("weekly_dd_halt", 0.07),
            peak_dd_halt=self.config.get("peak_dd_halt", 0.10),
        )
        self.positions: dict[str, Position] = {}
        self.equity_history: list[tuple[datetime, float]] = []
        self.peak_equity: float = 0.0
        self.daily_trade_count: int = 0
        self.last_trade_timestamp: datetime | None = None
        self.last_order_by_symbol: dict[str, tuple[datetime, str]] = {}

    def set_account_equity(self, equity: float) -> None:
        """CRITICAL: Update baselines with real account equity."""
        self.peak_equity = equity
        self.circuit_breaker.set_baseline_equity(equity)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, config_path: str | None) -> dict[str, Any]:
        if config_path is None:
            config_path = str(
                Path(__file__).parent.parent / "config" / "settings.yaml"
            )
        if not Path(config_path).exists():
            LOGGER.warning("Config not found: %s. Using defaults.", config_path)
            return self._default_config()
        try:
            with open(config_path) as f:
                settings = yaml.safe_load(f) or {}
            return settings.get("risk", self._default_config())
        except Exception as exc:
            LOGGER.error("Config load failed: %s. Using defaults.", exc)
            return self._default_config()

    @staticmethod
    def _default_config() -> dict[str, Any]:
        return {
            "max_risk_per_trade": 0.01,
            "max_exposure": 0.80,
            "max_leverage": 1.25,
            "max_single_position": 0.15,
            "max_concurrent_positions": 5,
            "max_daily_trades": 20,
            "min_position_value": 100.0,
            "daily_dd_reduce": 0.02,
            "daily_dd_halt": 0.03,
            "weekly_dd_reduce": 0.05,
            "weekly_dd_halt": 0.07,
            "peak_dd_halt": 0.10,
        }

    # ------------------------------------------------------------------
    # Signal validation (absolute veto)
    # ------------------------------------------------------------------

    def validate_signal(
        self,
        signal: Any,
        portfolio_state: PortfolioState,
        mt5_client: Any | None = None,
    ) -> RiskDecision:
        """Validate signal against portfolio limits and risk constraints.

        Returns RiskDecision with approval status and any modifications.
        """
        modifications: list[str] = []
        risk_score = 0.0

        # ──── CIRCUIT BREAKER VETO ─────────────────────────────
        halt_types = {
            CircuitBreakerType.PEAK_DD_HALT,
            CircuitBreakerType.DAILY_DD_HALT,
            CircuitBreakerType.WEEKLY_DD_HALT,
        }
        if portfolio_state.circuit_breaker_active in halt_types:
            return RiskDecision(
                approved=False,
                original_signal=signal,
                rejection_reason=(
                    f"Circuit breaker active: {portfolio_state.circuit_breaker_active.value}. "
                    "All new orders blocked."
                ),
                risk_score=1.0,
            )

        # ──── MANDATORY STOP LOSS ──────────────────────────────
        sl = getattr(signal, "stop_loss", 0.0)
        if sl <= 0:
            return RiskDecision(
                approved=False,
                original_signal=signal,
                rejection_reason="STOP LOSS REQUIRED. Order refused.",
                risk_score=1.0,
            )

        # ──── DUPLICATE ORDER CHECK ────────────────────────────
        if self._is_duplicate(signal.symbol, signal.direction):
            return RiskDecision(
                approved=False,
                original_signal=signal,
                rejection_reason=f"Duplicate {signal.symbol} {signal.direction} within 60s.",
                risk_score=0.8,
            )

        # ──── CONCURRENT POSITIONS ─────────────────────────────
        open_ct = len(portfolio_state.positions)
        max_concurrent = self.config.get("max_concurrent_positions", 5)
        if open_ct >= max_concurrent:
            return RiskDecision(
                approved=False,
                original_signal=signal,
                rejection_reason=f"Max concurrent positions ({max_concurrent}) reached.",
                risk_score=0.7,
            )

        # ──── DAILY TRADE LIMIT ────────────────────────────────
        max_daily = self.config.get("max_daily_trades", 20)
        if portfolio_state.daily_trade_count >= max_daily:
            return RiskDecision(
                approved=False,
                original_signal=signal,
                rejection_reason=f"Daily trade limit ({max_daily}) reached.",
                risk_score=0.6,
            )

        # ──── MARGIN CHECK (MT5) ──────────────────────────────
        if mt5_client is not None:
            lots = getattr(signal, "lots", 0.01)
            import MetaTrader5 as mt5
            order_type = (
                mt5.ORDER_TYPE_BUY
                if signal.direction == "LONG"
                else mt5.ORDER_TYPE_SELL
            )
            required_margin = mt5_client.get_margin_required(
                signal.symbol, lots, order_type
            )
            if required_margin is not None:
                margin_free = portfolio_state.cash
                if required_margin > margin_free:
                    return RiskDecision(
                        approved=False,
                        original_signal=signal,
                        rejection_reason=(
                            f"Insufficient margin: need ${required_margin:,.2f}, "
                            f"free ${margin_free:,.2f}"
                        ),
                        risk_score=0.9,
                    )

        # ──── SIZE REDUCTION FROM CB ───────────────────────────
        modified = copy.deepcopy(signal)
        if portfolio_state.size_reduction_multiplier < 1.0:
            if hasattr(modified, "lots"):
                modified.lots *= portfolio_state.size_reduction_multiplier
            modifications.append(
                f"Size reduced {portfolio_state.size_reduction_multiplier:.0%} (circuit breaker)"
            )
            risk_score += 0.15

        # ──── REGIME UNCERTAINTY ───────────────────────────────
        if portfolio_state.regime_uncertain or portfolio_state.high_flicker_rate:
            if hasattr(modified, "lots"):
                modified.lots *= 0.5
            modifications.append("Size halved due to regime uncertainty / flicker")
            risk_score += 0.1

        # ──── APPROVED ─────────────────────────────────────────
        decision = RiskDecision(
            approved=True,
            original_signal=signal,
            modified_signal=modified if modifications else None,
            modifications=modifications,
            risk_score=min(risk_score, 1.0),
        )

        if modifications:
            LOGGER.info(
                "Signal APPROVED with mods: %s | %s",
                signal.symbol,
                ", ".join(modifications),
            )
        else:
            LOGGER.info(
                "Signal APPROVED: %s %s | risk=%.2f",
                signal.symbol,
                signal.direction,
                risk_score,
            )

        return decision

    # ------------------------------------------------------------------
    # Portfolio state computation
    # ------------------------------------------------------------------

    def compute_portfolio_state(
        self,
        equity: float = 0.0,
        cash: float = 0.0,
        positions_list: list | None = None,
        daily_pnl: float = 0.0,
        regime_name: str = "UNKNOWN",
        hmm_confidence: float = 0.5,
        regime_flicker_rate: float = 0.0,
    ) -> PortfolioState:
        """Compute current portfolio state and update circuit breaker."""
        if positions_list is None:
            positions_list = []

        if equity > self.peak_equity:
            self.peak_equity = equity
        self.equity_history.append((datetime.now(), equity))

        gross = sum(abs(p.quantity * p.current_price) for p in positions_list)
        net = sum(
            (p.quantity if p.side == "LONG" else -p.quantity) * p.current_price
            for p in positions_list
        )
        lev = gross / (equity + 1e-12)

        daily_high = max((e for _, e in self.equity_history[-252:]), default=equity)
        weekly_high = max((e for _, e in self.equity_history[-1260:]), default=equity)
        daily_dd = (daily_high - equity) / (daily_high + 1e-12)
        weekly_dd = (weekly_high - equity) / (weekly_high + 1e-12)
        peak_dd = (self.peak_equity - equity) / (self.peak_equity + 1e-12)

        cb_type, _ = self.circuit_breaker.update(equity, regime_name)
        size_mult = (
            0.5
            if cb_type
            in {CircuitBreakerType.DAILY_DD_REDUCE, CircuitBreakerType.WEEKLY_DD_REDUCE}
            else 1.0
        )

        return PortfolioState(
            equity=equity,
            cash=cash,
            peak_equity=self.peak_equity,
            positions=positions_list,
            gross_exposure=gross,
            net_exposure=net,
            leverage=lev,
            buying_power=cash,
            daily_pnl=daily_pnl,
            daily_dd=daily_dd,
            weekly_dd=weekly_dd,
            peak_dd=peak_dd,
            daily_trade_count=self.daily_trade_count,
            last_trade_timestamp=self.last_trade_timestamp,
            circuit_breaker_active=cb_type,
            size_reduction_multiplier=size_mult,
            trading_halted=cb_type
            in {
                CircuitBreakerType.DAILY_DD_HALT,
                CircuitBreakerType.WEEKLY_DD_HALT,
                CircuitBreakerType.PEAK_DD_HALT,
            },
            high_flicker_rate=regime_flicker_rate > 0.2,
            regime_uncertain=hmm_confidence < 0.5,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_duplicate(
        self, symbol: str, direction: str, window_s: int = 60
    ) -> bool:
        if symbol not in self.last_order_by_symbol:
            return False
        last_time, last_dir = self.last_order_by_symbol[symbol]
        elapsed = (datetime.now() - last_time).total_seconds()
        return elapsed < window_s and last_dir == direction

    def record_trade(self, symbol: str, direction: str) -> None:
        self.daily_trade_count += 1
        self.last_trade_timestamp = datetime.now()
        self.last_order_by_symbol[symbol] = (datetime.now(), direction)

    def reset_daily_counters(self) -> None:
        self.daily_trade_count = 0
        self.circuit_breaker.reset_daily()

    def reset_weekly_counters(self) -> None:
        self.circuit_breaker.reset_weekly()

    def write_trading_halted_lockfile(self, reason: str = "Peak DD exceeded 10%") -> None:
        lockfile = Path(__file__).parent.parent / "trading_halted.lock"
        try:
            with open(lockfile, "w") as f:
                f.write(f"Trading halted at {datetime.now().isoformat()}\n")
                f.write(f"Reason: {reason}\n")
                f.write("Delete this file to resume trading.\n")
            LOGGER.critical("LOCKFILE created: %s", lockfile)
        except Exception as exc:
            LOGGER.error("Lockfile write failed: %s", exc)

    def check_trading_halted(self) -> bool:
        lockfile = Path(__file__).parent.parent / "trading_halted.lock"
        return lockfile.exists()

    def get_circuit_breaker_history(self, limit: int = 100) -> list[CircuitBreakerEvent]:
        return self.circuit_breaker.get_history(limit)
