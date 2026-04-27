"""Volatility-filtered EMA/VWAP bidirectional strategy for intraday trending."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-12


class TrendType(Enum):
    """Market regime classification from HMM volatility filter."""

    CHOP = "CHOP"
    TRENDING = "TRENDING"


class Direction(Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass
class StrategyConfig:
    """Configuration for EMA/VWAP technical strategy."""

    risk_percent: float = 0.01
    ema_9_period: int = 9
    ema_200_period: int = 200
    vwap_period: int = 20
    atr_period: int = 14
    exhaustion_atr_threshold: float = 2.0
    chop_volatility_threshold: float = 0.35
    low_vol_allocation: float = 0.95
    mid_vol_allocation: float = 0.60
    low_vol_leverage: float = 1.25
    rebalance_threshold: float = 0.10
    uncertainty_size_mult: float = 0.5
    confidence_threshold: float = 0.95


@dataclass
class OHLCVBar:
    """Single OHLCV candle for technical analysis."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class TechnicalSetup:
    """Computed technical indicators for the current bar."""

    price: float
    ema_9: float
    ema_200: float
    vwap: float
    atr_14: float
    swing_high: float
    swing_low: float
    distance_from_ema200: float
    volatility_percentile: float


@dataclass
class Signal:
    """Directional trading signal with risk/entry/stop parameters."""

    symbol: str
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    position_size: float
    timestamp: datetime
    reasoning: str
    regime_id: int
    regime_name: str
    regime_probability: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketContext:
    """Market snapshot for strategy evaluation."""

    price: float
    ema_9: float
    ema_200: float
    vwap: float
    atr_14: float
    swing_high: float
    swing_low: float
    timestamp: datetime


class VWAPCalculator:
    """Compute volume-weighted average price incrementally."""

    def __init__(self, window: int = 20) -> None:
        """Initialize VWAP calculator with rolling window."""
        self.window = window
        self.values: list[tuple[float, int]] = []

    def update(self, bar: OHLCVBar) -> float:
        """Add bar and return current VWAP."""
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        vp = typical_price * bar.volume
        self.values.append((vp, bar.volume))

        if len(self.values) > self.window:
            self.values.pop(0)

        total_vp = sum(vp for vp, _ in self.values)
        total_vol = sum(vol for _, vol in self.values)
        return total_vp / (total_vol + EPSILON)


class RegimeStrategy:
    """Volatility-filtered EMA/VWAP directional trading strategy."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        """Initialize strategy with configuration."""
        if config is None:
            config = StrategyConfig()
        self.config = config
        self._regime_type_cache: dict[int, TrendType] = {}
        self._volatility_cache: dict[int, float] = {}

    def detect_hmm_regime_type(self, regime_info: RegimeInfo, volatility_percentile: float | None = None) -> TrendType:
        """Classify regime as CHOP or TRENDING based on expected volatility."""
        if volatility_percentile is not None:
            is_chop = volatility_percentile < self.config.chop_volatility_threshold
        else:
            is_chop = regime_info.expected_volatility < 0.015

        return TrendType.CHOP if is_chop else TrendType.TRENDING

    def check_bullish_conditions(self, setup: TechnicalSetup) -> bool:
        """Check if bullish macro conditions are met."""
        return setup.price > setup.ema_200 and setup.ema_9 > setup.ema_200

    def check_bearish_conditions(self, setup: TechnicalSetup) -> bool:
        """Check if bearish macro conditions are met."""
        return setup.price < setup.ema_200 and setup.ema_9 < setup.ema_200

    def check_exhaustion(self, setup: TechnicalSetup) -> bool:
        """Check if price is overextended from EMA200."""
        abs_distance = abs(setup.distance_from_ema200)
        threshold = self.config.exhaustion_atr_threshold * setup.atr_14
        return abs_distance > threshold

    def evaluate_signal(
        self,
        symbol: str,
        regime_state: RegimeState,
        regime_info_map: dict[int, RegimeInfo],
        setup: TechnicalSetup,
    ) -> Signal:
        """Generate directional signal from HMM regime and EMA/VWAP technicals."""
        regime_id = regime_state.current_regime_id
        regime_info = regime_info_map[regime_id]
        regime_type = self.detect_hmm_regime_type(regime_info, volatility_percentile=setup.volatility_percentile)

        if regime_type == TrendType.CHOP:
            return Signal(
                symbol=symbol,
                direction=Direction.FLAT.value,
                confidence=0.0,
                entry_price=setup.price,
                stop_loss=setup.price,
                position_size=0.0,
                timestamp=datetime.now(),
                reasoning="CHOP regime detected: no trading, avoid whipsaws.",
                regime_id=regime_id,
                regime_name=regime_info.label,
                regime_probability=regime_state.state_probability,
                metadata={"regime_type": regime_type.value},
            )

        direction = None
        reasoning = ""
        stop_loss = 0.0
        entry_price = setup.price

        if self.check_bullish_conditions(setup):
            if self.check_exhaustion(setup):
                reasoning = "Bullish macro setup but EXHAUSTED (>2 ATR from EMA200): wait for reversion."
                direction = Direction.FLAT
            else:
                if setup.price > setup.ema_9 and setup.price > setup.vwap:
                    stop_loss = setup.swing_low
                    reasoning = f"LONG: Price CLOSE > EMA9 AND VWAP. Stop at swing low {stop_loss:.2f}."
                    direction = Direction.LONG
                else:
                    reasoning = "Bullish conditions met but waiting for EMA9/VWAP cross confirmation."
                    direction = Direction.FLAT

        elif self.check_bearish_conditions(setup):
            if self.check_exhaustion(setup):
                reasoning = "Bearish macro setup but EXHAUSTED (>2 ATR from EMA200): wait for reversion."
                direction = Direction.FLAT
            else:
                if setup.price < setup.ema_9 and setup.price < setup.vwap:
                    stop_loss = setup.swing_high
                    reasoning = f"SHORT: Price CLOSE < EMA9 AND VWAP. Stop at swing high {stop_loss:.2f}."
                    direction = Direction.SHORT
                else:
                    reasoning = "Bearish conditions met but waiting for EMA9/VWAP cross confirmation."
                    direction = Direction.FLAT

        else:
            reasoning = "No clear macro trend (price/EMA9 not aligned with EMA200)."
            direction = Direction.FLAT

        if direction == Direction.FLAT:
            position_size = 0.0
            confidence = 0.0
        else:
            stop_distance = abs(entry_price - stop_loss)
            if stop_distance < EPSILON:
                position_size = 0.0
                confidence = 0.0
                reasoning = "Stop loss distance too small to calculate position size."
            else:
                account_risk = 1.0
                position_size = (account_risk * 100.0) / stop_distance
                confidence = regime_state.state_probability

        return Signal(
            symbol=symbol,
            direction=direction.value,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=stop_loss,
            position_size=position_size,
            timestamp=datetime.now(),
            reasoning=reasoning,
            regime_id=regime_id,
            regime_name=regime_info.label,
            regime_probability=regime_state.state_probability,
            metadata={
                "regime_type": regime_type.value,
                "ema_9": setup.ema_9,
                "ema_200": setup.ema_200,
                "vwap": setup.vwap,
                "atr_14": setup.atr_14,
                "distance_from_ema200": setup.distance_from_ema200,
                "exhaustion_check": self.check_exhaustion(setup),
            },
        )

    def target_allocation(self, regime: int, confidence: float) -> float:
        """Return baseline target allocation from confidence-only fallback logic."""
        if confidence < self.config.confidence_threshold:
            return self.config.mid_vol_allocation * self.config.uncertainty_size_mult
        return self.config.low_vol_allocation if regime == 0 else self.config.mid_vol_allocation

    def should_rebalance(self, current_weight: float, target_weight: float) -> bool:
        """Decide whether portfolio rebalance threshold is exceeded."""
        return abs(target_weight - current_weight) > self.config.rebalance_threshold


# Backward-compatible aliases.
CrashDefensiveStrategy = RegimeStrategy
BearTrendStrategy = RegimeStrategy
MeanReversionStrategy = RegimeStrategy
BullTrendStrategy = RegimeStrategy
EuphoriaCautiousStrategy = RegimeStrategy

LABEL_TO_STRATEGY = {
    "CRASH": RegimeStrategy,
    "STRONG_BEAR": RegimeStrategy,
    "WEAK_BEAR": RegimeStrategy,
    "BEAR": RegimeStrategy,
    "NEUTRAL": RegimeStrategy,
    "WEAK_BULL": RegimeStrategy,
    "BULL": RegimeStrategy,
    "STRONG_BULL": RegimeStrategy,
    "EUPHORIA": RegimeStrategy,
}
