"""EMA9 + EMA200 + VWAP regime-filtered strategy for NAS100 M5.

Rules
-----
1. If HMM regime == CHOP / NEUTRAL -> FLAT (no trade).
2. If regime == TRENDING (BULL / STRONG_BULL / BEAR / STRONG_BEAR):
   - Evaluate on **candle close only**.
   - LONG:  close > EMA200 AND EMA9 > EMA200.
            Trigger: candle closes ABOVE both EMA9 and VWAP after a pullback.
   - SHORT: close < EMA200 AND EMA9 < EMA200.
            Trigger: candle closes BELOW both EMA9 and VWAP after a pullback.
3. Pullback = at least one of the last N candles touched EMA9 or VWAP.
4. Position size = 1% account risk / (entry − swing SL).
5. SL at swing low/high over lookback period. TP at 2:1 RR.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from core.hmm_engine import RegimeInfo, RegimeState

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-12


class TrendType(Enum):
    CHOP = "CHOP"
    TRENDING = "TRENDING"


class Direction(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Tunable strategy parameters."""

    risk_percent: float = 0.01          # 1% account risk per trade
    ema_9_period: int = 9
    ema_200_period: int = 200
    atr_period: int = 14
    pullback_lookback: int = 5          # candles to check for pullback
    swing_lookback: int = 20            # swing high/low window
    exhaustion_atr_threshold: float = 2.0
    rr_ratio: float = 2.0              # reward-to-risk for TP
    chop_labels: tuple[str, ...] = (
        "CHOP", "NEUTRAL", "WEAK_BEAR", "WEAK_BULL",
    )
    confidence_threshold: float = 0.90


# ------------------------------------------------------------------
# Data containers
# ------------------------------------------------------------------

@dataclass
class TechnicalSetup:
    """Pre-computed technical state from the last closed candle."""

    price: float                        # close of last bar
    ema_9: float
    ema_200: float
    vwap: float
    atr_14: float
    swing_high: float                   # SL for shorts
    swing_low: float                    # SL for longs
    distance_from_ema200: float
    volatility_percentile: float = 0.0
    had_pullback: bool = False          # at least one candle touched EMA9/VWAP


@dataclass
class Signal:
    """Directional trading signal with risk-management parameters."""

    symbol: str
    direction: str                      # "LONG" / "SHORT" / "FLAT"
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    lots: float                         # computed position size in lots
    position_size: float                # notional risk amount
    timestamp: datetime
    reasoning: str
    regime_id: int
    regime_name: str
    regime_probability: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # Aliases used by downstream order executor
    @property
    def target_weight(self) -> float:
        return self.lots


# ------------------------------------------------------------------
# Pullback detector
# ------------------------------------------------------------------

def detect_pullback(
    df: pd.DataFrame,
    direction: Direction,
    lookback: int = 5,
) -> bool:
    """Return True if a pullback to EMA9 or VWAP occurred in recent candles.

    For LONG: at least one bar's low touched or dipped below EMA9 or VWAP.
    For SHORT: at least one bar's high touched or rose above EMA9 or VWAP.
    """
    if len(df) < lookback + 1:
        return False

    recent = df.iloc[-(lookback + 1):-1]  # exclude the trigger candle

    if "ema_9" not in recent.columns or "vwap" not in recent.columns:
        return False

    if direction == Direction.LONG:
        touched_ema = (recent["low"] <= recent["ema_9"]).any()
        touched_vwap = (recent["low"] <= recent["vwap"]).any()
    else:
        touched_ema = (recent["high"] >= recent["ema_9"]).any()
        touched_vwap = (recent["high"] >= recent["vwap"]).any()

    return bool(touched_ema or touched_vwap)


# ------------------------------------------------------------------
# Position sizing
# ------------------------------------------------------------------

def calculate_position_size(
    account_equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    symbol_info: dict[str, Any] | None = None,
) -> float:
    """Calculate position size in lots based on 1% account risk.

    ``lots = (equity * risk%) / (|entry - SL| * contract_size)``

    For NAS100 on Exness the default contract size is 1 (1 lot = $1/point).
    """
    stop_distance = abs(entry_price - stop_loss)
    if stop_distance < EPSILON:
        return 0.0

    risk_amount = account_equity * risk_pct
    contract_size = 1.0  # NAS100 default

    if symbol_info:
        contract_size = symbol_info.get("contract_size", 1.0)

    raw_lots = risk_amount / (stop_distance * contract_size)

    # Quantize to 0.01 step and clamp
    lots = max(0.01, round(raw_lots, 2))
    return lots


# ------------------------------------------------------------------
# Strategy
# ------------------------------------------------------------------

class RegimeStrategy:
    """Volatility-filtered EMA/VWAP directional trading strategy."""

    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    # ---------- regime classification ----------

    def classify_regime(
        self,
        regime_info: RegimeInfo,
        regime_state: RegimeState,
    ) -> TrendType:
        """Classify regime as CHOP or TRENDING based on the HMM label."""
        label = regime_info.label.upper()

        if label in [c.upper() for c in self.config.chop_labels]:
            return TrendType.CHOP

        # Low confidence -> treat as chop
        if regime_state.state_probability < self.config.confidence_threshold:
            return TrendType.CHOP

        # Flickering -> chop
        if regime_state.is_flickering:
            return TrendType.CHOP

        return TrendType.TRENDING

    # ---------- macro conditions ----------

    @staticmethod
    def bullish_macro(setup: TechnicalSetup) -> bool:
        return setup.price > setup.ema_200 and setup.ema_9 > setup.ema_200

    @staticmethod
    def bearish_macro(setup: TechnicalSetup) -> bool:
        return setup.price < setup.ema_200 and setup.ema_9 < setup.ema_200

    @staticmethod
    def is_exhausted(setup: TechnicalSetup, threshold: float) -> bool:
        return abs(setup.distance_from_ema200) > threshold * setup.atr_14

    # ---------- main evaluation ----------

    def evaluate_signal(
        self,
        symbol: str,
        regime_state: RegimeState,
        regime_info_map: dict[int, RegimeInfo],
        setup: TechnicalSetup,
        account_equity: float = 10_000.0,
    ) -> Signal:
        """Generate directional signal from HMM regime + EMA/VWAP technicals.

        ONLY evaluated on candle close data (caller must ensure this).
        """
        rid = regime_state.current_regime_id
        rinfo = regime_info_map[rid]
        rtype = self.classify_regime(rinfo, regime_state)

        # ── CHOP -> FLAT ──────────────────────────────────────
        if rtype == TrendType.CHOP:
            return self._flat_signal(
                symbol, setup, rid, rinfo, regime_state,
                reason="CHOP regime — no trade.",
            )

        direction: Direction | None = None
        stop_loss = 0.0
        take_profit = 0.0
        reason = ""

        # ── BULLISH ───────────────────────────────────────────
        if self.bullish_macro(setup):
            if self.is_exhausted(setup, self.config.exhaustion_atr_threshold):
                return self._flat_signal(
                    symbol, setup, rid, rinfo, regime_state,
                    reason="Bullish but exhausted (>2 ATR from EMA200).",
                )

            trigger = setup.price > setup.ema_9 and setup.price > setup.vwap
            pullback = setup.had_pullback

            if trigger and pullback:
                direction = Direction.LONG
                stop_loss = setup.swing_low
                rr_dist = abs(setup.price - stop_loss) * self.config.rr_ratio
                take_profit = setup.price + rr_dist
                reason = (
                    f"LONG: Close > EMA9 & VWAP after pullback. "
                    f"SL={stop_loss:.2f}  TP={take_profit:.2f}"
                )
            else:
                reason = "Bullish macro but no trigger/pullback yet."
                return self._flat_signal(
                    symbol, setup, rid, rinfo, regime_state, reason=reason,
                )

        # ── BEARISH ───────────────────────────────────────────
        elif self.bearish_macro(setup):
            if self.is_exhausted(setup, self.config.exhaustion_atr_threshold):
                return self._flat_signal(
                    symbol, setup, rid, rinfo, regime_state,
                    reason="Bearish but exhausted (>2 ATR from EMA200).",
                )

            trigger = setup.price < setup.ema_9 and setup.price < setup.vwap
            pullback = setup.had_pullback

            if trigger and pullback:
                direction = Direction.SHORT
                stop_loss = setup.swing_high
                rr_dist = abs(stop_loss - setup.price) * self.config.rr_ratio
                take_profit = setup.price - rr_dist
                reason = (
                    f"SHORT: Close < EMA9 & VWAP after pullback. "
                    f"SL={stop_loss:.2f}  TP={take_profit:.2f}"
                )
            else:
                reason = "Bearish macro but no trigger/pullback yet."
                return self._flat_signal(
                    symbol, setup, rid, rinfo, regime_state, reason=reason,
                )

        # ── NO CLEAR TREND ────────────────────────────────────
        else:
            return self._flat_signal(
                symbol, setup, rid, rinfo, regime_state,
                reason="No clear macro alignment.",
            )

        # ── Position sizing (1% risk) ─────────────────────────
        lots = calculate_position_size(
            account_equity=account_equity,
            risk_pct=self.config.risk_percent,
            entry_price=setup.price,
            stop_loss=stop_loss,
        )
        stop_dist = abs(setup.price - stop_loss)
        risk_amount = account_equity * self.config.risk_percent

        return Signal(
            symbol=symbol,
            direction=direction.value,
            confidence=regime_state.state_probability,
            entry_price=setup.price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lots=lots,
            position_size=risk_amount,
            timestamp=datetime.now(),
            reasoning=reason,
            regime_id=rid,
            regime_name=rinfo.label,
            regime_probability=regime_state.state_probability,
            metadata={
                "regime_type": rtype.value,
                "ema_9": setup.ema_9,
                "ema_200": setup.ema_200,
                "vwap": setup.vwap,
                "atr_14": setup.atr_14,
                "swing_high": setup.swing_high,
                "swing_low": setup.swing_low,
                "had_pullback": setup.had_pullback,
            },
        )

    # ---------- helpers ----------

    @staticmethod
    def _flat_signal(
        symbol: str,
        setup: TechnicalSetup,
        rid: int,
        rinfo: RegimeInfo,
        rstate: RegimeState,
        reason: str,
    ) -> Signal:
        return Signal(
            symbol=symbol,
            direction=Direction.FLAT.value,
            confidence=0.0,
            entry_price=setup.price,
            stop_loss=0.0,
            take_profit=0.0,
            lots=0.0,
            position_size=0.0,
            timestamp=datetime.now(),
            reasoning=reason,
            regime_id=rid,
            regime_name=rinfo.label,
            regime_probability=rstate.state_probability,
        )


# Backward-compatible aliases
CrashDefensiveStrategy = RegimeStrategy
BearTrendStrategy = RegimeStrategy
MeanReversionStrategy = RegimeStrategy
BullTrendStrategy = RegimeStrategy
EuphoriaCautiousStrategy = RegimeStrategy

LABEL_TO_STRATEGY = {
    label: RegimeStrategy
    for label in [
        "CRASH", "STRONG_BEAR", "WEAK_BEAR", "BEAR",
        "NEUTRAL", "CHOP",
        "WEAK_BULL", "BULL", "STRONG_BULL", "EUPHORIA", "TRENDING",
    ]
}
