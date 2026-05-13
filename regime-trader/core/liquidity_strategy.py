"""Sweep & PBC Pattern strategy for US30m on Exness MT5.

Strategy phases:
  1. Liquidity level identification on M30 (pre-NY-open)
  2. Liquidity sweep detection on M5 (post-NY-open)
  3. PBC (Price-Volume-Confirmation) 2-candle pattern on M5
  4. Fibonacci filter for price quality
  5. Stop order entry on zone breakout (BUY_STOP / SELL_STOP)
  6. RR 1:2 take-profit, break-even at 1:1, time-based expiry

Magic Number: 202602
Max trades per day: 1
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-12


# ==================================================================
# Configuration
# ==================================================================

@dataclass
class LiquidityConfig:
    """All tunable parameters loaded from YAML ``liquidity_strategy`` section."""

    symbol: str = "US30m"
    broker_to_colombia_hours: float = 7.0
    use_colombia_session_times: bool = False
    session_start_colombia_time: str = "08:00"
    session_end_colombia_time: str = "12:00"
    entry_cutoff_colombia_time: str = "11:00"
    session_start_server_time: str = "13:00"
    session_end_server_time: str = "17:00"
    scan_start_server_time: str = "02:00"
    scan_end_server_time: str = "11:00"
    entry_cutoff_server_time: str = "16:00"
    server_timezone: str = "UTC+0"
    timeframe_context: str = "M30"
    timeframe_entry: str = "M5"
    max_trades_per_day: int = 1
    rr_ratio: float = 2.0
    breakeven_at_rr: float = 1.0
    min_rr_to_trade: float = 1.0
    risk_percent: float = 0.01
    entry_offset_points: float = 3.0
    sl_offset_points: float = 3.0
    min_sl_points: float = 30.0
    max_sl_points: float = 150.0
    min_candle_body_points: float = 5.0
    max_trade_duration_minutes: int = 30
    rejection_zone_points: float = 20.0
    magic_number: int = 202602
    alert_sound: bool = True
    alert_log_path: str = "monitoring/alert_log.csv"
    contract_size: float = 1.0

    # Derived helpers
    @staticmethod
    def _parse_hhmm(value: str) -> dt_time:
        h, m = map(int, value.split(":"))
        return dt_time(h, m)

    @staticmethod
    def _shift_time(base: dt_time, hours_delta: float) -> dt_time:
        anchor = datetime(2000, 1, 1, base.hour, base.minute)
        shifted = anchor + timedelta(hours=hours_delta)
        return shifted.time().replace(second=0, microsecond=0)

    def _colombia_to_server_time(self, value: str) -> dt_time:
        col_time = self._parse_hhmm(value)
        # Colombia = broker + offset  => broker = Colombia - offset
        return self._shift_time(col_time, -self.broker_to_colombia_hours)

    @property
    def session_start_time(self) -> dt_time:
        if self.use_colombia_session_times:
            return self._colombia_to_server_time(self.session_start_colombia_time)
        return self._parse_hhmm(self.session_start_server_time)

    @property
    def ny_open_time(self) -> dt_time:
        """Backward-compatible alias for legacy callers."""
        return self.session_start_time

    @property
    def session_end_time(self) -> dt_time:
        if self.use_colombia_session_times:
            return self._colombia_to_server_time(self.session_end_colombia_time)
        return self._parse_hhmm(self.session_end_server_time)

    @property
    def scan_start_time(self) -> dt_time:
        return self._parse_hhmm(self.scan_start_server_time)

    @property
    def scan_end_time(self) -> dt_time:
        return self._parse_hhmm(self.scan_end_server_time)

    @property
    def entry_cutoff_time(self) -> dt_time:
        if self.use_colombia_session_times:
            return self._colombia_to_server_time(self.entry_cutoff_colombia_time)
        return self._parse_hhmm(self.entry_cutoff_server_time)

    @classmethod
    def from_yaml(cls, yaml_section: dict[str, Any]) -> "LiquidityConfig":
        """Build config from a parsed YAML dict."""
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in yaml_section.items() if k in field_names}
        return cls(**filtered)


# ==================================================================
# Data containers
# ==================================================================

@dataclass
class SweepLevel:
    """Daily liquidity levels computed from pre-open M30 bars."""

    highest_high: float
    lowest_low: float
    calculated_at: datetime
    hh_bar_time: datetime
    ll_bar_time: datetime


@dataclass
class SweepInfo:
    """Information about a detected sweep event."""

    direction: str  # "BUY" or "SELL"
    sweep_high: float  # Max price during sweep
    sweep_low: float   # Min price during sweep
    first_sweep_time: datetime
    sweep_price: float = 0.0


@dataclass
class PBCPattern:
    """3-candle PBC pattern detected on M5 after a sweep."""

    direction: str  # "BUY" or "SELL"
    # Vela 1
    vela_1_open: float
    vela_1_high: float
    vela_1_low: float
    vela_1_close: float
    vela_1_time: datetime
    # Vela 2
    vela_2_open: float
    vela_2_high: float
    vela_2_low: float
    vela_2_close: float
    vela_2_time: datetime
    # Vela 3
    vela_3_open: float
    vela_3_high: float
    vela_3_low: float
    vela_3_close: float
    vela_3_time: datetime
    # Zone
    zona_high: float
    zona_low: float
    # Trade levels
    entry_price: float
    stop_loss: float
    take_profit: float
    # Fibonacci
    fib_ratio: float
    # Validity
    is_valid: bool = True
    invalidation_reason: str = ""


@dataclass
class Signal:
    """Trading signal compatible with OrderExecutor and RiskManager."""

    symbol: str
    direction: str  # "LONG" / "SHORT" / "FLAT"
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    lots: float
    position_size: float
    timestamp: datetime
    reasoning: str
    order_type: str = "STOP"
    # Compatibility fields
    regime_id: int = 0
    regime_name: str = "LIQUIDITY_PBC"
    regime_probability: float = 1.0
    # New fields for tracking sweep and liquidity
    sweep_high: float = 0.0
    sweep_low: float = 0.0
    liquidity_target: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def target_weight(self) -> float:
        return self.lots


PVCPattern = PBCPattern


# ==================================================================
# 1. Sweep Level Scanner
# ==================================================================

class SweepLevelScanner:
    """Scans M30 data from scan_start_server_time to scan_end_server_time for daily HH/LL levels."""

    def __init__(self, config: LiquidityConfig):
        self.config = config

    def scan_daily_levels(
        self,
        m30_data: pd.DataFrame,
        session_date: datetime,
        scan_end_time: dt_time | None = None,
    ) -> SweepLevel | None:
        """Find HighestHigh and LowestLow from scan_start_time to scan_end_time on session_date.

        Args:
            m30_data: M30 OHLCV with DatetimeIndex.
            session_date: Date to scan.
            scan_end_time: End of scan window (e.g. 09:30).

        Returns:
            SweepLevel or None if insufficient data.
        """
        day = session_date.date() if hasattr(session_date, "date") else session_date
        scan_start = self.config.scan_start_time
        scan_end = scan_end_time or self.config.scan_end_time

        pre_session = m30_data[
            (m30_data.index.date == day) &
            (m30_data.index.time >= scan_start) &
            (m30_data.index.time < scan_end)
        ]

        if pre_session.empty:
            LOGGER.info(
                "LEVELS: no M30 bars from %s to %s on %s",
                scan_start,
                scan_end,
                day,
            )
            return None

        hh_idx = pre_session["high"].idxmax()
        ll_idx = pre_session["low"].idxmin()
        hh = float(pre_session["high"].max())
        ll = float(pre_session["low"].min())

        LOGGER.info(
            "SCANNER: usando barras M30 de %s a %s | encontradas=%d | HH=%.1f | LL=%.1f",
            scan_start,
            scan_end,
            len(pre_session),
            hh,
            ll,
        )

        LOGGER.info(
            "LEVELS: HH=%.1f at %s | LL=%.1f at %s",
            hh, hh_idx, ll, ll_idx,
        )

        return SweepLevel(
            highest_high=hh,
            lowest_low=ll,
            calculated_at=datetime.now(),
            hh_bar_time=hh_idx.to_pydatetime() if hasattr(hh_idx, "to_pydatetime") else hh_idx,
            ll_bar_time=ll_idx.to_pydatetime() if hasattr(ll_idx, "to_pydatetime") else ll_idx,
        )


# ==================================================================
# 2. Sweep Detector
# ==================================================================

class SweepDetector:
    """Detects when price sweeps above HH or below LL on M5."""

    def detect(
        self,
        m5_data: pd.DataFrame,
        sweep_level: SweepLevel,
        session_start: pd.Timestamp,
        session_end: pd.Timestamp,
    ) -> SweepInfo | None:
        """Detect sweep direction and track min/max during sweep.

        Returns:
            SweepInfo with direction, sweep_high, sweep_low if sweep detected.
            Returns FIRST sweep detected, allowing fallback to opposite side.
            
        CORRECCIÓN 3: Permite monitorear ambos lados.
        Si ambos lados son barridos, retorna el PRIMERO.
        Si el primero es rechazado, el sistema puede intentar el segundo.
        """
        session = m5_data[
            (m5_data.index >= session_start) & (m5_data.index <= session_end)
        ]
        if session.empty:
            LOGGER.info("SWEEP: no M5 bars in session window")
            return None

        hh_swept = False
        ll_swept = False
        first_sweep: str | None = None
        first_sweep_bar = None
        first_sweep_price = 0.0
        first_sweep_time = None
        sweep_high = 0.0
        sweep_low = float('inf')
        
        # Track both sweePs for potential fallback
        hh_sweep_time = None
        ll_sweep_time = None

        for bar_time, bar in session.iterrows():
            # Track min/max prices during sweep period
            sweep_high = max(sweep_high, bar["high"])
            sweep_low = min(sweep_low, bar["low"])

            if not hh_swept and bar["high"] >= sweep_level.highest_high:
                hh_swept = True
                hh_sweep_time = bar_time
                if first_sweep is None:
                    first_sweep = "SELL"
                    first_sweep_bar = bar_time
                    first_sweep_price = bar["high"]
                    first_sweep_time = bar_time

            if not ll_swept and bar["low"] <= sweep_level.lowest_low:
                ll_swept = True
                ll_sweep_time = bar_time
                if first_sweep is None:
                    first_sweep = "BUY"
                    first_sweep_bar = bar_time
                    first_sweep_price = bar["low"]
                    first_sweep_time = bar_time

        if first_sweep:
            sweep_log = (
                f"SWEEP: {first_sweep} swept at {first_sweep_price:.1f} (bar {first_sweep_bar}) | "
                f"sweep_range=[{sweep_low:.1f}, {sweep_high:.1f}]"
            )
            
            # Log if both sides were swept
            if hh_swept and ll_swept:
                opposite = "SELL" if first_sweep == "BUY" else "BUY"
                sweep_log += f" | Also {opposite} swept at {hh_sweep_time if first_sweep == 'BUY' else ll_sweep_time}"
                LOGGER.info(sweep_log + " — will attempt both sides")
            else:
                LOGGER.info(sweep_log)
            
            return SweepInfo(
                direction=first_sweep,
                sweep_high=sweep_high,
                sweep_low=sweep_low,
                first_sweep_time=first_sweep_time,
            )
        else:
            LOGGER.info("NO SIGNAL: no sweep detected during session")

        return None


# ==================================================================
# 3. PBC Detector
# ==================================================================

class PBCDetector:
    """Detects 3-candle PBC pattern on M5 after a sweep event."""

    def detect_in_window(
        self,
        post_sweep: pd.DataFrame,
        direction: str,
        sweep_level: SweepLevel,
        config: LiquidityConfig,
        sweep_high: float,
        sweep_low: float,
        sweep_time: pd.Timestamp | datetime | None = None,
    ) -> PBCPattern | None:
        """Evaluate the latest PBC candidate in a filtered post-sweep window.

        The caller is responsible for passing only bars up to the current bar.
        """
        if len(post_sweep) < 3:
            return None

        c1 = post_sweep.iloc[-3]
        c2 = post_sweep.iloc[-2]
        c3 = post_sweep.iloc[-1]
        c1_time = post_sweep.index[-3]
        c2_time = post_sweep.index[-2]
        c3_time = post_sweep.index[-1]

        sweep_info = SweepInfo(
            direction=direction,
            sweep_high=sweep_high,
            sweep_low=sweep_low,
            first_sweep_time=sweep_time if sweep_time is not None else c1_time,
        )

        return self._check_pbc_triplet(
            c1, c2, c3, c1_time, c2_time, c3_time,
            sweep_info, sweep_level, config,
        )

    def detect(
        self,
        m5_data: pd.DataFrame,
        sweep_info: SweepInfo,
        sweep_level: SweepLevel,
        config: LiquidityConfig,
        session_start: pd.Timestamp,
        session_end: pd.Timestamp,
    ) -> PBCPattern | None:
        """Compatibility wrapper that preserves the old API."""
        session = m5_data[
            (m5_data.index >= session_start) & (m5_data.index <= session_end)
        ]
        if len(session) < 3:
            LOGGER.info("NO SIGNAL: insufficient M5 bars for PBC detection")
            return None

        sweep_bar_idx = self._find_sweep_bar(session, sweep_info, sweep_level)
        if sweep_bar_idx is None:
            return None

        post_sweep = session.iloc[sweep_bar_idx + 1:]
        if len(post_sweep) < 3:
            LOGGER.info("NO SIGNAL: insufficient bars after sweep for PBC")
            return None

        return self.detect_in_window(
            post_sweep,
            sweep_info.direction,
            sweep_level,
            config,
            sweep_info.sweep_high,
            sweep_info.sweep_low,
            sweep_info.first_sweep_time,
        )

    def _find_sweep_bar(
        self,
        session: pd.DataFrame,
        sweep_info: SweepInfo,
        sweep_level: SweepLevel,
    ) -> int | None:
        """Return iloc index of the bar that caused the sweep."""
        for i in range(len(session)):
            bar = session.iloc[i]
            if sweep_info.direction == "SELL" and bar["high"] >= sweep_level.highest_high:
                return i
            if sweep_info.direction == "BUY" and bar["low"] <= sweep_level.lowest_low:
                return i
        return None

    def _check_pbc_triplet(
        self,
        c1: pd.Series,
        c2: pd.Series,
        c3: pd.Series,
        c1_time: pd.Timestamp,
        c2_time: pd.Timestamp,
        c3_time: pd.Timestamp,
        sweep_info: SweepInfo,
        sweep_level: SweepLevel,
        config: LiquidityConfig,
    ) -> PBCPattern | None:
        """Check if a 3-candle sequence forms a valid PBC pattern.

        BUY after sweep down:
        - candle 1 bullish
        - candle 2 bearish
        - candle 3 bullish and closes above candle 2 high
        - entry above candle 3 high + offset
        - SL below candle 2 low

        SELL after sweep up:
        - candle 1 bearish
        - candle 2 bullish
        - candle 3 bearish and closes below candle 2 low
        - entry below candle 3 low - offset
        - SL above candle 2 high
        """
        if sweep_info.direction == "BUY":
            if not (c1["close"] > c1["open"] and c2["close"] < c2["open"] and c3["close"] > c3["open"]):
                return None

            c1_body = abs(c1["close"] - c1["open"])
            c2_body = abs(c2["close"] - c2["open"])
            c3_body = abs(c3["close"] - c3["open"])
            if min(c1_body, c2_body, c3_body) < config.min_candle_body_points:
                LOGGER.debug(
                    "PBC: REJECTED BUY reason=BODY_TOO_SMALL c1_body=%.1f c2_body=%.1f c3_body=%.1f min=%.1f",
                    c1_body, c2_body, c3_body, config.min_candle_body_points,
                )
                return None

            if float(c3["close"]) <= float(c2["high"]):
                LOGGER.debug(
                    "PBC: INVALID BUY reason=C3_NO_BREAK c3_close=%.1f c2_high=%.1f",
                    float(c3["close"]), float(c2["high"]),
                )
                return None

            zona_high = max(float(c2["high"]), float(c3["high"]))
            zona_low = min(float(c2["low"]), float(c3["low"]))
            entry = float(c3["high"]) + config.entry_offset_points
            sl = float(c2["low"]) - config.sl_offset_points
            tp = sweep_level.highest_high
            sl_distance = entry - sl
            rr = abs(tp - entry) / sl_distance if sl_distance > EPSILON else 0.0

        elif sweep_info.direction == "SELL":
            if not (c1["close"] < c1["open"] and c2["close"] > c2["open"] and c3["close"] < c3["open"]):
                return None

            c1_body = abs(c1["close"] - c1["open"])
            c2_body = abs(c2["close"] - c2["open"])
            c3_body = abs(c3["close"] - c3["open"])
            if min(c1_body, c2_body, c3_body) < config.min_candle_body_points:
                LOGGER.debug(
                    "PBC: REJECTED SELL reason=BODY_TOO_SMALL c1_body=%.1f c2_body=%.1f c3_body=%.1f min=%.1f",
                    c1_body, c2_body, c3_body, config.min_candle_body_points,
                )
                return None

            if float(c3["close"]) >= float(c2["low"]):
                LOGGER.debug(
                    "PBC: INVALID SELL reason=C3_NO_BREAK c3_close=%.1f c2_low=%.1f",
                    float(c3["close"]), float(c2["low"]),
                )
                return None

            zona_high = max(float(c2["high"]), float(c3["high"]))
            zona_low = min(float(c2["low"]), float(c3["low"]))
            entry = float(c3["low"]) - config.entry_offset_points
            sl = float(c2["high"]) + config.sl_offset_points
            tp = sweep_level.lowest_low
            sl_distance = sl - entry
            rr = abs(entry - tp) / sl_distance if sl_distance > EPSILON else 0.0

        else:
            return None

        if sl_distance < config.min_sl_points:
            LOGGER.info(
                "PBC: REJECTED reason=SL_TOO_SMALL sl_dist=%.1f min=%.1f",
                sl_distance, config.min_sl_points,
            )
            return PBCPattern(
                direction=sweep_info.direction,
                vela_1_open=float(c1["open"]), vela_1_high=float(c1["high"]),
                vela_1_low=float(c1["low"]), vela_1_close=float(c1["close"]),
                vela_1_time=c1_time,
                vela_2_open=float(c2["open"]), vela_2_high=float(c2["high"]),
                vela_2_low=float(c2["low"]), vela_2_close=float(c2["close"]),
                vela_2_time=c2_time,
                vela_3_open=float(c3["open"]), vela_3_high=float(c3["high"]),
                vela_3_low=float(c3["low"]), vela_3_close=float(c3["close"]),
                vela_3_time=c3_time,
                zona_high=zona_high, zona_low=zona_low,
                entry_price=entry, stop_loss=sl, take_profit=tp,
                fib_ratio=0.0,
                is_valid=False,
                invalidation_reason=f"SL distance {sl_distance:.1f} < min {config.min_sl_points}",
            )

        if sl_distance > config.max_sl_points:
            LOGGER.info(
                "PBC: REJECTED reason=SL_TOO_LARGE sl_dist=%.1f max=%.1f",
                sl_distance, config.max_sl_points,
            )
            return PBCPattern(
                direction=sweep_info.direction,
                vela_1_open=float(c1["open"]), vela_1_high=float(c1["high"]),
                vela_1_low=float(c1["low"]), vela_1_close=float(c1["close"]),
                vela_1_time=c1_time,
                vela_2_open=float(c2["open"]), vela_2_high=float(c2["high"]),
                vela_2_low=float(c2["low"]), vela_2_close=float(c2["close"]),
                vela_2_time=c2_time,
                vela_3_open=float(c3["open"]), vela_3_high=float(c3["high"]),
                vela_3_low=float(c3["low"]), vela_3_close=float(c3["close"]),
                vela_3_time=c3_time,
                zona_high=zona_high, zona_low=zona_low,
                entry_price=entry, stop_loss=sl, take_profit=tp,
                fib_ratio=0.0,
                is_valid=False,
                invalidation_reason=f"SL distance {sl_distance:.1f} > max {config.max_sl_points}",
            )

        fib_ratio = self._compute_fib_ratio(sweep_info.direction, sweep_level, zona_high, zona_low)

        if rr < config.min_rr_to_trade:
            reason = f"RR_too_low (RR={rr:.2f} < min={config.min_rr_to_trade})"
            LOGGER.info(
                "PBC: REJECTED reason=%s rr=%.2f",
                reason, rr,
            )
            return PBCPattern(
                direction=sweep_info.direction,
                vela_1_open=float(c1["open"]), vela_1_high=float(c1["high"]),
                vela_1_low=float(c1["low"]), vela_1_close=float(c1["close"]),
                vela_1_time=c1_time,
                vela_2_open=float(c2["open"]), vela_2_high=float(c2["high"]),
                vela_2_low=float(c2["low"]), vela_2_close=float(c2["close"]),
                vela_2_time=c2_time,
                vela_3_open=float(c3["open"]), vela_3_high=float(c3["high"]),
                vela_3_low=float(c3["low"]), vela_3_close=float(c3["close"]),
                vela_3_time=c3_time,
                zona_high=zona_high, zona_low=zona_low,
                entry_price=entry, stop_loss=sl, take_profit=tp,
                fib_ratio=fib_ratio,
                is_valid=False,
                invalidation_reason=reason,
            )

        LOGGER.info(
            "PBC: pattern found dir=%s zona=[%.1f, %.1f] fib=%.3f RR=%.2f",
            sweep_info.direction, zona_low, zona_high, fib_ratio, rr,
        )

        return PBCPattern(
            direction=sweep_info.direction,
            vela_1_open=float(c1["open"]), vela_1_high=float(c1["high"]),
            vela_1_low=float(c1["low"]), vela_1_close=float(c1["close"]),
            vela_1_time=c1_time,
            vela_2_open=float(c2["open"]), vela_2_high=float(c2["high"]),
            vela_2_low=float(c2["low"]), vela_2_close=float(c2["close"]),
            vela_2_time=c2_time,
            vela_3_open=float(c3["open"]), vela_3_high=float(c3["high"]),
            vela_3_low=float(c3["low"]), vela_3_close=float(c3["close"]),
            vela_3_time=c3_time,
            zona_high=zona_high, zona_low=zona_low,
            entry_price=entry, stop_loss=sl, take_profit=tp,
            fib_ratio=fib_ratio,
            is_valid=True,
        )

    @staticmethod
    def _compute_fib_ratio(
        direction: str,
        sweep_level: SweepLevel,
        zona_high: float,
        zona_low: float,
    ) -> float:
        """Compute Fibonacci retracement ratio of pattern relative to sweep impulse.

        For BUY:  impulse goes from HH down to zona_low.
                  fib = (HH - zona_high) / (HH - zona_low)
                  Higher ratio = pattern closer to sweep level = better.

        For SELL: impulse goes from LL up to zona_high.
                  fib = (zona_low - LL) / (zona_high - LL)
                  Higher ratio = pattern closer to sweep level = better.
        """
        if direction == "BUY":
            impulse = sweep_level.highest_high - zona_low
            if impulse < EPSILON:
                return 0.0
            retracement = sweep_level.highest_high - zona_high
            return retracement / impulse
        if direction == "SELL":
            impulse = zona_high - sweep_level.lowest_low
            if impulse < EPSILON:
                return 0.0
            retracement = zona_low - sweep_level.lowest_low
            return retracement / impulse
        return 0.0

PVCDetector = PBCDetector
# ==================================================================
# 4. Zone Manager
# ==================================================================

class ZoneManager:
    """Manages limit-order zone updates when a better PBC pattern appears."""

    def __init__(self) -> None:
        self.current_pattern: PBCPattern | None = None

    def update_zone(
        self,
        new_pattern: PBCPattern,
    ) -> PBCPattern:
        """Replace current zone with new_pattern if it offers a better entry.

        Returns the active pattern (new or existing).
        """
        if self.current_pattern is None:
            self.current_pattern = new_pattern
            return new_pattern

        old = self.current_pattern

        # Better price: for BUY a lower entry is better; for SELL a higher entry.
        is_better = False
        if new_pattern.direction == "BUY" and new_pattern.entry_price < old.entry_price:
            is_better = True
        elif new_pattern.direction == "SELL" and new_pattern.entry_price > old.entry_price:
            is_better = True

        if is_better:
            LOGGER.info(
                "ZONE UPDATE: old_entry=%.1f → new_entry=%.1f",
                old.entry_price, new_pattern.entry_price,
            )
            self.current_pattern = new_pattern
            return new_pattern

        return old

    def reset(self) -> None:
        self.current_pattern = None


# ==================================================================
# 5. Position Sizing
# ==================================================================

def calculate_position_size(
    account_equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    contract_size: float = 1.0,
) -> float:
    """Calculate lots for US30m.

    lots = (equity * risk%) / (SL_distance * value_per_point_per_lot)
    """
    sl_distance = abs(entry_price - stop_loss)
    if sl_distance < EPSILON:
        return 0.0
    risk_amount = account_equity * risk_pct
    raw_lots = risk_amount / (sl_distance * contract_size)
    return max(0.01, round(raw_lots, 2))


# ==================================================================
# 6. Liquidity Strategy (Orchestrator)
# ==================================================================

class LiquidityStrategy:
    """Orchestrates Levels -> Sweep -> PBC -> Fib -> Zone -> Signal.

    NOTE on order execution: this strategy generates STOP orders
    (``signal.order_type == "BUY_STOP"`` or ``"SELL_STOP"``).
    The current ``OrderExecutor`` only supports market execution
    (``TRADE_ACTION_DEAL``).  To execute stop orders you must extend
    ``OrderExecutor.submit_order`` to handle ``mt5.TRADE_ACTION_PENDING``
    with ``mt5.ORDER_TYPE_BUY_STOP`` / ``mt5.ORDER_TYPE_SELL_STOP``.

    Break-even management: when price reaches 1:1 RR, SL moves to entry.
    """

    def __init__(self, config: LiquidityConfig | None = None) -> None:
        self.config = config or LiquidityConfig()
        self.level_scanner = SweepLevelScanner(config)
        self.sweep_detector = SweepDetector()
        self.pbc_detector = PBCDetector()
        self.zone_manager = ZoneManager()

    def evaluate_session(
        self,
        m30_data: pd.DataFrame,
        m5_data: pd.DataFrame,
        session_date: pd.Timestamp | datetime,
        account_equity: float = 100_000.0,
    ) -> Signal | None:
        """Run the full strategy pipeline for one trading day.

        Args:
            m30_data: M30 OHLCV with DatetimeIndex.
            m5_data:  M5 OHLCV with DatetimeIndex.
            session_date: The calendar date of the session.
            account_equity: Current account equity for sizing.

        Returns:
            Signal (STOP) if valid setup found, None otherwise.
        """
        day = (
            session_date.date()
            if hasattr(session_date, "date") and callable(session_date.date)
            else session_date
        )
        session_start = pd.Timestamp.combine(day, self.config.session_start_time)
        session_end = pd.Timestamp.combine(day, self.config.session_end_time)

        LOGGER.info("=" * 50)
        LOGGER.info(
            "SESSION: %s | %s to %s",
            day, self.config.session_start_server_time, self.config.session_end_server_time,
        )

        # ── Phase 1: Daily liquidity levels ──────────────────
        levels = self.level_scanner.scan_daily_levels(
            m30_data, session_date, self.config.scan_end_time,
        )
        if levels is None:
            LOGGER.info("NO SIGNAL: no liquidity levels found")
            return None

        session = m5_data[
            (m5_data.index.date == day)
            & (m5_data.index.time >= self.config.session_start_time)
            & (m5_data.index.time <= self.config.session_end_time)
        ]
        if session.empty:
            LOGGER.info("NO SIGNAL: no M5 bars in session window")
            return None

        # ── Phase 2: Event-driven sweep monitoring ───────────
        self.zone_manager.reset()

        swept_high = False
        swept_low = False
        sell_sweep_bar = None
        buy_sweep_bar = None
        active_direction = None
        trade_taken = False
        running_high = float("-inf")
        running_low = float("inf")
        entry_cutoff_dt = pd.Timestamp.combine(day, self.config.entry_cutoff_time)

        for i, (bar_time, bar) in enumerate(session.iterrows()):
            running_high = max(running_high, float(bar["high"]))
            running_low = min(running_low, float(bar["low"]))

            if bar_time.time() > self.config.entry_cutoff_time:
                LOGGER.debug(
                    "SKIP: bar=%s after entry_cutoff=%s",
                    bar_time,
                    self.config.entry_cutoff_time,
                )
                continue

            if trade_taken:
                break

            # Detectar sweeps
            if not swept_high and bar["high"] >= levels.highest_high:
                swept_high = True
                sell_sweep_bar = i
                active_direction = "SELL"
                LOGGER.info("SWEEP SELL at %s price=%.1f", bar_time, float(bar["high"]))

            if not swept_low and bar["low"] <= levels.lowest_low:
                swept_low = True
                buy_sweep_bar = i
                active_direction = "BUY"
                LOGGER.info("SWEEP BUY at %s price=%.1f", bar_time, float(bar["low"]))

            # Buscar PBC SELL si hay sweep alcista y han pasado al menos 2 barras
            if swept_high and sell_sweep_bar is not None:
                bars_since_sell_sweep = i - sell_sweep_bar
                if bars_since_sell_sweep >= 3:
                    visible_m5 = session.iloc[:i + 1]
                    post_sweep = visible_m5.iloc[sell_sweep_bar:]
                    if len(post_sweep) >= 2:
                        pattern = self.pbc_detector.detect_in_window(
                            post_sweep,
                            "SELL",
                            levels,
                            self.config,
                            running_high,
                            running_low,
                            bar_time,
                        )
                        if pattern is not None and pattern.is_valid:
                            signal = self._build_signal(
                                pattern,
                                levels,
                                SweepInfo("SELL", running_high, running_low, bar_time, float(bar["high"])),
                                account_equity,
                            )
                            if signal is not None:
                                trade_taken = True
                                return signal

            # Buscar PBC BUY si hay sweep bajista y han pasado al menos 2 barras
            if swept_low and buy_sweep_bar is not None:
                bars_since_buy_sweep = i - buy_sweep_bar
                if bars_since_buy_sweep >= 3:
                    visible_m5 = session.iloc[:i + 1]
                    post_sweep = visible_m5.iloc[buy_sweep_bar:]
                    if len(post_sweep) >= 2:
                        pattern = self.pbc_detector.detect_in_window(
                            post_sweep,
                            "BUY",
                            levels,
                            self.config,
                            running_high,
                            running_low,
                            bar_time,
                        )
                        if pattern is not None and pattern.is_valid:
                            signal = self._build_signal(
                                pattern,
                                levels,
                                SweepInfo("BUY", running_high, running_low, bar_time, float(bar["low"])),
                                account_equity,
                            )
                            if signal is not None:
                                trade_taken = True
                                return signal

            # Ambos sweepados sin señal válida
            if swept_high and swept_low:
                all_bars_processed = (i == len(session) - 1)
                if all_bars_processed:
                    LOGGER.info("NO SIGNAL: both sides swept, no valid PBC found")
                    break

        return None
        return None

    def _build_signal(
        self,
        pattern: PBCPattern,
        levels: SweepLevel,
        sweep_info: SweepInfo,
        account_equity: float,
    ) -> Signal | None:
        """Construct a STOP Signal from a validated PBC pattern.

        Verifica antes de crear Signal que el RR mínimo se cumpla.
        Si no, return None inmediatamente.
        """
        entry = pattern.entry_price
        sl = pattern.stop_loss
        tp = pattern.take_profit

        direction = "LONG" if pattern.direction == "BUY" else "SHORT"

        # Calculate RR for validation as reward / risk
        if pattern.direction == "BUY":
            rr = abs(tp - entry) / (entry - sl) if (entry - sl) > EPSILON else 0.0
        else:
            rr = abs(entry - tp) / (sl - entry) if (sl - entry) > EPSILON else 0.0

        # KILLSWITCH 1: RR validation
        if rr < self.config.min_rr_to_trade:
            LOGGER.info(
                "REJECTED: RR=%.2f < min=%.2f",
                rr, self.config.min_rr_to_trade,
            )
            return None

        lots = calculate_position_size(
            account_equity=account_equity,
            risk_pct=self.config.risk_percent,
            entry_price=entry,
            stop_loss=sl,
            contract_size=self.config.contract_size,
        )
        risk_amount = account_equity * self.config.risk_percent

        order_type = "BUY_STOP" if direction == "LONG" else "SELL_STOP"

        reasoning = (
            f"PBC {direction}: sweep={sweep_info.direction} | "
            f"zona=[{pattern.zona_low:.1f}, {pattern.zona_high:.1f}] | "
            f"fib={pattern.fib_ratio:.3f} | RR={rr:.2f}"
        )

        LOGGER.info(
            "TRADE PARAMS: entry=%.1f SL=%.1f TP=%.1f RR=%.2f sweep_extreme=%.1f liquidity_target=%.1f",
            entry, sl, tp, rr,
            sweep_info.sweep_high if pattern.direction == "SELL" else sweep_info.sweep_low,
            tp,
        )

        ts = (
            pattern.vela_3_time.to_pydatetime()
            if hasattr(pattern.vela_3_time, "to_pydatetime")
            else pattern.vela_3_time
        )

        return Signal(
            symbol=self.config.symbol,
            direction=direction,
            confidence=1.0,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            lots=lots,
            position_size=risk_amount,
            timestamp=ts if isinstance(ts, datetime) else datetime.now(),
            reasoning=reasoning,
            order_type=order_type,
            sweep_high=sweep_info.sweep_high,
            sweep_low=sweep_info.sweep_low,
            liquidity_target=tp,
            metadata={
                "hh_level": levels.highest_high,
                "ll_level": levels.lowest_low,
                "hh_bar_time": str(levels.hh_bar_time),
                "ll_bar_time": str(levels.ll_bar_time),
                "sweep_direction": sweep_info.direction,
                "sweep_price": sweep_info.sweep_price,
                "sweep_bar_time": str(sweep_info.first_sweep_time),
                "pbc_zona_high": pattern.zona_high,
                "pbc_zona_low": pattern.zona_low,
                "pbc_vela_a_time": str(pattern.vela_1_time),
                "pbc_vela_b_time": str(pattern.vela_2_time),
                "pbc_vela_c_time": str(pattern.vela_3_time),
                "fib_ratio": pattern.fib_ratio,
                "rr_calculated": rr,
                "magic_number": self.config.magic_number,
                "sl_distance": abs(entry - sl),
                "breakeven_at_rr": self.config.breakeven_at_rr,
                "max_trade_duration_minutes": self.config.max_trade_duration_minutes,
                "session_end": self.config.session_end_server_time,
            },
        )

    def check_rejection_exit(
        self,
        m5_data: pd.DataFrame,
        current_price: float,
        take_profit: float,
        direction: str,
    ) -> bool:
        """Check for rejection candle pattern near TP level for early exit.

        Args:
            m5_data: M5 OHLCV data.
            current_price: Current price.
            take_profit: Target TP level.
            direction: "BUY" or "SELL".

        Returns:
            True if rejection pattern detected near TP, False otherwise.
        """
        if len(m5_data) < 2:
            return False

        distance_to_tp = abs(current_price - take_profit)

        # Check if close to TP
        if distance_to_tp >= self.config.rejection_zone_points:
            return False

        # Check last 2 candles for rejection
        c1 = m5_data.iloc[-2]
        c2 = m5_data.iloc[-1]

        if direction == "SELL":
            # Looking for lower wick > body (bullish rejection)
            wick_lower = min(c1["open"], c1["close"]) - c1["low"]
            body = abs(c1["open"] - c1["close"])
            if body < EPSILON:
                body = c1["high"] - c1["low"]

            rejection_1 = wick_lower > body * 1.5

            wick_lower_2 = min(c2["open"], c2["close"]) - c2["low"]
            body_2 = abs(c2["open"] - c2["close"])
            if body_2 < EPSILON:
                body_2 = c2["high"] - c2["low"]

            rejection_2 = wick_lower_2 > body_2 * 1.5

            if rejection_1 or rejection_2:
                LOGGER.info(
                    "REJECTION: SELL setup rejected at %.1f (TP=%.1f)",
                    current_price, take_profit,
                )
                return True

        else:  # BUY
            # Looking for upper wick > body (bearish rejection)
            wick_upper = c1["high"] - max(c1["open"], c1["close"])
            body = abs(c1["open"] - c1["close"])
            if body < EPSILON:
                body = c1["high"] - c1["low"]

            rejection_1 = wick_upper > body * 1.5

            wick_upper_2 = c2["high"] - max(c2["open"], c2["close"])
            body_2 = abs(c2["open"] - c2["close"])
            if body_2 < EPSILON:
                body_2 = c2["high"] - c2["low"]

            rejection_2 = wick_upper_2 > body_2 * 1.5

            if rejection_1 or rejection_2:
                LOGGER.info(
                    "REJECTION: BUY setup rejected at %.1f (TP=%.1f)",
                    current_price, take_profit,
                )
                return True

        return False
