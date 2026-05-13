"""Paper trading logger for Liquidity Sweep strategy.

Logs each setup detection (whether trade taken or not) for manual verification.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

PAPER_TRADING_LOG_PATH = Path("data/paper_trading_log.csv")


class PaperTradingLogger:
    """Logs paper trading setup detections and results for manual verification."""

    COLUMNS = [
        "date",
        "session_date",
        "setup_detected",
        "direction",
        "poi_type",
        "sweep_level",
        "pattern_candle_a",
        "pattern_candle_b",
        "entry_price",
        "stop_loss",
        "take_profit",
        "sl_distance",
        "tp_distance",
        "trader_agrees",
        "actual_result",
        "hypothetical_pnl",
        "notes",
    ]

    def __init__(self, log_path: str | Path = PAPER_TRADING_LOG_PATH):
        """Initialize paper trading logger.
        
        Args:
            log_path: Path to CSV file (creates if doesn't exist)
        """
        self.log_path = Path(log_path)
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Create CSV file with headers if it doesn't exist."""
        if not self.log_path.exists():
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            df = pd.DataFrame(columns=self.COLUMNS)
            df.to_csv(self.log_path, index=False)
            LOGGER.info("Created paper trading log: %s", self.log_path)
        else:
            LOGGER.debug("Paper trading log exists: %s", self.log_path)

    def log_setup_detected(
        self,
        session_date: pd.Timestamp | datetime,
        signal: Any,  # LiquidityStrategy.Signal
        poi_type: str,  # "BULLISH" or "BEARISH"
        sweep_level: float,
        pattern_candle_a: pd.Timestamp | str,
        pattern_candle_b: pd.Timestamp | str,
        sl_distance: float,
        tp_distance: float,
        notes: str = "",
    ) -> None:
        """Log a detected setup.
        
        Args:
            session_date: Date of the trading session
            signal: LiquidityStrategy.Signal object
            poi_type: "BULLISH" or "BEARISH"
            sweep_level: Price level where sweep occurred
            pattern_candle_a: First candle of confirmation pattern
            pattern_candle_b: Second candle of confirmation pattern
            sl_distance: Distance from entry to SL (in points)
            tp_distance: Distance from entry to TP (in points)
            notes: Additional notes (reason for rejection if any)
        """
        session_date_str = self._to_date_str(session_date)
        entry_price = float(signal.entry_price) if signal else None
        stop_loss = float(signal.stop_loss) if signal else None
        take_profit = float(signal.take_profit) if signal else None
        direction = signal.direction if signal else ""

        row = {
            "date": datetime.now().isoformat(),
            "session_date": session_date_str,
            "setup_detected": True,
            "direction": direction,
            "poi_type": poi_type,
            "sweep_level": round(sweep_level, 4) if sweep_level else "",
            "pattern_candle_a": str(pattern_candle_a),
            "pattern_candle_b": str(pattern_candle_b),
            "entry_price": round(entry_price, 4) if entry_price else "",
            "stop_loss": round(stop_loss, 4) if stop_loss else "",
            "take_profit": round(take_profit, 4) if take_profit else "",
            "sl_distance": round(sl_distance, 1) if sl_distance else "",
            "tp_distance": round(tp_distance, 1) if tp_distance else "",
            "trader_agrees": "",  # Manual fill
            "actual_result": "",  # Manual fill
            "hypothetical_pnl": "",  # Manual fill
            "notes": notes,
        }

        self._append_row(row)
        LOGGER.info(
            "Logged setup: %s %s @ %.2f | SL=%.1f | Notes: %s",
            direction, poi_type, entry_price, sl_distance, notes,
        )

    def log_setup_rejected(
        self,
        session_date: pd.Timestamp | datetime,
        reason: str,
        notes: str = "",
    ) -> None:
        """Log a rejected setup (e.g., SL out of range, entry cutoff exceeded).
        
        Args:
            session_date: Date of the trading session
            reason: Reason for rejection (e.g., "SL_TOO_LARGE", "AFTER_CUTOFF")
            notes: Additional context
        """
        session_date_str = self._to_date_str(session_date)

        row = {
            "date": datetime.now().isoformat(),
            "session_date": session_date_str,
            "setup_detected": False,
            "direction": "",
            "poi_type": "",
            "sweep_level": "",
            "pattern_candle_a": "",
            "pattern_candle_b": "",
            "entry_price": "",
            "stop_loss": "",
            "take_profit": "",
            "sl_distance": "",
            "tp_distance": "",
            "trader_agrees": "",
            "actual_result": "",
            "hypothetical_pnl": "",
            "notes": f"{reason}: {notes}",
        }

        self._append_row(row)
        LOGGER.info("Logged rejected setup: %s | %s", reason, notes)

    def log_no_setup_today(
        self,
        session_date: pd.Timestamp | datetime,
        reason: str = "No sweep detected",
    ) -> None:
        """Log a day with no setup detected.
        
        Args:
            session_date: Date of the trading session
            reason: Why no setup (e.g., "No POI", "No sweep", "No pattern")
        """
        session_date_str = self._to_date_str(session_date)

        row = {
            "date": datetime.now().isoformat(),
            "session_date": session_date_str,
            "setup_detected": False,
            "direction": "",
            "poi_type": "",
            "sweep_level": "",
            "pattern_candle_a": "",
            "pattern_candle_b": "",
            "entry_price": "",
            "stop_loss": "",
            "take_profit": "",
            "sl_distance": "",
            "tp_distance": "",
            "trader_agrees": "",
            "actual_result": "",
            "hypothetical_pnl": "",
            "notes": reason,
        }

        self._append_row(row)
        LOGGER.info("Logged no setup: %s", reason)

    def _append_row(self, row: dict[str, Any]) -> None:
        """Append a row to the CSV file."""
        try:
            df = pd.read_csv(self.log_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(self.log_path, index=False)
        except Exception as e:
            LOGGER.error("Failed to append row to paper trading log: %s", e)

    @staticmethod
    def _to_date_str(dt: pd.Timestamp | datetime | str) -> str:
        """Convert datetime to date string (YYYY-MM-DD)."""
        if isinstance(dt, str):
            return dt
        if isinstance(dt, pd.Timestamp):
            return dt.strftime("%Y-%m-%d")
        if isinstance(dt, datetime):
            return dt.strftime("%Y-%m-%d")
        return str(dt)

    def get_summary(self, days: int = 7) -> dict[str, Any]:
        """Get summary stats for the last N days.
        
        Args:
            days: Number of recent days to summarize
            
        Returns:
            Dict with total_days, setups_detected, trader_agreed, win_rate
        """
        try:
            df = pd.read_csv(self.log_path)
            
            if df.empty:
                return {
                    "total_days": 0,
                    "setups_detected": 0,
                    "trader_agreed": 0,
                    "win_rate": 0.0,
                }
            
            # Filter to recent days
            df["date"] = pd.to_datetime(df["date"])
            df = df[df["date"] >= (datetime.now() - pd.Timedelta(days=days))]
            
            if df.empty:
                return {
                    "total_days": 0,
                    "setups_detected": 0,
                    "trader_agreed": 0,
                    "win_rate": 0.0,
                }
            
            setups_detected = df["setup_detected"].sum()
            trader_agreed = df["trader_agrees"].notna().sum()
            
            # Win rate: count "WINNER" in actual_result
            winners = (df["actual_result"] == "WINNER").sum()
            win_rate = (winners / trader_agreed * 100) if trader_agreed > 0 else 0.0
            
            return {
                "total_days": df["session_date"].nunique(),
                "setups_detected": int(setups_detected),
                "trader_agreed": int(trader_agreed),
                "win_rate": round(win_rate, 1),
            }
        except Exception as e:
            LOGGER.error("Failed to compute summary: %s", e)
            return {
                "total_days": 0,
                "setups_detected": 0,
                "trader_agreed": 0,
                "win_rate": 0.0,
            }
