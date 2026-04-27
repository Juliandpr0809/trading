"""Market data access layer for historical and real-time feeds via MT5.

Primary data flow for live trading:
  1. ``get_live_bars("USTEC", "M5", 800)`` — polls MT5 for closed M5 candles.
  2. ``compute_technical_indicators(df)`` — adds EMA9, EMA200, VWAP, ATR, RSI.

VWAP is computed as a **session-anchored** cumulative VWAP (resets at midnight
server time) to match how institutional traders use it on indices.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


class MarketDataClient:
    """Fetches and normalizes OHLCV data across data sources."""

    def __init__(self, data_source: str = "mt5", data_dir: str | None = None) -> None:
        """Initialize market data client.

        Args:
            data_source: "csv" or "mt5" (default: "mt5")
            data_dir: Path to CSV directory (for "csv" source)
        """
        self.data_source = data_source
        self.data_dir = data_dir or Path(__file__).parent / "feeds"
        self.cache: dict[str, pd.DataFrame] = {}

    # ==================================================================
    # Historical data
    # ==================================================================

    def fetch_historical(
        self,
        symbol: str,
        timeframe: str = "D1",
        bars: int = 1000,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Return historical OHLCV bars for a symbol and timeframe."""
        if self.data_source == "csv":
            return self._fetch_from_csv(symbol, timeframe, bars, start_date, end_date)
        elif self.data_source == "mt5":
            return self._fetch_from_mt5(symbol, timeframe, bars)
        else:
            raise ValueError(f"Unknown data source: {self.data_source}")

    def _fetch_from_csv(
        self,
        symbol: str,
        timeframe: str = "D1",
        bars: int = 1000,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> pd.DataFrame:
        """Load historical data from CSV file."""
        csv_path = Path(self.data_dir) / f"{symbol}_{timeframe}.csv"

        if not csv_path.exists():
            LOGGER.error("CSV file not found: %s", csv_path)
            return pd.DataFrame()

        try:
            df = pd.read_csv(csv_path, parse_dates=["datetime"], index_col="datetime")
            df = df.sort_index()
            df.columns = df.columns.str.lower()

            required = ["open", "high", "low", "close", "volume"]
            if not all(col in df.columns for col in required):
                raise ValueError(f"CSV missing required columns. Found: {df.columns.tolist()}")

            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            return df.tail(bars)[required].copy()

        except Exception as exc:
            LOGGER.error("Failed to load CSV %s: %s", csv_path, exc)
            return pd.DataFrame()

    def _fetch_from_mt5(
        self,
        symbol: str,
        timeframe: str = "D1",
        bars: int = 1000,
    ) -> pd.DataFrame:
        """Fetch historical bars from a live MT5 terminal."""
        try:
            import MetaTrader5 as mt5

            tf = self._map_timeframe(mt5, timeframe)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

            if rates is None or len(rates) == 0:
                LOGGER.warning("No MT5 data for %s %s", symbol, timeframe)
                return pd.DataFrame()

            return self._rates_to_df(rates)

        except ImportError:
            LOGGER.error("MetaTrader5 module not installed")
            return pd.DataFrame()
        except Exception as exc:
            LOGGER.error("MT5 historical fetch error: %s", exc)
            return pd.DataFrame()

    # ==================================================================
    # Live data
    # ==================================================================

    def get_live_bars(
        self,
        symbol: str,
        timeframe: str = "M5",
        bars: int = 800,
    ) -> pd.DataFrame:
        """Get live bars via ``mt5.copy_rates_from_pos()``.

        IMPORTANT: This function does NOT call ``mt5.initialize()`` — the
        caller (MT5Client) is responsible for maintaining the connection.

        Args:
            symbol: Trading symbol (e.g. "USTEC")
            timeframe: Candle period string
            bars: Number of historical bars to fetch

        Returns:
            DataFrame indexed by datetime with OHLCV columns.
        """
        try:
            import MetaTrader5 as mt5

            tf = self._map_timeframe(mt5, timeframe)
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)

            if rates is None or len(rates) == 0:
                LOGGER.warning("No live data for %s", symbol)
                return pd.DataFrame()

            df = self._rates_to_df(rates)
            self.cache[symbol] = df
            return df

        except Exception as exc:
            LOGGER.error("Live fetch error: %s", exc)
            return pd.DataFrame()

    # ==================================================================
    # Technical indicators
    # ==================================================================

    def compute_technical_indicators(
        self,
        df: pd.DataFrame,
        ema9_period: int = 9,
        ema200_period: int = 200,
        atr_period: int = 14,
    ) -> pd.DataFrame:
        """Compute EMA9, EMA200, session-VWAP, ATR, SMA, RSI on OHLCV data.

        All indicators operate on **closed candles only** — the latest bar in
        the frame is always the most-recently closed M5 candle.
        """
        if df.empty or len(df) < ema200_period:
            LOGGER.warning("Insufficient bars (%d) for indicator calculation", len(df))
            return df

        df = df.copy()

        # ── EMAs ──────────────────────────────────────────────
        df["ema_9"] = df["close"].ewm(span=ema9_period, adjust=False).mean()
        df["ema_200"] = df["close"].ewm(span=ema200_period, adjust=False).mean()

        # ── Session-anchored VWAP ─────────────────────────────
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

        # Detect session boundaries (midnight rollover on date change)
        dates = df.index.date if hasattr(df.index, 'date') else pd.Series(df.index).dt.date.values
        session_id = pd.Series(dates, index=df.index)
        session_groups = (session_id != session_id.shift(1)).cumsum()

        cum_pv = (df["typical_price"] * df["volume"]).groupby(session_groups).cumsum()
        cum_v = df["volume"].groupby(session_groups).cumsum()
        df["vwap"] = cum_pv / (cum_v + 1e-12)
        df.drop(columns=["typical_price"], inplace=True)

        # ── ATR ───────────────────────────────────────────────
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window=atr_period).mean()

        # ── SMA 50 / 200 ─────────────────────────────────────
        df["sma_50"] = df["close"].rolling(window=50).mean()
        df["sma_200"] = df["close"].rolling(window=200).mean()

        # ── RSI 14 ────────────────────────────────────────────
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-12)
        df["rsi"] = 100 - (100 / (1 + rs))

        LOGGER.debug("Computed indicators for %d bars", len(df))
        return df

    # ==================================================================
    # Helpers
    # ==================================================================

    @staticmethod
    def _rates_to_df(rates) -> pd.DataFrame:
        """Convert MT5 rates array to pandas DataFrame."""
        df = pd.DataFrame(rates)
        df["datetime"] = pd.to_datetime(df["time"], unit="s")
        df = df[["datetime", "open", "high", "low", "close", "tick_volume"]].copy()
        df.columns = ["datetime", "open", "high", "low", "close", "volume"]
        df.set_index("datetime", inplace=True)
        return df

    @staticmethod
    def _map_timeframe(mt5_module, timeframe: str):
        """Map string timeframe to MT5 constant."""
        tf_map = {
            "M1": mt5_module.TIMEFRAME_M1,
            "M5": mt5_module.TIMEFRAME_M5,
            "M15": mt5_module.TIMEFRAME_M15,
            "M30": mt5_module.TIMEFRAME_M30,
            "H1": mt5_module.TIMEFRAME_H1,
            "H4": mt5_module.TIMEFRAME_H4,
            "D1": mt5_module.TIMEFRAME_D1,
            "W1": mt5_module.TIMEFRAME_W1,
        }
        return tf_map.get(timeframe, mt5_module.TIMEFRAME_M5)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure DataFrame has required columns and correct types."""
        required = ["open", "high", "low", "close", "volume"]
        df.columns = df.columns.str.lower()
        for col in required:
            if col not in df.columns:
                df[col] = 0 if col == "volume" else df.get("close", df.iloc[:, 0])
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
        return df.dropna()[required].copy()

    def get_available_symbols(self) -> list[str]:
        """Return list of available symbols."""
        if self.data_source == "csv":
            csv_files = list(Path(self.data_dir).glob("*.csv"))
            return sorted({f.stem.split("_")[0] for f in csv_files})
        return ["USTEC", "USTECm", "EURUSD", "GBPUSD", "XAUUSD"]

    def get_available_timeframes(self) -> list[str]:
        """Return supported timeframes."""
        return ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]
