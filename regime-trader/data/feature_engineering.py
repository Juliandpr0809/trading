"""Feature computation utilities for modeling and signal generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import ROCIndicator, RSIIndicator
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

EPSILON = 1e-12


def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    """Return rolling z-score standardized series."""
    rolling_mean = series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = series.rolling(window=lookback, min_periods=lookback).std(ddof=0)
    return (series - rolling_mean) / (rolling_std + EPSILON)


def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    """Compute rolling linear-regression slope."""
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    denom = np.sum(x_centered * x_centered)

    def _slope(values: np.ndarray) -> float:
        y = values.astype(float)
        y_centered = y - y.mean()
        return float(np.sum(x_centered * y_centered) / (denom + EPSILON))

    return series.rolling(window=window, min_periods=window).apply(_slope, raw=True)


def validate_ohlcv_columns(bars: pd.DataFrame) -> None:
    """Ensure required OHLCV columns exist before feature generation."""
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")


def compute_raw_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute unscaled observable features from OHLCV data."""
    validate_ohlcv_columns(bars)

    n_bars = len(bars)
    vol_window = min(20, max(10, n_bars // 12))
    vol_z_window = min(50, max(20, n_bars // 8))
    volume_slope_window = min(10, max(5, n_bars // 40))
    sma_fast_window = min(50, max(20, n_bars // 10))
    sma_long_window = min(200, max(80, n_bars // 2))
    rsi_norm_window = min(252, max(63, n_bars // 2))
    roc_window = min(20, max(10, n_bars // 12))

    close = bars["close"].astype(float)
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    volume = bars["volume"].astype(float)

    log_return_1 = np.log(close / close.shift(1))
    log_return_5 = np.log(close / close.shift(5))
    log_return_20 = np.log(close / close.shift(20))

    realized_vol_20 = log_return_1.rolling(window=vol_window, min_periods=vol_window).std(ddof=0)
    ret_vol_ratio = log_return_5 / (realized_vol_20 + EPSILON)

    volume_mean_50 = volume.rolling(window=vol_z_window, min_periods=vol_z_window).mean()
    volume_std_50 = volume.rolling(window=vol_z_window, min_periods=vol_z_window).std(ddof=0)
    volume_z_50 = (volume - volume_mean_50) / (volume_std_50 + EPSILON)
    volume_sma_10 = volume.rolling(window=volume_slope_window, min_periods=volume_slope_window).mean()
    volume_trend = rolling_slope(volume_sma_10, window=volume_slope_window)

    adx_14 = ADXIndicator(high=high, low=low, close=close, window=14).adx()

    sma_50 = close.rolling(window=sma_fast_window, min_periods=sma_fast_window).mean()
    sma_50_slope = rolling_slope(sma_50, window=volume_slope_window)

    rsi_14 = RSIIndicator(close=close, window=14).rsi()
    rsi_14_mean = rsi_14.rolling(window=rsi_norm_window, min_periods=rsi_norm_window).mean()
    rsi_14_std = rsi_14.rolling(window=rsi_norm_window, min_periods=rsi_norm_window).std(ddof=0)
    rsi_z = (rsi_14 - rsi_14_mean) / (rsi_14_std + EPSILON)

    sma_200 = close.rolling(window=sma_long_window, min_periods=sma_long_window).mean()
    distance_sma_200_pct = (close - sma_200) / (close + EPSILON)

    roc_20 = ROCIndicator(close=close, window=roc_window).roc()

    atr_14 = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    normalized_atr = atr_14 / (close + EPSILON)

    raw = pd.DataFrame(
        {
            "ret_1": log_return_1,
            "ret_5": log_return_5,
            "ret_20": log_return_20,
            "realized_vol_20": realized_vol_20,
            "ret_vol_ratio": ret_vol_ratio,
            "volume_z_50": volume_z_50,
            "volume_trend": volume_trend,
            "adx_14": adx_14,
            "sma_50_slope": sma_50_slope,
            "rsi_z": rsi_z,
            "distance_sma_200_pct": distance_sma_200_pct,
            "roc_20": roc_20,
            "normalized_atr": normalized_atr,
        },
        index=bars.index,
    )
    return raw


def standardize_features(raw_features: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Standardize features with explicit warmup handling and selective NaN filtering."""
    if raw_features.empty:
        return raw_features.copy()

    standardized = pd.DataFrame(index=raw_features.index)
    for column in raw_features.columns:
        standardized[column] = rolling_zscore(raw_features[column], lookback=lookback)

    # Remove unstable warmup region explicitly instead of relying on global dropna.
    warmup = max(0, lookback - 1)
    if warmup > 0:
        standardized = standardized.iloc[warmup:].copy()

    # Replace infs early to avoid silently dropping every row downstream.
    standardized = standardized.replace([np.inf, -np.inf], np.nan)

    # Keep rows where core regime-driving features are available.
    core_features = [
        "ret_1",
        "realized_vol_20",
        "volume_z_50",
        "adx_14",
        "distance_sma_200_pct",
        "normalized_atr",
    ]
    available_core = [col for col in core_features if col in standardized.columns]
    if available_core:
        standardized = standardized.dropna(subset=available_core)

    # Columns that are fully NaN on short windows are neutralized instead of killing rows.
    for column in standardized.columns:
        if standardized[column].isna().all():
            standardized[column] = 0.0

    # Fill residual gaps from long-window features without dropping the whole frame.
    standardized = standardized.ffill().bfill().dropna()
    return standardized


def compute_observable_features(bars: pd.DataFrame, zscore_lookback: int = 252) -> pd.DataFrame:
    """Compute standardized HMM observables from OHLCV input with adaptive lookback."""
    raw = compute_raw_features(bars)
    if raw.empty:
        return raw

    # For short slices (e.g., intraday polling windows), prevent destructive standardization.
    adaptive_lookback = min(zscore_lookback, max(30, len(raw) // 3))
    return standardize_features(raw, lookback=adaptive_lookback)


class FeatureEngineer:
    """Builds technical and statistical features from price data."""

    def __init__(self, zscore_lookback: int = 252) -> None:
        """Initialize feature engineering pipeline settings."""
        self.zscore_lookback = zscore_lookback

    def transform(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Transform raw bars into model-ready feature matrix."""
        return compute_observable_features(bars, zscore_lookback=self.zscore_lookback)

    def compute_features(self, bars: pd.DataFrame) -> pd.DataFrame:
        """Backward-compatible alias used by main pipeline."""
        return self.transform(bars)
