#!/usr/bin/env python3
"""Download NASDAQ 100 (QQQ) historical data via yfinance."""

import yfinance as yf
import pandas as pd
from pathlib import Path

# Create data/feeds directory if missing
Path("data/feeds").mkdir(parents=True, exist_ok=True)

print("[*] Downloading NASDAQ 100 (QQQ) daily data from 2024-01-01 to today...")

# Download daily bars (1500+ to ensure >504 features after zscore)
df = yf.download(
    "QQQ",
    start="2022-01-01",  # Extended to 2022 for more historical data
    end=pd.Timestamp.today().strftime("%Y-%m-%d"),
    interval="1d",
    progress=False
)

# Handle MultiIndex columns (yfinance returns MultiIndex for single ticker)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Normalize column names to lowercase
df.columns = df.columns.str.lower()

# Select required columns (yfinance returns: open, high, low, close, adj close, volume)
df = df[["open", "high", "low", "close", "volume"]].copy()
output_path = "data/feeds/NAS100_D1.csv"
df.to_csv(output_path)

print(f"[OK] Downloaded {len(df)} daily bars")
print(f"[OK] Saved to {output_path}")
print(f"\nData preview (last 5 bars):")
print(df.tail(5))
print(f"\nData shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
