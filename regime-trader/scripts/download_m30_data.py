#!/usr/bin/env python3
"""Download M30 historical data from MT5 (Exness) and save as CSV.

Usage:
    python scripts/download_m30_data.py
    python scripts/download_m30_data.py --bars 10000
    python scripts/download_m30_data.py --symbol USTEC_x100m

This script connects to your Exness MT5 terminal, downloads M30 candles,
and writes them to data/feeds/<SYMBOL>_M30.csv for offline backtesting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from broker.mt5_client import MT5Client


def resolve_symbol(mt5_module, preferred: str) -> str | None:
    """Try common NASDAQ aliases until one works."""
    candidates = [preferred, "USTEC_x100m", "USTECm", "USTEC", "NAS100m", "NAS100", "US100"]
    for sym in candidates:
        info = mt5_module.symbol_info(sym)
        if info is not None:
            if not info.visible:
                mt5_module.symbol_select(sym, True)
            return sym
    return None


def download_m30(symbol: str, bars: int, output_dir: Path) -> None:
    """Connect to MT5, download M30 bars, save as CSV."""

    # 1. Connect
    client = MT5Client()
    if not client.initialize():
        print("[FAIL] MT5 initialization failed")
        sys.exit(1)
    if not client.login():
        print("[FAIL] MT5 login failed")
        sys.exit(1)

    import MetaTrader5 as mt5

    # 2. Resolve symbol
    resolved = resolve_symbol(mt5, symbol)
    if resolved is None:
        print(f"[FAIL] Could not find symbol '{symbol}' or any NASDAQ alias")
        client.shutdown()
        sys.exit(1)
    print(f"[OK] Symbol resolved: {resolved}")

    # 3. Download M30 bars
    print(f"Downloading {bars:,} M30 bars for {resolved}...")
    rates = mt5.copy_rates_from_pos(resolved, mt5.TIMEFRAME_M30, 0, bars)

    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        print(f"[FAIL] No data returned. MT5 error: {err}")
        client.shutdown()
        sys.exit(1)

    # 4. Convert to DataFrame
    df = pd.DataFrame(rates)
    df["datetime"] = pd.to_datetime(df["time"], unit="s")
    df = df[["datetime", "open", "high", "low", "close", "tick_volume"]].copy()
    df.columns = ["datetime", "open", "high", "low", "close", "volume"]
    df.set_index("datetime", inplace=True)

    print(f"[OK] Received {len(df):,} bars")
    print(f"     From: {df.index[0]}")
    print(f"     To:   {df.index[-1]}")
    print(f"     Span: {(df.index[-1] - df.index[0]).days} days")

    # 5. Save to CSV
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save with the resolved symbol name
    csv_path = output_dir / f"{resolved}_M30.csv"
    df.to_csv(csv_path)
    print(f"[OK] Saved: {csv_path}  ({csv_path.stat().st_size / 1024:.0f} KB)")

    # Also save as USTEC_M30.csv and NAS100_M30.csv aliases
    for alias in ["USTEC_M30.csv", "NAS100_M30.csv"]:
        alias_path = output_dir / alias
        if alias_path.name != csv_path.name:
            df.to_csv(alias_path)
            print(f"[OK] Alias: {alias_path}")

    # 6. Quick data quality check
    null_count = df.isnull().sum().sum()
    zero_vol = (df["volume"] == 0).sum()
    gaps = df.index.to_series().diff().dt.total_seconds()
    median_gap = gaps.median()
    max_gap = gaps.max()

    print(f"\n--- Data Quality ---")
    print(f"  Nulls:       {null_count}")
    print(f"  Zero volume: {zero_vol} bars ({zero_vol/len(df)*100:.1f}%)")
    print(f"  Median gap:  {median_gap:.0f}s (expected 1800s)")
    print(f"  Max gap:     {max_gap:.0f}s ({max_gap/3600:.1f}h)")

    if zero_vol / len(df) > 0.3:
        print("[WARN] >30% zero-volume bars. Data may be unreliable.")

    client.shutdown()
    print("\n[DONE] M30 data ready for backtesting.")


def main():
    parser = argparse.ArgumentParser(
        description="Download M30 data from MT5 for offline backtesting"
    )
    parser.add_argument(
        "--symbol",
        default="USTEC",
        help="Symbol to download (default: USTEC)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=100000,
        help="Number of M30 bars to download (default: 100000)",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "feeds"),
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    download_m30(args.symbol, args.bars, Path(args.output))


if __name__ == "__main__":
    main()
