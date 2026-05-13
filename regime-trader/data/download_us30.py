#!/usr/bin/env python3
"""Download US30 (Dow Jones) M30 and M5 historical data from MT5 (Exness).

Usage:
    python data/download_us30.py
    python data/download_us30.py --years 3
    python data/download_us30.py --symbol US30m

This script connects to Exness MT5, downloads M30 and M5 candles for US30,
and saves them as separate CSVs for the liquidity-sweep backtester.

Downloads are chained in 5000-bar batches to overcome MT5's per-request limit.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from broker.mt5_client import MT5Client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)

# Maximum bars per MT5 request (safe limit)
BATCH_SIZE = 5000


def resolve_us30_symbol(mt5_module, preferred: str) -> str | None:
    """Try common US30 / Dow Jones aliases on Exness until one resolves."""
    candidates = [preferred, "US30m", "US30", "US30Cash", "DJ30", "DJ30m", "DJI"]
    for sym in candidates:
        info = mt5_module.symbol_info(sym)
        if info is not None:
            if not info.visible:
                mt5_module.symbol_select(sym, True)
            LOGGER.info("Symbol resolved: %s  (contract_size=%.2f, digits=%d)",
                        sym, info.trade_contract_size, info.digits)
            return sym
    return None


def download_chained(
    mt5_module,
    symbol: str,
    timeframe,
    timeframe_label: str,
    target_bars: int,
) -> pd.DataFrame:
    """Download historical bars by chaining requests backwards from now.

    MT5's copy_rates_from_pos has a practical limit per call.
    We chain requests moving backward in time to accumulate the full history.
    """
    all_frames: list[pd.DataFrame] = []
    total_downloaded = 0
    end_date = datetime.now()

    LOGGER.info("Downloading %s %s — target %d bars in batches of %d",
                symbol, timeframe_label, target_bars, BATCH_SIZE)

    while total_downloaded < target_bars:
        batch = min(BATCH_SIZE, target_bars - total_downloaded)
        rates = mt5_module.copy_rates_from(
            symbol, timeframe, end_date, batch
        )

        if rates is None or len(rates) == 0:
            LOGGER.warning("No more data available at %s. Total so far: %d bars",
                           end_date, total_downloaded)
            break

        df_batch = pd.DataFrame(rates)
        df_batch["datetime"] = pd.to_datetime(df_batch["time"], unit="s")
        all_frames.append(df_batch)
        total_downloaded += len(df_batch)

        # Move end_date back to 1 second before the oldest bar in this batch
        oldest_time = df_batch["datetime"].min()
        end_date = oldest_time - timedelta(seconds=1)

        LOGGER.info("  Batch: %d bars (oldest=%s) | Total: %d / %d",
                     len(df_batch), oldest_time, total_downloaded, target_bars)

        if len(rates) < batch:
            LOGGER.info("  Reached end of available history.")
            break

    if not all_frames:
        return pd.DataFrame()

    # Combine, deduplicate, sort
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["time"], keep="first")
    combined = combined.sort_values("time").reset_index(drop=True)

    # Normalize columns
    combined["datetime"] = pd.to_datetime(combined["time"], unit="s")
    result = combined[["datetime", "open", "high", "low", "close", "tick_volume"]].copy()
    result.columns = ["datetime", "open", "high", "low", "close", "volume"]
    result.set_index("datetime", inplace=True)

    LOGGER.info("Download complete: %d unique %s bars", len(result), timeframe_label)
    return result


def validate_data(df: pd.DataFrame, label: str) -> bool:
    """Run data quality checks and log results."""
    if df.empty:
        LOGGER.error("DATA CHECK %s: EMPTY — no bars downloaded", label)
        return False

    null_count = df.isnull().sum().sum()
    zero_vol = (df["volume"] == 0).sum()
    gaps = df.index.to_series().diff().dt.total_seconds()
    median_gap = gaps.median()
    max_gap = gaps.max()

    LOGGER.info("=" * 60)
    LOGGER.info("DATA CHECK: %s", label)
    LOGGER.info("  Bars:       %d", len(df))
    LOGGER.info("  Date start: %s", df.index[0])
    LOGGER.info("  Date end:   %s", df.index[-1])
    LOGGER.info("  Span:       %d days", (df.index[-1] - df.index[0]).days)
    LOGGER.info("  Close min:  %.2f", df["close"].min())
    LOGGER.info("  Close max:  %.2f", df["close"].max())
    LOGGER.info("  Avg volume: %.0f", df["volume"].mean())
    LOGGER.info("  Nulls:      %d", null_count)
    LOGGER.info("  Zero vol:   %d (%.1f%%)", zero_vol, 100 * zero_vol / len(df))
    LOGGER.info("  Median gap: %.0fs", median_gap)
    LOGGER.info("  Max gap:    %.0fs (%.1fh)", max_gap, max_gap / 3600)
    LOGGER.info("=" * 60)

    if null_count > 0:
        LOGGER.warning("Data contains %d null values", null_count)
    if zero_vol / len(df) > 0.5:
        LOGGER.warning(">50%% zero-volume bars — tick volume may be synthetic")

    return True


def check_alignment(m30: pd.DataFrame, m5: pd.DataFrame) -> None:
    """Verify that M30 and M5 data overlap in time range."""
    m30_start, m30_end = m30.index[0], m30.index[-1]
    m5_start, m5_end = m5.index[0], m5.index[-1]

    overlap_start = max(m30_start, m5_start)
    overlap_end = min(m30_end, m5_end)

    if overlap_start >= overlap_end:
        LOGGER.error("NO OVERLAP between M30 and M5 data!")
        LOGGER.error("  M30: %s to %s", m30_start, m30_end)
        LOGGER.error("  M5:  %s to %s", m5_start, m5_end)
        return

    overlap_days = (overlap_end - overlap_start).days
    LOGGER.info("ALIGNMENT CHECK:")
    LOGGER.info("  M30 range: %s to %s", m30_start, m30_end)
    LOGGER.info("  M5 range:  %s to %s", m5_start, m5_end)
    LOGGER.info("  Overlap:   %d days (%s to %s)", overlap_days, overlap_start, overlap_end)

    if overlap_days < 365:
        LOGGER.warning("Less than 1 year of overlapping data!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download US30 M30+M5 data from MT5 for liquidity sweep backtesting"
    )
    parser.add_argument(
        "--symbol", default="US30m",
        help="Symbol to download (default: US30m)",
    )
    parser.add_argument(
        "--years", type=float, default=3.0,
        help="Years of history to download (default: 3)",
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "data" / "feeds"),
        help="Output directory for CSV files",
    )
    args = parser.parse_args()

    # Calculate target bars
    # M5:  ~74,880 bars/year (261 trading days * 24h * 12 bars/h)
    # M30: ~12,528 bars/year (261 trading days * 24h * 2 bars/h)
    target_m5 = int(75_000 * args.years)
    target_m30 = int(12_600 * args.years)

    # 1. Connect to MT5
    client = MT5Client()
    if not client.initialize():
        LOGGER.error("MT5 initialization failed")
        sys.exit(1)
    if not client.login():
        LOGGER.error("MT5 login failed")
        sys.exit(1)

    import MetaTrader5 as mt5

    # 2. Resolve symbol
    resolved = resolve_us30_symbol(mt5, args.symbol)
    if resolved is None:
        LOGGER.error("Could not find symbol '%s' or any US30 alias", args.symbol)
        client.shutdown()
        sys.exit(1)

    # Log symbol properties
    sym_info = mt5.symbol_info(resolved)
    if sym_info:
        LOGGER.info("Symbol properties:")
        LOGGER.info("  Name:          %s", sym_info.name)
        LOGGER.info("  Description:   %s", sym_info.description)
        LOGGER.info("  Contract size: %.2f", sym_info.trade_contract_size)
        LOGGER.info("  Tick size:     %.5f", sym_info.trade_tick_size)
        LOGGER.info("  Tick value:    %.5f", sym_info.trade_tick_value)
        LOGGER.info("  Digits:        %d", sym_info.digits)
        LOGGER.info("  Volume min:    %.2f", sym_info.volume_min)
        LOGGER.info("  Volume step:   %.2f", sym_info.volume_step)

    # 3. Download M30
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("DOWNLOADING M30 DATA")
    LOGGER.info("=" * 60)
    m30_data = download_chained(mt5, resolved, mt5.TIMEFRAME_M30, "M30", target_m30)

    # 4. Download M5
    LOGGER.info("")
    LOGGER.info("=" * 60)
    LOGGER.info("DOWNLOADING M5 DATA")
    LOGGER.info("=" * 60)
    m5_data = download_chained(mt5, resolved, mt5.TIMEFRAME_M5, "M5", target_m5)

    client.shutdown()

    # 5. Validate
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    m30_ok = validate_data(m30_data, f"{resolved} M30")
    m5_ok = validate_data(m5_data, f"{resolved} M5")

    if not m30_ok or not m5_ok:
        LOGGER.error("Data validation failed. Check logs above.")
        sys.exit(1)

    # 6. Check alignment
    check_alignment(m30_data, m5_data)

    # 7. Save
    m30_path = output_dir / f"{resolved}_M30.csv"
    m5_path = output_dir / f"{resolved}_M5.csv"

    m30_data.to_csv(m30_path)
    LOGGER.info("Saved M30: %s  (%d KB)", m30_path, m30_path.stat().st_size // 1024)

    m5_data.to_csv(m5_path)
    LOGGER.info("Saved M5:  %s  (%d KB)", m5_path, m5_path.stat().st_size // 1024)

    # Also save without the 'm' suffix for config compatibility
    base_symbol = resolved.rstrip("m")
    if base_symbol != resolved:
        alias_m30 = output_dir / f"{base_symbol}_M30.csv"
        alias_m5 = output_dir / f"{base_symbol}_M5.csv"
        m30_data.to_csv(alias_m30)
        m5_data.to_csv(alias_m5)
        LOGGER.info("Aliases saved: %s, %s", alias_m30.name, alias_m5.name)

    LOGGER.info("")
    LOGGER.info("[DONE] US30 data ready for backtesting.")
    LOGGER.info("  M30: %s", m30_path)
    LOGGER.info("  M5:  %s", m5_path)


if __name__ == "__main__":
    main()
