#!/usr/bin/env python3
"""Download complete H1 historical data from MT5 via chained requests.

Downloads blocks of 5000 H1 bars backwards from current date until no more data.
Combines all blocks into single H1 CSV with deduplication.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from broker.mt5_client import MT5Client

def download_h1_full(symbol: str = "USTEC", output_dir: Path = None) -> None:
    """Download complete H1 data by chaining MT5 requests."""
    
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "feeds"
    
    # Connect
    client = MT5Client()
    if not client.initialize():
        print("[FAIL] MT5 initialization failed")
        sys.exit(1)
    if not client.login():
        print("[FAIL] MT5 login failed")
        sys.exit(1)

    import MetaTrader5 as mt5
    
    # Resolve symbol
    candidates = [symbol, "USTECm", "USTEC", "NAS100m", "NAS100", "US100"]
    resolved = None
    for sym in candidates:
        info = mt5.symbol_info(sym)
        if info is not None:
            if not info.visible:
                mt5.symbol_select(sym, True)
            resolved = sym
            break
    
    if resolved is None:
        print(f"[FAIL] Could not find symbol '{symbol}' or any NAS100 alias")
        client.shutdown()
        sys.exit(1)
    
    print(f"[OK] Symbol resolved: {resolved}")
    
    # Chain downloads: 5000 H1 bars per request, backwards from now
    all_rates = []
    end_date = datetime.now()
    block_size = 5000
    max_blocks = 50  # Safety limit: 50 * 5000 = 250,000 bars ≈ 3+ years H1
    
    for block_num in range(max_blocks):
        print(f"Downloading block {block_num + 1}: {block_size} H1 bars ending {end_date.isoformat()[:16]}...")
        
        rates = mt5.copy_rates_from(resolved, mt5.TIMEFRAME_H1, end_date, block_size)
        
        if rates is None or len(rates) == 0:
            print(f"[OK] No more data. Stopped at block {block_num}.")
            break
        
        df_block = pd.DataFrame(rates)
        all_rates.append(df_block)
        print(f"     Got {len(df_block)} bars")
        
        # Move anchor backwards
        oldest_time = rates[0]['time']  # First element is oldest
        end_date = datetime.fromtimestamp(oldest_time) - timedelta(hours=1)
    
    if not all_rates:
        print("[FAIL] No data downloaded")
        client.shutdown()
        sys.exit(1)
    
    # Combine and deduplicate
    print(f"\nCombining {len(all_rates)} blocks...")
    df_full = pd.concat(all_rates, ignore_index=True)
    print(f"Total bars before dedup: {len(df_full)}")
    
    # Deduplicate by time
    df_full = df_full.drop_duplicates(subset=['time'])
    df_full = df_full.sort_values('time').reset_index(drop=True)
    print(f"Total bars after dedup: {len(df_full)}")
    
    # Convert to DataFrame with datetime index
    df_full['datetime'] = pd.to_datetime(df_full['time'], unit='s')
    df_final = df_full[['datetime', 'open', 'high', 'low', 'close', 'tick_volume']].copy()
    df_final.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    df_final.set_index('datetime', inplace=True)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{resolved}_H1_full.csv"
    df_final.to_csv(output_path)
    print(f"\n[OK] Saved {len(df_final)} bars to {output_path}")
    print(f"     Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"     Date range: {df_final.index[0]} to {df_final.index[-1]}")
    print(f"     Span: {(df_final.index[-1] - df_final.index[0]).days} days")
    
    # Also create aliases
    for alias in [f"{symbol}_H1_full.csv", "NAS100_H1_full.csv"]:
        alias_path = output_dir / alias
        if alias_path.name != output_path.name:
            df_final.to_csv(alias_path)
            print(f"[OK] Alias: {alias_path}")
    
    # Data quality check
    null_count = df_final.isnull().sum().sum()
    zero_vol = (df_final['volume'] == 0).sum()
    gaps = df_final.index.to_series().diff().dt.total_seconds()
    median_gap = gaps.median()
    max_gap = gaps.max()
    
    print(f"\n--- Data Quality ---")
    print(f"  Nulls:       {null_count}")
    print(f"  Zero volume: {zero_vol} bars ({zero_vol/len(df_final)*100:.1f}%)")
    print(f"  Median gap:  {median_gap:.0f}s (expected 3600s)")
    print(f"  Max gap:     {max_gap:.0f}s ({max_gap/3600:.1f}h)")
    
    client.shutdown()
    print("\n[DONE] H1 data ready for backtesting.")

if __name__ == "__main__":
    download_h1_full(symbol="USTEC")
