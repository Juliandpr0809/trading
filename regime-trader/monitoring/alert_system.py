"""Alert system for Liquidity Sweep + PBC strategy.

This module only notifies the trader. It never executes orders.
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path

from core.liquidity_strategy import LiquidityConfig, Signal

LOGGER = logging.getLogger(__name__)


def _parse_timestamp(value: object) -> datetime | None:
    """Parse timestamp values from datetime or string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _format_mt5_colombia_time(value: object, broker_to_colombia_hours: float) -> str:
    """Format a timestamp as broker time and Colombia time."""
    ts_mt5 = _parse_timestamp(value)
    if ts_mt5 is None:
        return "N/A"
    ts_col = ts_mt5 + timedelta(hours=broker_to_colombia_hours)
    return f"{ts_mt5:%H:%M} MT5 / {ts_col:%I:%M %p} Colombia"

ALERT_HEADERS = [
    "timestamp",
    "date",
    "direction",
    "entry_price",
    "stop_loss",
    "take_profit",
    "sl_distance",
    "rr_calculated",
    "hh_level",
    "ll_level",
    "sweep_direction",
    "sweep_price",
    "sweep_bar_time",
    "pbc_zona_high",
    "pbc_zona_low",
    "pbc_vela_a_time",
    "pbc_vela_b_time",
    "pbc_vela_c_time",
    "fib_ratio",
    "trader_action",
    "trader_notes",
    "actual_result",
    "actual_pnl",
]


class AlertSystem:
    """Sends setup alerts and appends them to CSV log."""

    def __init__(self, config: LiquidityConfig):
        self.config = config
        self.log_path = Path(config.alert_log_path)
        self._ensure_log_exists()

    def _ensure_log_exists(self) -> None:
        """Create alert CSV with headers if missing."""
        if self.log_path.exists():
            return

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ALERT_HEADERS)
            writer.writeheader()

    def send_alert(self, signal: Signal) -> None:
        """Emit console alert, optional sound, and append CSV row."""
        meta = signal.metadata or {}
        direction = signal.direction
        entry = float(signal.entry_price)
        sl = float(signal.stop_loss)
        tp = float(signal.take_profit)
        sl_dist = abs(entry - sl)
        rr = float(meta.get("rr_calculated", 0.0))

        hh = float(meta.get("hh_level", 0.0))
        ll = float(meta.get("ll_level", 0.0))
        hh_time = str(meta.get("hh_bar_time", ""))
        ll_time = str(meta.get("ll_bar_time", ""))
        sweep_dir = str(meta.get("sweep_direction", ""))
        sweep_price = float(meta.get("sweep_price", 0.0))
        sweep_time = str(meta.get("sweep_bar_time", ""))
        zona_high = float(meta.get("pbc_zona_high", meta.get("pvc_zona_high", 0.0)))
        zona_low = float(meta.get("pbc_zona_low", meta.get("pvc_zona_low", 0.0)))
        vela_a_time = str(meta.get("pbc_vela_a_time", meta.get("pvc_vela_a_time", "")))
        vela_b_time = str(meta.get("pbc_vela_b_time", meta.get("pvc_vela_b_time", "")))
        vela_c_time = str(meta.get("pbc_vela_c_time", meta.get("pvc_vela_c_time", "")))
        fib_ratio = float(meta.get("fib_ratio", 0.0))
        broker_to_colombia_hours = float(self.config.broker_to_colombia_hours)

        print("\n" + "=" * 60)
        print("SETUP DETECTADO - REVISAR GRAFICO AHORA")
        print("=" * 60)
        print(f"  Instrumento: {self.config.symbol}")
        print(f"  Direccion:   {direction}")
        print(f"  Hora entrada:{_format_mt5_colombia_time(signal.timestamp, broker_to_colombia_hours)}")
        print(f"  Entrada:     {entry:.1f}")
        print(f"  Stop Loss:   {sl:.1f} ({sl_dist:.0f} pts)")
        print(f"  Take Profit: {tp:.1f} (RR {rr:.2f}:1)")
        print("-" * 60)
        print(f"  HH pre-sesion: {hh:.1f} ({hh_time})")
        print(f"  LL pre-sesion: {ll:.1f} ({ll_time})")
        print(f"  Hora sweep:    {_format_mt5_colombia_time(sweep_time, broker_to_colombia_hours)}")
        print(f"  Sweep:         {sweep_dir} en {sweep_price:.1f} ({sweep_time})")
        print(f"  Zona PBC:      [{zona_low:.1f} - {zona_high:.1f}]")
        print(f"  Vela A:        {vela_a_time}")
        print(f"  Vela B:        {vela_b_time}")
        print(f"  Vela C:        {vela_c_time}")
        print("-" * 60)
        print("  VERIFICA EN MT5 ANTES DE ENTRAR")
        print("  El bot NO ejecuta ordenes automaticamente")
        print("=" * 60 + "\n")

        if self.config.alert_sound:
            try:
                import winsound

                winsound.Beep(1000, 500)
                winsound.Beep(1200, 500)
                winsound.Beep(1000, 500)
            except Exception:
                print("\a")

        self._append_row(
            {
                "timestamp": datetime.now().isoformat(),
                "date": signal.timestamp.date().isoformat() if signal.timestamp else "",
                "direction": direction,
                "entry_price": round(entry, 4),
                "stop_loss": round(sl, 4),
                "take_profit": round(tp, 4),
                "sl_distance": round(sl_dist, 4),
                "rr_calculated": round(rr, 4),
                "hh_level": round(hh, 4),
                "ll_level": round(ll, 4),
                "sweep_direction": sweep_dir,
                "sweep_price": round(sweep_price, 4),
                "sweep_bar_time": sweep_time,
                "pbc_zona_high": round(zona_high, 4),
                "pbc_zona_low": round(zona_low, 4),
                "pbc_vela_a_time": vela_a_time,
                "pbc_vela_b_time": vela_b_time,
                "pbc_vela_c_time": vela_c_time,
                "fib_ratio": round(fib_ratio, 6),
                "trader_action": "",
                "trader_notes": "",
                "actual_result": "",
                "actual_pnl": "",
            }
        )

        LOGGER.info(
            "ALERT SENT: %s @ %.1f | SL=%.1f | TP=%.1f | RR=%.2f",
            direction,
            entry,
            sl,
            tp,
            rr,
        )

    def log_no_setup(self, date: str, reason: str) -> None:
        """Append a no-setup row for session traceability."""
        self._append_row(
            {
                "timestamp": datetime.now().isoformat(),
                "date": date,
                "direction": "NO_SETUP",
                "entry_price": "",
                "stop_loss": "",
                "take_profit": "",
                "sl_distance": "",
                "rr_calculated": "",
                "hh_level": "",
                "ll_level": "",
                "sweep_direction": "",
                "sweep_price": "",
                "sweep_bar_time": "",
                "pbc_zona_high": "",
                "pbc_zona_low": "",
                "pbc_vela_a_time": "",
                "pbc_vela_b_time": "",
                "pbc_vela_c_time": "",
                "fib_ratio": "",
                "trader_action": "NO_SETUP",
                "trader_notes": reason,
                "actual_result": "",
                "actual_pnl": "",
            }
        )
        LOGGER.info("NO SETUP LOGGED: date=%s reason=%s", date, reason)

    def _append_row(self, row: dict[str, object]) -> None:
        with self.log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ALERT_HEADERS)
            writer.writerow(row)
