"""Terminal dashboard using Rich library for live telemetry display."""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout


class LiveDashboard:
    """Terminal-based live dashboard for trading system telemetry."""

    def __init__(self, refresh_seconds: int = 5):
        """Initialize dashboard.
        
        Args:
            refresh_seconds: Refresh interval in seconds
        """
        self.refresh_seconds = refresh_seconds
        self.console = Console()
        self.last_snapshot: dict[str, Any] | None = None

    def render(self, snapshot: dict[str, Any]) -> str:
        """Generate dashboard content from current snapshot.
        
        Args:
            snapshot: Current system state dict with keys:
                - regime_name, regime_prob, stability_bars, flicker_rate
                - equity, daily_pnl, daily_pnl_pct, allocation, leverage
                - positions (list of position dicts)
                - recent_signals (list of signal tuples)
                - daily_dd, daily_dd_limit, peak_dd, peak_dd_limit
                - data_status, api_status, hmm_age_days, paper_live
                
        Returns:
            Dashboard as Rich renderables
        """
        self.last_snapshot = snapshot
        
        # Build dashboard layout
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="regime", size=4),
            Layout(name="portfolio", size=4),
            Layout(name="positions", size=8),
            Layout(name="signals", size=5),
            Layout(name="risk", size=3),
            Layout(name="system", size=2),
        )
        
        # Header
        layout["header"].update(Panel(
            Text("REGIME TRADER", justify="center", style="bold magenta"),
            style="blue"
        ))
        
        # Regime section
        regime_name = snapshot.get("regime_name", "UNKNOWN")
        regime_prob = snapshot.get("regime_prob", 0.0)
        stability_bars = snapshot.get("stability_bars", 0)
        flicker_rate = snapshot.get("flicker_rate", 0.0)
        
        regime_color = self._get_regime_color(regime_name)
        regime_text = f"{regime_name} ({regime_prob:.0%})"
        regime_content = f"""
Regime: {regime_text}
Stability: {stability_bars} bars | Flicker: {flicker_rate:.1%}
"""
        layout["regime"].update(Panel(
            regime_content.strip(),
            title="REGIME",
            style=regime_color
        ))
        
        # Portfolio section
        equity = snapshot.get("equity", 0.0)
        daily_pnl = snapshot.get("daily_pnl", 0.0)
        daily_pnl_pct = snapshot.get("daily_pnl_pct", 0.0)
        allocation = snapshot.get("allocation", 0.0)
        leverage = snapshot.get("leverage", 1.0)
        
        pnl_color = "green" if daily_pnl >= 0 else "red"
        portfolio_content = f"""
Equity: ${equity:,.0f} | Daily: ${daily_pnl:+,.0f} ({daily_pnl_pct:+.2%})
Allocation: {allocation:.0%} | Leverage: {leverage:.2f}x
"""
        layout["portfolio"].update(Panel(
            portfolio_content.strip(),
            title="PORTFOLIO",
            style="cyan"
        ))
        
        # Positions section
        positions = snapshot.get("positions", [])
        positions_table = Table(show_header=True, header_style="bold")
        positions_table.add_column("Symbol", style="cyan")
        positions_table.add_column("Type", style="magenta")
        positions_table.add_column("Price", justify="right")
        positions_table.add_column("P&L", justify="right")
        positions_table.add_column("Stop", justify="right")
        positions_table.add_column("Time", justify="right")
        
        for pos in positions[:5]:  # Show up to 5 positions
            symbol = pos.get("symbol", "?")
            pos_type = pos.get("direction", "?")
            price = pos.get("price", 0.0)
            pnl = pos.get("unrealized_pnl", 0.0)
            pnl_pct = pos.get("unrealized_pnl_pct", 0.0)
            stop = pos.get("stop_loss", 0.0)
            time_str = pos.get("time_open", "?")
            
            pnl_color = "green" if pnl >= 0 else "red"
            positions_table.add_row(
                symbol,
                pos_type,
                f"${price:.2f}",
                f"${pnl:+,.0f} ({pnl_pct:+.1%})",
                f"${stop:.2f}",
                time_str,
                style=pnl_color
            )
        
        layout["positions"].update(Panel(
            positions_table,
            title="POSITIONS",
            style="cyan"
        ))
        
        # Recent signals section
        recent_signals = snapshot.get("recent_signals", [])
        signals_content = ""
        for signal in recent_signals[:3]:  # Last 3 signals
            time_str = signal[0]
            symbol = signal[1]
            action = signal[2]
            reason = signal[3]
            signals_content += f"{time_str} | {symbol} | {action} | {reason}\n"
        
        layout["signals"].update(Panel(
            signals_content.strip() or "No recent signals",
            title="RECENT SIGNALS",
            style="cyan"
        ))
        
        # Risk section
        daily_dd = snapshot.get("daily_dd", 0.0)
        daily_dd_limit = snapshot.get("daily_dd_limit", 0.02)
        peak_dd = snapshot.get("peak_dd", 0.0)
        peak_dd_limit = snapshot.get("peak_dd_limit", 0.10)
        
        daily_dd_color = "red" if daily_dd >= daily_dd_limit else "yellow" if daily_dd >= daily_dd_limit * 0.75 else "green"
        peak_dd_color = "red" if peak_dd >= peak_dd_limit else "yellow" if peak_dd >= peak_dd_limit * 0.75 else "green"
        
        risk_content = f"""
Daily DD: {daily_dd:+.1%}/{daily_dd_limit:.1%} | Peak: {peak_dd:+.1%}/{peak_dd_limit:.1%}
"""
        layout["risk"].update(Panel(
            risk_content.strip(),
            title="RISK STATUS",
            style=daily_dd_color
        ))
        
        # System section
        data_status = snapshot.get("data_status", "UNKNOWN")
        api_status = snapshot.get("api_status", "UNKNOWN")
        hmm_age_days = snapshot.get("hmm_age_days", -1)
        paper_live = snapshot.get("paper_live", "PAPER")
        
        data_color = "green" if data_status == "OK" else "red"
        api_color = "green" if api_status == "OK" else "red"
        
        hmm_status = f"{hmm_age_days}d" if hmm_age_days >= 0 else "UNKNOWN"
        
        system_content = f"Data: [{data_status}] | API: [{api_status}] | HMM: {hmm_status} | [{paper_live}]"
        layout["system"].update(Panel(
            system_content,
            style="cyan"
        ))
        
        return layout

    def _get_regime_color(self, regime_name: str) -> str:
        """Get color for regime type.
        
        Args:
            regime_name: Regime identifier
            
        Returns:
            Rich color name
        """
        regime_colors = {
            "LOW_VOL": "green",
            "MID_VOL": "yellow",
            "HIGH_VOL": "red",
            "BULL": "green",
            "BEAR": "red",
            "NEUTRAL": "yellow",
        }
        return regime_colors.get(regime_name, "white")

    def run_live(self, update_callback) -> None:
        """Run dashboard in live mode with real-time updates.
        
        Args:
            update_callback: Function that returns snapshot dict when called
        """
        try:
            with Live(self.render(update_callback()), refresh_per_second=1/self.refresh_seconds) as live:
                while True:
                    snapshot = update_callback()
                    live.update(self.render(snapshot))
                    time.sleep(self.refresh_seconds)
        except KeyboardInterrupt:
            self.console.print("[red]Dashboard stopped[/red]")

    def print_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Print a single dashboard snapshot.
        
        Args:
            snapshot: Current system state
        """
        layout = self.render(snapshot)
        self.console.print(layout)
