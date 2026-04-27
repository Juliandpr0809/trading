"""Streamlit web dashboard for regime trader monitoring.

Run: streamlit run monitoring/dashboard_ui.py
Or:  python main.py --dashboard
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_lightweight_charts import st_lightweight_charts

LOGGER = logging.getLogger(__name__)


class StreamlitDashboard:
    """Streamlit-based web dashboard for live trading metrics."""

    def __init__(self, log_dir: str = "logs"):
        """Initialize Streamlit dashboard.
        
        Args:
            log_dir: Directory containing log files
        """
        self.log_dir = Path(log_dir)
        self.page_config = {
            "page_title": "Regime Trader Dashboard",
            "page_icon": "📊",
            "layout": "wide",
            "initial_sidebar_state": "expanded",
        }

    def run(self) -> None:
        """Run Streamlit dashboard."""
        st.set_page_config(**self.page_config)
        st.title("📊 Regime Trader Dashboard")
        
        # Sidebar
        with st.sidebar:
            st.markdown("## Navigation")
            page = st.radio(
                "Select Page:",
                ["Overview", "Equity Curve", "Drawdown", "Regime History", "Trades", "Settings"],
            )
            
            # Refresh interval
            refresh_interval = st.slider("Refresh Interval (seconds)", 1, 60, 5)
            st.markdown(f"_Refreshing every {refresh_interval}s_")
        
        # Load data
        main_logs = self._load_json_logs("main.log")
        trade_logs = self._load_json_logs("trades.log")
        regime_logs = self._load_json_logs("regime.log")
        alert_logs = self._load_json_logs("alerts.log")
        
        # Route to page
        if page == "Overview":
            self._page_overview(main_logs, trade_logs, regime_logs)
        elif page == "Equity Curve":
            self._page_equity_curve(main_logs)
        elif page == "Drawdown":
            self._page_drawdown(main_logs)
        elif page == "Regime History":
            self._page_regime_history(regime_logs)
        elif page == "Trades":
            self._page_trades(trade_logs)
        elif page == "Settings":
            self._page_settings()

    def _load_json_logs(self, filename: str) -> list[dict]:
        """Load JSON log file.
        
        Args:
            filename: Log filename
            
        Returns:
            List of log records
        """
        filepath = self.log_dir / filename
        records = []
        
        if not filepath.exists():
            return records
        
        try:
            with open(filepath, "r") as f:
                for line in f:
                    try:
                        record = json.loads(line.strip())
                        records.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            st.error(f"Failed to load logs: {e}")
        
        return records

    def _page_overview(
        self,
        main_logs: list[dict],
        trade_logs: list[dict],
        regime_logs: list[dict],
    ) -> None:
        """Render overview page."""
        st.markdown("## System Overview")
        
        # Get latest state
        if main_logs:
            latest = main_logs[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                equity = latest.get("equity", 0)
                st.metric("Equity", f"${equity:,.0f}")
            
            with col2:
                daily_pnl = latest.get("daily_pnl", 0)
                st.metric("Daily P&L", f"${daily_pnl:+,.0f}")
            
            with col3:
                positions = latest.get("positions", 0)
                st.metric("Positions", positions)
            
            with col4:
                drawdown = latest.get("drawdown", 0)
                st.metric("Drawdown", f"{drawdown:+.2%}")
        
        st.markdown("---")
        
        # Current regime
        st.markdown("## Current Regime")
        if regime_logs:
            latest_regime = regime_logs[-1]
            regime_name = latest_regime.get("regime", "UNKNOWN")
            regime_prob = latest_regime.get("probability", 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Regime", regime_name)
            with col2:
                st.metric("Confidence", f"{regime_prob:.0%}")
        
        st.markdown("---")
        
        # Recent trades
        st.markdown("## Recent Trades (Last 10)")
        if trade_logs:
            trade_df = pd.DataFrame([
                {
                    "Time": log.get("timestamp", ""),
                    "Message": log.get("message", ""),
                }
                for log in trade_logs[-10:]
            ])
            st.dataframe(trade_df, use_container_width=True)
        
        st.markdown("---")
        
        # Recent alerts
        alert_logs = self._load_json_logs("alerts.log")
        st.markdown("## Recent Alerts (Last 5)")
        if alert_logs:
            alert_df = pd.DataFrame([
                {
                    "Time": log.get("timestamp", ""),
                    "Level": log.get("level", ""),
                    "Message": log.get("message", ""),
                }
                for log in alert_logs[-5:]
            ])
            st.dataframe(alert_df, use_container_width=True)

    def _page_equity_curve(self, main_logs: list[dict]) -> None:
        """Render equity curve page."""
        st.markdown("## Equity Curve")
        
        if not main_logs:
            st.warning("No equity data available")
            return
        
        # Extract equity over time
        data = []
        for log in main_logs:
            if "equity" in log and "timestamp" in log:
                data.append({
                    "timestamp": log["timestamp"],
                    "equity": log["equity"],
                })
        
        if not data:
            st.warning("No equity data available")
            return
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Plot
        st.line_chart(df.set_index("timestamp"))
        
        # Statistics
        st.markdown("### Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            starting_equity = df["equity"].iloc[0]
            st.metric("Starting Equity", f"${starting_equity:,.0f}")
        
        with col2:
            ending_equity = df["equity"].iloc[-1]
            st.metric("Ending Equity", f"${ending_equity:,.0f}")
        
        with col3:
            total_return = (ending_equity - starting_equity) / starting_equity
            st.metric("Total Return", f"{total_return:+.2%}")
        
        with col4:
            max_equity = df["equity"].max()
            st.metric("Peak Equity", f"${max_equity:,.0f}")

    def _page_drawdown(self, main_logs: list[dict]) -> None:
        """Render drawdown analysis page."""
        st.markdown("## Drawdown Analysis")
        
        if not main_logs:
            st.warning("No drawdown data available")
            return
        
        # Extract drawdown over time
        data = []
        for log in main_logs:
            if "drawdown" in log and "timestamp" in log:
                data.append({
                    "timestamp": log["timestamp"],
                    "drawdown": log["drawdown"] * 100,  # Convert to percentage
                })
        
        if not data:
            st.warning("No drawdown data available")
            return
        
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Plot
        st.area_chart(df.set_index("timestamp"))
        
        # Statistics
        st.markdown("### Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            max_dd = df["drawdown"].min()  # Most negative
            st.metric("Max Drawdown", f"{max_dd:.2f}%")
        
        with col2:
            current_dd = df["drawdown"].iloc[-1]
            st.metric("Current Drawdown", f"{current_dd:.2f}%")
        
        with col3:
            avg_dd = df["drawdown"].mean()
            st.metric("Average Drawdown", f"{avg_dd:.2f}%")

    def _page_regime_history(self, regime_logs: list[dict]) -> None:
        """Render regime history page."""
        st.markdown("## Regime History")
        
        if not regime_logs:
            st.warning("No regime data available")
            return
        
        # Create regime history table
        data = []
        for log in regime_logs:
            data.append({
                "Time": log.get("timestamp", ""),
                "Regime": log.get("regime", "UNKNOWN"),
                "Confidence": f"{log.get('probability', 0):.0%}",
                "Message": log.get("message", ""),
            })
        
        df = pd.DataFrame(data[-50:])  # Last 50 regime changes
        st.dataframe(df, use_container_width=True)
        
        # Regime distribution
        st.markdown("### Regime Distribution")
        regime_counts = pd.Series([log.get("regime", "UNKNOWN") for log in regime_logs]).value_counts()
        st.bar_chart(regime_counts)

    def _page_trades(self, trade_logs: list[dict]) -> None:
        """Render trades page."""
        st.markdown("## Trade History")
        
        if not trade_logs:
            st.warning("No trades yet")
            return
        
        # Create trades table
        data = []
        for log in trade_logs:
            data.append({
                "Time": log.get("timestamp", ""),
                "Trade": log.get("message", ""),
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        
        # Summary stats
        st.markdown("### Trade Statistics")
        trade_count = len(trade_logs)
        st.metric("Total Trades", trade_count)

    def _page_settings(self) -> None:
        """Render settings page."""
        st.markdown("## Settings")
        
        # File paths
        st.markdown("### Log Files")
        st.info(f"Log directory: {self.log_dir.absolute()}")
        
        # Clearing logs
        st.markdown("### Maintenance")
        if st.button("Clear All Logs"):
            for log_file in self.log_dir.glob("*.log"):
                try:
                    log_file.unlink()
                    st.success(f"Deleted {log_file.name}")
                except Exception as e:
                    st.error(f"Failed to delete {log_file.name}: {e}")


def main() -> None:
    """Main entry point for Streamlit dashboard."""
    dashboard = StreamlitDashboard(log_dir="logs")
    dashboard.run()


if __name__ == "__main__":
    main()
