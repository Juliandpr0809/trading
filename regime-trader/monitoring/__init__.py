"""Monitoring, logging, and alerting package."""

from monitoring.alerts import AlertManager
from monitoring.dashboard import LiveDashboard
from monitoring.logger import StructuredLogger, get_structured_logger, setup_logger

# Optional: StreamlitDashboard requires streamlit (not installed by default)
try:
    from monitoring.dashboard_ui import StreamlitDashboard
    __all__ = [
        "StructuredLogger",
        "get_structured_logger",
        "setup_logger",
        "LiveDashboard",
        "AlertManager",
        "StreamlitDashboard",
    ]
except ImportError:
    __all__ = [
        "StructuredLogger",
        "get_structured_logger",
        "setup_logger",
        "LiveDashboard",
        "AlertManager",
    ]

