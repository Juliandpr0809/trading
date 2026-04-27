"""Structured JSON logging with rotating file handlers for trading events."""

from __future__ import annotations

import json
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON with trading context."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra context if provided
        if hasattr(record, "regime"):
            log_data["regime"] = record.regime
        if hasattr(record, "probability"):
            log_data["probability"] = record.probability
        if hasattr(record, "equity"):
            log_data["equity"] = record.equity
        if hasattr(record, "positions"):
            log_data["positions"] = record.positions
        if hasattr(record, "daily_pnl"):
            log_data["daily_pnl"] = record.daily_pnl
        if hasattr(record, "drawdown"):
            log_data["drawdown"] = record.drawdown
        
        # Include exception if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class StructuredLogger:
    """Structured logging manager with rotating file handlers."""

    def __init__(self, log_dir: str = "logs", max_bytes: int = 10 * 1024 * 1024, backup_count: int = 30):
        """Initialize structured logging system.
        
        Args:
            log_dir: Directory for log files
            max_bytes: Rotation size (default 10MB)
            backup_count: Days to keep backups
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        
        # Create loggers for different channels
        self.main_logger = self._create_logger("main", "main.log")
        self.trades_logger = self._create_logger("trades", "trades.log")
        self.alerts_logger = self._create_logger("alerts", "alerts.log")
        self.regime_logger = self._create_logger("regime", "regime.log")

    def _create_logger(self, name: str, filename: str) -> logging.Logger:
        """Create a rotating file logger.
        
        Args:
            name: Logger name
            filename: Log filename
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # Rotating file handler (10MB, 30 backups)
        filepath = self.log_dir / filename
        handler = logging.handlers.RotatingFileHandler(
            filepath,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
        )
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        
        return logger

    def log_trade(
        self,
        symbol: str,
        direction: str,
        price: float,
        volume: float,
        stop_loss: float,
        take_profit: float,
        reason: str,
    ) -> None:
        """Log a trade execution.
        
        Args:
            symbol: Trading symbol
            direction: LONG or SHORT
            price: Entry price
            volume: Position size
            stop_loss: Stop loss price
            take_profit: Take profit price
            reason: Trade reason/signal
        """
        record = logging.LogRecord(
            name="trades",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"{direction} {symbol} @ {price:.4f} x {volume} (SL: {stop_loss:.4f}, TP: {take_profit:.4f}) - {reason}",
            args=(),
            exc_info=None,
        )
        self.trades_logger.handle(record)

    def log_regime_change(
        self,
        regime_name: str,
        probability: float,
        stability_bars: int,
        flicker_rate: float,
    ) -> None:
        """Log regime state change.
        
        Args:
            regime_name: Regime identifier (LOW_VOL, MID_VOL, HIGH_VOL, etc.)
            probability: HMM confidence (0-1)
            stability_bars: Bars in current regime
            flicker_rate: Regime changes per 20 bars
        """
        record = logging.LogRecord(
            name="regime",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg=f"{regime_name} regime (prob={probability:.2%}, stable={stability_bars}b, flicker={flicker_rate:.2%})",
            args=(),
            exc_info=None,
        )
        record.regime = regime_name
        record.probability = probability
        self.regime_logger.handle(record)

    def log_alert(
        self,
        alert_type: str,
        severity: str,
        message: str,
        equity: float | None = None,
        drawdown: float | None = None,
    ) -> None:
        """Log an alert event.
        
        Args:
            alert_type: REGIME_CHANGE, CIRCUIT_BREAKER, LARGE_PNL, DATA_FEED_DOWN, API_ERROR, HMM_RETRAIN, FLICKER_HIGH
            severity: CRITICAL, WARNING, INFO
            message: Alert message
            equity: Current equity
            drawdown: Current drawdown
        """
        record = logging.LogRecord(
            name="alerts",
            level=logging.WARNING if severity == "CRITICAL" else logging.INFO,
            pathname="",
            lineno=0,
            msg=f"[{alert_type}] {severity}: {message}",
            args=(),
            exc_info=None,
        )
        record.equity = equity
        record.drawdown = drawdown
        self.alerts_logger.handle(record)

    def log_portfolio_state(
        self,
        equity: float,
        daily_pnl: float,
        positions_count: int,
        drawdown: float,
        regime: str,
        probability: float,
    ) -> None:
        """Log portfolio state snapshot.
        
        Args:
            equity: Current equity
            daily_pnl: Daily P&L
            positions_count: Number of open positions
            drawdown: Current drawdown
            regime: Current regime
            probability: HMM probability
        """
        record = logging.LogRecord(
            name="main",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg=f"Portfolio: Equity ${equity:,.2f}, Daily P&L ${daily_pnl:+,.2f}, Positions: {positions_count}, DD: {drawdown:+.2%}",
            args=(),
            exc_info=None,
        )
        record.equity = equity
        record.daily_pnl = daily_pnl
        record.positions = positions_count
        record.drawdown = drawdown
        record.regime = regime
        record.probability = probability
        self.main_logger.handle(record)

    def log_error(self, message: str, exception: Exception | None = None) -> None:
        """Log an error event.
        
        Args:
            message: Error message
            exception: Exception object if available
        """
        if exception:
            self.main_logger.error(message, exc_info=True)
        else:
            self.main_logger.error(message)


# Global singleton
_structured_logger: StructuredLogger | None = None


def get_structured_logger() -> StructuredLogger:
    """Get or create global structured logger."""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger()
    return _structured_logger


def setup_logger(name: str) -> logging.Logger:
    """Setup standard logger (legacy compatibility)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
