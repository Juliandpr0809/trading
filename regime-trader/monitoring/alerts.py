"""Rate-limited alerts for critical trading events via email and webhooks."""

from __future__ import annotations

import json
import logging
import os
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)


class AlertManager:
    """Dispatches alerts with rate limiting to prevent spam."""

    def __init__(self, rate_limit_minutes: int = 15):
        """Initialize alert manager.
        
        Args:
            rate_limit_minutes: Minimum minutes between alerts for same event type
        """
        self.rate_limit_minutes = rate_limit_minutes
        self.rate_limit_map: dict[str, datetime] = {}  # event_type -> last_alert_time
        
        # Load config from environment
        self.email_enabled = os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true"
        self.webhook_enabled = os.getenv("ALERT_WEBHOOK_ENABLED", "false").lower() == "true"
        
        self.email_from = os.getenv("ALERT_EMAIL_FROM", "")
        self.email_to = os.getenv("ALERT_EMAIL_TO", "").split(",") if os.getenv("ALERT_EMAIL_TO") else []
        self.smtp_server = os.getenv("ALERT_SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        self.smtp_password = os.getenv("ALERT_SMTP_PASSWORD", "")
        
        self.webhook_url = os.getenv("ALERT_WEBHOOK_URL", "")

    def _check_rate_limit(self, event_type: str) -> bool:
        """Check if event type is within rate limit.
        
        Args:
            event_type: Alert type (REGIME_CHANGE, CIRCUIT_BREAKER, etc.)
            
        Returns:
            True if can send alert, False if rate limited
        """
        now = datetime.now()
        last_time = self.rate_limit_map.get(event_type)
        
        if last_time is None:
            self.rate_limit_map[event_type] = now
            return True
        
        elapsed = now - last_time
        if elapsed >= timedelta(minutes=self.rate_limit_minutes):
            self.rate_limit_map[event_type] = now
            return True
        
        return False

    def send_alert(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Send alert via configured channels.
        
        Args:
            alert_type: Event type (REGIME_CHANGE, CIRCUIT_BREAKER, LARGE_PNL, DATA_FEED_DOWN, API_ERROR, HMM_RETRAIN, FLICKER_HIGH)
            severity: CRITICAL, WARNING, INFO
            title: Short alert title
            message: Detailed message
            context: Additional context dict
        """
        # Check rate limit
        if not self._check_rate_limit(alert_type):
            LOGGER.debug(f"Alert {alert_type} rate limited")
            return
        
        LOGGER.info(f"[{severity}] {alert_type}: {title}")
        
        # Send via email if enabled
        if self.email_enabled and self.email_from and self.email_to:
            self._send_email(alert_type, severity, title, message, context)
        
        # Send via webhook if enabled
        if self.webhook_enabled and self.webhook_url:
            self._send_webhook(alert_type, severity, title, message, context)

    def _send_email(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Send alert via email.
        
        Args:
            alert_type: Event type
            severity: Alert severity
            title: Alert title
            message: Alert message
            context: Additional context
        """
        try:
            # Create email
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{severity}] {title}"
            msg["From"] = self.email_from
            msg["To"] = ", ".join(self.email_to)
            
            # HTML body
            html = f"""
            <html>
                <body>
                    <h2>{title}</h2>
                    <p><strong>Type:</strong> {alert_type}</p>
                    <p><strong>Severity:</strong> {severity}</p>
                    <p><strong>Time:</strong> {datetime.now().isoformat()}</p>
                    <hr>
                    <p>{message}</p>
            """
            
            if context:
                html += "<h3>Context:</h3><ul>"
                for key, value in context.items():
                    html += f"<li><strong>{key}:</strong> {value}</li>"
                html += "</ul>"
            
            html += "</body></html>"
            
            msg.attach(MIMEText(html, "html"))
            
            # Send
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_from, self.smtp_password)
                server.send_message(msg)
            
            LOGGER.info(f"Email alert sent to {self.email_to}")
            
        except Exception as e:
            LOGGER.error(f"Failed to send email alert: {e}")

    def _send_webhook(
        self,
        alert_type: str,
        severity: str,
        title: str,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """Send alert via webhook.
        
        Args:
            alert_type: Event type
            severity: Alert severity
            title: Alert title
            message: Alert message
            context: Additional context
        """
        try:
            payload = {
                "alert_type": alert_type,
                "severity": severity,
                "title": title,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
            
            if context:
                payload["context"] = context
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=5,
                headers={"Content-Type": "application/json"},
            )
            
            if response.status_code in [200, 201, 204]:
                LOGGER.info(f"Webhook alert sent successfully")
            else:
                LOGGER.error(f"Webhook returned status {response.status_code}")
        
        except Exception as e:
            LOGGER.error(f"Failed to send webhook alert: {e}")

    # Convenience methods for specific alert types

    def alert_regime_change(self, old_regime: str, new_regime: str, probability: float) -> None:
        """Alert on regime change.
        
        Args:
            old_regime: Previous regime
            new_regime: New regime
            probability: HMM confidence
        """
        self.send_alert(
            alert_type="REGIME_CHANGE",
            severity="INFO",
            title=f"Regime Change: {old_regime} -> {new_regime}",
            message=f"Regime changed from {old_regime} to {new_regime} with confidence {probability:.0%}",
            context={"old_regime": old_regime, "new_regime": new_regime, "probability": f"{probability:.0%}"},
        )

    def alert_circuit_breaker(self, trigger_type: str, drawdown: float, limit: float) -> None:
        """Alert on circuit breaker trigger.
        
        Args:
            trigger_type: Type of trigger (daily, weekly, peak)
            drawdown: Current drawdown
            limit: Limit threshold
        """
        self.send_alert(
            alert_type="CIRCUIT_BREAKER",
            severity="CRITICAL",
            title=f"🔴 Circuit Breaker Triggered ({trigger_type})",
            message=f"{trigger_type.capitalize()} drawdown {drawdown:.2%} exceeded limit {limit:.2%}. All positions will be closed.",
            context={"trigger_type": trigger_type, "drawdown": f"{drawdown:.2%}", "limit": f"{limit:.2%}"},
        )

    def alert_large_pnl(self, pnl: float, pnl_pct: float) -> None:
        """Alert on large P&L move.
        
        Args:
            pnl: P&L amount
            pnl_pct: P&L percentage
        """
        self.send_alert(
            alert_type="LARGE_PNL",
            severity="WARNING",
            title=f"Large P&L: ${pnl:+,.2f}",
            message=f"Significant P&L move: ${pnl:+,.2f} ({pnl_pct:+.2%})",
            context={"pnl": f"${pnl:+,.2f}", "pnl_pct": f"{pnl_pct:+.2%}"},
        )

    def alert_data_feed_down(self) -> None:
        """Alert on data feed loss."""
        self.send_alert(
            alert_type="DATA_FEED_DOWN",
            severity="CRITICAL",
            title="Data Feed Down",
            message="Market data feed is unavailable. Signals paused, stop losses remain active.",
        )

    def alert_api_error(self, error_message: str) -> None:
        """Alert on API error.
        
        Args:
            error_message: Error details
        """
        self.send_alert(
            alert_type="API_ERROR",
            severity="CRITICAL",
            title="API Connection Error",
            message=f"Broker API error: {error_message}",
            context={"error": error_message},
        )

    def alert_hmm_retrain(self, old_n_components: int, new_n_components: int) -> None:
        """Alert on HMM retraining.
        
        Args:
            old_n_components: Previous regime count
            new_n_components: New regime count
        """
        self.send_alert(
            alert_type="HMM_RETRAIN",
            severity="INFO",
            title="HMM Model Retrained",
            message=f"HMM model retrained: {old_n_components} -> {new_n_components} regimes",
            context={"old_regimes": old_n_components, "new_regimes": new_n_components},
        )

    def alert_flicker_high(self, flicker_rate: float, threshold: float) -> None:
        """Alert on high regime flicker.
        
        Args:
            flicker_rate: Current flicker rate
            threshold: Flicker rate threshold
        """
        self.send_alert(
            alert_type="FLICKER_HIGH",
            severity="WARNING",
            title="High Regime Flicker",
            message=f"Regime instability detected: {flicker_rate:.0%} flicker rate (threshold: {threshold:.0%}). Reducing signal confidence.",
            context={"flicker_rate": f"{flicker_rate:.0%}", "threshold": f"{threshold:.0%}"},
        )
