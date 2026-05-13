"""MetaTrader5 API wrapper for account and market operations (Exness).

Provides connection management, health checks, and account queries.
Credentials are loaded from the project-root .env file.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)

# Resolve .env relative to project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
if not _ENV_PATH.exists():
    # Also check one level above project root (workspace root)
    _ENV_PATH = _PROJECT_ROOT.parent / ".env"


@dataclass
class MT5Credentials:
    """Credential bundle for MetaTrader5 connection."""

    account: int
    password: str
    server: str


class MT5Client:
    """Wrapper around MetaTrader5 (Exness) endpoints.

    Usage::

        client = MT5Client()
        client.initialize()
        client.login()
        client.health_check()
    """

    def __init__(self, credentials: MT5Credentials | None = None) -> None:
        load_dotenv(dotenv_path=str(_ENV_PATH))

        if credentials:
            self.credentials = credentials
        else:
            account = os.getenv("EXNESS_ACCOUNT")
            password = os.getenv("EXNESS_PASSWORD")
            server = os.getenv("EXNESS_SERVER")

            if not all([account, password, server]):
                raise ValueError(
                    "MT5 credentials not found. Set EXNESS_ACCOUNT, "
                    "EXNESS_PASSWORD, EXNESS_SERVER in .env"
                )

            self.credentials = MT5Credentials(
                account=int(account),
                password=password,
                server=server,
            )

        self.mt5 = None
        self.is_initialized = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def initialize(self, retries: int = 3, delay: float = 2.0) -> bool:
        """Initialize MetaTrader5 terminal connection with retry logic."""
        try:
            import MetaTrader5 as mt5

            self.mt5 = mt5
        except ImportError:
            LOGGER.error("MetaTrader5 module not installed. pip install MetaTrader5")
            return False

        for attempt in range(1, retries + 1):
            if mt5.initialize():
                LOGGER.info("MT5 initialized (attempt %d)", attempt)
                self.is_initialized = True
                return True
            LOGGER.warning(
                "MT5 init attempt %d/%d failed: %s",
                attempt,
                retries,
                mt5.last_error(),
            )
            if attempt < retries:
                time.sleep(delay)

        LOGGER.error("MT5 initialization failed after %d attempts", retries)
        return False

    def login(self) -> bool:
        """Authenticate with MetaTrader5 using stored credentials."""
        if not self.is_initialized or self.mt5 is None:
            LOGGER.error("MT5 not initialized. Call initialize() first.")
            return False

        try:
            ok = self.mt5.login(
                login=self.credentials.account,
                password=self.credentials.password,
                server=self.credentials.server,
            )
            if not ok:
                LOGGER.error("MT5 login failed: %s", self.mt5.last_error())
                return False

            LOGGER.info(
                "[OK] Logged in  server=%s  account=%s",
                self.credentials.server,
                self.credentials.account,
            )
            return True
        except Exception as exc:
            LOGGER.error("Login error: %s", exc)
            return False

    def shutdown(self) -> None:
        """Cleanly shutdown MT5 connection."""
        if self.mt5 and self.is_initialized:
            try:
                self.mt5.shutdown()
                self.is_initialized = False
                LOGGER.info("MT5 shutdown complete")
            except Exception as exc:
                LOGGER.error("MT5 shutdown error: %s", exc)

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self, test_symbol: str = "USTEC") -> bool:
        """Verify connection, account access, and symbol availability.

        Tries common NAS100 symbol aliases on Exness if the default fails.
        """
        if not self._ensure_ready():
            return False

        try:
            account_info = self.mt5.account_info()
            if account_info is None:
                LOGGER.error("Failed to get account info")
                return False

            LOGGER.info(
                "Health: account=%s  equity=%s  leverage=%dx",
                account_info.login,
                f"${account_info.equity:,.2f}",
                account_info.leverage,
            )

            # Symbol reachability
            resolved = self._resolve_symbol(test_symbol)
            if resolved is None:
                LOGGER.warning("Could not resolve any NAS100 symbol")
                return False

            LOGGER.info("[OK] Symbol resolved: %s", resolved)
            return True

        except Exception as exc:
            LOGGER.error("Health check failed: %s", exc)
            return False

    def _resolve_symbol(self, preferred: str) -> str | None:
        """Try to find an available NAS100 symbol on the broker."""
        candidates = [preferred, "USTECm", "USTEC", "NAS100m", "NAS100", "US100"]
        for sym in candidates:
            info = self.mt5.symbol_info(sym)
            if info is not None:
                # Ensure symbol is visible in MarketWatch
                if not info.visible:
                    self.mt5.symbol_select(sym, True)
                return sym
        return None

    # ------------------------------------------------------------------
    # Account queries
    # ------------------------------------------------------------------

    def get_account(self) -> dict[str, Any]:
        """Return account summary dict."""
        if not self._ensure_ready():
            return {}

        info = self.mt5.account_info()
        if info is None:
            return {}

        return {
            "account_id": info.login,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "leverage": info.leverage,
            "currency": info.currency,
            "profit": info.profit,
        }

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Fetch currently open positions from broker."""
        if not self._ensure_ready():
            return []

        try:
            positions = (
                self.mt5.positions_get(symbol=symbol)
                if symbol
                else self.mt5.positions_get()
            )
            if positions is None:
                return []

            return [
                {
                    "ticket": p.ticket,
                    "symbol": p.symbol,
                    "type": p.type,
                    "volume": p.volume,
                    "price_open": p.price_open,
                    "price_current": p.price_current,
                    "pnl": p.profit,
                    "commission": p.commission,
                    "sl": p.sl,
                    "tp": p.tp,
                    "time_open": p.time,
                }
                for p in positions
            ]
        except Exception as exc:
            LOGGER.error("Failed to get positions: %s", exc)
            return []

    def get_margin_required(self, symbol: str, volume: float, order_type: int) -> float | None:
        """Query MT5 for margin required to open a position.

        Returns margin in account currency, or None on failure.
        """
        if not self._ensure_ready():
            return None
        try:
            tick = self.mt5.symbol_info_tick(symbol)
            if tick is None:
                return None
            price = tick.ask if order_type == self.mt5.ORDER_TYPE_BUY else tick.bid
            margin = self.mt5.order_calc_margin(order_type, symbol, volume, price)
            return margin
        except Exception as exc:
            LOGGER.error("Margin calc error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ensure_ready(self) -> bool:
        if not self.is_initialized or self.mt5 is None:
            LOGGER.error("MT5 not initialized")
            return False
        return True
