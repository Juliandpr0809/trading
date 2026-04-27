"""MetaTrader5 API wrapper for account and market operations (Exness)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv

LOGGER = logging.getLogger(__name__)


@dataclass
class MT5Credentials:
    """Credential bundle for MetaTrader5 connection."""

    account: int | str
    password: str
    server: str


class MT5Client:
    """Wrapper around MetaTrader5 (Exness) endpoints."""

    def __init__(self, credentials: MT5Credentials | None = None) -> None:
        """Initialize MT5 client with credentials from .env or parameter.
        
        Args:
            credentials: MT5Credentials with account, password, server.
                        If None, loads from environment variables.
        """
        load_dotenv()
        
        if credentials:
            self.credentials = credentials
        else:
            # Load from .env
            account = os.getenv("EXNESS_ACCOUNT")
            password = os.getenv("EXNESS_PASSWORD")
            server = os.getenv("EXNESS_SERVER")
            
            if not all([account, password, server]):
                raise ValueError(
                    "MT5 credentials not found. Set EXNESS_ACCOUNT, EXNESS_PASSWORD, EXNESS_SERVER in .env"
                )
            
            self.credentials = MT5Credentials(
                account=int(account) if account.isdigit() else account,
                password=password,
                server=server,
            )
        
        self.mt5 = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """Initialize MetaTrader5 connection."""
        try:
            import MetaTrader5 as mt5
            
            self.mt5 = mt5
            
            # Initialize MT5
            if not mt5.initialize():
                LOGGER.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
            
            LOGGER.info("MT5 initialized successfully")
            self.is_initialized = True
            return True
            
        except ImportError:
            LOGGER.error("MetaTrader5 module not installed. Install via: pip install MetaTrader5")
            return False
        except Exception as e:
            LOGGER.error(f"MT5 initialization error: {e}")
            return False

    def login(self) -> bool:
        """Login to MetaTrader5 with stored credentials."""
        if not self.is_initialized or self.mt5 is None:
            LOGGER.error("MT5 not initialized. Call initialize() first.")
            return False
        
        try:
            login_result = self.mt5.login(
                login=int(self.credentials.account),
                password=self.credentials.password,
                server=self.credentials.server,
            )
            
            if not login_result:
                error = self.mt5.last_error()
                LOGGER.error(f"MT5 login failed: {error}")
                return False
            
            LOGGER.info(f"[OK] Logged in to {self.credentials.server} account {self.credentials.account}")
            return True
            
        except Exception as e:
            LOGGER.error(f"Login error: {e}")
            return False

    def health_check(self, test_symbol: str = "USTEC") -> bool:
        """Verify connection and test symbol availability.
        
        Args:
            test_symbol: Symbol to check (default: USTEC - NASDAQ 100 on Exness)
            
        Returns:
            True if connection healthy and symbol available
        """
        if not self.is_initialized or self.mt5 is None:
            LOGGER.error("MT5 not initialized")
            return False
        
        try:
            # Get account info
            account_info = self.mt5.account_info()
            if account_info is None:
                LOGGER.error("Failed to get account info")
                return False
            
            LOGGER.info(f"Account: {account_info.login}, Equity: ${account_info.equity:,.2f}")
            
            # Check symbol availability
            symbol_info = self.mt5.symbol_info(test_symbol)
            if symbol_info is None:
                LOGGER.warning(f"Symbol {test_symbol} not found. Trying alternatives...")
                for alt_symbol in ["USTECm", "USTEC", "NAS100m", "NAS100"]:
                    if self.mt5.symbol_info(alt_symbol):
                        LOGGER.info(f"[OK] {alt_symbol} available")
                        break
            else:
                LOGGER.info(f"[OK] {test_symbol} available")
            
            return True
            
        except Exception as e:
            LOGGER.error(f"Health check failed: {e}")
            return False

    def get_account(self) -> dict[str, Any]:
        """Fetch account summary and buying power information.
        
        Returns:
            Dict with equity, balance, margin, margin_free, etc.
        """
        if not self.is_initialized or self.mt5 is None:
            return {}
        
        try:
            account_info = self.mt5.account_info()
            if account_info is None:
                return {}
            
            return {
                "account_id": account_info.login,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "margin_free": account_info.margin_free,
                "leverage": account_info.leverage,
                "currency": account_info.currency,
            }
            
        except Exception as e:
            LOGGER.error(f"Failed to get account info: {e}")
            return {}

    def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Fetch currently open positions from broker.
        
        Args:
            symbol: Optional symbol filter. If None, returns all positions.
            
        Returns:
            List of position dicts with symbol, volume, price_open, etc.
        """
        if not self.is_initialized or self.mt5 is None:
            return []
        
        try:
            if symbol:
                positions = self.mt5.positions_get(symbol=symbol)
            else:
                positions = self.mt5.positions_get()
            
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                result.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": pos.type,  # 0=BUY, 1=SELL
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "price_current": pos.price_current,
                    "pnl": pos.profit,  # Unrealized P&L
                    "commission": pos.commission,
                    "time_open": pos.time,
                })
            
            return result
            
        except Exception as e:
            LOGGER.error(f"Failed to get positions: {e}")
            return []

    def shutdown(self) -> None:
        """Cleanly shutdown MT5 connection."""
        if self.mt5 and self.is_initialized:
            try:
                self.mt5.shutdown()
                LOGGER.info("MT5 shutdown complete")
            except Exception as e:
                LOGGER.error(f"MT5 shutdown error: {e}")


# Backward-compatible aliases for legacy code using Alpaca naming.
AlpacaCredentials = MT5Credentials
AlpacaClient = MT5Client
