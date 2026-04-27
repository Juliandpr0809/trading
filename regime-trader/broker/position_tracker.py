"""Position and PnL state tracking utilities via MetaTrader5."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import MetaTrader5 as mt5

LOGGER = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    """Normalized open-position representation."""

    symbol: str
    ticket: int
    direction: str  # "LONG" or "SHORT"
    qty: float  # Volume in lots
    entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    commission: float
    stop_loss: float
    take_profit: float
    time_open: datetime


class PositionTracker:
    """Tracks and aggregates current holdings and realized/unrealized PnL via MT5."""

    def __init__(self) -> None:
        """Initialize in-memory position state."""
        self.positions: dict[int, PositionSnapshot] = {}  # ticket -> PositionSnapshot
        self.closed_positions: list[PositionSnapshot] = []
        self.total_realized_pnl: float = 0.0

    def refresh(self, symbols: list[str] | None = None) -> list[PositionSnapshot]:
        """Refresh position snapshot from MT5 broker.
        
        Args:
            symbols: Optional list of symbols to filter. If None, gets all positions.
            
        Returns:
            List of current PositionSnapshot objects
        """
        try:
            # Fetch positions from MT5
            if symbols and len(symbols) == 1:
                positions = mt5.positions_get(symbol=symbols[0])
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                LOGGER.warning("No positions returned from MT5")
                return list(self.positions.values())
            
            # Update internal state
            active_tickets = set()
            snapshots = []
            
            for pos in positions:
                # Get current tick price
                tick = mt5.symbol_info_tick(pos.symbol)
                if tick is None:
                    LOGGER.warning(f"Could not get tick for {pos.symbol}")
                    current_price = pos.price_current
                else:
                    current_price = tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask
                
                # Create snapshot
                snapshot = PositionSnapshot(
                    symbol=pos.symbol,
                    ticket=pos.ticket,
                    direction="LONG" if pos.type == mt5.ORDER_TYPE_BUY else "SHORT",
                    qty=pos.volume,
                    entry_price=pos.price_open,
                    current_price=current_price,
                    market_value=pos.volume * current_price,
                    unrealized_pnl=pos.profit,
                    unrealized_pnl_pct=(pos.profit / (pos.volume * pos.price_open + 1e-12)) if pos.volume > 0 else 0.0,
                    commission=pos.commission,
                    stop_loss=pos.sl,
                    take_profit=pos.tp,
                    time_open=datetime.fromtimestamp(pos.time),
                )
                
                self.positions[pos.ticket] = snapshot
                active_tickets.add(pos.ticket)
                snapshots.append(snapshot)
            
            # Log positions
            if snapshots:
                total_pnl = sum(s.unrealized_pnl for s in snapshots)
                LOGGER.info(
                    f"[CHART] Positions: {len(snapshots)} open | "
                    f"Total unrealized P&L: ${total_pnl:,.2f}"
                )
                for snap in snapshots:
                    LOGGER.debug(
                        f"  {snap.symbol} {snap.direction}: {snap.qty} lots @ ${snap.entry_price:.4f} | "
                        f"Current: ${snap.current_price:.4f} | P&L: ${snap.unrealized_pnl:,.2f} "
                        f"({snap.unrealized_pnl_pct:+.2%})"
                    )
            else:
                LOGGER.info("No open positions")
            
            return snapshots
            
        except Exception as e:
            LOGGER.error(f"Failed to refresh positions: {e}")
            return list(self.positions.values())

    def get_position(self, ticket: int) -> PositionSnapshot | None:
        """Get specific position by ticket."""
        return self.positions.get(ticket)

    def get_positions_by_symbol(self, symbol: str) -> list[PositionSnapshot]:
        """Get all positions for a symbol."""
        return [p for p in self.positions.values() if p.symbol == symbol]

    def portfolio_pnl(self) -> tuple[float, float]:
        """Return aggregated portfolio unrealized and realized P&L.
        
        Returns:
            (unrealized_pnl, realized_pnl)
        """
        unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        return unrealized, self.total_realized_pnl

    def total_exposure(self) -> dict[str, Any]:
        """Calculate total portfolio exposure metrics.
        
        Returns:
            Dict with gross_exposure, net_exposure, leverage_ratio
        """
        total_long_value = sum(
            p.market_value for p in self.positions.values() if p.direction == "LONG"
        )
        total_short_value = sum(
            p.market_value for p in self.positions.values() if p.direction == "SHORT"
        )
        
        gross_exposure = total_long_value + total_short_value
        net_exposure = total_long_value - total_short_value
        
        # Get account equity
        account = mt5.account_info()
        equity = account.equity if account else 0.0
        
        leverage = gross_exposure / (equity + 1e-12) if equity > 0 else 0.0
        
        return {
            "gross_exposure": gross_exposure,
            "net_exposure": net_exposure,
            "long_value": total_long_value,
            "short_value": total_short_value,
            "leverage_ratio": leverage,
            "num_open_positions": len(self.positions),
        }

    def close_position(self, ticket: int, symbol: str) -> tuple[bool, str]:
        """Close a position by ticket.
        
        Args:
            ticket: Position ticket number
            symbol: Trading symbol
            
        Returns:
            (success, message)
        """
        try:
            # Get position
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False, f"Position {ticket} not found"
            
            pos = positions[0]
            
            # Determine close order type (opposite of open)
            if pos.type == mt5.ORDER_TYPE_BUY:  # Close long with sell
                order_type = mt5.ORDER_TYPE_SELL
                tick = mt5.symbol_info_tick(symbol)
                price = tick.bid if tick else 0.0
                side_text = "SELL"
            else:  # Close short with buy
                order_type = mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(symbol)
                price = tick.ask if tick else 0.0
                side_text = "BUY"
            
            # Send close order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": order_type,
                "price": price,
                "deviation": 20,
                "magic": 123456,
                "comment": f"Close position {ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                error = mt5.last_error() if result is None else result.comment
                return False, f"Close failed: {error}"
            
            # Record in closed positions
            if ticket in self.positions:
                closed_pos = self.positions[ticket]
                self.closed_positions.append(closed_pos)
                self.total_realized_pnl += closed_pos.unrealized_pnl
                del self.positions[ticket]
            
            LOGGER.info(
                f"[OK] Closed position {ticket}: {side_text} {pos.volume} {symbol} @ ${price:.4f}"
            )
            return True, f"Position closed, close ticket: {result.order}"
            
        except Exception as e:
            LOGGER.error(f"Close position error: {e}")
            return False, str(e)

    def close_all_positions(self, symbols: list[str] | None = None) -> tuple[int, int]:
        """Emergency close all positions.
        
        Args:
            symbols: Optional symbol filter. If None, closes all.
            
        Returns:
            (closed_count, failed_count)
        """
        closed = 0
        failed = 0
        
        positions_to_close = [
            p for p in self.positions.values()
            if symbols is None or p.symbol in symbols
        ]
        
        for pos in positions_to_close:
            success, msg = self.close_position(pos.ticket, pos.symbol)
            if success:
                closed += 1
            else:
                failed += 1
                LOGGER.error(f"Failed to close {pos.symbol}: {msg}")
        
        LOGGER.warning(f"[STOP] Emergency close: {closed} closed, {failed} failed")
        return closed, failed

    def get_position_history(self, symbol: str | None = None, limit: int = 100) -> list[PositionSnapshot]:
        """Get closed position history.
        
        Args:
            symbol: Optional symbol filter
            limit: Max positions to return
            
        Returns:
            List of closed positions
        """
        if symbol:
            return [p for p in self.closed_positions if p.symbol == symbol][-limit:]
        return self.closed_positions[-limit:]

    def get_summary(self) -> dict[str, Any]:
        """Get position summary statistics."""
        unrealized, realized = self.portfolio_pnl()
        exposure = self.total_exposure()
        
        open_long = len([p for p in self.positions.values() if p.direction == "LONG"])
        open_short = len([p for p in self.positions.values() if p.direction == "SHORT"])
        
        return {
            "open_positions": len(self.positions),
            "open_long": open_long,
            "open_short": open_short,
            "unrealized_pnl": unrealized,
            "realized_pnl": realized,
            "total_pnl": unrealized + realized,
            "gross_exposure": exposure["gross_exposure"],
            "net_exposure": exposure["net_exposure"],
            "leverage_ratio": exposure["leverage_ratio"],
            "closed_positions_count": len(self.closed_positions),
        }

