"""Order lifecycle handling for place/modify/cancel workflows via MetaTrader5.

All orders are market-execution with native MT5 SL/TP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import MetaTrader5 as mt5

LOGGER = logging.getLogger(__name__)


@dataclass
class OrderResponse:
    """Normalized order response from broker."""

    success: bool
    ticket: int | None = None
    order_id: str | None = None
    status: str = ""
    error_message: str = ""
    price: float = 0.0
    volume: float = 0.0


class OrderExecutor:
    """Executes and manages broker orders via MetaTrader5."""

    MAGIC_NUMBER = 202604  # Bot-identifying magic number

    def __init__(self, mt5_client: Any | None = None) -> None:
        self.mt5_client = mt5_client
        self.order_history: list[OrderResponse] = []

    # ------------------------------------------------------------------
    # Core order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        signal: Any,
        position_size_lots: float | None = None,
    ) -> OrderResponse:
        """Submit a market order to MT5 with SL/TP.

        Args:
            signal: TradeSignal with symbol, direction, stop_loss, take_profit,
                    and ``lots`` (computed by risk manager).
            position_size_lots: Override lot size. If None, reads from
                                ``signal.lots`` or falls back to ``signal.target_weight``.

        Returns:
            OrderResponse with success status and ticket.
        """
        try:
            # ── Validate ──────────────────────────────────────
            if not hasattr(signal, "symbol") or not hasattr(signal, "direction"):
                return OrderResponse(
                    success=False,
                    error_message="Invalid signal: missing symbol or direction",
                )

            if signal.direction == "FLAT":
                return OrderResponse(
                    success=False,
                    status="FLAT",
                    error_message="FLAT signal — no position",
                )

            # ── Order type & price ────────────────────────────
            if signal.direction == "LONG":
                order_type = mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(signal.symbol)
                price = tick.ask if tick else 0.0
                side = "BUY"
            elif signal.direction == "SHORT":
                order_type = mt5.ORDER_TYPE_SELL
                tick = mt5.symbol_info_tick(signal.symbol)
                price = tick.bid if tick else 0.0
                side = "SELL"
            else:
                return OrderResponse(
                    success=False,
                    error_message=f"Invalid direction: {signal.direction}",
                )

            if price <= 0:
                return OrderResponse(
                    success=False,
                    error_message=f"Invalid price for {signal.symbol}: {price}",
                )

            # ── Lot size ──────────────────────────────────────
            if position_size_lots is not None:
                lots = position_size_lots
            elif hasattr(signal, "lots") and signal.lots > 0:
                lots = signal.lots
            elif hasattr(signal, "target_weight"):
                lots = max(0.01, signal.target_weight)
            else:
                lots = 0.01

            # Quantize to broker step (typically 0.01 for indices)
            sym_info = mt5.symbol_info(signal.symbol)
            if sym_info:
                step = sym_info.volume_step
                lots = max(sym_info.volume_min, round(lots / step) * step)
                lots = min(lots, sym_info.volume_max)

            # ── SL / TP ──────────────────────────────────────
            sl = getattr(signal, "stop_loss", 0.0)
            tp = getattr(signal, "take_profit", 0.0)

            # ── Build MT5 request ─────────────────────────────
            comment = (
                getattr(signal, "reasoning", "")[:27]
                if hasattr(signal, "reasoning")
                else "Bot EMA9/200/VWAP"
            )

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": signal.symbol,
                "volume": lots,
                "type": order_type,
                "price": price,
                "sl": sl if sl > 0 else 0.0,
                "tp": tp if tp > 0 else 0.0,
                "deviation": 20,
                "magic": self.MAGIC_NUMBER,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # ── Send ──────────────────────────────────────────
            result = mt5.order_send(request)

            if result is None:
                err = mt5.last_error()
                LOGGER.error("order_send returned None for %s: %s", signal.symbol, err)
                return OrderResponse(success=False, error_message=str(err))

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                LOGGER.info(
                    "[OK] %s %.2f %s @ $%.4f | Ticket %s | SL $%.4f | TP $%.4f",
                    side,
                    lots,
                    signal.symbol,
                    price,
                    result.order,
                    sl,
                    tp,
                )
                resp = OrderResponse(
                    success=True,
                    ticket=result.order,
                    order_id=str(result.order),
                    status="FILLED",
                    price=price,
                    volume=lots,
                )
                self.order_history.append(resp)
                return resp

            LOGGER.error(
                "[REJECTED] %s %s: retcode=%s  comment=%s",
                signal.symbol,
                side,
                result.retcode,
                result.comment,
            )
            return OrderResponse(
                success=False,
                status="REJECTED",
                error_message=f"Code {result.retcode}: {result.comment}",
            )

        except Exception as exc:
            LOGGER.error("Order submission error: %s", exc, exc_info=True)
            return OrderResponse(success=False, error_message=str(exc))

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def close_position(self, ticket: int, symbol: str) -> OrderResponse:
        """Close a position by ticket using an opposing market order."""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return OrderResponse(
                    success=False,
                    error_message=f"Position ticket {ticket} not found",
                )

            pos = positions[0]

            if pos.type == mt5.ORDER_TYPE_BUY:
                order_type = mt5.ORDER_TYPE_SELL
                tick = mt5.symbol_info_tick(symbol)
                price = tick.bid if tick else 0.0
            else:
                order_type = mt5.ORDER_TYPE_BUY
                tick = mt5.symbol_info_tick(symbol)
                price = tick.ask if tick else 0.0

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": pos.volume,
                "type": order_type,
                "position": ticket,
                "price": price,
                "deviation": 20,
                "magic": self.MAGIC_NUMBER,
                "comment": f"Close #{ticket}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                err = mt5.last_error() if result is None else result.comment
                LOGGER.error("Failed to close position %s: %s", ticket, err)
                return OrderResponse(success=False, error_message=str(err))

            LOGGER.info("[OK] Closed position %s | Close ticket: %s", ticket, result.order)
            return OrderResponse(success=True, ticket=result.order, status="CLOSED")

        except Exception as exc:
            LOGGER.error("Close order error: %s", exc)
            return OrderResponse(success=False, error_message=str(exc))

    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> OrderResponse:
        """Modify stop loss and take profit for an open position."""
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return OrderResponse(
                    success=False, error_message=f"Position {ticket} not found"
                )

            pos = positions[0]
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": pos.symbol,
                "position": ticket,
                "sl": sl,
                "tp": tp,
            }

            result = mt5.order_send(request)

            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                err = mt5.last_error() if result is None else result.comment
                LOGGER.error("Modify failed for %s: %s", ticket, err)
                return OrderResponse(success=False, error_message=str(err))

            LOGGER.info("[OK] Modified %s | SL=$%.4f  TP=$%.4f", ticket, sl, tp)
            return OrderResponse(success=True, ticket=ticket, status="MODIFIED")

        except Exception as exc:
            LOGGER.error("Modify order error: %s", exc)
            return OrderResponse(success=False, error_message=str(exc))

    def get_order_history(self, limit: int = 100) -> list[OrderResponse]:
        """Return recent order history."""
        return self.order_history[-limit:]
