"""Signal orchestration combining regime model and strategy outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from core.hmm_engine import HMMEngine, RegimeState
from core.regime_strategies import RegimeStrategy, StrategyConfig, TechnicalSetup
from data.feature_engineering import FeatureEngineer

LOGGER = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Normalized signal payload consumed by order execution layer."""

    symbol: str
    target_weight: float
    regime: int
    regime_name: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    direction: str  # "LONG", "SHORT", "FLAT"
    timestamp: datetime = field(default_factory=datetime.now)
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SignalGenerator:
    """Converts HMM regime state + technical strategy into actionable portfolio signals."""

    def __init__(
        self,
        hmm_engine: HMMEngine | None = None,
        strategy_config: StrategyConfig | None = None,
        feature_engineer: FeatureEngineer | None = None,
    ) -> None:
        """Initialize signal generator with dependencies.
        
        Args:
            hmm_engine: Trained HMM instance (required)
            strategy_config: Technical strategy configuration
            feature_engineer: Feature computation engine
        """
        self.hmm_engine = hmm_engine
        self.strategy_config = strategy_config or StrategyConfig()
        self.feature_engineer = feature_engineer or FeatureEngineer()
        self.strategy = RegimeStrategy(config=self.strategy_config)
        self.last_signals: dict[str, TradeSignal] = {}
        self._account_equity = 10_000.0  # Default; can be set via set_account_equity()

    def set_account_equity(self, equity: float) -> None:
        """Set account equity for position sizing.
        
        Args:
            equity: Current account equity in base currency
        """
        self._account_equity = equity

    def generate(
        self,
        symbols: list[str],
        market_data: dict[str, pd.DataFrame],
        historical_features: pd.DataFrame | None = None,
    ) -> list[TradeSignal]:
        """Generate a set of target signals for the current cycle.
        
        Args:
            symbols: List of symbols to generate signals for
            market_data: Dict of {symbol: ohlcv_dataframe}
            historical_features: Pre-computed feature matrix (optional)
            
        Returns:
            List of TradeSignal objects with direction, weight, stop/target
        """
        signals = []
        
        if not self.hmm_engine:
            LOGGER.error("HMM engine not initialized. Cannot generate signals.")
            return signals
        
        for symbol in symbols:
            if symbol not in market_data:
                LOGGER.warning(f"No market data for {symbol}")
                continue
            
            try:
                signal = self._generate_signal_for_symbol(
                    symbol=symbol,
                    ohlcv_data=market_data[symbol],
                    historical_features=historical_features,
                )
                if signal:
                    signals.append(signal)
                    self.last_signals[symbol] = signal
            except Exception as e:
                LOGGER.error(f"Signal generation failed for {symbol}: {e}")
        
        return signals

    def _generate_signal_for_symbol(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame,
        historical_features: pd.DataFrame | None = None,
    ) -> TradeSignal | None:
        """Generate signal for a single symbol."""
        if ohlcv_data.empty or len(ohlcv_data) < 200:
            return None
        
        try:
            # Compute features if not provided
            if historical_features is None or historical_features.empty:
                features = self.feature_engineer.transform(ohlcv_data)
            else:
                features = historical_features
            
            if features.empty:
                return None
            
            # Get HMM regime state
            regime_state = self.hmm_engine.predict_regime_filtered(features)
            regime_info = self.hmm_engine.regime_info.get(regime_state.current_regime_id)
            
            if not regime_info:
                return None
            
            # Compute technical indicators from latest price data
            latest = ohlcv_data.iloc[-1]
            close_prices = ohlcv_data["close"]
            
            ema_9 = close_prices.ewm(span=9, adjust=False).mean().iloc[-1]
            ema_200 = close_prices.ewm(span=200, adjust=False).mean().iloc[-1]
            vwap = self._compute_vwap(ohlcv_data)
            atr_14 = self._compute_atr(ohlcv_data, 14)
            
            swing_high = ohlcv_data["high"].tail(20).max()
            swing_low = ohlcv_data["low"].tail(20).min()
            
            distance_from_ema200 = (latest["close"] - ema_200) / ema_200 if ema_200 > 0 else 0.0
            
            # Calculate real volatility percentile (normalized rolling std)
            close_prices = ohlcv_data["close"]
            if len(close_prices) >= 20:
                # Compute all rolling stds once
                rolling_stds = close_prices.rolling(20).std()
                current_std = rolling_stds.iloc[-1]
                
                if pd.notna(current_std) and current_std > 0:
                    # Count how many historical stds are <= current
                    historical = rolling_stds.dropna().iloc[:-1]  # exclude current
                    if len(historical) > 0:
                        percentile = (historical <= current_std).sum() / len(historical)
                        volatility_percentile = max(0.0, min(1.0, float(percentile)))
                    else:
                        volatility_percentile = 0.5
                else:
                    volatility_percentile = 0.5
            else:
                volatility_percentile = 0.5
            
            setup = TechnicalSetup(
                price=latest["close"],
                ema_9=ema_9,
                ema_200=ema_200,
                vwap=vwap,
                atr_14=atr_14,
                swing_high=swing_high,
                swing_low=swing_low,
                distance_from_ema200=distance_from_ema200,
                volatility_percentile=volatility_percentile,
            )
            
            # Evaluate strategy signal (pass account_equity if available, default 10000)
            account_equity = getattr(self, '_account_equity', 10_000.0)
            strategy_signal = self.strategy.evaluate_signal(
                symbol=symbol,
                regime_state=regime_state,
                regime_info_map=self.hmm_engine.regime_info,
                setup=setup,
                account_equity=account_equity,
            )
            
            # Convert to TradeSignal
            timestamp = ohlcv_data.index[-1] if isinstance(ohlcv_data.index, pd.DatetimeIndex) else datetime.now()
            
            trade_signal = TradeSignal(
                symbol=symbol,
                target_weight=strategy_signal.position_size,
                regime=regime_state.current_regime_id,
                regime_name=regime_info.label,
                confidence=regime_state.state_probability,
                entry_price=setup.price,
                stop_loss=strategy_signal.stop_loss,
                take_profit=strategy_signal.take_profit,
                direction=strategy_signal.direction,
                timestamp=timestamp,
                reasoning=strategy_signal.reasoning,
                metadata={
                    "regime_id": regime_state.current_regime_id,
                    "regime_name": regime_info.label,
                    "regime_expected_vol": regime_info.expected_volatility,
                    "regime_expected_return": regime_info.expected_return,
                    "ema_9": float(ema_9),
                    "ema_200": float(ema_200),
                    "vwap": float(vwap),
                    "atr_14": float(atr_14),
                    "swing_high": float(swing_high),
                    "swing_low": float(swing_low),
                    "state_probabilities": regime_state.state_probabilities.tolist() if hasattr(regime_state.state_probabilities, "tolist") else [],
                },
            )
            
            LOGGER.info(
                f"Signal for {symbol}: {trade_signal.direction} @ {trade_signal.entry_price:.4f}, "
                f"Regime: {trade_signal.regime_name}, Confidence: {trade_signal.confidence:.2%}"
            )
            
            return trade_signal
            
        except Exception as e:
            LOGGER.error(f"Failed to generate signal for {symbol}: {e}", exc_info=True)
            return None

    @staticmethod
    def _compute_vwap(ohlcv: pd.DataFrame) -> float:
        """Compute session-anchored volume-weighted average price."""
        if ohlcv.empty:
            return 0.0

        if isinstance(ohlcv.index, pd.DatetimeIndex):
            last_date = ohlcv.index[-1].date()
            session_data = ohlcv[ohlcv.index.date == last_date]
        else:
            # Fallback for non-datetime index: approximate intraday session.
            session_data = ohlcv.tail(96)

        if session_data.empty:
            session_data = ohlcv.tail(96)

        typical_price = (
            session_data["high"] + session_data["low"] + session_data["close"]
        ) / 3.0
        cum_pv = (typical_price * session_data["volume"]).sum()
        cum_v = session_data["volume"].sum()
        return float(cum_pv / (cum_v + 1e-12))

    @staticmethod
    def _compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
        """Compute average true range."""
        if len(ohlcv) < period:
            return float(ohlcv["high"].iloc[-1] - ohlcv["low"].iloc[-1])
        
        high_low = ohlcv["high"] - ohlcv["low"]
        high_close = abs(ohlcv["high"] - ohlcv["close"].shift())
        low_close = abs(ohlcv["low"] - ohlcv["close"].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return float(atr.iloc[-1]) if not atr.isna().all() else float(tr.iloc[-1])

    def get_last_signal(self, symbol: str) -> TradeSignal | None:
        """Retrieve the most recent signal for a symbol."""
        return self.last_signals.get(symbol)

    def validate_signal(self, signal: TradeSignal) -> bool:
        """Perform sanity checks on signal before transmission."""
        if signal.target_weight < 0 or signal.target_weight > 1.0:
            LOGGER.warning(f"Invalid weight {signal.target_weight} for {signal.symbol}")
            return False
        
        if signal.confidence < 0 or signal.confidence > 1.0:
            LOGGER.warning(f"Invalid confidence {signal.confidence} for {signal.symbol}")
            return False
        
        if signal.stop_loss <= 0 or signal.entry_price <= 0:
            LOGGER.warning(f"Invalid prices for {signal.symbol}")
            return False
        
        return True

