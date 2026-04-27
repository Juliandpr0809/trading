#!/usr/bin/env python3
"""Main orchestration loop for regime-detection trading system.

Modes:
  python main.py                # Live trading with MetaTrader5
  python main.py --dry-run      # Full pipeline without orders
  python main.py --train-only   # Train HMM and exit
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows to avoid UnicodeEncodeError with cp1252
if sys.platform == "win32":
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

import pandas as pd
import yaml

# ── Project imports ───────────────────────────────────────────────
from backtest.backtester import WalkForwardBacktester
from backtest.performance import PerformanceAnalyzer
from broker.mt5_client import MT5Client
from broker.order_executor import OrderExecutor
from broker.position_tracker import PositionTracker
from core.hmm_engine import HMMConfig, HMMEngine
from core.regime_strategies import (
    RegimeStrategy,
    StrategyConfig,
    TechnicalSetup,
    detect_pullback,
    Direction,
)
from core.risk_manager import CircuitBreakerType, RiskManager
from data.feature_engineering import (
    FeatureEngineer,
    compute_observable_features,
    compute_raw_features,
)
from data.market_data import MarketDataClient

# ── Logging ───────────────────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/trading.log"),
    ],
)
LOGGER = logging.getLogger(__name__)


# ==================================================================
# Trading System
# ==================================================================


class TradingSystem:
    """Main trading system orchestrator.

    Pipeline (per M5 bar):
      1. Fetch latest M5 candle (closed)
      2. Compute features -> HMM regime
      3. Evaluate EMA9/EMA200/VWAP strategy signal
      4. Risk manager veto/approval
      5. Submit to MT5 via OrderExecutor
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        dry_run: bool = False,
    ) -> None:
        self.config_path = config_path
        self.dry_run = dry_run
        self.config = self._load_config()

        # Components (initialized in startup)
        self.mt5_client: MT5Client | None = None
        self.market_data: MarketDataClient | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.hmm_engine: HMMEngine | None = None
        self.strategy: RegimeStrategy | None = None
        self.risk_manager: RiskManager | None = None
        self.order_executor: OrderExecutor | None = None
        self.position_tracker: PositionTracker | None = None

        # State
        self.is_running = False
        self.last_bar_time: datetime | None = None
        self.account_equity: float = 0.0
        self.session_state = {
            "start_time": datetime.now().isoformat(),
            "orders_submitted": 0,
            "signals_generated": 0,
            "signals_rejected": 0,
            "bars_processed": 0,
            "hmm_retrains": 0,
        }

        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            LOGGER.warning("Config not found: %s", self.config_path)
            return {}

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def startup(self) -> bool:
        """Initialize all sub-systems. Returns True on success."""
        try:
            LOGGER.info("=" * 60)
            LOGGER.info("REGIME TRADER — STARTUP")
            LOGGER.info("=" * 60)

            # 1. MT5 connection
            LOGGER.info("Connecting to MetaTrader5 (Exness)…")
            self.mt5_client = MT5Client()
            if not self.mt5_client.initialize():
                LOGGER.error("MT5 initialization failed")
                return False
            if not self.mt5_client.login():
                LOGGER.error("MT5 login failed")
                return False
            if not self.mt5_client.health_check():
                LOGGER.error("MT5 health check failed")
                return False

            account = self.mt5_client.get_account()
            self.account_equity = account.get("equity", 0.0)
            LOGGER.info(
                "[OK] Account %s | Equity $%,.2f | Leverage %dx",
                account.get("account_id"),
                self.account_equity,
                account.get("leverage", 0),
            )

            # 2. Market data client (live MT5)
            self.market_data = MarketDataClient(data_source="mt5")

            # 3. Feature engineer
            self.feature_engineer = FeatureEngineer()

            # 4. HMM engine
            hmm_cfg = HMMConfig(
                n_components=self.config.get("hmm", {}).get(
                    "n_components", [2, 3, 4]
                ),
                cv_tol=self.config.get("hmm", {}).get("cv_tol", 1e-3),
                cv_max_iter=self.config.get("hmm", {}).get("cv_max_iter", 200),
                train_bars=self.config.get("hmm", {}).get("train_bars", 504),
                stability_bars=self.config.get("hmm", {}).get("stability_bars", 5),
                flicker_window=self.config.get("hmm", {}).get("flicker_window", 40),
                flicker_threshold=self.config.get("hmm", {}).get(
                    "flicker_threshold", 2
                ),
                min_confidence=self.config.get("hmm", {}).get("min_confidence", 0.95),
            )
            self.hmm_engine = HMMEngine(hmm_cfg)

            hmm_path = Path(
                self.config.get("model_path", "models/hmm_model.pkl")
            )
            if hmm_path.exists():
                age_days = (
                    datetime.now()
                    - datetime.fromtimestamp(hmm_path.stat().st_mtime)
                ).days
                if age_days > 7:
                    LOGGER.info("HMM model is %d days old -- retraining", age_days)
                    self._retrain_hmm()
                else:
                    LOGGER.info("Loading HMM model (%d days old)", age_days)
                    self.hmm_engine.load_model(str(hmm_path))
            else:
                LOGGER.info("No HMM model found -- training")
                self._retrain_hmm()

            # 5. Strategy
            self.strategy = RegimeStrategy(StrategyConfig())

            # 6. Risk manager
            self.risk_manager = RiskManager()
            self.risk_manager.set_account_equity(self.account_equity)

            # 7. Position tracker
            self.position_tracker = PositionTracker()
            self.position_tracker.refresh()

            # 8. Order executor
            self.order_executor = OrderExecutor(mt5_client=self.mt5_client)

            # 9. Recovery state
            recovery = self._load_recovery_state()
            if recovery:
                self.session_state = recovery
                LOGGER.info("Recovered state from %s", recovery.get("start_time"))

            self.is_running = True
            LOGGER.info("[OK] System online. Entering main loop…")
            return True

        except Exception as exc:
            LOGGER.error("Startup failed: %s", exc, exc_info=True)
            return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def main_loop(self) -> None:
        """Run main trading loop — one iteration per new M5 bar close."""
        bar_count = 0
        LOGGER.info("Main loop started")

        try:
            while self.is_running:
                try:
                    current_bar = self._get_current_bar_time()

                    if current_bar == self.last_bar_time:
                        time.sleep(1)
                        continue

                    self.last_bar_time = current_bar
                    bar_count += 1

                    self._process_bar(bar_count)

                    # Weekly HMM retrain (1344 M5 bars ≈ 7 trading days)
                    if bar_count % 1344 == 0:
                        LOGGER.info("Weekly HMM retraining…")
                        self._retrain_hmm()

                except KeyboardInterrupt:
                    break
                except Exception as exc:
                    LOGGER.error("Loop error: %s", exc, exc_info=True)
                    time.sleep(5)
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def _process_bar(self, bar_count: int) -> None:
        """Process a single M5 bar close."""
        symbols = self.config.get("broker", {}).get("symbols", ["USTEC"])

        # Refresh account equity
        account = self.mt5_client.get_account()
        self.account_equity = account.get("equity", self.account_equity)
        margin_free = account.get("margin_free", 0.0)

        # Check circuit breaker
        positions_snap = self.position_tracker.refresh()
        portfolio_state = self.risk_manager.compute_portfolio_state(
            equity=self.account_equity,
            cash=margin_free,
            positions_list=[],  # Simplified: we track via MT5
        )

        if portfolio_state.circuit_breaker_active in {
            CircuitBreakerType.DAILY_DD_HALT,
            CircuitBreakerType.WEEKLY_DD_HALT,
            CircuitBreakerType.PEAK_DD_HALT,
        }:
            LOGGER.warning(
                "[HALT] Circuit breaker active: %s — closing all.",
                portfolio_state.circuit_breaker_active.value,
            )
            self.position_tracker.close_all_positions()
            return

        # Process each symbol
        for symbol in symbols:
            self._process_symbol(symbol, portfolio_state, margin_free)

        self.session_state["bars_processed"] = bar_count
        self._save_recovery_state()

    def _process_symbol(
        self,
        symbol: str,
        portfolio_state: Any,
        margin_free: float,
    ) -> None:
        """Generate and execute signal for a single symbol."""
        try:
            # ── 1. Fetch latest M5 data ───────────────────────
            df = self.market_data.get_live_bars(symbol, "M5", bars=800)
            if df.empty or len(df) < 250:
                LOGGER.warning("Insufficient data for %s (%d bars)", symbol, len(df))
                return

            # ── 2. Compute indicators ─────────────────────────
            df = self.market_data.compute_technical_indicators(df)

            # ── 3. HMM regime ────────────────────────────────
            features = self.feature_engineer.compute_features(df)
            if features is None or features.empty:
                LOGGER.warning("Feature computation returned empty for %s", symbol)
                return

            regime_state = self.hmm_engine.predict_regime_filtered(features)
            regime_info = self.hmm_engine.regime_info.get(
                regime_state.current_regime_id
            )
            if not regime_info:
                return

            LOGGER.info(
                "Regime: %s (id=%d, conf=%.2f%%, stable=%s)",
                regime_state.regime_label,
                regime_state.current_regime_id,
                regime_state.state_probability * 100,
                regime_state.is_stable,
            )

            # ── 4. Build technical setup from LAST CLOSED bar ─
            last = df.iloc[-1]
            setup = TechnicalSetup(
                price=float(last["close"]),
                ema_9=float(last["ema_9"]),
                ema_200=float(last["ema_200"]),
                vwap=float(last["vwap"]),
                atr_14=float(last["atr"]),
                swing_high=float(df["high"].tail(20).max()),
                swing_low=float(df["low"].tail(20).min()),
                distance_from_ema200=(
                    float(last["close"]) - float(last["ema_200"])
                ),
                had_pullback=detect_pullback(
                    df,
                    Direction.LONG
                    if float(last["close"]) > float(last["ema_200"])
                    else Direction.SHORT,
                    lookback=5,
                ),
            )

            # ── 5. Evaluate strategy signal ───────────────────
            strat_signal = self.strategy.evaluate_signal(
                symbol=symbol,
                regime_state=regime_state,
                regime_info_map=self.hmm_engine.regime_info,
                setup=setup,
                account_equity=self.account_equity,
            )

            self.session_state["signals_generated"] += 1

            if strat_signal.direction == "FLAT":
                LOGGER.info(
                    "Signal FLAT for %s: %s", symbol, strat_signal.reasoning
                )
                return

            LOGGER.info(
                "Signal: %s %s @ $%.2f | SL=$%.2f  TP=$%.2f  lots=%.2f | %s",
                strat_signal.direction,
                symbol,
                strat_signal.entry_price,
                strat_signal.stop_loss,
                strat_signal.take_profit,
                strat_signal.lots,
                strat_signal.reasoning,
            )

            # ── 6. Risk manager validation ────────────────────
            risk_decision = self.risk_manager.validate_signal(
                signal=strat_signal,
                portfolio_state=portfolio_state,
                mt5_client=self.mt5_client,
            )

            if not risk_decision.approved:
                LOGGER.info(
                    "Signal REJECTED for %s: %s",
                    symbol,
                    risk_decision.rejection_reason,
                )
                self.session_state["signals_rejected"] += 1
                return

            final_signal = risk_decision.modified_signal or strat_signal

            # ── 7. Execute ────────────────────────────────────
            if self.dry_run:
                LOGGER.info(
                    "[DRY RUN] Would submit: %s %.2f %s",
                    final_signal.direction,
                    final_signal.lots,
                    symbol,
                )
            else:
                response = self.order_executor.submit_order(final_signal)
                if response.success:
                    self.session_state["orders_submitted"] += 1
                    self.risk_manager.record_trade(symbol, final_signal.direction)
                    LOGGER.info("[OK] Order filled — ticket %s", response.ticket)
                else:
                    LOGGER.error("Order rejected: %s", response.error_message)

        except Exception as exc:
            LOGGER.error("Symbol processing error (%s): %s", symbol, exc, exc_info=True)

    # ------------------------------------------------------------------
    # HMM training
    # ------------------------------------------------------------------

    def _retrain_hmm(self) -> None:
        """Retrain HMM model with latest daily data."""
        try:
            symbols = self.config.get("broker", {}).get("symbols", ["USTEC"])
            data = self.market_data.fetch_historical(
                symbols[0], timeframe="D1", bars=3000
            )
            if data.empty:
                LOGGER.error("No data for HMM training")
                return

            features = compute_observable_features(data)
            if features.empty:
                raw = compute_raw_features(data)
                na_summary = (
                    raw.isna()
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .to_dict()
                )
                LOGGER.error(
                    "Features empty after engineering. NaN summary: %s", na_summary
                )
                return

            LOGGER.info("HMM training on %s features", features.shape)
            self.hmm_engine.fit(features)

            model_path = Path(
                self.config.get("model_path", "models/hmm_model.pkl")
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            self.hmm_engine.save_model(str(model_path))

            self.session_state["hmm_retrains"] += 1
            LOGGER.info("[OK] HMM model trained and saved")

        except Exception as exc:
            LOGGER.error("HMM retraining failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_current_bar_time() -> datetime:
        """Get current M5 bar close time (quantized to 5-min boundary)."""
        now = datetime.now()
        bar_minute = (now.minute // 5) * 5
        return now.replace(minute=bar_minute, second=0, microsecond=0)

    def _load_recovery_state(self) -> dict | None:
        try:
            f = Path("state_snapshot.json")
            if f.exists():
                return json.loads(f.read_text())
        except Exception as exc:
            LOGGER.warning("Recovery state load failed: %s", exc)
        return None

    def _save_recovery_state(self) -> None:
        try:
            Path("state_snapshot.json").write_text(
                json.dumps(self.session_state, indent=2)
            )
        except Exception as exc:
            LOGGER.error("Recovery state save failed: %s", exc)

    def _handle_shutdown(self, signum: int, frame: Any) -> None:
        LOGGER.info("Shutdown signal received (%s)", signum)
        self.is_running = False

    def _shutdown(self) -> None:
        LOGGER.info("=" * 60)
        LOGGER.info("REGIME TRADER — SHUTDOWN")
        LOGGER.info("=" * 60)
        self._save_recovery_state()
        self._print_session_summary()
        if self.mt5_client:
            self.mt5_client.shutdown()
        LOGGER.info("[OK] Shutdown complete")

    def _print_session_summary(self) -> None:
        LOGGER.info("Bars processed:    %d", self.session_state["bars_processed"])
        LOGGER.info("Signals generated: %d", self.session_state["signals_generated"])
        LOGGER.info("Signals rejected:  %d", self.session_state["signals_rejected"])
        LOGGER.info("Orders submitted:  %d", self.session_state["orders_submitted"])
        LOGGER.info("HMM retrains:      %d", self.session_state["hmm_retrains"])


# ==================================================================
# Entry point
# ==================================================================


def run_train_only() -> None:
    """Train HMM and exit."""
    LOGGER.info("Training HMM model…")

    # Connect to MT5 for data
    client = MT5Client()
    if not client.initialize() or not client.login():
        LOGGER.error("MT5 connection required for training data")
        sys.exit(1)

    market_data = MarketDataClient(data_source="mt5")
    data = market_data.fetch_historical("USTEC", timeframe="D1", bars=3000)

    if data.empty:
        LOGGER.error("No data retrieved")
        client.shutdown()
        sys.exit(1)

    features = compute_observable_features(data)
    if features.empty:
        LOGGER.error("Features empty after engineering")
        client.shutdown()
        sys.exit(1)

    hmm = HMMEngine(
        HMMConfig(n_components=[2, 3, 4])
    )
    hmm.fit(features)

    Path("models").mkdir(exist_ok=True)
    hmm.save_model("models/hmm_model.pkl")
    client.shutdown()
    LOGGER.info("[OK] HMM model trained and saved")


def run_backtester(config_path: str) -> None:
    """Run walk-forward backtest from local daily CSV data."""
    LOGGER.info("Starting walk-forward backtest…")

    config: dict[str, Any] = {}
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        LOGGER.warning("Config not found: %s", config_path)

    price_data_path = config.get("backtest_data_path", "data/feeds/USTEC_D1.csv")

    try:
        price_data = pd.read_csv(price_data_path)
        if "datetime" in price_data.columns:
            price_data["datetime"] = pd.to_datetime(price_data["datetime"])
            price_data = price_data.set_index("datetime")
        elif "Date" in price_data.columns:
            price_data["Date"] = pd.to_datetime(price_data["Date"])
            price_data = price_data.set_index("Date")
        else:
            raise ValueError(
                f"Missing datetime column in {price_data_path}. Found {price_data.columns.tolist()}"
            )
    except Exception as exc:
        LOGGER.error("Failed to load price data: %s", exc)
        sys.exit(1)

    price_data.columns = price_data.columns.str.lower()
    LOGGER.info("Loaded %d bars from %s", len(price_data), price_data_path)

    backtester = WalkForwardBacktester()
    results = backtester.run(price_data)

    if not results:
        LOGGER.error("Backtest produced no results")
        sys.exit(1)

    equity_curve: list[float] = []
    for result in results:
        equity_curve.extend(result.equity_curve)

    if not equity_curve:
        LOGGER.error("No equity curve data available")
        sys.exit(1)

    metrics = PerformanceAnalyzer().compute_metrics(equity_curve)

    LOGGER.info("=" * 60)
    LOGGER.info("BACKTEST RESULTS")
    LOGGER.info("=" * 60)
    LOGGER.info("Total Return: %.2f%%", metrics.total_return_pct)
    LOGGER.info("Sharpe Ratio: %.2f", metrics.sharpe_ratio)
    LOGGER.info("Sortino Ratio: %.2f", metrics.sortino_ratio)
    LOGGER.info("Calmar Ratio: %.2f", metrics.calmar_ratio)
    LOGGER.info("Max Drawdown: %.2f%%", metrics.max_drawdown_pct)
    LOGGER.info("Win Rate: %.1f%%", metrics.win_rate * 100)
    LOGGER.info("Profit Factor: %.2f", metrics.profit_factor)
    LOGGER.info("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regime-detection trading system (HMM + EMA/VWAP + MT5)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Signals only, no orders"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Train HMM and exit"
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run walk-forward backtest"
    )
    parser.add_argument(
        "--config", default="config/settings.yaml", help="Config file path"
    )
    args = parser.parse_args()

    if args.train_only:
        run_train_only()
        return

    if args.backtest:
        run_backtester(args.config)
        return

    system = TradingSystem(config_path=args.config, dry_run=args.dry_run)
    if not system.startup():
        LOGGER.error("Startup failed")
        sys.exit(1)

    system.main_loop()


if __name__ == "__main__":
    main()
