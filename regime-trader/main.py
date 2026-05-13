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
from datetime import date, datetime, time as dt_time, timedelta
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
from backtest.backtester import BacktestConfig
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
from monitoring.alert_system import AlertSystem

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


def _parse_utc_offset(timezone_str: str) -> float:
    """Parse strings like UTC+0, UTC-5, UTC+5.5 into hour offsets."""
    raw = (timezone_str or "UTC+0").strip().upper()
    if not raw.startswith("UTC"):
        return 0.0
    suffix = raw[3:]
    if not suffix:
        return 0.0
    try:
        return float(suffix)
    except ValueError:
        return 0.0


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
        strategy: str = "regime",
    ) -> None:
        self.config_path = config_path
        self.dry_run = dry_run
        self.strategy_mode = strategy  # "regime" or "liquidity"
        if self.strategy_mode not in ("regime", "liquidity"):
            raise ValueError(f"Unknown strategy: {strategy}. Must be 'regime' or 'liquidity'")
        self.config = self._load_config()

        # Components (initialized in startup)
        self.mt5_client: MT5Client | None = None
        self.market_data: MarketDataClient | None = None
        self.feature_engineer: FeatureEngineer | None = None
        self.hmm_engine: HMMEngine | None = None
        self.strategy: RegimeStrategy | None = None  # For regime strategy only
        self.liquidity_strategy: Any | None = None  # For liquidity strategy
        self.alert_system: AlertSystem | None = None
        self.risk_manager: RiskManager | None = None
        self.order_executor: OrderExecutor | None = None
        self.position_tracker: PositionTracker | None = None
        self.paper_trading_logger: Any | None = None  # For liquidity paper trading

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

    def _parse_time(self, time_str: str) -> dt_time:
        """Parse HH:MM format to dt_time."""
        if isinstance(time_str, dt_time):
            return time_str
        parts = time_str.split(":")
        return dt_time(int(parts[0]), int(parts[1]))

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
                "[OK] Account %s | Equity %s | Leverage %dx",
                account.get("account_id"),
                f"${self.account_equity:,.2f}",
                account.get("leverage", 0),
            )

            # 2. Market data client (live MT5)
            self.market_data = MarketDataClient(data_source="mt5")

            # 3. Feature engineer
            self.feature_engineer = FeatureEngineer()

            # 4. HMM engine (only for Regime strategy)
            if self.strategy_mode == "regime":
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
            else:
                LOGGER.info("HMM engine skipped (not needed for Liquidity Sweep)")

            # 5. Strategy (conditional based on strategy_mode)
            if self.strategy_mode == "regime":
                LOGGER.info("Loading Regime Strategy (HMM + EMA/VWAP)…")
                self.strategy = RegimeStrategy(StrategyConfig())
            elif self.strategy_mode == "liquidity":
                LOGGER.info("Loading Liquidity Sweep Alert Strategy…")
                from core.liquidity_strategy import LiquidityConfig, LiquidityStrategy

                liq_cfg = self.config.get("liquidity_strategy", {})
                strategy_cfg = LiquidityConfig.from_yaml(liq_cfg)
                self.liquidity_strategy = LiquidityStrategy(config=strategy_cfg)
                self.alert_system = AlertSystem(strategy_cfg)
                LOGGER.info("[OK] Alert system initialized: %s", strategy_cfg.alert_log_path)
            else:
                LOGGER.error(f"Unknown strategy: {self.strategy_mode}")
                return False

            # 6. Risk manager
            self.risk_manager = RiskManager()
            self.risk_manager.set_account_equity(self.account_equity)

            # 7. Position tracker
            self.position_tracker = PositionTracker()
            self.position_tracker.refresh()

            # 8. Order executor (regime mode only)
            if self.strategy_mode == "regime":
                self.order_executor = OrderExecutor(
                    mt5_client=self.mt5_client,
                    magic_number=202401,
                )
                LOGGER.info("Order executor initialized with magic_number=%d", 202401)
            else:
                self.order_executor = None
                LOGGER.info("Liquidity mode: order execution is disabled by design")

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
        """Route to appropriate main loop based on strategy mode."""
        if self.strategy_mode == "regime":
            self._main_loop_regime()
        elif self.strategy_mode == "liquidity":
            self._main_loop_liquidity()
        else:
            LOGGER.error(f"Unknown strategy mode: {self.strategy_mode}")

    def _main_loop_regime(self) -> None:
        """Run main trading loop for Regime strategy — one iteration per new M5 bar close."""
        bar_count = 0
        LOGGER.info("Main loop started (Regime strategy)")

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

    def _main_loop_liquidity(self) -> None:
        """Run main trading loop for Liquidity Sweep strategy — session-based (day-by-day)."""
        LOGGER.info("Main loop started (Liquidity Sweep strategy)")
        self._main_loop_liquidity_live()

    def _main_loop_liquidity_dry_run(self) -> None:
        """Deprecated: dry-run is replay-only for liquidity mode."""
        LOGGER.error("--dry-run is only supported with --replay in liquidity mode")
        self.is_running = False
        self._shutdown()

    def _main_loop_liquidity_live(self) -> None:
        """Live alert loop: detect setups and notify trader, never execute orders."""
        try:
            cfg = self.liquidity_strategy.config
            server_utc_offset = _parse_utc_offset(cfg.server_timezone)

            print("=" * 60)
            print("  LIQUIDITY SWEEP ALERT SYSTEM")
            print("  Estrategia 2 - Solo alertas, sin ejecucion")
            print(f"  Instrumento: {cfg.symbol} | Magic: {cfg.magic_number}")
            print(f"  Sesion: {cfg.session_start_server_time} - {cfg.session_end_server_time} MT5")
            print(f"  Escaneo HH/LL: {cfg.scan_start_server_time} - {cfg.scan_end_server_time} MT5")
            print(f"  Entry cutoff: {cfg.entry_cutoff_server_time} MT5")
            print("-" * 60)
            print("  Este sistema NO ejecuta ordenes")
            print("  Tu decides y ejecutas manualmente en MT5")
            print("=" * 60)

            now_server = datetime.utcnow() + timedelta(hours=server_utc_offset)
            if now_server.time() < cfg.session_start_time:
                wait_seconds = (
                    datetime.combine(now_server.date(), cfg.session_start_time)
                    - datetime.combine(now_server.date(), now_server.time())
                ).seconds
                LOGGER.info("Esperando sesion en %d minutos...", wait_seconds // 60)
                time.sleep(wait_seconds)

            setup_sent = False
            last_processed_bar = None
            while self.is_running:
                now_server = datetime.utcnow() + timedelta(hours=server_utc_offset)
                if now_server.time() > cfg.session_end_time:
                    break

                m30_data = self.market_data.get_live_bars(cfg.symbol, cfg.timeframe_context, bars=120)
                m5_data = self.market_data.get_live_bars(cfg.symbol, cfg.timeframe_entry, bars=900)
                if m30_data.empty or m5_data.empty:
                    LOGGER.info("Sin datos suficientes para evaluar session actual")
                else:
                    bar_age_seconds = (now_server - m5_data.index[-1].to_pydatetime()).total_seconds()
                    if bar_age_seconds > 360:
                        LOGGER.warning("Datos M5 stale (%.0fs). Esperando barra fresca...", bar_age_seconds)
                    elif last_processed_bar != m5_data.index[-1]:
                        last_processed_bar = m5_data.index[-1]
                        signal_found = self.liquidity_strategy.evaluate_session(
                            m30_data=m30_data,
                            m5_data=m5_data,
                            session_date=pd.Timestamp(now_server.date()),
                            account_equity=self.account_equity,
                        )
                        if signal_found is not None and not setup_sent:
                            self.alert_system.send_alert(signal_found)
                            setup_sent = True

                seconds_to_next = 300 - ((now_server.second + (now_server.minute * 60)) % 300)
                time.sleep(seconds_to_next + 2)

            if not setup_sent:
                self.alert_system.log_no_setup(
                    date=(datetime.utcnow() + timedelta(hours=server_utc_offset)).date().isoformat(),
                    reason="No se detecto setup valido en la sesion",
                )
        except KeyboardInterrupt:
            LOGGER.info("Liqudity alert loop interrupted by user")
        except Exception as exc:
            LOGGER.error("Liquidity alert loop error: %s", exc, exc_info=True)
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Bar processing (Regime only)
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

    # Force load H1 full data if available
    from pathlib import Path as PathlibPath
    h1_path = PathlibPath("data/feeds/USTEC_H1_full.csv").resolve()
    m5_path = PathlibPath("data/feeds/USTEC_M5.csv").resolve()
    
    if h1_path.exists():
        price_data_path = str(h1_path)
        LOGGER.info("Using H1 full data: %s", price_data_path)
    elif m5_path.exists():
        price_data_path = str(m5_path)
        LOGGER.info("Using M5 data (H1 not found): %s", price_data_path)
    else:
        price_data_path = config.get("backtest_data_path", "data/feeds/USTEC_H1_full.csv")
        LOGGER.info("Using config path: %s", price_data_path)

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

    # Resample to configured timeframe if needed (only for M5->H1 conversion)
    target_timeframe = config.get("broker", {}).get("timeframe", "M5")
    source_timeframe = "H1" if "H1" in price_data_path else "M5"
    if target_timeframe == "H1" and source_timeframe == "M5":
        LOGGER.info("Resampling M5 data to H1...")
        price_data = price_data.resample("1h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        }).dropna()
        LOGGER.info("Resampled to H1: %d bars", len(price_data))
    elif target_timeframe == "H1":
        LOGGER.info("Data already in H1: %d bars", len(price_data))

    backtest_cfg = BacktestConfig(
        is_window=config.get("backtest", {}).get("is_window", 14400),
        oos_window=config.get("backtest", {}).get("oos_window", 2880),
        step_size=config.get("backtest", {}).get("step_size", 2880),
        min_train_bars=config.get("backtest", {}).get("min_train_bars", 8640),
    )
    hmm_cfg = HMMConfig(
        n_components=config.get("hmm", {}).get("n_components", [2, 3, 4]),
        cv_tol=config.get("hmm", {}).get("cv_tol", 1e-3),
        cv_max_iter=config.get("hmm", {}).get("cv_max_iter", 200),
        train_bars=config.get("hmm", {}).get("train_bars", 14400),
        stability_bars=config.get("hmm", {}).get("stability_bars", 12),
        flicker_window=config.get("hmm", {}).get("flicker_window", 120),
        flicker_threshold=config.get("hmm", {}).get("flicker_threshold", 3),
        min_confidence=config.get("hmm", {}).get("min_confidence", 0.90),
        retrain_interval_bars=config.get("hmm", {}).get("retrain_interval_bars", 288),
        min_train_bars=config.get("hmm", {}).get("min_train_bars", 8640),
    )
    strategy_cfg = StrategyConfig(
        risk_percent=config.get("strategy", {}).get("risk_percent", 0.01),
        ema_9_period=config.get("strategy", {}).get("ema_9_period", 9),
        ema_200_period=config.get("strategy", {}).get("ema_200_period", 200),
        atr_period=config.get("strategy", {}).get("atr_period", 14),
        pullback_lookback=config.get("strategy", {}).get("pullback_lookback", 5),
        swing_lookback=config.get("strategy", {}).get("swing_lookback", 20),
        exhaustion_atr_threshold=config.get("strategy", {}).get("exhaustion_atr_threshold", 2.0),
        rr_ratio=config.get("strategy", {}).get("rr_ratio", 2.0),
    )

    backtester = WalkForwardBacktester(
        backtest_config=backtest_cfg,
        hmm_config=hmm_cfg,
        strategy_config=strategy_cfg,
    )
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
    all_trades: list[Any] = []
    for result in results:
        all_trades.extend(result.trades)
    trade_metrics = PerformanceAnalyzer().compute_trade_metrics(all_trades)

    LOGGER.info(
        "DATA CHECK: bars=%d, date_start=%s, date_end=%s, close_min=%.2f, close_max=%.2f",
        len(price_data),
        price_data.index[0],
        price_data.index[-1],
        price_data["close"].min(),
        price_data["close"].max(),
    )

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
    LOGGER.info("TRADE-LEVEL METRICS (DETALLADO)")
    LOGGER.info("Total Trades (closed):  %d", trade_metrics["total_trades"])
    LOGGER.info("Win Rate (por trade):   %.1f%%", trade_metrics["win_rate"] * 100)
    LOGGER.info("Profit Factor:          %.2f", trade_metrics["profit_factor"])
    LOGGER.info("RR Logrado:             %.2f", trade_metrics["avg_rr_achieved"])
    LOGGER.info("RR Diseñado:            2.00")
    LOGGER.info("Expectancy/trade:       $%.2f", trade_metrics["expectancy_per_trade"])
    LOGGER.info("Exits por TP:           %d", trade_metrics["tp_hit_count"])
    LOGGER.info("Exits por SL:           %d", trade_metrics["sl_hit_count"])
    LOGGER.info("Exits por Régimen FLAT: %d", trade_metrics["flat_exit_count"])
    LOGGER.info("=" * 60)


def run_liquidity_backtester(config_path: str) -> None:
    """Run walk-forward backtest for the Liquidity Sweep strategy on US30."""
    LOGGER.info("Starting Liquidity Sweep walk-forward backtest…")

    config: dict[str, Any] = {}
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        LOGGER.warning("Config not found: %s", config_path)

    liq_cfg = config.get("liquidity_strategy", {})
    bt_cfg = config.get("liquidity_backtest", {})
    symbol = liq_cfg.get("symbol", "US30m")

    # Load M30 and M5 data
    from pathlib import Path as PathlibPath
    feeds_dir = PathlibPath("data/feeds")
    m30_path = feeds_dir / f"{symbol}_M30.csv"
    m5_path = feeds_dir / f"{symbol}_M5.csv"

    if not m30_path.exists():
        LOGGER.error("M30 data not found: %s. Run data/download_us30.py first.", m30_path)
        sys.exit(1)
    if not m5_path.exists():
        LOGGER.error("M5 data not found: %s. Run data/download_us30.py first.", m5_path)
        sys.exit(1)

    m30_data = pd.read_csv(m30_path, parse_dates=["datetime"], index_col="datetime")
    m5_data = pd.read_csv(m5_path, parse_dates=["datetime"], index_col="datetime")
    m30_data.columns = m30_data.columns.str.lower()
    m5_data.columns = m5_data.columns.str.lower()

    LOGGER.info("Loaded M30: %d bars (%s to %s)", len(m30_data), m30_data.index[0], m30_data.index[-1])
    LOGGER.info("Loaded M5:  %d bars (%s to %s)", len(m5_data), m5_data.index[0], m5_data.index[-1])

    # Parse session times
    from datetime import time as dt_time_import
    s_start = liq_cfg.get("session_start_server_time", "13:00")
    s_end = liq_cfg.get("session_end_server_time", "16:30")
    s_cutoff = liq_cfg.get("entry_cutoff_server_time", "15:30")
    h1, m1 = map(int, s_start.split(":"))
    h2, m2 = map(int, s_end.split(":"))
    h3, m3 = map(int, s_cutoff.split(":"))

    from core.liquidity_strategy import LiquidityConfig
    from backtest.session_backtester import WalkForwardSessionBacktester, SessionBacktestConfig

    strat_config = LiquidityConfig(
        symbol=symbol,
        session_start_server_time=f"{h1:02d}:{m1:02d}",
        session_end_server_time=f"{h2:02d}:{m2:02d}",
        entry_cutoff_server_time=f"{h3:02d}:{m3:02d}",
        scan_start_server_time=liq_cfg.get("scan_start_server_time", "02:00"),
        scan_end_server_time=liq_cfg.get("scan_end_server_time", "11:00"),
        max_trades_per_day=liq_cfg.get("max_trades_per_day", 1),
        rr_ratio=liq_cfg.get("rr_ratio", 2.0),
        risk_percent=liq_cfg.get("risk_percent", 0.01),
        breakeven_at_rr=liq_cfg.get("breakeven_at_rr", 1.0),
        min_sl_points=liq_cfg.get("min_sl_points", 10),
        max_sl_points=liq_cfg.get("max_sl_points", 150),
        contract_size=liq_cfg.get("contract_size", 1.0),
    )
    backtest_config = SessionBacktestConfig(
        is_window_days=bt_cfg.get("is_window_days", 500),
        oos_window_days=bt_cfg.get("oos_window_days", 60),
        step_size_days=bt_cfg.get("step_size_days", 60),
        min_train_days=bt_cfg.get("min_train_days", 250),
        initial_capital=bt_cfg.get("initial_capital", 100000.0),
        slippage_points=bt_cfg.get("slippage_points", 2.0),
    )

    backtester = WalkForwardSessionBacktester(
        backtest_config=backtest_config,
        strategy_config=strat_config,
    )
    results = backtester.run(m30_data, m5_data)

    if not results:
        LOGGER.error("Backtest produced no results")
        sys.exit(1)

    # Aggregate all trades and equity
    all_trades = []
    equity_curve = []
    for r in results:
        all_trades.extend(r.trades)
        equity_curve.extend(r.equity_curve)

    # Compute metrics
    n = len(all_trades)
    if n == 0:
        LOGGER.error("No trades generated")
        sys.exit(1)

    winners = [t for t in all_trades if t.pnl > 0]
    losers = [t for t in all_trades if t.pnl <= 0]
    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    win_rate = len(winners) / n
    pf = gross_profit / (gross_loss + 1e-12)
    avg_win = gross_profit / max(len(winners), 1)
    avg_loss = gross_loss / max(len(losers), 1)
    true_rr = avg_win / (avg_loss + 1e-12) if avg_loss > 0 else 0.0
    expectancy = (gross_profit - gross_loss) / n

    tp_hits = sum(1 for t in all_trades if t.exit_reason == "TP_HIT")
    sl_hits = sum(1 for t in all_trades if t.exit_reason == "SL_HIT")
    trailing = sum(1 for t in all_trades if t.exit_reason == "TRAILING_EXIT")
    timeouts = sum(1 for t in all_trades if t.exit_reason == "TIMEOUT_2030")

    # Max drawdown
    peak = equity_curve[0] if equity_curve else 100000
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak
        max_dd = max(max_dd, dd)

    LOGGER.info("=" * 60)
    LOGGER.info("LIQUIDITY SWEEP BACKTEST RESULTS")
    LOGGER.info("=" * 60)
    LOGGER.info("Total Trades:       %d", n)
    LOGGER.info("Win Rate:           %.1f%%", win_rate * 100)
    LOGGER.info("Profit Factor:      %.2f", pf)
    LOGGER.info("TRUE RR:            %.2f", true_rr)
    LOGGER.info("Expectancy/trade:   $%.2f", expectancy)
    LOGGER.info("Max Drawdown:       %.1f%%", max_dd * 100)
    LOGGER.info("Avg Win:            $%.2f", avg_win)
    LOGGER.info("Avg Loss:           $%.2f", avg_loss)
    LOGGER.info("--- Exit Breakdown ---")
    LOGGER.info("  TP_HIT:           %d (%.1f%%)", tp_hits, 100 * tp_hits / n)
    LOGGER.info("  SL_HIT:           %d (%.1f%%)", sl_hits, 100 * sl_hits / n)
    LOGGER.info("  TRAILING_EXIT:    %d (%.1f%%)", trailing, 100 * trailing / n)
    LOGGER.info("  TIMEOUT_2030:     %d (%.1f%%)", timeouts, 100 * timeouts / n)
    LOGGER.info("=" * 60)


# ==================================================================
# Replay mode — offline single-day simulation for Strategy 2 (PBC)
# ==================================================================


def _broker_to_colombia(ts: datetime | None, broker_to_colombia_hours: float) -> datetime | None:
    """Convert broker timestamp to Colombia timestamp using configured hour offset."""
    if ts is None:
        return None
    return ts + timedelta(hours=broker_to_colombia_hours)


def _parse_timestamp(value: Any) -> datetime | None:
    """Parse timestamp values from datetime, pandas Timestamp or string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, str):
        try:
            return pd.Timestamp(value).to_pydatetime()
        except Exception:
            return None
    return None


def _format_mt5_colombia_time(value: Any, broker_to_colombia_hours: float) -> str:
    """Format a timestamp as broker and Colombia local times."""
    ts_mt5 = _parse_timestamp(value)
    if ts_mt5 is None:
        return "N/A"
    ts_col = _broker_to_colombia(ts_mt5, broker_to_colombia_hours)
    return f"{ts_mt5:%H:%M} MT5 / {ts_col:%I:%M %p} Colombia"


def run_replay(config_path: str, replay_date: str, override_symbol: str | None = None) -> None:
    """Replay a single day from CSVs bar-by-bar (no MT5 required).

    Usage:
        python main.py --strategy liquidity --replay --date 2026-03-24
        python main.py --strategy liquidity --replay --date 2026-03-24 --symbol USTEC
        python main.py --strategy liquidity --replay --date 2026-03-24 --symbol US30m
    """
    from core.liquidity_strategy import LiquidityConfig, LiquidityStrategy

    # ── Load config ──────────────────────────────────────────────
    config: dict[str, Any] = {}
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
    except FileNotFoundError:
        LOGGER.warning("Config not found: %s", config_path)

    liq_yaml = config.get("liquidity_strategy", {})
    cfg = LiquidityConfig.from_yaml(liq_yaml)
    
    # Override symbol if provided via --symbol argument
    if override_symbol:
        cfg.symbol = override_symbol

    target_date = pd.Timestamp(replay_date)
    day = target_date.date()
    session_start = pd.Timestamp.combine(day, cfg.session_start_time)
    session_end = pd.Timestamp.combine(day, cfg.session_end_time)
    day_start = pd.Timestamp.combine(day, dt_time(0, 0))

    LOGGER.info("=" * 60)
    LOGGER.info("REPLAY MODE — %s", day)
    LOGGER.info("Session: %s to %s (server time)", cfg.session_start_server_time, cfg.session_end_server_time)
    if cfg.use_colombia_session_times:
        LOGGER.info(
            "Session Colombia objetivo: %s to %s (cutoff %s)",
            cfg.session_start_colombia_time,
            cfg.session_end_colombia_time,
            cfg.entry_cutoff_colombia_time,
        )
        LOGGER.info(
            "Session broker derivada: %s to %s (cutoff %s)",
            cfg.session_start_time.strftime("%H:%M"),
            cfg.session_end_time.strftime("%H:%M"),
            cfg.entry_cutoff_time.strftime("%H:%M"),
        )
    LOGGER.info("=" * 60)

    # ── Load CSVs ────────────────────────────────────────────────
    feeds = Path("data/feeds")
    m30_path = feeds / f"{cfg.symbol}_M30.csv"
    m5_path = feeds / f"{cfg.symbol}_M5.csv"

    if not m30_path.exists():
        LOGGER.error("M30 data not found: %s", m30_path)
        sys.exit(1)
    if not m5_path.exists():
        LOGGER.error("M5 data not found: %s", m5_path)
        sys.exit(1)

    m30_all = pd.read_csv(m30_path, parse_dates=["datetime"], index_col="datetime")
    m5_all = pd.read_csv(m5_path, parse_dates=["datetime"], index_col="datetime")
    m30_all.columns = m30_all.columns.str.lower()
    m5_all.columns = m5_all.columns.str.lower()

    # Filter day's M30 (00:00 to session_end) and all M5 for the day
    m30_day = m30_all[(m30_all.index >= day_start) & (m30_all.index <= session_end)]
    m5_day = m5_all[(m5_all.index >= day_start) & (m5_all.index <= session_end)]

    if m30_day.empty:
        LOGGER.error("No M30 bars found for %s", day)
        sys.exit(1)
    if m5_day.empty:
        LOGGER.error("No M5 bars found for %s", day)
        sys.exit(1)

    LOGGER.info("M30 bars for day: %d (%s -> %s)", len(m30_day), m30_day.index[0], m30_day.index[-1])
    LOGGER.info("M5  bars for day: %d (%s -> %s)", len(m5_day), m5_day.index[0], m5_day.index[-1])

    strategy = LiquidityStrategy(cfg)
    signal_obj = strategy.evaluate_session(
        m30_data=m30_day,
        m5_data=m5_day,
        session_date=target_date,
        account_equity=100_000.0,
    )

    if signal_obj is None:
        _replay_no_signal(day, "No valid PBC setup in replay")
        return

    rr = float(signal_obj.metadata.get("rr_calculated", 0.0))
    broker_to_colombia_hours = float(cfg.broker_to_colombia_hours)
    sweep_time = signal_obj.metadata.get("sweep_bar_time")
    zona_low = float(signal_obj.metadata.get("pbc_zona_low", signal_obj.metadata.get("pvc_zona_low", 0.0)))
    zona_high = float(signal_obj.metadata.get("pbc_zona_high", signal_obj.metadata.get("pvc_zona_high", 0.0)))

    print("\n" + "=" * 70)
    print(f"  REPLAY RESULT - {day}")
    print("=" * 70)
    print(f"  Direccion:   {signal_obj.direction}")
    print(f"  Hora entrada:{_format_mt5_colombia_time(signal_obj.timestamp, broker_to_colombia_hours)}")
    print(f"  Entrada:     {signal_obj.entry_price:.1f}")
    print(f"  Stop Loss:   {signal_obj.stop_loss:.1f}")
    print(f"  Take Profit: {signal_obj.take_profit:.1f}")
    print(f"  RR:          {rr:.2f}")
    print(f"  Hora sweep:  {_format_mt5_colombia_time(sweep_time, broker_to_colombia_hours)}")
    print(f"  Sweep:       {signal_obj.metadata.get('sweep_direction', '')} @ {signal_obj.metadata.get('sweep_price', 0.0)}")
    print(f"  Zona PBC:    [{zona_low:.1f} - {zona_high:.1f}]")
    print(f"  HH/LL:       {signal_obj.metadata.get('hh_level', 0.0)} / {signal_obj.metadata.get('ll_level', 0.0)}")
    print("=" * 70 + "\n")


def _replay_no_signal(day, reason: str) -> None:
    """Print no-signal summary for replay."""
    print(f"\n{'═'*70}")
    print(f"  REPLAY RESULT — {day}")
    print(f"{'═'*70}")
    print(f"  {'DATE':<14}| {'DIR':<10}| {'ENTRY':>10} | {'SL':>10} | {'TP':>10} | {'RESULT':>10}")
    print(f"  {'─'*14}|{'─'*11}|{'─'*12}|{'─'*12}|{'─'*12}|{'─'*11}")
    print(f"  {str(day):<14}| {'—':<10}| {'—':>10} | {'—':>10} | {'—':>10} | {'NO SIGNAL':>10}")
    print(f"{'═'*70}")
    print(f"  Reason: {reason}")
    print(f"{'═'*70}\n")
    LOGGER.info("NO SIGNAL: %s", reason)


# ==================================================================
# Entry point
# ==================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trading system with Regime mode and Liquidity Alert mode"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Replay only for liquidity mode"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Train HMM and exit"
    )
    parser.add_argument(
        "--backtest", action="store_true", help="Run walk-forward backtest"
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="Replay a single historical day bar-by-bar (Strategy 2 only)",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date to replay in YYYY-MM-DD format (requires --replay)",
    )
    parser.add_argument(
        "--symbol", type=str, default=None,
        help="Symbol to replay (e.g. USTEC, US30m) - overrides config (replay only)",
    )
    parser.add_argument(
        "--strategy", default="regime", choices=["regime", "liquidity"],
        help="Strategy to use (default: regime)",
    )
    parser.add_argument(
        "--config", default="config/settings.yaml", help="Config file path"
    )
    args = parser.parse_args()

    if args.dry_run and not args.replay:
        LOGGER.error("--dry-run now only works together with --replay")
        sys.exit(1)

    if args.train_only:
        run_train_only()
        return

    if args.backtest:
        if args.strategy == "liquidity":
            run_liquidity_backtester(args.config)
        else:
            run_backtester(args.config)
        return

    if args.replay:
        if args.strategy != "liquidity":
            LOGGER.error("--replay is only available for --strategy liquidity")
            sys.exit(1)
        if not args.date:
            LOGGER.error("--replay requires --date YYYY-MM-DD")
            sys.exit(1)
        run_replay(args.config, args.date, override_symbol=args.symbol)
        return

    system = TradingSystem(
        config_path=args.config,
        dry_run=args.dry_run,
        strategy=args.strategy,
    )
    if not system.startup():
        LOGGER.error("Startup failed")
        sys.exit(1)

    system.main_loop()


if __name__ == "__main__":
    main()
