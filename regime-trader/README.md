# regime-trader

Volatility regime detection and allocation trading system using Hidden Markov Models (HMM) with MetaTrader5/Exness integration.

## Status

Phases 1-3 scaffold complete:
- Phase 1: Project structure and configuration
- Phase 2: HMM regime detection engine with forward-filtered inference (no look-ahead bias)
- Phase 3: Volatility-based allocation strategies (long-only)

Business logic implementation in progress.

## Structure

- `config/`: YAML settings and credential template for Exness/MetaTrader5.
- `core/`: HMM regime detection, volatility-based strategies, risk management, and signal generation.
- `broker/`: MetaTrader5 client wrapper, order execution, and position tracking.
- `data/`: Market data fetching and technical feature engineering.
- `monitoring/`: Structured logging, live dashboard, and alert routing.
- `backtest/`: Walk-forward backtesting engine and performance analytics.
- `tests/`: Unit tests for HMM forward algorithm, look-ahead bias detection, and strategies.

## Quick Start

1. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy environment template and fill with Exness credentials:

   ```bash
   cp .env.example .env
   # Edit .env and add:
   # EXNESS_ACCOUNT=your_account_number
   # EXNESS_PASSWORD=your_password
   # EXNESS_SERVER=Exness-Demo
   ```

4. Copy credential template and fill broker values:

   ```bash
   cp config/credentials.yaml.example config/credentials.yaml
   # Edit config/credentials.yaml with your Exness trading credentials
   ```

## Key Design Principles

- **Volatility Detection, Not Prediction**: HMM detects volatility regimes (calm, moderate, turbulent) for allocation sizing, not market direction.
- **Long-Only Allocation**: Markets have upward drift; shorting during recovery spikes destroys returns. Stays invested with varying levels based on volatility.
- **No Look-Ahead Bias**: Forward algorithm for regime inference processes only past and present data, eliminating backtest curve-fitting.
- **Uncertainty Handling**: Halves position sizes and forces leverage to 1.0x during low-confidence and flickering regimes.
- **Anti-Churn Rebalancing**: Only rebalances when target allocation differs by >10% from current to minimize slippage.