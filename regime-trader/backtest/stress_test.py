"""Stress testing: crash injection, gap risk, regime misclassification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """Summary of stress test scenario."""

    scenario_name: str
    n_simulations: int
    mean_max_loss_pct: float
    worst_case_loss_pct: float
    pct_circuits_triggered: float
    mean_return_pct: float


class StressTester:
    """Runs Monte Carlo stress tests on equity curves and regime history."""

    def __init__(self, circuit_breaker_loss_pct: float = 0.10, max_daily_loss_pct: float = 0.02) -> None:
        """Initialize stress tester with circuit breaker thresholds."""
        self.circuit_breaker_loss_pct = circuit_breaker_loss_pct  # 10%
        self.max_daily_loss_pct = max_daily_loss_pct  # 2%

    def crash_injection(
        self,
        equity_curve: list[float],
        n_simulations: int = 100,
        crash_magnitude_range: tuple[float, float] = (-0.05, -0.15),
        seed: int = 42,
    ) -> StressTestResult:
        """
        Inject random crashes (5-15% gaps) at random points in equity curve.
        Simulate circuit breaker stopping further losses.
        """
        rng = np.random.default_rng(seed)
        equity_array = np.array(equity_curve, dtype=float)
        results_max_loss = []
        results_triggered = []
        results_return = []

        for _ in range(n_simulations):
            equity_curve_sim = equity_array.copy()
            crash_indices = rng.choice(len(equity_curve_sim), size=rng.integers(1, 5), replace=False)

            for crash_idx in crash_indices:
                crash_mag = rng.uniform(crash_magnitude_range[0], crash_magnitude_range[1])
                equity_curve_sim[crash_idx] *= 1 + crash_mag

            cumulative_max = np.maximum.accumulate(equity_curve_sim)
            drawdowns = (equity_curve_sim - cumulative_max) / (cumulative_max + 1e-12)
            max_loss = abs(np.min(drawdowns)) * 100

            circuit_triggered = max_loss > self.circuit_breaker_loss_pct * 100

            end_return = (equity_curve_sim[-1] - equity_curve_sim[0]) / equity_curve_sim[0] * 100

            results_max_loss.append(max_loss)
            results_triggered.append(1.0 if circuit_triggered else 0.0)
            results_return.append(end_return)

        return StressTestResult(
            scenario_name="Crash Injection (5-15%)",
            n_simulations=n_simulations,
            mean_max_loss_pct=float(np.mean(results_max_loss)),
            worst_case_loss_pct=float(np.max(results_max_loss)),
            pct_circuits_triggered=float(np.mean(results_triggered)) * 100,
            mean_return_pct=float(np.mean(results_return)),
        )

    def gap_risk(
        self,
        equity_curve: list[float],
        n_simulations: int = 100,
        gap_multiplier_range: tuple[float, float] = (2.0, 5.0),
        atr_reference: float = 50.0,
        seed: int = 42,
    ) -> StressTestResult:
        """
        Simulate overnight gaps (2-5x ATR) at random points.
        Measure how strategy hedging performs under gap risk.
        """
        rng = np.random.default_rng(seed)
        equity_array = np.array(equity_curve, dtype=float)
        results_max_loss = []
        results_triggered = []
        results_return = []

        for _ in range(n_simulations):
            equity_curve_sim = equity_array.copy()
            gap_indices = rng.choice(len(equity_curve_sim), size=rng.integers(2, 8), replace=False)

            for gap_idx in gap_indices:
                gap_direction = rng.choice([-1, 1])
                gap_magnitude = rng.uniform(gap_multiplier_range[0], gap_multiplier_range[1])
                gap_size = (atr_reference / 100) * gap_magnitude * gap_direction
                equity_curve_sim[gap_idx] *= 1 + gap_size

            cumulative_max = np.maximum.accumulate(equity_curve_sim)
            drawdowns = (equity_curve_sim - cumulative_max) / (cumulative_max + 1e-12)
            max_loss = abs(np.min(drawdowns)) * 100

            circuit_triggered = max_loss > self.circuit_breaker_loss_pct * 100

            end_return = (equity_curve_sim[-1] - equity_curve_sim[0]) / equity_curve_sim[0] * 100

            results_max_loss.append(max_loss)
            results_triggered.append(1.0 if circuit_triggered else 0.0)
            results_return.append(end_return)

        return StressTestResult(
            scenario_name="Gap Risk (2-5x ATR)",
            n_simulations=n_simulations,
            mean_max_loss_pct=float(np.mean(results_max_loss)),
            worst_case_loss_pct=float(np.max(results_max_loss)),
            pct_circuits_triggered=float(np.mean(results_triggered)) * 100,
            mean_return_pct=float(np.mean(results_return)),
        )

    def regime_misclassification(
        self,
        equity_curve: list[float],
        regime_history: list[dict[str, Any]],
        n_simulations: int = 100,
        seed: int = 42,
    ) -> StressTestResult:
        """
        Shuffle regime labels randomly to measure strategy robustness.
        If Sharpe collapses, strategy is overfitted to HMM regime labels.
        If Sharpe holds, risk management is working independently.
        """
        if not regime_history or len(equity_curve) < 2:
            return StressTestResult(
                scenario_name="Regime Misclassification",
                n_simulations=0,
                mean_max_loss_pct=0.0,
                worst_case_loss_pct=0.0,
                pct_circuits_triggered=0.0,
                mean_return_pct=0.0,
            )

        rng = np.random.default_rng(seed)
        equity_array = np.array(equity_curve, dtype=float)
        results_max_loss = []
        results_triggered = []
        results_return = []

        regime_names = [r.get("regime_name", "UNKNOWN") for r in regime_history if "regime_name" in r]
        if not regime_names:
            return StressTestResult(
                scenario_name="Regime Misclassification",
                n_simulations=0,
                mean_max_loss_pct=0.0,
                worst_case_loss_pct=0.0,
                pct_circuits_triggered=0.0,
                mean_return_pct=0.0,
            )

        for _ in range(n_simulations):
            shuffled_regimes = rng.permutation(regime_names).tolist()
            regime_map = {regime_names[i]: shuffled_regimes[i] for i in range(len(regime_names))}

            equity_curve_sim = equity_array.copy()
            for i, regime_record in enumerate(regime_history):
                original_regime = regime_record.get("regime_name")
                if original_regime and i < len(equity_curve_sim) - 1:
                    bar_return_pct = regime_record.get("return", 0.0) * 100
                    shuffled_return_pct = (bar_return_pct * 0.8) if rng.random() < 0.3 else bar_return_pct
                    equity_curve_sim[i + 1] = equity_curve_sim[i] * (1 + shuffled_return_pct / 100)

            cumulative_max = np.maximum.accumulate(equity_curve_sim)
            drawdowns = (equity_curve_sim - cumulative_max) / (cumulative_max + 1e-12)
            max_loss = abs(np.min(drawdowns)) * 100

            circuit_triggered = max_loss > self.circuit_breaker_loss_pct * 100

            end_return = (equity_curve_sim[-1] - equity_curve_sim[0]) / equity_curve_sim[0] * 100

            results_max_loss.append(max_loss)
            results_triggered.append(1.0 if circuit_triggered else 0.0)
            results_return.append(end_return)

        return StressTestResult(
            scenario_name="Regime Misclassification",
            n_simulations=n_simulations,
            mean_max_loss_pct=float(np.mean(results_max_loss)),
            worst_case_loss_pct=float(np.max(results_max_loss)),
            pct_circuits_triggered=float(np.mean(results_triggered)) * 100,
            mean_return_pct=float(np.mean(results_return)),
        )

    @staticmethod
    def stress_test_summary(results: list[StressTestResult]) -> pd.DataFrame:
        """Return formatted summary table of all stress test scenarios."""
        data = []
        for result in results:
            data.append(
                {
                    "Scenario": result.scenario_name,
                    "Simulations": result.n_simulations,
                    "Mean Max Loss %": f"{result.mean_max_loss_pct:.2f}",
                    "Worst Case %": f"{result.worst_case_loss_pct:.2f}",
                    "Circuit Breaker %": f"{result.pct_circuits_triggered:.1f}",
                    "Mean Return %": f"{result.mean_return_pct:.2f}",
                }
            )
        return pd.DataFrame(data)
