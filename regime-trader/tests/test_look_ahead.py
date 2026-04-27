"""Tests to ensure no look-ahead bias in data and signal pipeline."""

import numpy as np
import pandas as pd

from core.hmm_engine import HMMConfig, HMMEngine


def _synthetic_features(n: int = 720, seed: int = 7) -> pd.DataFrame:
    """Create deterministic synthetic features with changing volatility regimes."""
    rng = np.random.default_rng(seed)

    low_vol = rng.normal(loc=0.0010, scale=0.0050, size=n // 3)
    mid_vol = rng.normal(loc=0.0004, scale=0.0100, size=n // 3)
    high_vol = rng.normal(loc=-0.0008, scale=0.0200, size=n - 2 * (n // 3))
    ret_1 = np.concatenate([low_vol, mid_vol, high_vol])

    ret_5 = pd.Series(ret_1).rolling(5, min_periods=1).sum().to_numpy()
    realized_vol_20 = pd.Series(ret_1).rolling(20, min_periods=1).std(ddof=0).fillna(0.0).to_numpy()
    momentum = pd.Series(ret_1).rolling(20, min_periods=1).mean().fillna(0.0).to_numpy()

    return pd.DataFrame(
        {
            "ret_1": ret_1,
            "ret_5": ret_5,
            "realized_vol_20": realized_vol_20,
            "momentum_20": momentum,
        }
    )


def test_no_look_ahead_bias() -> None:
    """Regime at T must be identical with data[0:T] vs data[0:T+100]."""
    features = _synthetic_features()

    config = HMMConfig(
        n_components=[3],
        cv_tol=1e-3,
        cv_max_iter=100,
        train_bars=504,
        stability_bars=3,
        flicker_window=20,
        flicker_threshold=4,
        min_confidence=0.9,
        min_train_bars=504,
        random_state=42,
    )

    engine = HMMEngine(config=config)
    engine.fit(features)

    t = 520
    regime_t = engine.predict_regime_path_filtered(features.iloc[:t])
    regime_long = engine.predict_regime_path_filtered(features.iloc[: t + 100])

    assert regime_t[-1] == regime_long[t - 1], "LOOK-AHEAD BIAS DETECTED"
