"""Hidden Markov Model engine for market regime detection.

Inference is performed via a forward-only filter
(``P(state_t | obs_1..t)``) to prevent look-ahead bias in live predictions.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

LOGGER = logging.getLogger(__name__)
EPSILON = 1e-12

LABELS_BY_COMPONENTS: dict[int, list[str]] = {
    2: ["CHOP", "TRENDING"],
    3: ["BEAR", "NEUTRAL", "BULL"],
    4: ["CRASH", "BEAR", "NEUTRAL", "BULL"],
    5: ["CRASH", "BEAR", "NEUTRAL", "BULL", "EUPHORIA"],
    6: ["CRASH", "STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL"],
    7: [
        "CRASH",
        "STRONG_BEAR",
        "WEAK_BEAR",
        "NEUTRAL",
        "WEAK_BULL",
        "STRONG_BULL",
        "EUPHORIA",
    ],
}


@dataclass
class HMMConfig:
    """Configuration for HMM model selection and filtering."""

    n_components: list[int]
    cv_tol: float = 1e-3
    cv_max_iter: int = 200
    train_bars: int = 504
    stability_bars: int = 5
    flicker_window: int = 40
    flicker_threshold: int = 2
    min_confidence: float = 0.95
    retrain_interval_bars: int = 21
    min_train_bars: int = 504
    random_state: int = 42


@dataclass
class RegimeInfo:
    """Human-readable metadata for a latent HMM state."""

    label: str
    state_id: int
    expected_return: float
    expected_volatility: float
    max_position_size: float
    max_leverage: float
    min_confidence_to_act: float


@dataclass
class RegimeState:
    """Current filtered regime state and confidence diagnostics."""

    current_regime_id: int
    state_probability: float
    state_probabilities: np.ndarray
    is_stable: bool
    is_flickering: bool
    days_in_regime: int
    regime_label: str = "UNKNOWN"
    size_multiplier: float = 1.0


class HMMEngine:
    """Selects and runs a GaussianMixture-based regime model.

    Training uses BIC-based model selection across candidate component counts.
    Live inference uses a **causal forward filter** so predictions never
    depend on future observations (no look-ahead bias).
    """

    def __init__(self, config: HMMConfig | None = None) -> None:
        if config is None:
            config = HMMConfig(n_components=[2, 3, 4])
        self.config = config

        self.model: GaussianMixture | None = None
        self.best_bic: float | None = None
        self.best_log_likelihood: float | None = None
        self.bic_scores: dict[int, float] = {}
        self.regime_info: dict[int, RegimeInfo] = {}
        self.state_label_by_id: dict[int, str] = {}
        self.training_data_labels: list[str] = []
        self.feature_columns: list[str] = []
        self.n_iter_: int | None = None
        self.converged_: bool | None = None
        self.last_retrain_index: int | None = None

        # Forward-filter cache for incremental updates
        self._cached_values: np.ndarray | None = None
        self._cached_index: pd.Index | None = None
        self._cached_alpha_logs: list[np.ndarray] = []

    # ==================================================================
    # Training
    # ==================================================================

    def fit(self, features: pd.DataFrame, returns: pd.Series | None = None) -> None:
        """Fit GaussianMixture candidates and select the best via BIC."""
        if features.empty:
            raise ValueError("Features are empty.")
        if len(features) < self.config.min_train_bars:
            raise ValueError(
                f"Need >={self.config.min_train_bars} bars, got {len(features)}."
            )

        train_features = features.tail(self.config.train_bars).copy()
        X = train_features.to_numpy(dtype=np.float64)
        self.feature_columns = train_features.columns.tolist()

        train_returns = (
            returns.reindex(train_features.index).astype(float)
            if returns is not None
            else train_features.iloc[:, 0].astype(float)
        )

        best_model: GaussianMixture | None = None
        best_bic = np.inf
        best_ll = -np.inf
        self.bic_scores = {}

        for n_states in self.config.n_components:
            try:
                model = GaussianMixture(
                    n_components=n_states,
                    covariance_type="full",
                    max_iter=self.config.cv_max_iter,
                    tol=self.config.cv_tol,
                    random_state=self.config.random_state,
                    n_init=10,
                )
                model.fit(X)
                ll = float(model.score(X))
                bic = float(model.bic(X))

                self.bic_scores[n_states] = bic
                LOGGER.info(
                    "HMM n=%d  LL=%.4f  BIC=%.4f  converged=%s  iters=%s",
                    n_states,
                    ll,
                    bic,
                    model.converged_,
                    model.n_iter_,
                )

                if bic < best_bic:
                    best_bic = bic
                    best_ll = ll
                    best_model = model
            except Exception as exc:
                LOGGER.warning("HMM n=%d training failed: %s", n_states, exc)
                continue

        if best_model is None:
            raise RuntimeError("No HMM candidate model was successfully trained.")

        self.model = best_model
        self._set_synthetic_transition_priors()
        self.best_bic = best_bic
        self.best_log_likelihood = best_ll
        self.n_iter_ = int(best_model.n_iter_)
        self.converged_ = bool(best_model.converged_)
        self.last_retrain_index = len(features) - 1

        LOGGER.info(
            "Selected HMM: n_components=%d  BIC=%.4f  converged=%s",
            self.model.n_components,
            self.best_bic,
            self.converged_,
        )

        self._build_regime_metadata(train_features, train_returns)
        self._reset_filter_cache()

    def maybe_retrain(
        self, features: pd.DataFrame, returns: pd.Series | None = None
    ) -> bool:
        """Retrain if enough new bars have arrived since the last fit."""
        if self.model is None or self.last_retrain_index is None:
            self.fit(features, returns=returns)
            return True

        bars_since = (len(features) - 1) - self.last_retrain_index
        if bars_since >= self.config.retrain_interval_bars:
            self.fit(features, returns=returns)
            return True
        return False

    # ==================================================================
    # Prediction (forward-only filtered inference)
    # ==================================================================

    def predict_regime(self, latest_features: pd.DataFrame) -> tuple[int, float]:
        """Return inferred regime index and posterior confidence."""
        state = self.predict_regime_filtered(latest_features)
        return state.current_regime_id, state.state_probability

    def predict_regime_filtered(self, features_up_to_now: pd.DataFrame) -> RegimeState:
        """Return the most-recent filtered regime state P(state|obs_1:t).

        This is the PRIMARY inference entry-point for live trading.
        It uses only past-and-current data — **no look-ahead bias**.
        """
        states = self.predict_regime_series_filtered(features_up_to_now)
        if not states:
            raise ValueError("No states available for empty features.")
        return states[-1]

    def predict_regime_path_filtered(
        self, features_up_to_now: pd.DataFrame
    ) -> list[int]:
        """Return full filtered regime path."""
        return [
            s.current_regime_id
            for s in self.predict_regime_series_filtered(features_up_to_now)
        ]

    def predict_regime_series_filtered(
        self, features_up_to_now: pd.DataFrame
    ) -> list[RegimeState]:
        """Return filtered regime state at every bar up to now."""
        if self.model is None:
            raise RuntimeError("Model is not trained.")
        if features_up_to_now.empty:
            return []

        X = features_up_to_now[self.feature_columns].to_numpy(dtype=np.float64)
        alpha_logs = self._forward_filter_log_probs(X, features_up_to_now.index)
        state_probs = [np.exp(a) for a in alpha_logs]
        filtered_path = [int(np.argmax(p)) for p in state_probs]

        stable_path, stable_flags, days_list, flicker_flags = (
            self._apply_stability_filter(filtered_path)
        )

        regime_states: list[RegimeState] = []
        for i, probs in enumerate(state_probs):
            sid = stable_path[i]
            conf = float(probs[sid])
            mult = 1.0
            if (
                not stable_flags[i]
                or flicker_flags[i]
                or conf < self.config.min_confidence
            ):
                mult = 0.75

            label = self.state_label_by_id.get(sid, f"REGIME_{sid}")

            regime_states.append(
                RegimeState(
                    current_regime_id=sid,
                    state_probability=conf,
                    state_probabilities=probs,
                    is_stable=stable_flags[i],
                    is_flickering=flicker_flags[i],
                    days_in_regime=days_list[i],
                    regime_label=label,
                    size_multiplier=mult,
                )
            )

        return regime_states

    def get_state_probabilities(self, latest_features: pd.DataFrame) -> np.ndarray:
        """Return posterior probabilities for all latent states."""
        return self.predict_regime_filtered(latest_features).state_probabilities

    # ==================================================================
    # Persistence
    # ==================================================================

    def save_model(self, file_path: str | Path) -> None:
        """Persist model and metadata bundle."""
        if self.model is None:
            raise RuntimeError("No trained model to save.")

        payload = {
            "model": self.model,
            "metadata": {
                "n_regimes": self.model.n_components,
                "bic": self.best_bic,
                "bic_scores": self.bic_scores,
                "convergence": self.converged_,
                "iterations": self.n_iter_,
                "training_data_labels": self.training_data_labels,
                "regime_info": {k: asdict(v) for k, v in self.regime_info.items()},
                "feature_columns": self.feature_columns,
            },
        }
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fp:
            pickle.dump(payload, fp)
        LOGGER.info("HMM model saved to %s", path)

    def load_model(self, file_path: str | Path) -> None:
        """Load model and metadata bundle."""
        with Path(file_path).open("rb") as fp:
            payload = pickle.load(fp)

        self.model = payload["model"]
        meta = payload.get("metadata", {})
        self.best_bic = meta.get("bic")
        self.bic_scores = meta.get("bic_scores", {})
        self.converged_ = meta.get("convergence")
        self.n_iter_ = meta.get("iterations")
        self.training_data_labels = meta.get("training_data_labels", [])
        self.feature_columns = meta.get("feature_columns", [])

        loaded = meta.get("regime_info", {})
        self.regime_info = {
            int(k): RegimeInfo(**v) for k, v in loaded.items()
        }
        self.state_label_by_id = {
            sid: info.label for sid, info in self.regime_info.items()
        }
        self._reset_filter_cache()
        LOGGER.info("HMM model loaded from %s", file_path)

    # Backward-compatible aliases
    save = save_model
    load = load_model

    def to_dict(self) -> dict[str, Any]:
        """Expose internal model metadata for logging."""
        return {
            "config": asdict(self.config),
            "selected_n_regimes": (
                int(self.model.n_components) if self.model else None
            ),
            "bic_scores": self.bic_scores,
            "selected_bic": self.best_bic,
            "log_likelihood": self.best_log_likelihood,
            "converged": self.converged_,
            "iterations": self.n_iter_,
            "regime_info": {
                sid: asdict(info) for sid, info in self.regime_info.items()
            },
        }

    # ==================================================================
    # Internals
    # ==================================================================

    def _build_regime_metadata(
        self, train_features: pd.DataFrame, train_returns: pd.Series
    ) -> None:
        """Create human-readable regime labels sorted by expected return."""
        if self.model is None:
            raise RuntimeError("Model is not trained.")

        state_path = self.predict_regime_path_filtered(train_features)
        df_states = pd.DataFrame(
            {"state": state_path, "ret": train_returns.values},
            index=train_features.index,
        )
        grouped = df_states.groupby("state")
        mean_ret = grouped["ret"].mean().to_dict()
        vol = grouped["ret"].std(ddof=0).fillna(0.0).to_dict()

        ordered = sorted(mean_ret.keys(), key=lambda s: mean_ret[s])
        labels = LABELS_BY_COMPONENTS.get(
            self.model.n_components,
            [f"REGIME_{i}" for i in range(self.model.n_components)],
        )

        self.regime_info = {}
        self.state_label_by_id = {}
        for rank, sid in enumerate(ordered):
            label = labels[rank] if rank < len(labels) else f"REGIME_{rank}"
            info = RegimeInfo(
                label=label,
                state_id=sid,
                expected_return=float(mean_ret[sid]),
                expected_volatility=float(vol.get(sid, 0.0)),
                max_position_size=1.0,
                max_leverage=1.0,
                min_confidence_to_act=self.config.min_confidence,
            )
            self.regime_info[sid] = info
            self.state_label_by_id[sid] = label

        self.training_data_labels = [
            self.state_label_by_id[s] for s in state_path
        ]

    @staticmethod
    def _compute_bic(
        model: GaussianMixture,
        log_likelihood: float,
        n_samples: int,
        n_features: int,
    ) -> float:
        """Compute Bayesian Information Criterion for GaussianMixture."""
        k = model.n_components
        n_start = k - 1
        n_means = k * n_features
        n_covs = k * (n_features * (n_features + 1) // 2)
        n_weights = k - 1
        n_params = n_start + n_weights + n_means + n_covs
        return float(-2.0 * log_likelihood * n_samples + n_params * np.log(max(n_samples, 1)))

    def _forward_filter_log_probs(
        self, X: np.ndarray, index: pd.Index
    ) -> list[np.ndarray]:
        """Run forward algorithm in log-space for filtered inference.

        NO look-ahead: α_t depends only on observations 1…t.
        Supports incremental cache when a single new bar is appended.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained.")

        n_obs = X.shape[0]
        if n_obs == 0:
            return []

        n_states = self.model.n_components

        # Check if we can reuse cache (append-only scenario)
        can_append = (
            self._cached_values is not None
            and self._cached_index is not None
            and len(index) == len(self._cached_index) + 1
            and index[:-1].equals(self._cached_index)
            and np.allclose(X[:-1], self._cached_values, atol=1e-12, rtol=1e-8)
        )

        if can_append:
            start_t = n_obs - 1
            alpha_logs = list(self._cached_alpha_logs)
        else:
            start_t = 0
            alpha_logs = []

        log_start = np.log(self.model.startprob_ + EPSILON)
        log_trans = np.log(self.model.transmat_ + EPSILON)

        for t in range(start_t, n_obs):
            emission_log = self._emission_log_prob(X[t])

            if t == 0:
                alpha_t = log_start + emission_log
            else:
                prev = alpha_logs[-1]
                predicted = logsumexp(
                    prev[:, np.newaxis] + log_trans, axis=0
                )
                alpha_t = predicted + emission_log

            # Normalize in log-space
            alpha_t -= logsumexp(alpha_t)
            alpha_logs.append(alpha_t)

        # Update cache
        self._cached_values = X.copy()
        self._cached_index = index.copy()
        self._cached_alpha_logs = alpha_logs
        return alpha_logs

    def _emission_log_prob(self, obs: np.ndarray) -> np.ndarray:
        """Compute log p(obs_t | state_k) for each state."""
        if self.model is None:
            raise RuntimeError("Model is not trained.")

        log_probs = np.empty(self.model.n_components, dtype=np.float64)
        for k in range(self.model.n_components):
            log_probs[k] = multivariate_normal.logpdf(
                obs,
                mean=self.model.means_[k],
                cov=self.model.covariances_[k],
                allow_singular=True,
            )
        return log_probs

    def _set_synthetic_transition_priors(self) -> None:
        """Attach persistence priors so filtered inference can run on GMM outputs."""
        if self.model is None:
            raise RuntimeError("Model is not trained.")

        n_states = self.model.n_components
        persistence = 0.97 if n_states > 1 else 1.0
        off_diag = (1.0 - persistence) / max(n_states - 1, 1)

        self.model.startprob_ = np.full(n_states, 1.0 / n_states, dtype=np.float64)
        self.model.transmat_ = np.full((n_states, n_states), off_diag, dtype=np.float64)
        np.fill_diagonal(self.model.transmat_, persistence)

    def _apply_stability_filter(
        self, filtered_path: list[int]
    ) -> tuple[list[int], list[bool], list[int], list[bool]]:
        """TRUE BLOCKING stability filter.

        The stable regime does NOT change until a *different* regime has been
        the MAP estimate for ``stability_bars`` **consecutive** bars.  Any
        interruption (even a single bar reverting) resets the counter.

        This prevents the rapid 0<->6, 2->3->0->2 oscillations observed in
        production logs.
        """
        if not filtered_path:
            return [], [], [], []

        n = len(filtered_path)
        required = self.config.stability_bars

        # ── initialise with first bar ─────────────────────────
        locked_regime = filtered_path[0]   # currently locked-in regime
        candidate: int | None = None       # candidate trying to replace it
        consec_count = 0                   # consecutive bars candidate held
        days_in_regime = 1

        stable_path: list[int] = [locked_regime]
        stable_flags: list[bool] = [True]
        days_list: list[int] = [1]
        flicker_flags: list[bool] = [False]

        for t in range(1, n):
            raw = filtered_path[t]

            if raw == locked_regime:
                # Raw agrees with locked regime -> reset any pending candidate
                candidate = None
                consec_count = 0
                days_in_regime += 1
                stable_path.append(locked_regime)
                stable_flags.append(True)

            else:
                # Raw disagrees with locked regime
                if raw == candidate:
                    consec_count += 1
                else:
                    # New candidate, restart counter
                    candidate = raw
                    consec_count = 1

                if consec_count >= required:
                    # ── TRANSITION ACCEPTED ────────────────────
                    LOGGER.info(
                        "Regime change confirmed: %d -> %d  (after %d consecutive bars)",
                        locked_regime,
                        candidate,
                        consec_count,
                    )
                    locked_regime = candidate
                    candidate = None
                    consec_count = 0
                    days_in_regime = 1
                    stable_path.append(locked_regime)
                    stable_flags.append(True)
                else:
                    # ── BLOCKED: keep old regime ──────────────
                    days_in_regime += 1
                    stable_path.append(locked_regime)
                    stable_flags.append(False)  # unstable (transition pending)

            days_list.append(days_in_regime)

            # ── Flicker detection ─────────────────────────────
            win_start = max(0, len(stable_path) - self.config.flicker_window)
            window = stable_path[win_start:]
            changes = sum(
                1 for i in range(1, len(window)) if window[i] != window[i - 1]
            )
            flicker_flags.append(changes > self.config.flicker_threshold)

        return stable_path, stable_flags, days_list, flicker_flags

    def _reset_filter_cache(self) -> None:
        self._cached_values = None
        self._cached_index = None
        self._cached_alpha_logs = []
