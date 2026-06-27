from __future__ import annotations

import numpy as np
import pandas as pd

from arhmm_explorer import ARHMM, RunConfig, prepare_observations


def test_prepare_observations_smoke() -> None:
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    prices = 100 * np.exp(np.cumsum(np.full(len(dates), 0.001)))
    prepared = prepare_observations(pd.DataFrame({"date": dates, "price": prices}), RunConfig(initial_train_years=1))
    assert {"return", "ewma_volatility", "log_ewma_volatility"}.issubset(prepared.columns)
    assert np.isfinite(prepared["log_ewma_volatility"]).all()


def test_arhmm_fit_smoke() -> None:
    rng = np.random.default_rng(123)
    y = np.zeros((180, 2))
    for t in range(1, len(y)):
        y[t, 0] = (0.001 if t < 90 else -0.001) + 0.15 * y[t - 1, 0] + rng.normal(0, 0.01)
        y[t, 1] = -4.5 + 0.2 * y[t - 1, 1] + rng.normal(0, 0.05)
    model = ARHMM(RunConfig(max_iter=5, n_initializations=1, initial_train_years=1)).fit(y)
    probs = model.filtered_probabilities(y)
    means, covs = model.predict_next_distribution(y[-1])
    assert probs.shape == (179, 3)
    assert means.shape == (3, 2)
    assert covs.shape == (3, 2, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
