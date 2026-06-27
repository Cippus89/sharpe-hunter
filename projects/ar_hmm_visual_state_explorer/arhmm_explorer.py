from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class RunConfig:
    date_column: str = "Date"
    price_column: str = "Adj Close"
    ewma_lambda: float = 0.94
    initial_train_years: int = 10
    n_states: int = 3
    max_iter: int = 100
    tol: float = 1e-4
    n_initializations: int = 5
    random_seed: int = 42
    covariance_floor: float = 1e-6
    transition_floor: float = 1e-6
    min_state_weight: float = 10.0
    alignment_alpha: float = 1.0
    max_refits: int | None = None

    def validate(self) -> None:
        if self.n_states != 3:
            raise ValueError("This MVP is intentionally fixed to 3 hidden states.")
        if not 0.0 < self.ewma_lambda < 1.0:
            raise ValueError("ewma_lambda must be between 0 and 1.")
        if self.initial_train_years < 1:
            raise ValueError("initial_train_years must be >= 1.")
        if self.n_initializations < 1:
            raise ValueError("n_initializations must be >= 1.")


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> RunConfig:
    data = {}
    if path:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML config files.")
        with Path(path).open("r", encoding="utf-8") as fh:
            data.update(yaml.safe_load(fh) or {})
    if overrides:
        data.update({k: v for k, v in overrides.items() if v is not None})
    cfg = RunConfig(**data)
    cfg.validate()
    return cfg


def load_price_csv(path_or_buffer, config: RunConfig) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    if config.date_column not in df.columns:
        raise ValueError(f"Missing date column: {config.date_column}")
    price_col = config.price_column if config.price_column in df.columns else None
    if price_col is None:
        for candidate in ("Adj Close", "Close", "close", "adj_close"):
            if candidate in df.columns:
                price_col = candidate
                break
    if price_col is None:
        raise ValueError(f"Missing price column: {config.price_column}")
    out = df.rename(columns={config.date_column: "date", price_col: "price"})[["date", "price"]]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["price"] = pd.to_numeric(out["price"], errors="coerce")
    out = out.dropna().sort_values("date").drop_duplicates("date", keep="last")
    out = out.loc[out["price"] > 0].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid positive prices after cleaning.")
    return out


def prepare_observations(price_df: pd.DataFrame, config: RunConfig) -> pd.DataFrame:
    df = price_df.copy()
    df["return"] = np.log(df["price"] / df["price"].shift(1))
    df["ewma_volatility"] = np.sqrt(_ewma_variance(df["return"].to_numpy(float), config.ewma_lambda))
    df["log_ewma_volatility"] = np.log(np.maximum(df["ewma_volatility"], np.finfo(float).tiny))
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    if len(df) < 252:
        raise ValueError("Too few observations after preprocessing.")
    return df


def data_quality_report(price_df: pd.DataFrame, prepared_df: pd.DataFrame) -> dict:
    return {
        "start_date": str(pd.Timestamp(price_df["date"].min()).date()),
        "end_date": str(pd.Timestamp(price_df["date"].max()).date()),
        "raw_rows": int(len(price_df)),
        "prepared_rows": int(len(prepared_df)),
        "min_price": float(price_df["price"].min()),
        "max_price": float(price_df["price"].max()),
        "min_return": float(prepared_df["return"].min()),
        "max_return": float(prepared_df["return"].max()),
        "return_outliers_abs_gt_10pct": int((prepared_df["return"].abs() > 0.10).sum()),
    }


def observation_matrix(prepared_df: pd.DataFrame) -> np.ndarray:
    return prepared_df[["return", "log_ewma_volatility"]].to_numpy(float)


@dataclass
class ARHMMParams:
    pi: np.ndarray
    transition: np.ndarray
    intercepts: np.ndarray
    ar_matrices: np.ndarray
    covariances: np.ndarray


@dataclass
class FitDiagnostics:
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    n_iterations: int
    best_initialization_id: int
    fit_time_seconds: float


class ARHMM:
    """State-specific AR(1) Gaussian HMM fitted by EM.

    y_t | y_{t-1}, z_t=k ~ Normal(c_k + B_k y_{t-1}, Sigma_k)
    """

    def __init__(self, config: RunConfig):
        self.config = config
        self.params_: ARHMMParams | None = None
        self.diagnostics_: FitDiagnostics | None = None

    def fit(self, y: np.ndarray) -> "ARHMM":
        y = _as_2d(y)
        start = perf_counter()
        best = None
        for init_id in range(self.config.n_initializations):
            params = self._initial_params(y, self.config.random_seed + init_id)
            prev_ll = -np.inf
            converged = False
            iteration = 0
            for iteration in range(1, self.config.max_iter + 1):
                gamma, xi, ll = self._e_step(y, params)
                params = self._m_step(y, gamma, xi, params)
                if np.isfinite(prev_ll) and abs(ll - prev_ll) < self.config.tol:
                    converged = True
                    break
                prev_ll = ll
            _, _, final_ll = self._e_step(y, params)
            if best is None or final_ll > best[0]:
                best = (final_ll, params, init_id, converged, iteration)
        assert best is not None
        ll, params, init_id, converged, iteration = best
        n_params = self._parameter_count(y.shape[1])
        n_samples = max(len(y) - 1, 1)
        self.params_ = params
        self.diagnostics_ = FitDiagnostics(
            log_likelihood=float(ll),
            aic=float(2 * n_params - 2 * ll),
            bic=float(n_params * np.log(n_samples) - 2 * ll),
            converged=bool(converged),
            n_iterations=int(iteration),
            best_initialization_id=int(init_id),
            fit_time_seconds=float(perf_counter() - start),
        )
        return self

    def filtered_probabilities(self, y: np.ndarray) -> np.ndarray:
        if self.params_ is None:
            raise RuntimeError("Model is not fitted.")
        gamma, _, _ = self._e_step(_as_2d(y), self.params_)
        return gamma

    def predict_next_distribution(self, y_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.params_ is None:
            raise RuntimeError("Model is not fitted.")
        y_prev = np.asarray(y_prev, dtype=float).reshape(-1)
        means = self.params_.intercepts + np.einsum("kij,j->ki", self.params_.ar_matrices, y_prev)
        return means, self.params_.covariances.copy()

    def _initial_params(self, y: np.ndarray, seed: int) -> ARHMMParams:
        x_prev, y_curr = y[:-1], y[1:]
        features = np.column_stack([y_curr, x_prev])
        try:
            labels = KMeans(n_clusters=self.config.n_states, n_init=10, random_state=seed).fit_predict(features)
        except Exception:
            labels = np.random.default_rng(seed).integers(0, self.config.n_states, size=len(y_curr))
        gamma = np.full((len(y_curr), self.config.n_states), 1e-3)
        gamma[np.arange(len(y_curr)), labels] = 1.0
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = np.full((max(len(y_curr) - 1, 0), self.config.n_states, self.config.n_states), 1e-3)
        for t in range(len(y_curr) - 1):
            xi[t, labels[t], labels[t + 1]] += 1.0
        if len(xi):
            xi /= xi.sum(axis=(1, 2), keepdims=True)
        return self._m_step(y, gamma, xi, self._fallback_params(y))

    def _fallback_params(self, y: np.ndarray) -> ARHMMParams:
        k, d = self.config.n_states, y.shape[1]
        cov = _regularize_cov(np.cov(y.T), self.config.covariance_floor)
        return ARHMMParams(
            pi=np.full(k, 1 / k),
            transition=np.full((k, k), 1 / k),
            intercepts=np.zeros((k, d)),
            ar_matrices=np.tile(np.eye(d) * 0.1, (k, 1, 1)),
            covariances=np.tile(cov, (k, 1, 1)),
        )

    def _e_step(self, y: np.ndarray, params: ARHMMParams) -> tuple[np.ndarray, np.ndarray, float]:
        emission = self._emission_logprob(y, params)
        n, k = emission.shape
        log_pi = np.log(_normalize(params.pi, self.config.transition_floor))
        log_a = np.log(_normalize_rows(params.transition, self.config.transition_floor))
        alpha = np.empty((n, k))
        alpha[0] = log_pi + emission[0]
        for t in range(1, n):
            alpha[t] = emission[t] + logsumexp(alpha[t - 1][:, None] + log_a, axis=0)
        ll = float(logsumexp(alpha[-1]))
        beta = np.zeros((n, k))
        for t in range(n - 2, -1, -1):
            beta[t] = logsumexp(log_a + emission[t + 1][None, :] + beta[t + 1][None, :], axis=1)
        gamma = np.exp(alpha + beta - ll)
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = np.empty((max(n - 1, 0), k, k))
        for t in range(n - 1):
            x = alpha[t][:, None] + log_a + emission[t + 1][None, :] + beta[t + 1][None, :] - ll
            xi[t] = np.exp(x)
            xi[t] /= xi[t].sum()
        return gamma, xi, ll

    def _m_step(self, y: np.ndarray, gamma: np.ndarray, xi: np.ndarray, previous: ARHMMParams) -> ARHMMParams:
        x_prev, y_curr = y[:-1], y[1:]
        n, d = y_curr.shape
        k = self.config.n_states
        design = np.column_stack([np.ones(n), x_prev])
        pi = _normalize(gamma[0], self.config.transition_floor)
        transition = _normalize_rows(xi.sum(axis=0), self.config.transition_floor) if len(xi) else previous.transition
        intercepts = np.empty((k, d))
        ar_matrices = np.empty((k, d, d))
        covariances = np.empty((k, d, d))
        for state in range(k):
            weights = gamma[:, state]
            if weights.sum() < self.config.min_state_weight:
                intercepts[state] = previous.intercepts[state]
                ar_matrices[state] = previous.ar_matrices[state]
                covariances[state] = previous.covariances[state]
                continue
            beta = _weighted_lstsq(design, y_curr, weights, self.config.covariance_floor)
            intercepts[state] = beta[0]
            ar_matrices[state] = beta[1:].T
            resid = y_curr - design @ beta
            cov = (resid * weights[:, None]).T @ resid / max(weights.sum(), 1.0)
            covariances[state] = _regularize_cov(cov, self.config.covariance_floor)
        return ARHMMParams(pi, transition, intercepts, ar_matrices, covariances)

    def _emission_logprob(self, y: np.ndarray, params: ARHMMParams) -> np.ndarray:
        x_prev, y_curr = y[:-1], y[1:]
        n, d = y_curr.shape
        out = np.empty((n, self.config.n_states))
        for state in range(self.config.n_states):
            mean = params.intercepts[state] + x_prev @ params.ar_matrices[state].T
            cov = _regularize_cov(params.covariances[state], self.config.covariance_floor)
            sign, logdet = np.linalg.slogdet(cov)
            if sign <= 0:
                raise np.linalg.LinAlgError("Non positive-definite covariance.")
            inv = np.linalg.inv(cov)
            diff = y_curr - mean
            mahal = np.einsum("ij,jk,ik->i", diff, inv, diff)
            out[:, state] = -0.5 * (d * np.log(2 * np.pi) + logdet + mahal)
        return out

    def _parameter_count(self, d: int) -> int:
        k = self.config.n_states
        return int((k - 1) + k * (k - 1) + k * ((d + 1) * d + d * (d + 1) // 2))


def run_walk_forward(prepared_df: pd.DataFrame, config: RunConfig) -> dict[str, pd.DataFrame]:
    config.validate()
    y_all = observation_matrix(prepared_df)
    start_idx = _first_walk_forward_index(prepared_df, config.initial_train_years)
    dist_rows, trans_rows, prob_rows, diag_rows, audit_rows = [], [], [], [], []
    refits = 0
    for idx in range(start_idx, len(prepared_df)):
        if config.max_refits is not None and refits >= config.max_refits:
            break
        refits += 1
        date = prepared_df.loc[idx, "date"]
        train = prepared_df.iloc[:idx]
        scaler = StandardScaler().fit(y_all[:idx])
        y_train_scaled = scaler.transform(y_all[:idx])
        model = ARHMM(config).fit(y_train_scaled)
        assert model.params_ is not None and model.diagnostics_ is not None
        probs = model.filtered_probabilities(y_train_scaled)[-1]
        means_scaled, covs_scaled = model.predict_next_distribution(y_train_scaled[-1])
        means_real, covs_real = [], []
        for state in range(config.n_states):
            mean, cov = _inverse_mean_cov(means_scaled[state], covs_scaled[state], scaler)
            means_real.append(mean)
            covs_real.append(cov)
        means_real, covs_real = np.asarray(means_real), np.asarray(covs_real)
        ret_mu = means_real[:, 0]
        ret_std = np.sqrt(np.maximum(covs_real[:, 0, 0], 0.0))
        logvol_mu = means_real[:, 1]
        logvol_std = np.sqrt(np.maximum(covs_real[:, 1, 1], 0.0))
        vol_mean = np.exp(logvol_mu + 0.5 * logvol_std**2)
        order = _alignment_order(ret_mu, ret_std, config.alignment_alpha)
        aligned_transition = model.params_.transition[np.ix_(order, order)]
        dist = _distribution_row(date, ret_mu, ret_std, logvol_mu, logvol_std, vol_mean, order)
        trans = _transition_row(date, model.params_.transition, aligned_transition)
        prob = _probability_row(date, probs, order)
        diag = {
            "date": date,
            "training_start": train["date"].iloc[0],
            "training_end": train["date"].iloc[-1],
            "n_observations": int(len(train)),
            "n_initializations": int(config.n_initializations),
            "random_seed": int(config.random_seed),
            **asdict(model.diagnostics_),
        }
        dist_rows.append(dist)
        trans_rows.append(trans)
        prob_rows.append(prob)
        diag_rows.append(diag)
        audit_rows.append({**prob, **{k: v for k, v in dist.items() if k != "date"}, **{k: v for k, v in trans.items() if k != "date"}, **{k: v for k, v in diag.items() if k != "date"}})
    return {
        "distributions": pd.DataFrame(dist_rows),
        "transitions": pd.DataFrame(trans_rows),
        "probabilities": pd.DataFrame(prob_rows),
        "diagnostics": pd.DataFrame(diag_rows),
        "audit": pd.DataFrame(audit_rows),
    }


def save_outputs(outputs: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "distributions": "daily_state_distributions",
        "transitions": "daily_transition_matrices",
        "probabilities": "daily_state_probabilities",
        "diagnostics": "model_diagnostics",
        "audit": "audit_table",
    }
    for key, stem in filenames.items():
        outputs[key].to_parquet(output_dir / f"{stem}.parquet", index=False)
        outputs[key].to_csv(output_dir / f"{stem}.csv", index=False)
    (output_dir / "run_manifest.json").write_text(json.dumps({"files": filenames}, indent=2), encoding="utf-8")


def gaussian_density_series(mu: float, sigma: float, points: int = 400) -> tuple[np.ndarray, np.ndarray]:
    sigma = max(float(sigma), 1e-12)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, points)
    return x, norm.pdf(x, mu, sigma)


def _distribution_row(date, ret_mu, ret_std, logvol_mu, logvol_std, vol_mean, order):
    row = {"date": date}
    for s in range(3):
        row[f"raw_state_{s}_return_mean"] = float(ret_mu[s])
        row[f"raw_state_{s}_return_std"] = float(ret_std[s])
        row[f"raw_state_{s}_logvol_mean"] = float(logvol_mu[s])
        row[f"raw_state_{s}_logvol_std"] = float(logvol_std[s])
        row[f"raw_state_{s}_vol_mean_lognormal"] = float(vol_mean[s])
    for label, raw_idx in zip(("low", "mid", "high"), order, strict=True):
        row[f"aligned_{label}_return_mean"] = float(ret_mu[raw_idx])
        row[f"aligned_{label}_return_std"] = float(ret_std[raw_idx])
        row[f"aligned_{label}_logvol_mean"] = float(logvol_mu[raw_idx])
        row[f"aligned_{label}_logvol_std"] = float(logvol_std[raw_idx])
        row[f"aligned_{label}_vol_mean_lognormal"] = float(vol_mean[raw_idx])
    return row


def _transition_row(date, raw, aligned):
    row = {"date": date}
    for i in range(3):
        for j in range(3):
            row[f"raw_p_{i}{j}"] = float(raw[i, j])
        row[f"raw_duration_{i}"] = float(1 / (1 - min(raw[i, i], 0.999999)))
    labels = ("low", "mid", "high")
    for i, src in enumerate(labels):
        for j, dst in enumerate(labels):
            row[f"aligned_p_{src}_{dst}"] = float(aligned[i, j])
        row[f"aligned_duration_{src}"] = float(1 / (1 - min(aligned[i, i], 0.999999)))
    return row


def _probability_row(date, raw_probs, order):
    row = {"date": date}
    for s in range(3):
        row[f"raw_prob_state_{s}"] = float(raw_probs[s])
    row["raw_dominant_state"] = int(np.argmax(raw_probs))
    aligned_values = []
    for label, raw_idx in zip(("low", "mid", "high"), order, strict=True):
        value = float(raw_probs[raw_idx])
        row[f"aligned_prob_{label}"] = value
        aligned_values.append(value)
    row["aligned_dominant_state"] = ("low", "mid", "high")[int(np.argmax(aligned_values))]
    return row


def _first_walk_forward_index(df: pd.DataFrame, years: int) -> int:
    cutoff = pd.Timestamp(df["date"].iloc[0]) + pd.DateOffset(years=years)
    idxs = np.flatnonzero(pd.to_datetime(df["date"]) >= cutoff)
    if len(idxs) == 0:
        raise ValueError(f"Dataset shorter than initial training window of {years} years.")
    return int(idxs[0])


def _alignment_order(ret_mu, ret_std, alpha):
    score = ret_mu * 252 - alpha * ret_std * np.sqrt(252)
    return np.argsort(score)


def _inverse_mean_cov(mean_scaled, cov_scaled, scaler):
    scale = np.asarray(scaler.scale_, dtype=float)
    center = np.asarray(scaler.mean_, dtype=float)
    mean = mean_scaled * scale + center
    cov = np.diag(scale) @ cov_scaled @ np.diag(scale)
    return mean, cov


def _ewma_variance(returns: np.ndarray, lam: float) -> np.ndarray:
    out = np.full_like(returns, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(returns))
    if len(valid) == 0:
        return out
    first = int(valid[0])
    out[first] = returns[first] ** 2
    for i in range(first + 1, len(returns)):
        out[i] = lam * out[i - 1] + (1 - lam) * returns[i] ** 2 if np.isfinite(returns[i]) else out[i - 1]
    return out


def _as_2d(y):
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not np.all(np.isfinite(y)):
        raise ValueError("Expected finite 2D array.")
    return y


def _normalize(v, floor):
    v = np.asarray(v, dtype=float) + floor
    return v / v.sum()


def _normalize_rows(m, floor):
    m = np.asarray(m, dtype=float) + floor
    return m / m.sum(axis=1, keepdims=True)


def _weighted_lstsq(x, y, weights, ridge):
    sw = np.sqrt(np.maximum(weights, 0))[:, None]
    lhs = (x * sw).T @ (x * sw) + ridge * np.eye(x.shape[1])
    rhs = (x * sw).T @ (y * sw)
    return np.linalg.solve(lhs, rhs)


def _regularize_cov(cov, floor):
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    return cov + floor * np.eye(cov.shape[0])


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--max-refits", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config, {"max_refits": args.max_refits})
    price = load_price_csv(args.input, cfg)
    prepared = prepare_observations(price, cfg)
    print(json.dumps(data_quality_report(price, prepared), indent=2))
    outputs = run_walk_forward(prepared, cfg)
    save_outputs(outputs, args.output)
    print(f"Saved outputs to {Path(args.output).resolve()}")


if __name__ == "__main__":
    cli_main()
