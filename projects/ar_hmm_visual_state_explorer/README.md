# AR-HMM Visual State Explorer

This is an MVP for a personal research app that shows how a 3-state **Autoregressive Hidden Markov Model** would have evolved through time using only data available up to the previous trading day.

It intentionally does **not** implement trading rules, portfolio exposure, tax logic, costs, or performance backtesting. The point is to validate whether the hidden states are stable, interpretable, and diagnosable before building a strategy on top.

## Model

Observation vector:

```text
y_t = [return_t, log_ewma_volatility_t]
```

State-dependent autoregressive emission:

```text
y_t | y_{t-1}, z_t = k ~ Normal(c_k + B_k y_{t-1}, Sigma_k)
```

Transition model:

```text
P(z_t | z_{t-1}) = A
```

Walk-forward protocol:

```text
initial training window = 10 years
each day T:
    train on data up to T-1
    refit AR-HMM with expanding window
    save predictive return distributions, state probabilities, transition matrix, durations, diagnostics
```

## Install

```bash
cd projects/ar_hmm_visual_state_explorer
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .[dev]
```

## Run offline with Yahoo Finance

```bash
python run_walk_forward.py \
  --ticker SPY \
  --start 2000-01-01 \
  --end 2026-06-27 \
  --output data/outputs/spy_run \
  --config configs/default.yaml
```

Yahoo examples:

```text
SPY       US ETF example
QQQ       Nasdaq-100 ETF example
SWDA.MI   iShares Core MSCI World UCITS ETF on Borsa Italiana
IMIE.MI   SPDR MSCI ACWI IMI UCITS ETF on Borsa Italiana
```

## Run offline with CSV

```bash
python run_walk_forward.py \
  --input data/raw/my_prices.csv \
  --output data/outputs/example_run \
  --config configs/default.yaml
```

Expected CSV columns:

```text
Date, Open, High, Low, Close, Adj Close, Volume
```

The default price column is `Adj Close`.

## Run dashboard

```bash
streamlit run app.py
```

The dashboard supports two data sources:

```text
Yahoo Finance ticker download
CSV upload
```

## Outputs

The runner saves both Parquet and CSV versions of:

```text
daily_state_distributions
daily_transition_matrices
daily_state_probabilities
model_diagnostics
audit_table
```

The app exposes raw states (`state_0`, `state_1`, `state_2`) and aligned states (`low`, `mid`, `high`) to reduce label-switching confusion.

## Validation gates before adding trading

Do not add trading rules until the diagnostics answer these questions:

```text
Are the three regimes statistically separable?
Are probabilities stable or noisy?
Does the transition matrix behave plausibly through stress periods?
Do the predictive return distributions change coherently?
Did the EM optimizer converge reliably?
```
