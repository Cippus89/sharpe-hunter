# Sharpe Hunter

Sharpe Hunter is a lightweight research harness for cross-sectional valuation and factor testing.

## Quick start
1. Install dependencies: `pip install -r requirements.txt`.
2. Update `config/config.yml` with the Finviz snapshot you want to analyse.
3. Run the pipeline: `python -m src.main`.

The script will:
- load and clean the Finviz snapshot;
- download or reuse cached price histories from Yahoo Finance;
- run an OLS regression to estimate fair values and mispricing;
- export CSV and text summaries under `data/results/`;
- print a compact ranking and run a naive long/short backtest.
