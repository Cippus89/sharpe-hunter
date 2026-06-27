# Yahoo Finance integration

This change adds Yahoo Finance as an alternative data source for the AR-HMM Visual State Explorer.

## Added

- `yfinance>=0.2` dependency
- `load_yfinance_price_history()` network loader
- `yfinance_to_price_df()` normalization helper for flat and MultiIndex yfinance outputs
- CLI support for `--ticker`, `--start`, and optional `--end`
- Streamlit data-source selector: `Yahoo Finance` or `CSV upload`
- Cached Yahoo download in the dashboard
- Stale-output invalidation when ticker/date/config changes
- Smoke tests for yfinance normalization
