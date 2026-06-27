from __future__ import annotations

from datetime import date
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from arhmm_explorer import (
    RunConfig,
    data_quality_report,
    gaussian_density_series,
    load_price_csv,
    load_yfinance_price_history,
    prepare_observations,
    run_walk_forward,
)


def _line(df: pd.DataFrame, x: str, y: str, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=y))
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y)
    return fig


def _multi_line(df: pd.DataFrame, cols: list[str], title: str) -> go.Figure:
    fig = go.Figure()
    for col in cols:
        fig.add_trace(go.Scatter(x=df["date"], y=df[col], mode="lines", name=col))
    fig.update_layout(title=title, xaxis_title="date")
    return fig


def _nearest_row(df: pd.DataFrame, selected_date):
    dates = pd.to_datetime(df["date"])
    idx = int(np.argmin(np.abs(dates - pd.Timestamp(selected_date))))
    return df.iloc[idx]


def _density_snapshot(df: pd.DataFrame, selected_date, aligned: bool) -> go.Figure:
    row = _nearest_row(df, selected_date)
    labels = ("low", "mid", "high") if aligned else ("0", "1", "2")
    fig = go.Figure()
    for label in labels:
        prefix = f"aligned_{label}" if aligned else f"raw_state_{label}"
        x, y = gaussian_density_series(row[f"{prefix}_return_mean"], row[f"{prefix}_return_std"])
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=str(label)))
    fig.update_layout(title=f"Predictive return distributions: {pd.Timestamp(row['date']).date()}", xaxis_title="daily return", yaxis_title="density")
    return fig


def _transition_heatmap(df: pd.DataFrame, selected_date, aligned: bool) -> go.Figure:
    row = _nearest_row(df, selected_date)
    labels = ("low", "mid", "high") if aligned else ("0", "1", "2")
    if aligned:
        z = np.array([[row[f"aligned_p_{i}_{j}"] for j in labels] for i in labels], dtype=float)
    else:
        z = np.array([[row[f"raw_p_{i}{j}"] for j in range(3)] for i in range(3)], dtype=float)
    fig = go.Figure(data=go.Heatmap(z=z, x=labels, y=labels, zmin=0, zmax=1, text=np.round(z, 3), texttemplate="%{text}"))
    fig.update_layout(title=f"Transition matrix: {pd.Timestamp(row['date']).date()}", xaxis_title="to", yaxis_title="from")
    return fig


@st.cache_data(show_spinner=False)
def _download_yahoo_cached(ticker: str, start: str, end: str | None, price_column: str) -> pd.DataFrame:
    cfg = RunConfig(price_column=price_column)
    return load_yfinance_price_history(ticker=ticker, start=start, end=end, config=cfg)


st.set_page_config(page_title="AR-HMM Visual State Explorer", layout="wide")
st.title("AR-HMM Visual State Explorer")
st.caption("3-state AR-HMM on daily return and log EWMA volatility. No trading rules.")

with st.sidebar:
    source = st.radio("Data source", ["Yahoo Finance", "CSV upload"], horizontal=True)
    if source == "Yahoo Finance":
        ticker = st.text_input("Yahoo ticker", "SPY").strip().upper()
        start_date = st.date_input("Start date", value=date(2000, 1, 1))
        end_date = st.date_input("End date", value=date.today())
    else:
        uploaded = st.file_uploader("Upload price CSV", type="csv")
        date_column = st.text_input("Date column", "Date")
    price_column = st.text_input("Price column", "Adj Close")
    ewma_lambda = st.slider("EWMA lambda", 0.80, 0.99, 0.94, 0.01)
    initial_train_years = st.number_input("Initial training years", 1, 30, 10)
    max_iter = st.number_input("EM max iterations", 10, 500, 100, step=10)
    n_initializations = st.number_input("Initializations", 1, 20, 3)
    max_refits = st.number_input("Max refits for trial; 0 = full", 0, 10000, 50)
    random_seed = st.number_input("Random seed", 0, 999999, 42)
    aligned = st.toggle("Use aligned low/mid/high states", True)
    run_button = st.button("Run walk-forward", type="primary")

cfg = RunConfig(
    date_column=locals().get("date_column", "Date"),
    price_column=price_column,
    ewma_lambda=float(ewma_lambda),
    initial_train_years=int(initial_train_years),
    max_iter=int(max_iter),
    n_initializations=int(n_initializations),
    random_seed=int(random_seed),
    max_refits=None if int(max_refits) == 0 else int(max_refits),
)

try:
    if source == "Yahoo Finance":
        if not ticker:
            st.info("Insert a Yahoo Finance ticker, for example SPY, QQQ, IWDA.AS, IMIE.MI, or SWDA.MI.")
            st.stop()
        if start_date >= end_date:
            st.error("Start date must be before end date.")
            st.stop()
        with st.spinner(f"Downloading {ticker} from Yahoo Finance..."):
            price_df = _download_yahoo_cached(ticker, str(start_date), str(end_date), price_column)
    else:
        if uploaded is None:
            st.info("Upload a CSV with Date and price columns, or switch to Yahoo Finance in the sidebar.")
            st.stop()
        price_df = load_price_csv(BytesIO(uploaded.getvalue()), cfg)
    prepared = prepare_observations(price_df, cfg)
except Exception as exc:
    st.error(f"Data error: {exc}")
    st.stop()

tabs = st.tabs([
    "Data Quality",
    "State Probabilities",
    "Return Distributions",
    "Transition Matrix",
    "State Duration",
    "Diagnostics",
    "Audit",
])

with tabs[0]:
    st.json(data_quality_report(price_df, prepared))
    st.plotly_chart(_line(prepared, "date", "price", "Price"), use_container_width=True)
    st.plotly_chart(_line(prepared, "date", "return", "Log returns"), use_container_width=True)
    st.plotly_chart(_line(prepared, "date", "ewma_volatility", "EWMA volatility"), use_container_width=True)
    st.plotly_chart(_line(prepared, "date", "log_ewma_volatility", "Log EWMA volatility"), use_container_width=True)

if run_button:
    with st.spinner("Fitting daily expanding-window AR-HMM. This can be slow."):
        st.session_state.outputs = run_walk_forward(prepared, cfg)

outputs = st.session_state.get("outputs")
if outputs is None:
    for tab in tabs[1:]:
        with tab:
            st.info("Run the walk-forward from the sidebar to populate this section.")
    st.stop()

prob = outputs["probabilities"]
dist = outputs["distributions"]
trans = outputs["transitions"]
diag = outputs["diagnostics"]
audit = outputs["audit"]
min_date = pd.to_datetime(dist["date"]).min().date()
max_date = pd.to_datetime(dist["date"]).max().date()

with tabs[1]:
    cols = [f"aligned_prob_{x}" for x in ("low", "mid", "high")] if aligned else [f"raw_prob_state_{i}" for i in range(3)]
    st.plotly_chart(_multi_line(prob, cols, "Filtered state probabilities"), use_container_width=True)
    st.dataframe(prob.tail(50), use_container_width=True)

with tabs[2]:
    selected = st.slider("Snapshot date", min_value=min_date, max_value=max_date, value=max_date)
    st.plotly_chart(_density_snapshot(dist, selected, aligned), use_container_width=True)
    mean_cols = [f"aligned_{x}_return_mean" for x in ("low", "mid", "high")] if aligned else [f"raw_state_{i}_return_mean" for i in range(3)]
    std_cols = [f"aligned_{x}_return_std" for x in ("low", "mid", "high")] if aligned else [f"raw_state_{i}_return_std" for i in range(3)]
    st.plotly_chart(_multi_line(dist, mean_cols, "Predictive return mean"), use_container_width=True)
    st.plotly_chart(_multi_line(dist, std_cols, "Predictive return std"), use_container_width=True)

with tabs[3]:
    selected = st.slider("Transition date", min_value=min_date, max_value=max_date, value=max_date, key="matrix_date")
    st.plotly_chart(_transition_heatmap(trans, selected, aligned), use_container_width=True)
    diag_cols = [f"aligned_p_{x}_{x}" for x in ("low", "mid", "high")] if aligned else [f"raw_p_{i}{i}" for i in range(3)]
    st.plotly_chart(_multi_line(trans, diag_cols, "Self-transition probability"), use_container_width=True)

with tabs[4]:
    dur_cols = [f"aligned_duration_{x}" for x in ("low", "mid", "high")] if aligned else [f"raw_duration_{i}" for i in range(3)]
    capped = trans.copy()
    for col in dur_cols:
        capped[col] = np.minimum(capped[col], 252)
    st.plotly_chart(_multi_line(capped, dur_cols, "Implied state duration, capped at 252 days"), use_container_width=True)

with tabs[5]:
    st.metric("Refits", len(diag))
    st.metric("Non-converged", int((~diag["converged"].astype(bool)).sum()))
    st.plotly_chart(_line(diag, "date", "log_likelihood", "Log likelihood"), use_container_width=True)
    st.plotly_chart(_line(diag, "date", "bic", "BIC"), use_container_width=True)
    st.plotly_chart(_line(diag, "date", "fit_time_seconds", "Fit time seconds"), use_container_width=True)
    st.dataframe(diag.tail(50), use_container_width=True)

with tabs[6]:
    st.dataframe(audit, use_container_width=True)
    st.download_button("Download audit CSV", audit.to_csv(index=False), "ar_hmm_audit_table.csv", "text/csv")
