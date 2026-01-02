#!/usr/bin/env python3
from __future__ import annotations

import os
import time
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from streamlit_autorefresh import st_autorefresh  # ✅ autorefresh

# 🔌 your helpers
from enso_streaming.utils import load_enso_indices, clean_enso


# -------------------
# Page
# -------------------
st.set_page_config(page_title="Live ENSO Forecast Dashboard", layout="wide")
st.title("🌊 Live ENSO Forecast Dashboard")


# -------------------
# Session-state config (LOCK after Save)
# -------------------
DEFAULTS = {
    "configured": False,
    "pred_csv": "./out/live_predictions.csv",
    "enso_path": "./data/nino34.long.anom.data.txt",
    "diag_csv": "./out/diagnostic.csv",
    "refresh_ms": 10000,
}

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------
# Sidebar (editable only until saved)
# -------------------
with st.sidebar:
    st.header("⚙️ Settings")

    if not st.session_state["configured"]:
        st.text_input("Predictions CSV", key="pred_csv")
        st.text_input("ENSO txt path", key="enso_path")
        st.text_input("Diagnostics CSV (output)", key="diag_csv")
        st.slider("Refresh every (ms)", 2000, 30000, step=1000, key="refresh_ms")

        st.caption(f"The page auto-refreshes every **{st.session_state['refresh_ms']/1000:.1f} seconds**.")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Save settings"):
                st.session_state["configured"] = True
                st.rerun()
        with c2:
            if st.button("↩ Reset defaults"):
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v
                st.rerun()

    else:
        st.success("Settings locked ✅ (persist across refresh)")
        st.write("**Pred CSV:**", st.session_state["pred_csv"])
        st.write("**ENSO path:**", st.session_state["enso_path"])
        st.write("**Diag CSV:**", st.session_state["diag_csv"])
        st.write("**Refresh (ms):**", st.session_state["refresh_ms"])

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔓 Unlock settings"):
                st.session_state["configured"] = False
                st.rerun()
        with c2:
            if st.button("↩ Reset defaults"):
                for k, v in DEFAULTS.items():
                    st.session_state[k] = v
                st.rerun()


# ✅ Streamlit-native autorefresh (reruns script, keeps session state)
st_autorefresh(interval=int(st.session_state["refresh_ms"]), key="enso_autorefresh")


# -------------------
# Metrics
# -------------------
def mae_rmse(y_true, y_pred) -> Tuple[float, float]:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.size == 0:
        return float("nan"), float("nan")
    err = y_pred - y_true
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))


def build_diagnostics_from_live(
    timestamps: pd.DatetimeIndex,
    actual_series: pd.Series,
    pred_series: pd.Series,
) -> pd.DataFrame:
    aligned = pd.concat(
        [actual_series.rename("actual"), pred_series.rename("pred")],
        axis=1,
    )

    diag_rows = []
    for t in aligned.index:
        sub = aligned.loc[:t].dropna()
        if not sub.empty:
            mae, rmse = mae_rmse(sub["actual"].values, sub["pred"].values)
            n_rows = len(sub)
        else:
            mae, rmse, n_rows = float("nan"), float("nan"), 0

        diag_rows.append(
            {
                "t": t,
                "actual": aligned.loc[t, "actual"],
                "pred": aligned.loc[t, "pred"],
                "mae_cum": mae,
                "rmse_cum": rmse,
                "n_rows": n_rows,
            }
        )

    return pd.DataFrame(diag_rows)


# -------------------
# Load ENSO via your utils
# -------------------
@st.cache_data(show_spinner=False)
def load_enso(path: str) -> pd.Series:
    raw = load_enso_indices(path)
    return clean_enso(raw)


# optional: small retry to avoid read errors while file is being written
def read_csv_retry(path: str, tries: int = 3, sleep_s: float = 0.15) -> pd.DataFrame:
    last_err = None
    for _ in range(tries):
        try:
            return pd.read_csv(path)
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise last_err


enso_path = st.session_state["enso_path"]
pred_csv = st.session_state["pred_csv"]
diag_csv = st.session_state["diag_csv"]

try:
    enso_raw = load_enso(enso_path)
except Exception as e:
    st.error(f"Failed to load ENSO file: {e}")
    st.stop()


# -------------------
# Load predictions CSV (timestamp, pred)
# -------------------
if not os.path.exists(pred_csv):
    st.warning(f"Prediction CSV not found: {pred_csv}")
    st.stop()

try:
    dfp = read_csv_retry(pred_csv)
except Exception as e:
    st.warning(f"Could not read '{pred_csv}': {e}")
    st.stop()

if "timestamp" not in dfp.columns or "pred" not in dfp.columns:
    st.warning("CSV must have columns: timestamp, pred")
    st.stop()

# Normalize timestamps (monthly start)
ts = pd.to_datetime(dfp["timestamp"], errors="coerce")
dfp.index = ts.dt.to_period("M").dt.to_timestamp(how="S")
dfp = dfp.sort_index()
dfp = dfp.loc[dfp.index.notna()]

if dfp.empty:
    st.info("No prediction rows yet.")
    st.stop()


# -------------------
# Build series
# -------------------
actual_series = enso_raw.reindex(dfp.index)
pred_series = dfp["pred"].astype(float)
forecast_series = pd.Series(pred_series.values, index=dfp.index + pd.DateOffset(months=1))

diag_df = build_diagnostics_from_live(dfp.index, actual_series, pred_series)

last_diag = diag_df.iloc[-1]
t_last = last_diag["t"]
actual_last = last_diag["actual"]
pred_last = last_diag["pred"]
mae = last_diag["mae_cum"]
rmse = last_diag["rmse_cum"]
n_rows = int(last_diag["n_rows"])


# -------------------
# Save diagnostics CSV (timestamp + metrics ONLY)
# -------------------
try:
    diag_df_out = diag_df[["t", "mae_cum", "rmse_cum", "n_rows"]].copy()
    diag_df_out.rename(columns={"t": "timestamp"}, inplace=True)
    diag_df_out.to_csv(diag_csv, index=False)
except Exception as e:
    st.warning(f"Could not write diagnostics CSV '{diag_csv}': {e}", icon="⚠️")


# -------------------
# KPI row
# -------------------
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Last month (t)", pd.to_datetime(t_last).strftime("%Y-%m-%d"))
with kpi_cols[1]:
    st.metric("Actual ENSO @ t", f"{actual_last:.3f}" if np.isfinite(actual_last) else "NaN")
with kpi_cols[2]:
    st.metric("Forecast @ t+1", f"{pred_last:.3f}")
with kpi_cols[3]:
    st.metric("MAE (cumulative)", f"{mae:.3f}" if np.isfinite(mae) else "NaN")

kpi_cols2 = st.columns(2)
with kpi_cols2[0]:
    st.metric("RMSE (cumulative)", f"{rmse:.3f}" if np.isfinite(rmse) else "NaN")
with kpi_cols2[1]:
    st.metric("n points used", f"{n_rows}")

st.markdown("---")


# -------------------
# Plots
# -------------------
col_main, col_diag = st.columns([2.2, 1.8], gap="large")

with col_main:
    st.subheader("Forecast vs Actual")

    df_actual = pd.DataFrame(
        {"timestamp": actual_series.index, "value": actual_series.values, "series": "Actual @ t"}
    ).dropna(subset=["value"])

    df_fore = pd.DataFrame(
        {"timestamp": forecast_series.index, "value": forecast_series.values, "series": "Forecast @ t+1"}
    ).dropna(subset=["value"])

    if df_actual.empty and df_fore.empty:
        st.info("Nothing to plot yet (both series empty).")
    else:
        df_long = pd.concat([df_actual, df_fore], axis=0)
        base = alt.Chart(df_long).encode(
            x=alt.X("timestamp:T", title="Month"),
            y=alt.Y("value:Q", title="ENSO anomaly"),
            color=alt.Color("series:N", title=None),
            tooltip=["timestamp:T", "series:N", "value:Q"],
        )
        st.altair_chart(
            (base.mark_line() + base.mark_point(size=40)).properties(height=420),
            use_container_width=True,
        )

with col_diag:
    st.subheader("Diagnostics over time")

    if diag_df.empty:
        st.caption("No diagnostics available.")
    else:
        d_plot = diag_df.copy()
        d_plot["mae_cum"] = pd.to_numeric(d_plot["mae_cum"], errors="coerce")
        d_plot["rmse_cum"] = pd.to_numeric(d_plot["rmse_cum"], errors="coerce")
        d_plot = d_plot.dropna(subset=["mae_cum", "rmse_cum"])

        if d_plot.empty:
            st.caption("Diagnostics have no numeric metrics yet.")
        else:
            m = d_plot.melt(
                id_vars=["t"],
                value_vars=["mae_cum", "rmse_cum"],
                var_name="metric",
                value_name="value",
            )
            m["metric"] = m["metric"].map({"mae_cum": "MAE (cum)", "rmse_cum": "RMSE (cum)"})

            diag_chart = (
                alt.Chart(m)
                .mark_line(point=True)
                .encode(
                    x=alt.X("t:T", title="timestamp (CSV)"),
                    y=alt.Y("value:Q", title="Metric value"),
                    color=alt.Color("metric:N", title=None),
                    tooltip=["t:T", "metric:N", "value:Q"],
                )
                .properties(height=220, title="Cumulative diagnostics (live CSV)")
            )
            st.altair_chart(diag_chart, use_container_width=True)

        st.caption("Last 6 diagnostic steps (internal view):")
        st.dataframe(
            d_plot.tail(6)[["t", "actual", "pred", "mae_cum", "rmse_cum", "n_rows"]],
            use_container_width=True,
        )

st.markdown("---")


# -------------------
# Downloads
# -------------------
st.subheader("Downloads")

dcol1, dcol2 = st.columns(2)

df_pred_out = pd.DataFrame({"timestamp": dfp.index, "pred": pred_series.values})
with dcol1:
    st.download_button(
        label="⬇️ Download predictions (timestamp + pred)",
        data=df_pred_out.to_csv(index=False),
        file_name="predictions_snapshot.csv",
        mime="text/csv",
        key="download_predictions_viewer",
    )

if os.path.exists(diag_csv) and os.path.getsize(diag_csv) > 0:
    try:
        diag_df_dl = pd.read_csv(diag_csv)
    except Exception:
        diag_df_dl = None
else:
    diag_df_dl = None

with dcol2:
    if diag_df_dl is not None and not diag_df_dl.empty:
        st.download_button(
            label="⬇️ Download diagnostics (timestamp + metrics)",
            data=diag_df_dl.to_csv(index=False),
            file_name="diagnostic_viewer.csv",
            mime="text/csv",
            key="download_diagnostics_viewer",
        )
    else:
        st.caption("No diagnostics available to download yet.")
