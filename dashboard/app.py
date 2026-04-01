"""
EnergyQuant interactive dashboard.

Streamlit application for visualising energy prices, spreads, volatility
and forecasts. Provides interactive charts, geographic and commodity filters,
and real-time risk alerts.
"""

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.risk import risk_summary
from analysis.spreads import (
    compute_clean_spark_spread,
    compute_dark_spread,
    compute_spark_spread,
)
from data.fetcher import fetch_day_ahead_prices
from forecasting.prophet_model import forecast, forecast_summary, prepare_data, train_model

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="EnergyQuant",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def load_prices(days: int) -> pd.Series:
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=days)
    df = fetch_day_ahead_prices("FR", start, end)
    return df["price_eur_mwh"]


@st.cache_data(ttl=3600, show_spinner=False)
def load_forecast(days: int) -> pd.DataFrame:
    prices = load_prices(days)
    df = prepare_data(prices)
    model = train_model(df)
    return forecast(model, periods=7)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Parameters")

period_label = st.sidebar.selectbox(
    "Analysis period",
    options=["7 days", "30 days", "90 days"],
    index=1,
)
period_days = {"7 days": 7, "30 days": 30, "90 days": 90}[period_label]

st.sidebar.markdown("---")
st.sidebar.subheader("Market assumptions")
gas_price = st.sidebar.slider("Gas (EUR/MWh)", min_value=10, max_value=100, value=30)
coal_price = st.sidebar.slider("Coal (EUR/MWh)", min_value=5, max_value=50, value=12)
carbon_price = st.sidebar.slider("Carbon EUA (EUR/tCO2)", min_value=10, max_value=120, value=65)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("EnergyQuant")
st.caption("Florian Bousso")
st.markdown(
    "Quantitative analysis tool for European energy markets. "
    "Real-time ENTSO-E data — France day-ahead."
)
st.markdown("---")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
with st.spinner("Loading ENTSO-E data..."):
    prices = load_prices(period_days)

# ---------------------------------------------------------------------------
# Section 1 : Day-ahead spot prices
# ---------------------------------------------------------------------------
st.header("Day-ahead spot prices — France")

st.markdown(
    "Day-ahead prices are set the day before for each delivery hour on the EPEX SPOT market. "
    "Negative prices reflect surplus non-dispatchable renewable generation."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Average price", f"{prices.mean():.1f} EUR/MWh")
col2.metric("Min price", f"{prices.min():.1f} EUR/MWh")
col3.metric("Max price", f"{prices.max():.1f} EUR/MWh")
col4.metric("Last price", f"{prices.iloc[-1]:.1f} EUR/MWh")

fig_prices = go.Figure()
fig_prices.add_trace(go.Scatter(
    x=prices.index,
    y=prices.values,
    mode="lines",
    name="Spot price",
    line=dict(color="#1f77b4", width=1.5),
    hovertemplate="%{x|%d/%m/%Y %H:%M}<br>%{y:.1f} EUR/MWh<extra></extra>",
))
fig_prices.update_layout(
    xaxis_title="Date",
    yaxis_title="EUR/MWh",
    hovermode="x unified",
    margin=dict(t=20, b=40),
    height=300,
)
st.plotly_chart(fig_prices, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 2 : Spread analysis
# ---------------------------------------------------------------------------
st.header("Spread analysis")

st.markdown(
    "Spreads measure the theoretical gross margin of different power plant types. "
    "The spark spread applies to gas-fired plants, the dark spread to coal plants. "
    "The clean spark spread incorporates the cost of carbon allowances (EUA)."
)

power_price = float(prices.iloc[-1])
spark = compute_spark_spread(power_price, gas_price)
dark = compute_dark_spread(power_price, coal_price)
clean_spark = compute_clean_spark_spread(power_price, gas_price, carbon_price)

spread_names = ["Spark spread", "Dark spread", "Clean spark spread"]
spread_values = [spark, dark, clean_spark]
spread_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in spread_values]

fig_spreads = go.Figure()
fig_spreads.add_trace(go.Bar(
    x=spread_names,
    y=spread_values,
    marker_color=spread_colors,
    text=[f"{v:.1f} EUR/MWh" for v in spread_values],
    textposition="outside",
    hovertemplate="%{x}<br>%{y:.2f} EUR/MWh<extra></extra>",
))
fig_spreads.add_hline(y=0, line_dash="dot", line_color="gray")
fig_spreads.update_layout(
    yaxis_title="EUR/MWh",
    margin=dict(t=20, b=40),
    height=320,
)
st.plotly_chart(fig_spreads, use_container_width=True)

with st.expander("Spread definitions"):
    st.markdown(
        """
        - Spark spread: `power - gas / 0.49` — theoretical margin of a gas-fired plant (CCGT)
        - Dark spread: `power - coal / 0.35` — theoretical margin of a coal-fired plant
        - Clean spark spread: spark spread minus the cost of CO2 allowances (EUA x 0.202 tCO2/MWh)
        """
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3 : Risk metrics
# ---------------------------------------------------------------------------
st.header("Risk metrics")

st.markdown(
    "Volatility is expressed as the standard deviation of daily average prices. "
    "The 95% VaR represents the maximum probable loss on a 1 MWh position "
    "in 95% of cases over the analysis period."
)

with st.spinner("Computing risk metrics..."):
    risk = risk_summary(prices)

def _fmt(val, decimals=2, unit=""):
    return f"{val:.{decimals}f}{unit}" if val is not None else "n/a"

col_v, col_var, col_cvar, col_skew = st.columns(4)
col_v.metric(
    "Volatility (daily std)",
    _fmt(risk["volatility_eur_mwh"], 1, " EUR/MWh"),
    delta=_fmt(risk["volatility_pct"], 1, " %"),
    delta_color="off",
)
col_var.metric("VaR 95%", _fmt(risk["var_95_eur_mwh"], 2, " EUR/MWh"))
col_cvar.metric("CVaR 95%", _fmt(risk["cvar_95_eur_mwh"], 2, " EUR/MWh"))
col_skew.metric("Skewness", _fmt(risk["skewness"], 3))

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4 : Price forecast J+7
# ---------------------------------------------------------------------------
st.header("Price Forecast J+1 to J+7 (Prophet)")

st.markdown(
    "Forecasts generated by the Prophet model (Meta) trained on the last 90 days "
    "of daily prices. The model captures the weekly seasonality of electricity markets."
)

with st.spinner("Training Prophet model..."):
    forecast_df = load_forecast(90)

summary = forecast_summary(forecast_df)

# Table
rows = [
    {
        "Date": date,
        "Forecast price (EUR/MWh)": vals["price"],
        "Lower bound": vals["lower"],
        "Upper bound": vals["upper"],
    }
    for date, vals in summary.items()
]
st.dataframe(
    pd.DataFrame(rows).set_index("Date"),
    use_container_width=True,
)

# Chart: forecast + recent history
daily_hist = prices.resample("D").mean().tail(30)

future_dates = [pd.Timestamp(d) for d in summary]
future_yhat  = [v["price"] for v in summary.values()]
future_lower = [v["lower"] for v in summary.values()]
future_upper = [v["upper"] for v in summary.values()]

fig_fc = go.Figure()
fig_fc.add_trace(go.Scatter(
    x=daily_hist.index,
    y=daily_hist.values,
    mode="lines",
    name="Historical (daily avg.)",
    line=dict(color="#1f77b4", width=2),
))
fig_fc.add_trace(go.Scatter(
    x=future_dates + future_dates[::-1],
    y=future_upper + future_lower[::-1],
    fill="toself",
    fillcolor="rgba(255,127,14,0.15)",
    line=dict(color="rgba(255,127,14,0)"),
    name="Confidence interval",
    hoverinfo="skip",
))
fig_fc.add_trace(go.Scatter(
    x=future_dates,
    y=future_yhat,
    mode="lines+markers",
    name="Forecast",
    line=dict(color="#ff7f0e", width=2, dash="dash"),
    marker=dict(size=7),
    hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f} EUR/MWh<extra></extra>",
))
fig_fc.update_layout(
    xaxis_title="Date",
    yaxis_title="EUR/MWh",
    hovermode="x unified",
    margin=dict(t=20, b=40),
    height=350,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_fc, use_container_width=True)

st.caption(
    "Model: Prophet (growth=flat, weekly seasonality, trained on 90 days). "
    "80% confidence interval."
)
