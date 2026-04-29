"""
EnergyQuant interactive dashboard.

Streamlit application for visualising energy prices, spreads, volatility
and forecasts. Provides interactive charts, geographic and commodity filters,
and real-time risk alerts.
"""

import base64
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.correlations import (
    build_commodity_dataframe,
    compute_correlation_matrix,
    correlation_summary,
)
from analysis.merit_order import CAPACITY_SOURCE, get_marginal_technology, get_merit_order
from analysis.risk import risk_summary
from analysis.seasonality import (
    compute_hourly_profile,
    compute_weekly_heatmap,
    seasonality_summary,
)
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
def load_commodity_df(days: int) -> pd.DataFrame:
    prices = load_prices(days)
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=days)
    return build_commodity_dataframe(prices, start, end)


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

try:
    _ttf_df = load_commodity_df(90)
except Exception:
    _ttf_df = pd.DataFrame()

_gas_default = int(round(_ttf_df["gas_ttf"].iloc[-1])) if not _ttf_df.empty and "gas_ttf" in _ttf_df.columns else 45
_gas_default = max(10, min(150, _gas_default))  # clamp to slider range

_carbon_default = int(round(_ttf_df["carbon_eua"].iloc[-1])) if not _ttf_df.empty and "carbon_eua" in _ttf_df.columns else 65
_carbon_default = max(10, min(120, _carbon_default))

# Initialise session_state keys on first run
if "gas_price"    not in st.session_state: st.session_state["gas_price"]    = _gas_default
if "coal_price"   not in st.session_state: st.session_state["coal_price"]   = 130
if "carbon_price" not in st.session_state: st.session_state["carbon_price"] = _carbon_default
if "demand"       not in st.session_state: st.session_state["demand"]       = 60

# Callback runs before widgets are re-instantiated — avoids the
# "cannot modify after instantiation" Streamlit constraint
def _reset_defaults():
    st.session_state["gas_price"]    = _gas_default
    st.session_state["coal_price"]   = 130
    st.session_state["carbon_price"] = _carbon_default
    st.session_state["demand"]       = 60


gas_price = st.sidebar.slider("Gas (EUR/MWh)", min_value=10, max_value=150, key="gas_price")
st.sidebar.caption("Default: last TTF closing price")
coal_price_tonne = st.sidebar.slider("Coal (EUR/tonne)", min_value=50, max_value=300, key="coal_price")
carbon_price = st.sidebar.slider("Carbon EUA (EUR/tCO2)", min_value=10, max_value=120, key="carbon_price")
st.sidebar.caption("Default: last EUA closing price")
demand_gw = st.sidebar.slider("Demand (GW)", min_value=20, max_value=100, step=1, key="demand")
st.sidebar.caption("Default: 60 GW — approximate average French daytime demand")
st.sidebar.button("Reset to defaults", on_click=_reset_defaults)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
_root = Path(__file__).resolve().parents[1]

def _svg_html(path: Path, width: int, extra_style: str = "") -> str:
    """Return an HTML img tag with the SVG embedded as a base64 data URI."""
    data = base64.b64encode(path.read_bytes()).decode()
    return (
        f'<div style="background:none; padding:0; {extra_style}">'
        f'<img src="data:image/svg+xml;base64,{data}" width="{width}">'
        f'</div>'
    )

_icon_b64 = base64.b64encode((_root / "assets/01-lattice/eq-square-icon.svg").read_bytes()).decode()
st.markdown(f"""
<div style="display: flex; align-items: center;
            justify-content: space-between;
            width: 100%; padding: 16px 0 24px 0;">
    <img src="data:image/svg+xml;base64,{_icon_b64}"
         style="height: 100px; width: auto;">
    <div style="text-align: center; flex: 1;">
        <div style="font-family: Arial, sans-serif;
                    font-weight: 800; font-size: 52px;
                    letter-spacing: -0.6px; color: #0d1117;">
            Energy<span style="color: #f5d547;">Q</span>uant
        </div>
        <div style="font-family: monospace; font-size: 13px;
                    letter-spacing: 2.2px; color: #6b7280;">
            QUANTITATIVE ENERGY ANALYTICS
        </div>
    </div>
    <div style="width: 100px;"></div>
</div>
""", unsafe_allow_html=True)
st.caption("Florian Bousso")
st.markdown(
    "Quantitative analysis tool for the French electricity market. "
    "Real-time ENTSO-E day-ahead prices, spread analysis, "
    "risk metrics and price forecasting."
)
def _nav_icon(path: Path, size: int = 36) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/svg+xml;base64,{data}" style="height:{size}px; width:{size}px;">'

_ni = {
    "candle":       _nav_icon(_root / "assets/06-candle/icon-light.svg"),
    "seasonality":  _nav_icon(_root / "assets/sections/01-seasonality/icon-light.svg"),
    "spread":       _nav_icon(_root / "assets/sections/02-spread/icon-light.svg"),
    "correlations": _nav_icon(_root / "assets/sections/03-correlations/icon-light.svg"),
    "merit":        _nav_icon(_root / "assets/02-merit-order/icon-light.svg"),
    "risk":         _nav_icon(_root / "assets/sections/04-risk/icon-light.svg"),
    "forecast":     _nav_icon(_root / "assets/sections/05-forecast/icon-light.svg"),
}

_btn = (
    "background-color: #f0f2f6;"
    "color: #262730;"
    "padding: 10px 16px;"
    "border-radius: 10px;"
    "text-decoration: none;"
    "font-weight: 500;"
    "font-size: 13px;"
    "border: 1px solid #d0d3da;"
    "display: flex;"
    "flex-direction: column;"
    "align-items: center;"
    "gap: 6px;"
    "min-width: 90px;"
    "text-align: center;"
)

st.markdown(
    f"""
    <div style="display: flex; gap: 10px; justify-content: center; margin: 10px 0;">
      <a href="#day-ahead-spot-prices-france" style="{_btn}">{_ni['candle']}Day-ahead Prices</a>
      <a href="#price-seasonality" style="{_btn}">{_ni['seasonality']}Price Seasonality</a>
      <a href="#spread-analysis" style="{_btn}">{_ni['spread']}Spread Analysis</a>
      <a href="#market-correlations" style="{_btn}">{_ni['correlations']}Market Correlations</a>
      <a href="#merit-order" style="{_btn}">{_ni['merit']}Merit Order</a>
      <a href="#risk-metrics" style="{_btn}">{_ni['risk']}Risk Metrics</a>
      <a href="#price-forecast-j-1-to-j-7-prophet" style="{_btn}">{_ni['forecast']}Price Forecast</a>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
with st.spinner("Loading ENTSO-E data..."):
    try:
        prices = load_prices(period_days)
    except Exception as e:
        st.error(
            f"ENTSO-E API unavailable: {e}\n\n"
            "The Transparency Platform may be temporarily down. Please try again in a few minutes."
        )
        st.stop()

# ---------------------------------------------------------------------------
# Section 1 : Day-ahead spot prices
# ---------------------------------------------------------------------------
_col_icon, _col_title = st.columns([0.05, 0.95])
with _col_icon:
    st.markdown(_svg_html(_root / "assets/06-candle/icon-light.svg", 45, "padding-top:12px"), unsafe_allow_html=True)
with _col_title:
    st.header("Day-ahead spot prices — France")

st.markdown(
    "Day-ahead prices are set the day before for each delivery hour on the EPEX SPOT market. "
    "Negative prices reflect surplus non-dispatchable renewable generation."
)

col1, col2, col3, col4 = st.columns(4)
_val_style = "font-size:28px; font-weight:600; margin:0"
with col1:
    st.caption("Average price")
    st.markdown(f"<p style='{_val_style}'>{prices.mean():.1f} EUR/MWh</p>", unsafe_allow_html=True)
with col2:
    st.caption("Min price")
    st.markdown(f"<p style='{_val_style}'>{prices.min():.1f} EUR/MWh</p>", unsafe_allow_html=True)
with col3:
    st.caption("Max price")
    st.markdown(f"<p style='{_val_style}'>{prices.max():.1f} EUR/MWh</p>", unsafe_allow_html=True)
with col4:
    st.caption("Last price")
    st.markdown(f"<p style='{_val_style}'>{prices.iloc[-1]:.1f} EUR/MWh</p>", unsafe_allow_html=True)

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
# Section 2 : Price seasonality
# ---------------------------------------------------------------------------
_col_icon, _col_title = st.columns([0.05, 0.95])
with _col_icon:
    st.markdown(_svg_html(_root / "assets/sections/01-seasonality/icon-light.svg", 45, "padding-top:12px"), unsafe_allow_html=True)
with _col_title:
    st.header("Price Seasonality")

st.markdown(
    "Electricity prices follow strong intraday and weekly patterns. "
    "Peak hours (evening) are systematically more expensive than off-peak hours, "
    "and weekdays show higher prices than weekends due to industrial demand patterns. "
    "The peak-to-mean ratio measures how much more expensive the peak hour is compared "
    "to the overall average price — a ratio of 2.0 means the peak hour is twice the average price."
)

season = seasonality_summary(prices)
profile = compute_hourly_profile(prices)
heatmap_df = compute_weekly_heatmap(prices)

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Peak hour",     f"{season['peak_hour']:02d}:00")
sc2.metric("Off-peak hour", f"{season['offpeak_hour']:02d}:00")
sc3.metric("Peak day",      season["peak_day"])
sc4.metric(
    "Peak-to-mean ratio",
    f"{season['peak_to_mean_ratio']:.3f}" if season["peak_to_mean_ratio"] is not None else "n/a",
)

col_heat, col_bar = st.columns(2)

with col_heat:
    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_df.values,
        x=heatmap_df.columns.tolist(),
        y=heatmap_df.index.tolist(),
        colorscale="RdYlGn_r",
        colorbar=dict(title="EUR/MWh"),
        hovertemplate="Hour %{y}:00 — %{x}<br>%{z:.1f} EUR/MWh<extra></extra>",
    ))
    fig_heat.update_layout(
        title="Price heatmap (EUR/MWh)",
        xaxis_title="Day of week",
        yaxis_title="Hour of day",
        margin=dict(t=40, b=40),
        height=380,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with col_bar:
    mean_vals = profile["mean"].values
    fig_hourly = go.Figure(go.Bar(
        x=profile.index.tolist(),
        y=mean_vals,
        marker=dict(
            color=mean_vals,
            colorscale="Blues",
            showscale=False,
        ),
        hovertemplate="Hour %{x}:00<br>Mean: %{y:.1f} EUR/MWh<extra></extra>",
    ))
    fig_hourly.update_layout(
        title="Average price by hour of day (EUR/MWh)",
        xaxis_title="Hour of day",
        yaxis_title="EUR/MWh",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2),
        margin=dict(t=40, b=40),
        height=380,
    )
    st.plotly_chart(fig_hourly, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3 : Spread analysis
# ---------------------------------------------------------------------------
_col_icon, _col_title = st.columns([0.05, 0.95])
with _col_icon:
    st.markdown(_svg_html(_root / "assets/sections/02-spread/icon-light.svg", 45, "padding-top:12px"), unsafe_allow_html=True)
with _col_title:
    st.header("Spread analysis")

st.markdown(
    "Spreads measure the theoretical gross margin of different power plant types. "
    "The spark spread applies to gas-fired plants, the dark spread to coal plants. "
    "The clean spark spread incorporates the cost of carbon allowances (EUA)."
)
st.latex(r"\text{Spark Spread} = P_{elec} - \frac{P_{gas}}{\eta_{gas}}, \quad \eta_{gas} = 0.49")
st.latex(r"\text{Dark Spread} = P_{elec} - \frac{P_{coal}}{\eta_{coal}}, \quad \eta_{coal} = 0.35")
st.latex(r"\text{Clean Spark Spread} = \text{Spark Spread} - P_{CO_2} \times EF, \quad EF = 0.202 \text{ tCO}_2/\text{MWh}")

power_price = float(prices.mean())
coal_price_mwh = coal_price_tonne / 8.14  # EUR/tonne → EUR/MWh (8.14 MWh/tonne)
spark = compute_spark_spread(power_price, gas_price)
dark = compute_dark_spread(power_price, coal_price_mwh)
clean_spark = compute_clean_spark_spread(power_price, gas_price, carbon_price)

st.caption(
    f"Power price used: {power_price:.1f} EUR/MWh (mean over selected period) — "
    f"Coal: {coal_price_tonne} EUR/tonne = {coal_price_tonne / 8.14:.1f} EUR/MWh"
)

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

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 3 : Market correlations
# ---------------------------------------------------------------------------
_col_icon, _col_title = st.columns([0.05, 0.95])
with _col_icon:
    st.markdown(_svg_html(_root / "assets/sections/03-correlations/icon-light.svg", 45, "padding-top:12px"), unsafe_allow_html=True)
with _col_title:
    st.header("Market Correlations")

st.markdown(
    "Correlation analysis between French day-ahead power prices, TTF natural gas "
    "and EUA carbon allowances. "
    "Note on data sources and methodology: power prices are computed as daily averages "
    "of 15-minute ENTSO-E spot prices, while TTF and EUA prices are daily closing prices "
    "from Yahoo Finance. Pearson correlation is computed on 90 trading days for statistical "
    "significance. A high power/carbon correlation reflects the influence of carbon costs "
    "on thermal generation marginal costs, while a low power/gas correlation in France "
    "reflects the dominance of nuclear baseload in the French power mix."
)

with st.spinner("Loading commodity prices (Yahoo Finance)..."):
    # Charts: follow sidebar period for visual consistency
    commodity_df = load_commodity_df(period_days)
    # Correlations: always 90 days for statistical significance
    commodity_df_90 = load_commodity_df(90)

if commodity_df.empty:
    st.warning("Commodity data unavailable for the selected period.")
else:
    # Correlations computed on 90-day window regardless of sidebar selection
    corr_source = commodity_df_90 if not commodity_df_90.empty else commodity_df
    corr_matrix = compute_correlation_matrix(corr_source)
    corr_sum = correlation_summary(corr_matrix)

    # Charts sliced to sidebar period (daily frequency, already in commodity_df)
    _chart_layout = dict(margin=dict(t=40, b=40), height=280)

    cc1, cc2, cc3 = st.columns(3)

    with cc1:
        fig_pow = go.Figure(go.Scatter(
            x=commodity_df.index, y=commodity_df["power"],
            mode="lines", line=dict(color="#1f77b4", width=1.5),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f} EUR/MWh<extra></extra>",
        ))
        fig_pow.update_layout(title="Power — FR Day-Ahead (EUR/MWh)",
                              yaxis_title="EUR/MWh", **_chart_layout)
        st.plotly_chart(fig_pow, use_container_width=True)

    with cc2:
        fig_gas = go.Figure(go.Scatter(
            x=commodity_df.index, y=commodity_df["gas_ttf"],
            mode="lines", line=dict(color="#ff7f0e", width=1.5),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f} EUR/MWh<extra></extra>",
        ))
        fig_gas.update_layout(title="Gas — TTF (EUR/MWh)",
                              yaxis_title="EUR/MWh", **_chart_layout)
        st.plotly_chart(fig_gas, use_container_width=True)

    with cc3:
        fig_co2 = go.Figure(go.Scatter(
            x=commodity_df.index, y=commodity_df["carbon_eua"],
            mode="lines", line=dict(color="#2ca02c", width=1.5),
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f} EUR/tCO2<extra></extra>",
        ))
        fig_co2.update_layout(title="Carbon — EUA (EUR/tCO2)",
                              yaxis_title="EUR/tCO2", **_chart_layout)
        st.plotly_chart(fig_co2, use_container_width=True)

    def _fmt_corr(val):
        return f"{val:.3f}" if val is not None else "n/a"

    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Power / Gas (TTF)",    _fmt_corr(corr_sum["power_gas"]))
    mc2.metric("Power / Carbon (EUA)", _fmt_corr(corr_sum["power_carbon"]))
    mc3.metric("Gas / Carbon (EUA)",   _fmt_corr(corr_sum["gas_carbon"]))
    st.caption("Correlations computed on 90 trading days for statistical significance.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4 : Merit order
# ---------------------------------------------------------------------------
_col_icon, _col_title = st.columns([0.05, 0.95])
with _col_icon:
    st.markdown(_svg_html(_root / "assets/02-merit-order/icon-light.svg", 45, "padding-top:12px"), unsafe_allow_html=True)
with _col_title:
    st.header("Merit Order")

st.markdown(
    "The merit order ranks power plants by short-run marginal cost (SRMC). "
    "The market clearing price is set by the most expensive plant called "
    "to meet demand — the marginal technology. "
    "Drag the demand slider to see how the marginal technology changes. "
    "Click 'Capacity assumptions' to view the underlying data."
)

mo_df = get_merit_order(gas_price, coal_price_tonne, carbon_price)
marginal = get_marginal_technology(mo_df, demand_gw)

with st.expander("Capacity assumptions"):
    cap_table = mo_df[["technology", "installed_gw", "capacity_factor", "capacity_gw", "marginal_cost"]].copy()
    cap_table["capacity_factor"] = (cap_table["capacity_factor"] * 100).round(0).astype(int)
    cap_table.columns = [
        "Technology",
        "Installed (GW)",
        "Capacity Factor (%)",
        "Available (GW)",
        "Marginal Cost (EUR/MWh)",
    ]
    st.dataframe(cap_table.set_index("Technology"), use_container_width=True)
    st.caption("Source: " + CAPACITY_SOURCE)
    st.caption(
        "Capacity factors reflect average availability. "
        "Renewable output is weather-dependent and varies significantly by season and time of day."
    )

    st.markdown("**Marginal cost assumptions**")
    st.markdown(
        "Supposed fixed marginal costs:\n"
        "- **Wind / Solar**: 0 EUR/MWh — no fuel cost\n"
        "- **Hydro run-of-river**: 5 EUR/MWh — O&M only\n"
        "- **Nuclear**: 8 EUR/MWh — fuel + O&M\n"
        "- **Hydro reservoir**: 10 EUR/MWh — water opportunity value\n"
        "- **Oil**: 200 EUR/MWh — fuel oil + O&M"
    )
    _coal_srmc = round((coal_price_tonne / 8.14) / 0.35 + carbon_price * 0.34, 1)
    _gas_srmc  = round(gas_price / 0.49 + carbon_price * 0.202, 1)
    st.markdown(
        "Market-linked marginal costs (computed from sidebar prices):\n"
        f"- **Coal**: fuel cost (ARA coal price) + carbon cost (EUA) — "
        f"Current value: **{_coal_srmc} EUR/MWh**\n"
        f"- **Gas CCGT**: fuel cost (TTF gas price) + carbon cost (EUA) — "
        f"Current value: **{_gas_srmc} EUR/MWh**"
    )

_tech_colors = {
    "Wind onshore":       "#2ecc71",
    "Wind offshore":      "#27ae60",
    "Solar":              "#f39c12",
    "Hydro run-of-river": "#3498db",
    "Nuclear":            "#9b59b6",
    "Hydro reservoir":    "#1a5276",
    "Coal":               "#6d4c41",
    "Gas CCGT":           "#e67e22",
    "Oil":                "#7f8c8d",
}

# --- Merit order step chart ---
fig_mo = go.Figure()
for _, row in mo_df.iterrows():
    x_start = row["cumulative_capacity"] - row["capacity_gw"]
    x_end   = row["cumulative_capacity"]
    color   = _tech_colors.get(row["technology"], "#aec7e8")
    fig_mo.add_trace(go.Scatter(
        x=[x_start, x_end],
        y=[row["marginal_cost"], row["marginal_cost"]],
        mode="lines",
        name=row["technology"],
        line=dict(color=color, width=6),
        hovertemplate=(
            f"<b>{row['technology']}</b><br>"
            f"Capacity: {row['capacity_gw']:.0f} GW<br>"
            f"SRMC: {row['marginal_cost']:.1f} EUR/MWh<extra></extra>"
        ),
    ))
fig_mo.add_vline(x=demand_gw, line_dash="dot", line_color="red", line_width=2)
fig_mo.add_annotation(
    x=demand_gw,
    y=marginal["marginal_cost"],
    text=f"Demand: {demand_gw} GW",
    showarrow=True,
    arrowhead=2,
    arrowcolor="red",
    font=dict(color="red", size=12),
    xanchor="left",
    yanchor="bottom",
    ax=20,
    ay=-30,
)
fig_mo.add_trace(go.Scatter(
    x=[demand_gw],
    y=[marginal["marginal_cost"]],
    mode="markers",
    marker=dict(color="red", size=10, symbol="circle"),
    name="Demand level",
    showlegend=False,
    hovertemplate=(
        f"Demand: {demand_gw} GW<br>"
        f"Marginal: {marginal['technology']}<br>"
        f"SRMC: {marginal['marginal_cost']:.1f} EUR/MWh<extra></extra>"
    ),
))
fig_mo.update_layout(
    title="French Merit Order — Marginal cost curve",
    xaxis_title="Cumulative capacity (GW)",
    yaxis_title="Marginal cost (EUR/MWh)",
    hovermode="closest",
    margin=dict(t=80, b=40),
    height=380,
    legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0),
)
st.plotly_chart(fig_mo, use_container_width=True)

# --- Merit order metrics for selected demand ---
moc1, moc2, moc3 = st.columns(3)
moc1.metric("Marginal technology", marginal["technology"])
moc2.metric("Marginal cost",       f"{marginal['marginal_cost']:.1f} EUR/MWh")
moc3.metric("Spare margin",        f"{marginal['margin_gw']:.1f} GW")


st.markdown("---")

# ---------------------------------------------------------------------------
# Section 5 : Risk metrics
# ---------------------------------------------------------------------------
_col_icon, _col_title = st.columns([0.05, 0.95])
with _col_icon:
    st.markdown(_svg_html(_root / "assets/sections/04-risk/icon-light.svg", 45, "padding-top:12px"), unsafe_allow_html=True)
with _col_title:
    st.header("Risk metrics")

st.markdown(
    "**Volatility (daily std)** — Standard deviation of daily average prices over the analysis "
    "period. Measures the typical price dispersion around the mean. Expressed in EUR/MWh.\n\n"
    "**VaR 95%** (Value at Risk) — Maximum probable price variation on a 1 MWh position that "
    "will not be exceeded in 95% of cases. Computed on raw 15-minute price levels over the "
    "analysis period. A VaR of X EUR/MWh means that in 95% of periods, the price will not "
    "drop by more than X EUR/MWh.\n\n"
    "**CVaR 95%** (Conditional Value at Risk / Expected Shortfall) — Average price variation "
    "in the worst 5% of cases, i.e. when the loss exceeds the VaR threshold. Always greater "
    "than VaR and gives a better picture of tail risk.\n\n"
    "**Skewness** — Measures the asymmetry of the price distribution. A positive skew means "
    "extreme upward price spikes are more frequent than downward ones, which is typical of "
    "electricity markets (scarcity events, cold snaps)."
)
st.latex(r"\sigma = \sqrt{\frac{1}{N}\sum_{t=1}^{N}(P_t - \bar{P})^2}")
st.latex(r"\text{VaR}_{95\%} = -\text{Quantile}_{5\%}(\Delta P_t)")
st.latex(r"\text{CVaR}_{95\%} = -\mathbb{E}[\Delta P_t \mid \Delta P_t < -\text{VaR}_{95\%}]")
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

with st.spinner("Computing risk metrics..."):
    risk = risk_summary(prices)

def _fmt(val, decimals=2, unit=""):
    return f"{val:.{decimals}f}{unit}" if val is not None else "n/a"

vol = risk["volatility_eur_mwh"]
mean_price = risk["mean_price"]
cv = (vol / mean_price * 100) if (vol is not None and mean_price) else None

col_v, col_var, col_cvar, col_skew = st.columns(4)
col_v.metric(
    "Volatility (daily std)",
    _fmt(vol, 1, " EUR/MWh"),
    delta=f"CV: {cv:.1f}%" if cv is not None else "n/a",
    delta_color="off",
)
col_v.caption("CV = volatility / mean price — relative price dispersion")
col_var.metric("VaR 95%", _fmt(risk["var_95_eur_mwh"], 2, " EUR/MWh"))
col_cvar.metric("CVaR 95%", _fmt(risk["cvar_95_eur_mwh"], 2, " EUR/MWh"))
col_skew.metric("Skewness", _fmt(risk["skewness"], 3))

st.markdown("---")

# ---------------------------------------------------------------------------
# Section 4 : Price forecast J+7
# ---------------------------------------------------------------------------
_col_icon, _col_title = st.columns([0.05, 0.95])
with _col_icon:
    st.markdown(_svg_html(_root / "assets/sections/05-forecast/icon-light.svg", 45, "padding-top:12px"), unsafe_allow_html=True)
with _col_title:
    st.header("Price Forecast J+1 to J+7 (Prophet)")

st.markdown(
    "Forecasts generated by the Prophet model (Meta) trained on the last 90 days "
    "of daily prices. The model captures the weekly seasonality of electricity markets. "
    "Prophet decomposes the time series as y(t) = g(t) + s(t) + e(t), where g(t) is the "
    "trend (flat for electricity), s(t) the weekly seasonality, and e(t) the error term."
)
st.latex(r"y(t) = g(t) + s(t) + \epsilon(t)")

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
daily_hist = prices.resample("D").mean().tail(period_days)

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
