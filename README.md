# EnergyQuant

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet?logo=anthropic&logoColor=white)

## Overview

Quantitative analysis tool for the French electricity market. EnergyQuant pulls real-time data from multiple sources — ENTSO-E, Yahoo Finance, RTE — and exposes it through an interactive Streamlit dashboard deployed publicly. It covers spot price analysis, spread modelling, risk metrics, cross-commodity correlations, price forecasting and a dynamic merit order model.

## Live Dashboard

[energyquant-florianb.streamlit.app](https://energyquant-florianb.streamlit.app)

## Features

1. **Real-time day-ahead spot prices** — France hourly prices via the ENTSO-E Transparency Platform API
2. **Price seasonality analysis** — hourly profile, hour × weekday heatmap, peak-to-mean ratio
3. **Spread analysis** — spark spread, dark spread and clean spark spread from live commodity prices
4. **Market correlations** — Pearson correlation matrix between Power, TTF Gas and EUA Carbon (Yahoo Finance)
5. **Risk metrics** — daily volatility, VaR 95%, CVaR 95%, skewness and kurtosis
6. **7-day price forecasting** — Prophet model (Meta) with flat trend and weekly seasonality
7. **French merit order** — dynamic demand slider, marginal technology identification, RTE installed capacity data (end of 2025)

## Project Structure

```
data/
  fetcher.py              ENTSO-E API client — fetches day-ahead prices for any bidding zone

analysis/
  spreads.py              Spark spread, dark spread, clean spark spread calculations
  risk.py                 Volatility, VaR 95%, CVaR 95%, skewness, kurtosis
  seasonality.py          Hourly profile, hour × weekday heatmap, seasonality summary
  correlations.py         Cross-commodity correlation (Power / TTF Gas / EUA Carbon)
  merit_order.py          French merit order stack with RTE capacity data (end of 2025)

forecasting/
  prophet_model.py        Prophet model — train, forecast 7 days, flat trend + weekly seasonality

dashboard/
  app.py                  Streamlit dashboard — all sections, sidebar controls, interactive charts
```

## Data Sources

- **ENTSO-E Transparency Platform** — day-ahead electricity prices (requires API key)
- **Yahoo Finance** via `yfinance` — TTF natural gas front-month futures, EUA carbon allowances
- **RTE** — French installed generation capacity, end of 2025

## Installation

```bash
git clone https://github.com/florianb/EnergyQuant.git
cd EnergyQuant
pip install -r requirements.txt
```

Create a `.env` file at the project root:

```
ENTSOE_API_KEY=your_api_key_here
```

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

## Tech Stack

Python · ENTSO-E API · yfinance · Prophet (Meta) · Streamlit · Plotly · pandas · scipy

## Author

**Florian Bousso** — Energy & Markets Engineer

## Built with

[Claude Code](https://claude.ai/code) by Anthropic
