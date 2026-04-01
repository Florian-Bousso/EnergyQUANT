# EnergyQuant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Built with Claude Code](https://img.shields.io/badge/Built%20with-Claude%20Code-blueviolet?logo=anthropic&logoColor=white)

## Overview

EnergyQuant is a quantitative analysis tool for European energy markets, built for analysts and energy traders who need fast, reliable market intelligence. It pulls real-time day-ahead price data directly from the ENTSO-E Transparency Platform, computes financial risk metrics (volatility, VaR, CVaR), and generates 7-day price forecasts using Meta's Prophet model. Results are exposed through an interactive Streamlit dashboard designed for rapid exploratory analysis.

## Features

- Real-time ENTSO-E day-ahead price data (France)
- Spark, dark and clean spark spread calculations with adjustable market assumptions
- Risk metrics: daily volatility, VaR 95%, CVaR 95%, skewness
- 7-day price forecasting with Prophet (weekly seasonality, flat growth)
- Interactive Streamlit dashboard with Plotly charts

## Project Structure

```
EnergyQuant/
├── data/
│   ├── fetcher.py            # ENTSO-E API client — fetch day-ahead prices
│   └── cleaner.py            # Data cleaning and normalisation (stub)
├── analysis/
│   ├── spreads.py            # Spark, dark and clean spark spread calculations
│   ├── volatility.py         # Rolling volatility and GARCH (stub)
│   └── risk.py               # VaR, CVaR, daily volatility, skewness
├── forecasting/
│   ├── prophet_model.py      # Prophet model — prepare, train, forecast
│   └── evaluator.py          # Forecast evaluation and backtesting (stub)
├── dashboard/
│   └── app.py                # Streamlit interactive dashboard
├── tests/
│   └── __init__.py
├── cli.py                    # Command-line interface (stub)
├── requirements.txt
├── TASKS.md
└── .gitignore
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/EnergyQuant.git
cd EnergyQuant

# Install dependencies
pip install -r requirements.txt

# Configure your ENTSO-E API key
# Get your key at: https://transparency.entsoe.eu → My Account → Web API Security Token
echo "ENTSOE_API_KEY=your_api_key_here" > .env
```

## Usage

Launch the dashboard:

```bash
streamlit run dashboard/app.py
```

The sidebar lets you select the analysis period (7, 30 or 90 days) and adjust market assumptions (gas, coal and carbon prices) used to compute spreads. All data is cached for one hour to avoid redundant API calls.

## Tech Stack

| Component | Library |
|---|---|
| Data source | ENTSO-E Transparency Platform |
| API client | entsoe-py |
| Forecasting | Prophet (Meta) |
| Dashboard | Streamlit |
| Charts | Plotly |
| Data processing | pandas, NumPy, SciPy |

## Author

**Florian Bousso** — Energy engineer & markets analyst

## Built with

Built with [Claude Code](https://claude.ai/code) by Anthropic.
