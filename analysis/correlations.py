"""
Cross-commodity correlation analysis for European energy markets.

Measures the statistical relationships between French day-ahead electricity
prices, TTF natural gas prices and EU carbon allowances (EUA). Strong
positive correlations between power and gas are expected given the marginal
role of gas-fired plants in setting the electricity price.
"""

import warnings
from datetime import datetime

import pandas as pd
import yfinance as yf


# Yahoo Finance tickers for European energy commodities
_TICKERS = {
    "gas_ttf":    "TTF=F",
    "carbon_eua": "CO2.L",
}


def fetch_commodity_prices(start: datetime, end: datetime) -> pd.DataFrame:
    """
    Download daily closing prices for TTF gas and EU carbon allowances (EUA).

    Uses Yahoo Finance via yfinance. Tickers may occasionally be unavailable
    or return empty data; such columns are silently dropped and a warning is
    issued so the rest of the pipeline can proceed.

    Parameters
    ----------
    start : datetime
        Start date of the download window.
    end : datetime
        End date of the download window.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex (daily) and a subset of columns:
        - gas_ttf    (float) : TTF front-month futures closing price (EUR/MWh)
        - carbon_eua (float) : EUA futures closing price (EUR/tCO2)
        Only columns with at least one valid value are included.
    """
    tickers = list(_TICKERS.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        return pd.DataFrame()

    # yfinance returns a MultiIndex when multiple tickers are requested
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Rename ticker symbols to human-readable column names
    reverse = {v: k for k, v in _TICKERS.items()}
    close = close.rename(columns=reverse)

    # Drop columns with no usable data and warn
    available = [col for col in _TICKERS if col in close.columns and close[col].notna().any()]
    missing = [col for col in _TICKERS if col not in available]
    for col in missing:
        warnings.warn(f"No data retrieved for '{col}' ({_TICKERS[col]}). Column excluded.", UserWarning)

    result = close[available].dropna(how="all")
    result.index = pd.to_datetime(result.index).tz_localize(None)
    return result


def build_commodity_dataframe(
    power_prices: pd.Series,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Build a joint daily DataFrame with power, gas and carbon prices.

    Aggregates intra-day electricity prices to daily averages, then joins
    them with commodity prices from Yahoo Finance on the common trading days
    (inner join). Rows with any remaining NaN are dropped.

    Parameters
    ----------
    power_prices : pd.Series
        Intra-day electricity prices with a DatetimeIndex (EUR/MWh).
    start : datetime
        Start date passed to fetch_commodity_prices.
    end : datetime
        End date passed to fetch_commodity_prices.

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with columns "power", "gas_ttf", "carbon_eua",
        indexed by date (daily). Only dates present in all series are kept.
    """
    daily_power = power_prices.resample("D").mean().dropna()
    daily_power.index = pd.to_datetime(daily_power.index).tz_localize(None)
    daily_power.name = "power"

    commodities = fetch_commodity_prices(start, end)

    df = daily_power.to_frame().join(commodities, how="inner")
    df = df.dropna()
    return df


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix for a commodity DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with numeric columns (e.g. power, gas_ttf, carbon_eua).

    Returns
    -------
    pd.DataFrame
        Square correlation matrix with values in [-1, 1].
    """
    return df.corr(method="pearson")


def correlation_summary(corr_matrix: pd.DataFrame) -> dict:
    """
    Extract the key pairwise correlations from a commodity correlation matrix.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square correlation matrix as returned by compute_correlation_matrix().

    Returns
    -------
    dict
        Dictionary containing:
        - power_gas    (float or None) : correlation between power and gas prices
        - power_carbon (float or None) : correlation between power and carbon prices
        - gas_carbon   (float or None) : correlation between gas and carbon prices
    """
    def _get(a, b):
        try:
            return round(corr_matrix.loc[a, b], 3)
        except KeyError:
            return None

    return {
        "power_gas":    _get("power", "gas_ttf"),
        "power_carbon": _get("power", "carbon_eua"),
        "gas_carbon":   _get("gas_ttf", "carbon_eua"),
    }


if __name__ == "__main__":
    import sys
    from datetime import timedelta
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.fetcher import fetch_day_ahead_prices

    end = datetime.utcnow()
    start = end - timedelta(days=90)

    print(f"Fetching France day-ahead prices from {start.date()} to {end.date()}...")
    power_prices = fetch_day_ahead_prices("FR", start, end)["price_eur_mwh"]
    print(f"{len(power_prices)} intra-day points loaded.")

    print("\nBuilding commodity DataFrame (power + gas + carbon)...")
    df = build_commodity_dataframe(power_prices, start, end)
    print(f"{len(df)} common trading days available.\n")
    print(df.describe().round(2))

    corr = compute_correlation_matrix(df)
    print("\nCorrelation matrix:")
    print(corr.round(3).to_string())

    summary = correlation_summary(corr)
    print("\n" + "=" * 40)
    print("Correlation summary")
    print("=" * 40)
    print(f"  Power / Gas (TTF)  : {summary['power_gas']}")
    print(f"  Power / Carbon EUA : {summary['power_carbon']}")
    print(f"  Gas   / Carbon EUA : {summary['gas_carbon']}")
