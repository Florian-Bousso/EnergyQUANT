"""
Seasonality analysis for European electricity market prices.

Identifies intra-day and intra-week price patterns through hourly profiles
and hour-by-weekday heatmaps. Electricity markets exhibit strong seasonality:
morning and evening consumption peaks, lower weekend demand, and seasonal
effects driven by temperature and renewable generation.
"""

import pandas as pd


_DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def compute_hourly_profile(prices: pd.Series) -> pd.DataFrame:
    """
    Compute the average intra-day price profile aggregated by hour of day.

    Groups all observations by their hour (0–23) regardless of the day,
    making it suitable for any input frequency (quarter-hourly, hourly).
    Useful for identifying morning and evening consumption peaks.

    Parameters
    ----------
    prices : pd.Series
        Time series of prices with a DatetimeIndex (EUR/MWh).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by hour (0–23) with columns:
        - mean (float) : average price in EUR/MWh
        - std  (float) : standard deviation in EUR/MWh
        - min  (float) : minimum observed price in EUR/MWh
        - max  (float) : maximum observed price in EUR/MWh
    """
    grouped = prices.groupby(prices.index.hour)
    profile = pd.DataFrame({
        "mean": grouped.mean(),
        "std":  grouped.std(),
        "min":  grouped.min(),
        "max":  grouped.max(),
    })
    profile.index.name = "hour"
    return profile


def compute_weekly_heatmap(prices: pd.Series) -> pd.DataFrame:
    """
    Build an hour-by-weekday average price matrix ready for a Plotly heatmap.

    Rows represent hours of the day (0–23), columns represent days of the
    week ordered Monday to Sunday. Each cell contains the mean price in
    EUR/MWh for that hour/day combination over the full series.

    Parameters
    ----------
    prices : pd.Series
        Time series of prices with a DatetimeIndex (EUR/MWh).

    Returns
    -------
    pd.DataFrame
        Pivot DataFrame with shape (24, 7):
        - Index  : hour (int, 0–23)
        - Columns: weekday name (str, Monday → Sunday)
        - Values : mean price in EUR/MWh
    """
    df = pd.DataFrame({
        "price":   prices.values,
        "hour":    prices.index.hour,
        "weekday": prices.index.day_name(),
    })
    pivot = df.pivot_table(values="price", index="hour", columns="weekday", aggfunc="mean")
    # Enforce Monday-to-Sunday column order, keep only days present in the data
    ordered_cols = [d for d in _DAY_ORDER if d in pivot.columns]
    return pivot[ordered_cols]


def seasonality_summary(prices: pd.Series) -> dict:
    """
    Compute a concise summary of intra-day and intra-week price seasonality.

    Parameters
    ----------
    prices : pd.Series
        Time series of prices with a DatetimeIndex (EUR/MWh).

    Returns
    -------
    dict
        Dictionary containing:
        - peak_hour         (int)   : hour with the highest mean price (0–23)
        - offpeak_hour      (int)   : hour with the lowest mean price (0–23)
        - peak_day          (str)   : weekday name with the highest mean price
        - offpeak_day       (str)   : weekday name with the lowest mean price
        - peak_to_mean_ratio (float): ratio of mean peak-hour price to overall mean price.
          More stable than a peak/offpeak ratio since the offpeak hour can have a mean
          price close to zero during solar generation surplus.
    """
    profile = compute_hourly_profile(prices)
    peak_hour    = int(profile["mean"].idxmax())
    offpeak_hour = int(profile["mean"].idxmin())

    daily_means = prices.groupby(prices.index.day_name()).mean()
    peak_day    = daily_means.idxmax()
    offpeak_day = daily_means.idxmin()

    peak_price   = profile.loc[peak_hour, "mean"]
    global_mean  = prices.mean()
    ratio = peak_price / global_mean if global_mean != 0 else None

    return {
        "peak_hour":        peak_hour,
        "offpeak_hour":     offpeak_hour,
        "peak_day":         peak_day,
        "offpeak_day":      offpeak_day,
        "peak_to_mean_ratio": round(ratio, 3) if ratio is not None else None,
    }


if __name__ == "__main__":
    import sys
    from datetime import datetime, timedelta
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.fetcher import fetch_day_ahead_prices

    end = datetime.utcnow()
    start = end - timedelta(days=30)

    print(f"Fetching France day-ahead prices from {start.date()} to {end.date()}...")
    prices = fetch_day_ahead_prices("FR", start, end)["price_eur_mwh"]
    print(f"{len(prices)} data points loaded.\n")

    summary = seasonality_summary(prices)
    print("=" * 45)
    print("Seasonality summary — France (last 30 days)")
    print("=" * 45)
    print(f"  Peak hour        : {summary['peak_hour']:02d}:00")
    print(f"  Off-peak hour    : {summary['offpeak_hour']:02d}:00")
    print(f"  Peak day         : {summary['peak_day']}")
    print(f"  Off-peak day     : {summary['offpeak_day']}")
    print(f"  Peak-to-mean ratio  : {summary['peak_to_mean_ratio']:.3f}")

    profile = compute_hourly_profile(prices)
    print("\nHourly profile (first 6 hours):")
    print(profile.head(6).to_string(float_format=lambda x: f"{x:.2f}"))
