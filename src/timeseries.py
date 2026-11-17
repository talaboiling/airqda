"""Time-series analysis: resampling, rolling means, seasonal decomposition."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

from . import config
from .eda import _save
from .utils import save_json


def city_series(df: pd.DataFrame, col: str = "PM2.5", station: str | None = None) -> pd.Series:
    """Hourly series averaged across sites (or one site)."""
    d = df if station is None else df[df["station"] == station]
    return d.groupby("datetime")[col].mean().asfreq("h")


def resample_levels(hourly: pd.Series) -> dict[str, pd.Series]:
    return {"daily": hourly.resample("D").mean(),
            "weekly": hourly.resample("W").mean(),
            "monthly": hourly.resample("MS").mean()}


def plot_rolling(daily: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(daily.index, daily, lw=0.6, alpha=0.45, label="Daily mean", color="grey")
    ax.plot(daily.index, daily.rolling(7, min_periods=4).mean(), lw=1.2, label="7-day MA")
    ax.plot(daily.index, daily.rolling(30, min_periods=15).mean(), lw=2, label="30-day MA")
    ax.plot(daily.index, daily.rolling(365, min_periods=180).mean(), lw=2.2, ls="--",
            color="black", label="365-day MA (trend)")
    ax.set_title("City-wide PM2.5: daily values and moving averages")
    ax.set_ylabel("PM2.5, µg/m³")
    ax.legend(ncol=4, loc="upper right")
    return _save(fig, "08_rolling_means")


def seasonal_strength(res) -> float:
    """Hyndman's strength of seasonality: 1 - Var(R) / Var(S + R)."""
    r, s = res.resid.dropna(), res.seasonal.loc[res.resid.dropna().index]
    return float(max(0.0, 1 - np.var(r) / np.var(s + r)))


def decompose(series: pd.Series, period: int, name: str, title: str) -> tuple[dict, str]:
    s = series.interpolate(limit_direction="both")
    res = seasonal_decompose(s, model="additive", period=period)
    fig = res.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(title, y=1.01)
    path = _save(fig, name)
    seasonal_profile = res.seasonal.iloc[:period].round(2)
    info = {"period": period, "seasonal_strength": round(seasonal_strength(res), 3),
            "seasonal_amplitude": round(float(seasonal_profile.max() - seasonal_profile.min()), 2),
            "trend_start": round(float(res.trend.dropna().iloc[0]), 2),
            "trend_end": round(float(res.trend.dropna().iloc[-1]), 2)}
    return info, path


def station_decomposition_table(df: pd.DataFrame) -> pd.DataFrame:
    """Seasonal strength and amplitude for each site (monthly, period 12)."""
    rows = []
    for st in sorted(map(str, df["station"].unique())):
        m = city_series(df, station=st).resample("MS").mean().interpolate(limit_direction="both")
        res = seasonal_decompose(m, model="additive", period=12)
        prof = res.seasonal.iloc[:12]
        rows.append({"station": st, "area_type": config.STATIONS[st][1],
                     "mean_pm25": round(float(m.mean()), 1),
                     "seasonal_strength": round(seasonal_strength(res), 3),
                     "amplitude": round(float(prof.max() - prof.min()), 1),
                     "peak_month": int(prof.idxmax().month),
                     "trend_change": round(float(res.trend.dropna().iloc[-1] - res.trend.dropna().iloc[0]), 1)})
    t = pd.DataFrame(rows).sort_values("mean_pm25", ascending=False)
    t.to_csv(config.TABLE_DIR / "station_decomposition.csv", index=False)
    return t


def run_timeseries(df: pd.DataFrame) -> dict:
    config.ensure_dirs()
    hourly = city_series(df)
    levels = resample_levels(hourly)
    figs = [plot_rolling(levels["daily"])]
    summary = {}
    summary["monthly_period12"], p = decompose(levels["monthly"], 12, "09_decomposition_monthly",
                                               "Seasonal decomposition of monthly PM2.5 (period = 12)")
    figs.append(p)
    summary["daily_period7"], p = decompose(levels["daily"], 7, "10_decomposition_weekly",
                                            "Seasonal decomposition of daily PM2.5 (period = 7)")
    figs.append(p)
    summary["hourly_period24"], _ = decompose(hourly, 24, "11_decomposition_diurnal",
                                              "Seasonal decomposition of hourly PM2.5 (period = 24)")
    save_json(summary, config.TABLE_DIR / "decomposition_summary.json")
    stations = station_decomposition_table(df)
    return {"summary": summary, "stations": stations, "figures": figs}
