"""Exploratory data analysis: descriptive stats, outliers, correlations, visualisations."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid", context="notebook")
SOURCE = "Source: Beijing Multi-Site Air Quality, UCI ML Repository (2013-03 – 2017-02)"


def _save(fig, name: str) -> str:
    path = config.FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[config.NUMERIC_COLS].describe(percentiles=[0.25, 0.5, 0.75, 0.95]).T
    stats["skew"] = df[config.NUMERIC_COLS].skew()
    stats["missing_pct"] = df[config.NUMERIC_COLS].isna().mean() * 100
    stats = stats.round(2)
    stats.to_csv(config.TABLE_DIR / "descriptive_stats.csv")
    return stats


def station_summary(df: pd.DataFrame) -> pd.DataFrame:
    s = (df.groupby("station", observed=True)[config.POLLUTANTS].mean().round(1)
           .sort_values("PM2.5", ascending=False))
    s["area_type"] = [config.STATIONS[str(i)][1] for i in s.index]
    s.to_csv(config.TABLE_DIR / "station_summary.csv")
    return s


def plot_boxplots(df: pd.DataFrame) -> str:
    """Outlier view (IQR whiskers) for each pollutant; log scale because of heavy right tail."""
    long = df[config.POLLUTANTS].melt(var_name="pollutant", value_name="value").dropna()
    long = long[long["value"] > 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=long, x="pollutant", y="value", ax=ax, color="#4C72B0",
                fliersize=1, flierprops={"alpha": 0.15})
    ax.set_yscale("log")
    ax.set_title("Distribution of pollutant concentrations (IQR whiskers, outliers as dots)")
    ax.set_xlabel("")
    ax.set_ylabel("Concentration, µg/m³ (log scale)")
    return _save(fig, "01_boxplots_pollutants")


def plot_station_boxplot(df: pd.DataFrame) -> str:
    order = (df.groupby("station", observed=True)["PM2.5"].median()
               .sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=df, x="station", y="PM2.5", order=order, hue="season",
                hue_order=["Winter", "Spring", "Summer", "Autumn"], ax=ax,
                showfliers=False, palette="Set2")
    ax.set_title("PM2.5 by monitoring site and season (outliers hidden)")
    ax.set_xlabel("")
    ax.set_ylabel("PM2.5, µg/m³")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Season", ncol=4, loc="upper right")
    return _save(fig, "02_station_season_boxplot")


def correlation(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    corr = df[config.NUMERIC_COLS].corr(method="pearson").round(2)
    corr.to_csv(config.TABLE_DIR / "correlation_pearson.csv")
    spearman = df[config.NUMERIC_COLS].corr(method="spearman").round(2)
    spearman.to_csv(config.TABLE_DIR / "correlation_spearman.csv")
    fig, ax = plt.subplots(figsize=(9, 7.5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation between pollutants and weather variables (hourly)")
    return corr, _save(fig, "03_correlation_heatmap")


def plot_hour_month_heatmap(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(index="hour", columns="month", values="PM2.5", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, cmap="rocket_r", ax=ax, cbar_kws={"label": "Mean PM2.5, µg/m³"})
    ax.set_title("Mean PM2.5 by hour of day and month (all sites)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Hour of day")
    return _save(fig, "04_hour_month_heatmap")


def plot_weather_influence(df: pd.DataFrame) -> str:
    """PM2.5 against binned weather variables (median + IQR band)."""
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
    for ax, (col, label) in zip(axes, [("WSPM", "Wind speed, m/s"), ("DEWP", "Dew point, °C"),
                                       ("TEMP", "Temperature, °C"), ("PRES", "Pressure, hPa")]):
        d = df[[col, "PM2.5"]].dropna()
        d["bin"] = pd.qcut(d[col], 20, duplicates="drop")
        g = d.groupby("bin", observed=True).agg(x=(col, "median"),
                                                 med=("PM2.5", "median"),
                                                 q1=("PM2.5", lambda s: s.quantile(.25)),
                                                 q3=("PM2.5", lambda s: s.quantile(.75)))
        ax.fill_between(g["x"], g["q1"], g["q3"], alpha=0.25, color="#4C72B0")
        ax.plot(g["x"], g["med"], marker="o", ms=3, color="#4C72B0")
        ax.set_xlabel(label)
    axes[0].set_ylabel("PM2.5, µg/m³ (median, IQR band)")
    fig.suptitle("Weather influence on PM2.5 (20 quantile bins)")
    return _save(fig, "05_weather_influence")


def interactive_trend_fig(df: pd.DataFrame):
    """Interactive Plotly chart: monthly mean PM2.5 per site."""
    monthly = (df.set_index("datetime").groupby("station", observed=True)["PM2.5"]
                 .resample("MS").mean().reset_index())
    monthly["station"] = monthly["station"].astype(str)
    fig = px.line(monthly, x="datetime", y="PM2.5", color="station",
                  title="Monthly mean PM2.5 by monitoring site (click legend to filter)",
                  labels={"datetime": "Month", "PM2.5": "PM2.5, µg/m³", "station": "Site"})
    fig.update_layout(template="plotly_white", hovermode="x unified")
    return fig


def plot_interactive_trend(df: pd.DataFrame) -> str:
    fig = interactive_trend_fig(df)
    path = config.FIG_DIR / "06_interactive_monthly_trend.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return str(path)


def interactive_wind_fig(df: pd.DataFrame):
    """Interactive polar chart: mean PM2.5 by wind direction."""
    g = df.groupby("wd", observed=True)["PM2.5"].mean().reindex(list(config.WIND_DEGREES)).reset_index()
    fig = px.bar_polar(g, r="PM2.5", theta="wd", color="PM2.5", color_continuous_scale="OrRd",
                       title="Mean PM2.5 by wind direction (wind blowing FROM)")
    fig.update_layout(template="plotly_white")
    return fig


def plot_interactive_wind(df: pd.DataFrame) -> str:
    fig = interactive_wind_fig(df)
    path = config.FIG_DIR / "07_interactive_wind_polar.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return str(path)


def run_eda(df: pd.DataFrame) -> dict:
    config.ensure_dirs()
    out = {"descriptive": descriptive_stats(df), "stations": station_summary(df)}
    corr, _ = correlation(df)
    out["corr"] = corr
    out["figures"] = [plot_boxplots(df), plot_station_boxplot(df), plot_hour_month_heatmap(df),
                      plot_weather_influence(df), plot_interactive_trend(df),
                      plot_interactive_wind(df)]
    return out
