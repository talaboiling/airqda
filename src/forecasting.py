"""Next-day PM2.5 forecasting: baselines vs regression models on a time-based split."""
import time
import tracemalloc

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .eda import _save
from .utils import save_json

WEATHER_LAGGED = ["TEMP", "DEWP", "PRES", "WSPM", "RAIN"]


def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
    """Daily panel (station x day). Every feature uses information up to day t-1 only."""
    daily = (df.set_index("datetime").groupby("station", observed=True)[["PM2.5"] + WEATHER_LAGGED]
               .resample("D").mean().reset_index())
    daily["station"] = daily["station"].astype(str)
    g = daily.groupby("station")["PM2.5"]
    for lag in (1, 2, 3, 7):
        daily[f"lag{lag}"] = g.shift(lag)
    daily["ma7"] = g.transform(lambda s: s.shift(1).rolling(7, min_periods=5).mean())
    daily["ma30"] = g.transform(lambda s: s.shift(1).rolling(30, min_periods=20).mean())
    for c in WEATHER_LAGGED:
        daily[f"{c}_lag1"] = daily.groupby("station")[c].shift(1)
    doy = daily["datetime"].dt.dayofyear
    daily["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    daily["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    daily["weekend"] = (daily["datetime"].dt.dayofweek >= 5).astype(int)
    daily = pd.get_dummies(daily, columns=["station"], prefix="st", dtype=int).assign(
        station=daily["station"])
    # identical rows for every approach -> fair comparison
    return daily.dropna(subset=["PM2.5", "lag1", "lag2", "lag3", "lag7", "ma7", "ma30"]
                        + [f"{c}_lag1" for c in WEATHER_LAGGED])


def feature_columns(daily: pd.DataFrame) -> list[str]:
    base = ["lag1", "lag2", "lag3", "lag7", "ma7", "ma30", "doy_sin", "doy_cos", "weekend"]
    return base + [f"{c}_lag1" for c in WEATHER_LAGGED] + [c for c in daily.columns if c.startswith("st_")]


def split(daily: pd.DataFrame):
    train = daily[daily["datetime"] < config.TEST_START]
    test = daily[daily["datetime"] >= config.TEST_START]
    return train, test


def _metrics(y, pred) -> dict:
    return {"MAE": round(float(mean_absolute_error(y, pred)), 2),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, pred))), 2)}


def _timed(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    t = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, round(t, 3), round(peak / 1024**2, 2)


def evaluate(df: pd.DataFrame) -> dict:
    config.ensure_dirs()
    daily = build_daily_features(df)
    train, test = split(daily)
    feats = feature_columns(daily)
    y_test = test["PM2.5"].values

    approaches = {
        "Naive (y[t-1])": ("baseline", lambda: test["lag1"].values),
        "Seasonal naive (y[t-7])": ("baseline", lambda: test["lag7"].values),
        "Moving average (7 days)": ("simple", lambda: test["ma7"].values),
    }

    def fit_lr():
        m = make_pipeline(StandardScaler(), LinearRegression())  # scaler fit on train only
        m.fit(train[feats], train["PM2.5"])
        return m.predict(test[feats])

    def fit_hgb():
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
        m.fit(train[feats], train["PM2.5"])
        return m.predict(test[feats])

    approaches["Linear Regression (lags + weather t-1)"] = ("model", fit_lr)
    approaches["Gradient Boosting (extension)"] = ("model", fit_hgb)

    rows, preds = [], {}
    for name, (kind, fn) in approaches.items():
        p, t, mem = _timed(fn)
        preds[name] = p
        rows.append({"approach": name, "type": kind, **_metrics(y_test, p),
                     "time_s": t, "peak_mem_mb": mem})
    res = pd.DataFrame(rows)
    naive_mae = res.loc[res["approach"].str.startswith("Naive"), "MAE"].iloc[0]
    res["skill_vs_naive_%"] = (100 * (1 - res["MAE"] / naive_mae)).round(1)
    res.to_csv(config.TABLE_DIR / "forecast_comparison.csv", index=False)

    # per-station MAE
    per_st = []
    for st, g in test.assign(**{k: v for k, v in preds.items()}).groupby("station"):
        row = {"station": st}
        for name in preds:
            row[name] = round(float(mean_absolute_error(g["PM2.5"], g[name])), 2)
        per_st.append(row)
    per_st = pd.DataFrame(per_st)
    per_st.to_csv(config.TABLE_DIR / "forecast_mae_by_station.csv", index=False)

    meta = {"train_rows": len(train), "test_rows": len(test),
            "train_period": [str(train["datetime"].min().date()), str(train["datetime"].max().date())],
            "test_period": [str(test["datetime"].min().date()), str(test["datetime"].max().date())],
            "n_features": len(feats)}
    save_json(meta, config.TABLE_DIR / "forecast_split.json")
    fig_paths = [plot_comparison(res), plot_forecast_example(test, preds)]
    return {"results": res, "per_station": per_st, "meta": meta, "figures": fig_paths}


def plot_comparison(res: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = {"baseline": "#A5A5A5", "simple": "#70AD47", "model": "#4472C4"}
    r = res.sort_values("MAE")
    bars = ax.barh(r["approach"], r["MAE"], color=[colors[t] for t in r["type"]])
    ax.bar_label(bars, fmt="%.1f", padding=3)
    ax.invert_yaxis()
    ax.set_xlabel("MAE on test period, µg/m³ (lower is better)")
    ax.set_title("Next-day PM2.5 forecast: approaches vs baselines")
    return _save(fig, "12_forecast_mae_comparison")


def plot_forecast_example(test: pd.DataFrame, preds: dict, station: str = "Dongsi") -> str:
    if station not in set(test["station"]):
        station = test["station"].iloc[0]
    mask = (test["station"] == station).values
    sub = test[mask]
    window = slice(-120, None)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(sub["datetime"].iloc[window], sub["PM2.5"].iloc[window], color="black", lw=2, label="Actual")
    for name in ("Naive (y[t-1])", "Moving average (7 days)", "Linear Regression (lags + weather t-1)"):
        ax.plot(sub["datetime"].iloc[window], preds[name][mask][window], lw=1.2, label=name)
    ax.set_title(f"Next-day PM2.5 forecasts, {station}, last 120 test days")
    ax.set_ylabel("PM2.5, µg/m³")
    ax.legend(loc="upper left", fontsize=8)
    return _save(fig, "13_forecast_example")
