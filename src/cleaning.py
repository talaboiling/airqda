"""Data quality audit and cleaning."""
import numpy as np
import pandas as pd

from . import config
from .utils import save_json

MAX_GAP_HOURS = 6  # interpolate only short gaps; longer gaps stay missing
# hourly values above these limits are treated as sensor errors (isolated spikes)
PLAUSIBLE_MAX = {"O3": 500.0}


def iqr_bounds(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def audit(df: pd.DataFrame) -> dict:
    """Full data quality report: types, missing values, duplicates, outliers, ranges."""
    report = {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "stations": sorted(map(str, df["station"].unique())),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "feature_types": {
            "numeric_continuous": config.NUMERIC_COLS,
            "categorical": ["station", "wd"],
            "temporal": ["year", "month", "day", "hour"],
        },
        "missing": {},
        "duplicates_full_rows": int(df.duplicated().sum()),
        "duplicates_station_time": int(df.duplicated(["station", "year", "month", "day", "hour"]).sum()),
        "outliers_iqr": {},
        "negative_values": {},
    }
    for c in df.columns:
        n = int(df[c].isna().sum())
        if n:
            report["missing"][c] = {"count": n, "pct": round(100 * n / len(df), 2)}
    for c in config.NUMERIC_COLS:
        s = df[c].dropna()
        lo, hi = iqr_bounds(s)
        n_out = int(((s < lo) | (s > hi)).sum())
        report["outliers_iqr"][c] = {"lower": round(lo, 2), "upper": round(hi, 2),
                                     "count": n_out, "pct": round(100 * n_out / len(s), 2),
                                     "min": float(s.min()), "max": float(s.max())}
        if c not in ("TEMP", "DEWP"):  # temperatures can legitimately be negative
            neg = int((s < 0).sum())
            if neg:
                report["negative_values"][c] = neg
    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Build datetime, remove duplicates/invalid values, fill short gaps per station."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.drop(columns=["No"], errors="ignore")
    df = df.drop_duplicates(["station", "datetime"]).sort_values(["station", "datetime"])

    # physically impossible negatives -> NaN
    for c in config.POLLUTANTS + ["RAIN", "WSPM"]:
        df.loc[df[c] < 0, c] = np.nan
    for c, limit in PLAUSIBLE_MAX.items():
        df.loc[df[c] > limit, c] = np.nan

    # IQR outliers are FLAGGED, not removed: PM spikes are real haze episodes
    lo, hi = iqr_bounds(df["PM2.5"].dropna())
    df["pm25_outlier"] = (df["PM2.5"] > hi).astype("int8")

    # time interpolation of short gaps inside each station
    parts = []
    for _, g in df.groupby("station", observed=True):
        g = g.set_index("datetime")
        g[config.NUMERIC_COLS] = g[config.NUMERIC_COLS].interpolate(
            method="time", limit=MAX_GAP_HOURS, limit_area="inside")
        g["wd"] = g["wd"].ffill(limit=MAX_GAP_HOURS)
        parts.append(g.reset_index())
    df = pd.concat(parts, ignore_index=True)

    df["season"] = df["month"].map(config.SEASONS).astype("category")
    df["station"] = df["station"].astype("category")
    df["wd"] = df["wd"].astype("category")
    for c in config.NUMERIC_COLS:
        df[c] = df[c].astype("float32")
    return df


def run_audit_and_clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = audit(raw)
    cleaned = clean(raw)
    after_missing = {c: int(cleaned[c].isna().sum()) for c in config.NUMERIC_COLS + ["wd"]}
    report = {"before": before,
              "after_cleaning_missing": after_missing,
              "rows_after": int(len(cleaned)),
              "rule_interpolation_max_gap_h": MAX_GAP_HOURS}
    save_json(report, config.AUDIT_PATH)
    cleaned.to_parquet(config.PARQUET_PATH, index=False)
    return cleaned, report


def load_clean() -> pd.DataFrame:
    if not config.PARQUET_PATH.exists():
        raise FileNotFoundError("Clean data not found. Run step 1 (load & clean) first.")
    return pd.read_parquet(config.PARQUET_PATH)
