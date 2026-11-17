"""High-level steps used by both the console menu and the full pipeline run."""
import pandas as pd

from . import cleaning, config, database, eda, forecasting, loader, timeseries
from .utils import save_json


def step_load_and_clean() -> pd.DataFrame:
    print("\n[1] Loading raw data (chunked, optimised dtypes) + CSV vs Parquet benchmark")
    bench = loader.benchmark_loading()
    print(f"  rows={bench['rows']:,}  memory: {bench['mem_default_mb']} MB -> "
          f"{bench['mem_optimized_mb']} MB (-{bench['mem_reduction_pct']}%)  "
          f"CSV {bench['csv_total_size_mb']} MB vs Parquet {bench['parquet_size_mb']} MB")
    raw = pd.read_parquet(config.RAW_PARQUET_PATH)
    print("  Running data quality audit and cleaning...")
    df, report = cleaning.run_audit_and_clean(raw)
    miss = report["before"]["missing"]
    print(f"  missing before: {sum(v['count'] for v in miss.values()):,} cells; "
          f"after: {sum(report['after_cleaning_missing'].values()):,}")
    return df


def step_build_db(df: pd.DataFrame) -> None:
    print("\n[2] Building SQLite database")
    counts = database.build_database(df)
    print(f"  {counts}")
    print(f"  ER diagram: {database.draw_er_diagram()}")
    bench = database.benchmark_pandas_vs_sql(df)
    print(f"  Pandas vs SQL: {bench}")


def step_sql() -> dict:
    print("\n[3] Running analytical SQL queries")
    return database.run_all_queries()


def step_eda(df: pd.DataFrame) -> dict:
    print("\n[4] EDA and visualisations")
    out = eda.run_eda(df)
    print(f"  {len(out['figures'])} figures saved to {config.FIG_DIR}")
    return out


def step_timeseries(df: pd.DataFrame) -> dict:
    print("\n[5] Time-series analysis")
    out = timeseries.run_timeseries(df)
    print(f"  {out['summary']}")
    return out


def step_forecast(df: pd.DataFrame) -> dict:
    print("\n[6] Forecast comparison (time-based split, test from %s)" % config.TEST_START)
    out = forecasting.evaluate(df)
    print(out["results"].to_string(index=False))
    return out


def run_all() -> None:
    config.ensure_dirs()
    df = step_load_and_clean()
    step_build_db(df)
    step_sql()
    step_eda(df)
    step_timeseries(df)
    step_forecast(df)
    save_json({"status": "ok", "rows": len(df)}, config.TABLE_DIR / "run_status.json")
    print("\nAll steps finished. Results: reports/figures, reports/tables. Dashboard: streamlit run dashboard.py")
