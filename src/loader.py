"""Loading raw CSVs in chunks with dtype optimisation; CSV vs Parquet benchmark."""
import zipfile
from pathlib import Path

import pandas as pd

from . import config
from .utils import df_memory_mb, file_size_mb, measure, save_json


def extract_archives(raw_dir: Path = config.RAW_DIR) -> int:
    """Unpack any .zip found in data/raw (also nested archives). Returns number unpacked."""
    done, n = set(), 0
    while True:
        zips = [z for z in raw_dir.rglob("*.zip") if z not in done]
        if not zips:
            return n
        for z in zips:
            try:
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(z.parent / z.stem)
                n += 1
            except zipfile.BadZipFile:
                print(f"Skipping corrupt archive: {z.name}")
            done.add(z)


def find_csv_files(raw_dir: Path = config.RAW_DIR) -> list[Path]:
    files = sorted(raw_dir.rglob(config.CSV_PATTERN))
    if not files and extract_archives(raw_dir):
        files = sorted(raw_dir.rglob(config.CSV_PATTERN))
    if not files:
        raise FileNotFoundError(
            f"No files matching {config.CSV_PATTERN} in {raw_dir}. "
            "Download the dataset and unpack it into data/raw/ (see README)."
        )
    return files


def read_csv_chunked(path: Path, dtypes: dict | None = None,
                     chunksize: int = config.CHUNK_SIZE) -> pd.DataFrame:
    """Read one station file chunk by chunk to keep peak memory bounded."""
    chunks = []
    for chunk in pd.read_csv(path, dtype=dtypes, chunksize=chunksize):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def load_raw(optimized: bool = True) -> pd.DataFrame:
    """Load all station files into one DataFrame."""
    dtypes = config.DTYPES if optimized else None
    frames = [read_csv_chunked(p, dtypes) for p in find_csv_files()]
    df = pd.concat(frames, ignore_index=True)
    if optimized:
        # concat of per-file categoricals with different categories -> re-categorise
        for col in ("station", "wd"):
            df[col] = df[col].astype("category")
    return df


def benchmark_loading() -> dict:
    """Compare naive vs optimised dtypes, and CSV vs Parquet (time, memory, size)."""
    config.ensure_dirs()
    res: dict = {}
    csv_files = find_csv_files()
    res["n_files"] = len(csv_files)
    res["csv_total_size_mb"] = round(sum(file_size_mb(p) for p in csv_files), 2)

    print("Loading CSV with default dtypes...")
    with measure("csv_default_dtypes", res):
        df_naive = load_raw(optimized=False)
    res["rows"] = len(df_naive)
    res["mem_default_mb"] = df_memory_mb(df_naive)
    del df_naive

    print("Loading CSV in chunks with optimised dtypes...")
    with measure("csv_chunked_optimized", res):
        df = load_raw(optimized=True)
    res["mem_optimized_mb"] = df_memory_mb(df)
    res["mem_reduction_pct"] = round(100 * (1 - res["mem_optimized_mb"] / res["mem_default_mb"]), 1)

    with measure("parquet_write", res):
        df.to_parquet(config.RAW_PARQUET_PATH, index=False, compression="snappy")
    res["parquet_size_mb"] = file_size_mb(config.RAW_PARQUET_PATH)

    print("Reading Parquet...")
    with measure("parquet_read", res):
        _ = pd.read_parquet(config.RAW_PARQUET_PATH)
    with measure("parquet_read_2cols", res):
        _ = pd.read_parquet(config.RAW_PARQUET_PATH, columns=["station", "PM2.5"])

    save_json(res, config.TABLE_DIR / "benchmark_loading.json")
    return res
