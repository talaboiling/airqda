"""Small helpers: timing, memory measurement, JSON output."""
import json
import time
import tracemalloc
from contextlib import contextmanager
from pathlib import Path

import pandas as pd


@contextmanager
def measure(label: str, results: dict | None = None):
    """Measure wall time (s) and peak Python memory (MB) of a code block."""
    tracemalloc.start()
    t0 = time.perf_counter()
    stats = {}
    try:
        yield stats
    finally:
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        stats.update(time_s=round(elapsed, 4), peak_mem_mb=round(peak / 1024**2, 2))
        if results is not None:
            results[label] = stats
        print(f"  [{label}] {stats['time_s']} s, peak {stats['peak_mem_mb']} MB")


def df_memory_mb(df: pd.DataFrame) -> float:
    return round(df.memory_usage(deep=True).sum() / 1024**2, 2)


def file_size_mb(path: Path) -> float:
    return round(Path(path).stat().st_size / 1024**2, 2)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def load_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None
