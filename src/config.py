"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
TABLE_DIR = ROOT / "reports" / "tables"

PARQUET_PATH = PROCESSED_DIR / "air_quality_clean.parquet"
RAW_PARQUET_PATH = PROCESSED_DIR / "air_quality_raw.parquet"
DB_PATH = PROCESSED_DIR / "air_quality.db"
AUDIT_PATH = TABLE_DIR / "data_quality_audit.json"

CSV_PATTERN = "PRSA_Data_*.csv"
CHUNK_SIZE = 50_000

POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]
NUMERIC_COLS = POLLUTANTS + WEATHER

# Optimised dtypes used while reading the raw CSVs in chunks
DTYPES = {
    "No": "int32", "year": "int16", "month": "int8", "day": "int8", "hour": "int8",
    **{c: "float32" for c in NUMERIC_COLS},
    "wd": "category", "station": "category",
}

# Station metadata (district and area type) used as the "region" dimension
STATIONS = {
    "Aotizhongxin":   ("Chaoyang",     "urban"),
    "Changping":      ("Changping",    "suburban"),
    "Dingling":       ("Changping",    "suburban"),
    "Dongsi":         ("Dongcheng",    "urban"),
    "Guanyuan":       ("Xicheng",      "urban"),
    "Gucheng":        ("Shijingshan",  "urban"),
    "Huairou":        ("Huairou",      "suburban"),
    "Nongzhanguan":   ("Chaoyang",     "urban"),
    "Shunyi":         ("Shunyi",       "suburban"),
    "Tiantan":        ("Dongcheng",    "urban"),
    "Wanliu":         ("Haidian",      "urban"),
    "Wanshouxigong":  ("Xicheng",      "urban"),
}

WIND_DEGREES = {
    "N": 0.0, "NNE": 22.5, "NE": 45.0, "ENE": 67.5, "E": 90.0, "ESE": 112.5,
    "SE": 135.0, "SSE": 157.5, "S": 180.0, "SSW": 202.5, "SW": 225.0, "WSW": 247.5,
    "W": 270.0, "WNW": 292.5, "NW": 315.0, "NNW": 337.5,
}

SEASONS = {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
           6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}

# Forecast evaluation: time-based split (no shuffling -> no leakage)
TEST_START = "2016-03-01"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
