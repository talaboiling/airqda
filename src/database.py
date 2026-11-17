"""SQLite database: normalised schema, loading, analytical SQL queries."""
import sqlite3
import time
from pathlib import Path

import pandas as pd

from . import config
from .utils import save_json

SCHEMA = """
DROP TABLE IF EXISTS measurements;
DROP TABLE IF EXISTS stations;
DROP TABLE IF EXISTS districts;
DROP TABLE IF EXISTS wind_directions;
DROP TABLE IF EXISTS seasons;

CREATE TABLE districts (
    district_id   INTEGER PRIMARY KEY,
    district_name TEXT NOT NULL UNIQUE,
    area_type     TEXT NOT NULL CHECK (area_type IN ('urban', 'suburban'))
);
CREATE TABLE stations (
    station_id   INTEGER PRIMARY KEY,
    station_name TEXT NOT NULL UNIQUE,
    district_id  INTEGER NOT NULL REFERENCES districts(district_id)
);
CREATE TABLE wind_directions (
    wd_id   INTEGER PRIMARY KEY,
    wd_code TEXT NOT NULL UNIQUE,
    degrees REAL NOT NULL
);
CREATE TABLE seasons (
    month  INTEGER PRIMARY KEY CHECK (month BETWEEN 1 AND 12),
    season TEXT NOT NULL
);
CREATE TABLE measurements (
    measurement_id INTEGER PRIMARY KEY,
    station_id INTEGER NOT NULL REFERENCES stations(station_id),
    ts    TEXT NOT NULL,          -- 'YYYY-MM-DD HH:MM:SS'
    date  TEXT NOT NULL,          -- 'YYYY-MM-DD'
    year  INTEGER, month INTEGER REFERENCES seasons(month), day INTEGER, hour INTEGER,
    pm25 REAL, pm10 REAL, so2 REAL, no2 REAL, co REAL, o3 REAL,
    temp REAL, pres REAL, dewp REAL, rain REAL,
    wd_id INTEGER REFERENCES wind_directions(wd_id),
    wspm REAL,
    pm25_outlier INTEGER,
    UNIQUE (station_id, ts)
);
"""

INDEXES = """
CREATE INDEX idx_meas_station_ts ON measurements(station_id, ts);
CREATE INDEX idx_meas_date ON measurements(date);
CREATE INDEX idx_meas_year_month ON measurements(year, month);
"""

COLUMN_MAP = {"PM2.5": "pm25", "PM10": "pm10", "SO2": "so2", "NO2": "no2", "CO": "co",
              "O3": "o3", "TEMP": "temp", "PRES": "pres", "DEWP": "dewp", "RAIN": "rain",
              "WSPM": "wspm"}


def get_connection(path: Path = config.DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(path)


def build_database(df: pd.DataFrame, path: Path = config.DB_PATH) -> dict:
    """Create the schema and load the cleaned data (lookup tables + fact table)."""
    t0 = time.perf_counter()
    con = get_connection(path)
    try:
        con.executescript(SCHEMA)

        districts = sorted({(d, a) for d, a in config.STATIONS.values()})
        district_ids = {d: i + 1 for i, (d, _) in enumerate(districts)}
        con.executemany("INSERT INTO districts VALUES (?,?,?)",
                        [(district_ids[d], d, a) for d, a in districts])

        stations = sorted(map(str, df["station"].unique()))
        station_ids = {s: i + 1 for i, s in enumerate(stations)}
        con.executemany("INSERT INTO stations VALUES (?,?,?)",
                        [(station_ids[s], s, district_ids[config.STATIONS[s][0]]) for s in stations])

        wd_ids = {w: i + 1 for i, w in enumerate(config.WIND_DEGREES)}
        con.executemany("INSERT INTO wind_directions VALUES (?,?,?)",
                        [(wd_ids[w], w, deg) for w, deg in config.WIND_DEGREES.items()])
        con.executemany("INSERT INTO seasons VALUES (?,?)", list(config.SEASONS.items()))

        fact = df.rename(columns=COLUMN_MAP)
        fact = pd.DataFrame({
            "station_id": fact["station"].astype(str).map(station_ids),
            "ts": fact["datetime"].dt.strftime("%Y-%m-%d %H:%M:%S"),
            "date": fact["datetime"].dt.strftime("%Y-%m-%d"),
            "year": fact["year"], "month": fact["month"], "day": fact["day"], "hour": fact["hour"],
            **{c: fact[c] for c in COLUMN_MAP.values() if c != "wspm"},
            "wd_id": fact["wd"].astype(str).map(wd_ids),
            "wspm": fact["wspm"],
            "pm25_outlier": fact["pm25_outlier"],
        })
        fact.to_sql("measurements", con, if_exists="append", index=False, chunksize=50_000)
        con.executescript(INDEXES)
        con.commit()
        counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("districts", "stations", "wind_directions", "seasons", "measurements")}
    finally:
        con.close()
    counts["load_time_s"] = round(time.perf_counter() - t0, 2)
    return counts


# ------------------------------------------------------------------ queries
QUERIES: dict[str, dict] = {
    "q1_station_avg_join": {
        "title": "Average pollutant levels per station with district (JOIN + GROUP BY)",
        "sql": """
SELECT s.station_name, d.district_name, d.area_type,
       ROUND(AVG(m.pm25),1) AS avg_pm25, ROUND(AVG(m.pm10),1) AS avg_pm10,
       ROUND(AVG(m.no2),1) AS avg_no2, COUNT(*) AS n_hours
FROM measurements m
JOIN stations s  ON s.station_id = m.station_id
JOIN districts d ON d.district_id = s.district_id
GROUP BY s.station_name, d.district_name, d.area_type
ORDER BY avg_pm25 DESC;"""},

    "q2_winter_having": {
        "title": "Stations whose winter mean PM2.5 exceeds 100 µg/m³ (GROUP BY + HAVING)",
        "sql": """
SELECT s.station_name, ROUND(AVG(m.pm25),1) AS winter_avg_pm25
FROM measurements m
JOIN stations s ON s.station_id = m.station_id
JOIN seasons se ON se.month = m.month
WHERE se.season = 'Winter'
GROUP BY s.station_name
HAVING AVG(m.pm25) > 100
ORDER BY winter_avg_pm25 DESC;"""},

    "q3_above_overall_subquery": {
        "title": "Stations above the city-wide mean PM2.5 (subquery)",
        "sql": """
SELECT s.station_name, ROUND(AVG(m.pm25),1) AS avg_pm25,
       ROUND((SELECT AVG(pm25) FROM measurements),1) AS city_avg
FROM measurements m JOIN stations s ON s.station_id = m.station_id
GROUP BY s.station_name
HAVING AVG(m.pm25) > (SELECT AVG(pm25) FROM measurements)
ORDER BY avg_pm25 DESC;"""},

    "q4_season_area_cte": {
        "title": "Seasonal means by area type (CTE + JOIN)",
        "sql": """
WITH seasonal AS (
    SELECT d.area_type, se.season, AVG(m.pm25) AS pm25, AVG(m.o3) AS o3, AVG(m.so2) AS so2
    FROM measurements m
    JOIN stations s  ON s.station_id = m.station_id
    JOIN districts d ON d.district_id = s.district_id
    JOIN seasons se  ON se.month = m.month
    GROUP BY d.area_type, se.season
)
SELECT area_type, season, ROUND(pm25,1) AS avg_pm25, ROUND(o3,1) AS avg_o3, ROUND(so2,1) AS avg_so2
FROM seasonal ORDER BY area_type, avg_pm25 DESC;"""},

    "q5_yearly_rank_window": {
        "title": "Ranking of stations by PM2.5 per monitoring year Mar–Feb (CTE + RANK window)",
        "sql": """
WITH yearly AS (
    SELECT CASE WHEN m.month >= 3 THEN m.year ELSE m.year - 1 END AS mon_year,
           s.station_name, AVG(m.pm25) AS avg_pm25
    FROM measurements m JOIN stations s ON s.station_id = m.station_id
    GROUP BY mon_year, s.station_name
)
SELECT mon_year || '-' || (mon_year + 1) AS monitoring_year, station_name,
       ROUND(avg_pm25,1) AS avg_pm25,
       RANK() OVER (PARTITION BY mon_year ORDER BY avg_pm25 DESC) AS rank_in_year
FROM yearly ORDER BY mon_year, rank_in_year;"""},

    "q6_rolling_7d_window": {
        "title": "Daily PM2.5 with 7-day moving average per station (window AVG ROWS)",
        "sql": """
WITH daily AS (
    SELECT s.station_name, m.date, AVG(m.pm25) AS pm25
    FROM measurements m JOIN stations s ON s.station_id = m.station_id
    GROUP BY s.station_name, m.date
)
SELECT station_name, date, ROUND(pm25,1) AS daily_pm25,
       ROUND(AVG(pm25) OVER (PARTITION BY station_name ORDER BY date
                             ROWS BETWEEN 6 PRECEDING AND CURRENT ROW),1) AS ma7_pm25
FROM daily ORDER BY station_name, date;"""},

    "q7_monthly_change_lag": {
        "title": "Month-over-month change of city-wide PM2.5 (CTE + LAG window)",
        "sql": """
WITH monthly AS (
    SELECT year, month, AVG(pm25) AS pm25 FROM measurements GROUP BY year, month
)
SELECT year, month, ROUND(pm25,1) AS avg_pm25,
       ROUND(pm25 - LAG(pm25) OVER (ORDER BY year, month),1) AS change_vs_prev,
       ROUND(pm25 - LAG(pm25,12) OVER (ORDER BY year, month),1) AS change_vs_last_year
FROM monthly ORDER BY year, month;"""},

    "q8_wind_direction_join": {
        "title": "PM2.5 and wind speed by wind direction (JOIN + GROUP BY)",
        "sql": """
SELECT w.wd_code, w.degrees, ROUND(AVG(m.pm25),1) AS avg_pm25,
       ROUND(AVG(m.wspm),2) AS avg_wind_speed, COUNT(*) AS n_hours
FROM measurements m JOIN wind_directions w ON w.wd_id = m.wd_id
GROUP BY w.wd_code, w.degrees ORDER BY w.degrees;"""},

    "q9_aqi_category_days": {
        "title": "Station-days per AQI category (China HJ 633-2012 PM2.5 breakpoints) per monitoring year (CTE + CASE)",
        "sql": """
WITH daily AS (
    SELECT station_id, date,
           CASE WHEN month >= 3 THEN year ELSE year - 1 END AS mon_year, AVG(pm25) AS pm25
    FROM measurements GROUP BY station_id, date
)
SELECT mon_year || '-' || (mon_year + 1) AS monitoring_year,
       CASE WHEN pm25 <= 35  THEN '1 Excellent'
            WHEN pm25 <= 75  THEN '2 Good'
            WHEN pm25 <= 115 THEN '3 Light'
            WHEN pm25 <= 150 THEN '4 Moderate'
            WHEN pm25 <= 250 THEN '5 Heavy'
            ELSE '6 Severe' END AS category,
       COUNT(*) AS station_days
FROM daily WHERE pm25 IS NOT NULL
GROUP BY mon_year, category ORDER BY mon_year, category;"""},

    "q10_top_episodes_window": {
        "title": "Top-3 most polluted days per station (CTE + ROW_NUMBER window)",
        "sql": """
WITH daily AS (
    SELECT s.station_name, m.date, AVG(m.pm25) AS pm25, AVG(m.wspm) AS wspm, AVG(m.dewp) AS dewp
    FROM measurements m JOIN stations s ON s.station_id = m.station_id
    GROUP BY s.station_name, m.date
), ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY station_name ORDER BY pm25 DESC) AS rn
    FROM daily
)
SELECT station_name, date, ROUND(pm25,1) AS daily_pm25, ROUND(wspm,2) AS avg_wind, ROUND(dewp,1) AS avg_dewp, rn
FROM ranked WHERE rn <= 3 ORDER BY station_name, rn;"""},
}


def run_query(key: str, path: Path = config.DB_PATH) -> pd.DataFrame:
    """Execute a named query and return the result as a DataFrame."""
    if key not in QUERIES:
        raise KeyError(f"Unknown query '{key}'. Available: {list(QUERIES)}")
    if not Path(path).exists():
        raise FileNotFoundError("Database not found. Build it first (menu step 2).")
    con = get_connection(path)
    try:
        return pd.read_sql_query(QUERIES[key]["sql"], con)
    except sqlite3.Error as e:
        print(f"SQL error in {key}: {e}")
        return pd.DataFrame()
    finally:
        con.close()


def run_all_queries(save: bool = True) -> dict[str, pd.DataFrame]:
    results = {}
    for key in QUERIES:
        t0 = time.perf_counter()
        results[key] = run_query(key)
        print(f"  {key}: {len(results[key])} rows, {time.perf_counter() - t0:.2f} s")
        if save:
            results[key].to_csv(config.TABLE_DIR / f"sql_{key}.csv", index=False)
    return results


def benchmark_pandas_vs_sql(df: pd.DataFrame) -> dict:
    """Same aggregation (monthly mean PM2.5 per station) in Pandas vs SQLite."""
    res = {}
    t0 = time.perf_counter()
    p = (df.groupby(["station", "year", "month"], observed=True)["PM2.5"].mean()
           .reset_index())
    res["pandas_groupby_s"] = round(time.perf_counter() - t0, 4)

    con = get_connection()
    try:
        t0 = time.perf_counter()
        s = pd.read_sql_query("""
            SELECT s.station_name, m.year, m.month, AVG(m.pm25) AS pm25
            FROM measurements m JOIN stations s ON s.station_id = m.station_id
            GROUP BY s.station_name, m.year, m.month""", con)
        res["sqlite_groupby_s"] = round(time.perf_counter() - t0, 4)

        t0 = time.perf_counter()
        _ = pd.read_sql_query("SELECT * FROM measurements", con)
        res["sqlite_full_read_s"] = round(time.perf_counter() - t0, 4)
    finally:
        con.close()
    t0 = time.perf_counter()
    _ = pd.read_parquet(config.PARQUET_PATH)
    res["parquet_full_read_s"] = round(time.perf_counter() - t0, 4)

    merged = p.merge(s, left_on=["station", "year", "month"],
                     right_on=["station_name", "year", "month"])
    res["max_abs_diff"] = float((merged["PM2.5"] - merged["pm25"]).abs().max())
    res["groups"] = len(p)
    save_json(res, config.TABLE_DIR / "benchmark_pandas_vs_sql.json")
    return res


ER_DOT = """digraph ER {
  graph [rankdir=LR, fontname="Helvetica", bgcolor="white", pad=0.3];
  node  [shape=plain, fontname="Helvetica", fontsize=11];
  edge  [arrowhead=crow, arrowtail=tee, dir=both, color="#555555"];
  districts [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
    <TR><TD BGCOLOR="#2F5597"><FONT COLOR="white"><B>districts</B></FONT></TD></TR>
    <TR><TD ALIGN="LEFT">PK district_id INTEGER</TD></TR>
    <TR><TD ALIGN="LEFT">district_name TEXT</TD></TR>
    <TR><TD ALIGN="LEFT">area_type TEXT</TD></TR></TABLE>>];
  stations [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
    <TR><TD BGCOLOR="#2F5597"><FONT COLOR="white"><B>stations</B></FONT></TD></TR>
    <TR><TD ALIGN="LEFT">PK station_id INTEGER</TD></TR>
    <TR><TD ALIGN="LEFT">station_name TEXT</TD></TR>
    <TR><TD ALIGN="LEFT">FK district_id INTEGER</TD></TR></TABLE>>];
  wind_directions [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
    <TR><TD BGCOLOR="#2F5597"><FONT COLOR="white"><B>wind_directions</B></FONT></TD></TR>
    <TR><TD ALIGN="LEFT">PK wd_id INTEGER</TD></TR>
    <TR><TD ALIGN="LEFT">wd_code TEXT</TD></TR>
    <TR><TD ALIGN="LEFT">degrees REAL</TD></TR></TABLE>>];
  seasons [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
    <TR><TD BGCOLOR="#2F5597"><FONT COLOR="white"><B>seasons</B></FONT></TD></TR>
    <TR><TD ALIGN="LEFT">PK month INTEGER</TD></TR>
    <TR><TD ALIGN="LEFT">season TEXT</TD></TR></TABLE>>];
  measurements [label=<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="5">
    <TR><TD BGCOLOR="#C55A11"><FONT COLOR="white"><B>measurements</B></FONT></TD></TR>
    <TR><TD ALIGN="LEFT">PK measurement_id INTEGER</TD></TR>
    <TR><TD ALIGN="LEFT">FK station_id INTEGER</TD></TR>
    <TR><TD ALIGN="LEFT">ts TEXT, date TEXT</TD></TR>
    <TR><TD ALIGN="LEFT">year, FK month, day, hour INTEGER</TD></TR>
    <TR><TD ALIGN="LEFT">pm25, pm10, so2, no2, co, o3 REAL</TD></TR>
    <TR><TD ALIGN="LEFT">temp, pres, dewp, rain, wspm REAL</TD></TR>
    <TR><TD ALIGN="LEFT">FK wd_id INTEGER</TD></TR>
    <TR><TD ALIGN="LEFT">pm25_outlier INTEGER</TD></TR></TABLE>>];
  districts -> stations [label=" 1:N "];
  stations -> measurements [label=" 1:N "];
  wind_directions -> measurements [label=" 1:N "];
  seasons -> measurements [label=" 1:N "];
}"""


def draw_er_diagram() -> Path:
    """Render the ER diagram with Graphviz (falls back to saving the .dot source)."""
    import shutil
    import subprocess
    dot_path = config.ROOT / "docs" / "er_diagram.dot"
    png_path = config.ROOT / "docs" / "er_diagram.png"
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    dot_path.write_text(ER_DOT, encoding="utf-8")
    if shutil.which("dot"):
        subprocess.run(["dot", "-Tpng", "-Gdpi=160", str(dot_path), "-o", str(png_path)], check=True)
        return png_path
    print("Graphviz not installed: saved docs/er_diagram.dot only.")
    return dot_path
