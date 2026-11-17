"""Streamlit mini-dashboard. Run: streamlit run dashboard.py"""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config, database

st.set_page_config(page_title="Beijing Air Quality", layout="wide")


@st.cache_data
def load() -> pd.DataFrame:
    df = pd.read_parquet(config.PARQUET_PATH)
    df["station"] = df["station"].astype(str)
    df["district"] = df["station"].map(lambda s: config.STATIONS[s][0])
    df["area_type"] = df["station"].map(lambda s: config.STATIONS[s][1])
    return df


try:
    df = load()
except FileNotFoundError:
    st.error("Clean data not found. Run `python main.py --all` first.")
    st.stop()

# ---------------- sidebar filters
st.sidebar.header("Filters")
years = sorted(df["year"].unique())
year_sel = st.sidebar.multiselect("Year", years, default=years)
area_sel = st.sidebar.multiselect("Region (area type)", ["urban", "suburban"], default=["urban", "suburban"])
stations_avail = sorted(df.loc[df["area_type"].isin(area_sel), "station"].unique())
station_sel = st.sidebar.multiselect("Monitoring site", stations_avail, default=stations_avail)
pollutant = st.sidebar.selectbox("Category (pollutant)", config.POLLUTANTS, index=0)
freq_label = st.sidebar.radio("Resample", ["Daily", "Weekly", "Monthly"], index=2)
freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq_label]

f = df[df["year"].isin(year_sel) & df["station"].isin(station_sel)]
if f.empty:
    st.warning("No data for the selected filters.")
    st.stop()

st.title("Air quality across Beijing monitoring sites")
st.caption("Beijing Multi-Site Air Quality dataset (UCI), hourly, 12 sites, 2013-03 – 2017-02")

# ---------------- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Mean {pollutant}", f"{f[pollutant].mean():.1f}")
c2.metric(f"Median {pollutant}", f"{f[pollutant].median():.1f}")
daily = f.groupby(["station", f["datetime"].dt.date])["PM2.5"].mean()
c3.metric("Station-days PM2.5 > 75", f"{(daily > 75).mean() * 100:.1f}%")
c4.metric("Hourly records", f"{len(f):,}")

# ---------------- charts
ts = (f.set_index("datetime").groupby("station")[pollutant].resample(freq).mean().reset_index())
st.plotly_chart(px.line(ts, x="datetime", y=pollutant, color="station",
                        title=f"{freq_label} mean {pollutant} by site",
                        labels={"datetime": "", pollutant: f"{pollutant}, µg/m³"}),
                width="stretch")

left, right = st.columns(2)
by_st = f.groupby(["station", "area_type"])[pollutant].mean().reset_index().sort_values(pollutant)
left.plotly_chart(px.bar(by_st, x=pollutant, y="station", color="area_type", orientation="h",
                         title=f"Mean {pollutant} by site"), width="stretch")
season = f.groupby(["season", "hour"], observed=True)[pollutant].mean().reset_index()
right.plotly_chart(px.line(season, x="hour", y=pollutant, color="season",
                           title=f"Diurnal profile of {pollutant} by season"), width="stretch")

left, right = st.columns(2)
corr = f[config.NUMERIC_COLS].corr().round(2)
left.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                            title="Correlation matrix"), width="stretch")
wd = f.groupby("wd", observed=True)[pollutant].mean().reindex(list(config.WIND_DEGREES)).reset_index()
right.plotly_chart(px.bar_polar(wd, r=pollutant, theta="wd", color=pollutant,
                                color_continuous_scale="OrRd",
                                title=f"{pollutant} by wind direction"), width="stretch")

# ---------------- SQL and forecast results
with st.expander("SQL queries (SQLite)"):
    key = st.selectbox("Query", list(database.QUERIES),
                       format_func=lambda k: database.QUERIES[k]["title"])
    st.code(database.QUERIES[key]["sql"], language="sql")
    try:
        st.dataframe(database.run_query(key), width="stretch")
    except FileNotFoundError as e:
        st.error(str(e))

with st.expander("Forecast comparison (next-day PM2.5, test from 2016-03-01)"):
    path = config.TABLE_DIR / "forecast_comparison.csv"
    if path.exists():
        st.dataframe(pd.read_csv(path), width="stretch")
    else:
        st.info("Run the forecast step first.")
