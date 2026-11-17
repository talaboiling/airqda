# Defense Q&A cheat sheet

**Why interpolate only gaps ≤ 6 h?** Pollution is strongly autocorrelated hour to hour, so short gaps are well approximated. Longer gaps would mean inventing data. 74K missing cells dropped to 33K.

**Why not remove outliers?** 4.7% of PM2.5 is above the IQR fence (247.5), but these are real haze episodes, the subject of the study. They are flagged in `pm25_outlier`. Only physically impossible values were removed: negatives, and O3 > 500.

**Why is chunked loading slower?** Concatenating chunks costs time. It halves peak memory (103 → 51 MB), which matters when files exceed RAM.

**Why Parquet?** It is columnar and compressed: 5.3× smaller than CSV and ~10× faster to read. It can also read only the needed columns.

**Why is the schema normalised?** Station, district, wind-direction and season attributes are stored once and referenced by keys (3NF). `area_type` depends on the district, so it lives in `districts`.

**Which window functions did you use?** RANK (q5), ROW_NUMBER (q10), LAG (q7), and AVG OVER ROWS 6 PRECEDING (q6). CTEs appear in q4–q7, q9 and q10.

**Why monitoring years Mar–Feb?** The data starts 1 Mar 2013 and ends 28 Feb 2017. Calendar 2013 lacks January–February and 2017 is winter only, so calendar averages would be biased.

**How did you avoid data leakage?** The split is by time (test = the last 12 months). All features are shifted to t−1, including moving averages and weather. The scaler is fitted on train inside a sklearn Pipeline. All approaches use the same test rows.

**Why is seasonal naive the worst?** It assumes a weekly cycle, but weekly seasonal strength is only 0.01.

**Why is naive hard to beat?** Episodes persist for days. Its weakness is repeating each peak one day late.

**Why choose Linear Regression over Gradient Boosting?** Boosting is 4.5% better on MAE but ~100× slower, uses 2× the memory, and is less interpretable.

**What is seasonal strength?** 1 − Var(residual) / Var(seasonal + residual), from Hyndman & Athanasopoulos. 0 means no seasonality and 1 means pure seasonality.

**Why is Pearson r for weather small?** The relationships are non-linear (for example, dew point rises then falls). Binned medians and Spearman (wind −0.33) show them better.

**Where is error handling?** In `main.py`: invalid menu option, non-numeric query number, unknown station, missing files. In `database.run_query`: `sqlite3.Error` and a missing DB. In `loader`: corrupt zip, no CSVs found.

**Known limitation to admit:** Linear Regression can predict slightly negative values on very clean days; clipping at 0 is a possible fix.
