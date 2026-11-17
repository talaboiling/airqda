"""Console entry point: interactive menu or full pipeline (python main.py --all)."""
import sys

from src import cleaning, config, database, pipeline

MENU = """
========= Beijing Air Quality Analysis =========
 1 - Load raw data, audit and clean
 2 - Build SQLite database (+ ER diagram, Pandas vs SQL benchmark)
 3 - Run SQL queries
 4 - EDA and visualisations
 5 - Time-series analysis (resample, rolling, decomposition)
 6 - Forecast comparison (baselines vs models)
 7 - Show statistics for one station
 8 - Run full pipeline
 0 - Exit
================================================"""


def get_data(state: dict):
    """Return cleaned data from memory or from the Parquet file."""
    if state.get("df") is None:
        state["df"] = cleaning.load_clean()
    return state["df"]


def choose_query() -> None:
    keys = list(database.QUERIES)
    for i, k in enumerate(keys, 1):
        print(f" {i:>2}. {database.QUERIES[k]['title']}")
    raw = input("Query number (Enter = all): ").strip()
    if raw == "":
        pipeline.step_sql()
        return
    idx = int(raw)  # ValueError handled by caller
    if not 1 <= idx <= len(keys):
        print("No such query.")
        return
    print(database.run_query(keys[idx - 1]).head(30).to_string(index=False))


def station_stats(state: dict) -> None:
    df = get_data(state)
    names = sorted(map(str, df["station"].unique()))
    print(", ".join(names))
    name = input("Station name: ").strip().title()
    if name not in names:
        print(f"Unknown station '{name}'.")
        return
    sub = df[df["station"] == name]
    print(sub[config.POLLUTANTS + config.WEATHER].describe().round(1).to_string())


def main() -> None:
    config.ensure_dirs()
    state: dict = {"df": None}
    actions = {
        "1": lambda: state.update(df=pipeline.step_load_and_clean()),
        "2": lambda: pipeline.step_build_db(get_data(state)),
        "3": choose_query,
        "4": lambda: pipeline.step_eda(get_data(state)),
        "5": lambda: pipeline.step_timeseries(get_data(state)),
        "6": lambda: pipeline.step_forecast(get_data(state)),
        "7": lambda: station_stats(state),
        "8": pipeline.run_all,
    }
    while True:
        print(MENU)
        choice = input("Select option: ").strip()
        if choice == "0":
            print("Bye.")
            break
        action = actions.get(choice)
        if action is None:
            print("Invalid option, enter a number from the menu.")
            continue
        try:
            action()
        except FileNotFoundError as e:
            print(f"File error: {e}")
        except ValueError as e:
            print(f"Invalid input: {e}")
        except KeyboardInterrupt:
            print("\nCancelled.")


if __name__ == "__main__":
    if "--all" in sys.argv:
        pipeline.run_all()
    else:
        main()
