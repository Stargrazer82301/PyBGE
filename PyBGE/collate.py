"""
pybge.collate
-------------
Collates parsed hourly usage data with Meteostat weather observations and
computes 24-hour rolling totals / averages.
"""

from __future__ import annotations

import os
import datetime
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats
import meteostat
from meteostat import Provider
from meteostat.enumerations import Parameter

from .utils import c_to_f

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_COL_SKIP = {"snow", "wpgt", "tsun"}

_RUNNING_COLS = [
    "usage_kwh_p24",
    "usage_therm_p24",
    "usage_ekwh_p24",
    "temp_p24",
    "tmin_p24",
    "tmax_p24",
    "rhum_p24",
    "prcp_p24",
    "wdir_p24",
    "wspd_p24",
    "pres_p24",
    "cldc_p24",
]
_RUNNING_COLS_TYPE = [
    "tot", "tot", "tot",
    "avg", "min", "max",
    "avg", "tot",
    "circavg", "avg", "avg", "avg",
]

# Temperature columns that get Fahrenheit copies when freedom_units=True
_TEMP_COLS = ["temp", "tmin", "tmax", "temp_p24", "tmin_p24", "tmax_p24"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _running_tots(
    frame: pd.DataFrame,
    running_cols: list[str],
    running_cols_type: list[str],
) -> pd.DataFrame:
    """Compute 24-hour rolling totals / averages for each column."""
    print(" ")
    time_skip = frame["date"].min() + datetime.timedelta(hours=24)

    for h in frame.index:
        if frame.loc[h, "date"] < time_skip:
            continue

        pct = 100 * h / len(frame.index)
        print(
            f"\rAveraging/Totalling weather data for "
            f"{frame.loc[h, 'date']}; {pct:.0f}% complete",
            end="",
        )

        for c, running_col in enumerate(running_cols):
            col = running_col.replace("_p24", "")
            if running_col == "tmin_p24":
                col = "temp"
            elif running_col == "tmax_p24":
                col = "temp"

            window_start = frame.loc[h, "date"] - datetime.timedelta(hours=24)
            mask = (frame["date"] > window_start) & (frame["date"] <= frame.loc[h, "date"])
            window_data = np.array(frame.loc[mask, col])

            stat = running_cols_type[c]
            if stat == "tot":
                frame.loc[h, running_col] = np.sum(window_data)
            elif stat == "avg":
                frame.loc[h, running_col] = np.mean(window_data)
            elif stat == "min":
                frame.loc[h, running_col] = np.min(window_data)
            elif stat == "max":
                frame.loc[h, running_col] = np.max(window_data)
            elif stat == "circavg":
                frame.loc[h, running_col] = scipy.stats.circmean(window_data, high=360)

    return frame


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def collate(
    data_dir: str,
    lat: float,
    lon: float,
    alt: float,
    dollars_per_kwhr: Optional[float] = None,
    dollars_per_therm: Optional[float] = None,
    freedom_units: bool = False,
) -> None:
    """Join usage data with Meteostat weather and write ``BGE_Collated.csv.gz``.

    Parameters
    ----------
    data_dir:
        Directory containing ``BGE_Usage.csv.gz``; output
        ``BGE_Collated.csv.gz`` is also written here.
    lat:
        Latitude of the property (decimal degrees).
    lon:
        Longitude of the property (decimal degrees).
    alt:
        Altitude of the property in metres above sea level.
    dollars_per_kwhr:
        Electricity cost per kWh.  When both this and ``dollars_per_therm``
        are provided, a ``usage_dollars_p24`` rolling column is added.
    dollars_per_therm:
        Gas cost per therm.  See ``dollars_per_kwhr``.
    freedom_units:
        If ``True``, Fahrenheit copies of all temperature columns are added
        to the output CSV alongside the native Celsius columns.  The Celsius
        columns are always retained.  Defaults to ``False``.
    """
    usage_path = os.path.join(data_dir, "BGE_Usage.csv.gz")
    out_path = os.path.join(data_dir, "BGE_Collated.csv.gz")

    if not os.path.exists(usage_path):
        raise FileNotFoundError(
            f"BGE_Usage.csv.gz not found in {data_dir!r}.  "
            "Run the parse step first."
        )

    use_dollars = (dollars_per_kwhr is not None) and (dollars_per_therm is not None)

    meteostat.config.cache_enable = False

    # ------------------------------------------------------------------ #
    # Load usage data
    # ------------------------------------------------------------------ #
    append = os.path.exists(out_path)

    thermo_hourly = pd.read_csv(usage_path)
    if "Unnamed: 0" in thermo_hourly.columns:
        thermo_hourly = thermo_hourly.drop("Unnamed: 0", axis=1)
    thermo_hourly["date"] = pd.to_datetime(thermo_hourly["date"])
    thermo_hourly = thermo_hourly.drop_duplicates(subset=["date"], keep="first")

    thermo_hourly_old: pd.DataFrame | None = None
    if append:
        thermo_hourly_old = pd.read_csv(out_path, parse_dates=["date"])
        if "Unnamed: 0" in thermo_hourly_old.columns:
            thermo_hourly_old = thermo_hourly_old.drop("Unnamed: 0", axis=1)

    # ------------------------------------------------------------------ #
    # Determine weather data start date
    # ------------------------------------------------------------------ #
    met_hourly_start = thermo_hourly["date"].min()
    if append and thermo_hourly_old is not None:
        valid_temp = thermo_hourly_old["date"][~np.isnan(thermo_hourly_old["temp"])]
        if not valid_temp.empty:
            met_hourly_start = valid_temp.max()

    # ------------------------------------------------------------------ #
    # Fetch Meteostat weather
    # ------------------------------------------------------------------ #
    print(" ")
    print("Querying Meteostat weather databases")
    met_point = meteostat.Point(lat, lon, alt)
    met_stations = meteostat.stations.nearby(met_point, limit=20)
    met_params = [
        Parameter.TEMP,
        Parameter.RHUM,
        Parameter.PRCP,
        Parameter.WDIR,
        Parameter.WSPD,
        Parameter.PRES,
        Parameter.CLDC,
    ]
    met_hourlies = meteostat.hourly(
        met_stations,
        met_hourly_start,
        thermo_hourly["date"].max(),
        providers=[Provider.HOURLY],
        parameters=met_params,
    )
    met_hourly = meteostat.interpolate(met_hourlies, met_point).fetch()

    # ------------------------------------------------------------------ #
    # Initialise weather columns
    # ------------------------------------------------------------------ #
    for col in met_hourly.columns:
        if col not in _COL_SKIP:
            thermo_hourly[col] = np.nan
    thermo_hourly["cldc"] = np.nan

    if "prcp" not in met_hourly.columns:
        met_hourly["prcp"] = np.nan
    where_prcp_nan = met_hourly.index[
        np.isnan(met_hourly["prcp"].values) & ~np.isnan(met_hourly["temp"].values)
    ]
    met_hourly.loc[where_prcp_nan, "prcp"] = 0.0

    # ------------------------------------------------------------------ #
    # Merge weather into usage frame
    # ------------------------------------------------------------------ #
    for h in thermo_hourly.index:
        current_date = thermo_hourly.loc[h, "date"]

        if current_date > met_hourly.index.max():
            continue

        old_fail = False
        if append and thermo_hourly_old is not None:
            if current_date in thermo_hourly_old["date"].values:
                old_idx = thermo_hourly_old.index[
                    thermo_hourly_old["date"].values == current_date
                ]
                for col in met_hourly.columns:
                    if col not in _COL_SKIP:
                        thermo_hourly.loc[h, col] = thermo_hourly_old.loc[old_idx, col].values
            else:
                old_fail = True

        if append and current_date < met_hourly_start:
            continue

        pct = 100 * h / len(thermo_hourly.index)
        print(
            f"\rCollating weather data for {current_date}; {pct:.0f}% complete",
            end="",
        )

        if not append or old_fail:
            met_idx = met_hourly.index[current_date == met_hourly.index]
            if len(met_idx) == 0:
                continue
            for col in met_hourly.columns:
                if col not in _COL_SKIP:
                    thermo_hourly.loc[h, col] = met_hourly.loc[met_idx, col].values[0]

    # ------------------------------------------------------------------ #
    # Build running column list (add dollars_p24 if rates provided)
    # ------------------------------------------------------------------ #
    running_cols = list(_RUNNING_COLS)
    running_cols_type = list(_RUNNING_COLS_TYPE)

    if use_dollars:
        running_cols.append("usage_dollars_p24")
        running_cols_type.append("tot")

    for running_col in running_cols:
        thermo_hourly[running_col] = np.nan

    thermo_hourly = _running_tots(thermo_hourly, running_cols, running_cols_type)

    # ------------------------------------------------------------------ #
    # Append Fahrenheit columns if requested
    # ------------------------------------------------------------------ #
    if freedom_units:
        for col in _TEMP_COLS:
            if col in thermo_hourly.columns:
                thermo_hourly[col + "_f"] = c_to_f(thermo_hourly[col])

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    thermo_hourly = thermo_hourly.drop_duplicates(subset=["date"], keep="first")
    thermo_hourly.to_csv(out_path)
    print(" ")
    print("Collation complete.")
