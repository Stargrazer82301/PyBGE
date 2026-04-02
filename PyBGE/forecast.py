"""
pybge.forecast
--------------
Uses the model bundle produced by ``pybge.correlate()`` together with
Open-Meteo weather forecasts to predict energy usage over the next 30 days.

Outputs saved to ``data_dir``:
  - ``Historical_Hourly.csv.gz``       -- cached long-run historical weather
  - ``BGE_Forecast_TempVsUsage.png``   -- scatter: temperature vs predicted use
  - ``BGE_Forecast_Daily.png``         -- bar chart of day-by-day predictions
"""

from __future__ import annotations

import datetime
import os

import cmasher
import cmocean
import hickle
import matplotlib
import matplotlib.dates
import matplotlib.pyplot as plt
import meteostat
from meteostat import Provider
from meteostat.enumerations import Parameter
import numpy as np
import pandas as pd
import seaborn as sns
import openmeteopy
from openmeteopy.hourly import HourlyForecast

from .collate import _running_tots
from .correlate import pred_descale
from .utils import c_to_f

plt.ioff()
sns.set(context="talk")
sns.set_style("darkgrid", {"font.sans-serif": "DejaVu Sans"})

# Running stat definitions for weather-only columns (no usage cols here)
_RUNNING_COLS = [
    "temp_p24", "tmin_p24", "tmax_p24",
    "rhum_p24", "prcp_p24", "wdir_p24",
    "wspd_p24", "pres_p24", "cldc_p24",
]
_RUNNING_COLS_TYPE = [
    "avg", "min", "max",
    "avg", "tot", "circavg",
    "avg", "avg", "avg",
]

# Mapping from meteostat column names to Open-Meteo parameter names
_OPENMET_DICT = {
    "temp":  "temperature_2m",
    "rhum":  "relativehumidity_2m",
    "prcp":  "precipitation",
    "wdir":  "winddirection_10m",
    "wspd":  "windspeed_10m",
    "pres":  "pressure_msl",
    "cldc":  "cloudcover",
}


def forecast(
    data_dir: str,
    lat: float,
    lon: float,
    *,
    alt: float = 69,
    forecast_days: int = 30,
    freedom_units: bool = False,
) -> None:
    """Predict energy usage for the coming ``forecast_days`` days.

    Parameters
    ----------
    data_dir:
        Directory containing ``BGE_Collated.csv.gz`` and
        ``BGE_Regressor.hkl`` (produced by ``pybge.run()`` and
        ``pybge.correlate()``).  All outputs are also written here.
    lat:
        Latitude of the property.  Defaults to 39.33125.
    lon:
        Longitude of the property.  Defaults to -76.63248.
    alt:
        Altitude of the property in metres.  Defaults to 69.
    forecast_days:
        Number of days ahead to forecast.  Defaults to 30.
    freedom_units:
        If ``True``, temperature axes on all plots use Fahrenheit.
        If not explicitly set, the value stored in the model bundle by
        ``pybge.correlate()`` is used as the default.  Defaults to ``False``
        when no bundle preference is present.
    """

    # ------------------------------------------------------------------ #
    # Load model bundle
    # ------------------------------------------------------------------ #
    hkl_path = os.path.join(data_dir, "BGE_Regressor.hkl")
    if not os.path.exists(hkl_path):
        raise FileNotFoundError(
            f"BGE_Regressor.hkl not found in {data_dir!r}. "
            "Run pybge.correlate() first."
        )
    regr_dict       = hickle.load(hkl_path)
    cols_train      = regr_dict["cols_train"]
    model           = regr_dict["model_curr"]
    features_scaler = regr_dict["features_scaler"]
    target_scaler   = regr_dict["target_scaler"]
    col_target      = regr_dict["col_target"]
    col_target_unit = regr_dict["col_target_unit"]
    # Honour freedom_units from correlate bundle unless caller overrides
    bundle_freedom  = regr_dict.get("freedom_units", False)
    # The kwarg default is False, but if the bundle says True we respect that;
    # an explicit True from the caller always wins.
    effective_freedom = freedom_units or bundle_freedom

    temp_label     = "Day Average Temperature (F)" if effective_freedom else "Day Average Temperature (C)"
    cbar_temp_label = "Avg Expected Temperature (F)" if effective_freedom else "Avg Expected Temperature (C)"

    # ------------------------------------------------------------------ #
    # Load collated data (used to build the forecast frame skeleton)
    # ------------------------------------------------------------------ #
    collated_path = os.path.join(data_dir, "BGE_Collated.csv.gz")
    if not os.path.exists(collated_path):
        raise FileNotFoundError(
            f"BGE_Collated.csv.gz not found in {data_dir!r}. "
            "Run pybge.run() first."
        )
    thermo_hourly = pd.read_csv(collated_path)
    thermo_hourly["date"] = pd.to_datetime(thermo_hourly["date"])
    thermo_hourly = thermo_hourly.drop_duplicates(subset=["date"], keep="first")

    # ------------------------------------------------------------------ #
    # Build empty hourly frame for the forecast window
    # ------------------------------------------------------------------ #
    typical_hourly = thermo_hourly.copy()
    drop_cols = [c for c in typical_hourly.columns
                 if c == "col1" or "unnamed" in c.lower()]
    if drop_cols:
        typical_hourly = typical_hourly.drop(columns=drop_cols)
    typical_hourly = typical_hourly.iloc[0:0]
    usage_cols = [c for c in typical_hourly.columns if "usage" in c]
    typical_hourly = typical_hourly.drop(columns=usage_cols)

    now = datetime.datetime.now()
    dt_start = (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    dt_end = dt_start + datetime.timedelta(days=forecast_days) + datetime.timedelta(hours=24)
    typical_hourly["date"] = pd.date_range(start=dt_start, end=dt_end, freq="h")

    typical_hourly["date_hour"]        = typical_hourly["date"].dt.hour
    typical_hourly["date_weekdayname"] = typical_hourly["date"].dt.day_name()
    typical_hourly["date_weekday"]     = typical_hourly["date"].dt.weekday
    typical_hourly["date_day"]         = typical_hourly["date"].dt.day
    typical_hourly["date_month"]       = typical_hourly["date"].dt.month
    typical_hourly["date_monthname"]   = typical_hourly["date"].dt.month_name()
    typical_hourly["date_year"]        = typical_hourly["date"].dt.year

    forecast_hourly = typical_hourly.copy()

    # ------------------------------------------------------------------ #
    # Historical climatology (cached)
    # ------------------------------------------------------------------ #
    met_hourly_path = os.path.join(data_dir, "Historical_Hourly.csv.gz")
    met_point    = meteostat.Point(lat, lon, alt)
    met_stations = meteostat.stations.nearby(met_point, limit=20)

    if not os.path.exists(met_hourly_path):
        print("Querying historical weather via Meteostat (one-time download)...")
        # Fetch 10 years of history for climatological averaging.
        hist_start = datetime.datetime.now() - datetime.timedelta(days=11*365)
        hist_end = datetime.datetime.now() - datetime.timedelta(days=366)
        meteostat.config.block_large_requests = False
        meteostat.config.cache_enable = False
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
            hist_start,
            hist_end,
            providers=[Provider.HOURLY],
            parameters=met_params,
        )

        # Try station interpolation; if that fails, just use neirest neighbour
        met_hourly_interp = meteostat.interpolate(met_hourlies, met_point)
        met_hourly_hist = met_hourly_interp.fetch()
        if met_hourly_hist is None:
            nearest_station_id = met_stations.index[0]
            met_hourly_hist = met_hourlies.fetch()
            met_hourly_hist = met_hourly_hist.loc[met_hourly_hist.index.get_level_values('station') == nearest_station_id]
            met_hourly_hist = met_hourly_hist.reset_index(level='station', drop=True)

        # If that still fails, raise exception
        if met_hourly_hist is None:
            raise RuntimeError(
                "Meteostat returned no historical weather data for the given "
                "location. Check that lat/lon/alt are correct."
            )
        met_hourly_hist.to_csv(met_hourly_path)

    met_hourly_hist = pd.read_csv(met_hourly_path)
    met_cols = met_hourly_hist.columns[1:]
    met_hourly_hist["date"]       = pd.to_datetime(met_hourly_hist["time"], format="%Y-%m-%d %H:%M:%S")
    met_hourly_hist["date_hour"]  = met_hourly_hist["date"].dt.hour
    met_hourly_hist["date_day"]   = met_hourly_hist["date"].dt.day
    met_hourly_hist["date_month"] = met_hourly_hist["date"].dt.month

    for h in typical_hourly.index:
        met_where = np.where(
            (met_hourly_hist["date_month"] == typical_hourly.loc[h, "date_month"])
            & (met_hourly_hist["date_day"]   == typical_hourly.loc[h, "date_day"])
            & (met_hourly_hist["date_hour"]  == typical_hourly.loc[h, "date_hour"])
        )
        for col in met_cols:
            typical_hourly.loc[h, col] = np.nanmean(met_hourly_hist.loc[met_where, col])

    typical_hourly = _running_tots(typical_hourly, _RUNNING_COLS, _RUNNING_COLS_TYPE)

    # ------------------------------------------------------------------ #
    # Open-Meteo live forecast
    # ------------------------------------------------------------------ #
    hourly_obj = HourlyForecast()
    options    = openmeteopy.options.ForecastOptions(lat, lon)
    openmet_query = openmeteopy.OpenMeteo(options, hourly_obj.temperature_2m())
    openmet_forecast = openmet_query.get_pandas()
    openmet_forecast["date"] = pd.to_datetime(
        openmet_forecast.index, format="%Y-%m-%dT%H:%M"
    )

    if "cloudcover" in openmet_forecast.columns:
        openmet_forecast["cloudcover"] = (openmet_forecast["cloudcover"] / 100.0) * 8.0

    for h in forecast_hourly.index:
        f = openmet_forecast.index[
            np.where(forecast_hourly.loc[h, "date"] == openmet_forecast["date"])
        ]
        if len(f) == 0:
            continue
        for col, om_col in _OPENMET_DICT.items():
            if om_col in openmet_forecast.columns:
                forecast_hourly.loc[h, col] = openmet_forecast.loc[f[0], om_col]

    # ------------------------------------------------------------------ #
    # Taper from live forecast to climatology
    # ------------------------------------------------------------------ #
    taper_end   = openmet_forecast["date"].max()
    taper_start = taper_end - datetime.timedelta(days=1)

    taper_array = np.full(len(forecast_hourly), np.nan)
    idx_start = int(np.argmax(forecast_hourly["date"] == taper_start))
    idx_end   = int(np.argmax(forecast_hourly["date"] == taper_end))
    taper_array[:idx_start]        = 1.0
    taper_array[idx_end:]          = 0.0
    taper_array[idx_start:idx_end] = np.linspace(1, 0, num=idx_end - idx_start)
    forecast_hourly["taper"] = taper_array
    taper_inv = 1.0 - taper_array

    for col in _OPENMET_DICT.keys():
        if col not in forecast_hourly.columns or col not in typical_hourly.columns:
            continue
        blended = (taper_array * forecast_hourly[col]) + (taper_inv * typical_hourly[col])
        nan_mask = np.where(np.isnan(blended))
        blended[nan_mask] = typical_hourly.loc[nan_mask, col].values
        forecast_hourly[col] = blended

    forecast_hourly = _running_tots(forecast_hourly, _RUNNING_COLS, _RUNNING_COLS_TYPE)

    forecast_daily = forecast_hourly.loc[
        (forecast_hourly["date_hour"] == 6) & (~np.isnan(forecast_hourly["temp_p24"]))
    ].copy()

    # ------------------------------------------------------------------ #
    # Predict usage
    # ------------------------------------------------------------------ #
    forecast_pred = pred_descale(
        forecast_daily[cols_train], model, features_scaler, target_scaler
    )
    forecast_daily = forecast_daily.copy()
    forecast_daily[col_target[0]] = forecast_pred
    forecast_pred_tot = float(np.nansum(forecast_pred))

    # ------------------------------------------------------------------ #
    # Temperature series for plots (convert if freedom_units)
    # ------------------------------------------------------------------ #
    plot_temp = c_to_f(forecast_daily["temp_p24"]) if effective_freedom else forecast_daily["temp_p24"]

    # ------------------------------------------------------------------ #
    # Plot 1: Temperature vs predicted usage
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    ax.scatter(
        plot_temp,
        forecast_daily[col_target],
        s=25,
        c=np.arange(len(forecast_daily)),
        cmap=cmocean.tools.crop_by_percent(cmocean.cm.dense, 45, which="both"),
    )
    ax.set_ylabel(f"Predicted Energy Use ({col_target_unit})", fontsize=15)
    ax.set_xlabel(temp_label, fontsize=15)
    p1_path = os.path.join(data_dir, "BGE_Forecast_TempVsUsage.png")
    fig.savefig(p1_path, dpi=175)
    plt.close(fig)
    print(f"Saved temp-vs-usage plot to {p1_path}")

    # ------------------------------------------------------------------ #
    # Plot 2: Day-by-day bar chart (colourbar = temperature)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.subplots_adjust(left=0.15, bottom=0.25, right=0.85, top=0.95)

    cbar_norm = matplotlib.colors.Normalize()
    cbar_norm.autoscale(plot_temp)
    cbar_map = cmasher.bubblegum(cbar_norm(plot_temp.values))

    ax.bar(
        matplotlib.dates.date2num(forecast_daily["date"].dt.date) - 1,
        forecast_daily[col_target[0]],
        align="center",
        color=cbar_map,
        width=0.8,
    )
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(matplotlib.dates.DayLocator(interval=3))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(45)
        lbl.set_ha("right")
        lbl.set_fontsize(12.5)
    ax.set_xlabel("Date", fontsize=17.5)
    ax.set_ylabel(f"Predicted Energy Use ({col_target_unit})", fontsize=17.5)
    ax.set_ylim(0, 1.1 * float(np.nanmax(forecast_daily[col_target[0]])))
    ax.text(
        0.02, 0.94,
        f"Predicted total: {forecast_pred_tot:.2f} {col_target_unit}",
        fontsize=15,
        transform=ax.transAxes,
    )
    cbar_sm = matplotlib.cm.ScalarMappable(norm=cbar_norm, cmap=cmasher.bubblegum)
    cbar_sm.set_array([])
    cbar = fig.colorbar(cbar_sm, ax=ax, label=cbar_temp_label)
    cbar.ax.tick_params(length=0)
    cbar.ax.yaxis.label.set_fontsize(17.5)

    p2_path = os.path.join(data_dir, "BGE_Forecast_Daily.png")
    fig.savefig(p2_path, dpi=175)
    plt.close(fig)
    print(f"Saved daily forecast plot to {p2_path}")

    print(f"Forecast total over {forecast_days} days: {forecast_pred_tot:.2f} {col_target_unit}")
    print("Forecast complete.  And furthermore, Carthage must be destroyed.")
