"""
pybge.api
---------
All public entry-points: ``run()``, ``correlate()``, ``forecast()``.
"""

from __future__ import annotations

import os
import datetime
from typing import Optional

from .scrape import scrape
from .parse import parse
from .collate import collate
from .correlate import correlate as _correlate
from .forecast import forecast as _forecast


def run(
    path: str,
    username: str,
    password: str,
    service_agreement_uuid: str,
    *,
    lat: float = 39.33125,
    lon: float = -76.63248,
    alt: float = 69,
    tesseract_cmd: Optional[str] = None,
    date_start: Optional[datetime.date] = None,
    date_end: Optional[datetime.date] = None,
    dollars_per_kwhr: Optional[float] = None,
    dollars_per_therm: Optional[float] = None,
    freedom_units: bool = False,
) -> None:
    """Scrape BGE usage screenshots, parse them into a CSV, then collate with
    weather data.

    Parameters
    ----------
    path:
        Directory where tabular output files are stored.  Must already exist.
    username:
        BGE account e-mail address.
    password:
        BGE account password.
    service_agreement_uuid:
        BGE's internal identifier for your specific utility service contract.
        It is embedded in the URL of your hourly usage page and tells the
        BGE website which meter/address to show data for.

        **How to find it:**

        1. Log into your BGE account at https://secure.bge.com.
        2. Navigate to My Account -> My Bill & Usage -> View My Usage.
        3. Select "Hourly" or "Daily" view for either electricity or gas.
           The URL in your browser address bar will look like::

               https://secure.bge.com/MyAccount/MyBillUsage/Pages/Secure/ViewMyUsage.aspx
               ?ou-data-browser=%2Fusage%2Felectricity%2Fday%2F2024-01-01
               %3FserviceAgreementUuid%3Dxxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx

        4. Your UUID is the value after ``serviceAgreementUuid%3D`` at the
           end of the URL. It is a 36-character string in the format
           ``xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`` (8-4-4-4-12 hex digits
           separated by hyphens), for example::

               a7f21c94-3e10-48bd-c291-58d7e4b06a13

        5. Copy that value and pass it as this argument.

        The UUID is account-specific and does not change over time, so you
        only need to look it up once.
    lat:
        Latitude of the property (used for Meteostat weather lookup).
        Defaults to 39.33125.
    lon:
        Longitude of the property.  Defaults to -76.63248.
    alt:
        Altitude of the property in metres.  Defaults to 69.
    tesseract_cmd:
        Full path to the ``tesseract`` executable.  ``None`` = auto-detect.
    date_start:
        Earliest date to process.  ``None`` = auto-detect from existing
        output files, or two years before ``date_end`` if no files exist.
    date_end:
        Latest date to process.  ``None`` = three days before current date.
    dollars_per_kwhr:
        Electricity cost per kWh.  When both this and ``dollars_per_therm``
        are provided:
          - ``usage_ekwh`` is recalculated using the cost ratio.
          - A ``usage_dollars`` column is added to ``BGE_Usage.csv.gz``.
          - A ``usage_dollars_p24`` column is added to ``BGE_Collated.csv.gz``.
    dollars_per_therm:
        Gas cost per therm.  See ``dollars_per_kwhr``.
    freedom_units:
        If ``True``, Fahrenheit copies of all temperature columns
        (``temp_f``, ``tmin_f``, ``tmax_f``, ``temp_p24_f``, etc.) are
        appended to ``BGE_Collated.csv.gz`` alongside the native Celsius
        columns.  Defaults to ``False``.
    """
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"data directory not found: {path!r}")

    img_dir = os.path.join(path, 'screenshots')
    os.makedirs(img_dir, exist_ok=True)

    if date_end is None:
        date_end = datetime.date.today() - datetime.timedelta(days=3)

    if date_start is None:
        collated_path = os.path.join(path, "BGE_Collated.csv.gz")
        usage_path = os.path.join(path, "BGE_Usage.csv.gz")
        if not os.path.exists(collated_path) and not os.path.exists(usage_path):
            date_start = date_end.replace(year=date_end.year - 2)

    print("=" * 60)
    print("PyBGE - Scraping BGE website")
    print("=" * 60)
    scrape(
        img_dir=img_dir,
        data_dir=path,
        username=username,
        password=password,
        service_agreement_uuid=service_agreement_uuid,
        tesseract_cmd=tesseract_cmd,
        date_start=date_start,
        date_end=date_end,
    )

    print("=" * 60)
    print("PyBGE - Parsing screenshots")
    print("=" * 60)
    parse(
        img_dir=img_dir,
        data_dir=path,
        tesseract_cmd=tesseract_cmd,
        date_start=date_start,
        date_end=date_end,
        dollars_per_kwhr=dollars_per_kwhr,
        dollars_per_therm=dollars_per_therm,
    )

    print("=" * 60)
    print("PyBGE - Collating usage with weather data")
    print("=" * 60)
    collate(
        data_dir=path,
        lat=lat,
        lon=lon,
        alt=alt,
        dollars_per_kwhr=dollars_per_kwhr,
        dollars_per_therm=dollars_per_therm,
        freedom_units=freedom_units,
    )


def correlate(
    data_dir: str,
    *,
    predict_days: int = 30,
    method: str = "Neural-Net",
    plot_unit: str = "auto",
    dollars_per_kwhr: Optional[float] = None,
    dollars_per_therm: Optional[float] = None,
    freedom_units: bool = False,
) -> None:
    """Train a weather-to-energy-usage regression model and save outputs.

    Parameters
    ----------
    data_dir:
        Directory containing ``BGE_Collated.csv.gz`` (produced by
        ``pybge.run()``).  All outputs are also written here.
    predict_days:
        Number of most-recent days to use as the prediction window.  All
        earlier data forms the training window.  Defaults to 30.
    method:
        Regression method.  One of: ``"GPR"`` (default), ``"linear"``,
        ``"BRR"``, ``"ARDR"``, ``"Random-Forest"``, ``"Quantile-Forest"``,
        ``"Neural-Net"``.
    plot_unit:
        ``"dollars"`` -- plot and model in USD (requires rate kwargs).
        ``"ekwh"``    -- plot and model in effective kWh.
        ``"auto"``    -- dollars if rates provided, otherwise ekWh (default).
    dollars_per_kwhr:
        Electricity cost per kWh.  Required when ``plot_unit="dollars"``.
    dollars_per_therm:
        Gas cost per therm.  Required when ``plot_unit="dollars"``.
    freedom_units:
        If ``True``, temperature axes on all plots use Fahrenheit.
        The model is trained on Celsius internally; only the display is
        converted.  Defaults to ``False``.
    """
    _correlate(
        data_dir=data_dir,
        predict_days=predict_days,
        method=method,
        plot_unit=plot_unit,
        dollars_per_kwhr=dollars_per_kwhr,
        dollars_per_therm=dollars_per_therm,
        freedom_units=freedom_units,
    )


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
        Latitude of the property in degrees.
    lon:
        Longitude of the property in degrees.
    alt:
        Altitude of the property in metres.  Defaults to 69.
    forecast_days:
        Number of days ahead to forecast.  Defaults to 30.
    freedom_units:
        If ``True``, temperature axes on all plots use Fahrenheit.
        If not explicitly set here, the value stored in the model bundle
        by ``pybge.correlate()`` is used automatically.  Defaults to
        ``False`` when no bundle preference is present.
    """
    _forecast(
        data_dir=data_dir,
        lat=lat,
        lon=lon,
        alt=alt,
        forecast_days=forecast_days,
        freedom_units=freedom_units,
    )
