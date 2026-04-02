"""
pybge.parse
-----------
Parses BGE usage bar-chart screenshot PNGs into a gzipped CSV of hourly
electricity and gas usage.

This is a direct refactoring of ``BGE_Parse.py``; all hard-coded paths
have been replaced with function parameters.
"""

from __future__ import annotations

import os
import time
import datetime
from typing import Optional

import numpy as np
import pandas as pd
import PIL.Image
import pytesseract

from .utils import time_est, configure_tesseract

# ---------------------------------------------------------------------------
# Constants -- pixel geometry of the BGE bar chart at 1440 x 1600 viewport
# ---------------------------------------------------------------------------
_BARS_X_MIN = 544
_BARS_X_MAX = 1173
_BAR_X_WIDTH = 20
_BAR_Y_MIN = 509
_BAR_Y_MAX = 813

_LABEL_X_MIN = 448
_LABEL_X_MAX = 542
_LABEL_Y_MIN = 500
_LABEL_Y_MAX = 516

_SORRY_X_MIN = 460
_SORRY_X_MAX = 1200
_SORRY_Y_MIN = 400
_SORRY_Y_MAX = 800

_BAR_RGB = np.array([23, 13, 103], dtype=np.uint8)
_HEADER_RGB = np.array([23, 13, 103], dtype=np.uint8)

_FUELS = ["Electricity", "Gas"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _y_offset(img: PIL.Image.Image) -> int:
    """Return the vertical pixel offset introduced by a page banner, if any."""
    where_header = np.all(np.array(img) == _HEADER_RGB, axis=-1)
    rows = np.where(where_header)[0]
    return int(rows.min()) if rows.size > 0 else 0


def _read_label_max(
    img: PIL.Image.Image,
    fuel: str,
    date_string: str,
    y_off: int,
) -> float:
    """OCR the y-axis maximum label and return it as a float (or NaN)."""
    label_img = img.crop(
        (_LABEL_X_MIN, _LABEL_Y_MIN + y_off, _LABEL_X_MAX, _LABEL_Y_MAX + y_off)
    )
    label_str = pytesseract.image_to_string(label_img).replace("\n", "").replace(",", "")

    if len(label_str) == 0:
        sorry_img = img.crop(
            (_SORRY_X_MIN, _SORRY_Y_MIN + y_off, _SORRY_X_MAX, _SORRY_Y_MAX + y_off)
        )
        sorry_str = pytesseract.image_to_string(sorry_img)
        if "sorry" in sorry_str.lower() or "authorized" in sorry_str.lower():
            print(f"No data for {fuel} for {date_string}; recording NaN usage")
        else:
            print(
                f"No webpage data for {date_string} {fuel}; "
                "perhaps page didn't finish loading?"
            )
        return float("nan")

    if fuel.startswith("E"):
        label_str = label_str.replace(" KWh", "").replace(" kWh", "")
    else:
        label_str = label_str.replace(" therms", "").replace(" therm", "")
    try:
        return float(label_str)
    except:
        breakpoint()


def _read_bar_energy(
    img: PIL.Image.Image,
    h: int,
    bars_x_pix: np.ndarray,
    label_max: float,
    y_off: int,
) -> float:
    """Return the energy reading for hour h by measuring bar height."""
    bar_x_center = bars_x_pix[h]
    bar_x_lo = bar_x_center - _BAR_X_WIDTH // 2
    bar_x_hi = bar_x_center + _BAR_X_WIDTH // 2
    img_bar = img.crop((bar_x_lo, _BAR_Y_MIN + y_off, bar_x_hi, _BAR_Y_MAX + y_off))

    img_bar_arr = np.flip(np.array(img_bar), axis=0)
    where_bar = np.where(np.all(img_bar_arr == _BAR_RGB, axis=-1))

    bar_max = where_bar[0].max() if where_bar[0].size > 0 else 0
    bar_frac = float(bar_max) / img_bar_arr.shape[0]
    return bar_frac * label_max


def _safe_write(frame: pd.DataFrame, out_path: str, max_retries: int = 10) -> None:
    """Write frame to out_path, retrying on file-lock errors."""
    for attempt in range(max_retries):
        try:
            frame.to_csv(out_path)
            return
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            time.sleep(10)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def parse(
    img_dir: str,
    data_dir: str,
    tesseract_cmd: Optional[str],
    date_start: Optional[datetime.date],
    date_end: datetime.date,
    dollars_per_kwhr: Optional[float] = None,
    dollars_per_therm: Optional[float] = None,
) -> None:
    """Parse screenshot PNGs and write hourly usage data to
    ``BGE_Usage.csv.gz`` inside *data_dir*.

    Parameters
    ----------
    img_dir:
        Directory containing the PNG screenshots.
    data_dir:
        Directory where ``BGE_Usage.csv.gz`` is read from / written to.
    tesseract_cmd:
        Path to the tesseract executable (``None`` = auto-detect).
    date_start:
        First date to parse, or ``None`` to auto-detect from existing data.
    date_end:
        Last date to parse (inclusive).
    dollars_per_kwhr:
        Electricity cost per kWh.  If both this and ``dollars_per_therm``
        are provided, ``usage_ekwh`` is recalculated using the cost ratio
        and a ``usage_dollars`` column is added.
    dollars_per_therm:
        Gas cost per therm.  See ``dollars_per_kwhr``.
    """
    configure_tesseract(tesseract_cmd)

    out_path = os.path.join(data_dir, "BGE_Usage.csv.gz")
    append = os.path.exists(out_path)

    # Determine ekwh conversion factor
    use_dollars = (dollars_per_kwhr is not None) and (dollars_per_therm is not None)
    if use_dollars:
        ekwh_per_therm = dollars_per_therm / dollars_per_kwhr
    else:
        ekwh_per_therm = 9.3  # original hard-coded default

    # ------------------------------------------------------------------ #
    # Determine date range
    # ------------------------------------------------------------------ #
    if date_start is None:
        if append:
            thermo_hourly_existing = pd.read_csv(out_path, parse_dates=["date"])
            good_rows = np.where(
                (thermo_hourly_existing["date_hour"] == 23)
                & (~np.isnan(thermo_hourly_existing["usage_kwh"]))
                & (~np.isnan(thermo_hourly_existing["usage_therm"]))
            )
            date_start = (
                thermo_hourly_existing.iloc[good_rows]["date"]
                .max()
                .to_pydatetime()
                .date()
            )
        else:
            raise FileNotFoundError(
                "date_start was not supplied and BGE_Usage.csv.gz does not exist "
                f"in {data_dir!r}.  Please provide an explicit date_start."
            )

    date_list = [
        date_start + datetime.timedelta(days=i)
        for i in range((date_end - date_start).days + 1)
    ]

    # ------------------------------------------------------------------ #
    # Load or initialise the accumulator
    # ------------------------------------------------------------------ #
    if append:
        thermo_frame = pd.read_csv(out_path, parse_dates=["date"])
        if "Unnamed: 0" in thermo_frame.columns:
            thermo_frame = thermo_frame.drop("Unnamed: 0", axis=1)
        thermo_dict = thermo_frame.transpose().to_dict()
        dict_list: list[dict] = [thermo_dict[i] for i in range(len(thermo_dict))]
    else:
        thermo_frame = pd.DataFrame()
        dict_list = []

    bars_x_pix = np.linspace(_BARS_X_MIN, _BARS_X_MAX, num=24).astype(int)

    # ------------------------------------------------------------------ #
    # Main parsing loop
    # ------------------------------------------------------------------ #
    time_list = [time.time()]
    for d, date in enumerate(date_list):
        date_string = date.strftime("%Y-%m-%d")

        fuel_data: dict[str, dict] = {}
        y_off = 0
        for fuel in _FUELS:
            img_path = os.path.join(img_dir, f"{date_string}_{fuel}.png")
            if not os.path.exists(img_path):
                print(f"Screenshot not found: {img_path}; skipping.")
                fuel_data[fuel] = {"img": None, "label_max": float("nan")}
                continue

            img = PIL.Image.open(img_path)
            y_off = _y_offset(img)
            label_max = _read_label_max(img, fuel, date_string, y_off)
            fuel_data[fuel] = {"img": img, "label_max": label_max}

        for h in range(24):
            hour_dict: dict = {
                "date":             datetime.datetime(date.year, date.month, date.day, h, 0, 0),
                "date_hour":        h,
                "date_weekdayname": date.strftime("%A"),
                "date_weekday":     date.strftime("%w"),
                "date_day":         date.strftime("%d"),
                "date_month":       date.strftime("%m"),
                "date_monthname":   date.strftime("%B"),
                "date_year":        date.strftime("%Y"),
            }

            for fuel in _FUELS:
                fd = fuel_data[fuel]
                if np.isnan(fd["label_max"]) or fd["img"] is None:
                    bar_energy = float("nan")
                else:
                    bar_energy = _read_bar_energy(
                        fd["img"], h, bars_x_pix, fd["label_max"], y_off
                    )

                if fuel.startswith("E"):
                    hour_dict["usage_kwh"] = bar_energy
                else:
                    hour_dict["usage_therm"] = bar_energy

            # Effective kWh using either cost-ratio or default conversion
            hour_dict["usage_ekwh"] = (
                hour_dict["usage_kwh"] + ekwh_per_therm * hour_dict["usage_therm"]
            )

            # Dollar cost column (only when rates provided)
            if use_dollars:
                kwh_cost = hour_dict["usage_kwh"] * dollars_per_kwhr
                therm_cost = hour_dict["usage_therm"] * dollars_per_therm
                if np.isnan(kwh_cost) or np.isnan(therm_cost):
                    hour_dict["usage_dollars"] = float("nan")
                else:
                    hour_dict["usage_dollars"] = kwh_cost + therm_cost

            # Merge into accumulator
            if append and not thermo_frame.empty:
                match_mask = thermo_frame["date"] == hour_dict["date"]
                if match_mask.any():
                    dict_list[int(np.argmax(match_mask))] = hour_dict
                else:
                    dict_list.append(hour_dict)
            else:
                dict_list.append(hour_dict)

            if len(dict_list) % 10 == 0:
                thermo_frame = pd.DataFrame(dict_list)
                thermo_frame.sort_values("date", inplace=True)
                thermo_frame.set_index(
                    pd.Index(np.arange(len(thermo_frame)).tolist()), inplace=True
                )
                _safe_write(thermo_frame, out_path)

        time_list.append(time.time())
        if len(time_list) > 2:
            est = time_est(time_list, len(date_list))
            pct = 100.0 * (d + 1) / len(date_list)
            print(
                f"\rParsing {date_string}; {pct:.1f}% complete; "
                f"estimate complete {est}",
                end="",
            )

    # Final flush
    if dict_list:
        thermo_frame = pd.DataFrame(dict_list)
        thermo_frame.sort_values("date", inplace=True)
        thermo_frame.set_index(
            pd.Index(np.arange(len(thermo_frame)).tolist()), inplace=True
        )
        _safe_write(thermo_frame, out_path)

    print("\nParsing complete.")
