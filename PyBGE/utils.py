"""
pybge.utils
-----------
Shared helper utilities used across scrape / parse / collate modules.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


def time_est(time_list: list[float], total: int) -> str:
    """Estimate wall-clock time of completion, converging on recent rate.

    Parameters
    ----------
    time_list:
        List of ``time.time()`` values captured at each iteration.
    total:
        Total number of iterations expected.

    Returns
    -------
    Human-readable estimated completion time string.
    """
    time_array = np.array(time_list)
    time_list_log = np.log10(time_array)
    iter_list = np.arange(0.0, float(len(time_list)))
    time_fit_log = np.polyfit(iter_list, time_list_log, 1)

    # Anchor the fit to the most recent actual value
    time_latest_actual = time_list[-1]
    time_latest_predicted_log = (
        time_fit_log[0] * float(len(time_list) - 1) + time_fit_log[1]
    )
    time_fit_log[1] += np.log10(time_latest_actual) - time_latest_predicted_log

    time_end_log = time_fit_log[0] * total + time_fit_log[1]
    time_end = 10.0 ** time_end_log
    return time.strftime("%H:%M:%S %a %d %b %Y", time.localtime(time_end))


def configure_tesseract(tesseract_cmd: Optional[str]) -> None:
    """Point PyTesseract at a specific ``tesseract`` binary if requested.

    Parameters
    ----------
    tesseract_cmd:
        Absolute path to the tesseract executable, or ``None`` to leave
        PyTesseract's default auto-detection in place.
    """
    if tesseract_cmd is not None:
        import pytesseract  # local import so the package stays importable
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd


def c_to_f(celsius):
    """Convert Celsius values to Fahrenheit.  Works on scalars, arrays, and
    pandas Series alike."""
    return celsius * 9.0 / 5.0 + 32.0
