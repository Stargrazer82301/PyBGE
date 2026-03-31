# PyBGE

A Python package that scrapes, parses, and collates hourly electricity and gas
usage data from the BGE (Baltimore Gas & Electric) customer portal, then
enriches it with hourly Meteostat weather observations.

## Requirements

* Python 3.11+
* [Google Chrome](https://www.google.com/chrome/) and a matching
  [ChromeDriver](https://chromedriver.chromium.org/) on `PATH`
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and
  (on Windows) either on `PATH` or passed via `tesseract_cmd`

## Installation

```bash
pip install .
```

Or in editable / development mode:

```bash
pip install -e ".[dev]"
```

## Quick start

```python
import pybge

pybge.run(
    path="/path/to/my/data",          # directory for CSV outputs
    username="you@example.com",       # BGE account e-mail
    password="yourpassword",          # BGE account password
)
```

On first run the directory must exist but can be empty — PyBGE will create
the `screenshots/` sub-directory automatically.  After the first run the
two output files `BGE_Usage.csv.gz` and `BGE_Collated.csv.gz` will be
present, and subsequent calls append only the newly available data.

## `pybge.run()` — full signature

```python
pybge.run(
    path,                               # (required) data directory
    username,                           # (required) BGE login e-mail
    password,                           # (required) BGE login password
    *,
    lat=39.3328,                        # property latitude  (Meteostat)
    lon=-76.6327,                       # property longitude (Meteostat)
    alt=69,                             # property altitude in metres
    service_agreement_uuid="1b3beab5-0a89-11ee-918c-0200170a5779",
    tesseract_cmd=None,                 # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    date_start=None,                    # datetime.date; None = auto-detect
    date_end=None,                      # datetime.date; None = today
    screenshots_subdir="screenshots",   # sub-directory name for PNGs
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Directory where CSV files are stored |
| `username` | `str` | — | BGE account e-mail |
| `password` | `str` | — | BGE account password |
| `lat` | `float` | `39.3328` | Latitude for Meteostat weather lookup |
| `lon` | `float` | `-76.6327` | Longitude for Meteostat weather lookup |
| `alt` | `float` | `69` | Altitude (metres) for Meteostat |
| `service_agreement_uuid` | `str` | *(original value)* | BGE service agreement UUID in usage page URLs |
| `tesseract_cmd` | `str \| None` | `None` | Absolute path to `tesseract` binary; `None` = use `PATH` |
| `date_start` | `datetime.date \| None` | `None` | Earliest date to process; auto-detected from existing data when `None` |
| `date_end` | `datetime.date \| None` | `None` | Latest date to process; defaults to today |
| `screenshots_subdir` | `str` | `"screenshots"` | Sub-directory inside `path` for PNG screenshots |

## Pipeline overview

`pybge.run()` executes three steps in sequence:

1. **Scrape** (`pybge.scrape`) — Launches a headless Chrome browser, logs into
   the BGE portal (handling MFA if prompted), and saves one PNG screenshot per
   day per fuel type.  Existing good screenshots are skipped.

2. **Parse** (`pybge.parse`) — Reads each PNG with Pillow and extracts hourly
   bar heights via colour matching, with OCR for the y-axis scale label.
   Writes `BGE_Usage.csv.gz`.

3. **Collate** (`pybge.collate`) — Joins the usage data with hourly Meteostat
   weather (temperature, humidity, precipitation, wind, pressure, cloud cover)
   and computes 24-hour rolling totals/averages.  Writes `BGE_Collated.csv.gz`.

## Output files

| File | Description |
|---|---|
| `BGE_Usage.csv.gz` | Hourly electricity (kWh), gas (therm), and effective kWh |
| `BGE_Collated.csv.gz` | Usage + Meteostat weather + 24-hour rolling statistics |
| `screenshots/<date>_<Fuel>.png` | Raw BGE portal screenshots |

## Notes

* Two-factor authentication: if BGE triggers an e-mail MFA challenge the
  script will pause and prompt you in the terminal to enter the code.
* The `service_agreement_uuid` defaults to the value from the original
  scripts.  If your account has a different UUID, pass it explicitly.
* Meteostat weather coordinates default to the original script's location
  (39.3328 N, 76.6327 W, 69 m — the Roland Park area of Baltimore).
  Pass `lat`, `lon`, and `alt` for a different address.
