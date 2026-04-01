# PyBGE

A small Python package to scrape, parse, and tabulate your hourly energy usage data from the BGE (Baltimore Gas and Electric) website, then do a little analysis with it to understand & predict household energy usage. 

## Introduction

I made this code as a little personal project, because we had done some energy efficiency upgrades to our house, and I wanted to understand how much of an impact these were having on our bills. The BGE  account website provides very useful hour-by-hour usage plots for electricity and gas. Cool! However, they don't provide a way to download that usage data in bulk.

`PyBGE` gets around this problem by logging into the BGE website on your behalf, navigating to the hourly usage plot for a given day, *screenshotting* it that plot, parsing the screenshot to get your hour-by-hour energy usage, and then repeating that process for every day within a range your define.`PyBGE` then tabulates all of this data, and cross-references it with publicly available weather data (via the Meteostat service).

`PyBGE` also provides functionality to let you use this data for some analysis!

Firstly, `PyBGE` can apply a range of (fairly basic) machine learning models to the usage and weather data, to train a model to predict your usage based on conditions. You can define training versus comparison date ranges, to see if your more recent usage is different from what the model would predict based on your past usage patterns. This allowed me to figure out how much my energy improvements were saving from my bills!

Secondly, `PyBGE` can combine weather forecasts and historical weather data (again using Metostat), to use its model of your usage to predict how much energy you are likely to use in the near future (I personally find this more accurate than BGE's own estimate of what my usage will be over a given billing period).

## Requirements

* Python >=3.11
* [Google Chrome](https://www.google.com/chrome/) and a matching  [ChromeDriver](https://chromedriver.chromium.org/) in `PATH`. Iif you have Chrome installed on your computer, you probably satisfy this requirement.
* The [Tesseract](https://github.com/tesseract-ocr/tesseract) open-source OCR (Optical Character Recognition) library. When running `PyBGE`, you will need to provide the path to the tesseract executable (see Quick Star section below).

## Installation

Install by downloading this repository, navigating in the terminal to the directory that contains `pyproject.toml`, then running the command:

```
pip install .
```
## Big Important Warnings

I made this code for personal use. Please do not use it to DoS the BGE website, or do anything else silly/nefarious. I appreciate BGE providing the hourly usage data they do, even though I *really* wish they would allow us to download it in bulk.

`PyBGE` needs you to provide, in plaintext, the email and password of your BGE account, so that it can log in to it and do its job. It is a **bad idea** to put your email and password into some random Python code you got off GitHub.

If you have MFA enabled on your BGE account (*and you should*), then `PyBGE` will also ask you to input the MFA code that BGE sends you when it tries to log in to your account. **This is a TERRIBLE idea**. You absolutely *should not* type your MFA code into some random Python code you got off GitHub! Why would you even do that? I could be a terrible person with evil plans! For heaven's sake, I chose to spend part of my limited time on this Earth writing code to scrape energy usage data off the website of a utility company - I clearly don't have good judgement! You absolutely should *not* trust my software with your MFA code.

I wrote `pyBGE` myself; however it was very much tailored to my own setup. I therefore used an LLM to refactor it into the more generic package structure you see here. I have a gaming PC I use regularly, so I judged that my net "compute and energy usage for silly purposes" budget was not meaningfully impacted by this use of an LLM.

## Quick Start

```python
import pybge

pybge.run(
    path="/path/to/my/data",          # directory for CSV outputs
    username="you@example.com",       # BGE account e-mail
    password="yourpassword",          # BGE account password
)
```

On first run the directory must exist but can be empty — PyBGE will create the `screenshots/` sub-directory automatically.  After the first run the two output files `BGE_Usage.csv.gz` and `BGE_Collated.csv.gz` will be present, and subsequent calls append only the newly available data.

## `pybge.run()` Guide

```python
pybge.run(
    path,                               # (required) data directory
    username,                           # (required) BGE login e-mail
    password,                           # (required) BGE login password
    service_agreement_uuid              # (required)BGE service agreement UUID in usage page URLs (see below),
    *,
    lat=39.3328,                        # property latitude (for weather data)
    lon=-76.6327,                       # property longitude (for weather data)
    alt=68,                             # property altitude in metres (for weather data)    
    tesseract_cmd=None,                 # e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    date_start=None,                    # datetime.date; None = auto-detect (auto-detect does not work for first run)
    date_end=None,                      # datetime.date; None = three days before current date
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Directory where CSV files are stored |
| `username` | `str` | — | BGE account e-mail |
| `password` | `str` | — | BGE account password |
| `service_agreement_uuid` | `str` | — | BGE service agreement UUID in usage page URLs |
| `lat` | `float` | `39.3328` | Latitude (degrees) for Meteostat weather lookup |
| `lon` | `float` | `-76.6327` | Longitude (degrees) for Meteostat weather lookup |
| `alt` | `float` | `68` | Altitude (metres) for Meteostat |
| `tesseract_cmd` | `str \| None` | `None` | Absolute path to `tesseract` binary; `None` = use `PATH` |
| `date_start` | `datetime.date \| None` | `None` | Earliest date to process; auto-detected from existing data when `None` |
| `date_end` | `datetime.date \| None` | `None` | Latest date to process; defaults to today |
| `screenshots_subdir` | `str` | `"screenshots"` | Sub-directory inside `path` for PNG screenshots |

Latitude, longitute, and altitude defaults values for weather lookup correspond approxmately to the Royal Farms global headquarters. This seemed an appropriately *Baltimore*  default location).

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
