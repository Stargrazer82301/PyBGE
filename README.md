# PyBGE

A small Python package to scrape, parse, and tabulate your hourly energy usage data from the BGE (Baltimore Gas and Electric) website, then do a little analysis with it to understand & predict household energy usage. 

## Introduction

I made this code as a little personal project, because we had done some energy efficiency upgrades to our house, and I wanted to understand how much of an impact these were having on our bills. The BGE  account website provides very useful hour-by-hour usage plots for electricity and gas. Cool! However, they don't provide a way to download that usage data in bulk.

PyBGE gets around this problem by logging into the BGE website on your behalf, navigating to the hourly usage plot for a given day, *screenshotting* that plot, parsing the screenshot to extract your hour-by-hour energy usage, and then repeating that process for every day within a date range you define.PyBGE then tabulates all of this usage data, and cross-references it with publicly available weather data (via the Meteostat service).

PyBGE also provides functionality to let you use this data for some analysis!

Firstly, PyBGE can apply a range of (fairly basic) machine learning models to the usage and weather data, to train a model to predict your usage based on conditions. You can define training versus comparison date ranges, to see if your more recent usage is different from what the model would predict based on your past usage patterns. This allowed me to figure out how much my energy improvements were saving from my bills!

Secondly, PyBGE can combine weather forecasts and historical weather data (again using Metostat), to use its model of your usage to predict how much energy you are likely to use in the near future (I personally find this more accurate than BGE's own estimate of what my usage will be over a given billing period).

## Requirements

* Python >=3.11
* [Google Chrome](https://www.google.com/chrome/) and a matching  [ChromeDriver](https://chromedriver.chromium.org/) in `PATH`. Iif you have Chrome installed on your computer, you probably satisfy this requirement.
* The [Tesseract](https://github.com/tesseract-ocr/tesseract) open-source OCR (Optical Character Recognition) library. When running PyBGE, you will need to provide the path to the tesseract executable (see Quick Star section below).

## Installation

Install by downloading this repository, navigating in the terminal to the directory that contains `pyproject.toml`, then running the command:

```
pip install .
```

## Big Important Warnings

I made this code for personal use. Please do not use it to DoS the BGE website, or do anything else silly/nefarious. I appreciate BGE providing the hourly usage data they do, even though I *really* wish they would allow us to download it in bulk.

PyBGE needs you to provide, in plaintext, the email and password of your BGE account, so that it can log in to it and do its job. It is a **bad idea** to put your email and password into some random Python code you got off GitHub.

If you have MFA enabled on your BGE account (*and you should*), then PyBGE will also ask you to input the MFA code that BGE sends you when it tries to log in to your account. **This is a TERRIBLE idea**. You absolutely *should not* type your MFA code into some random Python code you got off GitHub! Why would you even do that? I could be a terrible person with evil plans! For heaven's sake, I chose to spend part of my limited time on this Earth writing code to scrape energy usage data off the website of a utility company - I clearly don't have good judgement! You absolutely should *not* trust my software with your MFA code.

I wrote PyBGE myself; however it was very much tailored to my own setup. I therefore used an LLM to refactor it into the more generic package structure you see here. I have a gaming PC I use regularly, so I judged that my net "compute and energy usage for silly purposes" budget was not meaningfully impacted by this use of an LLM.

## Quick Start

Here is example code that illustrates a standard run of PyBGE. This code can also be found in the script `example.py` in the repository.

```
# Imports
import pybge
import datetime

# State name of directory to hold all the outputs PyBGE will produce
output_dir = 'PyBGE_Output'

# Provide path to where tesseract is installed on your system (if not already in PATH)
tesseract_cmd = '/opt/homebrew/bin/tesseract'

# Main PyBGE call, which will scrape, parse tabulate, and collate usage data
pybge.run(output_dir,
          'satisfied_bge_customer@bmail.com',
          'hax0rpa$$wurd',
          '3D2c3beab5-0b88-22eu-918c-0300170a5887',
          date_start = datetime.date(2025, 12, 1),
          date_end = datetime.date(2026, 3, 20),
          tesseract_cmd = tesseract_cmd)

# Function to train model to learn usage patterns based on weather
pybge.correlate('output_dir',
                predict_days = 120)

# Function to use the model trained above, and forecasts, to predict usage
pybge.forecast(output_dir,
               39.33125,
               -76.63248,
               alt=68,
               forecast_days = 30,
               freedom_units = False)
```

On first run of `pybge.run()` the output directory must exist, but can be empty — PyBGE will create the `screenshots/` sub-directory automatically.  After the first run the two output data files `BGE_Usage.csv.gz` and `BGE_Collated.csv.gz` will be present. On subsequent runs of `pybge.run()` to the same output directory, but with a different date range, the new dates will be appended to the existing output data files.

## `pybge.run()` Usage

The docstring for `pybge.run()` contains full explaination of the input parameters; they are summarised here for cenvenience:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Directory where CSV files are stored |
| `username` | `str` | — | BGE account e-mail |
| `password` | `str` | — | BGE account password |
| `service_agreement_uuid` | `str` | — | BGE service agreement UUID in usage page URLs (see below) |
| `lat` | `float` | `39.3328` | Latitude (degrees) for Meteostat weather lookup |
| `lon` | `float` | `-76.6327` | Longitude (degrees) for Meteostat weather lookup |
| `alt` | `float` | `68` | Altitude (metres) for Meteostat weather lookup |
| `tesseract_cmd` | `str \| None` | `None` | Absolute path to `tesseract`; if `None`, `PATH` is used |
| `date_start` | `datetime.date \| None` | `None` | Earliest date to process; auto-detected from existing data when `None` (auto-detect does not work for first run) |
| `date_end` | `datetime.date \| None` | `None` | Latest date to process; defaults to three days before current date |
| `dollars_per_kwhr` | `str` | `None` | Electricity cost per kWh |
| `dollars_per_therm` | `str` | `None` | Sub-directory inside `path` for PNG screenshots |
| `freedom_units` | `bool` | `False` | Sub-directory inside `path` for PNG screenshots |

Latitude, longitute, and altitude default values for weather lookup correspond approxmately to the Royal Farms headquarters. This seemed an appropriately *Baltimore*  default location. If you want more-accurate weather for the specific location of your address, you can find the latitude and longitude by, eg, right-clicking on a locaiton on Google maps.

If tesseract is in your path, you don't have to provide a value for the `tesseract_cmd` kwarg. Otherwise, typing `which tesseract` at the terminal (on UNIX systems) will tell you the path to your tesseract installation.

The output data files record the energy usage for your electricity usage in kWh, and gas usage in therms. They also record your combined usage in units of "equivalent kilowatt-hours", or ekWh, by using the approximate equivalency between 

The service agreement UUID is BGE's internal identifier for your account. It is embedded in the URL of your hourly usage page and tells the BGE website which meter/address to show data for. How to find yours:

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

       3D2c3beab5-0b88-22eu-918c-0300170a5887

### Output files

| File | Description |
|---|---|
| `BGE_Usage.csv.gz` | Hourly electricity (kWh), gas (therm), and effective kWh |
| `BGE_Collated.csv.gz` | Usage + Meteostat weather + 24-hour rolling statistics |
| `screenshots/<date>_<Fuel>.png` | Raw BGE portal screenshots |

### `pybge.run()` Pipeline Overview

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

## `pybge.correlate()` Usage

`pybge.correlate()` uses the tabulated outputs of `pybge.run`, and applies one of several possible machine learning models (mostly from `scikit-learn`) to understand the underyling relationship between weather conditions and energy usage. A certain number of the most recent days can be excluded from this modelling, to see if they deviate from the historical relationship.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_dir` | `str` | — | Directory where PyBGE CSV files are stored |
| `method` | `str` | `"GPR"` | Method to use to for modelling usage vs weather (`"GPR"`, `"linear"`, `"BRR"`, `"ARDR"`, `"Random-Forest"`, `"Quantile-Forest"`, `"Neural-Net"`) |
| `predict_days` | `int` | `30` | How many days into the past to predict (for comparing to model) |
| `dollars_per_kwhr` | `float` | `None` | None | Directory where CSV files are stored |
| `dollars_per_therm` | `float` | `None` | None| Directory where CSV files are stored |
| `plot_unit` | `str` | `"auto"` | `"ekwh"`, `"dollars"`, or `"auto"` |
| `freedom_units` | `bool` | `False` | Change temperature units in plots fron Censius to Farenheit |

## `pybge.forecast()` Usage

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | — | Directory where CSV files are stored |


