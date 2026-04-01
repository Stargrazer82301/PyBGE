"""
PyBGE -- BGE energy-usage scraper, parser, weather collator, correlator and forecaster.

Typical usage
-------------
import pybge

# Scrape, parse and collate
pybge.run(path="/data", username="you@example.com", password="secret")

# Train weather-to-usage regression model
pybge.correlate(data_dir="/data")

# Forecast next 30 days
pybge.forecast(data_dir="/data")
"""

from .api import run, correlate, forecast

__all__ = ["run", "correlate", "forecast"]
__version__ = "0.2.0"
