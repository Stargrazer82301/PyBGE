# Imports
import pybge
import datetime

# Provide the path to where tesseract is installed on your system
tesseract_cmd = '/opt/homebrew/bin/tesseract'

# State name of directory to hold all the outputs PyBGE will produce
output_dir = 'PyBGE_Output'

# Main PyBGE call, which will scrape, parse tabulate, and collate usage data
pybge.run(output_dir,
          'satisfied_bge_customer@gmail.com',
          'hax0rpa$$wurd',
          '3D2c3beab5-0b88-22eu-918c-0300170a5887',
          date_start = datetime.date(2025, 12, 1),
          date_end = datetime.date(2026, 3, 20),
          tesseract_cmd = tesseract_cmd)

# Function to train model to learn usage patterns based on weather
pybge.correlate('output_dir',
                predict_days = 10)

# Function to use the model trained above, and forecasts, to predict usage
pybge.forecast(output_dir,
               39.33125,
               -76.63248,
               alt=68,
               forecast_days = 30,
               freedom_units = False,)