# Imports
import pybge
import datetime

# Provide the path to where tesseract is installed on your system (or None if tesseract in PATH)
tesseract_cmd = '/opt/homebrew/bin/tesseract'

# State name of directory to hold all the outputs PyBGE will produce
output_dir = 'PyBGE_Output'

# Main PyBGE call, which will scrape, parse tabulate, and collate usage data
pybge.run(output_dir,
          'satisfied_bge_customer@bmail.com',           # Change to your BGE account email
          'hax0rpa$$wurd',                              # Change to your BGE account passowrd
          '3D2c3beab5-0b88-22eu-918c-0300170a5887',     # Change to your BGE account UUID (see readme)
          date_start = datetime.date(2025, 11, 1),
          date_end = datetime.date(2026, 3, 30),
          tesseract_cmd = tesseract_cmd)

# Function to train model to learn usage patterns based on weather
pybge.correlate(output_dir,
                predict_days = 30)

# Function to use the model trained above, and forecasts, to predict usage
pybge.forecast(output_dir,
               39.33125,
               -76.63248,
               alt=68,
               forecast_days = 30,
               freedom_units = False,)
