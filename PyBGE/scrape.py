"""
pybge.scrape
------------
Scrapes hourly energy-usage bar-chart screenshots from the BGE website for
a range of dates and saves them as PNG files.
"""

from __future__ import annotations

import gc
import os
import time
import datetime
from typing import Optional

import numpy as np
import pandas as pd
import PIL.Image
import pytesseract
import selenium.webdriver
import selenium.webdriver.chrome.options
import selenium.webdriver.common.action_chains
import selenium.webdriver.common.by
import selenium.webdriver.support.ui
import selenium.webdriver.support.expected_conditions
import sys
if sys.platform != "win32":
    import undetected_chromedriver as uc

from .utils import time_est, configure_tesseract


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _login(driver: selenium.webdriver.Chrome, username: str, password: str, img_dir: str) -> None:
    """Log into BGE and handle MFA if required."""
    By = selenium.webdriver.common.by.By
    EC = selenium.webdriver.support.expected_conditions
    Wait = selenium.webdriver.support.ui.WebDriverWait

    # Navigate to login page
    time.sleep(10)
    driver.get("https://secure.bge.com/accounts/login")
    time.sleep(20)
    try:
        username_box = driver.find_element(By.ID, "signInName")
        password_box = driver.find_element(By.ID, "password")
        login_button = driver.find_element(By.ID, "next")
    except:
        driver.save_screenshot(os.path.join(img_dir, 'debug_screenshot.png'))
        raise Exception('BGE login page not as expected; debug screenshot output to file')

    username_box.send_keys(username)
    password_box.send_keys(password)
    login_button.click()
    time.sleep(20)

    # Check for MFA prompt -- identical logic to original script
    code_box_search = driver.find_elements(By.ID, "emailVerificationCode")
    if len(code_box_search) == 1:
        code_box = code_box_search[0]
        continue_button = driver.find_element(By.ID, "continueButton")
        mfa_code = input("Please provide two-factor authentication code: ")
        mfa_code = mfa_code.replace(" ", "")
        if (len(mfa_code) != 6) or (not mfa_code.isnumeric()):
            raise Exception("Code not valid")
        code_box.send_keys(mfa_code)
        continue_button.click()
        time.sleep(20)
    print("Loaded " + driver.current_url)
    if 'secure.bge.com/accounts/dashboard' not in driver.current_url:
        driver.save_screenshot(os.path.join(img_dir, 'debug_screenshot.png'))
        raise Exception('BGE login page not as expected; debug screenshot output to file')


def _usage_url(fuel: str, date_string: str, service_agreement_uuid: str) -> str:
    """Build the BGE usage page URL for a given fuel and date."""
    return (
        "https://secure.bge.com/MyAccount/MyBillUsage/Pages/Secure/ViewMyUsage.aspx"
        f"?ou-data-browser=%2Fusage%2F{fuel.lower()}%2Fday%2F{date_string}"
        f"%3FserviceAgreementUuid%3D{service_agreement_uuid}"
    )


def _screenshot_is_good(img_path: str, check_bounds: tuple) -> bool:
    """Return True if an existing screenshot appears complete and usable."""
    check_img = PIL.Image.open(img_path)
    check_img = check_img.crop(check_bounds)
    check_str = pytesseract.image_to_string(check_img).lower()
    return (
        ("am" in check_str or "pm" in check_str or "day" in check_str)
        and ("kwh" in check_str or "therms" in check_str)
    )


def _remove_hover(driver: selenium.webdriver.Chrome, check_count: int) -> None:
    """Inject JS to hide any tooltip overlay that might be covering the chart."""
    css = """
    .weather-detailed.tooltip.simple {
        display: none !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    """
    driver.execute_script(
        f"""
        var style = document.createElement('style');
        style.type = 'text/css';
        style.appendChild(document.createTextNode(`{css}`));
        document.head.appendChild(style);
        """
    )
    driver.execute_script(
        """
        var tt = document.querySelector('div.weather-detailed.tooltip.simple');
        if (tt) { tt.remove(); }
        """
    )
    driver.execute_script(
        """
        var tt = document.querySelector('div.weather-detailed.tooltip.simple');
        if (tt) {
            tt.innerHTML = '';
            tt.style.opacity = '0';
            tt.style.pointerEvents = 'none';
        }
        """
    )
    actions = selenium.webdriver.common.action_chains.ActionChains(driver)
    mult = -1 if check_count % 2 == 0 else 1
    try:
        actions.move_by_offset(mult * 500, mult * 500).perform()
    except Exception:
        actions.move_by_offset(-mult * 500, -mult * 500).perform()
    time.sleep(5)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

def scrape(
    img_dir: str,
    data_dir: str,
    username: str,
    password: str,
    service_agreement_uuid: str,
    tesseract_cmd: Optional[str],
    date_start: Optional[datetime.date],
    date_end: datetime.date,
) -> None:
    """Scrape BGE usage screenshots and save them to *img_dir*."""
    configure_tesseract(tesseract_cmd)

    # ------------------------------------------------------------------ #
    # Determine date range
    # ------------------------------------------------------------------ #
    collated_path = os.path.join(data_dir, "BGE_Collated.csv.gz")
    if date_start is None:
        if os.path.exists(collated_path):
            thermo_hourly = pd.read_csv(collated_path, parse_dates=["date"])
            good_rows = np.where(
                (thermo_hourly["date_hour"] == 23)
                & (~np.isnan(thermo_hourly["usage_kwh"]))
                & (~np.isnan(thermo_hourly["usage_therm"]))
            )
            date_start = (
                thermo_hourly.iloc[good_rows]["date"].max().to_pydatetime().date()
            )
        else:
            raise FileNotFoundError(
                "date_start was not supplied and BGE_Collated.csv.gz does not exist "
                f"in {data_dir!r}.  Please provide an explicit date_start."
            )

    date_list = [
        date_start + datetime.timedelta(days=i)
        for i in range((date_end - date_start).days + 1)
    ]

    fuels = ["Electricity", "Gas"]
    check_bounds = (400, 400, 1300, 850)

    # ------------------------------------------------------------------ #
    # Start Selenium -- identical to original script
    # ------------------------------------------------------------------ #
    # On Windows, standard Selenium works fine. On Mac/Linux, use
    # undetected_chromedriver to bypass BGE's headless Chrome bot detection.
    import subprocess, re
    if sys.platform == "win32":
        options = selenium.webdriver.chrome.options.Options()
        options.add_argument("--headless")
        driver = selenium.webdriver.Chrome(options=options)
    else:
        options = uc.ChromeOptions()
        options.add_argument("--headless")

        # Auto-detect Chrome major version so ChromeDriver always matches
        chrome_version = None
        try:
            if sys.platform == "darwin":
                result = subprocess.run(
                    ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--version"],
                    capture_output=True, text=True
                )
            else:
                result = subprocess.run(
                    ["google-chrome", "--version"],
                    capture_output=True, text=True
                )
            match = re.search(r"(\d+)\.\d+\.\d+\.\d+", result.stdout + result.stderr)
            if match:
                chrome_version = int(match.group(1))
                print(f"Detected Chrome version: {chrome_version}")
        except Exception as e:
            print(f"Could not auto-detect Chrome version ({e}); letting undetected_chromedriver decide")

        driver = uc.Chrome(options=options, version_main=chrome_version)

    driver.set_window_size(1440, 1600)

    # ------------------------------------------------------------------ #
    # Pre-check pass -- identical logic to original script
    # ------------------------------------------------------------------ #
    date_list_keep = []
    time_list = [time.time()]
    for d, date in enumerate(date_list):
        date_string = date.strftime("%Y-%m-%d")
        for fuel in fuels:
            check_img_path = os.path.join(img_dir, f"{date_string}_{fuel}.png")
            if not os.path.exists(check_img_path):
                date_list_keep.append(date)
            else:
                check_img = PIL.Image.open(check_img_path)
                check_img = check_img.crop(check_bounds)
                check_str = pytesseract.image_to_string(check_img).lower()
                if (
                    ("am" in check_str or "pm" in check_str or "day" in check_str)
                    and ("kwh" in check_str or "therms" in check_str)
                ):
                    continue
                else:
                    date_list_keep.append(date)

        time_list.append(time.time())
        if len(time_list) > 2:
            est = time_est(time_list, len(date_list))
            pct = 100.0 * (d + 1) / len(date_list)
            print(
                f"\rPre-checking {date_string}; {pct:.1f}% complete; "
                f"estimate done {est}",
                end="",
            )
        if len(time_list) % 10 == 0:
            gc.collect()

    date_list_keep = list(set(date_list_keep))
    date_list_scrape = date_list_keep
    print(f"\nDates requiring scraping: {len(date_list_scrape)}")

    # ------------------------------------------------------------------ #
    # Main scraping loop -- identical logic to original script
    # ------------------------------------------------------------------ #
    time_list = [time.time()]
    for d, date in enumerate(date_list_scrape):

        # Initiate login for first date, and then once every years worth thereafter
        if (d == 0) or ((len(time_list) % 365) == 0):
            _login(driver, username, password, img_dir)

            # Set a short page load timeout AFTER login so driver.get() on the
            # JS-heavy usage pages gives up quickly rather than blocking for 120s.
            # The login page needs the default (unlimited) timeout to load fully.
            driver.set_page_load_timeout(20)

        date_string = date.strftime("%Y-%m-%d")

        for fuel in fuels:

            try:
                driver.get(_usage_url(fuel, date_string, service_agreement_uuid))
            except Exception:
                pass
            time.sleep(10)

            img_path = os.path.join(img_dir, f"{date_string}_{fuel}.png")
            check_count = 0
            check_done = False

            while not check_done:
                try:
                    driver.save_screenshot(img_path)
                except Exception as e:
                    print(f"save_screenshot failed with: {e}")
                    print("Attempting emergency screenshot to debug_screenshot.png")
                    try:
                        driver.save_screenshot(os.path.join(os.path.dirname(img_path), "debug_screenshot.png"))
                        print("Emergency screenshot saved")
                    except Exception as e2:
                        print(f"Emergency screenshot also failed with: {e2}")
                    breakpoint()
                check_img = PIL.Image.open(img_path).crop(check_bounds)
                check_str = pytesseract.image_to_string(check_img).lower()

                # Remove hover tooltip if present
                if ":00" in check_str and ("00am" in check_str or "00pm" in check_str):
                    print("Re-rendering to remove hover box")
                    _remove_hover(driver, check_count)
                    check_count += 1

                # Check if expected text is present
                if "loading" in check_str:
                    time.sleep(5)
                    check_count += 1
                elif ("sorry" in check_str) and ("information" in check_str):
                    check_done = True
                elif (
                    ("am" in check_str or "pm" in check_str or "day" in check_str)
                    and ("kwh" in check_str or "therms" in check_str)
                ):
                    check_done = True
                else:
                    time.sleep(5)
                    check_count += 1
                    if check_count > 10:
                        check_done = True

                # If server error message appears, reload to try again
                if "server error" in check_str:
                    driver.get(_usage_url(fuel, date_string, service_agreement_uuid))
                    time.sleep(10)
                    check_count += 1

        time_list.append(time.time())
        if len(time_list) > 2:
            est = time_est(time_list, len(date_list_scrape))
            pct = 100.0 * (d + 1) / len(date_list_scrape)
            print(
                f"\rScraping {date_string}; {pct:.1f}% complete; "
                f"estimate done {est}",
                end="",
            )
        if len(time_list) % 10 == 0:
            gc.collect()

    driver.quit()
    print("\nScraping complete.")
