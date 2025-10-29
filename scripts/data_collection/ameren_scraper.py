"""
Ameren WebFTP CSV Downloader

How it works:
1) Logs in to Ameren WebFTP using Selenium.
2) Finds all ".csv" links on the known folder page.
3) Uses the logged-in cookies with `requests` to stream-download each CSV.
4) Skips files that already exist locally.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

CREDENTIALS_FILE = Path("scripts/.secrets/config.yaml")
FOLDER_URL = "https://webftp.ameren.com/file/d/Switchbox/"
DOWNLOAD_DIR = Path("data/raw/ameren")

CHROME_BINARY = "/usr/bin/chromium"
CHROMEDRIVER_BINARY = "/usr/bin/chromedriver"

BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

PROGRESS_STEP_BYTES = 50 * 1024 * 1024

# Module logger
logger = logging.getLogger(__name__)


def load_credentials(path: Path) -> dict:
    """Load YAML with keys: username, password."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_driver() -> webdriver.Chrome:
    """Start headless Chrome for Selenium."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.binary_location = CHROME_BINARY
    service = Service(CHROMEDRIVER_BINARY)
    return webdriver.Chrome(service=service, options=options)


def login_and_get_cookies(driver: webdriver.Chrome, username: str, password: str) -> list[dict]:
    """Open login page, submit credentials, wait for redirect, return cookies."""
    logger.info("Logging in…")
    driver.get("https://webftp.ameren.com/login")

    # Wait until the username input exists on the page.
    username_field = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.NAME, "username")))
    password_field = driver.find_element(By.NAME, "password")

    username_field.send_keys(username)
    password_field.send_keys(password)

    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    driver.execute_script("arguments[0].click();", submit_button)

    # Consider login successful when we leave the login URL.
    WebDriverWait(driver, 15).until(lambda d: d.current_url != "https://webftp.ameren.com/login")
    logger.info("Login successful → %s", driver.current_url)
    return driver.get_cookies()


def list_csv_urls_in_folder(driver: webdriver.Chrome, folder_url: str) -> list[str]:
    """Visit folder page and collect all anchor hrefs ending with .csv."""
    logger.info("Opening folder: %s", folder_url)
    driver.get(folder_url)

    # Handle lazy-loading by scrolling until page height stops changing.
    last_h = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.8)
        new_h = driver.execute_script("return document.body.scrollHeight")
        if new_h == last_h:
            break
        last_h = new_h

    # Grab all <a> elements and keep those whose href ends with .csv
    csv_urls = []
    for a in driver.find_elements(By.TAG_NAME, "a"):
        href = a.get_attribute("href")
        if not href:
            continue
        abs_url = urljoin(folder_url, href)
        if abs_url.lower().endswith(".csv"):
            csv_urls.append(abs_url)

    # De-duplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for u in csv_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    logger.info("Found %d CSV file(s).", len(unique))
    return unique


def make_authenticated_session(cookies: list[dict], referer: str) -> requests.Session:
    """Build a requests.Session that carries Selenium cookies and headers."""
    s = requests.Session()
    for c in cookies:
        s.cookies.set(c["name"], c["value"])
    s.headers.update({"User-Agent": BROWSER_USER_AGENT, "Referer": referer, "Accept": "*/*"})
    return s


def _print_content_headers(r: requests.Response) -> int | None:
    """Log content-type/length if present; return total_bytes (or None)."""
    ctype = r.headers.get("content-type", "")
    clen = r.headers.get("content-length")
    if ctype:
        logger.info("Content-Type:   %s", ctype)
    if clen:
        total_bytes = int(clen)
        logger.info("Content-Length: %s bytes (~%.2f GB)", f"{total_bytes:,}", total_bytes / 1024 / 1024 / 1024)
        return total_bytes
    return None


def _stream_to_file(r: requests.Response, filepath: Path, total_bytes: int | None) -> int:
    """Write response body to `filepath` in 1 MB chunks; DEBUG-log ~50MB progress; return size on disk."""
    chunk_size = 1024 * 1024  # 1 MB
    downloaded = 0
    with filepath.open("wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            if total_bytes and downloaded % PROGRESS_STEP_BYTES == 0:
                pct = (downloaded / total_bytes) * 100
                downloaded_mb = downloaded // (1024 * 1024)
                total_mb = total_bytes // (1024 * 1024)
                logger.debug("Progress: %5.1f%% (%s MB / %s MB)", pct, f"{downloaded_mb:,}", f"{total_mb:,}")
    return filepath.stat().st_size


def download_file_with_cookies(file_url: str, cookies: list[dict], out_dir: Path) -> bool:
    """
    Stream one CSV to disk. Prints progress roughly every 50 MB.
    Returns True on success (non-empty file), False on failure (and removes partial).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = file_url.rstrip("/").split("/")[-1]
    filepath = out_dir / filename

    if filepath.exists():
        logger.info("[skip] %s already exists", filename)
        return True

    logger.info("Downloading: %s", filename)
    logger.info("From:        %s", file_url)

    session = make_authenticated_session(cookies, referer=file_url.rsplit("/", 1)[0] + "/")

    try:
        with session.get(file_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_bytes = _print_content_headers(r)
            size_on_disk = _stream_to_file(r, filepath, total_bytes)

        logger.info("Finished:   %s (%s bytes)", filename, f"{size_on_disk:,}")

        if size_on_disk == 0:
            logger.warning("File is empty; removing.")
            filepath.unlink(missing_ok=True)
            return False
        else:
            return True

    except Exception:
        # .exception() already includes the exception info + traceback.
        logger.exception("Download failed for %s", filename)
        with contextlib.suppress(Exception):
            filepath.unlink(missing_ok=True)
        return False


def download_all_csvs(cookies: list[dict], driver: webdriver.Chrome, out_dir: Path, folder_url: str) -> tuple[int, int]:
    """Find every .csv on the folder page and download each one."""
    urls = list_csv_urls_in_folder(driver, folder_url)
    ok, fail = 0, 0
    failed_urls: list[str] = []
    for u in urls:
        if download_file_with_cookies(u, cookies, out_dir):
            ok += 1
        else:
            fail += 1
            failed_urls.append(u)
    logger.info("Summary: %d succeeded, %d failed.", ok, fail)
    if failed_urls:
        logger.error("Failed downloads (%d):\n%s", len(failed_urls), "\n".join(f"  • {u}" for u in failed_urls))
    return ok, fail


def main() -> None:
    """Load credentials → login → list .csv links → download them → quit."""
    # Basic logging config; override with LOGLEVEL=DEBUG for more detail.
    logging.basicConfig(
        level=os.environ.get("LOGLEVEL", "INFO"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    creds = load_credentials(CREDENTIALS_FILE)
    username, password = creds["username"], creds["password"]

    driver = setup_driver()
    try:
        cookies = login_and_get_cookies(driver, username, password)
        download_all_csvs(cookies, driver, DOWNLOAD_DIR, FOLDER_URL)
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
