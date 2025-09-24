"""
Ameren WebFTP CSV Downloader

How it works:
1) Logs in to Ameren WebFTP using Selenium.
2) Finds all ".csv" links on the known folder page.
3) Uses the logged-in cookies with `requests` to stream-download each CSV.
4) Skips files that already exist locally.
"""

import contextlib
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
    print("Logging in…")
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
    print(f"Login successful → {driver.current_url}")
    return driver.get_cookies()


def list_csv_urls_in_folder(driver: webdriver.Chrome, folder_url: str) -> list[str]:
    """Visit folder page and collect all anchor hrefs ending with .csv."""
    print(f"Opening folder: {folder_url}")
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
    seen, unique = set(), []
    for u in csv_urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)

    print(f"Found {len(unique)} CSV file(s).")
    return unique


def make_authenticated_session(cookies: list[dict], referer: str) -> requests.Session:
    """Build a requests.Session that carries Selenium cookies and headers."""
    s = requests.Session()
    for c in cookies:
        # Domain/path can be added if needed
        s.cookies.set(c["name"], c["value"])
    s.headers.update({"User-Agent": BROWSER_USER_AGENT, "Referer": referer, "Accept": "*/*"})
    return s


def _print_content_headers(r: requests.Response) -> int | None:
    """Log content-type/length if present; return total_bytes (or None)."""
    ctype = r.headers.get("content-type", "")
    clen = r.headers.get("content-length")
    if ctype:
        print(f"Content-Type:   {ctype}")
    if clen:
        total_bytes = int(clen)
        print(f"Content-Length: {total_bytes:,} bytes (~{total_bytes / 1024 / 1024 / 1024:.2f} GB)")
        return total_bytes
    return None


def _stream_to_file(r: requests.Response, filepath: Path, total_bytes: int | None) -> int:
    """Write response body to `filepath` in 1 MB chunks; print 50MB progress; return size on disk."""
    chunk_size = 1024 * 1024  # 1 MB
    downloaded = 0
    with filepath.open("wb") as f:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            f.write(chunk)
            downloaded += len(chunk)
            # Progress every ~50MB if total size known
            if total_bytes and downloaded % PROGRESS_STEP_BYTES == 0:
                pct = (downloaded / total_bytes) * 100
                downloaded_mb = downloaded // (1024 * 1024)
                total_mb = total_bytes // (1024 * 1024)
                print(f"Progress: {pct:5.1f}% ({downloaded_mb:,} MB / {total_mb:,} MB)")
    return filepath.stat().st_size


def download_file_with_cookies(file_url: str, cookies: list[dict], out_dir: Path) -> bool:
    """
    Stream one CSV to disk. Prints progress roughly every 50 MB (original behavior).
    Returns True on success (non-empty file), False on failure (and removes partial).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = file_url.rstrip("/").split("/")[-1]
    filepath = out_dir / filename

    if filepath.exists():
        print(f"[skip] {filename} already exists")
        return True

    print(f"Downloading: {filename}")
    print(f"From:        {file_url}")

    session = make_authenticated_session(cookies, referer=file_url.rsplit("/", 1)[0] + "/")

    try:
        with session.get(file_url, stream=True, timeout=60) as r:
            r.raise_for_status()

            # Log headers & get total size (may be None)
            total_bytes = _print_content_headers(r)

            # Stream to disk with your 50MB progress behavior
            size_on_disk = _stream_to_file(r, filepath, total_bytes)

        print(f"Finished:   {filename} ({size_on_disk:,} bytes)")

        if size_on_disk == 0:
            print("File is empty; removing.")
            filepath.unlink(missing_ok=True)
            return False
        else:
            return True

    except Exception as e:
        print(f"Download failed for {filename}: {e}")
        # Clean up partial file so future runs start cleanly
        with contextlib.suppress(Exception):
            filepath.unlink(missing_ok=True)
        return False


def download_all_csvs(cookies: list[dict], driver: webdriver.Chrome, out_dir: Path, folder_url: str) -> tuple[int, int]:
    """Find every .csv on the folder page and download each one."""
    urls = list_csv_urls_in_folder(driver, folder_url)
    ok, fail = 0, 0
    for u in urls:
        if download_file_with_cookies(u, cookies, out_dir):
            ok += 1
        else:
            fail += 1
    print(f"Summary: {ok} succeeded, {fail} failed.")
    return ok, fail


def main() -> None:
    """Load credentials → login → list .csv links → download them → quit."""
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
