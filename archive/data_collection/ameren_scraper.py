"""
Enhanced Ameren WebFTP CSV Downloader with S3 Upload Integration

IMPORTANT: This script typically requires 2-3 runs to successfully download all
files due to external factors beyond my control:

1. Ameren's server intermittently blocks large file downloads after processing
   several GB of data, returning HTML error pages instead of CSV files
2. The WebFTP page uses heavy JavaScript that sometimes fails to load file
    listings on the first attempt.

The script checks S3 before downloading and skips files that already exist.
This means you can safely re-run the command multiple times until all files are
successfully uploaded.

Typical workflow:
- Run 1: Downloads 3 of 4 files (one fails due to server-side issues)
- Run 2: Skips the 3 successful files, downloads the remaining file

Workflow per file:
1. Check if CSV already exists in S3
2. Ask user for permission to proceed (unless --force)
3. Create isolated browser session with fresh authentication
4. Download CSV to local temp directory (with 3 automatic retries)
5. Upload to S3 in dated folder structure (ameren-data/YYYYMM/filename.csv)
6. Delete local file after successful upload
7. Close browser session

Usage:
    python ameren_downloader.py
    python ameren_downloader.py --force
    python ameren_downloader.py --bucket-name my-bucket
    python ameren_downloader.py --debug  # Enable debug logging

Known Limitations:
- Large file downloads (5-7 GB) occasionally fail due to network/server issues
- Page scraping intermittently returns 0 files; simply re-run if this occurs
- Ameren's server may rate-limit or invalidate sessions unpredictably
- If an S3 bucket is not provided, files are not saved locally--the program
just exits.
- If an S3 bucket is provided, the program will not check against local files
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import boto3
import requests
import yaml
from botocore.exceptions import ClientError
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ============================================================================
# Configuration
# ============================================================================

CREDENTIALS_FILE = Path(".secrets/config.yaml")
FOLDER_URL = "https://webftp.ameren.com/file/d/Switchbox/"
DOWNLOAD_DIR = Path("data/raw/ameren")
DEFAULT_S3_BUCKET = "smart-meter-data-sb"

CHROME_BINARY = "/usr/bin/chromium"
CHROMEDRIVER_BINARY = "/usr/bin/chromedriver"
BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

PROGRESS_STEP_BYTES = 50 * 1024 * 1024  # Print progress every 50MB

# Initialize logger
logger = logging.getLogger(__name__)


# ============================================================================
# Browser & Authentication
# ============================================================================


def load_credentials(path: Path) -> dict:
    """Load YAML credentials file with username and password."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_driver() -> webdriver.Chrome:
    """Initialize headless Chrome browser."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.binary_location = CHROME_BINARY
    service = Service(CHROMEDRIVER_BINARY)
    logger.debug("Initializing Chrome driver with headless configuration")
    return webdriver.Chrome(service=service, options=options)


def login_and_get_cookies(driver: webdriver.Chrome, username: str, password: str) -> list[dict]:
    """Login to Ameren WebFTP and return session cookies."""
    logger.info("Logging in to Ameren WebFTP...")
    driver.get("https://webftp.ameren.com/login")

    # Fill login form
    username_field = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.NAME, "username")))
    password_field = driver.find_element(By.NAME, "password")
    logger.debug("Found login form fields")
    username_field.send_keys(username)
    password_field.send_keys(password)

    # Submit and wait for redirect
    submit_button = driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]')
    driver.execute_script("arguments[0].click();", submit_button)
    logger.debug("Submitted login form, waiting for redirect...")
    WebDriverWait(driver, 15).until(lambda d: d.current_url != "https://webftp.ameren.com/login")

    logger.info("Login successful")
    return driver.get_cookies()


def list_csv_urls_in_folder(driver: webdriver.Chrome, folder_url: str) -> list[str]:
    """Scrape folder page and return list of CSV file URLs."""
    logger.info(f"Scanning folder: {folder_url}")
    driver.get(folder_url)

    # Wait for page content to load - look for table or file list elements
    try:
        WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "table")))
        logger.debug("Table element found on page")
    except TimeoutException:
        logger.warning("Table element not found, proceeding anyway...")

    # Additional wait for JavaScript to finish rendering
    logger.debug("Waiting for JavaScript to finish rendering...")
    time.sleep(5)

    # Scroll to trigger lazy-loading until page height stabilizes
    last_height = driver.execute_script("return document.body.scrollHeight")
    scroll_attempts = 0
    max_scroll_attempts = 10

    logger.debug("Starting scroll loop to trigger lazy-loading...")
    while scroll_attempts < max_scroll_attempts:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.5)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            logger.debug(f"Page height stabilized after {scroll_attempts + 1} scroll attempts")
            break
        last_height = new_height
        scroll_attempts += 1

    # Collect all CSV links
    csv_urls = []
    all_anchors = driver.find_elements(By.TAG_NAME, "a")
    logger.debug(f"Found {len(all_anchors)} total anchor elements on page")

    for anchor in all_anchors:
        href = anchor.get_attribute("href")
        text = anchor.text

        # Check multiple indicators that this is a CSV link
        is_csv = (
            (href and href.lower().endswith(".csv"))
            or (href and ".csv" in href.lower())
            or (text and text.lower().endswith(".csv"))
        )

        if is_csv and href:
            abs_url = urljoin(folder_url, href)
            if abs_url not in csv_urls:
                csv_urls.append(abs_url)
                logger.debug(f"Found CSV URL: {abs_url}")

    logger.info(f"Found {len(csv_urls)} CSV file(s)")

    # Debug info if no files found
    if not csv_urls:
        logger.warning(f"No CSV files found! Total anchor elements: {len(all_anchors)}")
        logger.debug("Sample of first 5 links:")
        for i, anchor in enumerate(all_anchors[:5]):
            logger.debug(f"  Link {i}: text='{anchor.text}' href='{anchor.get_attribute('href')}'")

    return csv_urls


# ============================================================================
# File Download
# ============================================================================


def make_authenticated_session(cookies: list[dict], referer: str) -> requests.Session:
    """Create requests session with Selenium cookies and headers."""
    session = requests.Session()
    for cookie in cookies:
        session.cookies.set(cookie["name"], cookie["value"])
    session.headers.update({"User-Agent": BROWSER_USER_AGENT, "Referer": referer, "Accept": "*/*"})
    logger.debug(f"Created authenticated session with {len(cookies)} cookies")
    return session


def is_valid_csv(filepath: Path) -> bool:
    """Check if downloaded file is actually a CSV and not an HTML error page."""
    if filepath.stat().st_size == 0:
        logger.debug(f"File {filepath.name} is empty (0 bytes)")
        return False

    # Check first 1KB for HTML markers
    with filepath.open("rb") as f:
        first_bytes = f.read(1024)
        if b"<!DOCTYPE html>" in first_bytes or b"<html" in first_bytes:
            logger.debug(f"File {filepath.name} contains HTML content instead of CSV")
            return False

    logger.debug(f"File {filepath.name} appears to be valid CSV")
    return True


def download_csv(file_url: str, cookies: list[dict], out_dir: Path, max_retries: int = 3) -> Path | None:
    """
    Download CSV file to local directory with retry logic.
    Returns filepath on success, None on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = file_url.rstrip("/").split("/")[-1]
    filepath = out_dir / filename

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            wait_time = 2**attempt  # Exponential backoff: 4, 8, 16 seconds
            logger.info(f"Retry attempt {attempt}/{max_retries} after {wait_time}s delay...")
            time.sleep(wait_time)
        else:
            logger.info(f"Downloading: {filename}")

        session = make_authenticated_session(cookies, referer=file_url.rsplit("/", 1)[0] + "/")

        try:
            with session.get(file_url, stream=True, timeout=120) as response:
                response.raise_for_status()

                # Get file size for progress tracking
                total_bytes = None
                content_length = response.headers.get("content-length")
                if content_length:
                    total_bytes = int(content_length)
                    size_gb = total_bytes / (1024**3)
                    logger.info(f"  Size: {total_bytes:,} bytes (~{size_gb:.2f} GB)")

                # Stream to disk with progress updates
                downloaded = 0
                with filepath.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Print progress every 50MB if size is known
                            if total_bytes and downloaded % PROGRESS_STEP_BYTES < 1024 * 1024:
                                percent = (downloaded / total_bytes) * 100
                                logger.info(f"  Progress: {percent:5.1f}%")

            # Verify file is valid CSV
            if not is_valid_csv(filepath):
                logger.error("Downloaded HTML error page instead of CSV (session expired)")
                filepath.unlink(missing_ok=True)
                continue  # Retry

        except Exception:
            logger.exception("Download failed")
            filepath.unlink(missing_ok=True)
            if attempt == max_retries:
                logger.exception("All retry attempts exhausted")
                return None
            # Continue to next retry attempt
        else:
            size = filepath.stat().st_size
            logger.info(f"Download complete: {size:,} bytes")
            return filepath

    return None


# ============================================================================
# S3 Operations
# ============================================================================


def extract_date_from_filename(filename: str) -> str | None:
    """Extract YYYYMM date from filename, returns None if not found."""
    base_name = filename.replace(".csv", "")
    matches = re.findall(r"\d{6}", base_name)

    for match in matches:
        year = int(match[:4])
        month = int(match[4:6])
        if 2000 <= year <= 2099 and 1 <= month <= 12:
            logger.debug(f"Extracted date '{match}' from filename '{filename}'")
            return match

    logger.debug(f"No valid date found in filename '{filename}'")
    return None


def get_s3_key(filename: str) -> str:
    """Generate S3 key with folder structure: ameren-data/YYYYMM/filename.csv"""
    date_str = extract_date_from_filename(filename)

    if date_str is None:
        logger.warning(f"Could not extract date from filename: {filename}")
        logger.warning("Expected format: filename with YYYYMM (e.g., usage_202401.csv)")
        logger.warning(f"File will be stored in 'undated' folder: ameren-data/undated/{filename}")
        folder = "undated"
    else:
        folder = date_str

    return f"ameren-data/{folder}/{filename}"


def file_exists_in_s3(s3_client, bucket: str, key: str) -> bool:
    """Check if file exists in S3 bucket."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        logger.debug(f"File exists in S3: s3://{bucket}/{key}")
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            logger.debug(f"File does not exist in S3: s3://{bucket}/{key}")
            return False
        raise
    else:
        return True


def upload_to_s3(s3_client, local_path: Path, bucket: str, key: str) -> bool:
    """Upload file to S3. Returns True on success."""
    try:
        file_size = local_path.stat().st_size
        size_gb = file_size / (1024**3)
        logger.info(f"Uploading to S3: s3://{bucket}/{key}")
        logger.info(f"  Size: {file_size:,} bytes (~{size_gb:.2f} GB)")

        s3_client.upload_file(str(local_path), bucket, key)
        logger.info("Upload complete")
    except Exception:
        logger.exception("Upload failed")
        return False
    else:
        return True


# ============================================================================
# User Interaction
# ============================================================================


def ask_overwrite_permission(filename: str, s3_key: str, bucket: str) -> bool:
    """Ask user if they want to overwrite existing S3 file."""
    logger.info("\nFile already exists in S3:")
    logger.info(f"  Location: s3://{bucket}/{s3_key}")

    while True:
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response in ("y", "yes"):
            logger.debug("User chose to overwrite")
            return True
        if response in ("n", "no"):
            logger.debug("User chose not to overwrite")
            return False
        logger.info("  Please enter 'y' or 'n'")


# ============================================================================
# Main Download Logic
# ============================================================================


def download_file_with_cookies(file_url: str, cookies: list[dict], s3_client, bucket: str, out_dir: Path) -> bool:
    """
    Download one file using existing cookies, then upload it to S3.
    Returns True on success, False on failure.
    """
    filename = file_url.rstrip("/").split("/")[-1]
    s3_key = get_s3_key(filename)

    # Download using provided cookies (no browser needed)
    local_path = download_csv(file_url, cookies, out_dir)
    if not local_path:
        return False

    upload_success = upload_to_s3(s3_client, local_path, bucket, s3_key)

    if upload_success:
        logger.debug(f"Cleaning up local file: {local_path}")
        local_path.unlink()

    return upload_success


def download_all_csvs(
    cookies: list[dict], driver: webdriver.Chrome, out_dir: Path, folder_url: str, s3_client, bucket: str
) -> tuple[int, int, list[str]]:
    """
    Find every .csv on the folder page and download each one.
    Returns: (successful_count, failed_count, list_of_failed_urls)
    """
    urls = list_csv_urls_in_folder(driver, folder_url)
    ok, fail = 0, 0
    failed_urls = []

    for u in urls:
        if download_file_with_cookies(u, cookies, s3_client, bucket, out_dir):
            ok += 1
        else:
            fail += 1
            failed_urls.append(u)

        # Brief pause between files to be nice to the server
        if u != urls[-1]:  # Not the last file
            logger.debug("Waiting 5 seconds before next file...")
            time.sleep(5)

    logger.info(f"Summary: {ok} succeeded, {fail} failed.")
    return ok, fail, failed_urls


# ============================================================================
# Main Workflow
# ============================================================================


def main():
    """Main entry point: process each file with its own isolated browser session."""
    parser = argparse.ArgumentParser(description="Download Ameren CSV files and upload to S3")
    parser.add_argument(
        "--bucket-name", default=DEFAULT_S3_BUCKET, help=f"S3 bucket name (default: {DEFAULT_S3_BUCKET})"
    )
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts for existing files")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger.info("Enhanced Ameren CSV Downloader with S3 Integration")
    logger.info(f"Target S3 bucket: {args.bucket_name}\n")

    # Load credentials
    try:
        creds = load_credentials(CREDENTIALS_FILE)
        username, password = creds["username"], creds["password"]
        logger.debug(f"Loaded credentials from {CREDENTIALS_FILE}")
    except Exception:
        logger.exception(f"Failed to load credentials from {CREDENTIALS_FILE}")
        return 1

    # Initialize S3 client. If the program can't connect it exits.
    try:
        s3_client = boto3.client("s3")
        s3_client.list_buckets()
        logger.debug("Successfully connected to S3")
    except Exception:
        logger.exception("Failed to connect to S3")
        logger.info("Make sure AWS credentials are configured")
        return 1

    # Get list of CSV files with initial browser session
    logger.info("Scanning for CSV files...")
    driver = setup_driver()
    try:
        # Login for session side-effects; cookies not used directly here
        login_and_get_cookies(driver, username, password)
        csv_urls = list_csv_urls_in_folder(driver, FOLDER_URL)
    finally:
        driver.quit()
        logger.debug("Closed initial browser session")

    if not csv_urls:
        logger.error("No CSV files found")
        return 1

    logger.info(f"\nFound {len(csv_urls)} file(s) to process")

    # Filter files already in S3
    files_to_download = []
    for url in csv_urls:
        filename = url.rstrip("/").split("/")[-1]
        s3_key = get_s3_key(filename)

        if file_exists_in_s3(s3_client, args.bucket_name, s3_key):
            if not args.force:
                logger.info(f"Skipping {filename} - already in S3")
                continue
            else:
                logger.info(f"File {filename} exists but will re-download (--force)")

        files_to_download.append(url)

    logger.info(f"\nNeed to download: {len(files_to_download)} file(s)")

    if not files_to_download:
        logger.info("Nothing to download!")
        return 0

    # Create new session for downloads
    driver = setup_driver()
    try:
        logger.info("Creating session...")
        cookies = login_and_get_cookies(driver, username, password)

        # Download all files using the new function
        successful, failed, failed_urls = download_all_csvs(
            cookies, driver, DOWNLOAD_DIR, FOLDER_URL, s3_client, args.bucket_name
        )
    finally:
        driver.quit()
        logger.info("Closed session")

    # Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("Processing complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Total: {len(csv_urls)}")

    # Print failed URLs if any
    if failed_urls:
        logger.info("\nFailed URLs:")
        for url in failed_urls:
            filename = url.rstrip("/").split("/")[-1]
            logger.info(f"  - {filename}")
            logger.info(f"    {url}")

        logger.info(f"\nNote: {failed} file(s) failed to download.")
        logger.info("This is expected behavior due to Ameren's server limitations.")
        logger.info("\nSimply re-run this command to complete the remaining downloads:")
        logger.info("  python scripts/data_collection/ameren_downloader.py --force")
        logger.info("\nThe script will automatically skip files already in S3 and only")
        logger.info("download the failed files. You may need to run 2-3 times total.")
        return 1
    else:
        logger.info("\nAll files successfully downloaded and uploaded to S3!")
        return 0


if __name__ == "__main__":
    main()
