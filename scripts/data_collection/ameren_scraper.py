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

Known Limitations:
- Large file downloads (5-7 GB) occasionally fail due to network/server issues
- Page scraping intermittently returns 0 files; simply re-run if this occurs
- Ameren's server may rate-limit or invalidate sessions unpredictably

These limitations are handled through the idempotent design - failures are 
expected and resolved by re-running the script.
"""

import argparse
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
    return webdriver.Chrome(service=service, options=options)


def login_and_get_cookies(
    driver: webdriver.Chrome, username: str, password: str
    ) -> list[dict]:
    """Login to Ameren WebFTP and return session cookies."""
    print("Logging in to Ameren WebFTP...")
    driver.get("https://webftp.ameren.com/login")

    # Fill login form
    username_field = WebDriverWait(
        driver, 15
        ).until(
            EC.presence_of_element_located(
                (By.NAME, "username")
                ))
    password_field = driver.find_element(By.NAME, "password")
    username_field.send_keys(username)
    password_field.send_keys(password)

    # Submit and wait for redirect
    submit_button = driver.find_element(
        By.CSS_SELECTOR, 
        'button[type="submit"]'
        )
    driver.execute_script(
        "arguments[0].click();", 
        submit_button
        )
    WebDriverWait(driver, 15).until(
        lambda d: d.current_url != "https://webftp.ameren.com/login"
        )

    print("Login successful")
    return driver.get_cookies()


def list_csv_urls_in_folder(
    driver: webdriver.Chrome, 
    folder_url: str
    ) -> list[str]:
    """Scrape folder page and return list of CSV file URLs."""
    print(f"Scanning folder: {folder_url}")
    driver.get(folder_url)

    # Wait for page content to load - look for table or file list elements
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located(
                (By.TAG_NAME, "table")
                ))
    except TimeoutException:
        print("Warning: Table element not found, proceeding anyway...")

    # Additional wait for JavaScript to finish rendering
    time.sleep(5)

    # Scroll to trigger lazy-loading until page height stabilizes
    last_height = driver.execute_script(
        "return document.body.scrollHeight"
        )
    scroll_attempts = 0
    max_scroll_attempts = 10

    while scroll_attempts < max_scroll_attempts:
        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
            )
        time.sleep(1.5)
        new_height = driver.execute_script(
            "return document.body.scrollHeight"
            )
        if new_height == last_height:
            break
        last_height = new_height
        scroll_attempts += 1

    # Collect all CSV links
    csv_urls = []
    all_anchors = driver.find_elements(By.TAG_NAME, "a")

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

    print(f"Found {len(csv_urls)} CSV file(s)")

    # Debug info if no files found
    if not csv_urls:
        print(f"Debug: Total anchor elements found: {len(all_anchors)}")
        print("Debug: Sample of first 5 links:")
        for i, anchor in enumerate(all_anchors[:5]):
            print(f"  Link {i}: text='{anchor.text}' href='{anchor.get_attribute('href')}'")

    return csv_urls


# ============================================================================
# File Download
# ============================================================================


def make_authenticated_session(
    cookies: list[dict], 
    referer: str
    ) -> requests.Session:
    """Create requests session with Selenium cookies and headers."""
    session = requests.Session()
    for cookie in cookies:
        session.cookies.set(cookie["name"], cookie["value"])
    session.headers.update(
        {"User-Agent": BROWSER_USER_AGENT, 
        "Referer": referer, 
        "Accept": "*/*"}
        )
    return session


def is_valid_csv(filepath: Path) -> bool:
    """Check if downloaded file is actually a CSV and not an HTML error page."""
    if filepath.stat().st_size == 0:
        return False

    # Check first 1KB for HTML markers
    with filepath.open("rb") as f:
        first_bytes = f.read(1024)
        if b"<!DOCTYPE html>" in first_bytes or b"<html" in first_bytes:
            return False

    return True


def download_csv(
    file_url: str, 
    cookies: list[dict], 
    out_dir: Path, 
    max_retries: int = 3
    ) -> Path | None:
    """
    Download CSV file to local directory with retry logic.
    Returns filepath on success, None on failure.
    """
    out_dir.mkdir(
        parents=True, 
        exist_ok=True
        )
    filename = file_url.rstrip("/").split("/")[-1]
    filepath = out_dir / filename

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            wait_time = 2**attempt  # Exponential backoff: 4, 8, 16 seconds
            print(f"Retry attempt {attempt}/{max_retries} after {wait_time}s delay...")
            time.sleep(wait_time)
        else:
            print(f"Downloading: {filename}")

        session = make_authenticated_session(
            cookies, 
            referer=file_url.rsplit("/", 1)[0] + "/"
            )

        try:
            with session.get(file_url, stream=True, timeout=120) as response:
                response.raise_for_status()

                # Get file size for progress tracking
                total_bytes = None
                content_length = response.headers.get("content-length")
                if content_length:
                    total_bytes = int(content_length)
                    size_gb = total_bytes / (1024**3)
                    print(f"  Size: {total_bytes:,} bytes (~{size_gb:.2f} GB)")

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
                                print(f"  Progress: {percent:5.1f}%")

            # Verify file is valid CSV
            if not is_valid_csv(filepath):
                print("ERROR: Downloaded HTML error page instead of CSV (session expired)")
                filepath.unlink(missing_ok=True)
                continue  # Retry

        except Exception as e:
            print(f"Download failed: {e}")
            filepath.unlink(missing_ok=True)
            if attempt == max_retries:
                print(f"All {max_retries} retry attempts exhausted")
                return None
            # Continue to next retry attempt
        else:
            size = filepath.stat().st_size
            print(f"Download complete: {size:,} bytes")
            return filepath

    return None


# ============================================================================
# S3 Operations
# ============================================================================


def extract_date_from_filename(
    filename: str
    ) -> str | None:
    """Extract YYYYMM date from filename, returns None if not found."""
    base_name = filename.replace(".csv", "")
    matches = re.findall(r"\d{6}", base_name)

    for match in matches:
        year = int(match[:4])
        month = int(match[4:6])
        if 2000 <= year <= 2099 and 1 <= month <= 12:
            return match

    return None


def get_s3_key(
    filename: str
    ) -> str:
    """Generate S3 key with folder structure: ameren-data/YYYYMM/filename.csv"""
    date_str = extract_date_from_filename(filename)
    folder = date_str if date_str else "undated"
    return f"ameren-data/{folder}/{filename}"


def file_exists_in_s3(
    s3_client, 
    bucket: str, 
    key: str
    ) -> bool:
    """Check if file exists in S3 bucket."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
        raise
    else:
        return True


def upload_to_s3(
    s3_client, 
    local_path: Path, 
    bucket: str, 
    key: str
    ) -> bool:
    """Upload file to S3. Returns True on success."""
    try:
        file_size = local_path.stat().st_size
        size_gb = file_size / (1024**3)
        print(f"Uploading to S3: s3://{bucket}/{key}")
        print(f"  Size: {file_size:,} bytes (~{size_gb:.2f} GB)")

        s3_client.upload_file(str(local_path), bucket, key)
        print("Upload complete")
    except Exception as e:
        print(f"Upload failed: {e}")
        return False
    else:
        return True


# ============================================================================
# User Interaction
# ============================================================================


def ask_overwrite_permission(
    filename: str, 
    s3_key: str, 
    bucket: str
    ) -> bool:
    """Ask user if they want to overwrite existing S3 file."""
    print("\nFile already exists in S3:")
    print(f"  Location: s3://{bucket}/{s3_key}")

    while True:
        response = input("  Overwrite? (y/n): ").strip().lower()
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'")


# ============================================================================
# Main Workflow
# ============================================================================


def process_single_file_with_fresh_session(
    file_url: str, 
    username: str, 
    password: str, 
    s3_client, 
    bucket: str, 
    force: bool
) -> bool:
    """
    Process a single file with its own browser session.
    Creates fresh login, downloads one file, then closes browser.
    """
    filename = file_url.rstrip("/").split("/")[-1]
    s3_key = get_s3_key(filename)

    print(f"\n{'=' * 80}")
    print(f"Processing: {filename}")
    print(f"{'=' * 80}")

    # Check if file already exists in S3
    exists = file_exists_in_s3(s3_client, bucket, s3_key)
    if exists and not force and not ask_overwrite_permission(
        filename, 
        s3_key, 
        bucket
        ):
        print(f"Skipping: {filename}")
        return True
    if exists:
        print("  Will overwrite existing file")

    # Create fresh browser session for this file
    driver = setup_driver()
    try:
        print("Creating fresh session...")
        cookies = login_and_get_cookies(driver, username, password)

        # Download to local temp directory
        local_path = download_csv(file_url, cookies, DOWNLOAD_DIR)
        if not local_path:
            return False

        # Upload to S3
        upload_success = upload_to_s3(s3_client, local_path, bucket, s3_key)

        # Clean up local file after successful upload
        if upload_success:
            try:
                local_path.unlink()
                print("Deleted local file")
            except Exception as e:
                print(f"Warning: Could not delete local file: {e}")

        return upload_success

    finally:
        driver.quit()
        print("Closed browser session")


def main():
    """Main entry point: process each file with its own isolated browser session."""
    parser = argparse.ArgumentParser(description="Download Ameren CSV files and upload to S3")
    parser.add_argument(
        "--bucket-name", default=DEFAULT_S3_BUCKET, help=f"S3 bucket name (default: {DEFAULT_S3_BUCKET})"
    )
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompts for existing files")
    args = parser.parse_args()

    print("Enhanced Ameren CSV Downloader with S3 Integration")
    print(f"Target S3 bucket: {args.bucket_name}\n")

    # Load credentials
    try:
        creds = load_credentials(CREDENTIALS_FILE)
        username, password = creds["username"], creds["password"]
    except Exception as e:
        print(f"ERROR: Failed to load credentials from {CREDENTIALS_FILE}: {e}")
        return

    # Initialize S3 client
    try:
        s3_client = boto3.client("s3")
        s3_client.list_buckets()
    except Exception as e:
        print(f"ERROR: Failed to connect to S3: {e}")
        print("Make sure AWS credentials are configured")
        return

    # Get list of CSV files with initial browser session
    print("Scanning for CSV files...")
    driver = setup_driver()
    try:
        # Login for session side-effects; cookies not used directly here
        login_and_get_cookies(driver, username, password)
        csv_urls = list_csv_urls_in_folder(driver, FOLDER_URL)
    finally:
        driver.quit()

    if not csv_urls:
        print("ERROR: No CSV files found")
        return

    print(f"\nFound {len(csv_urls)} file(s) to process")
    print("Each file will be downloaded in its own isolated session\n")

    # Process each file with its own fresh browser session
    successful = 0
    failed = 0

    for i, file_url in enumerate(csv_urls, 1):
        print(f"\nFile {i}/{len(csv_urls)}")

        if process_single_file_with_fresh_session(
            file_url, username, password, s3_client, args.bucket_name, args.force
        ):
            successful += 1
        else:
            failed += 1

        # Brief pause between files to be nice to the server
        if i < len(csv_urls):
            print("Waiting 5 seconds before next file...")
            time.sleep(5)

    # Summary
    print(f"\n{'=' * 80}")
    print("Processing complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(csv_urls)}")

    if failed > 0:
        print(f"\nNote: {failed} file(s) failed to download.")
        print("This is expected behavior due to Ameren's server limitations.")
        print("\nSimply re-run this command to complete the remaining downloads:")
        print("  python scripts/data_collection/ameren_downloader.py --force")
        print("\nThe script will automatically skip files already in S3 and only")
        print("download the failed files. You may need to run 2-3 times total.")
    else:
        print("\nAll files successfully downloaded and uploaded to S3!")

    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()