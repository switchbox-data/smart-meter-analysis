#!/usr/bin/env python
"""
Consolidated script for fetching ComEd smart meter data from SharePoint and
transferring to S3.
"""

import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from urllib.parse import quote

import boto3
import polars as pl
import requests
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

console = Console()

# Configuration
BASE_URL = "https://exeloncorp.sharepoint.com"
TARGET_FOLDER = "/sites/CDWAnonymousUploads/Shared Documents/ComEd Anonymous Zip +4"
BUCKET_NAME = "smart-meter-data-in-us-west-2"
AWS_REGION = "us-west-2"


def validate_cookies():
    """
    Check if we have the critical SharePoint auth cookies.

    Returns:
        tuple: (bool, str) - (is_valid, message)
    """
    cookie_file = pathlib.Path("cookies.txt")
    if not cookie_file.exists():
        return False, "cookies.txt not found"

    content = cookie_file.read_text()
    has_fedauth = "FedAuth" in content and "exeloncorp.sharepoint.com" in content
    has_rtfa = "rtFa" in content and ".sharepoint.com" in content

    if not has_fedauth:
        return False, "Missing FedAuth cookie - not authenticated"
    if not has_rtfa:
        return False, "Missing rtFa cookie - session invalid"

    return True, "Cookies valid"


def load_cookies():
    """
    Load cookies for authentication.

    Returns:
        MozillaCookieJar: Cookie jar with loaded cookies
    """
    cj = MozillaCookieJar("cookies.txt")
    cj.load(ignore_discard=True, ignore_expires=True)
    return cj


def get_file_list():
    """
    Get list of all files from SharePoint using REST API.

    Returns:
        pl.DataFrame: DataFrame with file information
    """
    cookies = load_cookies()
    headers = {"Accept": "application/json;odata=verbose", "User-Agent": "Mozilla/5.0"}

    files = []

    def recurse_folder(folder_path):
        """Recursively get all files in folder and subfolders."""
        # Get files in current folder
        folder_api = f"{BASE_URL}/_api/web/GetFolderByServerRelativeUrl('{quote(folder_path)}')"

        # Get files
        files_resp = requests.get(f"{folder_api}/Files", cookies=cookies, timeout=30, headers=headers)

        if files_resp.status_code == 200:
            data = files_resp.json()
            for file in data.get("d", {}).get("results", []):
                files.append({
                    "name": file["Name"],
                    "path": file["ServerRelativeUrl"],
                    "size_bytes": file["Length"],
                    "modified": file["TimeLastModified"],
                    "url": f"{BASE_URL}{file['ServerRelativeUrl']}",
                    "direct_url": f"{BASE_URL}{file['ServerRelativeUrl']}?download=1",
                })

        # Get subfolders and recurse
        folders_resp = requests.get(f"{folder_api}/Folders", cookies=cookies, timeout=30, headers=headers)

        if folders_resp.status_code == 200:
            data = folders_resp.json()
            for folder in data.get("d", {}).get("results", []):
                if not folder["Name"].startswith("_"):  # Skip hidden folders
                    recurse_folder(folder["ServerRelativeUrl"])

    # Start recursion from target folder
    recurse_folder(TARGET_FOLDER)

    # Save as Polars DataFrame
    df = pl.DataFrame(files)
    df.write_parquet("data/manifest.parquet")
    console.print(f"[green]Found {len(files)} files[/green]")
    return df


class SharePointToS3Transfer:
    """Handler for transferring files from SharePoint to S3."""

    def __init__(self, max_workers=5):
        self.bucket_name = BUCKET_NAME
        self.max_workers = max_workers
        self.s3_client = boto3.client("s3", region_name=AWS_REGION)
        self.cookies = load_cookies()
        self.session = self.create_session()

        # Load or create tracking
        self.manifest = pl.read_parquet("data/manifest.parquet")
        self.completed = self.load_completed()
        self.errors = []

    def create_session(self):
        """Create reusable session with cookies."""
        session = requests.Session()
        session.cookies = self.cookies
        session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
        return session

    def load_completed(self):
        """Load list of already completed transfers."""
        completed_file = Path("data/transfer_log.parquet")
        if completed_file.exists():
            return set(pl.read_parquet(completed_file)["s3_key"].to_list())
        return set()

    def fix_sharepoint_url(self, url):
        """Convert viewer URLs to direct download URLs."""
        if "/:u:/r/" in url or "/:x:/r/" in url or "/:f:/r/" in url:
            # Extract path after the sharing prefix
            parts = url.split("/r/", 1)[1] if "/r/" in url else url
            path = parts.split("?")[0]
            return f"https://exeloncorp.sharepoint.com/{path}?download=1"

        # Ensure download parameter
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}download=1"

    def calculate_s3_key(self, file_path):
        """Generate S3 key from SharePoint path."""
        # Extract relative path after "ComEd Anonymous Zip +4"
        if "ComEd Anonymous Zip +4" in file_path:
            relative = file_path.split("ComEd Anonymous Zip +4/")[1]
        else:
            relative = Path(file_path).name

        return f"ComEd/{relative}"

    def transfer_file(self, row):
        """Transfer single file from SharePoint to S3."""
        file_name = row["name"]
        source_url = row.get("direct_url", row["url"])
        s3_key = self.calculate_s3_key(row.get("path", file_name))

        # Skip if already completed
        if s3_key in self.completed:
            return {"status": "skipped", "file": file_name, "s3_key": s3_key}

        try:
            # Fix URL
            download_url = self.fix_sharepoint_url(source_url)

            # Download with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(download_url, stream=True, timeout=30, allow_redirects=True)

                    # Check if we got HTML instead of file
                    content_type = response.headers.get("Content-Type", "")
                    if "text/html" in content_type.lower():
                        raise ValueError("Invalid response")  # noqa: TRY003

                    response.raise_for_status()
                    break

                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2**attempt)  # Exponential backoff

            # Stream directly to S3
            self.s3_client.upload_fileobj(
                response.raw,
                self.bucket_name,
                s3_key,
                ExtraArgs={"Metadata": {"source_url": source_url, "download_time": datetime.now().isoformat()}},
            )

            # Record success
            self.completed.add(s3_key)
            return {
                "status": "success",
                "file": file_name,
                "s3_key": s3_key,
                "size": response.headers.get("Content-Length", 0),
            }

        except Exception as e:
            # Record error for retry
            error_info = {
                "file": file_name,
                "s3_key": s3_key,
                "url": source_url,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }
            self.errors.append(error_info)
            return {"status": "error", "file": file_name, "error": str(e)}

    def run_parallel_transfer(self):
        """Main transfer orchestration."""
        # Filter out completed files
        to_transfer = self.manifest.filter(~pl.col("name").is_in([Path(k).name for k in self.completed]))

        total_files = len(to_transfer)
        console.print(f"[green]Files to transfer: {total_files}[/green]")

        if total_files == 0:
            console.print("[yellow]All files already transferred![/yellow]")
            return

        results = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Transferring files...", total=total_files)

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all files
                futures = {executor.submit(self.transfer_file, row): row for row in to_transfer.iter_rows(named=True)}

                # Process as completed
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)

                    # Update progress
                    progress.update(task, advance=1)

                    # Show status
                    if result["status"] == "success":
                        console.print(f"✓ {result['file']}", style="green")
                    elif result["status"] == "error":
                        console.print(f"✗ {result['file']}: {result['error']}", style="red")

        # Save results
        self.save_results(results)

    def save_results(self, results):
        """Save transfer log and errors."""
        # Save successful transfers
        successful = [r for r in results if r["status"] == "success"]
        if successful:
            df = pl.DataFrame(successful)
            df = df.with_columns(pl.lit(datetime.now()).alias("transfer_time"))

            log_file = Path("data/transfer_log.parquet")
            if log_file.exists():
                existing = pl.read_parquet(log_file)
                df = pl.concat([existing, df])

            df.write_parquet(log_file)

        # Save errors for retry
        if self.errors:
            with open("data/errors.json", "w") as f:
                json.dump(self.errors, f, indent=2)
            console.print(f"[yellow]Saved {len(self.errors)} errors to data/errors.json[/yellow]")


def transfer_files(max_workers=5):
    """
    Transfer all files from SharePoint to S3.

    Args:
        max_workers: Number of parallel transfers
    """
    transfer = SharePointToS3Transfer(max_workers=max_workers)
    transfer.run_parallel_transfer()


def retry_failures():
    """Retry failed transfers from error log."""
    # Check for errors
    error_file = Path("data/errors.json")
    if not error_file.exists():
        console.print("[green]No errors to retry[/green]")
        return

    with open(error_file) as f:
        errors = json.load(f)

    console.print(f"[yellow]Retrying {len(errors)} failed transfers...[/yellow]")

    # Convert to DataFrame format expected by transfer
    retry_df = pl.DataFrame([
        {"name": e["file"], "url": e["url"], "direct_url": e["url"], "path": e.get("s3_key", e["file"])} for e in errors
    ])

    # Run transfer with lower parallelism for retries
    transfer = SharePointToS3Transfer(max_workers=2)
    transfer.manifest = retry_df
    transfer.run_parallel_transfer()

    # Clean up error file if all successful
    if not transfer.errors:
        error_file.unlink()
        console.print("[green bold]All retries successful![/green bold]")


def verify_transfer():
    """
    Compare manifest against S3 to find missing files.

    Returns:
        bool: True if all files transferred successfully
    """
    # Load manifest
    manifest = pl.read_parquet("data/manifest.parquet")
    manifest_files = set(manifest["name"].to_list())

    # List S3 contents
    s3 = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")

    s3_files = set()
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="ComEd/"):
        for obj in page.get("Contents", []):
            file_name = Path(obj["Key"]).name
            s3_files.add(file_name)

    # Compare
    missing = manifest_files - s3_files
    extra = s3_files - manifest_files

    # Display results
    table = Table(title="Transfer Verification")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", style="magenta")

    table.add_row("Files in Manifest", str(len(manifest_files)))
    table.add_row("Files in S3", str(len(s3_files)))
    table.add_row("Missing from S3", str(len(missing)))
    table.add_row("Extra in S3", str(len(extra)))

    console.print(table)

    # Save missing files for retry
    if missing:
        missing_df = manifest.filter(pl.col("name").is_in(list(missing)))
        missing_df.write_parquet("data/missing_files.parquet")
        console.print(f"[yellow]Saved {len(missing)} missing files to data/missing_files.parquet[/yellow]")

        # Show first few missing
        console.print("\n[red]First 5 missing files:[/red]")
        for file in list(missing)[:5]:
            console.print(f"  - {file}")

    return len(missing) == 0


def main():
    """
    Main orchestration function that runs the complete workflow.

    Workflow:
    1. Validate cookies exist and are valid
    2. Get list of all files from SharePoint
    3. Transfer files to S3
    4. Verify all files transferred
    5. Retry any failures
    """
    console.print("[bold cyan]Starting ComEd Data Fetch Process[/bold cyan]\n")

    # Step 1: Validate cookies
    console.print("[yellow]Step 1: Validating cookies...[/yellow]")
    valid, msg = validate_cookies()
    if not valid:
        console.print(f"[red]✗ {msg}[/red]")
        console.print("[red]Please export cookies from your browser first![/red]")
        return
    console.print(f"[green]✓ {msg}[/green]\n")

    # Step 2: Get file list
    console.print("[yellow]Step 2: Getting file list from SharePoint...[/yellow]")
    try:
        df = get_file_list()
        console.print(f"[green]✓ Successfully retrieved {len(df)} files[/green]\n")
    except Exception as e:
        console.print(f"[red]✗ Failed to get file list: {e}[/red]")
        return

    # Step 3: Transfer files
    console.print("[yellow]Step 3: Transferring files to S3...[/yellow]")
    transfer_files(max_workers=5)
    console.print()

    # Step 4: Verify transfer
    console.print("[yellow]Step 4: Verifying transfer...[/yellow]")
    all_good = verify_transfer()
    console.print()

    # Step 5: Retry failures if needed
    if not all_good:
        console.print("[yellow]Step 5: Retrying failed transfers...[/yellow]")
        retry_failures()

        # Verify again
        console.print("\n[yellow]Final verification...[/yellow]")
        all_good = verify_transfer()

    # Final status
    console.print("\n" + "=" * 50)
    if all_good:
        console.print("[green bold]✓ PROCESS COMPLETE: All files successfully transferred![/green bold]")
    else:
        console.print("[yellow]⚠ Process complete with some files missing.[/yellow]")
        console.print("[yellow]Run 'python scripts/fetch_comed_data.py retry' to retry failures[/yellow]")


if __name__ == "__main__":
    import sys

    # Check for specific command
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "validate":
            valid, msg = validate_cookies()
            console.print(f"[green]✓ {msg}[/green]" if valid else f"[red]✗ {msg}[/red]")

        elif command == "list":
            get_file_list()

        elif command == "transfer":
            transfer_files()

        elif command == "verify":
            all_good = verify_transfer()
            if all_good:
                console.print("[green bold]✓ All files successfully transferred![/green bold]")

        elif command == "retry":
            retry_failures()

        else:
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("Available commands: validate, list, transfer, verify, retry")
            console.print("Or run without arguments for full workflow")

    else:
        # Run full workflow
        main()
