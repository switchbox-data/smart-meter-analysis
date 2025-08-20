# scripts/03_transfer_files.py
import boto3
import requests
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import hashlib
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from http.cookiejar import MozillaCookieJar
import time

console = Console()

class SharePointToS3Transfer:
    def __init__(self, bucket_name="smart-meter-data-in-us-west-2", max_workers=5):
        self.bucket_name = bucket_name
        self.max_workers = max_workers
        self.s3_client = boto3.client('s3', region_name='us-west-2')
        self.cookies = self.load_cookies()
        self.session = self.create_session()
        
        # Load or create tracking
        self.manifest = pl.read_parquet("data/manifest.parquet")
        self.completed = self.load_completed()
        self.errors = []
        
    def load_cookies(self):
        """Load browser cookies for auth"""
        cj = MozillaCookieJar("cookies.txt")
        cj.load(ignore_discard=True, ignore_expires=True)
        return cj
    
    def create_session(self):
        """Create reusable session with cookies"""
        session = requests.Session()
        session.cookies = self.cookies
        session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*"
        })
        return session
    
    def load_completed(self):
        """Load list of already completed transfers"""
        completed_file = Path("data/transfer_log.parquet")
        if completed_file.exists():
            return set(pl.read_parquet(completed_file)["s3_key"].to_list())
        return set()
    
    def fix_sharepoint_url(self, url):
        """Convert viewer URLs to direct download URLs"""
        if "/:u:/r/" in url or "/:x:/r/" in url or "/:f:/r/" in url:
            # Extract path after the sharing prefix
            parts = url.split("/r/", 1)[1] if "/r/" in url else url
            path = parts.split("?")[0]
            return f"https://exeloncorp.sharepoint.com/{path}?download=1"
        
        # Ensure download parameter
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}download=1"
    
    def calculate_s3_key(self, file_path):
        """Generate S3 key from SharePoint path"""
        # Extract relative path after "ComEd Anonymous Zip +4"
        if "ComEd Anonymous Zip +4" in file_path:
            relative = file_path.split("ComEd Anonymous Zip +4/")[1]
        else:
            relative = Path(file_path).name
        
        return f"ComEd/{relative}"
    
    def transfer_file(self, row):
        """Transfer single file from SharePoint to S3"""
        file_name = row['name']
        source_url = row.get('direct_url', row['url'])
        s3_key = self.calculate_s3_key(row.get('path', file_name))
        
        # Skip if already completed
        if s3_key in self.completed:
            return {'status': 'skipped', 'file': file_name, 's3_key': s3_key}
        
        try:
            # Fix URL
            download_url = self.fix_sharepoint_url(source_url)
            
            # Download with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.session.get(
                        download_url,
                        stream=True,
                        timeout=30,
                        allow_redirects=True
                    )
                    
                    # Check if we got HTML instead of file
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type.lower():
                        raise ValueError(f"Got HTML instead of file - cookies may be expired")
                    
                    response.raise_for_status()
                    break
                    
                except requests.exceptions.Timeout:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
            
            # Stream directly to S3
            self.s3_client.upload_fileobj(
                response.raw,
                self.bucket_name,
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'source_url': source_url,
                        'download_time': datetime.now().isoformat()
                    }
                }
            )
            
            # Record success
            self.completed.add(s3_key)
            return {
                'status': 'success',
                'file': file_name,
                's3_key': s3_key,
                'size': response.headers.get('Content-Length', 0)
            }
            
        except Exception as e:
            # Record error for retry
            error_info = {
                'file': file_name,
                's3_key': s3_key,
                'url': source_url,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.errors.append(error_info)
            return {'status': 'error', 'file': file_name, 'error': str(e)}
    
    def run_parallel_transfer(self):
        """Main transfer orchestration"""
        # Filter out completed files
        to_transfer = self.manifest.filter(
            ~pl.col('name').is_in(
                [Path(k).name for k in self.completed]
            )
        )
        
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
            console=console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Transferring files...", 
                total=total_files
            )
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all files
                futures = {
                    executor.submit(self.transfer_file, row): row
                    for row in to_transfer.iter_rows(named=True)
                }
                
                # Process as completed
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    
                    # Update progress
                    progress.update(task, advance=1)
                    
                    # Show status
                    if result['status'] == 'success':
                        console.print(f"✓ {result['file']}", style="green")
                    elif result['status'] == 'error':
                        console.print(f"✗ {result['file']}: {result['error']}", style="red")
        
        # Save results
        self.save_results(results)
    
    def save_results(self, results):
        """Save transfer log and errors"""
        # Save successful transfers
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            df = pl.DataFrame(successful)
            df = df.with_columns(
                pl.lit(datetime.now()).alias('transfer_time')
            )
            
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

if __name__ == "__main__":
    transfer = SharePointToS3Transfer(max_workers=5)
    transfer.run_parallel_transfer()