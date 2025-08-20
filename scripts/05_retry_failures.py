# scripts/05_retry_failures.py
import json
from pathlib import Path
from scripts.03_transfer_files import SharePointToS3Transfer
import polars as pl

def retry_failures():
    """Retry failed transfers from error log"""
    
    # Check for errors
    error_file = Path("data/errors.json")
    if not error_file.exists():
        print("No errors to retry")
        return
    
    with open(error_file) as f:
        errors = json.load(f)
    
    print(f"Retrying {len(errors)} failed transfers...")
    
    # Convert to DataFrame format expected by transfer
    retry_df = pl.DataFrame([
        {
            'name': e['file'],
            'url': e['url'],
            'direct_url': e['url'],
            'path': e.get('s3_key', e['file'])
        }
        for e in errors
    ])
    
    # Run transfer
    transfer = SharePointToS3Transfer(max_workers=2)  # Lower parallelism for retries
    transfer.manifest = retry_df
    transfer.run_parallel_transfer()
    
    # Clean up error file if all successful
    if not transfer.errors:
        error_file.unlink()
        print("✓ All retries successful!")

if __name__ == "__main__":
    retry_failures()