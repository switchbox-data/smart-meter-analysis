# scripts/04_verify_transfer.py
import boto3
import polars as pl
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()

def verify_transfer():
    """Compare manifest against S3 to find missing files"""
    
    # Load manifest
    manifest = pl.read_parquet("data/manifest.parquet")
    manifest_files = set(manifest['name'].to_list())
    
    # List S3 contents
    s3 = boto3.client('s3', region_name='us-west-2')
    paginator = s3.get_paginator('list_objects_v2')
    
    s3_files = set()
    for page in paginator.paginate(
        Bucket='smart-meter-data-in-us-west-2',
        Prefix='ComEd/'
    ):
        for obj in page.get('Contents', []):
            file_name = Path(obj['Key']).name
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
        missing_df = manifest.filter(pl.col('name').is_in(list(missing)))
        missing_df.write_parquet("data/missing_files.parquet")
        console.print(f"[yellow]Saved {len(missing)} missing files to data/missing_files.parquet[/yellow]")
        
        # Show first few missing
        console.print("\n[red]First 5 missing files:[/red]")
        for file in list(missing)[:5]:
            console.print(f"  - {file}")
    
    return len(missing) == 0

if __name__ == "__main__":
    all_good = verify_transfer()
    if all_good:
        console.print("[green bold]✓ All files successfully transferred![/green bold]")
    else:
        console.print("[yellow]⚠ Some files need retry - run 05_retry_failures.py[/yellow]")