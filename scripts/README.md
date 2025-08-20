# Scripts Execution Order

Run these scripts in order for the full data pipeline:

1. `extract_cookies.py` - Extract authentication cookies
2. `get_file_list.py` - Get list of files to download
3. `transfer_files.py` - Transfer files from SharePoint to S3
4. `verify_transfer.py` - Verify successful transfers
5. `retry_failures.py` - Retry any failed transfers
