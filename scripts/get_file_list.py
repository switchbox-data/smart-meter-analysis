# scripts/get_file_list.py
from http.cookiejar import MozillaCookieJar
from urllib.parse import quote

import polars as pl
import requests

BASE_URL = "https://exeloncorp.sharepoint.com"
TARGET_FOLDER = "/sites/CDWAnonymousUploads/Shared Documents/ComEd Anonymous Zip +4"


def load_cookies():
    """Load cookies for authentication"""
    cj = MozillaCookieJar("cookies.txt")
    cj.load(ignore_discard=True, ignore_expires=True)
    return cj


def get_files_via_api():
    """Primary method: Use SharePoint REST API"""
    cookies = load_cookies()

    headers = {"Accept": "application/json;odata=verbose", "User-Agent": "Mozilla/5.0"}

    files = []

    def recurse_folder(folder_path):
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

    recurse_folder(TARGET_FOLDER)

    # Save as Polars DataFrame
    df = pl.DataFrame(files)
    df.write_parquet("data/manifest.parquet")
    print(f"Found {len(files)} files")
    return df
