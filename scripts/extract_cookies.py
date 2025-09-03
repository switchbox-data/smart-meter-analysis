# scripts/extract_cookies.py
import pathlib


def validate_cookies():
    """Check if we have the critical SharePoint auth cookies"""
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


if __name__ == "__main__":
    valid, msg = validate_cookies()
    print(f"✓ {msg}" if valid else f"✗ {msg}")
