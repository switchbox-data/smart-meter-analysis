import os  # Add os module import
import re
from datetime import datetime  # Add datetime import

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.exceptions import RequestException


def fetch_heating_data(state_abbrev):
    """
    Fetch home heating source data for a given state from EIA website.

    Parameters:
    state_abbrev (str): Two-letter state abbreviation

    Returns:
    list: dataframe of heating sources
    """
    url = f"https://www.eia.gov/state/print.php?sid={state_abbrev}"
    try:
        # Fetch the HTML content
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        html_content = response.text
    except RequestException as e:
        raise Exception(f"Failed to fetch data from URL: {e}") from e

    # Parse the HTML content
    soup = BeautifulSoup(html_content, "html.parser")

    # Find all tables
    tables = soup.find_all("table", class_="contable")

    # Look for the specific table with heating sources
    # Find the consumption & expenditures table
    consumption_table = None
    tables = soup.find_all("table", class_="contable")

    for table in tables:
        headers = table.find_all("th")
        for header in headers:
            if "Consumption & Expenditures" in header.text:
                consumption_table = table
                break
        if consumption_table:
            break

    if not consumption_table:
        raise ValueError(f"Consumption & Expenditures table not found for state {state_abbrev}")

    # Find the section with home heating data
    heating_section = None
    summary_rows = consumption_table.find_all("tr", class_="summary")

    for row in summary_rows:
        if "Energy Source Used for Home Heating" in row.text:
            heating_section = row
            break

    if not heating_section:
        raise ValueError(f"Home heating section not found for state {state_abbrev}")

    # Extract all rows for the home heating section
    data = []
    current_row = heating_section.find_next_sibling("tr")

    # Loop until we hit another summary row or end of table
    while current_row and current_row.get("class", "") != "summary":
        cells = current_row.find_all("td")

        if len(cells) >= 3:  # Ensuring we have enough cells
            energy_source = cells[0].text.strip()
            state_value = cells[1].text.strip()
            us_value = cells[2].text.strip()

            # Get period (if available)
            period = cells[4].text.strip() if len(cells) > 4 else ""

            # Add to data list
            data.append({
                "Energy Source": energy_source,
                f"{state_abbrev} (%)": state_value,
                "U.S. Average (%)": us_value,
                "Period": period,
            })

        # Move to next row
        current_row = current_row.find_next_sibling("tr")
        if not current_row:
            break

    return pd.DataFrame(data)


def clean_percentage_data(df, state_abbrev):
    """
    Clean the percentage data in the DataFrame:
    - Remove whitespace and % symbols
    - Convert percentages to decimal fractions

    Parameters:
    df (pandas.DataFrame): DataFrame with the original data
    state_abbrev (str): Two-letter state abbreviation

    Returns:
    pandas.DataFrame: Cleaned DataFrame
    """
    # Make a copy to avoid modifying the original DataFrame
    cleaned_df = df.copy()

    # Clean state percentage column
    state_col = f"{state_abbrev} (%)"
    cleaned_df[state_col] = cleaned_df[state_col].apply(lambda x: re.sub(r"[^\d.]", "", x) if isinstance(x, str) else x)

    # Clean US average percentage column
    cleaned_df["U.S. Average (%)"] = cleaned_df["U.S. Average (%)"].apply(
        lambda x: re.sub(r"[^\d.]", "", x) if isinstance(x, str) else x
    )

    # Convert to float and divide by 100 to get decimal fractions
    cleaned_df[state_col] = cleaned_df[state_col].astype(float) / 100
    cleaned_df["U.S. Average (%)"] = cleaned_df["U.S. Average (%)"].astype(float) / 100

    # Rename columns to reflect that they're now decimal fractions
    cleaned_df = cleaned_df.rename(columns={state_col: state_abbrev, "U.S. Average (%)": "U.S. Average"})

    return cleaned_df


def main(state_abbrev="MA"):
    try:
        df_raw = fetch_heating_data(state_abbrev)
        df_cleaned = clean_percentage_data(df_raw, state_abbrev)

        # Create output directory if it doesn't exist
        output_dir = "/workspaces/reports2/data/eia/state_energy_profiles"
        os.makedirs(output_dir, exist_ok=True)

        # Get current date in YYYYMMDD format
        current_date = datetime.now().strftime("%Y%m%d")

        # Save to CSV with date in filename
        output_file = f"{output_dir}/{state_abbrev.lower()}_heating_sources_{current_date}.csv"
        df_cleaned.to_csv(output_file, index=False)
        print(rf"\EIA state profile ({state_abbrev}) saved to '{output_file}'")

        return df_cleaned

    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    import sys

    # Get state abbreviation from command line or use 'MA' as default
    state_abbrev = sys.argv[1].upper() if len(sys.argv) > 1 else "MA"

    main(state_abbrev)
