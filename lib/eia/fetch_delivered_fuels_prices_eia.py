import os

import pandas as pd
import requests
from requests.exceptions import RequestException


def get_eia_petroleum_data(
    api_key: str,
    state_abbrev: str,
    duoarea: str,
    fuel_type: str = "heating_oil",
    retail_or_wholesale: str | None = "retail",
    frequency: str = "weekly",
    path_csv: str = "/workspaces/reports2/data/eia/delivered_fuels/xx_eia_delivered_fuels_prices_weekly.csv",
    path_parquet: str = "/workspaces/reports2/data/eia/delivered_fuels/xx_eia_delivered_fuels_prices_weekly.parquet",
    sort_column: str = "period",
    sort_direction: str = "desc",
    offset: int = 0,
    length: int = 5000,
) -> str:
    """
    Fetch petroleum pricing data from EIA API and save to CSV.

    Args:
        api_key: Your EIA API key
        state_abbrev: State abbreviation (e.g., 'RI', 'PADD_1A')
        duoarea: EIA duoarea code (e.g., 'R1X' or 'SRI')
        fuel_type: Fuel type (default: "heating_oil", or 'propane')
        retail_or_wholesale: 'retail' or 'wholesale' (default: 'retail')
        frequency: Data frequency (default: "weekly")
        path_csv: Path where to save the CSV file (default: "/workspaces/reports2/data/eia/delivered_fuels/xx_eia_heating_oil_prices_weekly.csv")
        path_parquet: Path where to save the Parquet file (default: "/workspaces/reports2/data/eia/delivered_fuels/xx_eia_heating_oil_prices_weekly.parquet")
        sort_column: Column to sort by (default: "period")
        sort_direction: Sort direction "asc" or "desc" (default: "desc")
        offset: Starting offset for pagination (default: 0)
        length: Number of records to return (default: 5000)

    Returns:
        Path to the saved Parquet file

    Raises:
        requests.exceptions.RequestException: If the API request fails
        OSError: If there's an issue creating directories or saving the file
    """

    # Base URL
    base_url = "https://api.eia.gov/v2/petroleum/pri/wfr/data/"

    # Build parameters
    params = {
        "frequency": frequency,
        "data[0]": "value",
        "sort[0][column]": sort_column,
        "sort[0][direction]": sort_direction,
        "offset": offset,
        "length": length,
        "api_key": api_key,
    }

    # Add duoarea
    params["facets[duoarea][0]"] = duoarea

    # Add processes
    # "Process" is a EIA "process" for collecting and aggregating data
    # 1. "PRS" (Price Delivered to Residential Consumers)
    # 2. "PWR" (Price Delivered to Wholesale Consumers)
    if retail_or_wholesale == "wholesale" and fuel_type == "propane":
        params["facets[process][0]"] = "PWR"
    else:
        params["facets[process][0]"] = "PRS"

    # Add series
    # "Series" is an EIA time series
    # Composed of...
    # 1. "W" (Weekly)
    # 2. "EPD2F" (No.2 Heating Oild)
    # 3. "PRS" (Price Delivered to Residential Consumers)
    # 4. "duaoarea" (duoarea: state (e.g. 'RI') or region (eg 'R1X' for PADD_1A))
    # 5. "DPG" (Dollars per Gallon)
    fuel_code_eia = "EPD2F" if fuel_type == "heating_oil" else "EPLLPA"

    params["facets[series][0]"] = f"W_{fuel_code_eia}_{params['facets[process][0]']}_{duoarea}_DPG"

    try:
        # Make API request
        response = requests.get(
            base_url, params=params, headers={"Accept": "application/json", "User-Agent": "reports"}, timeout=30
        )
        response.raise_for_status()
        data = response.json()

        # Extract data records
        records = data.get("response", {}).get("data", [])

        if not records:
            print("No data returned from API")
            return data

        # Convert to DataFrame
        df = pd.DataFrame(records)

        # Create directory if it doesn't exist
        path_csv = path_csv.replace("xx", state_abbrev.lower()).replace("delivered_fuels", fuel_type)
        path_parquet = path_parquet.replace("xx", state_abbrev.lower()).replace("delivered_fuels", fuel_type)
        os.makedirs(os.path.dirname(path_csv), exist_ok=True)
        os.makedirs(os.path.dirname(path_parquet), exist_ok=True)

        # Save to CSV
        df.to_csv(path_csv, index=False)
        print(f"Data saved to {path_csv}")

        # Save to Parquet
        df.to_parquet(path_parquet, index=False)
        print(f"Data saved to {path_parquet}")

        return path_parquet

    except RequestException as e:
        print(f"Error fetching data from EIA API: {e}")
        raise
    except Exception as e:
        print(f"Error processing or saving data: {e}")
        raise


def clean_eia_petroleum_data(path_parquet, state_abbrev, fuel_type):
    """
    Clean the EIA petroleum data.
    """
    heating_oil_df = pd.read_parquet(path_parquet)

    heating_oil_df["year"] = heating_oil_df["period"].str.split("-").str[0].astype(int)
    heating_oil_df["month"] = heating_oil_df["period"].str.split("-").str[1].astype(int)
    heating_oil_df["day"] = heating_oil_df["period"].str.split("-").str[2].astype(int)

    # Fuel Oil: 145.945 MJ per gallon (https://www.eia.gov/energyexplained/units-and-calculators/energy-conversion-calculators.php)
    # 40.2778 kWh/gallon
    if (heating_oil_df["units"] == "$/GAL").all():
        if fuel_type == "heating_oil":
            heating_oil_df["value"] = heating_oil_df["value"].astype(float) / 40.2778
            heating_oil_df["units"] = "dollars_per_kwh"
        else:  # fuel_type == 'propane':
            # propane = 26.8kWh per gallon (1 kWh = 3.41214163312794 BTU, 1 gallon of propane = 91,452 BTU)
            # # https://www.eia.gov/energyexplained/units-and-calculators/british-thermal-units.php
            heating_oil_df["value"] = heating_oil_df["value"].astype(float) / 26.8
            heating_oil_df["units"] = "dollars_per_kwh"
    else:
        print(f"Expected '$/GAL' in 'units' column, but got something else. Please check the data at {path_parquet}.")

    # Group by year and month and calculate mean prices
    heating_oil_df = (
        heating_oil_df.groupby(["year", "month"])
        .agg({
            "value": "mean",
            "units": "first",
            "duoarea": "first",
            "area-name": "first",
            "product": "first",
            "process": "first",
            "process-name": "first",
            "series": "first",
            "series-description": "first",
        })
        .reset_index()
    )

    # Rename 'value' to 'supply_rate' to match expected column naming convention
    heating_oil_df = heating_oil_df.rename(columns={"value": "supply_rate"})

    # add a "fuel_oil_utility" column
    if fuel_type == "heating_oil":
        heating_oil_df["fuel_oil_utility"] = "generic_retail"
    else:  # fuel_type == 'propane':
        heating_oil_df["propane_utility"] = "generic_retail"

    # add a "state" column
    heating_oil_df["state"] = state_abbrev

    # To-Do: Linearly interpolate the data for missing values

    # Save
    # ----
    # parquet
    heating_oil_df.to_parquet(
        path_parquet.replace("xx", state_abbrev.lower()).replace("weekly", "monthly"), index=False
    )
    print(f"Saved {path_parquet}")

    # csv
    path_csv = path_parquet.replace("xx", state_abbrev.lower()).replace("weekly", "monthly").replace(".parquet", ".csv")
    heating_oil_df.to_csv(path_csv, index=False)
    print(f"Saved {path_csv}")

    return heating_oil_df


# def plot_eia_petroleum_data(path_parquet, state_abbrev):
#     """
#     Plot the EIA petroleum data.
#     """
#     path_parquet = path_parquet.replace('xx', state_abbrev.lower())
#     heating_oil_df = pd.read_parquet(path_parquet)

#     import matplotlib.pyplot as plt
#     # Plot the data
#     ax = heating_oil_df.plot(x='year', y=['heating_oil_residential_price_dollars_per_gallon_r1x', 'heating_oil_residential_price_dollars_per_gallon_sri'],
#                             label=['R1X', 'SRI'], kind='line')
#     ax.legend(['R1X', 'SRI'])
#     plt.show()


def main(state_abbrev, duoarea, fuel_type):
    try:
        with open("/workspaces/reports2/.secrets/config") as f:
            secrets = dict(line.strip().split("=", 1) for line in f if "=" in line.strip())
        EIA_API_KEY = secrets["EIA_API_KEY"]
    except Exception as e:
        print(f"Error: {e}")
        return None

    path_parquet = get_eia_petroleum_data(
        state_abbrev=state_abbrev,
        duoarea=duoarea,
        api_key=EIA_API_KEY,
        frequency="weekly",
        retail_or_wholesale="retail",
        fuel_type=fuel_type,
    )

    clean_eia_petroleum_data(path_parquet, state_abbrev, fuel_type)


if __name__ == "__main__":
    import sys

    state_to_duoarea = {"CT": "SCT", "RI": "SRI", "PADD_1A": "R1X"}

    # Get list of area codes from command line or use default ['SRI', 'R1X']
    if len(sys.argv) > 3:
        print(len(sys.argv))
        print(
            "Input is a state abbreviation, PADD code (regional), or 'list' to list all available states and PADD codes"
        )

    elif sys.argv[1] == "list":
        print("Available states and PADD codes (EIA alias):")
        print("--------------------------------")
        [print(f"{key} ({state_to_duoarea[key]})") for key in sorted(state_to_duoarea.keys())]
    elif sys.argv[1].upper() in list(state_to_duoarea.keys()):
        state_abbrev = sys.argv[1].upper()
        duoarea = state_to_duoarea[state_abbrev]
        main(state_abbrev, duoarea, sys.argv[2])
    else:
        print(f"Invalid state abbreviation: {sys.argv[1]}")
