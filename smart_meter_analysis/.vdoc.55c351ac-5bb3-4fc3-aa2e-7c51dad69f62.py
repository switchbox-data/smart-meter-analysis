# type: ignore
# flake8: noqa
#
import polars as pl
import cenpy as cen
import pygris

# After installing the relevant packages, we retrieve the ACS census data.

# Finding the correct survey and year
available_surveys = cen.explorer.available()
available_surveys['title']

# API connection
conn = cen.remote.APIConnection('ACSDP5Y2023')

vars_to_download = {
  'DP02_0001E': 'Households by type: Total Households',
  'DP02_0016E': 'Average household size',
  'DP02_0017E': 'Average family size',
  'DP02_0059E': 'Educational attainment 25 years and over',
  'DP03_0052E': 'Income and benefits less than $10,000',
  'DP03_0053E': 'Income and benefits $10,000 to $14,999',
  'DP03_0054E': 'Income and benefits $15,000 to $24,999',
  'DP03_0055E': 'Income and benefits $25,000 to $34,999',
  'DP03_0056E': 'Income and benefits 35,000 to $49,999',
  'DP03_0057E': 'Income and benefits 50,000 to $74,999',
  'DP03_0058E': 'Income and benefits $75,000 to $99,999',
  'DP03_0059E': 'Income and benefits $100,000 to $149,999',
  'DP03_0060E': 'Income and benefits 150,000 to $199,999',
  'DP03_0061E': 'Income and benefits 200,000 or more',
  'DP03_0062E': 'Median household income',
  'DP04_0001E': 'Total housing units',
  'DP04_0002E': 'Occupied housing units',
  'DP04_0003E': 'Vacant housing units',
  'DP04_0017E': 'Built 2020 or after',
  'DP04_0018E': 'Built 2010 to 2019',
  'DP04_0019E': 'Built 2000 to 2009',
  'DP04_0020E': 'Built 1990 to 1999',
  'DP04_0021E': 'Built 1980 to 1989',
  'DP04_0022E': 'Built 1970 to 1979',
  'DP04_0023E': 'Built 1960 to 1969',
  'DP04_0024E': 'Built 1950 to 1959',
  'DP04_0025E': 'Built 1940 to 1949',
  'DP04_0026E': 'Built 1939 or earlier',
  'DP04_0028E': '1 room',
  'DP04_0029E': '2 rooms',
  'DP04_0030E': '3 rooms',
  'DP04_0031E': '4 rooms',
  'DP04_0032E': '5 rooms',
  'DP04_0033E': '6 rooms',
  'DP04_0034E': '7 rooms',
  'DP04_0035E': '8 rooms',
  'DP04_0036E': '9 rooms or more',
  'DP04_0046E': 'Owner occupied',
  'DP04_0047E': 'Renter occupied',
  'DP04_0063E': 'Utility gas heating',
  'DP04_0065E': 'Electric heating',
  'DP05_0006E': '5 to 9 years old',
  'DP05_0007E': '10 to 14 years old',
  'DP05_0008E': '15 to 19 years old',
  'DP05_0009E': '20 to 24 years old',
  'DP05_0010E': '25 to 34 years old',
  'DP05_0011E': '35 to 44 years old',
  'DP05_0012E': '45 to 54 years old',
  'DP05_0013E': '55 to 59 years old',
  'DP05_0014E': '60 to 64 years old',
  'DP05_0015E': '65 to 74 years old',
  'DP05_0016E': '75 to 84 years old',
  'DP05_0017E': '85 years or older',
  'DP04_0081E': 'Value less than $50,000',
  'DP04_0082E': 'Value $50,000 to $99,999',
  'DP04_0083E': 'Value $100,000 to $149,999',
  'DP04_0084E': 'Value $150,000 to $199,999',
  'DP04_0085E': 'Value $200,000 to $299,999',
  'DP04_0086E': 'Value $300,000 to $499,999',
  'DP04_0087E': 'Value $500,000 to $999,999',
  'DP04_0088E': 'Value $1,000,000 or more'
}

vars_to_download_list = list(vars_to_download.keys())

df_pandas = conn.query(
    cols=['NAME', 'GEOID', 'state', 'county', 'tract'] + vars_to_download_list,
    geo_unit='tract',
    geo_filter={'state': '17'}
)
# 5. Convert to Polars
df_polars_raw = pl.from_pandas(df_pandas)

# 6. Rename columns in Polars
rename_map = {
k: v for k, v in vars_to_download.items() if k in df_polars_raw.columns
}
df_polars = df_polars_raw.rename(rename_map)
#
#
#
#
#
