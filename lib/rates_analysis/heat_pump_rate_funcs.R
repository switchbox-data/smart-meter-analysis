library(tidyverse)
library(scales)
library(viridis)
library(ggplot2)
library(patchwork)
library(lubridate)
library(arrow)


########################################################
# Constants and Conversions
########################################################

# Gas
# https://www.rienergy.com/site/-/media/rie-jss-app/home/ways-to-save/rates-and-shopping/service-rates/residential-rates/Rates-5-01-25.ashx?sc_lang=en&hash=EB0DA36E10360398E10E6365C5AD7FBB
convert_gas_therm_to_mmbtu <- 1
convert_gas_therm_to_ccf <- 1 / 1.028
convert_gas_ccf_to_therm <- 1.028
convert_gas_therm_to_kwh <- 29.3

# Delivered Fuels
convert_gal_fuel_oil_to_kwh <- 40.2778
convert_propane_gal_to_kwh <- 26.8


########################################################
# Misc Utility Functions
########################################################

## --- Adding/Manipulating housing_units table ---
add_upgrade_alias_columns <- function(path_rs_db, table) {
  # Connect to database
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)

  # Pull table
  df <- tbl(con, table) |> collect()

  # Add columns
  df <- df |>
    mutate(
      hvac = case_when(
        upgrade == '0' ~ "current",
        upgrade == '3' ~ "low_hp",
        upgrade == '4' ~ "high_hp",
        upgrade == '7' ~ "low_hp",
        upgrade == '8' ~ "high_hp",
        upgrade == '9' ~ "high_hp",
        upgrade == '10' ~ "high_hp"
      )
    ) |>
    mutate(
      shell = case_when(
        upgrade == '0' ~ "current",
        upgrade == '3' ~ "current",
        upgrade == '4' ~ "current",
        upgrade == '7' ~ "current",
        upgrade == '8' ~ "current",
        upgrade == '9' ~ "basic_shell",
        upgrade == '10' ~ "enhanced_shell"
      )
    ) |>
    mutate(
      appliances = case_when(
        upgrade == '0' ~ "current",
        upgrade == '3' ~ "current",
        upgrade == '4' ~ "current",
        upgrade == '7' ~ "low_elec",
        upgrade == '8' ~ "high_elec",
        upgrade == '9' ~ "high_elec",
        upgrade == '10' ~ "high_elec"
      )
    )

  # Write back to database
  DBI::dbWriteTable(con, table, df, overwrite = TRUE)

  # Close connection
  DBI::dbDisconnect(con)
}


add_season_column <- function(df) {
  # Winter/Summer Months Enumerations
  summer_months_eversource <- c(6, 7, 8, 9)
  winter_months_eversource <- c(10, 11, 12, 1, 2, 3, 4, 5)

  summer_months_nationalgrid <- c(5, 6, 7, 8, 9, 10)
  winter_months_nationalgrid <- c(11, 12, 1, 2, 3, 4)

  summer_months_unitil <- c(5, 6, 7, 8, 9, 10)
  winter_months_unitil <- c(11, 12, 1, 2, 3, 4)

  summer_months_municipal <- c(6, 7, 8, 9)
  winter_months_municipal <- c(10, 11, 12, 1, 2, 3, 4, 5)

  df |>
    mutate(
      season = case_when(
        electric_utility == "eversource" ~ if_else(
          month %in% summer_months_eversource,
          "summer",
          "winter"
        ),
        electric_utility == "nationalgrid" ~ if_else(
          month %in% summer_months_nationalgrid,
          "summer",
          "winter"
        ),
        electric_utility == "unitil" ~ if_else(
          month %in% summer_months_unitil,
          "summer",
          "winter"
        ),
        electric_utility == "municipal" ~ if_else(
          month %in% summer_months_municipal,
          "summer",
          "winter"
        )
      )
    )
}


add_baseline_heating_type <- function(path_rs_db) {
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)

  housing_units <- tbl(con, "housing_units") |> collect()

  housing_units <- housing_units |>
    select(-baseline_heating_type)

  housing_units <- housing_units |>
    mutate(
      baseline_heating_type = case_when(
        `in.heating_fuel` == 'Natural Gas' ~ 'Natural Gas',
        `in.heating_fuel` == 'Electricity' &
          `in.hvac_heating_type` %in%
            c('Ducted Heating', 'Non-Ducted Heating') ~ 'Resistance',
        `in.heating_fuel` == 'Electricity' &
          `in.hvac_heating_type` == 'Ducted Heat Pump' ~ 'Heat Pump',
        `in.heating_fuel` == 'Fuel Oil' ~ 'Fuel Oil',
        `in.heating_fuel` == 'Propane' ~ 'Propane',
        `in.heating_fuel` == 'Other Fuel' ~ 'Other Fuel',
        TRUE ~ 'None'
      )
    )

  DBI::dbWriteTable(con, "housing_units", housing_units, overwrite = TRUE)

  DBI::dbDisconnect(con)
}

add_baseline_cooling_type <- function(path_rs_db) {
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)

  housing_units <- tbl(con, "housing_units") |> collect()

  # housing_units <- housing_units |>
  #   select(-baseline_cooling_type)

  housing_units <- housing_units |>
    mutate(
      baseline_cooling_type = case_when(
        `in.hvac_cooling_type` == 'Central AC' ~ 'Yes AC',
        `in.hvac_cooling_type` == 'Room AC' ~ 'Yes AC',
        `in.hvac_cooling_type` == 'None' ~ 'No AC',
        TRUE ~ 'No AC'
      )
    )

  DBI::dbWriteTable(con, "housing_units", housing_units, overwrite = TRUE)

  DBI::dbDisconnect(con)
}


add_building_type_group <- function(path_rs_db) {
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)

  housing_units <- tbl(con, "housing_units") |> collect()

  housing_units <- housing_units |>
    mutate(building_type = `in.geometry_building_type_acs`) |>
    mutate(
      building_type_group = case_when(
        `in.geometry_building_type_acs` %in%
          c(
            "Single-Family Detached",
            "Mobile Home",
            "Single-Family Attached"
          ) ~ "Single-Family",
        `in.geometry_building_type_acs` %in%
          c("2 Unit", "3 or 4 Unit") ~ "2-4 Units",
        `in.geometry_building_type_acs` %in%
          c(
            "5 to 9 Unit",
            "10 to 19 Unit",
            "20 to 49 Unit",
            "50 or more Unit"
          ) ~ "5+ Units",
        TRUE ~ "Other"
      )
    )

  DBI::dbWriteTable(con, "housing_units", housing_units, overwrite = TRUE)

  DBI::dbDisconnect(con)
}


# get_income_distribution() - get the distribution within bounds of income
get_income_distribution <- function(
  min_income = 0,
  max_income = 500000,
  state = "MA",
  year = 2022
) {
  pums_path <- paste0(
    "/workspaces/reports2/data/census/pums/pums_",
    state,
    "_",
    year,
    ".Rds"
  )

  if (file.exists(pums_path)) {
    message("Loading PUMS data from cache...")
    pums_data <- readRDS(pums_path)
  } else {
    message("Fetching PUMS data from Census API...")
    # Define variables we need
    vars <- c("HINCP", "WGTP")

    # Fetch PUMS data
    pums_data <- get_pums(
      variables = vars,
      state = state,
      year = year,
      survey = "acs1",
      rep_weights = "housing",
      recode = TRUE
    ) |>
      distinct(SERIALNO, .keep_all = TRUE)

    saveRDS(pums_data, pums_path)
  }

  # Filter to our income bracket and remove NAs
  min_income <- max(8000, min_income)
  income_bracket <- pums_data |>
    filter(HINCP > min_income, HINCP <= max_income) |>
    filter(!is.na(HINCP), !is.na(WGTP))
}


assign_precise_income_dollars <- function(
  path_rs_db,
  state = 'MA',
  year = 2022
) {
  # Setting precise dollar income levels
  source("/workspaces/reports2/lib/inflation.R")
  library(fredr)
  library(tidycensus)

  # 3 Steps
  # -------
  # 1. For each ResStock income bracket of N, get the "real" income distribution from the Census; from the Census PUMS
  #     income distribution, take N random samples and assign them to bldg_ids in the ResStock database
  # 2. Inflate those income values from 2019 (year of ResStock data) to 2024 (current year)
  # 3. Write the housing_units table back to the database

  # 0. Get the housing_units table
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)
  on.exit(DBI::dbDisconnect(con))

  housing_units <- tbl(con, "housing_units") |>
    collect()

  # Drop income-related columns if they exist
  cols_to_drop <- c(
    "income_dollars",
    "income_low",
    "income_high",
    "income_sim",
    "assigned_income",
    "assigned_income_uninflated",
    "assigned_income_uninflated.x",
    "assigned_income_uninflated.y"
  )

  housing_units <- housing_units |>
    select(-any_of(cols_to_drop))

  housing_units_with_ranges <- housing_units |>
    mutate(
      income = case_match(
        in.income,
        "<10000" ~ "0-10000",
        "200000+" ~ "200000-500000",
        "Not Available" ~ NA,
        .default = in.income
      )
    ) |>
    mutate(
      income_low = if_else(
        !is.na(income),
        as.numeric(str_extract(income, "^[0-9]+")),
        NA_real_
      ),
      income_high = if_else(
        !is.na(income),
        as.numeric(str_extract(income, "[0-9]+$")),
        NA_real_
      )
    )

  # Create an empty tibble to store results
  housing_with_incomes <- tibble()

  # Process each income bracket separately
  for (income_bracket in unique(housing_units_with_ranges$in.income)) {
    # Get buildings in this income bracket
    buildings_in_bracket <- housing_units_with_ranges |>
      filter(in.income == income_bracket) |>
      select(bldg_id, income_low, income_high)

    # # Skip if missing income bounds
    # if (nrow(buildings_in_bracket) == 0 ||
    #     is.na(buildings_in_bracket$income_low[1]) ||
    #     is.na(buildings_in_bracket$income_high[1])) {
    #   next
    # }

    # Get income distribution for this bracket
    distribution <- get_income_distribution(
      min_income = buildings_in_bracket$income_low[1],
      max_income = buildings_in_bracket$income_high[1]
    )

    # Skip if no valid distribution
    # if (nrow(distribution) == 0) {
    #   next
    # }

    # Sample incomes directly
    n_buildings <- nrow(buildings_in_bracket)
    sampled_incomes <- sample(
      x = distribution$HINCP,
      size = n_buildings,
      replace = TRUE,
      prob = distribution$WGTP
    )

    # Assign to buildings
    bracket_result <- buildings_in_bracket |>
      select(bldg_id, income_low, income_high) |> # Just keep bldg_id
      mutate(assigned_income_uninflated = sampled_incomes)

    # Add to results
    housing_with_incomes <- bind_rows(housing_with_incomes, bracket_result)
  }

  # Join back to housing_units to add the assigned incomes
  housing_units_with_incomes <- housing_units |>
    left_join(housing_with_incomes, by = "bldg_id")

  # ------------------------------------------
  # 2. Inflate those income values from 2019 (year of ResStock data) to 2024 (current year)
  #census_api_key(api_key = "c79b508d59918868944680000000000000000000")
  FRED_KEY <- file("/workspaces/reports2/.secrets/fred.key") |> readLines() # Needs API key
  fredr_set_key(FRED_KEY)

  # inflation adjustment based on the Employment Cost Index
  # source: https://fred.stlouisfed.org/series/CIU2020000000212I
  ECI <- "CIU2020000000212I" # Employment Cost Index: Wages and salaries for Private industry workers in the Middle Atlantic
  wage_index <- get_inflation_index(
    series = ECI,
    start = "2019-01-01",
    end = "2024-12-31",
    api_key = FRED_KEY
  )

  # pull the 2019 inflation rate
  inflation_adj_resstock <- get_inflation_factor(
    wage_index,
    input_year = 2019,
    target_year = 2024
  )

  # Apply inflation adjustment to assigned incomes
  housing_units_with_incomes <- housing_units_with_incomes |>
    mutate(
      assigned_income = assigned_income_uninflated * inflation_adj_resstock
    )

  # ------------------------------------------
  # 3. Write the housing_units table back to the database
  DBI::dbWriteTable(
    con,
    "housing_units",
    housing_units_with_incomes,
    overwrite = TRUE
  )

  DBI::dbDisconnect(con)
}

get_housing_units_column_counts <- function(
  column_tariff_name,
  county_code_rs
) {
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)

  if (county_code_rs == 'statewide') {
    counts <- DBI::dbGetQuery(
      con,
      sprintf(
        "
          SELECT
            %s,
            COUNT(*) as count
          FROM housing_units
          GROUP BY %s
          ORDER BY count DESC
        ",
        column_tariff_name,
        column_tariff_name
      )
    )
  } else {
    counts <- DBI::dbGetQuery(
      con,
      sprintf(
        "
          SELECT
            %s,
            COUNT(*) as count
          FROM housing_units
          WHERE \"in.county\" IN ('%s')
          GROUP BY %s
          ORDER BY count DESC
        ",
        column_tariff_name,
        county_code_rs,
        column_tariff_name
      )
    )
  }

  DBI::dbDisconnect(con)

  # Add percentage column
  counts <- counts |>
    mutate(pct = round((count / sum(count)) * 100, 1))

  return(counts)
}

get_bldgs_by_heating_fuel <- function(path_rs_db, heating_fuel) {
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  heated_bldgs <- tbl(con, "housing_units") |>
    filter(
      (`in.heating_fuel` == heating_fuel) &
        (`in.hvac_cooling_type` != "Heat Pump")
    ) |>
    select(bldg_id) |>
    collect()

  return(heated_bldgs)
}

get_bldgs_by_building_type <- function(path_rs_db, building_types) {
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  heated_bldgs <- tbl(con, "housing_units") |>
    filter(`in.geometry_building_type_acs` %in% building_types) |>
    select(bldg_id) |>
    collect()

  return(heated_bldgs)
}

## --- Resampling functions ---
resample_to_match_housing_distribution <- function(
  annual_change_table,
  target_pcts,
  print_summary_table = FALSE
) {
  # Get the order of the building type groups from the target percentages
  # This function resamples a dataset to match target housing type distributions
  # Steps:
  # 1. The the current count and pct of each building type group
  # 2. Calculate "availability": pct_original / target_pcts
  # 3. Calculate weights: The group with the lowest availability will keep all its samples, so weight = 1.0
  #     Other groups will have weights < 1.0 (min_availability / availability)
  # 4. Calculate target counts: weights * n_buildings_original
  # 5. Sample the building IDs for each type
  # 6. Return the filtered dataframe containing only the sampled building IDs that match the target distribution

  # Args:
  #   annual_change_table: A dataframe containing building data with columns:
  #     - building_type_group: The type of building (Single-Family, 2-4 Units, 5+ Units)
  #     - bldg_id: Unique identifier for each building
  #   target_pcts: tariff_named vector of target percentages for each building type group
  #   print_summary_table: Boolean, whether to print summary statistics
  # Returns:
  #   A filtered dataframe containing only the sampled building IDs that match
  #   the target distribution
  building_type_counts_order <- tariff_names(target_pcts)

  # First get counts of each building type in original data
  building_type_counts <- annual_change_table |>
    group_by(building_type_group) |>
    summarise(n_buildings_original = n_distinct(bldg_id)) |>
    mutate(pct_original = n_buildings_original / sum(n_buildings_original)) |>
    arrange(match(building_type_group, building_type_counts_order))

  # Calculate how many samples we can keep for each type
  # We'll use the smallest group as our baseline to ensure we don't exceed any group's count
  availability <- building_type_counts$pct_original /
    target_pcts[building_type_counts$building_type_group]

  min_availability <- min(availability)

  weights <- c(
    (min_availability / availability[1]),
    (min_availability / availability[2]),
    (min_availability / availability[3])
  )

  # Calculate target counts for each type
  target_counts <- c(
    floor(weights[1] * building_type_counts$n_buildings_original[1]),
    floor(weights[2] * building_type_counts$n_buildings_original[2]),
    floor(weights[3] * building_type_counts$n_buildings_original[3])
  )
  tariff_names(target_counts) <- building_type_counts_order

  # Create a tariff_named vector of target counts that matches the building_type_group tariff_names
  target_counts_tariff_named <- settariff_names(
    target_counts,
    building_type_counts_order
  )

  # Use map_dfr to create the sampled IDs data frame in one operation
  sampled_bldg_ids <- map_dfr(
    building_type_counts_order,
    function(group) {
      sampled_ids <- annual_change_table |>
        filter(building_type_group == group) |>
        distinct(bldg_id) |>
        pull(bldg_id) |>
        sample(size = target_counts_tariff_named[group])

      tibble(
        bldg_id = sampled_ids,
        building_type_group = group
      )
    }
  )

  # Then get all rows for those buildings
  annual_change_table_resampled <- annual_change_table |>
    inner_join(sampled_bldg_ids, by = c("bldg_id", "building_type_group"))

  if (print_summary_table) {
    building_type_counts_resampled <- annual_change_table_resampled |>
      group_by(building_type_group) |>
      summarise(n_buildings_resampled = n_distinct(bldg_id)) |>
      mutate(
        pct_resampled = n_buildings_resampled / sum(n_buildings_resampled)
      ) |>
      arrange(match(building_type_group, building_type_counts_order))

    # Create a summary table combining all the information
    summary_table <- building_type_counts |>
      # Add availability and weights
      mutate(
        availability = availability,
        weights = weights,
        target_counts = target_counts
      ) |>
      # Join with resampled counts
      left_join(
        building_type_counts_resampled,
        by = "building_type_group"
      ) |>
      # Format percentages
      mutate(
        pct_original = scales::percent(pct_original, accuracy = 0.1),
        pct_resampled = scales::percent(pct_resampled, accuracy = 0.1)
      ) |>
      # Select and arrange columns
      select(
        building_type_group,
        n_buildings_original,
        pct_original,
        availability,
        weights,
        target_counts,
        n_buildings_resampled,
        pct_resampled
      ) |>
      # Arrange by building type order
      arrange(match(building_type_group, building_type_counts_order))

    # Print the summary table
    print(summary_table)
  }

  return(annual_change_table_resampled)
}

########################################################
# Rate Analysis Functions
########################################################
calc_stats_by_rate_version <- function(
  annual_change_table,
  hp_eff_for_stats,
  print_table = TRUE
) {
  scenario_stats <- annual_change_table |>
    filter(hvac == hp_eff_for_stats) |>
    group_by(version_elec) |>
    summarise(
      median_change = median(annual_bill_change),
      pct_that_save = mean(annual_bill_change < 0) * 100,
      pct_that_save_big = mean(annual_bill_change < -1000) * 100,
      pct_that_lose = mean(annual_bill_change > 0) * 100,
      pct_that_lose_big = mean(annual_bill_change > 1000) * 100,
      median_savings = median(annual_bill_change[annual_bill_change < 0]),
      median_loss = median(annual_bill_change[annual_bill_change > 0])
    )

  if (print_table) {
    print(knitr::kable(
      scenario_stats,
      caption = glue::glue(
        "Summary Statistics by Rate Version (HP Efficiency = {hp_eff_for_stats})"
      ),
      format = "pipe",
      digits = 1,
      col.tariff_names = c(
        "Rate Version",
        "Median Bill Change ($)",
        "% That Save",
        "Median Savings ($)",
        "Median Loss ($)"
      )
    ))
  }

  return(scenario_stats)
}

########################################################
# Supply Rate Functions
########################################################
get_month_hour_supply <- function(
  supply_rates,
  start_year,
  end_year,
  dynamic_or_all_hours,
  all_hours_tariff_name,
  on_peak_tariff_name,
  off_peak_tariff_name
) {
  # Extract supply rates for given year
  supply_rates_year <- supply_rates |>
    filter(
      year >= start_year & year <= end_year,
      tariff_name %in%
        c(all_hours_tariff_name, on_peak_tariff_name, off_peak_tariff_name)
    ) |>
    select(
      month = month,
      hour = 0,
      tariff_name = tariff_name,
      rate = rate
    )

  # Define peak/off-peak hours
  on_peak_hours <- c(
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23
  )
  off_peak_hours <- c(0, 1, 2, 3, 4, 5, 6, 7)

  # Create month-hour grid
  month_hour_grid <- expand.grid(
    month = 1:12,
    hour = 0:23
  ) |>
    arrange(month, hour)

  month_hour_supply <- month_hour_grid |>
    mutate(
      rate_type = if (dynamic_or_all_hours == "dynamic") {
        case_when(
          hour %in% on_peak_hours ~ on_peak_tariff_name,
          hour %in% off_peak_hours ~ off_peak_tariff_name,
          TRUE ~ all_hours_tariff_name # Fallback to all-hours rate
        )
      } else {
        all_hours_tariff_name # Use all_hours for every row when not dynamic
      }
    ) |>
    left_join(
      supply_rates_year,
      by = c("month", "rate_type" = "tariff_name") # Join on both month and rate type
    ) |>
    select(month, hour, supply_rate = rate) # Select and rename the rate column
  return(month_hour_supply)
}

get_supply_rates_monthly <- function(
  supply_rates,
  target_year = 2024,
  type = NULL,
  zone = NULL,
  electric_utility = NULL
) {
  supply_rates_year <- supply_rates |>
    filter(year == as.numeric(target_year)) |>
    mutate(
      year = year,
      month = month,
      type = type,
      zone = zone,
      electric_utility = electric_utility,
      supply_rate = supply_rate
    ) |>
    select(year, month, type, zone, electric_utility, supply_rate)

  if (!is.null(type)) {
    supply_rates_year <- supply_rates_year |> filter(type == !!type)
  }
  if (!is.null(zone)) {
    supply_rates_year <- supply_rates_year |> filter(zone == !!zone)
  }
  if (!is.null(electric_utility)) {
    supply_rates_year <- supply_rates_year |>
      filter(electric_utility == !!electric_utility)
  }

  return(supply_rates_year)
}

########################################################
# Delivery Tariff Functions
########################################################
assign_tariffs_to_month_hour_grid <- function(tariffs) {
  #' Assign tariffs to a month-hour grid
  #'
  #' Takes a data frame of tariffs with month and hour ranges and expands them into a complete
  #' month-hour grid with assigned rates. Handles tariffs that wrap around calendar boundaries.
  #'
  #' @param tariffs A data frame containing tariff definitions with columns:
  #'   \itemize{
  #'     \item start_month - Starting month (1-12)
  #'     \item end_month - Ending month (1-12)
  #'     \item start_hour - Starting hour (0-23)
  #'     \item end_hour - Ending hour (0-23)
  #'     \item tariff_tariff_name - tariff_name/identifier of the tariff
  #'     \item value - Rate value for the tariff
  #'   }
  #'
  #' @return A data frame with columns:
  #'   \itemize{
  #'     \item month - Month number (1-12)
  #'     \item hour - Hour number (0-23)
  #'     \item tariff_tariff_name - tariff_name of the tariff
  #'     \item rate - Rate value for that month-hour combination
  #'   }
  #'
  #' @details
  #' The function handles tariffs that wrap around calendar boundaries (e.g. Nov-Feb)
  #' or day boundaries (e.g. 22:00-06:00). It first creates a complete month-hour grid,
  #' then joins the expanded tariff definitions to assign rates to each time slot.
  #' The result is pivoted to create separate columns for each tariff type before
  #' being converted back to long format.
  #'
  #' @examples
  #' tariffs <- data.frame(
  #'   start_month = c(6, 10),
  #'   end_month = c(9, 5),
  #'   start_hour = c(14, 0),
  #'   end_hour = c(20, 23),
  #'   tariff_tariff_name = c("r3", "r4"),
  #'   value = c(0.25, 0.15)
  #' )
  #' month_hour_rates <- assign_tariffs_to_month_hour_grid(tariffs)

  # Build the month-hour grid
  month_hour_grid <- expand.grid(
    month = 1:12,
    hour = 0:23
  ) |>
    arrange(month, hour)

  # Process each tariff and create a lookup table
  lookup_table <- tariffs |>
    # Must operate rowwise()
    rowwise() |>
    # create lists for months and hours
    mutate(
      months = list(
        if (start_month <= end_month) {
          start_month:end_month
        } else {
          c(start_month:12, 1:end_month)
        }
      ),
      hours = list(
        if (start_hour <= end_hour) {
          start_hour:end_hour
        } else {
          c(start_hour:23, 0:end_hour)
        }
      )
    ) |>
    # Expand the months and hours
    unnest(months) |>
    unnest(hours) |>
    # Keep relevant columns including tariff_tariff_name
    select(month = months, hour = hours, tariff_tariff_name, value)

  # First pivot to wide format by tariff_tariff_name
  month_hour_tariff_wide <- month_hour_grid |>
    left_join(lookup_table, by = c("month", "hour")) |>
    pivot_wider(
      tariff_names_from = tariff_tariff_name,
      values_from = value
    )

  # Then pivot back to long format
  month_hour_tariff_long <- month_hour_tariff_wide |>
    pivot_longer(
      cols = c(r3, r4), # specify the rate columns
      tariff_names_to = "tariff_tariff_name",
      values_to = "rate"
    ) |>
    # replace any NA values with 0
    mutate(rate = replace_na(rate, 0)) |>
    # Ensure all columns have the correct type
    mutate(
      month = as.integer(month),
      hour = as.integer(hour),
      tariff_tariff_name = as.character(tariff_tariff_name),
      rate = as.numeric(rate)
    )

  return(month_hour_tariff_long)
}

assign_tariffs_to_month_grid <- function(tariffs) {
  # Create a lookup table by expanding the months for each tariff
  lookup_table <- tariffs |>
    # Must operate rowwise()
    rowwise() |>

    # Create a list of months covered by this tariff
    mutate(
      months = list(
        if (start_month <= end_month) {
          start_month:end_month
        } else {
          c(start_month:12, 1:end_month)
        }
      )
    ) |>

    # Expand the months
    unnest(months) |>

    # Select only the columns we need for the final output
    select(
      utility,
      version,
      customer_class,
      tariff_tariff_name,
      domain,
      class,
      type,
      tariff_name,
      value,
      month = months
    )

  # Check for duplicate month combinations
  duplicate_check <- lookup_table |>
    group_by(
      utility,
      version,
      customer_class,
      tariff_tariff_name,
      type,
      month
    ) |>
    summarize(
      count = n(),
      values = paste(unique(value), collapse = ", "),
      .groups = "drop"
    ) |>
    filter(count > 1)

  if (nrow(duplicate_check) > 0) {
    warning("Found duplicate values for some month combinations:")
    print(duplicate_check)

    # Use the first value for each month combination
    lookup_table <- lookup_table |>
      group_by(
        utility,
        version,
        customer_class,
        tariff_tariff_name,
        type,
        month
      ) |>
      slice(1) |>
      ungroup()
  }

  return(lookup_table)
}


########################################################
# Bill Calculation Functions
########################################################
get_monthly_consumption <- function(
  path_monthly_data,
  fuel,
  functional_group,
  use_these_states,
  use_these_upgrades = c("00")
) {
  #' Calculate Monthly Energy Consumption
  #'
  #' This function calculates monthly energy consumption for a given fuel and functional group.
  #'
  #' @param path_monthly_data Character string. Path to the directory containing monthly
  #'   load data in Arrow dataset format.
  #' @param fuel Character string. The fuel to calculate consumption for.
  #' @param functional_group Character string. The functional group to calculate consumption for.
  #' @param use_these_upgrades Character vector. The upgrades to use for the calculation.
  #'
  #' @return A data frame containing monthly energy consumption with the following columns:
  #'   \item bldg_id - Building identifier
  #'   \item upgrade - Upgrade scenario identifier
  #'   \item month - Month number (1-12)
  #'   \item consumption - Energy consumption in kWh
  #'
  #' @examples
  #' \dontrun{
  #' monthly_consumption <- get_monthly_consumption(
  #'   path_monthly_data = "/workspaces/reports2/data/ResStock/2024_release2_tmy3/load_curve_monthly_10/state=RI",
  #'   fuel = "natural_gas",
  #'   functional_group = "heating",
  #'   use_these_upgrades = c("00", "01", "02", "03")
  #' )
  #' }

  # load the data dictionary labeled (ddl) from feather
  ddl <- read_feather(
    "/workspaces/reports2/lib/resstock/2024/end_use_groups.feather"
  )

  # filter ddl by fuel and functional_group to get the target columns
  ddl_filtered <- ddl |>
    filter(
      .data$fuel == .env$fuel,
      .data$functional_group == .env$functional_group
    )

  # target columns
  target_columns <- ddl_filtered$timeseries_field_name

  # Read the dataset from the parquet directory
  data <- open_dataset(path_monthly_data)

  monthly_consumption <- data |>
    #filter(year(timestamp) == 2018) |>
    filter(state %in% use_these_states) |>
    filter(upgrade %in% use_these_upgrades) |>
    mutate(
      bldg_id = as.integer(bldg_id),
      month = as.integer(month(timestamp))
    ) |>
    select(all_of(c("bldg_id", "upgrade", "month", target_columns))) |>
    collect()

  monthly_consumption <- monthly_consumption |>
    mutate(
      "consumption_kwh" := rowSums(across(all_of(target_columns)), na.rm = TRUE)
    ) |>
    select(all_of(c("bldg_id", "upgrade", "month", "consumption_kwh")))

  return(monthly_consumption)
}


# monthly bills
calc_monthly_bills <- function(
  monthly_consumption,
  fuel_type,
  delivery_tariffs,
  supply_rates,
  housing_units,
  supply_year,
  state = NULL,
  enable_lmi_discount = TRUE,
  use_these_upgrades = c("00")
) {
  #' Calculate Monthly Electric Bills
  #'
  #' This function calculates monthly electric bills for housing units based on their
  #' electricity consumption, delivery tariffs, and supply rates. It handles different
  #' utility companies, customer classes, and Low to Moderate Income (LMI) discounts.
  #'
  #' @param path_monthly_data Character string. Path to the directory containing monthly
  #'   load data in Arrow dataset format.
  #' @param fuel_type Character string. The type of fuel to calculate bills for.
  #' @param delivery_tariffs Data frame. Contains electric delivery tariffs
  #'   with columns for utility, customer_class, type (customer_charge, delivery_rate,
  #'   sales_tax_rate), value, month, and lmi status.
  #' @param supply_rates Data frame. Contains electric supply rates with
  #'   columns for electric_utility, month, year, and electric_supply_rate.
  #' @param housing_units Data frame. Contains housing unit metadata including bldg_id,
  #'   electric_utility, baseline_heating_type, lmi status, and discount_rate.
  #' @param supply_year Integer. The year for which to calculate bills (used to filter
  #'   supply rates).
  #' @param state Character string. The state to use for the calculation.
  #' @param use_these_upgrades Character vector. The upgrades to use for the calculation.
  #'
  #' @return A data frame containing monthly electric bills with the following columns:
  #'   \itemize{
  #'     \item bldg_id - Building identifier
  #'     \item upgrade - Upgrade scenario identifier
  #'     \item month - Month number (1-12)
  #'     \item consumption - {fuel_type} consumption in kWh
  #'     \item utility - Utility company tariff_name
  #'     \item baseline_heating_type - Type of heating system
  #'     \item lmi - Low to Moderate Income status (logical)
  #'     \item discount_rate - {fuel_type} discount rate for LMI customers
  #'     \item customer_charge - Fixed monthly customer charge
  #'     \item delivery_rate - {fuel_type} delivery rate per kWh
  #'     \item delivery_charge - Total delivery charges
  #'     \item supply_charge - Total supply charges
  #'     \item total_pretax_bill - Total bill before taxes
  #'     \item sales_tax_charge - Sales tax amount
  #'     \item total_bill - Final total bill including taxes
  #'   }
  #'
  #' @details The function performs the following operations:
  #'   \enumerate{
  #'     \item Loads monthly electricity consumption data from the specified path
  #'     \item Filters for baseline upgrades (upgrade "00") and year 2018 data
  #'     \item Joins consumption data with housing unit metadata
  #'     \item Applies appropriate delivery tariffs based on utility and LMI status
  #'     \item Calculates delivery charges, supply charges, and taxes
  #'     \item Returns comprehensive billing information for each building-month
  #'   }
  #'
  #' @note The function uses Arrow datasets for efficient processing of large monthly
  #'   load files. It assumes the monthly load data contains columns for bldg_id,
  #'   upgrade, timestamp, and out.electricity.total.energy_consumption.
  #'
  #' @seealso \code{\link{calc_annual_bills_from_monthly}} for aggregating monthly
  #'   bills to annual totals
  #'
  #' @examples
  #' \dontrun{
  #' monthly_bills <- calc_monthly_bills_elec(
  #'   path_monthly_data = "/path/to/monthly/loads",
  #'   electric_delivery_tariffs = delivery_tariffs,
  #'   electric_supply_rates = supply_rates,
  #'   housing_units = housing_data,
  #'   year = 2024
  #' )
  #' }
  # upgrade_filters <- paste0("upgrade == '", use_these_upgrades, "'", collapse = " | ")
  # upgrade_filter_expr <- parse(text = upgrade_filters)[[1]]

  # Set some columns names based on fuel_type
  # ------------
  # Electricity
  if (fuel_type == "electricity") {
    fuel_consumption_column <- "out.electricity.total.energy_consumption"
    discount_rate <- "discount_rate_elec"
    utility <- "electric_utility"
    # ------------
    # Gas
  } else if (fuel_type == "natural_gas") {
    fuel_consumption_column <- "out.natural_gas.total.energy_consumption"
    discount_rate <- "discount_rate_gas"
    utility <- "gas_utility"
    # ------------
    # Fuel Oil
  } else if (fuel_type == "fuel_oil") {
    fuel_consumption_column <- "out.fuel_oil.total.energy_consumption"
    discount_rate <- "discount_rate_fuel_oil"
    utility <- "fuel_oil_utility"
    # ------------
    # Propane
  } else if (fuel_type == "propane") {
    fuel_consumption_column <- "out.propane.total.energy_consumption"
    discount_rate <- "discount_rate_propane"
    utility <- "propane_utility"
  }
  # ------------

  final_result <- monthly_consumption |>
    left_join(
      housing_units |>
        mutate(bldg_id = as.integer(bldg_id)) |>
        select(
          bldg_id,
          upgrade,
          hvac,
          !!sym(utility),
          lmi,
          !!sym(discount_rate)
        ) |>
        mutate(
          !!sym(discount_rate) := if (enable_lmi_discount) {
            !!sym(discount_rate)
          } else {
            0
          }
        ),
      by = c("bldg_id", "upgrade")
    ) |>
    left_join(
      delivery_tariffs |>
        filter(type == "customer_charge") |>
        select(
          month,
          !!sym(utility),
          customer_charge = value,
          tariff_name,
          version,
          lmi
        ),
      by = c("month", utility, "lmi")
    ) |>
    left_join(
      delivery_tariffs |>
        filter(type == "delivery_rate") |>
        select(
          month,
          !!sym(utility),
          delivery_rate = value,
          tariff_name,
          version,
          lmi
        ),
      by = c("month", utility, "version", "lmi", "tariff_name")
    ) |>
    left_join(
      delivery_tariffs |>
        filter(type == "sales_tax_rate") |>
        select(
          month,
          !!sym(utility),
          sales_tax_rate = value,
          tariff_name,
          version,
          lmi
        ),
      by = c("month", utility, "version", "lmi", "tariff_name")
    ) |>
    left_join(
      supply_rates |>
        filter(year == supply_year) |>
        select(month, !!sym(utility), supply_rate, year),
      by = c("month", utility)
    ) |>
    mutate(
      customer_charge = customer_charge * (1 - !!sym(discount_rate)),
      delivery_charge = consumption_kwh *
        delivery_rate *
        (1 - !!sym(discount_rate)),
      supply_charge = consumption_kwh *
        supply_rate *
        (1 - !!sym(discount_rate)),
      total_pretax_bill = delivery_charge + supply_charge + customer_charge,
      sales_tax_charge = total_pretax_bill * sales_tax_rate,
      monthly_bill = total_pretax_bill + sales_tax_charge,
      monthly_bill_undiscounted = monthly_bill / (1 - !!sym(discount_rate))
    ) |>
    select(
      bldg_id,
      upgrade,
      hvac,
      month,
      year,
      !!sym(utility),
      consumption_kwh,
      version,
      tariff_name,
      delivery_rate,
      supply_rate,
      customer_charge,
      delivery_charge,
      supply_charge,
      total_pretax_bill,
      sales_tax_charge,
      monthly_bill,
      monthly_bill_undiscounted
    ) |>
    arrange(bldg_id, month, year, !!sym(utility), version, tariff_name) |>
    collect()

  return(final_result)
}

calc_monthly_bills_gas <- function(
  path_monthly_data,
  gas_delivery_tariffs,
  gas_supply_rates,
  housing_units,
  supply_year,
  state = NULL,
  enable_lmi_discount = TRUE,
  use_these_upgrades = c("00")
) {
  #' Calculate Monthly Gas Bills
  #'
  #' This function calculates monthly electric bills for housing units based on their
  #' gas consumption, delivery tariffs, and supply rates. It handles different
  #' utility companies, customer classes, and Low to Moderate Income (LMI) discounts.
  #'
  #' @param path_monthly_data Character string. Path to the directory containing monthly
  #'   load data in Arrow dataset format.
  #' @param gas_delivery_tariffs Data frame. Contains gas delivery tariffs
  #'   with columns for utility, customer_class, type (customer_charge, delivery_rate,
  #'   sales_tax_rate), value, month, and lmi status.
  #' @param gas_supply_rates Data frame. Contains gas supply rates with
  #'   columns for gas_utility, month, year, and gas_supply_rate.
  #' @param housing_units Data frame. Contains housing unit metadata including bldg_id,
  #'   gas_utility, baseline_heating_type, lmi status, and discount_rate_gas.
  #' @param supply_year Integer. The year for which to calculate bills (used to filter
  #'   supply rates).
  #' @param state Character string. The state to use for the calculation.
  #' @param use_these_upgrades Character vector. The upgrades to use for the calculation.
  #'
  #' @return A data frame containing monthly electric bills with the following columns:
  #'   \itemize{
  #'     \item bldg_id - Building identifier
  #'     \item upgrade - Upgrade scenario identifier
  #'     \item month - Month number (1-12)
  #'     \item gas_consumption - Gas consumption in Therms
  #'     \item gas_utility - Utility company tariff_name
  #'     \item baseline_heating_type - Type of heating system
  #'     \item lmi - Low to Moderate Income status (logical)
  #'     \item discount_rate_gas - Gas discount rate for LMI customers
  #'     \item customer_charge - Fixed monthly customer charge
  #'     \item delivery_rate - Delivery rate per kWh                <- YES, per kWh
  #'     \item delivery_charge - Total delivery charges
  #'     \item supply_charge - Total supply charges
  #'     \item total_pretax_bill - Total bill before taxes
  #'     \item sales_tax_charge - Sales tax amount
  #'     \item total_bill - Final total bill including taxes
  #'   }
  #'
  #' @details The function performs the following operations:
  #'   \enumerate{
  #'     \item Loads monthly electricity consumption data from the specified path
  #'     \item Filters for baseline upgrades (upgrade "00") and year 2018 data
  #'     \item Joins consumption data with housing unit metadata
  #'     \item Applies appropriate delivery tariffs based on utility and LMI status
  #'     \item Calculates delivery charges, supply charges, and taxes
  #'     \item Returns comprehensive billing information for each building-month
  #'   }
  #'
  #' @note The function uses Arrow datasets for efficient processing of large monthly
  #'   load files. It assumes the monthly load data contains columns for bldg_id,
  #'   upgrade, timestamp, and out.electricity.total.energy_consumption.
  #'
  #' @seealso \code{\link{calc_annual_bills_from_monthly}} for aggregating monthly
  #'   bills to annual totals
  #'
  #' @examples
  #' \dontrun{
  #' monthly_bills <- calc_monthly_bills_elec(
  #'   path_monthly_data = "/path/to/monthly/loads",
  #'   electric_delivery_tariffs = delivery_tariffs,
  #'   electric_supply_rates = supply_rates,
  #'   housing_units = housing_data,
  #'   year = 2024
  #' )
  #' }

  # Build the dataset path based on state if provided
  if (!is.null(state)) {
    # For partitioned datasets, Arrow can filter more efficiently if we point
    # directly to the state partition
    data_path <- file.path(path_monthly_data, paste0("state=", toupper(state)))
    if (!dir.exists(data_path)) {
      stop("State partition not found: ", data_path)
    }
    print(data_path)

    # Option 1: Open the dataset at the state level and filter by upgrades
    # Arrow will automatically detect the upgrade partitions
    monthly_load_ds <- open_dataset(data_path) |>
      filter(upgrade %in% use_these_upgrades)
  } else {
    # Open the full dataset if no state specified
    monthly_load_ds <- open_dataset(path_monthly_data)

    # Need to filter by upgrades when no state is specified
    # Build a filter expression
    if (length(use_these_upgrades) == 1) {
      monthly_load_ds <- monthly_load_ds |>
        filter(upgrade == use_these_upgrades[1])
    } else if (length(use_these_upgrades) == 2) {
      monthly_load_ds <- monthly_load_ds |>
        filter(
          upgrade == use_these_upgrades[1] | upgrade == use_these_upgrades[2]
        )
    } else {
      # For more upgrades, build dynamically
      filter_expr <- paste0(
        "upgrade == '",
        use_these_upgrades,
        "'",
        collapse = " | "
      )
      monthly_load_ds <- monthly_load_ds |>
        filter(eval(parse(text = paste0("(", filter_expr, ")"))))
    }
  }

  # Now do all operations in a single pipeline
  final_result <- monthly_load_ds |>
    filter(year(timestamp) == 2018) |>
    mutate(
      month = as.integer(month(timestamp)),
      bldg_id = as.integer(bldg_id)
    ) |>
    rename(gas_consumption = out.natural_gas.total.energy_consumption) |>
    select(bldg_id, upgrade, month, gas_consumption) |>

    # Add housing unit metadata
    left_join(
      housing_units |>
        mutate(bldg_id = as.integer(bldg_id)) |>

        # for gas, we need to flag heat_non_heat
        mutate(
          heat_non_heat = case_when(
            hvac == "natural_gas" ~ "heat",
            TRUE ~ "non_heat"
          )
        ) |>
        select(
          bldg_id,
          upgrade,
          gas_utility,
          lmi,
          hvac,
          heat_non_heat,
          discount_rate_gas
        ) |>
        mutate(
          discount_rate_gas = if (enable_lmi_discount) discount_rate_gas else 0
        ),
      by = c("bldg_id", "upgrade")
    ) |>

    # Inline the tariff filtering and selection directly in joins
    left_join(
      gas_delivery_tariffs |>
        filter(type == "customer_charge") |>
        select(
          month,
          gas_utility,
          customer_charge = value,
          lmi,
          heat_non_heat,
          tariff_name,
          version
        ),
      by = c("month", "gas_utility", "lmi", "heat_non_heat")
    ) |>
    left_join(
      gas_delivery_tariffs |>
        filter(type == "delivery_rate") |>
        select(
          month,
          gas_utility,
          delivery_rate = value,
          lmi,
          heat_non_heat,
          tariff_name,
          version
        ),
      by = c(
        "month",
        "gas_utility",
        "lmi",
        "heat_non_heat",
        "tariff_name",
        "version"
      )
    ) |>
    left_join(
      gas_delivery_tariffs |>
        filter(type == "sales_tax_rate") |>
        select(
          month,
          gas_utility,
          sales_tax_rate = value,
          lmi,
          heat_non_heat,
          tariff_name,
          version
        ),
      by = c(
        "month",
        "gas_utility",
        "lmi",
        "heat_non_heat",
        "tariff_name",
        "version"
      )
    ) |>
    left_join(
      gas_supply_rates |>
        select(month, gas_utility, heat_non_heat, gas_supply_rate, year),
      by = c("month", "gas_utility", "heat_non_heat")
    ) |>
    # handle NAs and invalid numbers
    mutate(
      year = if_else(is.na(year), supply_year, year),
      delivery_rate = if_else(is.na(delivery_rate), 0, delivery_rate),
      gas_supply_rate = if_else(is.na(gas_supply_rate), 0, gas_supply_rate),
      customer_charge = if_else(is.na(customer_charge), 0, customer_charge),
      sales_tax_rate = if_else(is.na(sales_tax_rate), 0, sales_tax_rate)
    ) |>
    # calculate monthly bills by component
    mutate(
      customer_charge = customer_charge * (1 - discount_rate_gas),
      delivery_charge = gas_consumption *
        delivery_rate *
        (1 - discount_rate_gas),
      supply_charge = gas_consumption *
        gas_supply_rate *
        (1 - discount_rate_gas),
      total_pretax_bill = delivery_charge + supply_charge + customer_charge,
      sales_tax_charge = total_pretax_bill * sales_tax_rate,
      monthly_bill = total_pretax_bill + sales_tax_charge,
      monthly_bill_undiscounted = monthly_bill / (1 - discount_rate_gas)
    ) |>
    # handle NAs and invalid numbers
    mutate(
      delivery_charge = if_else(is.na(delivery_charge), 0, delivery_charge),
      supply_charge = if_else(is.na(supply_charge), 0, supply_charge),
      total_pretax_bill = if_else(
        is.na(total_pretax_bill),
        0,
        total_pretax_bill
      ),
      sales_tax_charge = if_else(is.na(sales_tax_charge), 0, sales_tax_charge),
      monthly_bill = if_else(is.na(monthly_bill), 0, monthly_bill),
      monthly_bill_undiscounted = if_else(
        is.na(monthly_bill_undiscounted),
        0,
        monthly_bill_undiscounted
      )
    ) |>
    select(
      bldg_id,
      upgrade,
      heat_non_heat,
      hvac,
      month,
      year,
      gas_utility,
      gas_consumption,
      version,
      tariff_name,
      delivery_rate,
      gas_supply_rate,
      customer_charge,
      delivery_charge,
      supply_charge,
      total_pretax_bill,
      sales_tax_charge,
      monthly_bill,
      monthly_bill_undiscounted
    ) |>
    arrange(bldg_id, month, year, version, tariff_name) |>
    collect() # Only collect at the very end

  return(final_result)
}

calc_monthly_changes <- function(monthly_bills, fuel_type) {
  # Set some columns names based on fuel_type
  # ------------
  # Electricity
  if (fuel_type == "electricity") {
    discount_rate <- "discount_rate_elec"
    utility <- "electric_utility"
    # ------------
    # Gas
  } else if (fuel_type == "natural_gas") {
    discount_rate <- "discount_rate_gas"
    utility <- "gas_utility"
    # ------------
    # Fuel Oil
  } else if (fuel_type == "fuel_oil") {
    discount_rate <- "discount_rate_fuel_oil"
    utility <- "fuel_oil_utility"
    # ------------
    # Propane
  } else if (fuel_type == "propane") {
    discount_rate <- "discount_rate_propane"
    utility <- "propane_utility"
  }
  # ------------

  monthly_changes <- monthly_bills |>
    # first get the baseline bill for comparison
    filter(upgrade %in% c(0, "00") & version == "baseline") |>
    select(bldg_id, year, month, !!sym(utility), monthly_bill) |>
    rename(baseline_bill = monthly_bill) |>
    # join the baseline bill column to the full dataset
    right_join(monthly_bills, by = c("bldg_id", "year", "month", utility)) |>
    mutate(monthly_change = monthly_bill - baseline_bill) |>
    select(
      bldg_id,
      upgrade,
      hvac,
      year,
      month,
      !!sym(utility),
      version,
      tariff_name,
      baseline_bill,
      electrified_bill = monthly_bill,
      monthly_change
    )

  return(monthly_changes)
}

# Annual Bills
calc_annual_bills_from_monthly <- function(
  monthly_bills,
  fuel_type,
  months = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
) {
  # First assign groups_cols based on fuel_type
  groups_cols <- if (fuel_type == "electricity") {
    c(
      "bldg_id",
      "upgrade",
      "hvac",
      "year",
      "electric_utility",
      "version",
      "tariff_name"
    )
  } else if (fuel_type == "natural_gas") {
    c(
      "bldg_id",
      "upgrade",
      "hvac",
      "year",
      "gas_utility",
      "version",
      "tariff_name"
    )
  } else if (fuel_type == "fuel_oil") {
    c(
      "bldg_id",
      "upgrade",
      "hvac",
      "year",
      "fuel_oil_utility",
      "version",
      "tariff_name"
    )
  } else if (fuel_type == "propane") {
    c(
      "bldg_id",
      "upgrade",
      "hvac",
      "year",
      "propane_utility",
      "version",
      "tariff_name"
    )
  }

  monthly_bills |>
    filter(month %in% months) |>
    group_by(across(all_of(groups_cols))) |>
    summarize(
      annual_bill = sum(monthly_bill[
        !is.na(monthly_bill) & is.finite(monthly_bill)
      ]),
      .groups = "drop"
    ) |>
    ungroup()
}

calc_annual_change_from_monthly <- function(
  monthly_changes,
  fuel_type,
  months = c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
) {
  # First assign groups_cols based on fuel_type
  groups_cols <- if (fuel_type == "electricity") {
    c(
      "bldg_id",
      "upgrade",
      "hvac",
      "year",
      "electric_utility",
      "version",
      "tariff_name"
    )
  } else if (fuel_type == "natural_gas") {
    c(
      "bldg_id",
      "upgrade",
      "hvac",
      "year",
      "gas_utility",
      "version",
      "tariff_name"
    )
  } else if (fuel_type == "fuel_oil") {
    c(
      "bldg_id",
      "upgrade",
      "hvac",
      "year",
      "fuel_oil_utility",
      "version",
      "tariff_name"
    )
  } else if (fuel_type == "propane") {
    c(
      "bldg_id",
      "upgrade",
      "hvac",
      "year",
      "propane_utility",
      "version",
      "tariff_name"
    )
  }

  monthly_changes |>
    filter(month %in% months) |>
    group_by(across(all_of(groups_cols))) |>
    summarize(
      annual_bill = sum(electrified_bill[
        !is.na(electrified_bill) & is.finite(electrified_bill)
      ]),
      annual_bill_baseline = sum(baseline_bill[
        !is.na(baseline_bill) & is.finite(baseline_bill)
      ]),
      annual_change = sum(monthly_change[
        !is.na(monthly_change) & is.finite(monthly_change)
      ]),
      .groups = "drop"
    ) |>
    ungroup()
}

calc_annual_bills_total <- function(
  annual_bills_elec,
  annual_bills_gas,
  annual_bills_fuel_oil,
  annual_bills_propane
) {
  annual_bills_total <- annual_bills_elec |>
    select(bldg_id, upgrade, year, annual_bill, version, tariff_name) |>
    rename(
      annual_bill_elec = annual_bill,
      version_elec = version,
      tariff_name_elec = tariff_name
    ) |>
    # gas bills
    left_join(
      annual_bills_gas |>
        select(bldg_id, upgrade, year, annual_bill, version, tariff_name) |>
        rename(
          annual_bill_gas = annual_bill,
          version_gas = version,
          tariff_name_gas = tariff_name
        ),
      by = c("bldg_id", "upgrade", "year")
    ) |>
    # fuel oil bills
    left_join(
      annual_bills_fuel_oil |>
        select(bldg_id, upgrade, year, annual_bill, version, tariff_name) |>
        rename(
          annual_bill_fuel_oil = annual_bill,
          version_fuel_oil = version,
          tariff_name_fuel_oil = tariff_name
        ),
      by = c("bldg_id", "upgrade", "year")
    ) |>
    # propane bills
    left_join(
      annual_bills_propane |>
        select(bldg_id, upgrade, year, annual_bill, version, tariff_name) |>
        rename(
          annual_bill_propane = annual_bill,
          version_propane = version,
          tariff_name_propane = tariff_name
        ),
      by = c("bldg_id", "upgrade", "year")
    ) |>
    # total bills
    mutate(
      annual_bill_total = annual_bill_elec +
        annual_bill_gas +
        annual_bill_fuel_oil +
        annual_bill_propane
    ) |>
    # add some metadata from housing_units
    left_join(
      housing_units |>
        select(
          bldg_id,
          upgrade,
          hvac,
          in.representative_income,
          baseline_heating_type,
          building_type_group,
          baseline_cooling_type,
          dollar_tier,
          smi_tier,
          occupants_group
        ),
      by = c("bldg_id", "upgrade")
    ) |>

    # Remove homes with no income
    filter(in.representative_income > 1000) |>
    filter(occupants_group != "Vacant") |>

    # Energy Burdens
    mutate(
      burden_elec = annual_bill_elec / in.representative_income,
      burden_gas = annual_bill_gas / in.representative_income,
      burden_fuel_oil = annual_bill_fuel_oil / in.representative_income,
      burden_propane = annual_bill_propane / in.representative_income,
      burden_total = annual_bill_total / in.representative_income
    )

  return(annual_bills_total)
}

calc_annual_changes_total <- function(
  annual_changes_elec,
  annual_changes_gas,
  annual_changes_fuel_oil,
  annual_changes_propane
) {
  annual_changes_total <- annual_changes_elec |>
    select(bldg_id, upgrade, year, annual_change, version, tariff_name) |>
    rename(
      annual_change_elec = annual_change,
      version_elec = version,
      tariff_name_elec = tariff_name
    ) |>
    # gas bills
    left_join(
      annual_changes_gas |>
        select(bldg_id, upgrade, year, annual_change, version, tariff_name) |>
        rename(
          annual_change_gas = annual_change,
          version_gas = version,
          tariff_name_gas = tariff_name
        ),
      by = c("bldg_id", "upgrade", "year")
    ) |>
    # fuel oil bills
    left_join(
      annual_changes_fuel_oil |>
        select(bldg_id, upgrade, year, annual_change, version, tariff_name) |>
        rename(
          annual_change_fuel_oil = annual_change,
          version_fuel_oil = version,
          tariff_name_fuel_oil = tariff_name
        ),
      by = c("bldg_id", "upgrade", "year")
    ) |>
    # propane bills
    left_join(
      annual_changes_propane |>
        select(bldg_id, upgrade, year, annual_change, version, tariff_name) |>
        rename(
          annual_change_propane = annual_change,
          version_propane = version,
          tariff_name_propane = tariff_name
        ),
      by = c("bldg_id", "upgrade", "year")
    ) |>
    # total bills
    mutate(
      annual_change_total = annual_change_elec +
        annual_change_gas +
        annual_change_fuel_oil +
        annual_change_propane
    ) |>
    # add some metadata from housing_units
    left_join(
      housing_units |>
        select(
          bldg_id,
          upgrade,
          hvac,
          baseline_heating_type,
          building_type_group,
          baseline_cooling_type,
          dollar_tier,
          smi_tier,
          occupants_group
        ),
      by = c("bldg_id", "upgrade")
    )

  return(annual_changes_total)
}

# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
# Bill Calculation Functions OLD
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
get_monthly_bills_elec_old <- function(path_rs_db, year) {
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  query <- sprintf(
    "
  WITH base_data AS (
    SELECT m.bldg_id, m.month, m.upgrade, m.hvac, m.shell, m.appliances,
           m.total_elec_kwh as elec_consumption,
           h.electric_utility, h.gas_utility, h.city, h.county, h.county_and_puma, h.building_type, h.occupants
    FROM ma_monthly m
    RIGHT JOIN (
      SELECT bldg_id, electric_utility, gas_utility, \"in.city\" as city, \"in.county\" as county, \"in.county_and_puma\" as county_and_puma, \"in.geometry_building_type_acs\" as building_type, \"in.occupants\" as occupants
      FROM housing_units
    ) h ON m.bldg_id = h.bldg_id
  ),
  with_delivery AS (
    SELECT b.*,
           d1.value as customer_charge,
           d2.value as delivery_rate,
           d3.value as sales_tax_rate,
           d1.version,
           d1.tariff_id
    FROM base_data b
    LEFT JOIN (
      SELECT month, electric_utility, value, version, tariff_id
      FROM delivery_tariffs_elec
      WHERE type = 'customer_charge'
    ) d1 ON b.month = d1.month AND b.electric_utility = d1.electric_utility
    LEFT JOIN (
      SELECT month, electric_utility, value, version, tariff_id
      FROM delivery_tariffs_elec
      WHERE type = 'delivery_rate'
    ) d2 ON b.month = d2.month
      AND b.electric_utility = d2.electric_utility
      AND d2.version = d1.version
      AND d2.tariff_id = d1.tariff_id
    LEFT JOIN (
      SELECT month, electric_utility, value, version, tariff_id
      FROM delivery_tariffs_elec
      WHERE type = 'sales_tax_rate'
    ) d3 ON b.month = d3.month
      AND b.electric_utility = d3.electric_utility
      AND d3.version = d1.version
      AND d3.tariff_id = d1.tariff_id
  ),
  final AS (
    SELECT w.*,
           s.supply_rate,
           s.year,
           w.elec_consumption * w.delivery_rate as delivery_charge,
           w.elec_consumption * s.supply_rate as supply_charge,
           (w.elec_consumption * w.delivery_rate + w.elec_consumption * s.supply_rate + w.customer_charge) as total_pretax_bill,
           (w.elec_consumption * w.delivery_rate + w.elec_consumption * s.supply_rate + w.customer_charge) * w.sales_tax_rate as sales_tax_charge,
           (w.elec_consumption * w.delivery_rate + w.elec_consumption * s.supply_rate + w.customer_charge) +
           ((w.elec_consumption * w.delivery_rate + w.elec_consumption * s.supply_rate + w.customer_charge) * w.sales_tax_rate) as monthly_bill
    FROM with_delivery w
    LEFT JOIN (
      SELECT month, electric_utility, supply_rate, year
      FROM supply_rates_elec
      WHERE year = %d
    ) s ON w.month = s.month AND w.electric_utility = s.electric_utility
  )
  SELECT
    bldg_id,
    upgrade,
    hvac,
    shell,
    appliances,
    month,
    year,
    electric_utility,
    gas_utility,
    city,
    county,
    county_and_puma,
    building_type,
    occupants,
    elec_consumption,
    version,
    tariff_id,
    delivery_rate,
    supply_rate,
    customer_charge,
    delivery_charge,
    supply_charge,
    total_pretax_bill,
    sales_tax_charge,
    monthly_bill
  FROM final
  ORDER BY bldg_id, month, year, hvac, shell, appliances, electric_utility, version, tariff_id
  ",
    year
  )

  monthly_bills_elec <- DBI::dbGetQuery(con, query)
  return(monthly_bills_elec)
}

get_monthly_bills_gas <- function(path_rs_db, year) {
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  query <- sprintf(
    "
  WITH base_data AS (
    SELECT m.bldg_id, m.month, m.upgrade, m.hvac, m.shell, m.appliances,
           m.total_gas_kwh as gas_consumption,
           h.electric_utility, h.gas_utility, h.city, h.county, h.county_and_puma, h.building_type, h.occupants,
           CASE
             WHEN m.hvac = 'current' THEN 'R-3'
             WHEN m.hvac IN ('low_hp','mid_hp','high_hp') THEN 'R-1'
           END as customer_class
    FROM ma_monthly m
   RIGHT JOIN (
      SELECT bldg_id, electric_utility, gas_utility, \"in.city\" as city, \"in.county\" as county, \"in.county_and_puma\" as county_and_puma, \"in.geometry_building_type_acs\" as building_type, \"in.occupants\" as occupants
      FROM housing_units
    ) h ON m.bldg_id = h.bldg_id
  ),
  with_delivery AS (
    SELECT b.*,
           d1.value as customer_charge,
           d2.value as delivery_rate,
           d3.value as sales_tax_rate,
           d1.version,
           d1.tariff_id,
           d1.customer_class
    FROM base_data b
    LEFT JOIN (
      SELECT month, gas_utility, value, version, tariff_id, customer_class
      FROM delivery_tariffs_gas
      WHERE type = 'customer_charge'
    ) d1 ON b.month = d1.month AND b.gas_utility = d1.gas_utility AND b.customer_class = d1.customer_class
    LEFT JOIN (
      SELECT month, gas_utility, value, version, tariff_id, customer_class
      FROM delivery_tariffs_gas
      WHERE type = 'delivery_rate'
    ) d2 ON b.month = d2.month
      AND b.gas_utility = d2.gas_utility
      AND b.customer_class = d2.customer_class
      AND d2.version = d1.version
      AND d2.tariff_id = d1.tariff_id
    LEFT JOIN (
      SELECT month, gas_utility, value, version, tariff_id, customer_class
      FROM delivery_tariffs_gas
      WHERE type = 'sales_tax_rate'
    ) d3 ON b.month = d3.month
      AND b.gas_utility = d3.gas_utility
      AND b.customer_class = d3.customer_class
      AND d3.version = d1.version
      AND d3.tariff_id = d1.tariff_id
  ),
  final AS (
    SELECT w.*,
           s.supply_rate,
           s.year,
           s.rate_class,
           w.gas_consumption * w.delivery_rate as delivery_charge,
           w.gas_consumption * s.supply_rate as supply_charge,
           (w.gas_consumption * w.delivery_rate + w.gas_consumption * s.supply_rate + w.customer_charge) as total_pretax_bill,
           (w.gas_consumption * w.delivery_rate + w.gas_consumption * s.supply_rate + w.customer_charge) * w.sales_tax_rate as sales_tax_charge,
           (w.gas_consumption * w.delivery_rate + w.gas_consumption * s.supply_rate + w.customer_charge) +
           ((w.gas_consumption * w.delivery_rate + w.gas_consumption * s.supply_rate + w.customer_charge) * w.sales_tax_rate) as monthly_bill
    FROM with_delivery w
    LEFT JOIN (
      SELECT month, gas_utility, supply_rate, year, rate_class
      FROM supply_rates_gas
      WHERE year = %d
    ) s ON w.month = s.month AND w.gas_utility = s.gas_utility AND s.rate_class = w.customer_class
  )
  SELECT
    bldg_id,
    upgrade,
    hvac,
    shell,
    appliances,
    month,
    year,
    electric_utility,
    gas_utility,
    city,
    county,
    county_and_puma,
    building_type,
    occupants,
    gas_consumption,
    version,
    tariff_id,
    delivery_rate,
    supply_rate,
    customer_charge,
    delivery_charge,
    supply_charge,
    total_pretax_bill,
    sales_tax_charge,
    monthly_bill
  FROM final
  ORDER BY bldg_id, month, year, hvac, shell, appliances, gas_utility, version, tariff_id
  ",
    year
  )

  # Print the query before executing it
  #cat("About to execute query:\n", query, "\n")

  monthly_bills_gas <- DBI::dbGetQuery(con, query)

  # Force all rows to have the year passed as argument
  monthly_bills_gas$year <- year

  return(monthly_bills_gas)
}


get_monthly_bills_fueloil <- function(path_rs_db, year) {
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  query <- sprintf(
    "
  WITH base_data AS (
    SELECT m.bldg_id, m.month, m.upgrade, m.hvac, m.shell, m.appliances,
           m.total_fuel_oil_kwh as fueloil_consumption,
           h.electric_utility, h.gas_utility, h.city, h.county, h.county_and_puma, h.building_type, h.occupants
    FROM ma_monthly m
    LEFT JOIN (
      SELECT bldg_id, electric_utility, gas_utility, \"in.city\" as city, \"in.county\" as county, \"in.county_and_puma\" as county_and_puma, \"in.geometry_building_type_acs\" as building_type, \"in.occupants\" as occupants
      FROM housing_units
    ) h ON m.bldg_id = h.bldg_id
  ),
  with_delivery AS (
    SELECT b.*,
           d1.value as customer_charge,
           d2.value as delivery_rate,
           d3.value as sales_tax_rate
    FROM base_data b
    LEFT JOIN (
      SELECT month, value
      FROM delivery_tariffs_fueloil
      WHERE type = 'customer_charge'
    ) d1 ON b.month = d1.month
    LEFT JOIN (
      SELECT month, value
      FROM delivery_tariffs_fueloil
      WHERE type = 'delivery_rate'
    ) d2 ON b.month = d2.month
    LEFT JOIN (
      SELECT month, value
      FROM delivery_tariffs_fueloil
      WHERE type = 'sales_tax_rate'
    ) d3 ON b.month = d3.month
  ),
  final AS (
    SELECT w.*,
           fuel_oil_dollars_per_kwh as supply_rate,
           %d as year,
           w.fueloil_consumption * supply_rate as supply_charge,
           w.fueloil_consumption * w.delivery_rate as delivery_charge,
           (w.fueloil_consumption * w.delivery_rate + w.fueloil_consumption * supply_rate + w.customer_charge) as total_pretax_bill,
           (w.fueloil_consumption * w.delivery_rate + w.fueloil_consumption * supply_rate + w.customer_charge) * w.sales_tax_rate as sales_tax_charge,
           (w.fueloil_consumption * w.delivery_rate + w.fueloil_consumption * supply_rate + w.customer_charge) +
           ((w.fueloil_consumption * w.delivery_rate + w.fueloil_consumption * supply_rate + w.customer_charge) * w.sales_tax_rate) as monthly_bill
    FROM with_delivery w
    LEFT JOIN (
      SELECT month, fuel_oil_dollars_per_kwh
      FROM supply_rates_fueloil
      WHERE year = %d
    ) s ON w.month = s.month
  )
  SELECT
    bldg_id,
    upgrade,
    hvac,
    shell,
    appliances,
    month,
    year,
    electric_utility,
    gas_utility,
    city,
    county,
    county_and_puma,
    building_type,
    occupants,
    fueloil_consumption,
    delivery_rate,
    supply_rate,
    customer_charge,
    delivery_charge,
    supply_charge,
    total_pretax_bill,
    sales_tax_charge,
    monthly_bill
  FROM final
  ORDER BY bldg_id, month
  ",
    year,
    year
  )

  # Print the query before executing it
  #cat("About to execute query:\n", query, "\n")

  monthly_bills_fueloil <- DBI::dbGetQuery(con, query)

  return(monthly_bills_fueloil)
}

get_monthly_bills_propane <- function(path_rs_db, year) {
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)
  on.exit(DBI::dbDisconnect(con), add = TRUE)

  query <- sprintf(
    "
  WITH base_data AS (
    SELECT m.bldg_id, m.month, m.upgrade, m.hvac, m.shell, m.appliances,
           m.total_propane_kwh as propane_consumption,
           h.electric_utility, h.gas_utility, h.city, h.county, h.county_and_puma, h.building_type, h.occupants
    FROM ma_monthly m
    LEFT JOIN (
      SELECT bldg_id, electric_utility, gas_utility, \"in.city\" as city, \"in.county\" as county, \"in.county_and_puma\" as county_and_puma, \"in.geometry_building_type_acs\" as building_type, \"in.occupants\" as occupants
      FROM housing_units
    ) h ON m.bldg_id = h.bldg_id
  ),
  with_delivery AS (
    SELECT b.*,
           d1.value as customer_charge,
           d2.value as delivery_rate,
           d3.value as sales_tax_rate
    FROM base_data b
    LEFT JOIN (
      SELECT month, value
      FROM delivery_tariffs_propane
      WHERE type = 'customer_charge'
    ) d1 ON b.month = d1.month
    LEFT JOIN (
      SELECT month, value
      FROM delivery_tariffs_propane
      WHERE type = 'delivery_rate'
    ) d2 ON b.month = d2.month
    LEFT JOIN (
      SELECT month, value
      FROM delivery_tariffs_propane
      WHERE type = 'sales_tax_rate'
    ) d3 ON b.month = d3.month
  ),
  final AS (
    SELECT w.*,
           propane_dollars_per_kwh as supply_rate,
           %d as year,
           w.propane_consumption * supply_rate as supply_charge,
           w.propane_consumption * w.delivery_rate as delivery_charge,
           (w.propane_consumption * w.delivery_rate + w.propane_consumption * supply_rate + w.customer_charge) as total_pretax_bill,
           (w.propane_consumption * w.delivery_rate + w.propane_consumption * supply_rate + w.customer_charge) * w.sales_tax_rate as sales_tax_charge,
           (w.propane_consumption * w.delivery_rate + w.propane_consumption * supply_rate + w.customer_charge) +
           ((w.propane_consumption * w.delivery_rate + w.propane_consumption * supply_rate + w.customer_charge) * w.sales_tax_rate) as monthly_bill
    FROM with_delivery w
    LEFT JOIN (
      SELECT month, propane_dollars_per_kwh
      FROM supply_rates_propane
      WHERE year = %d
    ) s ON w.month = s.month
  )
  SELECT
    bldg_id,
    upgrade,
    hvac,
    shell,
    appliances,
    month,
    year,
    electric_utility,
    gas_utility,
    city,
    county,
    county_and_puma,
    building_type,
    occupants,
    propane_consumption,
    delivery_rate,
    supply_rate,
    customer_charge,
    delivery_charge,
    supply_charge,
    total_pretax_bill,
    sales_tax_charge,
    monthly_bill
  FROM final
  ORDER BY bldg_id, month, year
  ",
    year,
    year
  )

  # Print the query before executing it
  #cat("About to execute query:\n", query, "\n")

  monthly_bills_propane <- DBI::dbGetQuery(con, query)

  return(monthly_bills_propane)
}


## Apply low-income discounts
apply_low_income_discounts <- function(
  path_rs_db,
  url_sheet_low_income_thresholds,
  sheet_tariff_name,
  monthly_bills,
  fuel_type
) {
  # Get low income thresholds from Google Sheet
  googlesheets4::gs4_deauth()
  low_income_thresholds <- googlesheets4::read_sheet(
    url_sheet_low_income_thresholds,
    sheet = sheet_tariff_name
  ) |>
    select(-source, -note)

  # Get housing units data
  con <- DBI::dbConnect(duckdb::duckdb(), path_rs_db)
  housing_units <- DBI::dbReadTable(con, "housing_units")
  DBI::dbDisconnect(con)

  # Calculate discount rates for each building
  result <- housing_units |>
    # Cross join with low_income_thresholds to get all possible combinations
    cross_join(low_income_thresholds) |>
    # Filter to keep only rows where conditions are met
    filter(
      if (fuel_type == "electricity") {
        electric_utility.x == electric_utility.y
      } else if (fuel_type == "gas") {
        gas_utility.x == gas_utility.y
      } else if (fuel_type == "fueloil") {
        TRUE
      } else if (fuel_type == "propane") {
        TRUE
      },
      in.occupants == occupants_min,
      assigned_income >= income_threshold_lower,
      assigned_income <= income_threshold_upper
    ) |>
    # Group by housing unit to handle multiple matching thresholds
    group_by(bldg_id) |>
    # Take highest matching discount rate or 0 if none match
    summarize(
      discount_rate = if (n() > 0) max(discount_rate) else 0,
      .groups = "drop"
    ) |>
    # Right join to housing_units to keep all buildings
    right_join(housing_units, by = "bldg_id") |>
    # Fill NA discounts with 0
    mutate(discount_rate = coalesce(discount_rate, 0)) |>
    select(bldg_id, in.occupants, assigned_income, discount_rate)

  # Apply discounts to annual bills
  monthly_bills_discounted <- monthly_bills |>
    left_join(result, by = c("bldg_id", "occupants" = "in.occupants")) |>
    mutate(monthly_bill_raw = monthly_bill) |>
    mutate(discount = -(monthly_bill_raw * discount_rate)) |>
    mutate(monthly_bill = monthly_bill_raw + discount)

  return(monthly_bills_discounted)
}


## Annual Bills
calc_annual_bills_from_monthly_old <- function(monthly_bills, fuel_type) {
  # First assign groups_cols based on fuel_type
  groups_cols <- if (fuel_type == "electricity") {
    c(
      "bldg_id",
      "year",
      "hvac",
      "shell",
      "appliances",
      "county",
      "county_and_puma",
      "building_type",
      "occupants",
      "electric_utility",
      "version",
      "tariff_id"
    )
  } else if (fuel_type == "gas") {
    c(
      "bldg_id",
      "year",
      "hvac",
      "shell",
      "appliances",
      "county",
      "building_type",
      "occupants",
      "gas_utility",
      "version",
      "tariff_id"
    )
  } else if (fuel_type == "fueloil") {
    c(
      "bldg_id",
      "year",
      "hvac",
      "shell",
      "appliances",
      "county",
      "building_type",
      "occupants"
    )
  } else if (fuel_type == "propane") {
    c(
      "bldg_id",
      "year",
      "hvac",
      "shell",
      "appliances",
      "county",
      "building_type",
      "occupants"
    )
  }

  # monthly_bills |>
  #   group_by(across(all_of(groups_cols))) |>
  #   summarize(
  #     annual_bill = sum(monthly_bill),
  #     annual_bill_raw = sum(monthly_bill_raw)
  #   )|>
  #   ungroup()

  monthly_bills |>
    group_by(across(all_of(groups_cols))) |>
    summarize(
      annual_bill = sum(monthly_bill[
        !is.na(monthly_bill) & is.finite(monthly_bill)
      ]),
      annual_bill_raw = sum(monthly_bill_raw[
        !is.na(monthly_bill_raw) & is.finite(monthly_bill_raw)
      ])
    ) |>
    ungroup()
}

calculate_annual_elec_bills_from_monthly <- function(monthly_bills) {
  monthly_bills |>
    group_by(bldg_id, year, upgrade, electric_utility, version, tariff_id) |>
    summarize(
      annual_bill = sum(monthly_bill[
        !is.na(monthly_bill) & is.finite(monthly_bill)
      ])
    ) |>
    filter(annual_bill > 0) |>
    ungroup()
}


calculate_annual_gas_bills_from_monthly <- function(monthly_bills) {
  monthly_bills |>
    group_by(bldg_id, year, upgrade, gas_utility, version, tariff_id) |>
    summarize(
      annual_bill = sum(total_bill[!is.na(total_bill) & is.finite(total_bill)])
    ) |>
    filter(annual_bill > 0) |>
    ungroup()
}


########################################################
# Plotting Functions
########################################################

## Supply Plots ----------------------
### NY Supply Rates
plot_supply_rates <- function(
  supply_rates,
  y,
  start_year,
  end_year = 2024,
  highlight_years = c("2020", "2024")
) {
  # Filter data to desired year range
  supply_rates_filtered <- supply_rates |>
    filter(year >= start_year & year <= end_year, tariff_name == y)

  p <- ggplot(supply_rates_filtered, aes(x = month, y = rate, group = year)) +

    # Add grey lines for years not in highlight years
    geom_line(
      data = filter(
        supply_rates_filtered,
        !as.character(year) %in% highlight_years
      ),
      color = "grey80"
    ) +
    geom_point(
      data = filter(
        supply_rates_filtered,
        !as.character(year) %in% highlight_years
      ),
      color = "grey80",
      size = 0.1
    ) +

    # Add colored lines for highlight years
    geom_line(
      data = filter(
        supply_rates_filtered,
        as.character(year) %in% highlight_years
      ),
      aes(color = as.factor(year))
    ) +
    geom_point(
      data = filter(
        supply_rates_filtered,
        as.character(year) %in% highlight_years
      ),
      aes(color = as.factor(year)),
      size = 0.25
    ) +

    scale_x_continuous(breaks = 1:12, labels = month.abb) +
    scale_y_continuous(labels = scales::label_dollar()) +
    scale_color_viridis_d() +
    labs(
      title = paste("ConEd", y, "Supply Rates by Year"),
      x = "",
      y = "Wholesale Supply Cost ($/kWh)",
      color = "Year"
    ) +
    theme(legend.position = "right")

  return(p)
}

plot_supply_rates_ribbon <- function(
  supply_rates,
  first_year,
  last_year,
  ribbon_top,
  ribbon_bottom,
  middle
) {
  # Create color palette
  year_range <- last_year - first_year + 1
  year_colors <- viridis::viridis(
    year_range,
    begin = 0,
    end = 1,
    direction = -1
  ) |>
    settariff_names(first_year:last_year)

  # Filter and reshape data
  plot_data <- supply_rates |>
    filter(year >= first_year & year <= last_year) |>
    mutate(date = make_date(year, month, 1)) |>
    pivot_wider(
      tariff_names_from = tariff_name,
      values_from = rate
    )

  # Create plot
  p <- ggplot(plot_data, aes(x = date, group = year)) +
    # Add ribbon between specified rates
    geom_ribbon(
      aes(
        ymin = .data[[ribbon_bottom]],
        ymax = .data[[ribbon_top]],
        fill = as.factor(year)
      ),
      alpha = 0.3
    ) +
    # Add middle line
    geom_line(
      aes(
        y = .data[[middle]],
        color = as.factor(year)
      ),
      linewidth = 1
    ) +
    # Formatting
    scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
    scale_y_continuous(labels = scales::label_dollar()) +
    scale_fill_manual(values = year_colors) +
    scale_color_manual(values = year_colors) +
    labs(
      title = paste("Supply Rates:", first_year, "-", last_year),
      subtitle = "Showing peak/off-peak spread and all-hours rate",
      x = "",
      y = "Supply Rate ($/kWh)",
      fill = "Year",
      color = "Year"
    ) +
    theme(legend.position = "None")

  return(p)
}

### MA Gas Supply Rates
plot_gas_supply_rates <- function(data, save_path, width = 6, height = 3) {
  # Get year range for shading
  start_year <- as.numeric(format(min(data$effective_date), "%Y"))
  end_year <- as.numeric(format(max(data$effective_date), "%Y"))

  # Create data frame for summer period rectangles
  summer_periods <- data.frame(
    xmin = as.Date(sprintf("%d-05-01", start_year:end_year)),
    xmax = as.Date(sprintf("%d-10-31", start_year:end_year))
  )

  # Create the plot
  p <- ggplot(data, aes(x = effective_date, y = gaf, color = gas_utility)) +
    # Add summer period shading
    theme(
      panel.grid.minor.x = element_blank(), # Remove minor gridlines
      panel.grid.major.y = element_line(linewidth = 0.5, color = "grey80"), # Explicit y gridlines
      panel.grid.minor.y = element_line(linewidth = 0.25, color = "grey90") # Explicit minor y gridlines
    ) +
    # geom_rect(data = summer_periods,
    #       aes(xmin = xmin, xmax = xmax,
    #           ymin = -Inf, ymax = Inf),
    #       fill = "blue",
    #       color = NA,
    #       inherit.aes = FALSE) +
    geom_point(size = 0.25) +
    geom_line() +
    labs(
      title = "Gas Supply (GAF) Rates by Utility",
      x = "Effective Date",
      y = "GAF Rate ($/Therm)",
      color = "gas_utility"
    ) +
    scale_color_viridis_d(option = "viridis") +
    scale_x_date(
      date_breaks = "1 year",
      date_labels = "%Y",
      limits = c(
        as.Date(paste0(min(format(data$effective_date, "%Y")), "-01-01")),
        max(data$effective_date)
      )
    ) +
    scale_y_continuous(limits = c(0, NA))

  # Save the plot
  ggsave(save_path, p, width = width, height = height, bg = "white")

  # Display the plot
  print(p)

  # Return the plot object invisibly
  invisible(p)
}

### Plot supply rates for 12 months
plot_supply_rates_12_months <- function(
  supply_rates_monthly_long,
  electric_utility,
  highlight_years = c("2025")
) {
  # Spaghetti plot of supply rates for 12 months
  # X: Month
  # Y: Supply Rate
  # Color: Year
  # Add grey lines for years not in highlight years
  # Add colored lines for highlight years

  # Filter data for the selected utility
  filtered_data <- supply_rates_monthly_long |>
    filter(electric_utility == !!electric_utility)

  ggplot(filtered_data, aes(x = month, y = supply_rate, group = year)) +
    # Add grey lines for years not in highlight years
    geom_line(
      data = filter(filtered_data, !year %in% highlight_years),
      color = "grey80"
    ) +
    geom_point(
      data = filter(filtered_data, !year %in% highlight_years),
      color = "grey80",
      size = 0.1
    ) +

    # Add colored lines for highlight years
    geom_line(
      data = filter(filtered_data, year %in% highlight_years),
      aes(color = factor(year))
    ) +
    geom_point(
      data = filter(filtered_data, year %in% highlight_years),
      aes(color = factor(year)),
      size = 0.25
    ) +

    scale_x_continuous(breaks = 1:12, labels = month.abb) +
    scale_y_continuous(labels = label_dollar()) +
    scale_color_viridis_d() +
    labs(
      title = paste(
        "Massachusetts Basic Service Rates:",
        gsub("_", " ", electric_utility)
      ),
      x = "",
      y = "Wholesale Supply Cost ($/kWh)"
    ) +
    theme(legend.position = "right")
}

## Annual Bill Plots --------------------
