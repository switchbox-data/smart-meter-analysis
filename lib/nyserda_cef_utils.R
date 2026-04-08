library(tidyverse)
library(readxl)
library(janitor)


read_nyserda_cef_data <- function(prog_path, part_path) {
  programs <- read_csv(prog_path, col_types = cols(Year = col_integer())) |>
    clean_names()
  participants <- read_csv(part_path, col_types = cols(Year = col_integer())) |>
    clean_names()

  cols <- c(
    "program_administrator",
    "fuel_type_funding_source",
    "portfolio",
    "primary_end_use_sector",
    "program_name",
    "nys_clean_heat",
    "new_efficiency_new_york",
    "lmi_market_rate",
    "active_inactive",
    "year",
    "reporting_period"
  )

  programs_by_quarter <- programs |>
    group_by(across(all_of(cols))) |>
    summarize(
      expenditures = sum(total_program_dollars_expenditures_this_quarter),
      co2e_reductions_annual = sum(
        direct_annual_co2e_emission_reductions_metric_tons_acquired_this_quarter
      ),
      co2e_reductions_gross_lifetime = sum(
        direct_gross_lifetime_co2e_emission_reductions_metric_tons_acquired_this_quarter
      )
    ) |>
    ungroup()

  participants_by_quarter <- participants |>
    group_by(across(all_of(cols))) |>
    summarize(
      participants = sum(participants_acquired_this_quarter)
    ) |>
    ungroup()

  joined <- programs_by_quarter |>
    left_join(participants_by_quarter, by = cols)

  return(joined)
}
