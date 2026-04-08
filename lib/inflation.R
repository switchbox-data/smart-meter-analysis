library(fredr)
library(tidyverse)

get_inflation_index <- function(series, start, end, api_key, freq = "q") {
  fredr_set_key(api_key)
  fredr(
    series_id = series,
    observation_start = as_date(start),
    observation_end = as_date(end),
    frequency = freq
  ) |>
    filter(!is.na(value)) |>
    filter(month(date) == month(max(date))) |> # filter to the most recent reported month
    mutate(
      pct_change = last(value) / value,
      year = year(date)
    ) |>
    select(year, pct_change)
}

get_inflation_factor <- function(inflation_index, input_year, target_year) {
  # scaling factor to convert input_year dollars to target_year dollars

  if (max(inflation_index) != target_year) {
    stop("inflation_index is not referenced to target year.")
  }
  inflation_index |>
    filter(year == input_year) |>
    pull(pct_change)
}
