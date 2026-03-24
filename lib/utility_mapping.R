library(yaml)
library(tidyverse)
library(sf)
library(tigris)
library(arrow)

########################################################
# Making or updating utility crosswalks
########################################################
make_empty_utility_crosswalk <- function(path_to_rs2024_metadata) {
  #' Make an empty utility crosswalk CSV file
  #'
  #' @param path_to_rs2024_metadata A path to the ResStock metadata parquet file
  #' @return A CSV file containing a list of all Restock 2024.2 bldg_ids, their states, and their heating fuels
  #' @examples
  #' make_empty_utility_crosswalk("/workspaces/reports2/data/resstock/2024_resstock_amy2018_release_2/res_2024_tmy3_2/metadata/RI/up00/metadata.parquet")

  use_these_columns <- c(
    "bldg_id",
    'in.state',
    'in.heating_fuel',
    'out.natural_gas.total.energy_consumption'
  )

  # Read metadata parquet file with selected columns
  bldg_utility_mapping <- arrow::read_parquet(
    file.path(path_to_rs2024_metadata, "metadata.parquet"),
    col_select = use_these_columns
  )

  # Add empty utility columns
  bldg_utility_mapping <- bldg_utility_mapping |>
    mutate(
      electric_utility = NA_character_,
      gas_utility = NA_character_
    )
  bldg_utility_mapping |> head() |> print()

  # Write to feather
  write_feather(
    bldg_utility_mapping,
    file.path(path_to_rs2024_metadata, "rs2024_bldg_utility_crosswalk.feather")
  )

  # Write to csv
  write_csv(
    bldg_utility_mapping,
    file.path(path_to_rs2024_metadata, "rs2024_bldg_utility_crosswalk.csv")
  )
}

########################################################
# Forced Utility Mapping
########################################################
forced_utility_crosswalk_ri <- function(path_to_rs2024_metadata) {
  use_these_columns <- c(
    "bldg_id",
    'in.state',
    'in.heating_fuel',
    'out.natural_gas.total.energy_consumption'
  )

  # Read metadata parquet file with selected columns
  bldg_utility_mapping <- read_parquet(
    file.path(path_to_rs2024_metadata, "metadata.parquet"),
    col_select = use_these_columns
  )

  # Add empty utility columns
  bldg_utility_mapping <- bldg_utility_mapping |>
    mutate(
      electric_utility = NA_character_,
      gas_utility = NA_character_
    )

  bldg_utility_mapping <- bldg_utility_mapping |>
    mutate(
      electric_utility = case_when(
        `in.state` == "RI" ~ "rhode_island_energy",
        TRUE ~ electric_utility
      ),
      gas_utility = case_when(
        `in.state` == "RI" &
          `out.natural_gas.total.energy_consumption` >
            10 ~ "rhode_island_energy",
        TRUE ~ gas_utility
      )
    )

  # Write to CSV
  write_feather(
    bldg_utility_mapping,
    file.path(path_to_rs2024_metadata, "rs2024_bldg_utility_crosswalk.feather")
  )

  # Write to CSV
  write_csv(
    bldg_utility_mapping,
    file.path(path_to_rs2024_metadata, "rs2024_bldg_utility_crosswalk.csv")
  )
}

########################################################
# GIS Utility Mapping
########################################################
state_configs <- list(
  "NY" = list(
    state_fips = "36",
    state_crs = 2260, # New York state plane (meters)
    hh_utilities_path = "/workspaces/reports2/data/resstock/utility_lookups/NY_hh_utilities.csv",
    resstock_path = "/workspaces/reports2/data/resstock/2022_resstock_amy2018_release_1.1/20230922.db",
    electric_poly_path = "/workspaces/reports2/data/buildings2/Utilities/NYS_Electric_Utility_Service_Territories.csv",
    gas_poly_path = "/workspaces/reports2/data/buildings2/Utilities/NYS_Gas_Utility_Service_Territories.csv",
    utility_name_map = tribble(
      ~state_name                       , ~std_name      ,
      "Bath Electric Gas and Water"     , "bath"         ,
      "Central Hudson Gas and Electric" , "cenhud"       ,
      "Chautauqua Utilities, Inc."      , "chautauqua"   ,
      "Consolidated Edison"             , "coned"        ,
      "Corning Natural Gas"             , "corning"      ,
      "Fillmore Gas Company"            , "fillmore"     ,
      "National Grid - NYC"             , "kedny"        ,
      "National Grid - Long Island"     , "kedli"        ,
      "National Grid"                   , "nimo"         ,
      "None"                            , "none"         ,
      "National Fuel Gas Distribution"  , "nationalfuel" ,
      "NYS Electric and Gas"            , "nyseg"        ,
      "Orange and Rockland Utilities"   , "or"           ,
      "Long Island Power Authority"     , "pseg-li"      ,
      "Reserve Gas Company"             , "reserve"      ,
      "Rochester Gas and Electric"      , "rge"          ,
      "St. Lawrence Gas"                , "stlawrence"   ,
      "Valley Energy"                   , "valley"       ,
      "Woodhull Municipal Gas Company"  , "woodhull"
    )
  ),
  "MA" = list(
    state_fips = "25",
    state_crs = 26986, # Massachusetts state plane (meters)
    hh_utilities_path = "/workspaces/reports2/data/resstock/utility_lookups/MA_hh_utilities.csv",
    resstock_path = "/workspaces/reports2/data/resstock/2022_resstock_amy2018_release_1.1/rs_20250326.db",
    electric_poly_path = "/workspaces/reports2/data/datamagov/MA_utility_territory_shapefiles_20250326/TOWNS_POLY_V_ELEC.shp",
    gas_poly_path = "/workspaces/reports2/data/datamagov/MA_utility_territory_shapefiles_20250326/TOWNS_POLY_V_GAS.shp",
    utility_name_map = tribble(
      ~state_name                                      , ~std_name      ,
      "The Berkshire Gas Company"                      , "berkshire"    ,
      "Eversource Energy"                              , "eversource"   ,
      "NSTAR Electric d/b/a Eversource Energy"         , "eversource"   ,
      "Liberty Utilities"                              , "liberty"      ,
      "Municipal"                                      , "municipal"    ,
      "National Grid"                                  , "nationalgrid" ,
      "Massachusetts Electric d/b/a National Grid"     , "nationalgrid" ,
      "Nantucket Electric Company d/b/a National Grid" , "nationalgrid" ,
      "No Natural Gas Service"                         , "none"         ,
      "Unitil"                                         , "unitil"       ,
      "UNITIL"                                         , "unitil"
    )
  )
)


# puma_centroids_default <- "/workspaces/reports2/data/buildings2/2010_Gaz_PUMAs_national.tsv" # Census PUMA centroids
# tract_lookup_default <- "/workspaces/reports2/data/resstock/spatial_tract_lookup_table.csv"

get_bldg_by_utility <- function(
  state_code,
  utility_electric = NULL,
  utility_gas = NULL,
  config = state_configs
) {
  #' Get buildings by utility service area
  #'
  #' @description
  #' Returns buildings that are served by the specified utilities in the given state.
  #' If only electric utility is specified, returns all buildings served by those electric utility.
  #' If only gas utility is specified, returns all buildings served by those gas utility.
  #' If both are specified, returns buildings served by both utilities.
  #' If neither is specified, returns all buildings in the state and their associated utilities.
  #'
  #' @param state (str) State code. Must be one of:
  #'   - "NY" (New York)
  #'   - "MA" (Massachusetts) #not ready yet
  #'
  #' @param utility_electric (list) Electric utility identifier. For NY, must be any of:
  #'   - "nimo" (National Grid)
  #'   - "cenhud" (Central Hudson Gas and Electric)
  #'   - "coned" (Consolidated Edison)
  #'   - "rge" (Rochester Gas and Electric)
  #'   - "nyseg" (NYS Electric and Gas)
  #'   - "pseg-li" (Long Island Power Authority)
  #'   - "or" (Orange and Rockland Utilities)
  #'
  #' @param utility_gas (list) Gas utility identifier. For NY, must be any of:
  #'   - "kedny" (National Grid - NYC)
  #'   - "kedli" (National Grid - Long Island)
  #'   - "nimo" (National Grid)
  #'   - "coned" (Consolidated Edison)
  #'   - "nationalfuel" (National Fuel Gas Distribution)
  #'   - "rge" (Rochester Gas and Electric)
  #'   - "nyseg" (NYS Electric and Gas)
  #'
  #' @return A dataframe containing:
  #'   - bldg_id: ResStock building identifier
  #'   - std_name.electric: Standardized electric utility name
  #'   - std_name.gas: Standardized gas utility name
  #'
  #' @examples
  #' # Get all buildings served by National Grid electric
  #' get_bldg_by_utility("NY", utility_electric = "nimo")
  #'
  #' # Get buildings served by both ConEd electric and gas
  #' get_bldg_by_utility("NY", utility_electric = "coned", utility_gas = "coned")
  #'
  #' # Get all buildings in NY with their utility assignments
  #' get_bldg_by_utility("NY")

  state_config <- config[[state_code]]
  if (is.null(state_config)) {
    stop(sprintf("No configuration available for state: %s", state_code))
  }

  if (file.exists(state_config$hh_utilities_path)) {
    print("Loading existing hh_utilities file")
    hh_utilities <- read_csv(
      state_config$hh_utilities_path,
      show_col_types = FALSE
    )
  } else {
    print("Creating new hh_utilities file")
    hh_utilities <- create_hh_utilities(
      state_code = state_code,
      config = config
    )
  }

  hh_utilities |>
    filter(
      (is.null(utility_electric) | electric_utility %in% utility_electric) &
        (is.null(utility_gas) | gas_utility %in% utility_gas)
    ) |>
    select(bldg_id, electric_utility, gas_utility)
}

create_hh_utilities <- function(
  state_code,
  config = state_configs,
  puma_year = 2019,
  save_file = TRUE,
  db_path = NULL
) {
  #' Create a dataframe of households with their associated utilities
  #'
  #' @description
  #' Returns a dataframe of households with their associated utilities.
  #'
  #' @param state_code (str) State code. Must be one of:
  #'   - "NY" (New York)
  #'   - "MA" (Massachusetts) #not ready yet
  #'
  #' @param state_config (list) State configuration.
  #'
  #' @param save_file (bool) Whether to save the file to the state_config$hh_utilities_path.
  #'
  #' @return A dataframe of households with their associated utilities.

  state_config <- config[[state_code]]
  if (is.null(state_config)) {
    stop(sprintf("No configuration available for state: %s", state))
  }

  utility_name_map <- state_config$utility_name_map

  if (is.null(db_path)) {
    db_path <- state_config$resstock_path
  }

  # load PUMAS
  pumas <- pumas(
    state = state_code,
    year = puma_year,
    cb = TRUE # Use cartographic boundaries (simplified)
  )

  if (state_code == "MA") {
    electric_utility_polygons <- merge_ma_electric_polygons(
      state_config$electric_poly_path
    )

    gas_utility_polygons <- merge_ma_gas_polygons(state_config$gas_poly_path)
  } else {
    electric_utility_polygons <- read_csv(
      state_config$electric_poly_path,
      show_col_types = FALSE
    ) |>
      st_as_sf(wkt = "the_geom") |>
      st_set_crs(4326) |> # Set WGS84 as the CRS for the input data|>
      rename(utility = COMP_FULL)

    gas_utility_polygons <- read_csv(
      state_config$gas_poly_path,
      show_col_types = FALSE
    ) |>
      st_as_sf(wkt = "the_geom") |>
      st_set_crs(4326) |> # Set WGS84 as the CRS for the input data|>
      rename(utility = COMP_FULL)
  }

  # calculate overlap between PUMAS and utilities
  puma_elec_overlap <- pumas |>
    st_transform(state_config$state_crs) |>
    mutate(puma_area = st_area(geometry)) |> # Calculate total area of each PUMA
    st_intersection(
      electric_utility_polygons |> st_transform(state_config$state_crs)
    ) |> # Intersect with utilities
    mutate(
      overlap_area = st_area(geometry), # Calculate area of each overlap
      pct_overlap = as.numeric(overlap_area / puma_area * 100) # Calculate percentage
    ) |>
    st_drop_geometry() |>
    select(puma_id = PUMACE10, pct_overlap, contains("utility"))
  puma_gas_overlap <- pumas |>
    st_transform(state_config$state_crs) |> # Transform to Massachusetts state plane (meters)
    mutate(puma_area = st_area(geometry)) |> # Calculate total area of each PUMA
    st_intersection(
      gas_utility_polygons |> st_transform(state_config$state_crs)
    ) |> # Intersect with utilities
    mutate(
      overlap_area = st_area(geometry), # Calculate area of each overlap
      pct_overlap = as.numeric(overlap_area / puma_area * 100) # Calculate percentage
    ) |>
    st_drop_geometry() |>
    select(puma_id = PUMACE10, pct_overlap, contains("utility"))

  if (state_code == "MA") {
    puma_elec_overlap <- split_multi_service_areas(puma_elec_overlap)
    puma_gas_overlap <- split_multi_service_areas(puma_gas_overlap)
  }
  puma_elec_probs <- puma_elec_overlap |>
    left_join(utility_name_map, by = c("utility" = "state_name")) |>
    mutate(utility = coalesce(std_name, utility)) |>
    select(-std_name) |>
    mutate(
      utility = case_when(
        str_detect(utility, "^Municipal Utility:") ~ paste0(
          "muni-",
          str_to_lower(str_trim(str_remove(utility, "^Municipal Utility:")))
        ),
        .default = utility
      )
    ) |>
    group_by(puma_id) |>
    mutate(
      probability = pct_overlap / sum(pct_overlap)
    ) |>
    ungroup() |>
    select(puma_id, utility, probability) |>
    pivot_wider(
      names_from = utility,
      values_from = probability,
      values_fill = 0
    )

  puma_gas_probs <- puma_gas_overlap |>
    left_join(utility_name_map, by = c("utility" = "state_name")) |>
    mutate(utility = coalesce(std_name, utility)) |>
    select(-std_name) |>
    group_by(puma_id) |>
    mutate(
      probability = pct_overlap / sum(pct_overlap)
    ) |>
    ungroup() |>
    select(puma_id, utility, probability) |>
    filter(utility != "none") |>
    pivot_wider(
      names_from = utility,
      values_from = probability,
      values_fill = 0
    )
  # get resstock data
  # Create connection when needed
  con <- DBI::dbConnect(
    duckdb::duckdb(),
    dbdir = db_path,
    read_only = TRUE
  )
  on.exit(DBI::dbDisconnect(con), add = TRUE) # Ensure connection is closed

  bldgs <- tbl(con, "housing_units") |>
    select(bldg_id, puma = in.puma, heating_fuel = `in.heating_fuel`) |>
    mutate(puma = str_sub(puma, start = -5)) |>
    collect()

  # assign elec to bldgs
  building_elec <- bldgs |>
    left_join(puma_elec_probs, by = c("puma" = "puma_id")) |>
    rowwise() |>
    mutate(
      utility = sample(
        names(pick(everything()))[-(1:3)], # Get utility names from columns, skip first 3 (bldg_id, puma, heating_fuel)
        size = 1,
        prob = c_across(-(1:3)) # Get probabilities, skip first 3 columns
      )
    ) |>
    ungroup() |>
    select(bldg_id, electric_utility = utility)

  # assign gas to bldgs
  building_gas <- bldgs |>
    left_join(puma_gas_probs, by = c("puma" = "puma_id")) |>
    rowwise() |>
    mutate(
      utility = case_when(
        heating_fuel == "Natural Gas" ~ sample(
          names(pick(everything()))[-(1:3)], # Get utility names from columns, skip first 3 (bldg_id, puma, heating_fuel)
          size = 1,
          prob = c_across(-(1:3)) # Get probabilities, skip first 3 columns
        ),
        .default = NA
      )
    ) |>
    ungroup() |>
    select(bldg_id, gas_utility = utility)

  building_utilities <- building_elec |>
    left_join(building_gas, by = "bldg_id")

  if (save_file) {
    write_csv(building_utilities, state_config$hh_utilities_path)
  }

  return(building_utilities)
}


# create_hh_utilities <- function(
#     state_code,
#     state_config,
#     puma_centroids_path = puma_centroids_default,
#     tract_lookup = tract_lookup_default,
#     save_file = TRUE) {
#   # Get state-specific configs
#   state_config <- state_configs[[state_code]]
#   if (is.null(state_config)) {
#     stop(sprintf("No configuration available for state: %s", state))
#   }

#   # Use state_config values
#   state_fips <- state_config$state_fips
#   electric_poly_path <- state_config$electric_poly_path
#   gas_poly_path <- state_config$gas_poly_path

#   # Create connection when needed
#   resstock <- DBI::dbConnect(
#     duckdb::duckdb(),
#     dbdir = state_config$resstock_path,
#     read_only = TRUE
#   )
#   on.exit(DBI::dbDisconnect(resstock), add = TRUE) # Ensure connection is closed

#   # # We need to map households to gas and electric utility service territories, via their PUMA.
#   # # ResStock PUMA codes don't match Census PUMA codes, so we map those first.

#   puma_centroids <- read_tsv(puma_centroids_path, show_col_types = FALSE) |> # Census PUMA centroids
#     select(GEOID, INTPTLAT, INTPTLONG)

#   resstock_puma_mapping <- read_csv(tract_lookup, show_col_types = FALSE) |> # Census tracts with ResStock and Census PUMAS
#     select(nhgis_2010_puma_gisjoin, puma_tsv) |>
#     separate_wider_delim(puma_tsv, delim = ", ", names = c("state", "puma")) |> # Fix FIPS code: "NY, 001341" -> "NY", "36001341"
#     mutate(puma = paste0(state_fips, puma)) |>
#     filter(state == state_code) |>
#     select(-state) |>
#     distinct() |> # from 1 per tract to 1 per PUMA
#     left_join(puma_centroids, by = c("puma" = "GEOID")) |>
#     st_as_sf(coords = c("INTPTLONG", "INTPTLAT"))

#   electric_utility_polygons <- read_csv(electric_poly_path, show_col_types = FALSE) |>
#     st_as_sf(wkt = "the_geom")

#   gas_utility_polygons <- read_csv(gas_poly_path, show_col_types = FALSE) |>
#     st_as_sf(wkt = "the_geom")

#   utility_name_mapping <- tribble(
#     ~std_name, ~bills_name, ~state_name,
#     "kedny", "KEDNY", "National Grid - NYC",
#     "kedli", "KEDLI", "National Grid - Long Island",
#     "nimo", "NiMO", "National Grid",
#     "cenhud", "CenHud", "Central Hudson Gas and Electric",
#     "coned", "ConEd", "Consolidated Edison",
#     "nationalfuel", "NF", "National Fuel Gas Distribution",
#     "rge", "RGE", "Rochester Gas and Electric",
#     "nyseg", "NYSEG", "NYS Electric and Gas",
#     "pseg-li", "PSEG-LI", "Long Island Power Authority",
#     "or", "O&R", "Orange and Rockland Utilities",
#   )

#   puma_electric <- resstock_puma_mapping |>
#     st_join(electric_utility_polygons,
#       join = st_covered_by
#     ) |>
#     select(-puma) |>
#     rename(
#       puma = nhgis_2010_puma_gisjoin,
#       utility_full_name = COMP_FULL,
#       utility_short_name = COMP_SHORT
#     ) |>
#     select(puma, utility_full_name, utility_short_name) |>
#     st_drop_geometry() |>
#     left_join(utility_name_mapping, by = c("utility_full_name" = "state_name"))

#   puma_gas <- resstock_puma_mapping |>
#     st_join(gas_utility_polygons,
#       join = st_covered_by
#     ) |>
#     select(-puma) |>
#     rename(
#       puma = nhgis_2010_puma_gisjoin,
#       utility_full_name = COMP_FULL,
#       utility_short_name = COMP_SHORT
#     ) |>
#     select(puma, utility_full_name, utility_short_name) |>
#     st_drop_geometry() |>
#     left_join(utility_name_mapping, by = c("utility_full_name" = "state_name"))

#   hh_utilities <- tbl(resstock, "housing_units") |>
#     left_join(puma_gas, by = c("in.puma" = "puma"), copy = TRUE) |>
#     left_join(puma_electric, by = c("in.puma" = "puma"), copy = TRUE, suffix = c(".gas", ".electric")) |>
#     select(bldg_id, std_name.gas, std_name.electric, bills_name.gas, bills_name.electric) |>
#     mutate(
#       gas_missing = is.na(bills_name.gas),
#       electric_missing = is.na(bills_name.electric),
#       bills_name.gas = coalesce(bills_name.gas, "NiMO"),
#       bills_name.electric = coalesce(bills_name.electric, "NiMO"),
#       std_name.gas = coalesce(std_name.gas, "nimo"),
#       std_name.electric = coalesce(std_name.electric, "nimo")
#     ) |>
#     collect()

#   if (save_file) {
#     write_csv(hh_utilities, state_config$hh_utilities_path)
#   }

#   return(hh_utilities)
# }

merge_ma_electric_polygons <- function(electric_poly_path) {
  # MA electric utility polygons are mapped to municipalities, we want to merge polygons by utility

  electric_utility_poly <- st_read(electric_poly_path) |>
    mutate(
      utility_1 = str_extract(ELEC_LABEL, "^[^,]+"), # Extract everything before first comma
      utility_2 = str_extract(ELEC_LABEL, "(?<=, ).+") # Extract everything after comma and space
    ) |>
    # Clean up by trimming any whitespace
    mutate(
      across(c(utility_1, utility_2), str_trim),
      multi_utility = ifelse(is.na(utility_2), 0, 1)
      # utility_1 = case_when(utility_1 == "Municipal" ~ paste0("muni-", str_to_lower(TOWN)),
      #   .default = utility_1
      # )
    )

  merged_utilities <- electric_utility_poly |>
    rename(utility = ELEC_LABEL) |>
    group_by(utility) |>
    summarise(
      n_towns = n(), # Count number of towns per utility
      utility_1 = first(utility_1),
      utility_2 = first(utility_2),
      multi_utility = first(multi_utility)
    ) |>
    ungroup()

  return(merged_utilities)
}

merge_ma_gas_polygons <- function(gas_poly_path) {
  # MA gas utility polygons are mapped to municipalities, we want to merge polygons by utility

  gas_utility_poly <- st_read(gas_poly_path) |>
    mutate(
      utility_1 = str_extract(GAS_LABEL, "^[^,]+"), # Extract everything before first comma
      utility_2 = str_extract(GAS_LABEL, "(?<=, ).+") # Extract everything after comma and space
    ) |>
    # Clean up by trimming any whitespace
    mutate(
      across(c(utility_1, utility_2), str_trim),
      multi_utility = ifelse(is.na(utility_2), 0, 1)
      # utility_1 = case_when(utility_1 == "Municipal" ~ paste0("muni-", str_to_lower(TOWN)),
      #   .default = utility_1
      # )
    )

  merged_utilities <- gas_utility_poly |>
    rename(utility = GAS_LABEL) |>
    group_by(utility) |>
    summarise(
      n_towns = n(), # Count number of towns per utility
      utility_1 = first(utility_1),
      utility_2 = first(utility_2),
      multi_utility = first(multi_utility)
    ) |>
    ungroup()
  return(merged_utilities)
}

split_multi_service_areas <- function(puma_utility_overlap) {
  # some service areas are labeled with 2 utilities, we want to split them into 2 rows and split the area
  mutli_utils <- puma_utility_overlap |>
    filter(multi_utility == 1) |>
    select(-utility, -multi_utility) |>
    pivot_longer(-c(puma_id, pct_overlap), values_to = "utility") |>
    mutate(pct_overlap = pct_overlap / 2) |>
    select(-name)

  single_utils <- puma_utility_overlap |>
    filter(multi_utility == 0) |>
    select(puma_id, utility, pct_overlap)

  pct_overlap_final <- bind_rows(mutli_utils, single_utils) |>
    summarise(.by = c(puma_id, utility), pct_overlap = sum(pct_overlap))

  return(pct_overlap_final)
}
write_utilities_to_db <- function(state_code, db_path, config = state_configs) {
  if (!state_code %in% names(config)) {
    message(sprintf(
      "Cannot add utilities, state %s not supported. Only %s are currently supported.",
      state_code,
      paste(names(config), collapse = " and ")
    ))
    return(NULL)
  }

  building_utilities <- create_hh_utilities(state_code, db_path = db_path)

  con <- DBI::dbConnect(
    duckdb::duckdb(),
    dbdir = db_path,
    read_only = FALSE
  )
  on.exit(DBI::dbDisconnect(con), add = TRUE) # Ensure connection is closed
  # Write the results to the housing_units table
  # Add columns one at a time
  DBI::dbExecute(
    con,
    "ALTER TABLE housing_units ADD COLUMN IF NOT EXISTS electric_utility VARCHAR;"
  )
  DBI::dbExecute(
    con,
    "ALTER TABLE housing_units ADD COLUMN IF NOT EXISTS gas_utility VARCHAR;"
  )

  # Update the housing_units table with utility assignments
  # Write utilities to temporary table
  DBI::dbWriteTable(
    con,
    "temp_utilities",
    building_utilities |> select(bldg_id, electric_utility, gas_utility),
    temporary = TRUE,
    overwrite = TRUE
  )

  # Update housing_units with utility assignments
  DBI::dbExecute(
    con,
    "
      UPDATE housing_units AS h
      SET
        electric_utility = t.electric_utility,
        gas_utility = t.gas_utility
      FROM temp_utilities AS t
      WHERE h.bldg_id = t.bldg_id;
    "
  )

  # Remove columns one at a time
  # DBI::dbExecute(con, "ALTER TABLE housing_units DROP COLUMN IF EXISTS electric_utility;")
  # DBI::dbExecute(con, "ALTER TABLE housing_units DROP COLUMN IF EXISTS gas_utility;")
}
