#!/usr/bin/env Rscript

# =============================================================================
# Stage 2 Multinomial Logit (Block Group): Cluster Composition ~ Census Predictors
# =============================================================================
#
# Key principles for regulatory defensibility
# -------------------------------------------
# - Predictors are inferred from the census parquet to avoid fragile hard-coding.
# - Some predictors are "shares" that sum to 100 within a group (compositional).
#   To ensure a unique model, one pre-specified reference share per group is omitted
#   from estimation. Omitted shares are documented explicitly in outputs.
# - Predictors are NOT mean-centered. Optional scaling divides by SD only.
# - If VGAM is used, any remaining rank-deficient terms are dropped deterministically.
#
# Outputs
# -------
# - regression_results.parquet
# - regression_diagnostics.json
# - stage2_input_qc.json
# - regression_data_blockgroups_wide.parquet
# - omitted_predictors.parquet   <-- NEW: documents reference omissions + rank drops
# - stage2_manifest.json
# - stage2_metadata.json
#
# =============================================================================


# -----------------------------
# CLI args (no external deps)
# -----------------------------
print_help_and_exit <- function(exit_code = 0) {
  cat(
    paste0(
      "\nStage 2 Multinomial Logit (Block Group)\n\n",
      "Usage:\n",
      "  Rscript stage2_multinom_blockgroup_weighted.R [options]\n\n",
      "Required:\n",
      "  --clusters PATH        Cluster assignments parquet (ZIP+4 household-day rows)\n",
      "  --crosswalk PATH       ZIP+4 -> Block Group crosswalk (tab-delimited txt)\n",
      "  --census PATH          Census predictors parquet (Block Group level)\n",
      "  --out-dir PATH         Output directory\n\n",
      "Optional:\n",
      "  --baseline-cluster K           Baseline cluster label (default: choose most frequent)\n",
      "  --min-obs-per-bg N             Drop BGs with total household-days < N (default: 50)\n",
      "  --allow-missing-predictors 0|1 If 0, abort if predictor NA would drop any BGs (default: 0)\n",
      "  --standardize 0|1              If 1, divide predictors by SD (NO mean-centering) (default: 0)\n",
      "  --use-vgam 0|1                 Use VGAM::vglm() (IRLS) instead of nnet::multinom (default: 1)\n",
      "  --verbose 0|1                  Verbose logging (default: 1)\n",
      "  --no-emoji 0|1                 Disable unicode icons (default: 0)\n",
      "  --help                         Print this help and exit\n\n",
      "Notes:\n",
      "  - Predictors are inferred from the census parquet columns.\n",
      "  - Some predictors are compositional shares; one reference share per group is omitted\n",
      "    deterministically and documented in omitted_predictors.parquet.\n",
      "  - Model uses COUNT response: cbind(count_clusterA, count_clusterB, ...)\n",
      "  - Zeros are handled naturally by the multinomial likelihood; no smoothing is applied.\n\n"
    )
  )
  quit(status = exit_code)
}

args <- commandArgs(trailingOnly = TRUE)
if (any(args %in% c("--help", "-h"))) print_help_and_exit(0)

get_arg <- function(flag, default = NULL) {
  hit <- grep(paste0("^", flag, "="), args)
  if (length(hit) > 0) return(sub(paste0("^", flag, "="), "", args[hit[1]]))
  hit2 <- which(args == flag)
  if (length(hit2) > 0 && hit2[1] < length(args)) return(args[hit2[1] + 1])
  default
}

parse_bool01 <- function(x, default = 0L) {
  if (is.null(x) || is.na(x) || x == "") return(as.integer(default))
  s <- tolower(trimws(as.character(x)))
  if (s %in% c("1", "true", "t", "yes", "y")) return(1L)
  if (s %in% c("0", "false", "f", "no", "n")) return(0L)
  as.integer(default)
}

parse_int <- function(x, default = NA_integer_) {
  if (is.null(x) || is.na(x) || x == "") return(default)
  suppressWarnings(v <- as.integer(x))
  if (is.na(v)) default else v
}

stopf <- function(fmt, ...) stop(sprintf(fmt, ...), call. = FALSE)

# Required I/O paths
CLUSTERS_PATH <- get_arg("--clusters", default = NULL)
CROSSWALK_PATH <- get_arg("--crosswalk", default = NULL)
CENSUS_PATH <- get_arg("--census", default = NULL)
OUT_DIR <- get_arg("--out-dir", default = NULL)

# Model/config knobs
BASELINE_CLUSTER_ARG <- get_arg("--baseline-cluster", default = NULL)
MIN_OBS_PER_BG <- parse_int(get_arg("--min-obs-per-bg", default = "50"), default = 50L)

ALLOW_MISSING_PREDICTORS <- parse_bool01(get_arg("--allow-missing-predictors", default = "0"), default = 0L)
STANDARDIZE <- parse_bool01(get_arg("--standardize", default = "0"), default = 0L)
USE_VGAM <- parse_bool01(get_arg("--use-vgam", default = "1"), default = 1L)

VERBOSE <- parse_bool01(get_arg("--verbose", default = "1"), default = 1L)
NO_EMOJI <- parse_bool01(get_arg("--no-emoji", default = "0"), default = 0L)

if (is.null(CLUSTERS_PATH) || is.null(CROSSWALK_PATH) || is.null(CENSUS_PATH) || is.null(OUT_DIR)) {
  cat("Missing required argument(s). Use --help.\n")
  quit(status = 2)
}

# -----------------------------
# Deps
# -----------------------------
require_pkg <- function(pkg) requireNamespace(pkg, quietly = TRUE)

if (!require_pkg("arrow")) stopf("Missing R package 'arrow'. Install with: install.packages('arrow')")
if (!require_pkg("jsonlite")) stopf("Missing R package 'jsonlite'. Install with: install.packages('jsonlite')")
if (!require_pkg("dplyr")) stopf("Missing R package 'dplyr'. Install with: install.packages('dplyr')")
if (!require_pkg("tibble")) stopf("Missing R package 'tibble'. Install with: install.packages('tibble')")
if (!require_pkg("nnet")) stopf("Missing R package 'nnet'. Install with: install.packages('nnet')")
if (USE_VGAM == 1L && !require_pkg("VGAM")) stopf("Missing R package 'VGAM'. Install with: install.packages('VGAM')")

suppressPackageStartupMessages({
  library(arrow)
  library(jsonlite)
  library(dplyr)
  library(tibble)
  library(nnet)
  if (USE_VGAM == 1L) library(VGAM)
})

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

icon_for <- function() {
  if (NO_EMOJI == 1L) return(list(ok = "OK", warn = "[WARN]", crit = "[CRIT]"))
  list(ok = "\u2705", warn = "\U0001F7E1", crit = "\U0001F534")
}
IC <- icon_for()

logi <- function(...) if (VERBOSE == 1L) cat(sprintf(...), "\n")

safe_write_json <- function(obj, path) {
  jsonlite::write_json(obj, path, pretty = TRUE, auto_unbox = TRUE)
}

logi("%s Config: standardize=%d use_vgam=%d", IC$ok, STANDARDIZE, USE_VGAM)
t_total_start <- Sys.time()


# -----------------------------
# Reference-category specification (deterministic omissions)
# -----------------------------
REFERENCE_GROUPS <- list(
  list(
    group_id = "income_3bin",
    group_name = "Household income distribution (3 bins)",
    variables = c("pct_income_under_25k", "pct_income_25k_to_75k", "pct_income_75k_plus"),
    reference = "pct_income_75k_plus",
    rationale = "Shares sum to 100; omit one bin to ensure a unique model. Highest-income share is a stable reference for policy interpretation."
  ),
  list(
    group_id = "tenure_2bin",
    group_name = "Housing tenure (occupied units)",
    variables = c("pct_owner_occupied", "pct_renter_occupied"),
    reference = "pct_owner_occupied",
    rationale = "Owner + renter shares sum to 100; omit owners so renter share is interpreted relative to owners."
  ),
  list(
    group_id = "vintage_3bin",
    group_name = "Housing vintage distribution (3 bins)",
    variables = c("pct_housing_built_2000_plus", "pct_housing_built_1980_1999", "old_building_pct"),
    reference = "pct_housing_built_2000_plus",
    rationale = "Shares sum to 100; omit newest stock so older stock shares are interpreted relative to newer construction."
  ),
  list(
    group_id = "home_value_3bin",
    group_name = "Owner-occupied home value distribution (3 bins)",
    variables = c("pct_home_value_under_150k", "pct_home_value_150k_to_299k", "pct_home_value_300k_plus"),
    reference = "pct_home_value_300k_plus",
    rationale = "Shares sum to 100; omit highest-value share so lower/middle value shares are interpreted relative to highest-value homes."
  ),
  list(
    group_id = "age_6bin",
    group_name = "Population age distribution (6 bins)",
    variables = c(
      "pct_population_under_5", "pct_population_5_to_17", "pct_population_18_to_24",
      "pct_population_25_to_44", "pct_population_45_to_64", "pct_population_65_plus"
    ),
    reference = "pct_population_25_to_44",
    rationale = "Shares sum to 100; omit 25–44 as a central working-age reference group."
  )
)

REFERENCE_VARS <- unique(vapply(REFERENCE_GROUPS, function(g) g$reference, character(1)))

implied_definitions <- lapply(REFERENCE_GROUPS, function(g) {
  incl <- setdiff(g$variables, g$reference)
  list(
    group_id = g$group_id,
    group_name = g$group_name,
    reference = g$reference,
    implied_as = paste0(g$reference, " = 100 - (", paste(incl, collapse = " + "), ")")
  )
})

REFERENCE_CAVEAT <- paste(
  "Some predictors are shares that sum to 100 within a group (compositional).",
  "To ensure a unique multinomial logit model, one pre-specified reference share per group is omitted from estimation.",
  "The omitted share is not independently estimated; it is implied by the included shares via the identity in the diagnostics."
)


# -----------------------------
# Helpers: keys + inference
# -----------------------------
normalize_zip4 <- function(x) {
  s <- as.character(x)
  s <- trimws(s)
  s[s == ""] <- NA_character_
  out <- s
  is9 <- !is.na(s) & grepl("^[0-9]{9}$", s)
  out[is9] <- paste0(substr(s[is9], 1, 5), "-", substr(s[is9], 6, 9))
  out
}

zfill4 <- function(x) {
  s <- as.character(x)
  s <- trimws(s)
  s[s == ""] <- NA_character_
  s <- gsub("[^0-9]", "", s)
  s <- ifelse(is.na(s), NA_character_, sprintf("%04d", as.integer(s)))
  s
}

infer_geoid_col <- function(df) {
  nms <- names(df)
  low <- tolower(nms)
  if ("geoid" %in% low) return(nms[which(low == "geoid")[1]])
  if ("censuskey2023" %in% low) return(nms[which(low == "censuskey2023")[1]])
  if ("censuskey2020" %in% low) return(nms[which(low == "censuskey2020")[1]])
  NULL
}

infer_predictors <- function(census_df) {
  geoid_col <- infer_geoid_col(census_df)
  if (is.null(geoid_col)) {
    stopf("Census predictors must include a GEOID-like column. Found: %s", paste(names(census_df), collapse = ", "))
  }

  id_like <- unique(c(geoid_col, "GEOID", "NAME"))
  candidates <- setdiff(names(census_df), id_like)

  is_numish <- vapply(
    census_df[candidates],
    function(x) is.numeric(x) || is.integer(x) || is.logical(x),
    logical(1)
  )
  preds <- candidates[is_numish]

  all_na <- preds[vapply(census_df[preds], function(x) all(is.na(x)), logical(1))]
  preds <- setdiff(preds, all_na)

  list(geoid_col = geoid_col, predictors = preds, dropped_all_na = all_na)
}

drop_rank_deficient_terms <- function(model_df, predictors) {
  ftmp <- stats::as.formula(paste0("~ ", paste(predictors, collapse = " + ")))
  Xmm <- stats::model.matrix(ftmp, data = model_df)

  is_const <- apply(Xmm, 2, function(v) {
    if (!is.numeric(v)) return(FALSE)
    rng <- range(v, na.rm = TRUE)
    is.finite(rng[1]) && is.finite(rng[2]) && abs(rng[2] - rng[1]) < 1e-12
  })
  is_const[colnames(Xmm) == "(Intercept)"] <- FALSE
  const_cols <- colnames(Xmm)[is_const]

  if (length(const_cols) > 0) {
    logi("%s Dropping %d constant design columns before rank check.", IC$warn, length(const_cols))
    keep <- setdiff(colnames(Xmm), const_cols)
    Xmm <- Xmm[, keep, drop = FALSE]
  }

  qrX <- qr(Xmm)
  rk <- qrX$rank
  full_cols <- colnames(Xmm)
  dropped_predictors <- character(0)

  if (rk < ncol(Xmm)) {
    piv <- qrX$pivot
    keep_cols <- full_cols[piv[seq_len(rk)]]
    drop_cols <- setdiff(full_cols, keep_cols)

    if (!"(Intercept)" %in% keep_cols) stopf("Internal error: intercept was dropped by rank procedure.")

    dropped_predictors <- intersect(drop_cols, predictors)
    kept_predictors <- setdiff(predictors, dropped_predictors)

    logi(
      "%s Rank-deficient design detected: rank=%d of %d columns. Dropping %d term(s).",
      IC$warn, rk, ncol(Xmm), length(dropped_predictors)
    )
    if (length(dropped_predictors) > 0) {
      logi("%s Dropped predictors (rank-deficient): %s", IC$warn, paste(dropped_predictors, collapse = ", "))
    }

    return(list(
      predictors = kept_predictors,
      dropped_predictors = dropped_predictors,
      rank = rk,
      ncol_design = ncol(Xmm)
    ))
  }

  return(list(
    predictors = predictors,
    dropped_predictors = character(0),
    rank = rk,
    ncol_design = ncol(Xmm)
  ))
}


# -----------------------------
# Read + aggregate clusters (memory-safe)
# -----------------------------
if (!file.exists(CLUSTERS_PATH)) stopf("Clusters parquet not found: %s", CLUSTERS_PATH)
if (!file.exists(CROSSWALK_PATH)) stopf("Crosswalk file not found: %s", CROSSWALK_PATH)
if (!file.exists(CENSUS_PATH)) stopf("Census predictors parquet not found: %s", CENSUS_PATH)

logi("%s Aggregating clusters from parquet (Arrow Dataset compute): %s", IC$ok, CLUSTERS_PATH)

clusters_ds <- tryCatch(
  arrow::open_dataset(sources = CLUSTERS_PATH, format = "parquet"),
  error = function(e) NULL
)
if (is.null(clusters_ds)) stopf("Failed to open clusters parquet as Arrow Dataset: %s", CLUSTERS_PATH)

zip_cluster_counts <- clusters_ds %>%
  dplyr::select(zip_code, cluster) %>%
  dplyr::filter(!is.na(zip_code), !is.na(cluster)) %>%
  dplyr::group_by(zip_code, cluster) %>%
  dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%
  dplyr::collect()

zip_cluster_counts <- zip_cluster_counts %>%
  dplyr::mutate(
    zip4 = normalize_zip4(zip_code),
    cluster = suppressWarnings(as.integer(as.character(cluster))),
    n = suppressWarnings(as.integer(n))
  ) %>%
  dplyr::filter(!is.na(zip4), !is.na(cluster), !is.na(n), n > 0)

if (nrow(zip_cluster_counts) == 0) stopf("No usable ZIP+4×cluster counts after basic filtering.")
household_day_rows_total <- sum(zip_cluster_counts$n, na.rm = TRUE)


# -----------------------------
# Read crosswalk
# -----------------------------
logi("%s Reading crosswalk: %s", IC$ok, CROSSWALK_PATH)

cw_tbl <- tryCatch(
  arrow::read_csv_arrow(CROSSWALK_PATH, delim = "\t"),
  error = function(e) NULL
)

cw <- NULL
if (!is.null(cw_tbl)) {
  cw <- as.data.frame(cw_tbl)
} else {
  cw <- tryCatch(
    read.delim(CROSSWALK_PATH, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE),
    error = function(e) NULL
  )
}
if (is.null(cw)) stopf("Failed to read crosswalk: %s", CROSSWALK_PATH)

low_cw <- tolower(names(cw))
zip_col <- if ("zip" %in% low_cw) names(cw)[which(low_cw == "zip")[1]] else NA_character_
zip4_col <- if ("zip4" %in% low_cw) names(cw)[which(low_cw == "zip4")[1]] else NA_character_
geoid_col_cw <- if ("censuskey2023" %in% low_cw) names(cw)[which(low_cw == "censuskey2023")[1]] else NA_character_

if (is.na(zip_col) || is.na(zip4_col)) stopf("Crosswalk must include columns Zip and Zip4. Found: %s", paste(names(cw), collapse = ", "))
if (is.na(geoid_col_cw)) stopf("Crosswalk must include column CensusKey2023. Found: %s", paste(names(cw), collapse = ", "))

zip4_present <- unique(zip_cluster_counts$zip4)

cw <- cw %>%
  dplyr::transmute(
    Zip = as.character(.data[[zip_col]]),
    Zip4 = zfill4(.data[[zip4_col]]),
    zip4 = ifelse(!is.na(Zip) & !is.na(Zip4), paste0(Zip, "-", Zip4), NA_character_),
    block_group_geoid = as.character(.data[[geoid_col_cw]])
  ) %>%
  dplyr::filter(!is.na(zip4), !is.na(block_group_geoid)) %>%
  dplyr::filter(zip4 %in% zip4_present) %>%
  dplyr::mutate(block_group_geoid = substr(block_group_geoid, 1, 12)) %>%
  dplyr::filter(!is.na(block_group_geoid), nchar(block_group_geoid) == 12)

if (nrow(cw) == 0) stopf("Crosswalk produced 0 usable rows after cleaning/filtering.")

cw <- cw %>%
  dplyr::arrange(zip4, block_group_geoid) %>%
  dplyr::group_by(zip4) %>%
  dplyr::slice(1) %>%
  dplyr::ungroup()

dup_zip4 <- cw %>% dplyr::count(zip4) %>% dplyr::filter(n > 1)
if (nrow(dup_zip4) > 0) stopf("Crosswalk still has non-unique zip4 after deterministic resolution. Found %d duplicates.", nrow(dup_zip4))


# -----------------------------
# Join + aggregate to BG counts
# -----------------------------
logi("%s Joining ZIP+4×cluster counts to crosswalk...", IC$ok)

clusters2 <- zip_cluster_counts %>%
  dplyr::select(zip4, cluster, n) %>%
  dplyr::inner_join(cw, by = "zip4")

household_day_rows_after_crosswalk <- sum(clusters2$n, na.rm = TRUE)
dropped_missing_crosswalk <- household_day_rows_total - household_day_rows_after_crosswalk

if (nrow(clusters2) == 0) stopf("All cluster counts dropped after crosswalk join. Check zip4 normalization and crosswalk keying.")

logi("%s Aggregating to block group counts...", IC$ok)

bg_counts <- clusters2 %>%
  dplyr::group_by(block_group_geoid, cluster) %>%
  dplyr::summarize(n = sum(n, na.rm = TRUE), .groups = "drop")

total_by_bg <- clusters2 %>%
  dplyr::group_by(block_group_geoid) %>%
  dplyr::summarize(total_household_days = sum(n, na.rm = TRUE), .groups = "drop")

clusters_observed <- sort(unique(bg_counts$cluster))
if (length(clusters_observed) < 2) stopf("Need at least 2 clusters observed after aggregation; found: %s", paste(clusters_observed, collapse = ","))

bg_wide <- total_by_bg %>% dplyr::rename(GEOID = block_group_geoid)

for (k in clusters_observed) {
  colname <- paste0("cluster_", k)
  tmp <- bg_counts %>%
    dplyr::filter(cluster == k) %>%
    dplyr::transmute(GEOID = block_group_geoid, n = as.integer(n))

  bg_wide <- bg_wide %>%
    dplyr::left_join(tmp, by = "GEOID") %>%
    dplyr::mutate(!!colname := ifelse(is.na(n), 0L, as.integer(n))) %>%
    dplyr::select(-n)
}

bg_wide <- bg_wide %>% dplyr::filter(total_household_days >= as.integer(MIN_OBS_PER_BG))
if (nrow(bg_wide) == 0) stopf("No block groups remain after --min-obs-per-bg filtering (N=%d).", MIN_OBS_PER_BG)

zero_stats <- list()
for (k in clusters_observed) {
  colname <- paste0("cluster_", k)
  if (colname %in% names(bg_wide)) {
    n_zeros <- sum(bg_wide[[colname]] == 0, na.rm = TRUE)
    pct_zeros <- 100 * n_zeros / nrow(bg_wide)
    zero_stats[[as.character(k)]] <- list(n_zeros = as.integer(n_zeros), pct_zeros = as.numeric(pct_zeros))
  }
}


# -----------------------------
# Read census predictors + infer predictors
# -----------------------------
logi("%s Reading census predictors: %s", IC$ok, CENSUS_PATH)

census <- arrow::read_parquet(CENSUS_PATH, as_data_frame = TRUE)
inf <- infer_predictors(census)
CENSUS_GEOID_COL <- inf$geoid_col
PREDICTORS_RAW <- inf$predictors
DROPPED_ALL_NA <- inf$dropped_all_na

if (length(PREDICTORS_RAW) == 0) stopf("No usable numeric predictors inferred from census parquet.")

census <- census %>%
  dplyr::mutate(GEOID = as.character(.data[[CENSUS_GEOID_COL]])) %>%
  dplyr::select(GEOID, dplyr::any_of(PREDICTORS_RAW)) %>%
  dplyr::distinct(GEOID, .keep_all = TRUE)

logi("%s Joining census predictors to BG counts...", IC$ok)
bg_model <- bg_wide %>% dplyr::inner_join(census, by = "GEOID")

# Missing predictor handling
pred_mat <- bg_model %>% dplyr::select(dplyr::any_of(PREDICTORS_RAW))
any_na <- apply(is.na(pred_mat), 1, any)

drop_missing_pred_bg <- sum(any_na)
drop_missing_pred_hhday <- if (drop_missing_pred_bg > 0) sum(bg_model$total_household_days[any_na], na.rm = TRUE) else 0

if (drop_missing_pred_bg > 0 && ALLOW_MISSING_PREDICTORS == 0L) {
  stopf(
    "Predictor missingness would drop %d block groups (%.0f household-days). Refuse to proceed. Set --allow-missing-predictors=1 to override.",
    drop_missing_pred_bg, drop_missing_pred_hhday
  )
}
if (drop_missing_pred_bg > 0 && ALLOW_MISSING_PREDICTORS == 1L) {
  bg_model <- bg_model[!any_na, , drop = FALSE]
}
if (nrow(bg_model) == 0) stopf("No block groups remain after predictor missingness filtering.")


# -----------------------------
# Deterministic reference-variable omission (if present)
# -----------------------------
omit_refs_present <- intersect(REFERENCE_VARS, PREDICTORS_RAW)
PREDICTORS <- setdiff(PREDICTORS_RAW, omit_refs_present)

if (length(omit_refs_present) > 0) {
  logi("%s Omitted %d reference-share predictor(s): %s", IC$warn, length(omit_refs_present), paste(omit_refs_present, collapse = ", "))
}

if (length(PREDICTORS) == 0) stopf("No predictors remain after reference omissions.")


# -----------------------------
# Optional: scale by SD only (NO mean-centering)
# -----------------------------
scaling_info <- NULL
zero_var_predictors <- character(0)

if (STANDARDIZE == 1L) {
  logi("%s Scaling %d predictors by SD only (no mean-centering)...", IC$ok, length(PREDICTORS))
  scaling_info <- list()

  for (col in PREDICTORS) {
    if (!col %in% names(bg_model)) next
    x <- bg_model[[col]]
    if (is.logical(x)) x <- as.numeric(x)

    sigma <- stats::sd(x, na.rm = TRUE)
    scaling_info[[col]] <- list(sd = as.numeric(sigma), centered = FALSE)

    if (is.finite(sigma) && sigma > 1e-10) {
      bg_model[[col]] <- x / sigma
    } else {
      zero_var_predictors <- c(zero_var_predictors, col)
      cat(sprintf("%s Predictor '%s' has ~zero variance; not scaled.\n", IC$warn, col))
      bg_model[[col]] <- x
    }
  }
}


# -----------------------------
# Choose baseline + response matrix
# -----------------------------
resp_cols <- paste0("cluster_", clusters_observed)
resp_cols <- resp_cols[resp_cols %in% names(bg_model)]
if (length(resp_cols) < 2) stopf("Need >=2 cluster count columns; found: %s", paste(resp_cols, collapse = ","))

cluster_totals_for_baseline <- sapply(resp_cols, function(cn) sum(bg_model[[cn]], na.rm = TRUE))
names(cluster_totals_for_baseline) <- resp_cols

baseline_cluster <- NA_integer_
if (!is.null(BASELINE_CLUSTER_ARG) && nchar(BASELINE_CLUSTER_ARG) > 0) {
  baseline_cluster <- suppressWarnings(as.integer(BASELINE_CLUSTER_ARG))
} else {
  max_col <- names(cluster_totals_for_baseline)[which.max(cluster_totals_for_baseline)]
  baseline_cluster <- as.integer(sub("^cluster_", "", max_col))
}

if (!paste0("cluster_", baseline_cluster) %in% resp_cols) {
  stopf("Baseline cluster %d not present among observed clusters: %s", baseline_cluster, paste(resp_cols, collapse = ", "))
}

resp_cols_ordered <- c(setdiff(resp_cols, paste0("cluster_", baseline_cluster)), paste0("cluster_", baseline_cluster))
Y <- as.matrix(bg_model[, resp_cols_ordered, drop = FALSE])

if (any(is.na(Y))) stopf("NA values detected in response matrix.")
if (any(Y < 0, na.rm = TRUE)) stopf("Negative counts detected in response matrix.")
rs <- rowSums(Y)
if (any(rs <= 0, na.rm = TRUE)) {
  bad_rows <- which(rs <= 0)
  stopf("Block groups with non-positive total counts. Example row indices: %s", paste(head(bad_rows, 10), collapse = ", "))
}

model_df <- bg_model[, c("GEOID", "total_household_days", PREDICTORS), drop = FALSE]
model_df$Y <- I(Y)

nonbase_cols <- resp_cols_ordered[resp_cols_ordered != paste0("cluster_", baseline_cluster)]
nonbase_clusters <- as.integer(sub("^cluster_", "", nonbase_cols))


# -----------------------------
# Rank check / drop terms (VGAM full-rank requirement)
# -----------------------------
dropped_predictors_rank <- character(0)
design_rank <- NA_integer_
design_ncol <- NA_integer_

if (USE_VGAM == 1L) {
  rk <- drop_rank_deficient_terms(model_df, PREDICTORS)
  PREDICTORS <- rk$predictors
  dropped_predictors_rank <- rk$dropped_predictors
  design_rank <- as.integer(rk$rank)
  design_ncol <- as.integer(rk$ncol_design)

  model_df <- bg_model[, c("GEOID", "total_household_days", PREDICTORS), drop = FALSE]
  model_df$Y <- I(Y)
}

if (length(PREDICTORS) == 0) stopf("No predictors remain after rank-deficiency handling.")


# -----------------------------
# Fit model
# -----------------------------
logi(
  "%s Fitting multinomial logit (counts) with %d BGs, %d predictors, %d clusters (baseline=%d)...",
  IC$ok, nrow(model_df), length(PREDICTORS), length(resp_cols_ordered), baseline_cluster
)

rhs <- paste(PREDICTORS, collapse = " + ")
form <- stats::as.formula(paste0("Y ~ ", rhs))

fit_start <- Sys.time()
fit_warnings <- character(0)
fit0_warnings <- character(0)

if (USE_VGAM == 1L) {
  logi("%s Using VGAM::vglm() (IRLS) for full-rank MLE...", IC$ok)

  fit <- tryCatch(
    withCallingHandlers(
      VGAM::vglm(form, family = VGAM::multinomial(refLevel = length(resp_cols_ordered)), data = model_df),
      warning = function(w) { fit_warnings <<- c(fit_warnings, w$message); invokeRestart("muffleWarning") }
    ),
    error = function(e) stopf("Model fit failed (VGAM::vglm): %s", e$message)
  )

  fit0 <- tryCatch(
    withCallingHandlers(
      VGAM::vglm(Y ~ 1, family = VGAM::multinomial(refLevel = length(resp_cols_ordered)), data = model_df),
      warning = function(w) { fit0_warnings <<- c(fit0_warnings, w$message); invokeRestart("muffleWarning") }
    ),
    error = function(e) stopf("Null model fit failed (VGAM::vglm): %s", e$message)
  )
} else {
  fit <- tryCatch(
    withCallingHandlers(
      nnet::multinom(form, data = model_df, trace = FALSE, maxit = 500),
      warning = function(w) { fit_warnings <<- c(fit_warnings, w$message); invokeRestart("muffleWarning") }
    ),
    error = function(e) stopf("Model fit failed (nnet::multinom): %s", e$message)
  )

  fit0 <- tryCatch(
    withCallingHandlers(
      nnet::multinom(Y ~ 1, data = model_df, trace = FALSE, maxit = 500),
      warning = function(w) { fit0_warnings <<- c(fit0_warnings, w$message); invokeRestart("muffleWarning") }
    ),
    error = function(e) stopf("Null model fit failed (nnet::multinom): %s", e$message)
  )
}

fit_end <- Sys.time()
fit_duration <- as.numeric(difftime(fit_end, fit_start, units = "secs"))
logi("%s Model fit completed in %.1f seconds", IC$ok, fit_duration)


# -----------------------------
# Convergence heuristics
# -----------------------------
convergence_ok <- TRUE
convergence_message <- "Model converged successfully"

all_warns <- c(fit_warnings, fit0_warnings)
if (length(all_warns) > 0) {
  conv_warnings <- grep("converg|iteration|maxit|step", all_warns, ignore.case = TRUE, value = TRUE)
  if (length(conv_warnings) > 0) {
    convergence_ok <- FALSE
    convergence_message <- paste0("CONVERGENCE WARNING: ", paste(unique(conv_warnings), collapse = "; "))
    cat(sprintf("\n%s %s\n", IC$warn, convergence_message))
  }
}

iter_used <- NA_integer_
if (USE_VGAM == 0L) {
  if (!is.null(fit$iter)) iter_used <- suppressWarnings(as.integer(fit$iter))
  if (is.na(iter_used) && !is.null(fit$niter)) iter_used <- suppressWarnings(as.integer(fit$niter))
  if (!is.na(iter_used) && iter_used >= 500) {
    convergence_ok <- FALSE
    convergence_message <- paste0(convergence_message, "; Reached maxit=500 (iter_used=", iter_used, ")")
    cat(sprintf("\n%s Reached maxit=500 (iter_used=%d). Treating as potential non-convergence.\n", IC$warn, iter_used))
  }
}

deviance_full <- as.numeric(stats::deviance(fit))
deviance_null <- as.numeric(stats::deviance(fit0))
deviance_ratio <- deviance_full / deviance_null
if (is.finite(deviance_ratio) && deviance_ratio > 0.95) {
  cat(sprintf(
    "\n%s Model deviance (%.2f) is very close to null deviance (%.2f). Low explanatory power and/or convergence issues possible.\n",
    IC$warn, deviance_full, deviance_null
  ))
  convergence_message <- paste0(convergence_message, "; High deviance_ratio=", sprintf("%.3f", deviance_ratio))
}


# -----------------------------
# Core stats
# -----------------------------
ll_full <- as.numeric(stats::logLik(fit))
ll_null <- as.numeric(stats::logLik(fit0))
pseudo_r2 <- 1.0 - (ll_full / ll_null)


# -----------------------------
# Coefficients table (nnet + VGAM) -> ALWAYS returns 'cluster'
# -----------------------------
extract_coef_table <- function(fit_obj, nonbase_clusters) {
  if (inherits(fit_obj, "multinom")) {
    coefs <- summary(fit_obj)$coefficients
    ses <- summary(fit_obj)$standard.errors
    pred_names <- colnames(coefs)

    rows <- vector("list", length = nrow(coefs) * length(pred_names))
    idx <- 0L
    for (i in seq_len(nrow(coefs))) {
      for (j in seq_along(pred_names)) {
        idx <- idx + 1L
        b <- as.numeric(coefs[i, j])
        se <- as.numeric(ses[i, j])
        z <- b / se
        p <- 2 * pnorm(-abs(z))
        rows[[idx]] <- list(
          eq = as.integer(i),
          cluster = as.integer(nonbase_clusters[i]),
          predictor = ifelse(pred_names[j] == "(Intercept)", "Intercept", pred_names[j]),
          coefficient = b,
          std_err = se,
          z_stat = z,
          p_value = p
        )
      }
    }
    return(tibble::as_tibble(do.call(rbind.data.frame, rows)))
  }

  if (inherits(fit_obj, "vglm")) {
    b <- stats::coef(fit_obj)

    V <- tryCatch(stats::vcov(fit_obj), error = function(e) NULL)
    if (is.null(V)) stopf("Failed to compute vcov() for VGAM fit; cannot produce standard errors.")
    if (is.null(colnames(V))) stopf("vcov(VGAM) returned no colnames; cannot align SEs to coefficients.")

    se_all <- sqrt(diag(V))
    names(se_all) <- colnames(V)

    se <- se_all[names(b)]
    if (any(is.na(se))) {
      missing <- names(b)[is.na(se)]
      stopf(
        "Internal error: missing SE for %d VGAM coefficient(s). Example: %s",
        length(missing),
        paste(head(missing, 5), collapse = ", ")
      )
    }

    nm <- names(b)

    parse_vgam_eq_and_term <- function(nm_vec) {
      eq <- rep(NA_integer_, length(nm_vec))
      term <- rep(NA_character_, length(nm_vec))

      m_a <- regexec("^(.*):([0-9]+)$", nm_vec)
      r_a <- regmatches(nm_vec, m_a)
      has_a <- lengths(r_a) == 3
      if (any(has_a)) {
        term[has_a] <- vapply(r_a[has_a], function(x) x[[2]], character(1))
        eq[has_a] <- suppressWarnings(as.integer(vapply(r_a[has_a], function(x) x[[3]], character(1))))
      }

      need_b <- !has_a
      if (any(need_b)) {
        nm_b <- nm_vec[need_b]
        eq_b <- suppressWarnings(as.integer(sub("^.*mu\\[,([0-9]+)\\].*$", "\\1", nm_b)))
        term_b <- sub("^.*\\):", "", nm_b)
        eq[need_b] <- eq_b
        term[need_b] <- term_b
      }

      term <- ifelse(term == "(Intercept)", "Intercept", term)
      list(eq = eq, term = term)
    }

    parsed <- parse_vgam_eq_and_term(nm)
    eq <- parsed$eq
    term <- parsed$term

    if (any(is.na(eq))) {
      stopf(
        "Failed to parse VGAM equation indices from coefficient names. Example names: %s",
        paste(head(nm, 10), collapse = " | ")
      )
    }
    if (max(eq, na.rm = TRUE) > length(nonbase_clusters)) {
      stopf("Internal error: VGAM eq index exceeds nonbaseline cluster count.")
    }

    cluster <- as.integer(nonbase_clusters[eq])
    z <- as.numeric(b) / as.numeric(se)
    p <- 2 * pnorm(-abs(z))

    return(tibble::tibble(
      eq = as.integer(eq),
      cluster = as.integer(cluster),
      predictor = as.character(term),
      coefficient = as.numeric(b),
      std_err = as.numeric(se),
      z_stat = as.numeric(z),
      p_value = as.numeric(p)
    ))
  }

  stopf("Unknown fit object type for coefficient extraction.")
}

res_tbl <- extract_coef_table(fit, nonbase_clusters)

res_tbl <- res_tbl %>%
  dplyr::group_by(cluster) %>%
  dplyr::mutate(q_value = p.adjust(p_value, method = "BH")) %>%
  dplyr::ungroup() %>%
  dplyr::mutate(
    r_squared = as.numeric(pseudo_r2),
    nobs = as.integer(nrow(bg_model)),
    baseline_cluster = as.integer(baseline_cluster)
  ) %>%
  dplyr::select(
    cluster, predictor, coefficient, std_err, z_stat, p_value, q_value, r_squared, nobs, baseline_cluster
  )


# -----------------------------
# QC + metadata outputs
# -----------------------------
household_day_rows_modeled <- sum(bg_model$total_household_days, na.rm = TRUE)

# New omitted predictors artifact (reference omissions + rank drops)
omitted_tbl <- tibble::tibble(
  predictor = c(omit_refs_present, dropped_predictors_rank),
  reason = c(
    rep("omitted_reference_share", length(omit_refs_present)),
    rep("dropped_rank_deficient", length(dropped_predictors_rank))
  )
) %>%
  dplyr::mutate(
    caveat = dplyr::case_when(
      reason == "omitted_reference_share" ~ REFERENCE_CAVEAT,
      TRUE ~ "Dropped because the fitted design matrix was not full rank (VGAM full-rank requirement)."
    )
  )

# Add group metadata for reference omissions
if (nrow(omitted_tbl) > 0) {
  ref_map <- do.call(rbind.data.frame, lapply(REFERENCE_GROUPS, function(g) {
    data.frame(
      predictor = g$reference,
      group_id = g$group_id,
      group_name = g$group_name,
      implied_definition = implied_definitions[[which(vapply(implied_definitions, function(x) x$group_id, character(1)) == g$group_id)[1]]]$implied_as,
      rationale = g$rationale,
      stringsAsFactors = FALSE
    )
  }))
  omitted_tbl <- omitted_tbl %>%
    dplyr::left_join(ref_map, by = "predictor")
}

input_qc <- list(
  inputs = list(
    clusters = CLUSTERS_PATH,
    crosswalk = CROSSWALK_PATH,
    census_predictors = CENSUS_PATH
  ),
  notes = list(
    "Predictors are inferred from the census parquet columns (numeric/logical), excluding GEOID/NAME.",
    "Counts are computed memory-safely using Arrow Dataset aggregation; the full household-day parquet is not read into RAM.",
    "Zero counts are expected in BG×cluster composition; multinomial likelihood handles zeros naturally (no smoothing/alpha).",
    "Reference-share variables (compositional groups) are omitted deterministically when present to ensure identifiability.",
    "If VGAM is used, remaining rank-deficient predictors are dropped deterministically to satisfy full-rank design requirement."
  ),
  reference_categories = list(
    reference_groups = REFERENCE_GROUPS,
    reference_variables = REFERENCE_VARS,
    reference_variables_present_in_input = omit_refs_present,
    implied_definitions = implied_definitions,
    caveat = REFERENCE_CAVEAT
  ),
  counts = list(
    household_day_rows_total_after_basic_filter = as.integer(household_day_rows_total),
    household_day_rows_dropped_missing_crosswalk = as.integer(dropped_missing_crosswalk),
    household_day_rows_after_crosswalk = as.integer(household_day_rows_after_crosswalk),
    blockgroups_after_min_obs = as.integer(nrow(bg_wide)),
    blockgroups_dropped_missing_predictors = as.integer(drop_missing_pred_bg),
    household_day_rows_dropped_missing_predictors = as.integer(drop_missing_pred_hhday),
    blockgroups_final_complete_case = as.integer(nrow(bg_model)),
    household_days_modeled = as.integer(household_day_rows_modeled)
  ),
  inferred_predictors = list(
    geoid_column = CENSUS_GEOID_COL,
    predictors_inferred = inf$predictors,
    predictors_used = PREDICTORS,
    predictors_omitted_reference_share = omit_refs_present,
    predictors_dropped_all_na = DROPPED_ALL_NA,
    predictors_dropped_rank_deficient = dropped_predictors_rank,
    predictors_zero_variance_not_scaled = zero_var_predictors
  ),
  model = list(
    clusters_observed = as.integer(clusters_observed),
    baseline_cluster = as.integer(baseline_cluster),
    min_obs_per_bg = as.integer(MIN_OBS_PER_BG),
    allow_missing_predictors = as.integer(ALLOW_MISSING_PREDICTORS),
    standardized = as.logical(STANDARDIZE == 1L),
    scaling_parameters = if (STANDARDIZE == 1L) scaling_info else NULL,
    use_vgam = as.logical(USE_VGAM == 1L),
    design_rank = if (USE_VGAM == 1L) as.integer(design_rank) else NULL,
    design_ncol_checked = if (USE_VGAM == 1L) as.integer(design_ncol) else NULL,
    zero_count_statistics = zero_stats
  )
)

n_bg <- nrow(bg_model)
n_hhday <- sum(bg_model$total_household_days, na.rm = TRUE)
n_params <- length(stats::coef(fit))
df_residual <- n_bg - n_params
aic_full <- tryCatch(as.numeric(stats::AIC(fit)), error = function(e) NA_real_)
bic_full <- tryCatch(as.numeric(stats::BIC(fit)), error = function(e) NA_real_)

cluster_totals <- sapply(resp_cols_ordered, function(cn) sum(bg_model[[cn]], na.rm = TRUE))
cluster_props <- cluster_totals / sum(cluster_totals)

diag <- list(
  fit = list(
    converged = convergence_ok,
    convergence_message = convergence_message,
    iter_used = if (is.na(iter_used)) NULL else as.integer(iter_used),
    deviance_full = as.numeric(deviance_full),
    deviance_null = as.numeric(deviance_null),
    deviance_ratio = as.numeric(deviance_ratio),
    logLik_full = ll_full,
    logLik_null = ll_null,
    pseudo_r2_mcfadden = pseudo_r2,
    aic = aic_full,
    bic = bic_full,
    n_parameters = as.integer(n_params),
    df_residual = as.integer(df_residual),
    nobs_blockgroups = as.integer(n_bg),
    nobs_household_days = as.integer(n_hhday),
    clusters_observed = as.integer(clusters_observed),
    baseline_cluster = as.integer(baseline_cluster)
  ),
  reference_categories = list(
    reference_groups = REFERENCE_GROUPS,
    reference_variables = REFERENCE_VARS,
    reference_variables_present_in_input = omit_refs_present,
    implied_definitions = implied_definitions,
    caveat = REFERENCE_CAVEAT
  ),
  omitted_predictors_summary = list(
    n_omitted_reference_share = as.integer(length(omit_refs_present)),
    n_dropped_rank_deficient = as.integer(length(dropped_predictors_rank))
  ),
  cluster_marginal_distributions = lapply(names(cluster_totals), function(cn) {
    k <- as.integer(sub("^cluster_", "", cn))
    list(cluster = k, household_days = as.integer(cluster_totals[[cn]]), proportion = as.numeric(cluster_props[[cn]]))
  })
)

manifest <- list(
  timestamp_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  outputs = list(
    regression_results = file.path(OUT_DIR, "regression_results.parquet"),
    regression_diagnostics = file.path(OUT_DIR, "regression_diagnostics.json"),
    stage2_input_qc = file.path(OUT_DIR, "stage2_input_qc.json"),
    regression_data_blockgroups_wide = file.path(OUT_DIR, "regression_data_blockgroups_wide.parquet"),
    omitted_predictors = file.path(OUT_DIR, "omitted_predictors.parquet"),
    stage2_manifest = file.path(OUT_DIR, "stage2_manifest.json"),
    stage2_metadata = file.path(OUT_DIR, "stage2_metadata.json")
  )
)

logi("%s Writing outputs to: %s", IC$ok, OUT_DIR)

arrow::write_parquet(res_tbl, file.path(OUT_DIR, "regression_results.parquet"))
arrow::write_parquet(bg_model, file.path(OUT_DIR, "regression_data_blockgroups_wide.parquet"))
arrow::write_parquet(omitted_tbl, file.path(OUT_DIR, "omitted_predictors.parquet"))
safe_write_json(diag, file.path(OUT_DIR, "regression_diagnostics.json"))
safe_write_json(input_qc, file.path(OUT_DIR, "stage2_input_qc.json"))
safe_write_json(manifest, file.path(OUT_DIR, "stage2_manifest.json"))

t_total_end <- Sys.time()
runtime_total_seconds <- as.numeric(difftime(t_total_end, t_total_start, units = "secs"))

metadata <- list(
  timestamp_utc = manifest$timestamp_utc,
  runtime_seconds_total = as.numeric(runtime_total_seconds),
  runtime_seconds_fit = as.numeric(fit_duration),
  package_versions = list(
    R = as.character(getRversion()),
    arrow = as.character(utils::packageVersion("arrow")),
    dplyr = as.character(utils::packageVersion("dplyr")),
    nnet = as.character(utils::packageVersion("nnet")),
    jsonlite = as.character(utils::packageVersion("jsonlite")),
    VGAM = if (USE_VGAM == 1L) as.character(utils::packageVersion("VGAM")) else NULL
  )
)
safe_write_json(metadata, file.path(OUT_DIR, "stage2_metadata.json"))

cat("\n")
cat(sprintf("%s Stage 2 multinomial logit complete.\n", IC$ok))
cat(sprintf("  Block Groups (modeled): %d\n", nrow(bg_model)))
cat(sprintf("  Household-days (modeled): %s\n", format(household_day_rows_modeled, big.mark = ",")))
cat(sprintf("  Predictors (used): %d\n", length(PREDICTORS)))
cat(sprintf("  Omitted reference predictors (present): %d\n", length(omit_refs_present)))
cat(sprintf("  Rank-deficient predictors dropped: %d\n", length(dropped_predictors_rank)))
cat(sprintf("  Pseudo R^2 (McFadden): %.4f\n", pseudo_r2))
cat(sprintf("  Converged: %s\n", ifelse(convergence_ok, "true", "false")))
cat("\n")

quit(status = 0)
