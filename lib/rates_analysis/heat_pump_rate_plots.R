library(lubridate)
library(patchwork)
library(tidyverse)
library(ggrepel)

hist_for_single_rate_version <- function(
  annual_change_table,
  bill_type,
  version_elec,
  baseline_heating_type_option,
  hvac_option,
  supply_year,
  season = 'Annual',
  title = "auto",
  second_subtitle = NULL,
  x_limits = c(-3000, 3000),
  y_limits = c(0, 1000),
  binwidth = 100,
  show_category_labels = TRUE
) {
  # --------------------------------------
  # Plot parameters
  x_min <- x_limits[1]
  x_max <- x_limits[2]

  # Define the breaks for binning
  breaks <- seq(
    floor(x_min / binwidth) * binwidth,
    ceiling(x_max / binwidth) * binwidth,
    by = binwidth
  )

  # More balanced color palette with less contrast
  dark_red <- "#db8b87" # dark red
  light_red <- "#eaada9" # light red
  light_green <- "#8ebd85" # light green
  dark_green <- "#5e8a5e" # dark green
  # --------------------------------------

  # Filter data for mid_hp
  plot_data <- annual_change_table |>
    filter(hvac == hvac_option, version_elec == !!version_elec) |>
    filter(
      if (baseline_heating_type_option != "All Fuels") {
        baseline_heating_type == !!baseline_heating_type_option
      } else {
        TRUE
      }
    ) |>
    # Add category column before binning with new cutoffs
    mutate(
      bill_category = case_when(
        annual_change_total <= -1000 ~ "large_savings",
        annual_change_total < 0 ~ "small_savings",
        annual_change_total < 1000 ~ "small_increase",
        TRUE ~ "large_increase"
      )
    )

  # Calculate percentages for each category
  category_percentages <- plot_data |>
    group_by(bill_category) |>
    summarise(count = n(), .groups = "drop") |>
    mutate(percentage = round(count / sum(count) * 100, 1))

  # Add a row for 'pct_that_save' as the sum of 'small_savings' and 'large_savings'
  category_percentages <- category_percentages |>
    bind_rows(
      tibble(
        bill_category = "pct_that_save",
        count = sum(category_percentages$count[
          category_percentages$bill_category %in%
            c("small_savings", "large_savings")
        ]),
        percentage = sum(category_percentages$percentage[
          category_percentages$bill_category %in%
            c("small_savings", "large_savings")
        ])
      )
    ) |>
    bind_rows(
      tibble(
        bill_category = "pct_that_lose",
        count = sum(category_percentages$count[
          category_percentages$bill_category %in%
            c("small_increase", "large_increase")
        ]),
        percentage = sum(category_percentages$percentage[
          category_percentages$bill_category %in%
            c("small_increase", "large_increase")
        ])
      )
    )

  # Get percentages for annotation
  pct_large_savings <- category_percentages |>
    filter(bill_category == "large_savings") |>
    pull(percentage) |>
    round(1)
  if (length(pct_large_savings) == 0) {
    pct_large_savings <- 0
  }

  pct_small_savings <- category_percentages |>
    filter(bill_category == "small_savings") |>
    pull(percentage) |>
    round(1)
  if (length(pct_small_savings) == 0) {
    pct_small_savings <- 0
  }

  pct_that_save <- pct_large_savings + pct_small_savings

  pct_small_increase <- category_percentages |>
    filter(bill_category == "small_increase") |>
    pull(percentage) |>
    round(1)
  if (length(pct_small_increase) == 0) {
    pct_small_increase <- 0
  }

  pct_large_increase <- category_percentages |>
    filter(bill_category == "large_increase") |>
    pull(percentage) |>
    round(1)
  if (length(pct_large_increase) == 0) {
    pct_large_increase <- 0
  }

  pct_that_lose <- pct_small_increase + pct_large_increase

  # Manually bin the data and calculate counts - this preserves the bill_category
  binned_data <- plot_data |>
    mutate(
      bin = cut(
        annual_change_total,
        breaks = breaks,
        include.lowest = TRUE,
        right = FALSE
      )
    ) |>
    group_by(bin, bill_category) |>
    summarise(count = n(), .groups = "drop") |>
    # Add bin midpoint for plotting
    mutate(
      bin_mid = as.numeric(as.character(
        sapply(bin, function(b) {
          # Extract bin boundaries and calculate midpoint
          vals <- as.numeric(gsub(
            "\\(|\\]|\\[|\\)",
            "",
            strsplit(as.character(b), ",")[[1]]
          ))
          return(mean(vals))
        })
      ))
    )

  # Position the bars at the very top and text just below
  bar_y_pos <- y_limits[2] # Exactly at the top
  text_y_pos <- y_limits[2] * 0.95 # Just below the colored line

  # Calculate x positions for percentage labels (center of each category)
  x_large_savings <- mean(c(x_min, -1000))
  x_small_savings <- mean(c(-1000, 0))
  x_small_increase <- mean(c(0, 1000))
  x_large_increase <- mean(c(1000, x_max))

  # Create the plot using geom_col() with binned data
  px_hist_by_rate_version <- ggplot(
    binned_data,
    aes(x = bin_mid, y = count, fill = bill_category)
  ) +
    # Add vertical dotted lines at the category boundaries
    geom_vline(
      xintercept = c(-1000, 0, 1000),
      linetype = "dotted",
      color = "gray50",
      size = 0.5
    ) +

    # Add the bars
    geom_col(position = "stack", width = binwidth * 0.9) +

    # Set the fill colors manually with expanded categories
    scale_fill_manual(
      values = c(
        "large_savings" = "#5e8a5e", # dark green
        "small_savings" = "#8ebd85", # light green
        "small_increase" = "#eaada9", # light red
        "large_increase" = "#db8b87" # dark red
      ),
      guide = "none"
    ) +

    # Add horizontal lines at the very top of the plot
    # With colors matching their respective categories
    annotate(
      "segment",
      x = x_min,
      xend = -1000,
      y = bar_y_pos,
      yend = bar_y_pos,
      color = dark_green,
      size = 1
    ) +

    annotate(
      "segment",
      x = -1000,
      xend = 0,
      y = bar_y_pos,
      yend = bar_y_pos,
      color = light_green,
      size = 1
    ) +

    annotate(
      "segment",
      x = 0,
      xend = 1000,
      y = bar_y_pos,
      yend = bar_y_pos,
      color = light_red,
      size = 1
    ) +

    annotate(
      "segment",
      x = 1000,
      xend = x_max,
      y = bar_y_pos,
      yend = bar_y_pos,
      color = dark_red,
      size = 1
    ) +

    # Add percentage labels for each category directly below the lines
    annotate(
      "text",
      x = x_large_savings,
      y = text_y_pos * 0.95,
      label = paste0(pct_large_savings, "%"),
      size = 3,
      fontface = "bold"
    ) +
    annotate(
      "text",
      x = x_small_savings,
      y = text_y_pos * 0.95,
      label = paste0(pct_small_savings, "%"),
      size = 3,
      fontface = "bold"
    ) +
    annotate(
      "text",
      x = x_small_increase,
      y = text_y_pos * 0.95,
      label = paste0(pct_small_increase, "%"),
      size = 3,
      fontface = "bold"
    ) +
    annotate(
      "text",
      x = x_large_increase,
      y = text_y_pos * 0.95,
      label = paste0(pct_large_increase, "%"),
      size = 3,
      fontface = "bold"
    ) +

    # Rest of the plot styling
    labs(
      title = if (title == "auto") {
        title = case_when(
          bill_type == "annual_change_total" ~ glue::glue(
            "Change in Total {season} Energy Bills after Switching to Heat Pump from {baseline_heating_type_option}"
          ),
          bill_type == "annual_change_elec" ~ glue::glue(
            "Switching from {baseline_heating_type_option} to Heat Pump: Change in {season} Electric Bill"
          ),
          bill_type == "annual_change_gas" ~ glue::glue(
            "Switching from {baseline_heating_type_option} to Heat Pump: Change in {season} Gas Bill"
          ),
          bill_type == "annual_change_fuel_oil" ~ glue::glue(
            "Switching from {baseline_heating_type_option} to Heat Pump: Change in {season} Fuel Oil Bill"
          ),
          bill_type == "annual_change_propane" ~ glue::glue(
            "Switching from {baseline_heating_type_option} to Heat Pump: Change in {season} Propane Bill"
          ),
          TRUE ~ "Heat Pump Rate Bill Changes" # Default case
        )
      } else if (title == "No Title") {
        title = NULL
      } else {
        title = title
      },
      x = glue::glue("{season} Bill Change ($)"),
      y = "# of Homes"
    ) +
    scale_y_continuous(
      labels = function(x) paste0(round(x * 242.13 * 0.001), "k"),
      name = "# of Homes",
      limits = y_limits,
      breaks = seq(0, y_limits[2] * 242.13, by = 20000) / 242.13
    ) +
    scale_x_continuous(
      labels = scales::dollar_format(),
      limits = x_limits,
      breaks = seq(x_min, x_max, by = 500)
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),
      panel.grid.major = element_line(linewidth = 0.2),
      panel.grid.minor = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none",
      aspect.ratio = 0.6
    ) +
    {
      if (show_category_labels) {
        list(
          annotate(
            "text",
            x = x_large_savings,
            y = y_limits[2] * 0.97,
            label = "savings > $1k",
            size = 3,
            fontface = "bold"
          ),
          annotate(
            "text",
            x = x_small_savings,
            y = y_limits[2] * 0.97,
            label = "savings $0–1k",
            size = 3,
            fontface = "bold"
          ), # make bold
          annotate(
            "text",
            x = x_small_increase,
            y = y_limits[2] * 0.97,
            label = "losses  $0–1k",
            size = 3,
            fontface = "bold"
          ), # make bold
          annotate(
            "text",
            x = x_large_increase,
            y = y_limits[2] * 0.97,
            label = "losses  > $1k",
            size = 3,
            fontface = "bold"
          )
        )
      }
    } +
    coord_cartesian(clip = "off")

  return(list(px_hist_by_rate_version, category_percentages))
}


plot_bill_change_histograms <- function(
  annual_change_table,
  bill_type,
  baseline_heating_type_option,
  season = 'Annual',
  version_elec = c("baseline", "baseline", "baseline"),
  hvac_option = c("hp_low", "hp_mid", "hp_best"),
  supply_year = 2024,
  second_subtitle = NULL,
  x_limits = c(-2500, 2500),
  y_limits = c(0, 100000) / 242.13,
  binwidth = 100
) {
  # Create nicer version labels
  version_labels <- c(
    "hp_low" = "HSPF 9.2 - Energy Star Minimum",
    "hp_high" = "HSPF 11 - Minimum for Climate Zone 5",
    "hp_best" = "HSPF 13 - Best Available"
  )

  # Create three histograms without individual titles

  # Plot 1: hp_low
  # -----------------------------------------------
  result_1 <- hist_for_single_rate_version(
    annual_change_table = annual_change_table,
    bill_type = bill_type,
    version_elec = version_elec[1],
    baseline_heating_type_option = baseline_heating_type_option,
    hvac_option = hvac_option[1],
    supply_year = supply_year,
    title = 'auto',
    x_limits = x_limits,
    y_limits = y_limits,
    binwidth = binwidth,
    show_category_labels = FALSE
  )

  p1 <- result_1[[1]] +
    labs(title = NULL) + # Remove title
    ylab(NULL) + # Remove y-axis label
    theme(
      axis.title.x = element_blank(), # Remove x axis title from top two plots
      axis.text.x = element_blank(), # Remove x tick labels from top two plots
      axis.ticks.x = element_blank(), # Remove x ticks from top two plots
      plot.margin = margin(5, 5, 0, 5), # Consistent margins (top, right, bottom, left)
      axis.title.y = element_blank(), # Remove y title from top plot
      aspect.ratio = 0.18
    ) +
    # Add elegant version label on right side
    annotate(
      "text",
      x = x_limits[2] * 0.95,
      y = y_limits[2] * 0.7,
      label = glue::glue(
        "{version_labels[hvac_option[1]]}\n{version_elec[1]} rates"
      ),
      hjust = 1,
      fontface = "bold",
      size = 4,
      color = "#023047"
    )

  # Plot 2: hp_mid
  # -----------------------------------------------
  result_2 <- hist_for_single_rate_version(
    annual_change_table = annual_change_table,
    bill_type = bill_type,
    version_elec = version_elec[2],
    baseline_heating_type_option = baseline_heating_type_option,
    hvac_option = hvac_option[2],
    supply_year = supply_year,
    title = 'auto',
    x_limits = x_limits,
    y_limits = y_limits,
    binwidth = binwidth,
    show_category_labels = FALSE
  )

  p2 <- result_2[[1]] +
    labs(title = NULL) + # Remove title
    ylab(NULL) +
    theme(
      axis.title.x = element_blank(), # Remove x axis title from top two plots
      axis.text.x = element_blank(), # Remove x tick labels from top two plots
      axis.ticks.x = element_blank(), # Remove x ticks from top two plots
      axis.title.y = element_blank(), # Remove y title from middle plot
      plot.margin = margin(0, 5, 0, 5), # Consistent margins (no top/bottom margin)
      aspect.ratio = 0.18
    ) +
    # Add elegant version label on right side
    annotate(
      "text",
      x = x_limits[2] * 0.95,
      y = y_limits[2] * 0.7,
      label = glue::glue(
        "{version_labels[hvac_option[2]]}\n{version_elec[2]} rates"
      ),
      hjust = 1,
      fontface = "bold",
      size = 4,
      color = "#FC9706"
    )

  # Plot 3: hp_best
  # -----------------------------------------------
  result_3 <- hist_for_single_rate_version(
    annual_change_table = annual_change_table,
    bill_type = bill_type,
    version_elec = version_elec[3],
    baseline_heating_type_option = baseline_heating_type_option,
    hvac_option = hvac_option[3],
    supply_year = supply_year,
    title = 'auto',
    x_limits = x_limits,
    y_limits = y_limits,
    binwidth = binwidth,
    show_category_labels = FALSE
  )

  p3 <- result_3[[1]] +
    labs(title = NULL) + # Remove title
    ylab(NULL) +
    theme(
      axis.title.y = element_blank(), # Remove y title from bottom plot
      plot.margin = margin(0, 5, 5, 5), # Consistent margins
      axis.title.x = element_text(size = 9),
      aspect.ratio = 0.18
    ) +
    # Add elegant version label on right side
    annotate(
      "text",
      x = x_limits[2] * 0.95,
      y = y_limits[2] * 0.7,
      label = glue::glue(
        "{version_labels[hvac_option[3]]}\n{version_elec[3]} rates"
      ),
      hjust = 1,
      fontface = "bold",
      size = 4,
      color = "#68BED8"
    )

  # Combine plots with shared x and y axes
  combined_plot <- p1 /
    p2 /
    p3 +
    plot_layout(heights = c(1, 1, 1)) + # Equal heights for all plots
    plot_annotation(
      title = case_when(
        bill_type == "annual_change_total" ~ glue::glue(
          "Change in Total Annual Energy Bills after Switching to Heat Pump from {baseline_heating_type_option}"
        ),
        bill_type == "annual_change_elec" ~ glue::glue(
          "Switching from {baseline_heating_type_option} to Heat Pump: Change in Annual Electric Bill"
        ),
        bill_type == "annual_change_gas" ~ glue::glue(
          "Switching from {baseline_heating_type_option} to Heat Pump: Change in Annual Gas Bill"
        ),
        bill_type == "annual_change_fuel_oil" ~ glue::glue(
          "Switching from {baseline_heating_type_option} to Heat Pump: Change in Annual Fuel Oil Bill"
        ),
        bill_type == "annual_change_propane" ~ glue::glue(
          "Switching from {baseline_heating_type_option} to Heat Pump: Change in Annual Propane Bill"
        ),
        TRUE ~ "Heat Pump Rate Bill Changes" # Default case
      ),
      subtitle = second_subtitle,
      theme = theme(
        plot.margin = margin(10, 25, 10, 10), # Add more right margin for labels
        plot.title = element_text(hjust = 0.5, size = 10, face = "bold"),
        plot.subtitle = element_text(hjust = 0.5, size = 8),
      )
    ) +
    # Apply consistent theming to all plots
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_line(linewidth = 0.2, color = "gray90"),
      panel.grid.major.y = element_line(linewidth = 0.2, color = "gray90"),
    )

  # Wrap the combined plot to add a shared y-axis label
  combined_plot <- patchwork::wrap_elements(combined_plot) +
    labs(tag = "# of Homes") +
    theme(
      plot.tag = element_text(
        angle = 90,
        vjust = 0.5,
        hjust = 0.5,
        face = "plain",
        size = 9
      ),
      plot.tag.position = "left"
    )

  return(list(combined_plot, result_1[[2]], result_2[[2]], result_3[[2]]))
}


plot_energy_burden_histogram_standalone <- function(
  data,
  name,
  x_limits = c(0, 1.0),
  y_limits = c(0, 50000 / 242.13),
  binwidth = 0.01,
  show_x_axis = TRUE,
  show_y_axis = TRUE
) {
  low_burden_threshold <- 0.04
  moderate_burden_threshold <- 0.06
  high_burden_threshold <- 0.12

  # Calculate bill categories
  data <- data |>
    mutate(
      burden_group = case_when(
        burden_total < low_burden_threshold ~ "low_energy_burden",
        burden_total < moderate_burden_threshold ~ "moderate_energy_burden",
        burden_total < high_burden_threshold ~ "high_energy_burden",
        TRUE ~ "very_high_energy_burden"
      )
    )

  # Calculate percentages for each category
  category_percentages <- data |>
    group_by(burden_group) |>
    summarise(count = n(), .groups = "drop") |>
    mutate(percentage = count / sum(count) * 100)

  # Get percentages for annotation
  pct_low_energy_burden <- category_percentages |>
    filter(burden_group == "low_energy_burden") |>
    pull(percentage) |>
    round(1)

  pct_moderate_energy_burden <- category_percentages |>
    filter(burden_group == "moderate_energy_burden") |>
    pull(percentage) |>
    round(1)

  pct_high_energy_burden <- category_percentages |>
    filter(burden_group == "high_energy_burden") |>
    pull(percentage) |>
    round(1)

  pct_very_high_energy_burden <- category_percentages |>
    filter(burden_group == "very_high_energy_burden") |>
    pull(percentage) |>
    round(1)

  # Calculate bin breaks and midpoints
  breaks <- seq(
    floor(x_limits[1] / binwidth) * binwidth,
    ceiling(x_limits[2] / binwidth) * binwidth,
    by = binwidth
  )

  # Manually bin the data
  binned_data <- data |>
    mutate(
      bin = cut(
        burden_total,
        breaks = breaks,
        include.lowest = TRUE,
        right = FALSE
      )
    ) |>
    group_by(bin, burden_group) |>
    summarise(count = n(), .groups = "drop") |>
    mutate(
      bin_mid = as.numeric(as.character(
        sapply(bin, function(b) {
          vals <- as.numeric(gsub(
            "\\(|\\]|\\[|\\)",
            "",
            strsplit(as.character(b), ",")[[1]]
          ))
          return(mean(vals))
        })
      ))
    )

  # Position the bars at the top and text just below
  bar_y_pos <- y_limits[2]
  text_y_pos <- y_limits[2] * 0.90

  # Calculate x positions for percentage labels
  x_low_energy_burden <- mean(c(0, low_burden_threshold - 0.002))
  x_moderate_energy_burden <- mean(c(
    low_burden_threshold,
    moderate_burden_threshold
  ))
  x_high_energy_burden <- mean(c(
    moderate_burden_threshold,
    high_burden_threshold
  ))
  x_very_high_energy_burden <- mean(c(high_burden_threshold, x_limits[2]))

  # Create the plot
  plot <- ggplot(
    binned_data,
    aes(x = bin_mid, y = count, fill = burden_group)
  ) +
    # Add vertical dotted lines at the category boundaries
    geom_vline(
      xintercept = c(
        low_burden_threshold,
        moderate_burden_threshold,
        high_burden_threshold
      ),
      linetype = "dotted",
      color = "gray50",
      size = 0.5
    ) +
    # Add bars
    geom_col(position = "stack", width = binwidth * 0.9) +

    # Set the fill colors for the bars
    scale_fill_manual(
      values = energy_burden_colors,
      guide = "none"
    ) +
    # Add vertical lines at the category boundaries
    annotate(
      "segment",
      x = x_limits[1],
      xend = low_burden_threshold,
      y = bar_y_pos,
      yend = bar_y_pos,
      color = energy_burden_colors["low_energy_burden"],
      size = 1
    ) +
    annotate(
      "segment",
      x = low_burden_threshold,
      xend = moderate_burden_threshold,
      y = bar_y_pos,
      yend = bar_y_pos,
      color = energy_burden_colors["moderate_energy_burden"],
      size = 1
    ) +
    annotate(
      "segment",
      x = moderate_burden_threshold,
      xend = high_burden_threshold,
      y = bar_y_pos,
      yend = bar_y_pos,
      color = energy_burden_colors["high_energy_burden"],
      size = 1
    ) +
    annotate(
      "segment",
      x = high_burden_threshold,
      xend = x_limits[2],
      y = bar_y_pos,
      yend = bar_y_pos,
      color = energy_burden_colors["very_high_energy_burden"],
      size = 1
    ) +
    annotate(
      "text",
      x = x_low_energy_burden,
      y = text_y_pos * 0.95,
      label = paste0(pct_low_energy_burden, "%"),
      size = 3,
      fontface = "bold"
    ) +
    annotate(
      "text",
      x = x_moderate_energy_burden,
      y = text_y_pos * 0.95,
      label = paste0(pct_moderate_energy_burden, "%"),
      size = 3,
      fontface = "bold"
    ) +
    annotate(
      "text",
      x = x_high_energy_burden,
      y = text_y_pos * 0.95,
      label = paste0(pct_high_energy_burden, "%"),
      size = 3,
      fontface = "bold"
    ) +
    annotate(
      "text",
      x = x_very_high_energy_burden,
      y = text_y_pos * 0.95,
      label = paste0(pct_very_high_energy_burden, "%"),
      size = 3,
      fontface = "bold"
    ) +
    scale_y_continuous(
      labels = function(x) paste0(round(x * 242.13 * 0.001), "k"),
      #name = "# of Homes",
      limits = y_limits,
      breaks = c(
        0,
        25000,
        50000,
        75000,
        100000,
        125000,
        150000,
        175000,
        200000,
        225000,
        250000
      ) /
        242.13
    ) +
    scale_x_continuous(
      labels = scales::percent_format(accuracy = 1),
      name = "Annual Energy Burden (All fuels, no transportation costs)",
      limits = x_limits,
      breaks = c(
        low_burden_threshold,
        moderate_burden_threshold,
        high_burden_threshold
      ),
    ) +
    theme_minimal() +
    theme(
      panel.grid.major = element_line(linewidth = 0.2),
      panel.grid.minor = element_blank(),
      aspect.ratio = 0.18,
      axis.title.x = element_text(size = 8)
    )

  # Remove x-axis elements if requested
  if (!show_x_axis) {
    plot <- plot +
      theme(
        axis.title.x = element_blank(),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank()
      )
  }

  # Remove y-axis elements if requested
  if (!show_y_axis) {
    plot <- plot +
      theme(
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank()
      )
  }

  return(list(plot = plot, category_percentages = category_percentages))
}
