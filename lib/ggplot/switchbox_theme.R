# Load and register IBM Plex Sans font

library(sysfonts)
library(showtext)
library(ggplot2)

# Download fonts to the same directory as this theme file
# Get project root (go up from current working directory to find project root)
project_root <- getwd()
while (!file.exists(file.path(project_root, "lib"))) {
  parent <- dirname(project_root)
  if (parent == project_root) {
    # Reached filesystem root, use current directory as fallback
    project_root <- getwd()
    break
  }
  project_root <- parent
}

theme_dir <- file.path(project_root, "lib", "ggplot")

# Download fonts if they don't exist
regular_path <- file.path(theme_dir, "IBMPlexSans-Regular.otf")
bold_path <- file.path(theme_dir, "IBMPlexSans-Bold.otf")

if (!file.exists(regular_path)) {
  download.file(
    "https://switchbox-data.github.io/reports/fonts/ibm_plex_sans/IBMPlexSans-Regular.otf",
    regular_path,
    mode = "wb"
  )
}

if (!file.exists(bold_path)) {
  download.file(
    "https://switchbox-data.github.io/reports/fonts/ibm_plex_sans/IBMPlexSans-Bold.otf",
    bold_path,
    mode = "wb"
  )
}

font_add(
  family = "IBM-Plex-Sans",
  regular = regular_path,
  bold = bold_path
)
showtext_auto()
showtext::showtext_opts(dpi = 300)
theme_set(theme_minimal())
theme_update(
  panel.background = element_rect(fill = "white", color = "white"),
  legend.title = element_text(hjust = 0.5), # Centers the legend title
  axis.line = element_line(linewidth = 0.5),
  axis.ticks = element_line(color = "black"),
  # panel.grid.minor.x = element_blank(),
  text = element_text(family = "IBM-Plex-Sans", size = 12),
  axis.text = element_text(
    family = "IBM-Plex-Sans",
    size = 12
  ),
  axis.title = element_text(
    family = "IBM-Plex-Sans",
    size = 12
  ),
  strip.text = element_text(
    size = 12, # Font size
    family = "IBM-Plex-Sans",
  ),
  axis.title.x = element_text(margin = margin(t = 3)),
  axis.title.y = element_text(margin = margin(r = 3))
)


sb_colors <- c(
  "sky" = "#68BED8", # primary
  "midnight" = "#023047", # primary
  "carrot" = "#FC9706", # primary
  "saffron" = "#FFC729", # secondary
  "pistachio" = "#A0AF12", # secondary
  "black" = "#000000", # utilitarian
  "white" = "#FFFFFF", # utilitarian
  "midnight_text" = "#0B6082", # lighter, text only
  "pistachio_text" = "#546800" # darker, text only
)
