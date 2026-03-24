"""Switchbox brand theme for plotnine.

Applies Switchbox brand fonts, colors, and sizes to all chart text
elements, producing a consistent visual hierarchy across reports.

Typography guide
----------------
- **Title** (GT Planar Bold 15pt black): chart headline, echoes H3 headings.
- **Labeling layer** (GT Planar 13pt #333333): subtitle, axis titles, strip
  (facet) labels, legend title.  All share the same font/size/color.
- **Data-reference layer** (IBM Plex Sans 11pt #4D4D4D): axis tick labels,
  legend text — smallest text, for reading values off axes.

Usage::

    from lib.plotnine import theme_switchbox, SB_COLORS

    (
        ggplot(df, aes("x", "y"))
        + geom_col(fill=SB_COLORS["sky"])
        + theme_switchbox()
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
from matplotlib import font_manager
from plotnine import element_line, element_rect, element_text, theme
from plotnine.themes.theme_minimal import theme_minimal

_MarginKey = Literal["t", "b", "l", "r", "unit"]

_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "reports" / ".style" / "fonts"
_FONT_IBM_PLEX = "IBM Plex Sans"
_FONT_GT_PLANAR = "GT Planar"
_FONT_FARNHAM = "Farnham Text"
_FONTS_REGISTERED = False

_FONT_FILES: list[str] = [
    "ips-regular.otf",
    "ips-bold.otf",
    "gtp-regular.otf",
    "gtp-bold.otf",
    "gtp-black.otf",
    "ft-regular.otf",
    "ft-bold.otf",
]


def _register_fonts() -> None:
    """Register all brand fonts with matplotlib (idempotent)."""
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return

    for filename in _FONT_FILES:
        path = _FONT_DIR / filename
        if path.exists():
            font_manager.fontManager.addfont(str(path))

    mpl.rcParams["svg.fonttype"] = "none"
    _FONTS_REGISTERED = True


SB_COLORS: dict[str, str] = {
    "sky": "#68BED8",
    "midnight": "#023047",
    "carrot": "#FC9706",
    "saffron": "#FFC729",
    "pistachio": "#A0AF12",
    "black": "#000000",
    "white": "#FFFFFF",
    "midnight_text": "#0B6082",
    "pistachio_text": "#546800",
}


_COLOR_LABEL = "#333333"
_COLOR_DATA = "#4D4D4D"


class theme_switchbox(theme_minimal):
    """Switchbox brand theme for plotnine.

    Three-tier text hierarchy:

    ====  ====================  ===========================
    Tier  Elements              Spec
    ====  ====================  ===========================
    1     plot_title            GT Planar Bold · 15pt · black
    2     subtitle, axis        GT Planar · 13pt · #333333
          titles, strip text,
          legend title
    3     axis tick labels,     IBM Plex Sans · 11pt · #4D4D4D
          legend text
    ====  ====================  ===========================
    """

    def __init__(self, base_size: int = 11):
        _register_fonts()
        super().__init__(base_size=base_size, base_family=_FONT_IBM_PLEX)
        margin_title: dict[_MarginKey, Any] = {"b": 8, "unit": "pt"}
        margin_x: dict[_MarginKey, Any] = {"t": 8, "unit": "pt"}
        margin_y: dict[_MarginKey, Any] = {"r": 8, "unit": "pt"}
        self += theme(
            panel_background=element_rect(fill="white", color="white"),
            # Tier 1 — title
            plot_title=element_text(
                family=_FONT_GT_PLANAR,
                fontweight="bold",
                size=15,
                color="black",
                margin=margin_title,
            ),
            # Tier 2 — labeling layer
            plot_subtitle=element_text(
                family=_FONT_GT_PLANAR,
                size=13,
                color=_COLOR_LABEL,
            ),
            axis_title_x=element_text(
                family=_FONT_GT_PLANAR,
                size=13,
                color=_COLOR_LABEL,
                margin=margin_x,
            ),
            axis_title_y=element_text(
                family=_FONT_GT_PLANAR,
                size=13,
                color=_COLOR_LABEL,
                margin=margin_y,
            ),
            strip_text=element_text(
                family=_FONT_GT_PLANAR,
                size=13,
                color=_COLOR_LABEL,
            ),
            legend_title=element_text(
                family=_FONT_GT_PLANAR,
                size=13,
                color=_COLOR_LABEL,
            ),
            # Tier 3 — data-reference layer (explicit x/y so all axis tick labels match)
            axis_text=element_text(
                family=_FONT_IBM_PLEX,
                size=11,
                color=_COLOR_DATA,
            ),
            axis_text_x=element_text(
                family=_FONT_IBM_PLEX,
                size=11,
                color=_COLOR_DATA,
            ),
            axis_text_y=element_text(
                family=_FONT_IBM_PLEX,
                size=11,
                color=_COLOR_DATA,
            ),
            legend_text=element_text(
                family=_FONT_IBM_PLEX,
                size=11,
                color=_COLOR_DATA,
            ),
            # Structure
            axis_line=element_line(size=0.5),
            axis_ticks=element_line(color="black"),
        )
