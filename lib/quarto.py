"""Helpers for Quarto Manuscript rendering."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def display_svg(fig: Figure) -> None:
    """Render a matplotlib figure as SVG and display it via IPython.

    Quarto Manuscript's ``{{< embed >}}`` breaks on cells that produce
    multi-MIME output (e.g. a raw matplotlib Figure emits both PNG and
    text/plain).  Finishing the cell with a single ``IPython.display.SVG``
    object avoids this by producing only an ``image/svg+xml`` MIME type.

    After saving to SVG the figure is closed to free memory.
    """
    import matplotlib.pyplot as plt
    from IPython.display import SVG, display

    fig.savefig(buf := io.BytesIO(), format="svg", bbox_inches="tight")
    plt.close(fig)
    display(SVG(data=buf.getvalue()))
