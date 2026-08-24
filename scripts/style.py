"""One visual language for every figure: palette, plotly template, colormaps.

Rules this file exists to enforce, so no individual figure has to remember them:

* **Categorical colours are assigned in a fixed order and never cycled.** Slots
  1-3 are safe for scatter-type figures (where every pair of series can end up
  adjacent on screen); slots 1-4 are safe for bars and lines. Past that, fold
  into "other" or use small multiples.
* **Colour means one thing per figure.** From notebook 02 onward, *blue is a
  circular feed and orange is a linear feed*, always -- that pairing is the
  spine of the whole series, so nothing else gets those two colours in a figure
  where feeds appear.
* **Magnitude gets a sequential map, signed quantities get a diverging map with
  a neutral midpoint.** Stokes I uses ``afmhot``, the EHT house style; Q, U and
  V are signed and use ``RdBu_r``. Never a rainbow.
* **Identity is never colour alone**: two or more series get a legend, and the
  four Stokes bars are directly labelled with their values.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# --- ink and surface -------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8983"
GRID = "#e6e5e1"
AXIS = "#c9c8c2"

# --- categorical series, in fixed order -----------------------------------
SERIES = ("#2a78d6",  # 1 blue
          "#eb6834",  # 2 orange
          "#1baf7a",  # 3 aqua
          "#eda100",  # 4 yellow
          "#e87ba4",  # 5 magenta
          "#008300",  # 6 green
          "#4a3aa7",  # 7 violet
          "#e34948")  # 8 red

#: Safe for figures where any two series can sit side by side (scatter, uv plots).
SERIES_SCATTER = SERIES[:3]

# --- semantic roles --------------------------------------------------------
FEED_CIRCULAR = SERIES[0]   # blue  -- R/L feeds
FEED_LINEAR = SERIES[1]     # orange -- X/Y feeds
FEED_MIXED = SERIES[6]      # violet -- a baseline pairing the two
STOKES = {"I": SERIES[0], "Q": SERIES[1], "U": SERIES[2], "V": SERIES[3]}
#: The polarization ellipse. Slot 3, so it is distinct from the two field
#: components it is drawn beside (blue E_x, orange E_y) and still inside the
#: three-colour set that is safe when every pair can appear side by side.
ELLIPSE = SERIES[2]     # aqua/green
POL_TICK = "#0b0b0b"        # EVPA ticks: ink, so they read on any brightness map
HIGHLIGHT = SERIES[1]

FONT = ("Source Sans Pro, -apple-system, BlinkMacSystemFont, Segoe UI, "
        "Helvetica, Arial, sans-serif")
MATH_FONT = "Latin Modern Math, Cambria Math, serif"

TEMPLATE_NAME = "csc494"

# Literal unicode, never HTML entities. Plotly parses entities in 2D layout text
# but *not* in 3D trace text, where "&#176;" renders as those seven characters --
# so the whole report uses the real glyphs and the question never arises.
DEG = "°"      # degrees
MU = "µ"       # micro, as in microarcseconds
LAMBDA = "λ"   # wavelength
SIGMA = "σ"
PHI = "φ"
OMEGA = "ω"
UAS = f"{MU}as"


def mpl_colorscale(name: str, n_steps: int = 64) -> list[list]:
    """Convert a matplotlib colormap into a plotly colorscale.

    Lets plotly figures use the perceptual colormaps the radio community
    already reads fluently (``afmhot`` for total intensity) instead of plotly's
    defaults.

    Parameters
    ----------
    name : str
        Any matplotlib colormap name.
    n_steps : int, optional
        Number of sampled steps.

    Returns
    -------
    list
        ``[[position, 'rgb(r,g,b)'], ...]`` suitable for ``colorscale=``.
    """
    cmap = mpl.colormaps[name]
    positions = np.linspace(0.0, 1.0, n_steps)
    scale = []
    for position in positions:
        r, g, b, _ = cmap(position)
        scale.append([float(position),
                      f"rgb({int(255 * r)},{int(255 * g)},{int(255 * b)})"])
    return scale


#: Total intensity (magnitude, one direction only).
CMAP_INTENSITY = "afmhot"
#: Signed quantities: Q, U, V, residuals. Neutral in the middle.
CMAP_DIVERGING = "RdBu_r"
#: Polarization fraction, bounded 0-1.
CMAP_FRACTION = "viridis"


def register_template() -> str:
    """Register the ``csc494`` plotly template and make it the default.

    Recessive axes and grid, generous margins, a single shared font, and the
    categorical palette in fixed order. Idempotent.

    Returns
    -------
    str
        The template name, for ``fig.update_layout(template=...)``.
    """
    axis = {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": True,
        "zerolinecolor": AXIS,
        "zerolinewidth": 1,
        "linecolor": AXIS,
        "ticks": "outside",
        "ticklen": 4,
        "tickcolor": AXIS,
        "tickfont": {"color": TEXT_SECONDARY, "size": 12},
        "title": {"font": {"color": TEXT_SECONDARY, "size": 13}},
        "automargin": True,
    }
    pio.templates[TEMPLATE_NAME] = go.layout.Template(
        layout={
            "colorway": list(SERIES),
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            "font": {"family": FONT, "size": 13, "color": TEXT_PRIMARY},
            "title": {"font": {"size": 15, "color": TEXT_PRIMARY}, "x": 0.012,
                      "xanchor": "left", "y": 0.955, "yanchor": "top"},
            "xaxis": axis,
            "yaxis": axis,
            "margin": {"l": 60, "r": 20, "t": 50, "b": 50},
            "legend": {"bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                       "font": {"color": TEXT_SECONDARY, "size": 12}},
            "hoverlabel": {"font": {"family": FONT, "size": 12},
                           "bgcolor": SURFACE, "bordercolor": AXIS},
            "hovermode": "closest",
            "colorscale": {"sequential": mpl_colorscale(CMAP_INTENSITY),
                           "diverging": mpl_colorscale(CMAP_DIVERGING)},
        }
    )
    pio.templates.default = TEMPLATE_NAME
    return TEMPLATE_NAME


def use_mpl_style() -> None:
    """Match matplotlib to the plotly template, for the few static figures."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 12,
        "axes.titlelocation": "left",
        "axes.grid": True,
        "axes.prop_cycle": mpl.cycler(color=list(SERIES)),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
        "font.size": 11,
        "figure.dpi": 110,
        "savefig.bbox": "tight",
        "image.cmap": CMAP_INTENSITY,
        "image.origin": "lower",
        "legend.frameon": False,
    })


def set_title(fig: go.Figure, headline: str, subtitle: str | None = None,
              has_subplots: bool = False) -> go.Figure:
    """Give a figure one headline and at most one subtitle, with room for both.

    Three layers of text used to pile into the same band at the top of a figure --
    a title, a legend, and the subplot titles -- and they overlapped. The rule
    this function enforces: the headline sits on its own line, any secondary
    reading (polarization fractions, measured amplitudes) becomes a smaller
    subtitle beneath it, subplot titles are nudged clear, and the top margin
    grows to fit whatever is actually there. Legends go *below* the plot; see
    :func:`legend_below`.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Modified in place.
    headline : str
        Short, stable across slider drags, so the reader's eye is not chasing it.
    subtitle : str, optional
        The part that changes as controls move.
    has_subplots : bool, optional
        True if the figure was built with ``subplot_titles``; their annotations
        are then pushed down to clear the header.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    text = headline
    if subtitle:
        text += (f"<br><span style='font-size:0.82em;color:{TEXT_SECONDARY}'>"
                 f"{subtitle}</span>")
    fig.update_layout(title={"text": text})

    # Subplot titles are annotations at the top of each cell's domain, and the
    # domain is measured inside the margins -- so growing the top margin moves
    # them down with the axes, and all the header needs is room above them.
    top = 52 + (24 if subtitle else 0) + (30 if has_subplots else 0)
    fig.update_layout(margin={**(fig.layout.margin.to_plotly_json() or {}), "t": top})
    return fig


def legend_below(fig: go.Figure, y: float = -0.18) -> go.Figure:
    """Put the legend under the plot, horizontally, out of the title's way.

    A legend across the top of a figure collides with the title and with subplot
    titles, which is exactly what it used to do. Below the axes it always fits,
    and for two or three series it reads like a caption.
    """
    fig.update_layout(showlegend=True,
                      legend={"orientation": "h", "x": 0.0, "xanchor": "left",
                              "y": y, "yanchor": "top"})
    return fig


def square_axes(fig: go.Figure, row: int | None = None, col: int | None = None) -> go.Figure:
    """Lock a 1:1 aspect ratio -- mandatory for anything on the sky or the uv plane.

    An image or a uv track with unequal axis scales is a lie about the geometry,
    so this is applied rather than left to the reader's eye.
    """
    kwargs = {"scaleanchor": "x", "scaleratio": 1}
    if row is None:
        fig.update_yaxes(**kwargs)
    else:
        fig.update_yaxes(row=row, col=col, **kwargs)
    return fig


def annotate_note(fig: go.Figure, text: str, y: float = -0.16) -> go.Figure:
    """Add a small caption under a figure -- the one sentence it should leave you with."""
    fig.add_annotation(text=text, xref="paper", yref="paper", x=0.0, y=y,
                       xanchor="left", yanchor="top", showarrow=False,
                       font={"size": 11, "color": TEXT_MUTED}, align="left")
    return fig


register_template()
use_mpl_style()
