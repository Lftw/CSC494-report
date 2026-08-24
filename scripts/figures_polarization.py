"""Figures for notebook 01: waves, ellipses, Stokes bars, Poincare sphere, images.

Every builder returns a plain ``go.Figure`` so it can be rendered in a notebook,
exported to HTML, or wrapped in ``go.FigureWidget`` by an ``explore`` module and
updated in place. Traces that get updated are given stable ``name``s; use
:func:`trace_index` to find them instead of relying on trace order.
"""

from __future__ import annotations

import images as im
import numpy as np
import plotly.graph_objects as go
import polarization as pol
import style
from plotly.subplots import make_subplots

__all__ = ["trace_index", "wave_ellipse_figure", "wave_3d_figure", "basis_bars_figure",
           "stokes_bars_figure", "poincare_figure", "depolarization_figure",
           "pol_image_figure", "stokes_panels_figure", "faraday_figure"]


def trace_index(fig: go.Figure, name: str) -> int:
    """Index of the first trace called ``name``.

    Raises
    ------
    KeyError
        If no trace has that name -- a loud failure beats silently updating the
        wrong curve.
    """
    for idx, trace in enumerate(fig.data):
        if trace.name == name:
            return idx
    raise KeyError(f"no trace named {name!r} (have: {[t.name for t in fig.data]})")


def _segments(ticks: dict) -> tuple[list[float], list[float]]:
    """Flatten tick endpoints into one x/y pair with ``None`` breaks between ticks."""
    xs: list[float] = []
    ys: list[float] = []
    for x0, y0, x1, y1 in zip(ticks["x0"], ticks["y0"], ticks["x1"], ticks["y1"],
                             strict=True):
        xs += [x0, x1, None]
        ys += [y0, y1, None]
    return xs, ys


# ---------------------------------------------------------------------------
# Beats 1-2: the wave and its ellipse
# ---------------------------------------------------------------------------

def wave_ellipse_figure(field: np.ndarray, phase_marker: float = 0.0) -> go.Figure:
    """Two panels: the traced polarization ellipse, and the two components vs time.

    Left is what the tip of the electric field vector does in the plane
    transverse to the wave; right is the same information as two oscillations
    with a phase offset. Dragging the phase offset moves both.

    Parameters
    ----------
    field : numpy.ndarray
        Complex ``[E_X, E_Y]``.
    phase_marker : float, optional
        Phase ``omega t`` at which to draw the instantaneous field vector.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    phase, e_x, e_y = pol.wave_trace(field, n_periods=1.0, n_samples=361)
    now = np.exp(1.0j * phase_marker)
    tip = (np.real(field[0] * now), np.real(field[1] * now))
    limit = 1.15 * max(np.abs(field).max(), 1e-9)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.42, 0.58],
                        horizontal_spacing=0.13,
                        subplot_titles=("the polarization ellipse",
                                        "the two components, offset in phase"))

    fig.add_trace(go.Scatter(x=e_x, y=e_y, name="ellipse", mode="lines",
                             showlegend=False,
                             line={"color": style.TEXT_PRIMARY, "width": 2},
                             hovertemplate="E_x %{x:.2f}<br>E_y %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, tip[0]], y=[0, tip[1]], name="vector",
                             mode="lines+markers", showlegend=False,
                             line={"color": style.HIGHLIGHT, "width": 2},
                             marker={"size": [1, 9], "color": style.HIGHLIGHT,
                                     "line": {"color": style.SURFACE, "width": 1}},
                             hovertemplate="the field, right now<extra></extra>"),
                  row=1, col=1)

    for index, (label, values) in enumerate((("E_x", e_x), ("E_y", e_y))):
        fig.add_trace(go.Scatter(x=phase, y=values, name=label, mode="lines",
                                 showlegend=False,
                                 line={"color": style.SERIES[index], "width": 2},
                                 hovertemplate=label + " %{y:.2f}<extra></extra>"),
                      row=1, col=2)
        # Direct label at the end of the curve: no legend box to collide with.
        fig.add_annotation(x=phase[-1], y=values[-1], text=f" {label}", row=1, col=2,
                           showarrow=False, xanchor="left", yanchor="middle",
                           font={"size": 12, "color": style.SERIES[index]})
    fig.add_trace(go.Scatter(x=[phase_marker, phase_marker], y=[-limit, limit],
                             name="now", mode="lines", showlegend=False,
                             line={"color": style.TEXT_MUTED, "width": 1, "dash": "dot"},
                             hoverinfo="skip"),
                  row=1, col=2)

    fig.update_xaxes(title_text="E_x", range=[-limit, limit], row=1, col=1)
    fig.update_yaxes(title_text="E_y", range=[-limit, limit], row=1, col=1,
                     scaleanchor="x", scaleratio=1)
    fig.update_xaxes(title_text=f"phase {style.OMEGA}t (rad)", range=[0, 2.35 * np.pi],
                     row=1, col=2)
    fig.update_yaxes(title_text="field", range=[-limit, limit], row=1, col=2)
    fig.update_layout(height=400, showlegend=False)
    style.set_title(fig, "One wave, two views", ellipse_summary(pol.stokes_from_field(field)),
                    has_subplots=True)
    return fig


def wave_3d_figure(field: np.ndarray, n_periods: float = 2.0) -> go.Figure:
    """The wave travelling through space: flat plane, or corkscrew.

    The figure the phrase "the field spirals as it travels" needs. The wave runs
    left to right; the two faint curves are its shadows on the floor and the back
    wall, which are the two components the algebra works with. At the near face
    the polarization ellipse is drawn, which is what the whole corkscrew looks
    like end-on.

    Parameters
    ----------
    field : numpy.ndarray
        Complex ``[E_X, E_Y]``.
    n_periods : float, optional
        Number of wavelengths to draw.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    phase, e_x, e_y = pol.wave_trace(field, n_periods=n_periods,
                                     n_samples=int(240 * n_periods))
    z = phase / (2 * np.pi)                      # distance, in wavelengths
    limit = 1.25 * max(np.abs(field).max(), 1e-9)
    stokes = pol.stokes_from_field(field)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=z, y=e_x, z=e_y, name="the field", mode="lines",
                               line={"color": style.TEXT_PRIMARY, "width": 6},
                               hoverinfo="skip", showlegend=False))
    # Shadows on the floor and the back wall: the two components, separately.
    fig.add_trace(go.Scatter3d(x=z, y=e_x, z=np.full_like(z, -limit), name="E_x shadow",
                               mode="lines", line={"color": style.SERIES[0], "width": 3},
                               opacity=0.75, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(x=z, y=np.full_like(z, limit), z=e_y, name="E_y shadow",
                               mode="lines", line={"color": style.SERIES[1], "width": 3},
                               opacity=0.75, hoverinfo="skip", showlegend=False))
    # The ellipse at the near face: the corkscrew seen end-on.
    _, ell_x, ell_y = pol.wave_trace(field, n_periods=1.0, n_samples=181)
    fig.add_trace(go.Scatter3d(x=np.zeros_like(ell_x), y=ell_x, z=ell_y,
                               name="polarization ellipse", mode="lines",
                               line={"color": style.ELLIPSE, "width": 5},
                               hoverinfo="skip", showlegend=False))

    transverse = {"range": [-limit, limit], "backgroundcolor": style.SURFACE,
                  "gridcolor": style.GRID, "zerolinecolor": style.AXIS,
                  "showticklabels": False,
                  "title": {"font": {"size": 12, "color": style.TEXT_SECONDARY}}}
    fig.update_layout(
        height=440, margin={"l": 0, "r": 0, "b": 12},
        scene={"xaxis": {"title": {"text": "direction of travel →",
                                  "font": {"size": 12, "color": style.TEXT_SECONDARY}},
                         "showticklabels": False, "backgroundcolor": style.SURFACE,
                         "gridcolor": style.GRID, "zeroline": False},
               "yaxis": {**transverse, "title": {**transverse["title"], "text": "E_x"}},
               "zaxis": {**transverse, "title": {**transverse["title"], "text": "E_y"}},
               "aspectmode": "manual",
               "aspectratio": {"x": 1.5, "y": 1, "z": 1},
               # Framing: the camera looks slightly below centre so the wave sits
               # high in the frame instead of leaving a band of empty space above it.
               "camera": {"eye": {"x": -1.25, "y": -1.55, "z": 0.8},
                          "center": {"x": 0, "y": 0, "z": -0.12},
                          "up": {"x": 0, "y": 0, "z": 1}}})
    style.set_title(fig, "One wave, travelling", ellipse_summary(stokes))
    return fig


# ---------------------------------------------------------------------------
# Beat 3: two bases, one wave
# ---------------------------------------------------------------------------

def basis_bars_figure(field: np.ndarray) -> go.Figure:
    """The same wave decomposed onto linear (X, Y) and circular (R, L) feeds.

    This is the whole mixed-polarization problem in one figure, three notebooks
    early: two telescopes can report completely different pairs of numbers for
    identical light, because they project it onto different feeds.

    Parameters
    ----------
    field : numpy.ndarray
        Complex ``[E_X, E_Y]``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    lin = np.asarray(field, dtype=complex).reshape(2)
    circ = pol.lin_to_circ(lin)

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.16,
                        subplot_titles=("linear feeds (X, Y)", "circular feeds (R, L)"))
    for col, (labels, amps, colour) in enumerate(
            [(["X", "Y"], np.abs(lin), style.FEED_LINEAR),
             (["R", "L"], np.abs(circ), style.FEED_CIRCULAR)], start=1):
        fig.add_trace(go.Bar(x=labels, y=amps, name=f"panel{col}",
                             marker={"color": colour}, width=0.5,
                             text=[f"{a:.2f}" for a in amps], textposition="outside",
                             textfont={"color": style.TEXT_SECONDARY},
                             hovertemplate="|E| %{y:.3f}<extra></extra>",
                             showlegend=False),
                      row=1, col=col)
        phases = np.rad2deg(np.angle(lin if col == 1 else circ))
        # Inside the axes, so it cannot collide with the subplot title above them.
        fig.add_annotation(row=1, col=col, x=0.5, y=0.99, xref="x domain", yref="y domain",
                           yanchor="top", showarrow=False,
                           text=f"phases {phases[0]:+.0f}{style.DEG}, "
                                f"{phases[1]:+.0f}{style.DEG}",
                           font={"size": 11, "color": style.TEXT_MUTED})

    top = 1.45 * max(np.abs(lin).max(), np.abs(circ).max(), 1e-9)
    fig.update_yaxes(title_text="amplitude |E|", range=[0, top], row=1, col=1)
    fig.update_yaxes(range=[0, top], row=1, col=2)
    fig.update_layout(height=360, bargap=0.45, width=760)
    style.set_title(fig, "The same wave, read out two ways", has_subplots=True)
    return fig


# ---------------------------------------------------------------------------
# Beats 4-6: Stokes, depolarization, Poincare
# ---------------------------------------------------------------------------

def stokes_bars_figure(stokes: np.ndarray, normalise: bool = True) -> go.Figure:
    """The four Stokes parameters as labelled bars, with the polarization budget.

    Q, U and V are signed, so the bars run both ways from zero; the annotation
    states the fractions, because the numbers are the point and colour alone
    should never have to carry them.

    Parameters
    ----------
    stokes : numpy.ndarray
        ``[I, Q, U, V]``.
    normalise : bool, optional
        Divide through by I so the axis is always ``[-1, 1]``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    i, q, u, v = (np.asarray(stokes, dtype=float) /
                  (stokes[0] if normalise and stokes[0] else 1.0))
    labels = ["I", "Q", "U", "V"]
    values = [i, q, u, v]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=values, name="stokes",
                         marker={"color": [style.STOKES[k] for k in labels]},
                         width=0.55, text=[f"{val:+.3f}" for val in values],
                         textposition="outside", textfont={"color": style.TEXT_SECONDARY},
                         hovertemplate="%{x} = %{y:.4f}<extra></extra>"))
    fig.update_yaxes(title_text="value / I" if normalise else "value", range=[-1.25, 1.25])
    fig.update_layout(height=360, bargap=0.5, showlegend=False, width=620)
    style.set_title(fig, "Stokes parameters", ellipse_summary(stokes))
    return fig


def ellipse_summary(stokes: np.ndarray) -> str:
    """One-line human summary of a polarization state, for figure subtitles.

    Only mentions what is meaningful: the EVPA of light with no linear component
    is undefined, and saying "EVPA +45" under a circular wave is worse than
    saying nothing.
    """
    i, q, u, v = pol.require_single_state("ellipse_summary", *stokes)
    geo = pol.ellipse_from_stokes(i, q, u, v)
    hand = {1: "right-handed", -1: "left-handed", 0: ""}[geo["handedness"]]

    if geo["p_total"] < 0.005:
        return "unpolarized"
    parts = [f"p = {geo['p_total']:.0%} polarized"]
    if geo["p_lin"] > 0.005:
        parts.append(f"{geo['p_lin']:.0%} linear at EVPA "
                     f"{np.rad2deg(geo['evpa']):+.1f}{style.DEG}")
    if abs(geo["p_circ"]) > 0.005:
        parts.append(f"{abs(geo['p_circ']):.0%} circular ({hand})")
    if len(parts) == 2 and geo["p_lin"] > 0.005:
        parts[1] += " (a straight line)"
    return "  ·  ".join(parts)


def poincare_figure(stokes: np.ndarray) -> go.Figure:
    """The Poincare sphere with the current state marked.

    Surface = fully polarized, centre = unpolarized, equator = linear, poles =
    circular. Azimuth is *twice* the EVPA, which is exactly why position angles
    are ambiguous by 180 degrees.

    Parameters
    ----------
    stokes : numpy.ndarray
        ``[I, Q, U, V]``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    point = pol.poincare_point(*stokes)

    u_ang, v_ang = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=np.cos(u_ang) * np.sin(v_ang), y=np.sin(u_ang) * np.sin(v_ang), z=np.cos(v_ang),
        name="sphere", opacity=0.12, showscale=False, hoverinfo="skip",
        colorscale=[[0, style.TEXT_MUTED], [1, style.TEXT_MUTED]]))

    ring = np.linspace(0, 2 * np.pi, 121)
    fig.add_trace(go.Scatter3d(x=np.cos(ring), y=np.sin(ring), z=np.zeros_like(ring),
                               name="linear (equator)", mode="lines",
                               line={"color": style.TEXT_MUTED, "width": 2},
                               hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=[point[0]], y=[point[1]], z=[point[2]], name="state", mode="markers",
        marker={"size": 7, "color": style.HIGHLIGHT,
                "line": {"color": style.SURFACE, "width": 2}},
        hovertemplate="Q/I %{x:.2f}<br>U/I %{y:.2f}<br>V/I %{z:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[0, point[0]], y=[0, point[1]], z=[0, point[2]],
                               name="radius", mode="lines", showlegend=False,
                               line={"color": style.HIGHLIGHT, "width": 4,
                                     "dash": "dot"},
                               hoverinfo="skip"))

    # Only the poles are labelled: the equator ring already reads as "linear", and
    # six labels plus three axis labels collided with each other.
    marks = {"right circular": (0, 0, 1.02), "left circular": (0, 0, -1.02)}
    fig.add_trace(go.Scatter3d(
        x=[p[0] for p in marks.values()], y=[p[1] for p in marks.values()],
        z=[p[2] for p in marks.values()], text=list(marks), name="landmarks",
        mode="markers+text", textposition="top center",
        textfont={"size": 11, "color": style.TEXT_SECONDARY},
        marker={"size": 4, "color": style.TEXT_SECONDARY}, hoverinfo="skip",
        showlegend=False))

    # Label the axes with arrows *inside* the scene rather than with plotly's 3D
    # axes: the boxed axes ate a third of the figure and clipped their own titles.
    ends = {"Q / I": (1.3, 0, 0), "U / I": (0, 1.3, 0), "V / I": (0, 0, 1.3)}
    axis_x, axis_y, axis_z = [], [], []
    for end in ends.values():
        axis_x += [0, end[0], None]
        axis_y += [0, end[1], None]
        axis_z += [0, end[2], None]
    fig.add_trace(go.Scatter3d(x=axis_x, y=axis_y, z=axis_z, name="axes", mode="lines",
                               line={"color": style.AXIS, "width": 2},
                               hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=[e[0] for e in ends.values()], y=[e[1] for e in ends.values()],
        z=[e[2] for e in ends.values()], text=list(ends), name="axis_labels",
        mode="text", textfont={"size": 13, "color": style.TEXT_SECONDARY},
        hoverinfo="skip", showlegend=False))

    blank = {"visible": False, "range": [-1.35, 1.35], "showspikes": False}
    fig.update_layout(
        height=420, width=520, showlegend=False,
        margin={"l": 0, "r": 0, "b": 10},
        scene={"xaxis": blank, "yaxis": blank, "zaxis": blank,
               "aspectmode": "cube",
               "camera": {"eye": {"x": 1.0, "y": 1.0, "z": 0.72}}})
    style.set_title(fig, "The Poincaré sphere", ellipse_summary(stokes))
    return fig


def depolarization_figure(spread_rad: float, n_waves: int = 400,
                          evpa0_rad: float = 0.0, seed: int = 0) -> go.Figure:
    """Why real light is only *partially* polarized.

    Left: the position angles of many independently emitting wavelets. Right:
    the linear polarization fraction of their incoherent sum, against the spread
    that produced it. Order emerges only if the emitters agree with each other.

    Parameters
    ----------
    spread_rad : float
        Standard deviation of wavelet EVPAs, in radians.
    n_waves : int, optional
        Number of wavelets.
    evpa0_rad : float, optional
        Mean EVPA.
    seed : int, optional
        Reproducibility seed, shared with :func:`~polarization.incoherent_sum`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    rng = np.random.default_rng(seed)
    angles = evpa0_rad + spread_rad * rng.standard_normal(min(n_waves, 300))

    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55], horizontal_spacing=0.14,
                        subplot_titles=(f"{len(angles)} wavelets, each fully polarized",
                                        "polarization survives only if they agree"))
    xs: list[float] = []
    ys: list[float] = []
    for ang in angles:
        xs += [-np.cos(ang), np.cos(ang), None]
        ys += [-np.sin(ang), np.sin(ang), None]
    fig.add_trace(go.Scatter(x=xs, y=ys, name="wavelets", mode="lines",
                             line={"color": style.SERIES[0], "width": 1},
                             opacity=0.28, hoverinfo="skip", showlegend=False),
                  row=1, col=1)

    sweep = np.linspace(0.0, np.pi / 2, 40)
    p_lin = [pol.frac_lin(*pol.incoherent_sum(n_waves, s, evpa0_rad, seed=seed)[:3])
             for s in sweep]
    fig.add_trace(go.Scatter(x=np.rad2deg(sweep), y=p_lin, name="p_lin", mode="lines",
                             line={"color": style.TEXT_PRIMARY, "width": 2},
                             hovertemplate=f"spread %{{x:.0f}}{style.DEG} -> "
                                           "p = %{y:.2f}<extra></extra>",
                             showlegend=False),
                  row=1, col=2)
    here = pol.frac_lin(*pol.incoherent_sum(n_waves, spread_rad, evpa0_rad, seed=seed)[:3])
    fig.add_trace(go.Scatter(x=[np.rad2deg(spread_rad)], y=[here], name="now",
                             mode="markers+text", text=[f"  p = {here:.2f}"],
                             textposition="middle right",
                             textfont={"color": style.HIGHLIGHT},
                             marker={"size": 10, "color": style.HIGHLIGHT},
                             hoverinfo="skip", showlegend=False),
                  row=1, col=2)

    fig.update_xaxes(range=[-1.2, 1.2], showgrid=False, zeroline=False,
                     showticklabels=False, row=1, col=1)
    fig.update_yaxes(range=[-1.2, 1.2], showgrid=False, zeroline=False,
                     showticklabels=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="spread of wavelet angles (deg)", row=1, col=2)
    fig.update_yaxes(title_text="linear polarization fraction", range=[0, 1.05],
                     row=1, col=2)
    fig.update_layout(height=390, showlegend=False)
    style.set_title(fig, "Polarization survives only if the emitters agree",
                    has_subplots=True)
    return fig


# ---------------------------------------------------------------------------
# Beats 8-9: images, and Faraday rotation
# ---------------------------------------------------------------------------

def pol_image_figure(image: dict, tick_step: int = 6, tick_scale: float = 6.0,
                     i_cut: float = 0.1, show_ticks: bool = True,
                     length_mode: str = "polarized",
                     title: str = "Stokes I with EVPA ticks") -> go.Figure:
    """A polarized image the way the EHT publishes one: brightness map plus ticks.

    Ticks are line segments, not arrows, and their length encodes polarized
    intensity -- so a tick is only drawn where there is light to polarize.

    Parameters
    ----------
    image : dict
        Image dictionary from :mod:`images`.
    tick_step : int, optional
        Draw a tick every ``tick_step`` pixels.
    tick_scale : float, optional
        Tick length multiplier. Length is proportional to polarized intensity
        measured against the peak of Stokes I, so a more polarized source really
        does get longer ticks.
    i_cut : float, optional
        Fraction of the peak below which ticks are suppressed.
    show_ticks : bool, optional
        Draw the ticks at all.
    length_mode : {'polarized', 'fraction', 'uniform'}, optional
        Passed to :func:`~images.evpa_ticks`.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    extent = image["extent"]
    axis = np.linspace(extent[0], extent[1], image["I"].shape[0])

    peak = image["I"].max() or 1.0
    fig = go.Figure()
    # Normalised brightness: the absolute Jy/pixel scale is an artefact of the
    # pixel size, and "500u" on a colourbar teaches nobody anything.
    fig.add_trace(go.Heatmap(z=image["I"] / peak, x=axis, y=axis, name="stokes_i",
                             colorscale=style.mpl_colorscale(style.CMAP_INTENSITY),
                             zmin=0.0, zmax=1.0,
                             colorbar={"title": {"text": "Stokes I / peak",
                                                 "side": "right",
                                                 "font": {"size": 12}},
                                       "thickness": 11, "outlinewidth": 0, "len": 0.88,
                                       "tickvals": [0, 0.25, 0.5, 0.75, 1.0]},
                             hovertemplate=f"%{{x:.1f}}, %{{y:.1f}} {style.UAS}"
                                           "<br>I %{z:.2f} of peak<extra></extra>"))
    ticks = im.evpa_ticks(image, step=tick_step, scale=tick_scale, i_cut=i_cut,
                          length_mode=length_mode)
    xs, ys = _segments(ticks)
    fig.add_trace(go.Scatter(x=xs if show_ticks else [], y=ys if show_ticks else [],
                             name="evpa_ticks", mode="lines",
                             line={"color": style.POL_TICK, "width": 2},
                             hoverinfo="skip", showlegend=False))

    fig.update_xaxes(title_text=f"relative RA ({style.UAS})", range=[extent[1], extent[0]],
                     showgrid=False, zeroline=False)
    fig.update_yaxes(title_text=f"relative Dec ({style.UAS})", range=[extent[2], extent[3]],
                     showgrid=False, zeroline=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(height=500, width=580)
    style.set_title(fig, title)
    return fig


def stokes_panels_figure(image: dict) -> go.Figure:
    """Four panels: I on a brightness map, Q, U and V on a diverging one.

    Q, U and V take both signs, so they get a two-hue map with a neutral
    midpoint and a symmetric colour range -- a sequential map here would hide
    the sign, which is the only interesting thing about them.

    Parameters
    ----------
    image : dict
        Image dictionary from :mod:`images`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    extent = image["extent"]
    axis = np.linspace(extent[0], extent[1], image["I"].shape[0])
    span = max(float(np.abs(image[k]).max()) for k in ("Q", "U", "V")) or 1.0

    fig = make_subplots(rows=1, cols=4, horizontal_spacing=0.045,
                        subplot_titles=("I (total)", "Q", "U", "V (circular)"))
    fig.add_trace(go.Heatmap(z=image["I"], x=axis, y=axis, name="I", showscale=False,
                             colorscale=style.mpl_colorscale(style.CMAP_INTENSITY),
                             hovertemplate="I %{z:.2e}<extra></extra>"), row=1, col=1)
    for col, key in enumerate(("Q", "U", "V"), start=2):
        fig.add_trace(go.Heatmap(z=image[key], x=axis, y=axis, name=key,
                                 zmid=0.0, zmin=-span, zmax=span,
                                 showscale=(col == 4),
                                 colorbar={"thickness": 10, "outlinewidth": 0, "len": 0.85},
                                 colorscale=style.mpl_colorscale(style.CMAP_DIVERGING),
                                 hovertemplate=key + " %{z:.2e}<extra></extra>"),
                      row=1, col=col)
    for col in range(1, 5):
        fig.update_xaxes(range=[extent[1], extent[0]], showgrid=False, zeroline=False,
                         showticklabels=False, row=1, col=col)
        fig.update_yaxes(range=[extent[2], extent[3]], showgrid=False, zeroline=False,
                         showticklabels=False, scaleanchor=f"x{col if col > 1 else ''}",
                         scaleratio=1, row=1, col=col)
    fig.update_layout(height=300, margin={"b": 10})
    style.set_title(fig, "The four Stokes images", has_subplots=True)
    return fig


def faraday_figure(rm_rad_m2: float, evpa0_deg: float = 30.0,
                   freq_ghz: tuple[float, ...] = (86.0, 230.0, 345.0)) -> go.Figure:
    """EVPA against wavelength squared: the Faraday rotation signature.

    A straight line whose slope is the rotation measure. The marked frequencies
    are the EHT/ALMA bands, so the figure also shows how much leverage a
    multi-frequency observation actually has.

    Parameters
    ----------
    rm_rad_m2 : float
        Rotation measure in rad m^-2.
    evpa0_deg : float, optional
        Intrinsic EVPA in degrees.
    freq_ghz : tuple of float, optional
        Frequencies to mark.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    c = 299792458.0
    lam = np.linspace(0.0005, 0.005, 200)          # 0.5 mm to 5 mm
    evpa_deg = np.rad2deg(pol.faraday_evpa(np.deg2rad(evpa0_deg), rm_rad_m2, lam))
    lam_marks = np.array([c / (f * 1e9) for f in freq_ghz])
    evpa_marks = np.rad2deg(pol.faraday_evpa(np.deg2rad(evpa0_deg), rm_rad_m2, lam_marks))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lam**2 * 1e6, y=evpa_deg, name="EVPA", mode="lines",
                             showlegend=False,
                             line={"color": style.TEXT_PRIMARY, "width": 2},
                             hovertemplate=f"{style.LAMBDA}2 %{{x:.2f}} mm2<br>"
                                           f"EVPA %{{y:.1f}}{style.DEG}<extra></extra>"))
    # The three bands cluster at small lambda^2, so labelling the markers directly
    # made the labels overlap each other and the line. Dotted verticals with the
    # label parked at the top of the frame always have room.
    for wavelength, freq in zip(lam_marks, freq_ghz, strict=True):
        fig.add_vline(x=wavelength**2 * 1e6,
                      line={"color": style.TEXT_MUTED, "width": 1, "dash": "dot"})
        fig.add_annotation(x=wavelength**2 * 1e6, y=0.99, yref="y domain", yanchor="top",
                           text=f"{freq:.0f} GHz", showarrow=False, textangle=-90,
                           xshift=-7, font={"size": 10, "color": style.TEXT_MUTED})
    fig.add_trace(go.Scatter(x=lam_marks**2 * 1e6, y=evpa_marks, name="bands",
                             mode="markers", showlegend=False,
                             marker={"size": 9, "color": style.HIGHLIGHT,
                                     "line": {"color": style.SURFACE, "width": 1.5}},
                             hovertemplate=f"%{{y:.1f}}{style.DEG}<extra></extra>"))

    fig.update_xaxes(title_text=f"{style.LAMBDA}<sup>2</sup> (mm<sup>2</sup>)")
    fig.update_yaxes(title_text="observed EVPA (deg)")
    fig.update_layout(height=400, width=720, margin={"t": 74})
    style.set_title(fig, "Faraday rotation is chromatic",
                    f"RM = {rm_rad_m2:,.0f} rad m<sup>-2</sup> · a straight line in "
                    f"{style.LAMBDA}<sup>2</sup>, so two frequencies measure it")
    return fig
