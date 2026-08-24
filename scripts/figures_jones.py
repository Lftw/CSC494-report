"""Figures for notebook 03: 2x2 matrices, baseline slots, and what goes wrong.

Most of these draw a *matrix*, which is unusual for a plot and deliberate here:
the whole subject is a chain of 2x2 complex matrices, and the fastest way to see
that a field rotation is diagonal for circular feeds and a real rotation for
linear ones is to look at the four numbers.
"""

from __future__ import annotations

import images as im
import jones
import numpy as np
import plotly.graph_objects as go
import polarization as pol
import style
from plotly.subplots import make_subplots

__all__ = ["matrix_figure", "jones_chain_figure", "slot_grid_figure",
           "slots_to_stokes_figure", "stokes_comparison_figure",
           "leakage_image_figure", "field_rotation_figure"]


def _complex_text(value: complex, digits: int = 2) -> str:
    """A complex number short enough to sit inside a matrix cell."""
    if abs(value.imag) < 5e-4:
        return f"{value.real:.{digits}f}"
    if abs(value.real) < 5e-4:
        return f"{value.imag:+.{digits}f}i"
    return f"{value.real:.{digits}f}{value.imag:+.{digits}f}i"


def matrix_figure(matrices: dict[str, np.ndarray], row_labels=("p1", "p2"),
                  col_labels=("p1", "p2"), title: str = "",
                  subtitle: str | None = None, digits: int = 2) -> go.Figure:
    """Draw complex matrices side by side, shaded by magnitude and labelled by value.

    Parameters
    ----------
    matrices : dict
        ``{name: (n, n) complex array}``, drawn left to right in insertion order.
    row_labels, col_labels : sequence of str
        Axis tick labels.
    title, subtitle : str
        Passed to :func:`style.set_title`.
    digits : int, optional
        Decimals in the cell labels.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    names = list(matrices)
    biggest = max(np.abs(matrix).max() for matrix in matrices.values()) or 1.0
    fig = make_subplots(rows=1, cols=len(names), horizontal_spacing=0.08,
                        subplot_titles=names)
    for column, name in enumerate(names, start=1):
        matrix = np.asarray(matrices[name], dtype=complex)
        text = [[_complex_text(value, digits) for value in row] for row in matrix]
        fig.add_trace(go.Heatmap(
            z=np.abs(matrix)[::-1], x=list(col_labels), y=list(row_labels)[::-1],
            text=text[::-1], texttemplate="%{text}", name=name,
            colorscale=style.mpl_colorscale("Blues"), zmin=0.0, zmax=biggest,
            showscale=False, xgap=3, ygap=3,
            textfont={"size": 13, "color": style.TEXT_PRIMARY},
            hovertemplate="%{y}%{x}: %{text}<extra></extra>"), row=1, col=column)
        fig.update_xaxes(showgrid=False, zeroline=False, row=1, col=column)
        fig.update_yaxes(showgrid=False, zeroline=False, row=1, col=column)
    fig.update_layout(height=300, width=min(260 * len(names) + 90, 900))
    style.set_title(fig, title, subtitle, has_subplots=True)
    return fig


def jones_chain_figure(gain_p1: complex, gain_p2: complex, d_p1: complex,
                       d_p2: complex, feed_type: str, rotation: float) -> go.Figure:
    """The three factors of ``J = G (I + D) Phi``, and their product.

    Reading left to right is reading the signal path backwards, which is how
    matrix products work: the rightmost factor acts on the sky first.
    """
    labels = (feed_type[0].upper(), feed_type[1].upper())
    matrices = {
        "G (gain)": jones.gain_matrix(gain_p1, gain_p2),
        "I + D (leakage)": jones.dterm_matrix(d_p1, d_p2),
        "Φ (field rotation)": jones.field_rotation_matrix(feed_type, rotation),
        "J = G (I+D) Φ": jones.jones_matrix(gain_p1, gain_p2, d_p1, d_p2,
                                            feed_type, rotation),
    }
    kind = "circular" if feed_type == "rl" else "linear"
    return matrix_figure(
        matrices, row_labels=labels, col_labels=labels,
        title="One station, one instant",
        subtitle=f"{kind} feeds · field rotation {np.rad2deg(rotation):+.0f}°")


def slot_grid_figure(feed1: str, feed2: str) -> go.Figure:
    """The four correlation products of a baseline, named.

    Rows are station 1's feeds, columns are station 2's. The labels in the cells
    are what the correlator writes to disk -- and on a mixed baseline they are
    products of feeds that measure different things.
    """
    labels = jones.slot_labels(feed1, feed2)
    grid = [[labels[0], labels[2]], [labels[3], labels[1]]]
    mixed = jones.is_mixed(feed1, feed2)
    colour = style.FEED_MIXED if mixed else (
        style.FEED_LINEAR if feed1 == "xy" else style.FEED_CIRCULAR)

    fig = go.Figure(go.Heatmap(
        z=[[1, 1], [1, 1]], x=[f"{feed2[0].upper()} (station 2)",
                               f"{feed2[1].upper()} (station 2)"],
        y=[f"{feed1[1].upper()} (station 1)", f"{feed1[0].upper()} (station 1)"],
        text=[grid[1], grid[0]], texttemplate="%{text}", name="slots",
        colorscale=[[0, colour], [1, colour]], opacity=0.18, showscale=False,
        xgap=4, ygap=4, textfont={"size": 22, "color": style.TEXT_PRIMARY},
        hovertemplate="%{text}<extra></extra>"))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False)
    fig.update_layout(height=320, width=540)
    style.set_title(
        fig, "What this baseline writes to disk",
        f"polbasis '{jones.polbasis(feed1, feed2)}' · "
        + ("the two ends disagree: a MIXED baseline" if mixed
           else "both ends agree"))
    return fig


def slots_to_stokes_figure(feed1: str, feed2: str) -> go.Figure:
    """The 4x4 matrix taking this baseline's four slots to I, Q, U, V.

    One matrix per feed pairing. In a homogeneous array it is the same for every
    baseline and can be applied once to the whole dataset; in a mixed array it is
    not, which is the entire software problem in one picture.
    """
    matrix = jones.slots_to_stokes_matrix(feed1, feed2)
    labels = jones.slot_labels(feed1, feed2)
    fig = matrix_figure({"slots → Stokes": matrix},
                        row_labels=("I", "Q", "U", "V"), col_labels=labels,
                        title="", digits=2)
    fig.update_layout(height=360, width=560)
    style.set_title(fig, "The conversion this baseline needs",
                    f"feeds {feed1.upper()} × {feed2.upper()}", has_subplots=True)
    return fig


def stokes_comparison_figure(truth, recovered, title: str,
                             subtitle: str | None = None,
                             names=("on the sky", "what comes out")) -> go.Figure:
    """Two Stokes vectors side by side: the truth, and what the pipeline produced."""
    keys = ["I", "Q", "U", "V"]
    fig = go.Figure()
    for index, (name, values) in enumerate(zip(names, (truth, recovered),
                                               strict=True)):
        fig.add_trace(go.Bar(
            x=keys, y=np.real(values), name=name, width=0.34,
            marker={"color": style.SERIES[index]},
            text=[f"{value:+.3f}" for value in np.real(values)],
            textposition="outside", textfont={"color": style.TEXT_SECONDARY},
            hovertemplate="%{x} = %{y:.4f}<extra>" + name + "</extra>"))
    span = 1.45 * max(np.abs(np.real(truth)).max(), np.abs(np.real(recovered)).max(), 0.1)
    fig.update_yaxes(title_text="value", range=[-span, span])
    fig.update_layout(height=400, width=620, barmode="group", bargap=0.35)
    style.legend_below(fig)
    style.set_title(fig, title, subtitle)
    return fig


def leakage_image_figure(d_p1: complex, d_p2: complex, feed_type: str = "rl",
                         p_lin: float = 0.0, npix: int = 96) -> go.Figure:
    """A ring with leakage applied, drawn the way a real result would be.

    Every pixel is pushed through the same corrupted baseline, so the ticks are
    the polarization an observer would report. Set the source's own polarization
    to zero and every tick you see is manufactured by the instrument.
    """
    truth = im.polarized_ring(npix=npix, fov_uas=100.0, p_lin=p_lin, pitch_deg=45.0)
    corrupt = jones.jones_matrix(1.0, 1.0, d_p1, d_p2, feed_type)

    # One 4x4 operator does the whole image: leakage is linear in the Stokes vector.
    operator = np.column_stack([
        jones.stokes_from_slots(
            jones.observe(unit, feed_type, feed_type, corrupt, corrupt),
            feed_type, feed_type)
        for unit in np.eye(4)])
    stokes = np.stack([truth[key] for key in "IQUV"])
    observed = np.tensordot(operator, stokes, axes=(1, 0))
    image = dict(truth, I=observed[0], Q=observed[1], U=observed[2], V=observed[3])

    fraction = pol.frac_lin(*[float(observed[k].sum()) for k in range(3)])
    from figures_polarization import pol_image_figure
    fig = pol_image_figure(image, tick_step=8, tick_scale=8.0, i_cut=0.15,
                           title="What the telescope reports")
    style.set_title(
        fig, "What the telescope reports",
        f"source is {p_lin:.0%} polarized · leakage invents "
        f"{jones.spurious_polarization(d_p1, d_p2, feed_type):.1%} · "
        f"net measured {fraction:.1%}")
    return fig


def field_rotation_figure(angle: float) -> go.Figure:
    """The same rotation of the sky, written in a circular and a linear feed basis.

    Left is what a circular-feed station sees: a pure phase on each feed, and no
    change in amplitude. Right is a linear-feed station: a real rotation that
    mixes the feeds -- and therefore mixes Stokes Q into U.
    """
    return matrix_figure(
        {"Φ, circular feeds (R, L)": jones.field_rotation_matrix("rl", angle),
         "Φ, linear feeds (X, Y)": jones.field_rotation_matrix("xy", angle)},
        row_labels=("feed 1", "feed 2"), col_labels=("feed 1", "feed 2"),
        title="One rotation, two very different corrections",
        subtitle=f"field rotation {np.rad2deg(angle):+.0f}° · circular feeds take a "
                 "phase, linear feeds get mixed")
