"""Smoke tests for every figure builder.

Widgets need a live kernel, but the figures do not -- so every builder is
constructed headless here. Plotly validates properties on construction, so this
catches a mistyped attribute or a broken subplot reference before a notebook
ever opens, which is the whole reason the builders are plain functions.
"""

import figures_polarization as figs
import images as im
import numpy as np
import polarization as pol
import pytest
import style

FIELD = pol.jones_vector(1.0, 0.6, np.deg2rad(-60.0))
STOKES = pol.stokes_from_ellipse(1.0, 0.5, 0.4, 0.2)
IMAGE = im.polarized_ring(npix=64, p_lin=0.3)


def test_template_is_registered_and_default():
    import plotly.io as pio
    assert style.TEMPLATE_NAME in pio.templates
    assert pio.templates.default == style.TEMPLATE_NAME


def test_colorscale_conversion():
    scale = style.mpl_colorscale("afmhot", n_steps=8)
    assert len(scale) == 8
    assert scale[0][0] == 0.0 and scale[-1][0] == 1.0
    assert all(entry[1].startswith("rgb(") for entry in scale)


def test_wave_ellipse_figure_has_its_named_traces():
    fig = figs.wave_ellipse_figure(FIELD, phase_marker=0.7)
    for name in ("ellipse", "vector", "E_x", "E_y", "now"):
        assert figs.trace_index(fig, name) >= 0


def test_trace_index_raises_on_unknown_name():
    fig = figs.wave_ellipse_figure(FIELD)
    with pytest.raises(KeyError, match="no trace named"):
        figs.trace_index(fig, "does_not_exist")


def test_wave_3d_figure_draws_the_wave_its_shadows_and_the_ellipse():
    fig = figs.wave_3d_figure(FIELD, n_periods=2.0)
    assert [t.name for t in fig.data] == ["the field", "E_x shadow", "E_y shadow",
                                          "polarization ellipse"]
    # Stretched along the direction of travel, and the amplitude ticks are hidden:
    # the shape is the point, not the numbers.
    assert fig.layout.scene.aspectratio.x > fig.layout.scene.aspectratio.z
    assert fig.layout.scene.yaxis.showticklabels is False


def test_basis_bars_figure_shows_both_bases():
    fig = figs.basis_bars_figure(FIELD)
    assert len(fig.data) == 2
    circular = pol.lin_to_circ(FIELD)
    assert np.allclose(fig.data[1].y, np.abs(circular))
    # Feed colours are semantic and must not drift between figures.
    assert fig.data[0].marker.color == style.FEED_LINEAR
    assert fig.data[1].marker.color == style.FEED_CIRCULAR


def test_stokes_bars_figure_normalises_and_labels():
    fig = figs.stokes_bars_figure(np.array([2.0, 1.0, 0.0, 0.0]))
    assert list(fig.data[0].y) == pytest.approx([1.0, 0.5, 0.0, 0.0])
    assert fig.data[0].text[1] == "+0.500"      # values are always written out


def test_ellipse_summary_only_reports_what_is_meaningful():
    """An EVPA under circular light is undefined, so it must not be printed."""
    circular = figs.ellipse_summary(pol.stokes_from_ellipse(1, 0.0, 0, 0.5))
    assert "right-handed" in circular and "EVPA" not in circular
    assert "left-handed" in figs.ellipse_summary(pol.stokes_from_ellipse(1, 0.0, 0, -0.5))

    linear = figs.ellipse_summary(pol.stokes_from_ellipse(1, 0.5, 0, 0.0))
    assert "linear" in linear and "circular" not in linear

    assert figs.ellipse_summary(np.array([1.0, 0.0, 0.0, 0.0])) == "unpolarized"


def test_poincare_figure_marks_the_state():
    fig = figs.poincare_figure(STOKES)
    state = fig.data[figs.trace_index(fig, "state")]
    assert (state.x[0], state.y[0], state.z[0]) == pytest.approx(
        tuple(pol.poincare_point(*STOKES)))


def test_depolarization_figure():
    fig = figs.depolarization_figure(np.deg2rad(25.0), n_waves=200)
    assert len(fig.data) == 3


def test_pol_image_figure_has_a_reversed_ra_axis():
    fig = figs.pol_image_figure(IMAGE, tick_step=8)
    assert fig.layout.xaxis.range[0] > fig.layout.xaxis.range[1]   # RA increases left
    assert fig.layout.yaxis.scaleanchor == "x"                     # square pixels
    assert len(fig.data) == 2


def test_pol_image_figure_can_hide_ticks():
    fig = figs.pol_image_figure(IMAGE, show_ticks=False)
    assert len(fig.data[figs.trace_index(fig, "evpa_ticks")].x) == 0


def test_stokes_panels_figure_uses_a_symmetric_diverging_scale():
    fig = figs.stokes_panels_figure(IMAGE)
    assert len(fig.data) == 4
    for trace in fig.data[1:]:
        assert trace.zmid == 0.0
        assert trace.zmin == pytest.approx(-trace.zmax)


def test_faraday_figure():
    fig = figs.faraday_figure(1.0e5, evpa0_deg=20.0, freq_ghz=(86.0, 230.0))
    assert len(fig.data) == 2
    assert len(fig.data[1].x) == 2


def test_all_builders_survive_extreme_but_legal_inputs():
    """Zero-amplitude and unpolarized states are legal and must not divide by zero."""
    dark = pol.jones_vector(0.0, 0.0, 0.0)
    figs.wave_ellipse_figure(dark)
    figs.wave_3d_figure(dark)
    figs.basis_bars_figure(dark)
    figs.stokes_bars_figure(np.array([1.0, 0.0, 0.0, 0.0]))
    figs.poincare_figure(np.array([1.0, 0.0, 0.0, 0.0]))
    figs.pol_image_figure(im.polarized_ring(npix=32, p_lin=0.0))
    figs.stokes_panels_figure(im.gaussian_blob(npix=32, p_lin=0.0))
    figs.faraday_figure(0.0)


def test_no_shapes_are_drawn_on_log_axes():
    """Plotly reads shape coordinates on a log axis as powers of ten.

    ``add_hline(y=42)`` on a log axis puts the line at 10**42, which wrecks the
    autorange and silently hides the annotation. The resolution figure was drawn
    that way; landmark lines are traces now. This keeps every log-axis figure
    honest.
    """
    import figures_telescope as figs_tel
    import interactive_telescope as tel

    for explorer in (tel.resolution_explorer, tel.visibility_profile_explorer):
        figure = explorer(static=True)
        for axis_name in ("xaxis", "yaxis"):
            if figure.layout[axis_name].type != "log":
                continue
            letter = axis_name[0]
            offenders = [shape for shape in (figure.layout.shapes or ())
                         if getattr(shape, f"{letter}ref", "").startswith(letter)]
            assert not offenders, f"{explorer.__name__}: shape on a log {letter} axis"

    # And the landmark lines really are where they claim to be.
    figure = figs_tel.resolution_figure(1.3, 1e4)
    landmark_ys = {trace.y[0] for trace in figure.data if trace.name in figs_tel.LANDMARKS}
    assert landmark_ys == set(figs_tel.LANDMARKS.values())
