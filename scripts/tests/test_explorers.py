"""Every explorer, built headless in static mode.

Widgets need a kernel, but the *builders* behind them do not: ``static=True``
runs each explorer's builder on its controls' initial values and returns a plain
figure. That exercises the whole stack -- controls, physics, figures -- and
catches the kind of mistake (a bad kwarg, a renamed function, a subplot
reference) that would otherwise only show up when a reader opens the notebook.

The trace-count check matters: an explorer updates its figure in place, so a
builder that emits a different number of traces for different inputs would break
as soon as a slider moved.
"""

import interactive_jones as interactive_jones_mod
import interactive_polarization as interactive_pol
import interactive_telescope as interactive_tel
import numpy as np
import plotly.graph_objects as go
import pytest
import widgets

POL_EXPLORERS = [getattr(interactive_pol, name) for name in interactive_pol.__all__]
EHT_EXPLORERS = [getattr(interactive_tel, name) for name in interactive_tel.__all__
                 if callable(getattr(interactive_tel, name))]
JONES_EXPLORERS = [getattr(interactive_jones_mod, name)
                   for name in interactive_jones_mod.__all__]


@pytest.mark.parametrize("explorer", POL_EXPLORERS, ids=lambda f: f.__name__)
def test_polarization_explorers_build_statically(explorer):
    figure = explorer(static=True)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) > 0


@pytest.mark.parametrize("explorer", EHT_EXPLORERS + JONES_EXPLORERS,
                         ids=lambda f: f.__name__)
def test_eht_explorers_build_statically(explorer):
    figure = explorer(static=True)
    assert isinstance(figure, go.Figure)
    assert len(figure.data) > 0


@pytest.mark.parametrize("explorer", POL_EXPLORERS + EHT_EXPLORERS + JONES_EXPLORERS,
                         ids=lambda f: f.__name__)
def test_every_explorer_documents_itself(explorer):
    """The docstring is what the notebook shows next to the figure."""
    assert explorer.__doc__ and len(explorer.__doc__) > 80


def test_controls_are_plain_specs_not_widgets():
    """So static rendering and the tests need nothing installed but numpy and plotly."""
    control = widgets.slider(1.0, 0.0, 2.0, 0.1, "x")
    assert isinstance(control, widgets.Control)
    assert (control.value, control.low, control.high) == (1.0, 0.0, 2.0)
    with pytest.raises(ValueError, match="unknown control kind"):
        widgets._build_widget(widgets.Control("nonsense", 1, "x"))


def test_wave_explorer_presets_stay_in_range():
    """A preset that a slider cannot represent would silently do nothing."""
    controls = {
        "amp_x": widgets.slider(1.0, 0.0, 1.0, 0.05, "a_x"),
        "amp_y": widgets.slider(1.0, 0.0, 1.0, 0.05, "a_y"),
        "delta_deg": widgets.slider(45.0, -180.0, 180.0, 5.0, "delta"),
    }
    for settings in ({"amp_x": 1.0, "amp_y": 1.0, "delta_deg": -90.0},
                     {"amp_x": 1.0, "amp_y": 0.0, "delta_deg": 0.0}):
        for name, value in settings.items():
            assert controls[name].low <= value <= controls[name].high


def test_static_figures_are_stable_across_rebuilds():
    """Random draws are seeded, so a notebook re-run does not silently change."""
    first = interactive_pol.depolarization_explorer(static=True)
    second = interactive_pol.depolarization_explorer(static=True)
    assert np.allclose(np.array(first.data[1].y, dtype=float),
                       np.array(second.data[1].y, dtype=float))


@pytest.mark.parametrize("model", ["ring", "blob", "double"])
def test_the_three_sky_models_have_the_flux_they_claim(model):
    image = interactive_tel._model_image(model)
    assert image["I"].sum() == pytest.approx(1.0, rel=1e-9)
    assert image["I"].min() >= 0.0


def test_unknown_sky_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        interactive_tel._model_image("spiral")


def test_array_map_keeps_a_constant_trace_count_through_the_night():
    """Stations rise and set, but the figure's shape must not change."""
    import arrays as arr
    import figures_telescope as figs_tel

    array = arr.array_2017()
    ra, dec = arr.SOURCES["M87"]
    counts = {len(figs_tel.array_map(array, interactive_tel.MJD_2017_APR11 + hour / 24.0,
                                     ra, dec).data)
              for hour in range(0, 24, 3)}
    assert len(counts) == 1


# ---------------------------------------------------------------------------
# The live widget path
# ---------------------------------------------------------------------------

def _numpy_paths(obj, path=""):
    """Every place a numpy array is hiding inside a nested structure."""
    if isinstance(obj, np.ndarray):
        return [path]
    if isinstance(obj, dict):
        return [p for key, value in obj.items() for p in _numpy_paths(value, f"{path}.{key}")]
    if isinstance(obj, (list, tuple)):
        return [p for i, value in enumerate(obj) for p in _numpy_paths(value, f"{path}[{i}]")]
    return []


@pytest.mark.parametrize("explorer", POL_EXPLORERS + EHT_EXPLORERS + JONES_EXPLORERS,
                         ids=lambda f: f.__name__)
def test_no_numpy_arrays_reach_the_widget_state(explorer):
    """Regression: numpy in a FigureWidget breaks every pan, zoom and hover.

    plotly keeps whatever you hand it, but serialises arrays to the browser
    base64-encoded. When the browser echoes them back, plotly's own property
    pruning evaluates ``if not input_val`` against the numpy array it still
    holds and raises "truth value of an array ... is ambiguous" -- from deep
    inside ipywidgets' message handler, on an interaction rather than on a call.
    ``widgets.plain()`` converts on the way in; this keeps it that way.
    """
    pytest.importorskip("ipywidgets")
    widget = explorer()
    figure = next((child for child in widget.children
                   if type(child).__name__ == "FigureWidget"), None)
    if figure is None:
        pytest.skip("this explorer redraws instead of updating in place")

    assert not _numpy_paths(figure._data, "data")
    assert not _numpy_paths(figure._layout, "layout")

    control = next((c for c in widget.children[0].children if hasattr(c, "value")), None)
    if control is not None:
        control.value = control.value
    assert not _numpy_paths(figure._data, "redrawn")


def test_plain_converts_arrays_scalars_and_nested_structures():
    plain = widgets.plain({"z": np.arange(4.0).reshape(2, 2),
                           "n": np.float64(1.5),
                           "deep": [{"x": np.array([1, 2])}, ("a", np.int64(3))]})
    assert plain == {"z": [[0.0, 1.0], [2.0, 3.0]], "n": 1.5,
                     "deep": [{"x": [1, 2]}, ["a", 3]]}
    assert not _numpy_paths(plain)


def test_image_bearing_figures_still_use_the_live_widget():
    """Even figures containing a 2-D array update in place, with the echo muted.

    Plotly's ``FigureWidget`` cannot round trip a heatmap or a surface: the
    browser echoes the array back in a different shape and plotly's property
    pruning raises from inside ipywidgets' message handler. The fix is to detach
    the two observers that process that echo -- nothing here reads the
    browser-computed defaults they record. Redrawing into an ``Output`` instead
    was the previous approach, and it depends on the front end rendering a rich
    mime bundle nested in a widget, which VS Code does not do reliably.
    """
    pytest.importorskip("ipywidgets")
    with_images = [explorer for explorer in POL_EXPLORERS + EHT_EXPLORERS + JONES_EXPLORERS
                   if widgets.needs_rerender(explorer(static=True))]
    assert with_images, "expected some figures to contain a 2-D array trace"

    for explorer in with_images:
        widget = explorer()
        kinds = [type(child).__name__ for child in widget.children]
        assert "FigureWidget" in kinds, explorer.__name__


@pytest.mark.parametrize("explorer", POL_EXPLORERS + EHT_EXPLORERS + JONES_EXPLORERS,
                         ids=lambda f: f.__name__)
def test_the_browser_echo_is_muted_on_every_live_figure(explorer):
    """The two observers that crash on 2-D arrays must not be attached."""
    pytest.importorskip("ipywidgets")
    widget = explorer()
    figure = next((child for child in widget.children
                   if type(child).__name__ == "FigureWidget"), None)
    if figure is None:
        pytest.skip("this explorer redraws instead of updating in place")

    for trait in widgets._ECHO_TRAITS:
        # Compare by handler *name*: traitlets stores the class-level
        # ObserveHandler, so checking for the bound method would pass vacuously --
        # which is exactly how a broken mute shipped once already.
        remaining = [getattr(entry, "name", None)
                     for entry in figure._trait_notifiers.get(trait, {}).get("change", [])]
        assert f"_handler{trait}" not in remaining, (
            f"{explorer.__name__}: {trait} still observed")


@pytest.mark.parametrize("explorer", POL_EXPLORERS + EHT_EXPLORERS + JONES_EXPLORERS,
                         ids=lambda f: f.__name__)
def test_every_control_can_be_driven_through_its_whole_range(explorer):
    """Move every slider to both ends, pick every dropdown option, click every preset.

    This is the test that catches builders whose trace list changes shape --
    selecting the same station in two dropdowns, or switching a map for a 3D
    scene -- which raises only when the reader touches that particular control.
    """
    pytest.importorskip("ipywidgets")
    widget = explorer()
    for control in widget.children[0].children:
        if not hasattr(control, "value"):
            continue
        if type(control).__name__ == "Checkbox":
            control.value = not control.value
            control.value = not control.value
        elif hasattr(control, "options"):
            for option in list(control.options):
                control.value = option
        else:
            control.value = control.max
            control.value = control.min
    for row in widget.children[1:2]:
        for button in getattr(row, "children", []):
            if type(button).__name__ == "Button":
                button.click()


def test_the_widget_state_guard_actually_fires():
    """The guard must catch a numpy leak, or it is decoration."""
    pytest.importorskip("ipywidgets")
    import figures_polarization as figs
    import images as im
    import plotly.graph_objects as go

    leaky = go.FigureWidget(figs.pol_image_figure(im.polarized_ring(npix=16, p_lin=0.2)))
    with pytest.raises(RuntimeError, match="numpy arrays reached the FigureWidget"):
        widgets._check_widget_state(leaky)


def test_redrawn_figures_never_accumulate_copies():
    """One figure in the Output, no matter how many times a control moves.

    Regression: the redraw path used ``clear_output(wait=True)`` and then
    ``display(figure)``. Plotly publishes its own mime bundle, which does not
    always land inside the Output's capture context, so the pending clear never
    fired and each redraw appended another copy of the plot. ``render_into``
    assigns ``outputs`` directly, which is synchronous.
    """
    pytest.importorskip("ipywidgets")
    import plotly.graph_objects as go

    # No explorer takes this path today -- every figure in the report updates in
    # place -- so the machinery is driven directly. ``rerender=True`` is what a
    # builder that changes the figure's shape would ask for, and it is also the
    # fallback when ``FigureWidget`` is unavailable.
    def build(count):
        return go.Figure(go.Scatter(y=list(range(int(count)))))

    widget = widgets.reactive(build, {"count": widgets.slider(3, 1, 9, 1, "count")},
                              rerender=True)
    out = next(child for child in widget.children
               if type(child).__name__ == "Output")
    assert len(out.outputs) == 1

    control = next(child for child in widget.children[0].children
                   if hasattr(child, "max"))
    for _ in range(3):
        control.value = min(control.max, control.value + (control.step or 1))
        assert len(out.outputs) == 1, "the redraw path accumulated copies"


#: Trace types that make plotly.js fetch map data from its CDN before drawing.
GEO_TRACES = ("scattergeo", "choropleth", "scattermapbox", "scattermap")


def test_the_array_map_is_the_only_figure_that_depends_on_plotly_geo():
    """One figure may need a CDN fetch to draw. Exactly one.

    A ``scattergeo`` trace makes plotly.js download world coastline data before it
    renders anything, which is fast on a good connection and can stall on a
    restricted one. ``array_explorer`` pays that cost deliberately -- the
    coastlines are what make the array legible, and its caption says so, with
    ``figures_telescope.array_globe_3d`` named as the offline fallback. Every
    other figure in the report has to draw with nothing external.
    """
    for explorer in POL_EXPLORERS + EHT_EXPLORERS + JONES_EXPLORERS:
        figure = explorer(static=True)
        offenders = sorted({trace.type for trace in figure.data
                            if trace.type in GEO_TRACES})
        if explorer.__name__ == "array_explorer":
            assert offenders == ["scattergeo"], "the array map should be a geo figure"
        else:
            assert not offenders, f"{explorer.__name__} needs {offenders}"


def test_the_offline_globe_still_works_and_needs_nothing_external():
    """The fallback the array map's caption points at, exercised for real.

    Nothing in the notebooks calls ``array_globe_3d`` any more, so without this
    it would rot -- and the caption would be promising a figure that no longer
    builds.
    """
    import arrays as arr
    import figures_telescope as figs_tel

    array = arr.array_2017()
    ra, dec = arr.SOURCES["M87"]
    figure = figs_tel.array_globe_3d(array, interactive_tel.MJD_2017_APR11, ra, dec)
    assert not [trace for trace in figure.data if trace.type in GEO_TRACES]
    assert {trace.type for trace in figure.data} == {"surface", "scatter3d"}
    labelled = [trace for trace in figure.data if trace.type == "scatter3d"
                and trace.text]
    assert sorted(code for trace in labelled for code in trace.text) == sorted(array)


def test_muting_really_stops_plotly_from_pruning_properties():
    """The end-to-end proof, not just an absent observer.

    Deliver the browser echo by hand and watch whether plotly's
    ``_remove_overlapping_props`` -- the function that raises on a 2-D array --
    gets called at all. Unmuted it is reached; muted it is not.
    """
    pytest.importorskip("ipywidgets")
    import figures_jones as figs_jones
    import plotly.graph_objects as go
    from plotly.basewidget import BaseFigureWidget

    calls = []
    original = BaseFigureWidget._remove_overlapping_props
    BaseFigureWidget._remove_overlapping_props = staticmethod(
        lambda *args, **kwargs: (calls.append(1), [])[1])
    try:
        def send_echo(figure):
            figure._last_trace_edit_id = 7
            figure._js2py_traceDeltas = {
                "trace_deltas": [{"uid": figure.data[0].uid, "z": [[0.0, 1.0]]}],
                "trace_edit_id": 7}

        def fresh():
            return go.FigureWidget(
                figs_jones.jones_chain_figure(1.0, 1.0, 0.02, 0.02, "rl", 0.5))

        unmuted = fresh()
        send_echo(unmuted)
        assert calls, "expected plotly to prune properties on an unmuted widget"

        calls.clear()
        muted = fresh()
        assert widgets._mute_browser_echo(muted) == list(widgets._ECHO_TRAITS)
        send_echo(muted)
        assert not calls, "muting did not stop plotly's property pruning"
    finally:
        BaseFigureWidget._remove_overlapping_props = original
