"""The one piece of widget plumbing every interactive figure here uses.

:func:`reactive` takes a figure *builder* -- an ordinary function mapping
parameters to a ``go.Figure`` -- plus a set of :class:`Control` specs, and wires
them together.

Controls are declared as plain dataclasses rather than as ipywidgets, which buys
three things: the physics and figures never import a widget library, an explorer
can be rendered statically with nothing installed but numpy and plotly, and the
whole stack is testable headless (``tests/test_explorers.py`` builds every
explorer this way).

Interactivity itself does need ``ipywidgets`` and a live kernel -- Jupyter Lab,
VS Code, or Binder. If it is missing, the error message says what to do.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import plotly.graph_objects as go
import style

_MISSING_MSG = (
    "This figure is interactive and needs ipywidgets with a live kernel.\n"
    "    pip install -r requirements.txt\n"
    "Then run the notebook in Jupyter Lab or VS Code. Every explorer also takes\n"
    "static=True to return a plain (non-interactive) figure instead."
)

_SLIDER_WIDTH = "380px"
_LABEL_WIDTH = "150px"
_CONTROL_MARGIN = "0 18px 6px 0"


@dataclass
class Control:
    """Declarative spec for one input: what it is, not how it is rendered.

    Attributes
    ----------
    kind : {'float', 'int', 'bool', 'choice'}
        Which widget to build when a live kernel is available.
    value : Any
        Current value -- and, before anything is dragged, the value a static
        render uses.
    """

    kind: str
    value: Any
    description: str
    low: float | None = None
    high: float | None = None
    step: float | None = None
    options: Sequence[Any] = field(default_factory=tuple)
    continuous: bool = True
    readout_format: str = ".2f"


def slider(value: float, low: float, high: float, step: float, description: str,
           readout_format: str = ".2f", continuous: bool = True) -> Control:
    """A continuous control."""
    return Control("float", value, description, low, high, step,
                   continuous=continuous, readout_format=readout_format)


def int_slider(value: int, low: int, high: int, step: int, description: str,
               continuous: bool = True) -> Control:
    """An integer control (pixel counts, sample counts)."""
    return Control("int", value, description, low, high, step, continuous=continuous)


def toggle(value: bool, description: str) -> Control:
    """An on/off control."""
    return Control("bool", value, description)


def dropdown(options: Sequence[Any], value: Any, description: str) -> Control:
    """A pick-one control."""
    return Control("choice", value, description, options=options)


def require_widgets():
    """Import and return ``ipywidgets``, with an actionable message if absent."""
    try:
        import ipywidgets
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(_MISSING_MSG) from exc
    return ipywidgets


def _build_widget(control: Control):
    """Turn a :class:`Control` into the ipywidgets widget it describes."""
    if control.kind not in ("float", "int", "bool", "choice"):
        raise ValueError(f"unknown control kind {control.kind!r}")
    widgets = require_widgets()
    layout = widgets.Layout(width=_SLIDER_WIDTH, margin=_CONTROL_MARGIN)
    style = {"description_width": _LABEL_WIDTH}
    if control.kind == "float":
        return widgets.FloatSlider(value=control.value, min=control.low, max=control.high,
                                   step=control.step, description=control.description,
                                   continuous_update=control.continuous,
                                   readout_format=control.readout_format,
                                   style=style, layout=layout)
    if control.kind == "int":
        return widgets.IntSlider(value=control.value, min=control.low, max=control.high,
                                 step=control.step, description=control.description,
                                 continuous_update=control.continuous,
                                 style=style, layout=layout)
    if control.kind == "bool":
        return widgets.Checkbox(value=control.value, description=control.description,
                                indent=False,
                                layout=widgets.Layout(width="260px",
                                                      margin=_CONTROL_MARGIN))
    return widgets.Dropdown(options=list(control.options), value=control.value,
                            description=control.description, style=style,
                            layout=widgets.Layout(width="330px",
                                                  margin=_CONTROL_MARGIN))


def caption(text: str):
    """A small muted note under a figure -- the sentence the figure is making."""
    widgets = require_widgets()
    return widgets.HTML(
        f"<div style='color:{style.TEXT_MUTED};font-size:0.86em;line-height:1.45;"
        f"max-width:60em;margin:2px 0 10px 4px'>{text}</div>"
    )


def plain(value):
    """Recursively replace numpy arrays and scalars with plain Python equivalents.

    Necessary, not cosmetic. A ``FigureWidget`` keeps whatever you hand it, so a
    numpy ``z`` stays a numpy array in the widget's internal state. When the
    browser echoes that array back it arrives base64-encoded -- which is a *dict*
    in plotly 6 -- and plotly's own property pruning then evaluates
    ``if not input_val`` on the numpy array it still holds, raising

        ValueError: The truth value of an array with more than one element is
        ambiguous.

    on every pan, zoom or hover over an image-bearing figure. Plain lists make
    that truthiness test legal, so the arrays are converted on the way in.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def numpy_paths(obj, path: str = "") -> list[str]:
    """Every place a numpy array is hiding inside a nested structure.

    Used to enforce the invariant :func:`plain` exists to maintain.
    """
    if isinstance(obj, np.ndarray):
        return [path]
    if isinstance(obj, dict):
        return [p for key, value in obj.items() for p in numpy_paths(value, f"{path}.{key}")]
    if isinstance(obj, (list, tuple)):
        return [p for i, value in enumerate(obj) for p in numpy_paths(value, f"{path}[{i}]")]
    return []


def render_into(out, figure: go.Figure) -> None:
    """Replace an ``Output`` widget's contents with exactly one figure.

    Deliberately not ``clear_output()`` plus ``display()``. Plotly renders a
    figure by publishing its own mime bundle, which does not always land inside
    the ``Output``'s capture context -- so a ``clear_output(wait=True)`` sits
    there waiting for output that never arrives, and every redraw *appends*
    another copy of the figure. Assigning ``outputs`` directly is synchronous and
    can only ever leave one.
    """
    bundle = figure._repr_mimebundle_()
    if isinstance(bundle, tuple):           # IPython allows (data, metadata)
        bundle = bundle[0]
    if not bundle:                          # no notebook renderer configured
        bundle = {"application/vnd.plotly.v1+json": figure.to_plotly_json()}
    out.outputs = ({"output_type": "display_data", "data": bundle, "metadata": {}},)


def _check_widget_state(view) -> None:
    """Fail loudly, here, if a numpy array made it into the widget's state.

    Otherwise the consequence surfaces as a ValueError from inside ipywidgets'
    message handler the first time the reader touches the figure -- a traceback
    with none of our code in it. Better to break at build time with a sentence
    that names the property.
    """
    found = numpy_paths(view._data, "data") + numpy_paths(view._layout, "layout")
    if found:
        raise RuntimeError(
            "numpy arrays reached the FigureWidget state at "
            f"{', '.join(found[:5])}. Pass them through widgets.plain() first -- "
            "plotly keeps them, then chokes on them when the browser echoes "
            "the figure back. See widgets.plain.__doc__."
        )


#: Trace types whose data is a 2-D array. Plotly's ``FigureWidget`` cannot round
#: trip these unaided -- see :func:`_mute_browser_echo` for what goes wrong and
#: how it is dealt with. Kept because the tests use it to say which figures the
#: problem applies to.
ARRAY_2D_TRACES = frozenset({"heatmap", "heatmapgl", "contour", "contourcarpet",
                             "surface", "image"})


def needs_rerender(figure: go.Figure) -> bool:
    """True if this figure contains a trace whose data is a 2-D array."""
    return any(trace.type in ARRAY_2D_TRACES for trace in figure.data)


#: Plotly traits carrying what the *browser* computed, echoed back to Python.
_ECHO_TRAITS = ("_js2py_traceDeltas", "_js2py_layoutDelta")


def _mute_browser_echo(view: go.FigureWidget) -> list[str]:
    """Stop plotly from processing the figure state the browser sends back.

    Plotly registers observers on ``_js2py_traceDeltas`` and ``_js2py_layoutDelta``
    to record the property defaults the browser computed. Those handlers call
    ``_remove_overlapping_props``, which cannot cope with a 2-D array: the array is
    held one way in Python and echoed back another, and the comparison raises --
    ``ValueError: truth value of an array is ambiguous`` in one shape, a bare
    ``AssertionError`` in another -- from inside ipywidgets' message handler, on a
    pan or a hover rather than on a call.

    Nothing here reads those computed defaults; they exist for
    ``full_figure_for_development()``. Detaching the two observers removes the
    whole failure mode and lets every figure, images included, use the smooth
    in-place update path.

    Returns
    -------
    list of str
        The traits successfully muted, for the tests to check.
    """
    muted = []
    for trait in _ECHO_TRAITS:
        # traitlets registers an @observe-decorated method as the *class-level*
        # ObserveHandler, not as a bound method. ``getattr(view, name)`` hands back
        # a bound method that is not the object in the notifier list, so
        # unobserving it silently removes nothing.
        handler = getattr(type(view), f"_handler{trait}", None)
        if handler is None:  # pragma: no cover - plotly internals moved
            continue
        try:
            view.unobserve(handler, names=trait)
        except (ValueError, KeyError):  # pragma: no cover - plotly internals moved
            continue
        remaining = view._trait_notifiers.get(trait, {}).get("change", [])
        if not any(getattr(entry, "name", None) == f"_handler{trait}"
                   for entry in remaining):
            muted.append(trait)
    return muted


def _sync(view: go.FigureWidget, new: go.Figure) -> None:
    """Copy a freshly built figure into an existing FigureWidget, in one repaint."""
    if len(view.data) != len(new.data):
        raise ValueError(
            f"builder changed the trace count ({len(view.data)} -> {len(new.data)}); "
            "builders must always return the same traces so the figure can update in place"
        )
    with view.batch_update():
        for old_trace, new_trace in zip(view.data, new.data, strict=True):
            old_trace.update(plain({key: val
                                    for key, val in new_trace.to_plotly_json().items()
                                    if key not in ("type", "uid")}))
        view.layout.update(plain({key: val
                                  for key, val in new.layout.to_plotly_json().items()
                                  if key != "template"}))


def reactive(builder: Callable[..., go.Figure], controls: Mapping[str, Control],
             note: str | None = None,
             presets: Mapping[str, Mapping[str, Any]] | None = None,
             static: bool = False, rerender: bool = False):
    """Wire controls to a figure builder and return something displayable.

    Parameters
    ----------
    builder : callable
        Called as ``builder(**values)``, keyed by the names in ``controls``.
        Must return a figure with the same set of traces every time, so the
        figure can be updated in place instead of re-rendered.
    controls : mapping of str to Control
        The controls, in display order. Keys are the builder's keyword arguments.
    note : str, optional
        Caption placed under the figure.
    presets : mapping, optional
        Named sets of control values offered as buttons -- the fastest way to
        walk a reader through the special cases worth knowing.
    static : bool, optional
        Return the figure built from the controls' initial values, with no
        widgets at all. Used by the tests and by static exports.
    rerender : bool, optional
        Force the redraw-from-scratch path. Needed when a control changes the
        *shape* of the figure -- a different set of traces, or a map where there
        was a 3D scene -- which cannot be applied to a live figure in place.

    Returns
    -------
    ipywidgets.Widget or plotly.graph_objects.Figure
    """
    initial = {name: control.value for name, control in controls.items()}
    if static:
        return builder(**initial)

    widgets = require_widgets()
    live = {name: _build_widget(control) for name, control in controls.items()}
    figure = builder(**initial)

    def current() -> dict[str, Any]:
        return {name: widget.value for name, widget in live.items()}

    def make_output_view():
        """Re-render the whole figure on every change. Slower, always correct."""
        out = widgets.Output()

        def redraw(_change=None):
            render_into(out, builder(**current()))

        redraw()
        return out, redraw

    if rerender:
        # The builder returns structurally different figures (a map, then a 3D
        # scene): nothing can be updated in place, so redraw from scratch.
        view, redraw = make_output_view()
    else:
        try:
            # Build from plain values, not numpy arrays -- see plain() for why.
            view = go.FigureWidget(plain(figure.to_plotly_json()))
            _check_widget_state(view)
            _mute_browser_echo(view)

            def redraw(_change=None):
                _sync(view, builder(**current()))

        except Exception:  # pragma: no cover - only if FigureWidget is unavailable
            view, redraw = make_output_view()

    for widget in live.values():
        widget.observe(redraw, names="value")

    rows = [widgets.Box(list(live.values()),
                        layout=widgets.Layout(display="flex", flex_flow="row wrap",
                                              margin="0 0 8px 0"))]
    if presets:
        buttons = []
        for label, settings in presets.items():
            button = widgets.Button(description=label, layout=widgets.Layout(width="auto"))
            button.on_click(_preset_setter(live, settings, redraw))
            buttons.append(button)
        rows.append(widgets.HBox(
            [widgets.HTML(f"<span style='color:{style.TEXT_SECONDARY};"
                          f"font-size:0.86em'>try&nbsp;&nbsp;</span>"), *buttons],
            layout=widgets.Layout(margin="0 0 10px 0")))

    children = [*rows, view] + ([caption(note)] if note else [])
    return widgets.VBox(children)


def _preset_setter(live: Mapping[str, Any], settings: Mapping[str, Any],
                   redraw: Callable[..., None]) -> Callable[[Any], None]:
    """Button callback that sets several controls with a single repaint at the end."""
    def on_click(_button):
        for name in settings:
            live[name].unobserve(redraw, names="value")
        try:
            for name, value in settings.items():
                live[name].value = value
        finally:
            for name in settings:
                live[name].observe(redraw, names="value")
        redraw()
    return on_click
