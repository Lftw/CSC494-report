"""Interactive figures for notebook 03: the corruption chain, and mixed feeds.

Same pattern as the other two: read the controls, call :mod:`jones`, hand the
arrays to :mod:`figures_jones`. Nothing here knows any physics.
"""

from __future__ import annotations

import figures_jones as figs
import jones
import numpy as np
import widgets

__all__ = ["jones_chain_explorer", "rime_explorer", "field_rotation_explorer",
           "leakage_explorer", "slot_explorer", "conversion_explorer",
           "misread_explorer"]

_FEEDS = ("rl", "xy")
_FEED_NAMES = {"rl": "circular (R, L)", "xy": "linear (X, Y)"}


def _feed_dropdown(value: str, description: str) -> widgets.Control:
    return widgets.dropdown(_FEEDS, value, description)


# ---------------------------------------------------------------------------
# The corruption chain
# ---------------------------------------------------------------------------

def jones_chain_explorer(static: bool = False):
    """Build one station's Jones matrix a factor at a time.

    Everything a station does to the signal is in these four numbers. Turn the
    gains off centre and the whole matrix scales; add leakage and the
    off-diagonal entries fill in; rotate the field and watch how differently the
    two feed types respond.
    """
    controls = {
        "feed_type": _feed_dropdown("rl", "feeds"),
        "gain_amp": widgets.slider(1.0, 0.5, 1.5, 0.05, "gain amplitude |g|"),
        "gain_phase_deg": widgets.slider(0.0, -180.0, 180.0, 10.0,
                                         "gain phase (deg)", ".0f"),
        "d_amp": widgets.slider(0.02, 0.0, 0.2, 0.005, "leakage |d|", ".3f"),
        "d_phase_deg": widgets.slider(0.0, -180.0, 180.0, 10.0,
                                      "leakage phase (deg)", ".0f"),
        "rotation_deg": widgets.slider(30.0, -90.0, 90.0, 5.0,
                                       "field rotation (deg)", ".0f"),
    }

    def build(feed_type, gain_amp, gain_phase_deg, d_amp, d_phase_deg, rotation_deg):
        gain = gain_amp * np.exp(1j * np.deg2rad(gain_phase_deg))
        leak = d_amp * np.exp(1j * np.deg2rad(d_phase_deg))
        return figs.jones_chain_figure(gain, gain, leak, np.conj(leak), feed_type,
                                       np.deg2rad(rotation_deg))

    return widgets.reactive(
        build, controls, static=static,
        note="Cells are shaded by magnitude and labelled with the complex value. The "
             "product on the right is what actually multiplies the sky. Note that the "
             "gain never moves signal between feeds (it is diagonal) while leakage and "
             "field rotation both do, which is why gains can be calibrated with "
             "closure quantities and the other two cannot.")


def rime_explorer(static: bool = False):
    """A whole baseline: sky, corruption, and what calibration can undo.

    The measurement equation is ``V = J₁ C J₂†``. Corrupt the two stations
    differently and the recovered Stokes vector is wrong. Then note that
    inverting the *known* Jones matrices recovers it exactly. Calibration is easy
    when you know the answer; the hard part is that you never do.
    """
    controls = {
        "p_lin": widgets.slider(0.3, 0.0, 0.9, 0.05, "source p_lin"),
        "evpa_deg": widgets.slider(30.0, -90.0, 90.0, 5.0, "source EVPA (deg)", ".0f"),
        "d_amp": widgets.slider(0.06, 0.0, 0.3, 0.01, "leakage |d| (both ends)", ".2f"),
        "rotation_deg": widgets.slider(25.0, -90.0, 90.0, 5.0,
                                       "field rotation (deg)", ".0f"),
        "feed_type": _feed_dropdown("rl", "feeds (both ends)"),
    }

    def build(p_lin, evpa_deg, d_amp, rotation_deg, feed_type):
        import polarization as pol
        truth = pol.stokes_from_ellipse(1.0, p_lin, np.deg2rad(evpa_deg), 0.0)
        corrupt = jones.jones_matrix(1.0, 1.0, d_amp, d_amp, feed_type,
                                     np.deg2rad(rotation_deg))
        slots = jones.observe(truth, feed_type, feed_type, corrupt, corrupt)
        naive = jones.stokes_from_slots(slots, feed_type, feed_type)
        return figs.stokes_comparison_figure(
            truth, naive, "The measurement equation, uncorrected",
            "what the sky has, against what the four correlations say it has",
            names=("on the sky", "read straight off the correlator"))

    return widgets.reactive(
        build, controls, static=static,
        note="Stokes I barely moves: total intensity is robust. Q and U are wrecked by "
             "a few percent of leakage, because they are small differences of large "
             "numbers. That asymmetry is why polarimetry needs calibration an order of "
             "magnitude better than total-intensity imaging.")


def field_rotation_explorer(static: bool = False):
    """The same physical rotation, in a circular and a linear feed basis.

    Nothing about the sky differs between the two panels. The correction a station
    has to apply does, and if the two ends of a baseline are of different types,
    the two corrections are not even the same kind of operation.
    """
    controls = {"rotation_deg": widgets.slider(40.0, -90.0, 90.0, 5.0,
                                               "field rotation (deg)", ".0f")}

    def build(rotation_deg):
        return figs.field_rotation_figure(np.deg2rad(rotation_deg))

    return widgets.reactive(
        build, controls, static=static,
        note="Circular feeds: the matrix stays diagonal, so each feed only picks up a "
             "phase and the amplitudes |R| and |L| never change. Linear feeds: a real "
             "rotation that mixes the feeds, which mixes Stokes Q and U by twice the "
             "angle. Same rotation, two different-looking problems.")


def leakage_explorer(static: bool = False):
    """Leakage manufacturing polarization out of nothing.

    Set the source's polarization to zero and drag the leakage up. Every tick that
    appears is instrumental. This is the systematic that sets the floor on EHT
    polarimetry, and the reason D-terms have to be solved to a fraction of a
    percent before a polarized image means anything.
    """
    controls = {
        "d_amp": widgets.slider(0.05, 0.0, 0.2, 0.005, "leakage |d|", ".3f"),
        "d_phase_deg": widgets.slider(0.0, -180.0, 180.0, 15.0,
                                      "leakage phase (deg)", ".0f"),
        "p_lin": widgets.slider(0.0, 0.0, 0.4, 0.05, "source polarization"),
        "feed_type": _feed_dropdown("rl", "feeds"),
    }

    def build(d_amp, d_phase_deg, p_lin, feed_type):
        leak = d_amp * np.exp(1j * np.deg2rad(d_phase_deg))
        return figs.leakage_image_figure(leak, np.conj(leak), feed_type, p_lin=p_lin)

    return widgets.reactive(
        build, controls, static=static,
        note="With the source polarization at zero, the subtitle is the whole story: a "
             "few percent of leakage produces a few percent of polarization, laid out in "
             "a smooth, plausible pattern that looks nothing like noise.")


# ---------------------------------------------------------------------------
# Mixed feeds
# ---------------------------------------------------------------------------

def slot_explorer(static: bool = False):
    """What a baseline writes to disk, for any pairing of feed types.

    Two circular stations give RR, LL, RL, LR. Two linear stations give XX, YY,
    XY, YX. One of each gives XR, YL, XL, YR, four products that are neither,
    and that no single-basis imaging code has a rule for.
    """
    controls = {"feed1": _feed_dropdown("xy", "station 1 feeds"),
                "feed2": _feed_dropdown("rl", "station 2 feeds")}

    def build(feed1, feed2):
        return figs.slot_grid_figure(feed1, feed2)

    return widgets.reactive(
        build, controls, static=static,
        presets={"both circular": {"feed1": "rl", "feed2": "rl"},
                 "both linear": {"feed1": "xy", "feed2": "xy"},
                 "mixed (ALMA × SMA)": {"feed1": "xy", "feed2": "rl"}},
        note="eht-imaging stores this as a per-row <code>polbasis</code> string, and calls "
             "a dataset containing more than one of them <code>polrep='mixed'</code>. A "
             "homogeneous dataset has one value here for every row; that assumption is "
             "what the mixed-pol work had to remove.")


def conversion_explorer(static: bool = False):
    """The 4x4 matrix that turns this baseline's four numbers into I, Q, U, V.

    In a homogeneous array it is one matrix for the whole dataset, applied once.
    In a mixed array every baseline needs its own, which is the difference
    between a global conversion and a per-row one.
    """
    controls = {"feed1": _feed_dropdown("xy", "station 1 feeds"),
                "feed2": _feed_dropdown("rl", "station 2 feeds")}

    def build(feed1, feed2):
        return figs.slots_to_stokes_figure(feed1, feed2)

    return widgets.reactive(
        build, controls, static=static,
        presets={"both circular": {"feed1": "rl", "feed2": "rl"},
                 "both linear": {"feed1": "xy", "feed2": "xy"},
                 "mixed (ALMA × SMA)": {"feed1": "xy", "feed2": "rl"}},
        note="Compare the two homogeneous cases: circular feeds put I and V on the "
             "parallel-hand slots and Q, U on the cross-hands, linear feeds do the "
             "opposite. The mixed case has no such clean split: every Stokes parameter "
             "draws on every slot.")


def misread_explorer(static: bool = False):
    """What happens when the software assumes the wrong basis. The money plot.

    Feed a known sky through a baseline with real feed types, then convert the
    result using the matrix for a *different* pairing, which is exactly what a
    pipeline written for a homogeneous array does to an ALMA baseline. Nothing
    crashes. The numbers are simply wrong, and nothing downstream can tell.
    """
    controls = {
        "p_lin": widgets.slider(0.3, 0.0, 0.9, 0.05, "source p_lin"),
        "evpa_deg": widgets.slider(30.0, -90.0, 90.0, 5.0, "source EVPA (deg)", ".0f"),
        "p_circ": widgets.slider(0.0, -0.4, 0.4, 0.05, "source p_circ"),
        "true1": _feed_dropdown("xy", "station 1 really has"),
        "true2": _feed_dropdown("rl", "station 2 really has"),
        "assumed": _feed_dropdown("rl", "software assumes both are"),
    }

    def build(p_lin, evpa_deg, p_circ, true1, true2, assumed):
        import polarization as pol
        truth = pol.stokes_from_ellipse(1.0, p_lin, np.deg2rad(evpa_deg), p_circ)
        recovered = jones.misread_stokes(truth, (true1, true2), (assumed, assumed))
        honest = jones.is_mixed(true1, true2) or true1 != assumed
        return figs.stokes_comparison_figure(
            truth, recovered,
            "Reading a baseline in the wrong basis",
            (f"true feeds {true1.upper()} × {true2.upper()}, interpreted as "
             f"{assumed.upper()} × {assumed.upper()}"
             + ("" if honest else " (which is correct, so nothing moves)")),
            names=("on the sky", "what the pipeline reports"))

    return widgets.reactive(
        build, controls, static=static,
        presets={"correct (all circular)": {"true1": "rl", "true2": "rl",
                                            "assumed": "rl"},
                 "ALMA read as circular": {"true1": "xy", "true2": "rl",
                                           "assumed": "rl"},
                 "all linear read as circular": {"true1": "xy", "true2": "xy",
                                                 "assumed": "rl"}},
        note="Set the source to pure linear polarization and read an all-linear array as "
             "circular: Stokes V picks up the entire signal. Circular polarization in "
             "M87 is at the few-tenths-of-a-percent level, so an error of this kind does "
             "not look like a bug. It looks like a discovery.")
