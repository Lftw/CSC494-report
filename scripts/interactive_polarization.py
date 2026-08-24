"""Interactive figures for notebook 01. One explorer per idea.

Each function here is a thin composition: read control values, call
:mod:`the physics modules`, hand the arrays to :mod:`the figure modules`. Nothing
in this module knows any physics, and nothing in ``core`` knows a widget exists.

All explorers accept ``static=True`` to return a plain figure instead of a
widget, so the notebooks still say something on GitHub's static preview.
"""

from __future__ import annotations

import figures_polarization as figs
import images as im
import numpy as np
import polarization as pol
import style
import widgets

__all__ = ["wave_explorer", "wave_3d_explorer", "basis_explorer",
           "depolarization_explorer", "stokes_explorer", "poincare_explorer",
           "synchrotron_explorer", "faraday_explorer", "image_explorer"]


# ---------------------------------------------------------------------------
# Beats 1-2: any wiggle is two components and a phase offset
# ---------------------------------------------------------------------------

def wave_explorer(static: bool = False):
    """Sliders on ``(a_x, a_y, delta)`` -> the polarization ellipse and the two components.

    The single most important control is ``delta``: with the amplitudes equal, it
    walks the state from linear (0), through elliptical, to circular (90), to the
    other linear (180) and the other circular (270). Nothing about the *amount*
    of light changes -- only its geometry.
    """
    controls = {
        "amp_x": widgets.slider(1.0, 0.0, 1.0, 0.05, "x amplitude a_x"),
        "amp_y": widgets.slider(1.0, 0.0, 1.0, 0.05, "y amplitude a_y"),
        "delta_deg": widgets.slider(45.0, -180.0, 180.0, 5.0, "phase offset δ (deg)", ".0f"),
        "phase_deg": widgets.slider(0.0, 0.0, 355.0, 5.0, "time ωt (deg)", ".0f"),
    }

    def build(amp_x, amp_y, delta_deg, phase_deg):
        field = pol.jones_vector(amp_x, amp_y, np.deg2rad(delta_deg))
        # The figure sets its own headline and subtitle; don't overwrite them.
        return figs.wave_ellipse_figure(field, phase_marker=np.deg2rad(phase_deg))

    return widgets.reactive(
        build, controls, static=static,
        presets={
            "linear, horizontal": {"amp_x": 1.0, "amp_y": 0.0, "delta_deg": 0.0},
            "linear, 45°": {"amp_x": 1.0, "amp_y": 1.0, "delta_deg": 0.0},
            "right circular": {"amp_x": 1.0, "amp_y": 1.0, "delta_deg": -90.0},
            "left circular": {"amp_x": 1.0, "amp_y": 1.0, "delta_deg": 90.0},
            "elliptical": {"amp_x": 1.0, "amp_y": 0.55, "delta_deg": 60.0},
        },
        note="The ellipse is the whole story: its tilt is the EVPA, its fatness the "
             "ellipticity, and the direction it is traced is the handedness. "
             "Right circular is δ = −90° in this convention, where the field turns "
             "counter-clockwise in the plane as drawn.")


def wave_3d_explorer(static: bool = False):
    """The wave travelling through space: a flat plane, or a corkscrew.

    Starts flat, so the reader is the one who opens it into a helix. Makes the
    word "transverse" concrete, and shows where the polarization ellipse comes
    from: it is the corkscrew seen end-on.
    """
    controls = {
        "amp_x": widgets.slider(1.0, 0.0, 1.0, 0.05, "x amplitude a_x"),
        "amp_y": widgets.slider(1.0, 0.0, 1.0, 0.05, "y amplitude a_y"),
        "delta_deg": widgets.slider(0.0, -180.0, 180.0, 5.0, "phase offset δ (deg)", ".0f"),
        "n_periods": widgets.slider(2.0, 1.0, 5.0, 0.5, "wavelengths shown", ".1f"),
    }

    def build(amp_x, amp_y, delta_deg, n_periods):
        field = pol.jones_vector(amp_x, amp_y, np.deg2rad(delta_deg))
        return figs.wave_3d_figure(field, n_periods=n_periods)

    return widgets.reactive(
        build, controls, static=static,
        presets={"flat (linear)": {"amp_x": 1.0, "amp_y": 1.0, "delta_deg": 0.0},
                 "corkscrew (right circular)": {"amp_x": 1.0, "amp_y": 1.0,
                                                "delta_deg": -90.0},
                 "corkscrew (left circular)": {"amp_x": 1.0, "amp_y": 1.0,
                                               "delta_deg": 90.0},
                 "stretched": {"amp_x": 1.0, "amp_y": 0.5, "delta_deg": -60.0}},
        note="It starts flat: at δ = 0 the wave stays in a single plane. Drag δ toward "
             "±90° and the plane opens into a corkscrew. The blue curve on the floor is "
             "the E_x component on its own, the orange one on the back wall is E_y, and "
             "the green loop at the near face is the polarization ellipse, the corkscrew "
             "seen end-on. Drag the figure itself to rotate it.")


# ---------------------------------------------------------------------------
# Two bases, one wave: the idea everything later depends on
# ---------------------------------------------------------------------------

def basis_explorer(static: bool = False):
    """The same wave read out by linear (X, Y) feeds and by circular (R, L) feeds.

    Set the state to pure right circular: the circular feeds report all of the
    light in one number and nothing in the other, while the linear feeds split it
    evenly and hide the whole story in a phase difference. Same light, two
    descriptions -- and telescopes that disagree about which description to use
    are what the later notebooks are about.
    """
    controls = {
        "amp_x": widgets.slider(1.0, 0.0, 1.0, 0.05, "x amplitude a_x"),
        "amp_y": widgets.slider(1.0, 0.0, 1.0, 0.05, "y amplitude a_y"),
        "delta_deg": widgets.slider(-90.0, -180.0, 180.0, 5.0, "phase offset δ (deg)", ".0f"),
    }

    def build(amp_x, amp_y, delta_deg):
        field = pol.jones_vector(amp_x, amp_y, np.deg2rad(delta_deg))
        return figs.basis_bars_figure(field)

    return widgets.reactive(
        build, controls, static=static,
        presets={
            "right circular": {"amp_x": 1.0, "amp_y": 1.0, "delta_deg": -90.0},
            "left circular": {"amp_x": 1.0, "amp_y": 1.0, "delta_deg": 90.0},
            "linear, horizontal": {"amp_x": 1.0, "amp_y": 0.0, "delta_deg": 0.0},
        },
        note="R = (X + iY)/√2 and L = (X − iY)/√2, a rotation of the description and "
             "not of the light. A circular-feed dish and a linear-feed dish record different "
             "numbers for identical incoming waves.")


# ---------------------------------------------------------------------------
# Beats 4-6: partial polarization, Stokes, Poincare
# ---------------------------------------------------------------------------

def depolarization_explorer(static: bool = False):
    """How ordered the emitters have to be for polarization to survive averaging.

    Real light is not one wave. It is an incoherent pile of them, and only their
    Stokes parameters add. Spread the emitters' angles by a few tens of degrees
    and most of the polarization cancels -- which is why the EHT measures a few
    tens of percent, not 100%, and why *net* polarization from an unresolved
    source is such a weak measurement.
    """
    controls = {
        "spread_deg": widgets.slider(20.0, 0.0, 90.0, 2.0, "spread of angles (deg)", ".0f"),
        "n_waves": widgets.int_slider(400, 10, 2000, 10, "number of wavelets"),
    }

    def build(spread_deg, n_waves):
        return figs.depolarization_figure(np.deg2rad(spread_deg), n_waves=n_waves)

    return widgets.reactive(build, controls, static=static,
                        note="Stokes parameters add; field amplitudes do not. This is the "
                             "reason the honest object is the coherency matrix ⟨EE†⟩, whose "
                             "four real degrees of freedom are exactly I, Q, U, V.")


def stokes_explorer(static: bool = False):
    """Drive ``(p_lin, EVPA, p_circ)`` and watch Stokes, the ellipse and the sphere agree.

    Three sliders, three views of one state. The constraint
    ``I² ≥ Q² + U² + V²`` is not decoration: push the fractions past unity in
    quadrature and the state stops existing, which the explorer refuses to draw.
    """
    controls = {
        "p_lin": widgets.slider(0.6, 0.0, 1.0, 0.02, "linear fraction p_lin"),
        "evpa_deg": widgets.slider(30.0, -90.0, 90.0, 5.0, "EVPA (deg)", ".0f"),
        "p_circ": widgets.slider(0.2, -1.0, 1.0, 0.02, "circular fraction p_circ"),
    }

    def build(p_lin, evpa_deg, p_circ):
        norm = np.hypot(p_lin, p_circ)
        if norm > 1.0:  # keep the state physical rather than raising at the reader
            p_lin, p_circ = p_lin / norm, p_circ / norm
        stokes = pol.stokes_from_ellipse(1.0, p_lin, np.deg2rad(evpa_deg), p_circ)
        return figs.stokes_bars_figure(stokes)

    return widgets.reactive(
        build, controls, static=static,
        note="EVPA = ½·arctan(U/Q). The factor of one half is why a position angle of 0° "
             "and 180° are the same state, and why polarimetric calibration errors so "
             "often show up as a 90° flip.")


def poincare_explorer(static: bool = False):
    """The Poincare sphere: every polarization state, once.

    Same three sliders as :func:`stokes_explorer`, plotted as a point. Surface is
    fully polarized, centre unpolarized, equator linear, poles circular.
    """
    controls = {
        "p_lin": widgets.slider(0.6, 0.0, 1.0, 0.02, "linear fraction p_lin"),
        "evpa_deg": widgets.slider(30.0, -90.0, 90.0, 5.0, "EVPA (deg)", ".0f"),
        "p_circ": widgets.slider(0.2, -1.0, 1.0, 0.02, "circular fraction p_circ"),
    }

    def build(p_lin, evpa_deg, p_circ):
        norm = np.hypot(p_lin, p_circ)
        if norm > 1.0:
            p_lin, p_circ = p_lin / norm, p_circ / norm
        stokes = pol.stokes_from_ellipse(1.0, p_lin, np.deg2rad(evpa_deg), p_circ)
        return figs.poincare_figure(stokes)

    return widgets.reactive(build, controls, static=static,
                        note="Walking the EVPA slider through 180° carries the point all the "
                             "way around the equator: 360° of azimuth for 180° of angle.")


# ---------------------------------------------------------------------------
# Beats 7-8: where polarization comes from, and what happens on the way out
# ---------------------------------------------------------------------------

def synchrotron_explorer(static: bool = False):
    """Magnetic field direction in, EVPA ticks out. The reason polarimetry exists.

    Relativistic electrons spiralling around a magnetic field radiate with their
    electric vector perpendicular to the field's projection on the sky (in the
    optically thin case). So an EVPA map *is* a magnetic field map, rotated by
    90° -- the only way we have of seeing the field that launches a jet.
    """
    controls = {
        "b_angle_deg": widgets.slider(0.0, -90.0, 90.0, 5.0, "field angle on sky (deg)", ".0f"),
        "p_lin": widgets.slider(0.3, 0.0, 0.75, 0.05, "polarization fraction"),
        "optically_thick": widgets.toggle(False, "optically thick (EVPA flips 90°)"),
    }

    def build(b_angle_deg, p_lin, optically_thick):
        evpa_deg = b_angle_deg + (0.0 if optically_thick else 90.0)
        image = im.gaussian_blob(npix=96, fov_uas=100.0, fwhm_uas=45.0,
                                p_lin=p_lin, evpa_deg=evpa_deg)
        figure = figs.pol_image_figure(
            image, tick_step=8, tick_scale=2.2, i_cut=0.15, length_mode="fraction",
            title="Synchrotron emission: the EVPA is perpendicular to B")
        # Draw the field direction itself, for comparison with the ticks. Note the
        # RA axis is reversed, so +x is to the *left*: anchor the label accordingly.
        angle = np.deg2rad(b_angle_deg)
        span = 38.0
        end = (span * np.cos(angle), span * np.sin(angle))
        figure.add_shape(type="line", x0=-end[0], y0=-end[1], x1=end[0], y1=end[1],
                         line={"color": style.FEED_CIRCULAR, "width": 2.5, "dash": "dash"})
        figure.add_annotation(x=end[0], y=end[1], text="B", showarrow=False,
                              xanchor="right", yanchor="bottom", xshift=-4,
                              font={"size": 13, "color": style.FEED_CIRCULAR})
        return figure

    return widgets.reactive(
        build, controls, static=static,
        note="Dashed blue is the magnetic field; black ticks are what a telescope measures. "
             "Optically thick emission flips the relationship by 90°, so knowing which "
             "regime you are in matters before you claim a field geometry.")


def faraday_explorer(static: bool = False):
    """Rotation measure in, EVPA-versus-λ² line out.

    Magnetised plasma between us and the source rotates the EVPA by RM·λ².
    Because the effect is chromatic, it is measured by observing several
    frequencies -- and every frequency channel has to be handled with the same
    polarization machinery. That is what makes multi-frequency and mixed
    polarization the same project.
    """
    controls = {
        "rm_rad_m2": widgets.slider(50000.0, -500000.0, 500000.0, 10000.0,
                                "rotation measure (rad/m²)", ".0f"),
        "evpa0_deg": widgets.slider(30.0, -90.0, 90.0, 5.0, "intrinsic EVPA (deg)", ".0f"),
    }

    def build(rm_rad_m2, evpa0_deg):
        return figs.faraday_figure(rm_rad_m2, evpa0_deg=evpa0_deg)

    return widgets.reactive(build, controls, static=static,
                        note="M87's core shows |RM| of order 10⁵ rad/m². At 230 GHz "
                             "(λ ≈ 1.3 mm) that is a few degrees of rotation; at 86 GHz it is "
                             "an order of magnitude more, which is the leverage.")


# ---------------------------------------------------------------------------
# Beat 9: the thing we are actually trying to reconstruct
# ---------------------------------------------------------------------------

def image_explorer(static: bool = False):
    """A polarized ring, drawn the way the EHT draws one.

    This is the target: not a number, an *image* of four Stokes parameters. The
    ``pitch`` slider sweeps the EVPA pattern from radial to azimuthal -- for M87
    the measured spiral is what argued for a dynamically important, partly
    poloidal field.
    """
    controls = {
        "p_lin": widgets.slider(0.25, 0.0, 0.7, 0.05, "polarization fraction"),
        "pitch_deg": widgets.slider(45.0, 0.0, 90.0, 5.0, "EVPA pitch angle (deg)", ".0f"),
        "asymmetry": widgets.slider(0.4, 0.0, 0.9, 0.05, "brightness asymmetry"),
        "tick_step": widgets.int_slider(6, 3, 14, 1, "tick spacing (pixels)"),
        "tick_scale": widgets.slider(6.0, 1.0, 16.0, 0.5, "tick length"),
        "i_cut": widgets.slider(0.1, 0.0, 0.5, 0.02, "hide ticks below (× peak)"),
    }

    def build(p_lin, pitch_deg, asymmetry, tick_step, tick_scale, i_cut):
        image = im.polarized_ring(npix=128, fov_uas=100.0, p_lin=p_lin,
                                  pitch_deg=pitch_deg, asymmetry=asymmetry)
        totals = im.image_stokes_totals(image)
        net = totals["p_lin_net"]
        figure = figs.pol_image_figure(image, tick_step=tick_step, tick_scale=tick_scale,
                                     i_cut=i_cut,
                                     title="A polarized image: what we are trying to make")
        style.set_title(
            figure, "A polarized image: what we are trying to make",
            f"every pixel is {totals['p_lin_mean']:.0%} polarized, but the whole ring "
            f"nets only {'<0.1%' if net < 0.001 else format(net, '.1%')}")
        return figure

    return widgets.reactive(
        build, controls, static=static,
        note="Note the title: opposite sides of the ring cancel, so an unresolved "
             "measurement would see almost no polarization at all. Resolving the "
             "polarization structure is the entire point of doing this with an array.")
