"""Interactive figures for notebook 02: the array, its coverage, and its images.

Same pattern as :mod:`polarization` -- read control values,
call ``core``, hand the arrays to ``viz``. Anything that recomputes a full image
uses non-continuous sliders, so the figure updates when you let go of the handle
rather than sixty times on the way.
"""

from __future__ import annotations

import arrays as arr
import figures_telescope as figs
import images as im
import interferometry as itf
import numpy as np
import style
import widgets

__all__ = ["resolution_explorer", "fringe_explorer", "fourier_explorer",
           "visibility_profile_explorer", "array_explorer", "visibility_window_explorer",
           "coverage_explorer", "imaging_explorer", "field_rotation_explorer",
           "closure_explorer", "noise_explorer", "MJD_2017_APR11"]

#: The night the EHT observed M87. Every figure in notebook 02 uses it.
MJD_2017_APR11 = 57854.0

_SOURCES = ("M87", "SgrA*")


def _model_image(model: str, npix: int = 96, fov_uas: float = 200.0,
                 diameter_uas: float = 42.0) -> dict:
    """The three sky models notebook 02 plays with."""
    if model == "ring":
        return im.polarized_ring(npix=npix, fov_uas=fov_uas, diameter_uas=diameter_uas,
                                 width_uas=0.28 * diameter_uas, asymmetry=0.4)
    if model == "blob":
        return im.gaussian_blob(npix=npix, fov_uas=fov_uas, fwhm_uas=diameter_uas)
    if model == "double":
        left = im.gaussian_blob(npix=npix, fov_uas=fov_uas, fwhm_uas=0.3 * diameter_uas)
        image = dict(left)
        shift = max(1, int(npix * diameter_uas / fov_uas / 2))
        image["I"] = 0.6 * np.roll(left["I"], shift, axis=1) + \
            0.4 * np.roll(left["I"], -shift, axis=1)
        return image
    raise ValueError(f"unknown model {model!r}")


# ---------------------------------------------------------------------------
# Part A: one baseline
# ---------------------------------------------------------------------------

def resolution_explorer(static: bool = False):
    """How big an aperture do you need? Drag the wavelength and find out.

    The EHT observes at 1.3 mm not because that is convenient -- it is a
    miserable wavelength, absorbed by water vapour, which is why the stations
    sit on volcanoes and in deserts -- but because resolution scales with
    ``lambda / D`` and ``D`` is already capped at the size of the planet.
    """
    controls = {
        "wavelength_mm": widgets.slider(1.3, 0.4, 20.0, 0.1, "wavelength (mm)"),
        "aperture_km": widgets.slider(1.0e4, 1.0, 2.0e4, 100.0, "aperture (km)", ".0f"),
    }
    return widgets.reactive(figs.resolution_figure, controls, static=static,
                        presets={"a single dish": {"aperture_km": 0.012},
                                 "the whole Earth": {"aperture_km": 1.274e4}},
                        note="The blue line is the diffraction limit itself, "
                             "<b>θ = λ / D</b>: pick an aperture on the horizontal axis "
                             "and read off the smallest thing it can resolve. Both axes "
                             "are logarithmic, so the straight line is the inverse "
                             "relationship. The green marker is where that line crosses "
                             "M87's ring, the aperture the EHT had to have. To put the "
                             "ring on more than a couple of resolution elements you need "
                             "the Earth, which is the sentence that created this "
                             "instrument.")


def fringe_explorer(static: bool = False):
    """One pair of dishes: a comb of fringes laid across the figs.

    Short baselines have wide fringes and respond to the source's total flux;
    long baselines have fine fringes and respond only to structure at that
    scale. An array is a set of dishes chosen to give a useful spread of comb
    spacings.
    """
    controls = {
        "baseline_km": widgets.slider(3000.0, 20.0, 11000.0, 20.0, "baseline (km)", ".0f"),
        "wavelength_m": widgets.slider(1.3e-3, 0.4e-3, 3.5e-3, 0.1e-3, "wavelength (m)",
                                   ".4f"),
    }
    return widgets.reactive(figs.fringe_figure, controls, static=static,
                        note="The orange bar is M87's ring. When the fringe spacing is "
                             "much larger than the source, every part of the source sits "
                             "on the same part of the comb and the response saturates: "
                             "that baseline learns only the total flux.")


def fourier_explorer(static: bool = False):
    """Drag a baseline around and watch which Fourier component it measures.

    The single most important idea in interferometry: **a baseline does not see
    a piece of the picture, it sees a piece of the picture's Fourier
    transform.** Length sets the fringe spacing, orientation sets the fringe
    direction, and the correlator reports how strongly the sky resembles that
    pattern -- as one complex number.
    """
    controls = {
        "model": widgets.dropdown(("ring", "blob", "double"), "ring", "sky model"),
        "length_glambda": widgets.slider(3.0, 0.1, 8.5, 0.1, "baseline length (Gλ)"),
        "angle_deg": widgets.slider(0.0, -90.0, 90.0, 5.0, "baseline angle (deg)", ".0f"),
    }

    def build(model, length_glambda, angle_deg):
        angle = np.deg2rad(angle_deg)
        u = length_glambda * 1e9 * np.cos(angle)
        v = length_glambda * 1e9 * np.sin(angle)
        # A tighter field of view so the source fills the panel it is drawn in.
        return figs.fourier_component_figure(_model_image(model, fov_uas=140.0), u, v)

    return widgets.reactive(build, controls, static=static,
                        note="Watch the amplitude collapse when the fringe spacing matches "
                             "the ring diameter: the source lines up with the positive and "
                             "negative stripes equally, and cancels. That null is how the "
                             "EHT measured the ring's size.")


def visibility_profile_explorer(static: bool = False):
    """Ring diameter in, null position out. The measurement, in one curve.

    The EHT's headline number -- a 42 microarcsecond ring -- is essentially a
    reading of where this curve dips.
    """
    controls = {
        "diameter_uas": widgets.slider(42.0, 15.0, 90.0, 1.0, "ring diameter (µas)",
                                   ".0f"),
        "model": widgets.dropdown(("ring", "blob", "double"), "ring", "sky model"),
    }

    def build(diameter_uas, model):
        image = _model_image(model, npix=128, fov_uas=300.0, diameter_uas=diameter_uas)
        return figs.visibility_profile_figure(image)

    return widgets.reactive(build, controls, static=static,
                        note="A smooth blob has no null at all; its visibility just falls "
                             "away. Sharp edges and holes are what put structure into the "
                             "curve, and structure in the curve is what long baselines can "
                             "measure.")


# ---------------------------------------------------------------------------
# Part B: an array on a spinning planet
# ---------------------------------------------------------------------------

def array_explorer(static: bool = False):
    """The 2017 array through one night, seen from the source.

    An orthographic map centred on the point where the source is at the zenith,
    so the stations you can see are the stations that can see it. Run the clock:
    the Earth turns underneath a fixed source, stations rise and set, and the set
    of live baselines is different every hour.

    Coastlines make this readable in a way a bare sphere is not, which is why the
    3D alternative in :func:`figures_telescope.array_globe_3d` is no longer wired
    to a dropdown here. It draws without the network, though -- see the caption.
    """
    controls = {
        "hour": widgets.slider(4.0, 0.0, 24.0, 0.25, "hours UT", ".2f"),
        "source": widgets.dropdown(_SOURCES, "M87", "source"),
        "show_baselines": widgets.toggle(True, "draw live baselines"),
    }

    def build(hour, source, show_baselines):
        ra, dec = arr.SOURCES[source]
        mjd = MJD_2017_APR11 + hour / 24.0
        return figs.array_map(arr.array_2017(), mjd, ra, dec,
                              show_baselines=show_baselines,
                              title=f"{source} at {hour:.2f} h UT")

    return widgets.reactive(build, controls, static=static,
                        note=figs.feed_legend_note() + " Marker size tracks collecting "
                             "area, and ALMA is 37 dishes phased together, which is "
                             "why it dominates the array's sensitivity. A station fades "
                             "out when the source drops below its horizon; the grey "
                             "chords are the baselines live at that moment. Two pairs "
                             "share a mountain, and so share a point on the map: ALMA "
                             "and APEX on Chajnantor, 2.6 km apart, and SMA and JCMT on "
                             "Mauna Kea, 164 m apart. Each pair gets one label above the "
                             "point and one below. ALMA and APEX are the shortest "
                             "mixed-feed baseline in the array. <b>The first draw needs a "
                             "network connection</b>: plotly fetches the "
                             "coastline data from its CDN, and on a restricted link it "
                             "can stall. If the map never appears, "
                             "<code>figures_telescope.array_globe_3d</code> draws the "
                             "same array with nothing external.")


def visibility_window_explorer(static: bool = False):
    """Who is on the air, when.

    A global array is never global all at once. For M87 the South Pole is
    permanently blind, and the useful overlap between Hawaii and Spain is a
    couple of hours -- which is why the uv coverage looks the way it does.
    """
    controls = {"source": widgets.dropdown(_SOURCES, "M87", "source"),
                "elev_min_deg": widgets.slider(10.0, 0.0, 30.0, 1.0,
                                           "elevation limit (deg)", ".0f")}

    def build(source, elev_min_deg):
        ra, dec = arr.SOURCES[source]
        return figs.elevation_strip(arr.array_2017(), MJD_2017_APR11,
                                   np.linspace(0, 24, 289), ra, dec,
                                   elev_min_deg=elev_min_deg)

    return widgets.reactive(build, controls, static=static,
                        note="Raising the elevation limit is what bad weather effectively "
                             "does: low-elevation observations look through more "
                             "atmosphere, and dropping them costs baselines.")


def coverage_explorer(static: bool = False):
    """Turn stations on and off, and watch the uv coverage they buy.

    Each baseline draws an arc as the Earth turns. Eight stations give 28
    baselines and still leave most of the plane empty -- and the empty regions
    are, precisely, the structures the data cannot constrain.
    """
    array = arr.array_2017()
    controls = {f"use_{code}": widgets.toggle(True, code) for code in array}
    controls["source"] = widgets.dropdown(_SOURCES, "M87", "source")

    def build(source, **flags):
        ra, dec = arr.SOURCES[source]
        chosen = {code: site for code, site in array.items() if flags[f"use_{code}"]}
        if len(chosen) < 2:
            chosen = array
        coverage = arr.uv_coverage(chosen, MJD_2017_APR11, np.linspace(0, 24, 145), ra, dec)
        beam = (f"{itf.beam_size_uas(coverage['u'], coverage['v']):.0f} µas beam"
                if len(coverage["u"]) else "nothing observable")
        return figs.uv_coverage_figure(
            coverage, title=f"{len(chosen)} stations  ·  {len(coverage['u'])} samples"
                            f"  ·  {beam}")

    return widgets.reactive(build, controls, static=static,
                        note="Turn ALMA off and the plane empties out, since it anchors "
                             "almost every long baseline. Turn off everything but the two "
                             "Hawaii dishes and you are left with a single very short "
                             "baseline that measures nothing but total flux.")


def imaging_explorer(static: bool = False):
    """Truth, dirty beam, dirty image -- with the array you choose.

    The dirty image is what the data looks like drawn in the image plane. It is
    not a picture of the sky and it never will be: the beam's sidelobes are
    permanently baked in, and removing them requires assuming something about
    what a sky is allowed to look like.
    """
    array = arr.array_2017()
    controls = {f"use_{code}": widgets.toggle(True, code) for code in array}
    controls["model"] = widgets.dropdown(("ring", "blob", "double"), "ring", "sky model")

    def build(model, **flags):
        ra, dec = arr.SOURCES["M87"]
        chosen = {code: site for code, site in array.items() if flags[f"use_{code}"]}
        if len(chosen) < 2:
            chosen = array
        coverage = arr.uv_coverage(chosen, MJD_2017_APR11, np.linspace(0, 24, 97), ra, dec)
        truth = _model_image(model, npix=64, fov_uas=200.0)
        vis = itf.sample_visibilities(truth, coverage["u"], coverage["v"])
        dirty = itf.dirty_image(vis, coverage["u"], coverage["v"], npix=64, fov_uas=200.0)
        beam = itf.dirty_beam(coverage["u"], coverage["v"], npix=64, fov_uas=200.0)
        return figs.sampling_figure(truth, dirty, beam)

    return widgets.reactive(build, controls, static=static,
                        note="Drop stations and watch the beam's sidelobes grow into "
                             "structure that is not in the figs. Everything called "
                             "“imaging” is the job of deciding which features "
                             "of the middle panel you are allowed to believe.")


# ---------------------------------------------------------------------------
# Part C: what a station records
# ---------------------------------------------------------------------------

def field_rotation_explorer(static: bool = False):
    """The feeds turn against the sky, at a rate set by latitude and mount.

    This is the first place polarimetry stops being a property of the source and
    starts being a property of the *instrument*. A circular feed picks up a
    phase from this rotation; a linear feed has its Q and U mixed by twice the
    angle. Same sky, two entirely different corrections -- which is exactly the
    problem when both are in the same array.
    """
    all_sites = arr.load_sites()
    choices = ("ALMA", "SMA", "APEX", "LMT", "JCMT", "PV", "PDB", "GLT")
    controls = {
        "source": widgets.dropdown(_SOURCES, "M87", "source"),
        "s1": widgets.dropdown(choices, "ALMA", "station 1"),
        "s2": widgets.dropdown(choices, "SMA", "station 2"),
        "s3": widgets.dropdown(choices, "LMT", "station 3"),
        "s4": widgets.dropdown(choices, "PDB", "station 4"),
    }

    def build(source, s1, s2, s3, s4):
        ra, dec = arr.SOURCES[source]
        chosen = {code: all_sites[code] for code in dict.fromkeys([s1, s2, s3, s4])}
        return figs.field_rotation_figure(chosen, MJD_2017_APR11, np.linspace(0, 24, 289),
                                         ra, dec)

    return widgets.reactive(build, controls, static=static,
                        note="SMA is Nasmyth-mounted with a 45° offset, APEX is "
                             "Nasmyth the other way, JCMT is a plain alt-az, and NOEMA's "
                             "equatorial mount does not rotate the feeds at all. Curves "
                             "break where the source is below the horizon.")


def noise_explorer(static: bool = False):
    """Thermal noise on one baseline, and why a mixed baseline has four values.

    ``sigma = sqrt(SEFD_1 SEFD_2 / 2 B tau) / 0.88``: the geometric mean of the
    two stations. The SEFD belongs to a *feed*, not to a station, so on a
    baseline joining a circular-feed station to a linear-feed one, the four
    correlation products can each have their own noise level. Getting that
    pairing right in eht-imaging was part of this project.
    """
    all_sites = arr.load_sites()
    codes = tuple(all_sites)
    controls = {
        "s1": widgets.dropdown(codes, "ALMA", "station 1"),
        "s2": widgets.dropdown(codes, "SMA", "station 2"),
        "tint_s": widgets.slider(10.0, 1.0, 300.0, 1.0, "integration time (s)", ".0f"),
        "bandwidth_ghz": widgets.slider(2.0, 0.1, 8.0, 0.1, "bandwidth (GHz)"),
    }

    def build(s1, s2, tint_s, bandwidth_ghz):
        import plotly.graph_objects as go
        site1, site2 = all_sites[s1], all_sites[s2]
        sigma = arr.thermal_noise(site1.sefd_jy, site2.sefd_jy, tint_s,
                                 bandwidth_ghz * 1e9)
        labels = [f"{a.upper()}{b.upper()}" for a in site1.feeds for b in site2.feeds]
        fig = go.Figure(go.Bar(
            x=labels, y=[sigma] * 4, name="sigma", width=0.55,
            marker={"color": [style.FEED_LINEAR if site1.is_linear or site2.is_linear
                              else style.FEED_CIRCULAR] * 4},
            text=[f"{sigma * 1e3:.1f} mJy"] * 4, textposition="outside",
            textfont={"color": style.TEXT_SECONDARY},
            hovertemplate="%{x}: %{y:.4f} Jy<extra></extra>"))
        fig.update_yaxes(title_text="noise per correlation (Jy)",
                         range=[0, 1.45 * sigma])
        mixed = site1.is_linear != site2.is_linear
        fig.update_layout(height=380, width=620, bargap=0.5, showlegend=False)
        style.set_title(
            fig, f"The four products on the {s1}–{s2} baseline",
            f"{'MIXED basis' if mixed else 'same basis on both ends'}  ·  "
            f"{style.SIGMA} = {sigma * 1e3:.1f} mJy per product")
        return fig

    return widgets.reactive(build, controls, static=static,
                        note="Pick ALMA and any other station: the slot labels become "
                             "things like XR and YL, which no imaging code written for a "
                             "single polarization basis knows how to interpret. That is "
                             "the whole of notebook 03 in one bar chart.")


# ---------------------------------------------------------------------------
# Part D: from visibilities to images
# ---------------------------------------------------------------------------

def closure_explorer(static: bool = False):
    """Scramble the station phases and watch the closure phase refuse to move.

    Atmospheric phase at 230 GHz is hopeless -- it changes by radians in
    seconds, and it is *per station*. A closure phase adds three visibility
    phases around a triangle, and each station's corruption enters twice with
    opposite signs. What survives is a genuine constraint on the source.
    """
    controls = {
        "phase_rms_deg": widgets.slider(60.0, 0.0, 180.0, 5.0, "station phase noise (deg)",
                                    ".0f"),
        "model": widgets.dropdown(("ring", "double", "blob"), "ring", "sky model"),
        "seed": widgets.int_slider(0, 0, 20, 1, "random seed"),
    }
    triangle = ("ALMA", "LMT", "SMT")

    def build(phase_rms_deg, model, seed):
        ra, dec = arr.SOURCES["M87"]
        array = arr.load_sites()
        hours = np.linspace(0, 12, 90)
        mjd = MJD_2017_APR11 + hours / 24.0
        truth = _model_image(model, npix=64, fov_uas=200.0)

        rng = np.random.default_rng(seed)
        gains = {code: np.exp(1j * np.deg2rad(phase_rms_deg)
                              * rng.standard_normal(len(hours)))
                 for code in triangle}

        phases = {}
        product = np.ones(len(hours), dtype=complex)
        for code1, code2 in zip(triangle, triangle[1:] + triangle[:1], strict=True):
            u, v = arr.uv_coordinates(array[code1], array[code2], mjd, ra, dec)
            vis = itf.sample_visibilities(truth, u, v)
            corrupted = itf.apply_station_gains(vis, gains[code1], gains[code2])
            phases[f"{code1}–{code2}"] = np.angle(corrupted)
            product *= corrupted
        return figs.closure_phase_figure(triangle, hours, phases, np.angle(product))

    return widgets.reactive(build, controls, static=static,
                        note="Push the noise to 180° and the individual phases become "
                             "pure garbage, while the bottom panel does not move at all. "
                             "The EHT's first images were made almost entirely from "
                             "quantities with this property.")
