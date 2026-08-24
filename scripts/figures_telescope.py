"""Every figure notebook 02 draws: the array, the sky, and the Fourier plane.

Two halves. The first is the array as a physical object -- stations on a globe,
who can see the source when, how each station's feeds rotate against the sky.
The second is the Fourier side -- fringes, uv coverage, dirty beams, dirty images.

``u`` and ``v`` are plotted in gigawavelengths, the unit every EHT paper uses: the
array's longest baselines reach about 8 Glambda, and a 42 microarcsecond ring puts
a null near 3.4 Glambda.
"""

from __future__ import annotations

import arrays as arr
import interferometry as itf
import numpy as np
import plotly.graph_objects as go
import style
from plotly.subplots import make_subplots

__all__ = ["array_map", "array_globe_3d", "elevation_strip", "field_rotation_figure",
           "feed_legend_note", "resolution_figure", "fringe_figure",
           "fourier_component_figure", "visibility_profile_figure",
           "uv_coverage_figure", "sampling_figure", "closure_phase_figure"]

EARTH_RADIUS_M = 6371e3
GIGA = 1e9

#: How many station curves the field-rotation figure draws -- and always draws.
MAX_STATION_CURVES = 4

#: Angular sizes worth having on screen when talking about resolution.
M87_RING = "M87's ring, 42 µas"
LANDMARKS = {M87_RING: 42.0, "Hubble at 500 nm, 50,000 µas": 5e4}

#: Apertures worth having on screen, in metres.
APERTURES = {"a 12 m dish": 12.0, "the 300 m Arecibo dish": 300.0,
             "Earth's diameter": 1.274e7}


# ---------------------------------------------------------------------------
# The array as a physical object
# ---------------------------------------------------------------------------

def _feed_colour(site: arr.Site) -> str:
    """Blue for circular feeds, orange for linear. Fixed across the whole report."""
    return style.FEED_LINEAR if site.is_linear else style.FEED_CIRCULAR


def _marker_size(site: arr.Site) -> float:
    """Marker area tracks collecting area, so ALMA looks like what it is."""
    return float(8.0 + 14.0 * np.sqrt(site.collecting_area / 4000.0))


#: Stations that share a mountain land on the same pixel, so their labels are
#: pushed apart by hand -- ALMA and APEX are 2.6 km apart on Chajnantor, SMA
#: and JCMT 164 m apart on Mauna Kea. Left to plotly, the two labels overprint and
#: neither is readable, which is worse than either one missing. Offsets stay
#: vertical: a sideways label near the limb is clipped at the subplot's edge.
_LABEL_POSITIONS = {"ALMA": "bottom center", "APEX": "top center",
                    "SMA": "bottom center", "JCMT": "top center"}
_DEFAULT_LABEL_POSITION = "top center"


def feed_legend_note() -> str:
    """The one-line legend used wherever stations are coloured by feed basis."""
    return ("Blue stations record a <b>circular</b> (R/L) basis, "
            "orange ones a <b>linear</b> (X/Y) basis.")


def sub_source_point(mjd: float, ra_hours: float, dec_deg: float) -> tuple[float, float]:
    """Longitude and latitude where the source is directly overhead, in degrees."""
    lon = (ra_hours - arr.gmst_hours(mjd)) * 15.0
    return float((lon + 180.0) % 360.0 - 180.0), float(dec_deg)


def array_map(sites: dict[str, arr.Site], mjd: float, ra_hours: float, dec_deg: float,
              show_baselines: bool = True, elev_min_deg: float = 10.0,
              title: str | None = None) -> go.Figure:
    """The array on a globe, viewed from the source.

    Parameters
    ----------
    sites : dict
        Stations to draw, from :func:`arrays.load_sites`.
    mjd : float
        Time of the snapshot, including time of day.
    ra_hours, dec_deg : float
        Source position; the globe is rotated so the source is overhead at the
        centre of the disc.
    show_baselines : bool, optional
        Draw the live baselines (both ends above the elevation limit).
    elev_min_deg : float, optional
        Elevation limit below which a station cannot observe.
    title : str, optional
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    lon0, lat0 = sub_source_point(mjd, ra_hours, dec_deg)
    codes = list(sites)
    elev = {code: float(np.rad2deg(arr.elevation(site, mjd, ra_hours, dec_deg)))
            for code, site in sites.items()}
    up = {code: elev[code] > elev_min_deg for code in codes}

    fig = go.Figure()

    # Always present, empty when switched off, so the trace list keeps its shape.
    lons: list[float] = []
    lats: list[float] = []
    if show_baselines:
        for code1, code2 in arr.baselines(sites):
            if not (up[code1] and up[code2]):
                continue
            for code in (code1, code2):
                lons.append(np.rad2deg(sites[code].longitude))
                lats.append(np.rad2deg(sites[code].latitude))
            lons.append(None)
            lats.append(None)
    fig.add_trace(go.Scattergeo(lon=lons, lat=lats, name="baselines", mode="lines",
                                line={"width": 1, "color": style.TEXT_MUTED},
                                opacity=0.55, hoverinfo="skip", showlegend=False))

    # Both groups are always added, even when empty: an explorer updates this
    # figure in place, which requires the trace list to stay the same shape.
    # Biggest dish first within a group, so a small station is never buried
    # under a large one drawn on top of it.
    by_size = sorted(codes, key=lambda code: -_marker_size(sites[code]))
    for observing in (False, True):
        selected = [code for code in by_size if up[code] == observing]
        fig.add_trace(go.Scattergeo(
            lon=[np.rad2deg(sites[code].longitude) for code in selected],
            lat=[np.rad2deg(sites[code].latitude) for code in selected],
            text=selected, name="observing" if observing else "source below horizon",
            mode="markers+text", showlegend=False,
            textposition=[_LABEL_POSITIONS.get(code, _DEFAULT_LABEL_POSITION)
                          for code in selected] or _DEFAULT_LABEL_POSITION,
            textfont={"size": 10, "color": style.TEXT_SECONDARY if observing
                      else style.TEXT_MUTED},
            marker={"size": [_marker_size(sites[code]) for code in selected] or [1],
                    "color": [_feed_colour(sites[code]) for code in selected] or [
                        style.FEED_CIRCULAR],
                    "opacity": 1.0 if observing else 0.25,
                    "line": {"width": 1.5 if observing else 0.5, "color": style.SURFACE}},
            customdata=[[sites[code].name, sites[code].feeds.upper(),
                         sites[code].sefd_jy, elev[code]] for code in selected],
            hovertemplate="<b>%{text}</b> · %{customdata[0]}<br>"
                          "feeds %{customdata[1]} · SEFD %{customdata[2]:.0f} Jy<br>"
                          "elevation %{customdata[3]:.0f}°<extra></extra>"))

    # The station markers are coloured by feed basis and faded when the source is
    # below the horizon, so an automatic legend would label a colour with a state.
    # These two carry no data and exist only to say what the colours mean.
    for linear, label in ((False, "circular feeds"), (True, "linear feeds")):
        fig.add_trace(go.Scattergeo(
            lon=[None], lat=[None], mode="markers", name=label, hoverinfo="skip",
            marker={"size": 11, "line": {"width": 1.5, "color": style.SURFACE},
                    "color": style.FEED_LINEAR if linear else style.FEED_CIRCULAR}))

    fig.update_geos(projection={"type": "orthographic",
                                "rotation": {"lon": lon0, "lat": lat0}},
                    showland=True, landcolor="#efeee9", showocean=True,
                    oceancolor="#dfe4ea", showcountries=True,
                    countrycolor="#cfcec8", coastlinecolor="#b9b8b2",
                    showframe=True, framecolor=style.AXIS, bgcolor=style.SURFACE)
    fig.update_layout(height=500, margin={"l": 0, "r": 0, "b": 0})
    style.legend_below(fig, y=0.02)
    style.set_title(fig, "The array, seen from the source", title)
    return fig


def array_globe_3d(sites: dict[str, arr.Site], mjd: float, ra_hours: float,
                   dec_deg: float, elev_min_deg: float = 10.0) -> go.Figure:
    """The same array in 3D, with baselines drawn as chords through the Earth.

    A featureless sphere with no coastlines on it, which is why notebook 02 shows
    :func:`array_map` instead: the geography is what makes the picture readable.
    Kept, and kept tested, because it is the one view of the array that needs
    nothing from the network -- plot it directly if plotly's coastline data
    cannot be fetched.
    """
    fig = go.Figure()

    phi, cos_theta = np.mgrid[0:2 * np.pi:80j, -1:1:40j]
    sin_theta = np.sqrt(1 - cos_theta**2)
    radius = EARTH_RADIUS_M / 1e6
    fig.add_trace(go.Surface(x=radius * sin_theta * np.cos(phi),
                             y=radius * sin_theta * np.sin(phi), z=radius * cos_theta,
                             name="earth", opacity=0.22, showscale=False,
                             hoverinfo="skip",
                             colorscale=[[0, "#b9c4cf"], [1, "#b9c4cf"]]))

    codes = list(sites)
    up = {code: float(np.rad2deg(arr.elevation(site, mjd, ra_hours, dec_deg))) > elev_min_deg
          for code, site in sites.items()}
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for code1, code2 in arr.baselines(sites):
        if not (up[code1] and up[code2]):
            continue
        for code in (code1, code2):
            position = sites[code].xyz / 1e6
            xs.append(position[0])
            ys.append(position[1])
            zs.append(position[2])
        xs.append(None)
        ys.append(None)
        zs.append(None)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, name="baselines", mode="lines",
                               line={"width": 2, "color": style.TEXT_MUTED},
                               opacity=0.6, hoverinfo="skip"))

    for linear in (False, True):
        selected = [code for code in codes if sites[code].is_linear == linear]
        positions = (np.array([sites[code].xyz / 1e6 for code in selected])
                     if selected else np.zeros((0, 3)))
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0], y=positions[:, 1], z=positions[:, 2], text=selected,
            name="linear feeds" if linear else "circular feeds",
            mode="markers+text",
            textposition=[_LABEL_POSITIONS.get(code, _DEFAULT_LABEL_POSITION)
                          for code in selected] or _DEFAULT_LABEL_POSITION,
            textfont={"size": 10, "color": style.TEXT_SECONDARY},
            marker={"size": [0.45 * _marker_size(sites[code]) for code in selected] or [1],
                    "color": style.FEED_LINEAR if linear else style.FEED_CIRCULAR},
            hovertemplate="<b>%{text}</b><extra></extra>"))

    axis = {"showbackground": False, "showticklabels": False, "title": "",
            "showgrid": False, "zeroline": False, "range": [-7.5, 7.5]}
    fig.update_layout(height=500, margin={"l": 0, "r": 0, "b": 0},
                      scene={"xaxis": axis, "yaxis": axis, "zaxis": axis,
                             "aspectmode": "cube",
                             "camera": {"eye": {"x": 1.15, "y": 1.15, "z": 0.8}}})
    style.legend_below(fig, y=0.04)
    style.set_title(fig, "A baseline is a chord through the planet",
                    "the two dishes act as two points on one Earth-sized aperture")
    return fig


def elevation_strip(sites: dict[str, arr.Site], mjd_start: float, hours: np.ndarray,
                    ra_hours: float, dec_deg: float,
                    elev_min_deg: float = 10.0) -> go.Figure:
    """Elevation of the source at every station, over a night.

    One row per station, time across. Blank means below the horizon, so the
    figure reads as "who is on the air when" -- and the answer, for a source at
    M87's declination, is never everybody at once.
    """
    codes = list(sites)
    mjd = mjd_start + np.asarray(hours) / 24.0
    grid = np.array([np.rad2deg(arr.elevation(sites[code], mjd, ra_hours, dec_deg))
                     for code in codes])
    grid = np.where(grid > elev_min_deg, grid, np.nan)

    labels = [f"{code}  {'XY' if sites[code].is_linear else 'RL'}" for code in codes]
    fig = go.Figure(go.Heatmap(
        z=grid, x=hours, y=labels, name="elevation",
        colorscale=style.mpl_colorscale("viridis"), zmin=0, zmax=90,
        colorbar={"title": {"text": "elevation", "side": "right"}, "thickness": 12,
                  "outlinewidth": 0, "ticksuffix": "°"},
        hovertemplate="%{y}<br>%{x:.1f} h UT<br>%{z:.0f}°<extra></extra>"))
    fig.update_xaxes(title_text="hours UT", showgrid=False)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_layout(height=96 + 34 * len(codes), showlegend=False)
    style.set_title(fig, "When each station can see the source",
                    "blank means below the horizon, and the array is never all on the air")
    return fig


def field_rotation_figure(sites: dict[str, arr.Site], mjd_start: float,
                          hours: np.ndarray, ra_hours: float, dec_deg: float,
                          elev_min_deg: float = 10.0) -> go.Figure:
    """Feed rotation angle against time, one curve per station.

    Every station's feeds turn against the sky at its own rate, set by its
    latitude and by its mount: an alt-az mount follows the parallactic angle, a
    Nasmyth optical path adds or subtracts the elevation, and an equatorial
    mount does not rotate at all. Since the correction is per station, it cannot
    be calibrated away with baseline-based tricks -- it has to be modelled.
    """
    codes = list(sites)[:MAX_STATION_CURVES]  # four is readable; more is a rainbow
    mjd = mjd_start + np.asarray(hours) / 24.0

    fig = go.Figure()
    # Always the same number of traces, even when fewer stations are given: the
    # explorers update this figure in place, and a changing trace count breaks
    # that. Picking the same station in two dropdowns used to do exactly that.
    for index in range(MAX_STATION_CURVES):
        if index >= len(codes):
            fig.add_trace(go.Scatter(x=[], y=[], name=f"unused{index}", mode="lines",
                                     line={"width": 2, "color": style.SERIES[index]},
                                     hoverinfo="skip", showlegend=False))
            continue
        code = codes[index]
        site = sites[code]
        angle = np.rad2deg(arr.field_rotation_angle(site, mjd, ra_hours, dec_deg))
        visible = np.rad2deg(arr.elevation(site, mjd, ra_hours, dec_deg)) > elev_min_deg
        angle = np.where(visible, np.unwrap(angle, period=360.0), np.nan)
        mount = ("equatorial" if (site.fr_par, site.fr_elev) == (0.0, 0.0)
                 else "alt-az" if site.fr_elev == 0.0
                 else "Nasmyth")
        fig.add_trace(go.Scatter(
            x=hours, y=angle, name=f"{code} ({mount})", mode="lines",
            line={"width": 2, "color": style.SERIES[index]},
            hovertemplate=f"{code}<br>%{{x:.1f}} h UT<br>φ %{{y:.0f}}°"
                          "<extra></extra>"))

    fig.update_xaxes(title_text="hours UT")
    fig.update_yaxes(title_text=f"feed rotation {style.PHI} (deg)")
    fig.update_layout(height=430, width=820)
    style.legend_below(fig, y=-0.16)
    style.set_title(fig, "The feeds turn against the sky, differently at every station",
                    "curves break where the source is below the horizon")
    return fig


# ---------------------------------------------------------------------------
# The Fourier plane
# ---------------------------------------------------------------------------

def resolution_figure(wavelength_mm: float = 1.3, aperture_km: float = 1e4) -> go.Figure:
    """The finest detail a telescope can see, against how big it is.

    One curve, ``theta = lambda / D``, with the sizes worth caring about drawn
    across it. Read it by picking an aperture on the horizontal axis and reading
    the smallest thing it can resolve off the vertical one -- and note where the
    curve crosses M87's 42 microarcsecond ring, because that crossing is the
    aperture the EHT had to have.

    Everything is drawn as a trace rather than with ``add_hline``: shape
    coordinates on a log axis are interpreted as powers of ten, which silently
    put the landmark lines at 10^42 instead of 42.
    """
    aperture_m = np.logspace(0, 7.4, 300)
    resolution = np.array([itf.resolution_uas(wavelength_mm * 1e-3, d)
                           for d in aperture_m])
    aperture_km_axis = aperture_m / 1e3
    chosen = itf.resolution_uas(wavelength_mm * 1e-3, aperture_km * 1e3)
    needed_km = itf.required_aperture_m(wavelength_mm * 1e-3, LANDMARKS[M87_RING]) / 1e3

    fig = go.Figure()

    # The angular sizes worth resolving, as full-width dotted lines.
    for label, size in LANDMARKS.items():
        fig.add_trace(go.Scatter(
            x=[aperture_km_axis[0], aperture_km_axis[-1]], y=[size, size],
            name=label, mode="lines",
            line={"color": style.TEXT_MUTED, "width": 1, "dash": "dot"},
            hovertemplate=f"{label}<extra></extra>", showlegend=False))
        fig.add_annotation(x=np.log10(0.0025), y=np.log10(size), text=label,
                           showarrow=False, xanchor="left", yanchor="bottom",
                           font={"size": 11, "color": style.TEXT_SECONDARY})

    # The aperture M87's ring demands, which is the point of the whole figure.
    fig.add_trace(go.Scatter(
        x=[needed_km, needed_km], y=[resolution.min(), resolution.max()],
        name="needed", mode="lines", showlegend=False,
        line={"color": style.SERIES[2], "width": 1.5, "dash": "dash"},
        hovertemplate="%{x:,.0f} km<extra></extra>"))
    fig.add_annotation(x=np.log10(needed_km), y=np.log10(resolution.max()),
                       text=f"{needed_km:,.0f} km needed<br>for M87's ring ",
                       showarrow=False, xanchor="right", yanchor="top",
                       font={"size": 11, "color": style.SERIES[2]})

    # The curve itself, labelled on the curve so it needs no legend.
    fig.add_trace(go.Scatter(
        x=aperture_km_axis, y=resolution, name="resolution", mode="lines",
        line={"color": style.SERIES[0], "width": 2.5},
        hovertemplate="%{x:,.3g} km across &#8594; sees %{y:,.3g} "
                      + style.UAS + "<extra></extra>"))
    label_at = int(0.72 * len(aperture_km_axis))
    fig.add_annotation(x=np.log10(aperture_km_axis[label_at]),
                       y=np.log10(resolution[label_at]),
                       text=f"  &#955; / D at {wavelength_mm:.2f} mm", showarrow=False,
                       xanchor="left", yanchor="bottom",
                       font={"size": 13, "color": style.SERIES[0]})

    # Real telescopes, for scale.
    fig.add_trace(go.Scatter(
        x=[d / 1e3 for d in APERTURES.values()],
        y=[itf.resolution_uas(wavelength_mm * 1e-3, d) for d in APERTURES.values()],
        text=[f" {label} " for label in APERTURES], name="apertures",
        mode="markers+text",
        textposition="middle right",
        textfont={"size": 10, "color": style.TEXT_SECONDARY},
        marker={"size": 7, "color": style.TEXT_SECONDARY}, hoverinfo="skip",
        showlegend=False))
    fig.add_trace(go.Scatter(
        x=[aperture_km], y=[chosen], name="chosen", mode="markers",
        marker={"size": 14, "color": style.HIGHLIGHT,
                "line": {"color": style.SURFACE, "width": 2}},
        hovertemplate="your aperture: %{x:,.0f} km<br>sees %{y:,.3g} "
                      + style.UAS + "<extra></extra>", showlegend=False))

    fig.update_xaxes(title_text="aperture, or baseline length (km)", type="log",
                     range=[np.log10(6e-4), np.log10(9e4)])
    fig.update_yaxes(title_text=f"finest detail it can resolve ({style.UAS})",
                     type="log",
                     range=[np.log10(resolution.min()) - 0.4,
                            np.log10(resolution.max()) + 0.4])
    fig.update_layout(height=440, width=800, showlegend=False)
    style.set_title(
        fig, "How big a telescope do you need?",
        f"lower is better &#183; {aperture_km:,.0f} km at {wavelength_mm:.2f} mm "
        f"resolves {chosen:,.1f} {style.UAS}")
    return fig


def _image_axis(image: dict) -> np.ndarray:
    return np.linspace(image["extent"][0], image["extent"][1], image["I"].shape[0])


def _sky_heatmap(image: dict, name: str, colorscale=None, **kwargs) -> go.Heatmap:
    axis = _image_axis(image)
    return go.Heatmap(z=image["I"], x=axis, y=axis, name=name,
                      colorscale=colorscale or style.mpl_colorscale(style.CMAP_INTENSITY),
                      hovertemplate="%{x:.0f}, %{y:.0f} µas<br>%{z:.3f}"
                                    "<extra></extra>", **kwargs)


def fringe_figure(baseline_km: float, wavelength_m: float = 1.3e-3,
                  span_uas: float = 150.0) -> go.Figure:
    """The fringe pattern of a two-element interferometer, across the figs_tel.

    A pair of dishes is not a camera: it is a comb of fringes laid across the
    sky, and the correlator reports how much of the source lines up with the
    comb. Lengthen the baseline and the comb gets finer -- which is why long
    baselines are sensitive to small structure and blind to large structure.
    """
    offsets = np.linspace(-span_uas, span_uas, 1200)
    response = itf.fringe_response(offsets, baseline_km * 1e3, wavelength_m)
    spacing = wavelength_m / (baseline_km * 1e3) / itf.UAS

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=offsets, y=response, name="fringe", mode="lines",
                             line={"color": style.SERIES[0], "width": 2},
                             hovertemplate="%{x:.0f} µas<br>%{y:+.2f}"
                                           "<extra></extra>"))
    fig.add_trace(go.Scatter(x=[-21, 21], y=[1.08, 1.08], name="M87 ring",
                             mode="lines+text", text=["", "  M87 ring (42 µas)"],
                             textposition="middle right",
                             textfont={"size": 11, "color": style.HIGHLIGHT},
                             line={"color": style.HIGHLIGHT, "width": 3},
                             hoverinfo="skip", showlegend=False))
    fig.update_xaxes(title_text=f"angle on the sky ({style.UAS})")
    fig.update_yaxes(title_text="correlator response", range=[-1.25, 1.35])
    fig.update_layout(height=380, width=780, showlegend=False)
    style.set_title(fig, "One pair of dishes is a comb of fringes",
                    f"{baseline_km:,.0f} km baseline gives a fringe spacing of "
                    f"{spacing:.0f} {style.UAS}")
    return fig


def fourier_component_figure(image: dict, u: float, v: float) -> go.Figure:
    """One baseline, one Fourier component: the sky, the fringe it matches, and V.

    Left is the source with the baseline's fringe pattern laid over it; right is
    where that sample lands in the uv plane, with the measured amplitude and
    phase. Rotating the baseline rotates the fringes; lengthening it makes them
    finer. Every complex number the EHT records is one of these.
    """
    visibility = itf.sample_visibilities(image, np.array([u]), np.array([v]))[0]
    axis = _image_axis(image)
    l_grid, m_grid = itf.image_coordinates(image["I"].shape[0], image["fov_uas"])
    fringe = np.cos(2 * np.pi * (u * l_grid + v * m_grid))

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45],
                        horizontal_spacing=0.13,
                        subplot_titles=("the source, and this baseline's fringes",
                                        "where the sample lands"))
    fig.add_trace(_sky_heatmap(image, "sky", showscale=False), row=1, col=1)
    # Only the crests of the comb, so the pattern reads as a set of stripes
    # instead of a wash of thin contours.
    fig.add_trace(go.Contour(z=fringe, x=axis, y=axis, name="fringes",
                             contours={"start": 0.995, "end": 1.0, "size": 1.0,
                                       "coloring": "lines"},
                             line={"width": 1.6}, opacity=0.85, showscale=False,
                             colorscale=[[0, "#8fd0ff"], [1, "#8fd0ff"]],
                             hoverinfo="skip"), row=1, col=1)

    limit = 9.0
    ring = np.linspace(0, 2 * np.pi, 121)
    for radius in (3.0, 6.0, 9.0):
        fig.add_trace(go.Scatter(x=radius * np.cos(ring), y=radius * np.sin(ring),
                                 name=f"ring{radius}", mode="lines", showlegend=False,
                                 line={"color": style.GRID, "width": 1},
                                 hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, u / GIGA], y=[0, v / GIGA], name="baseline",
                             mode="lines", line={"color": style.HIGHLIGHT, "width": 2},
                             hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[u / GIGA, -u / GIGA], y=[v / GIGA, -v / GIGA],
                             name="sample", mode="markers+text",
                             text=["  this baseline", "  its free conjugate"],
                             textposition="middle right",
                             textfont={"size": 10, "color": style.TEXT_SECONDARY},
                             marker={"size": [11, 7], "color": style.HIGHLIGHT,
                                     "opacity": [1.0, 0.45],
                                     "line": {"color": style.SURFACE, "width": 1.5}},
                             hovertemplate="u %{x:.2f}, v %{y:.2f} Gλ<extra></extra>"),
                  row=1, col=2)

    fig.update_xaxes(title_text="µas", range=[image["extent"][1], image["extent"][0]],
                     showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[image["extent"][2], image["extent"][3]], showgrid=False,
                     zeroline=False, scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="u (Gλ)", range=[limit, -limit], row=1, col=2)
    fig.update_yaxes(title_text="v (Gλ)", range=[-limit, limit],
                     scaleanchor="x2", scaleratio=1, row=1, col=2)
    fig.update_layout(height=430, showlegend=False)
    style.set_title(
        fig, "One baseline measures one Fourier component",
        f"baseline {np.hypot(u, v) / GIGA:.2f} G{style.LAMBDA}  ·  |V| = "
        f"{abs(visibility):.3f} Jy  ·  phase "
        f"{np.rad2deg(np.angle(visibility)):+.0f}{style.DEG}", has_subplots=True)
    return fig


def visibility_profile_figure(image: dict, u_max_glambda: float = 9.0,
                              coverage: dict | None = None) -> go.Figure:
    """Visibility amplitude against baseline length, with the ring's nulls.

    A ring of diameter ``d`` has a visibility profile that oscillates and dips
    to a deep minimum near ``u ~ 1.22 / d`` -- for M87's 42 uas, about 3.4
    Glambda. The EHT's ability to *see* that null is what pinned the ring
    diameter, and it is why the array needed baselines of exactly that length.
    """
    radius = np.linspace(0.02, u_max_glambda, 400) * GIGA
    profiles = {
        "east-west cut": itf.sample_visibilities(image, radius, np.zeros_like(radius)),
        "north-south cut": itf.sample_visibilities(image, np.zeros_like(radius), radius),
    }

    fig = go.Figure()
    for index, (label, values) in enumerate(profiles.items()):
        fig.add_trace(go.Scatter(x=radius / GIGA, y=np.abs(values), name=label,
                                 mode="lines",
                                 line={"width": 2, "color": style.SERIES[index]},
                                 hovertemplate="%{x:.2f} Gλ<br>|V| %{y:.3f} Jy"
                                               "<extra></extra>"))
    if coverage is not None:
        sampled = np.hypot(coverage["u"], coverage["v"]) / GIGA
        fig.add_trace(go.Scatter(
            x=sampled, y=np.full_like(sampled, 1e-4), name="where the EHT samples",
            mode="markers", marker={"size": 5, "color": style.TEXT_MUTED, "symbol": "line-ns-open"},
            hovertemplate="sampled at %{x:.2f} Gλ<extra></extra>"))

    fig.update_xaxes(title_text="baseline length (Gλ)")
    fig.update_yaxes(title_text="visibility amplitude (Jy)", type="log",
                     range=[-4, 0.2])
    fig.update_layout(height=420, width=780)
    style.legend_below(fig)
    style.set_title(fig, "Long baselines see small structure",
                    "the first deep minimum is a direct reading of the ring diameter")
    return fig


def uv_coverage_figure(coverage: dict, title: str | None = None,
                       highlight: tuple[str, str] | None = None) -> go.Figure:
    """Everything the array measured, in the plane where it measured it.

    Each point is one baseline at one moment; the conjugate half comes free
    because the sky is real. Earth rotation drags each baseline along an arc, so
    an eight-station array with 28 baselines still fills only a few percent of
    the plane -- and the empty parts are exactly the structures the data cannot
    constrain.
    """
    u = np.concatenate([coverage["u"], -coverage["u"]]) / GIGA
    v = np.concatenate([coverage["v"], -coverage["v"]]) / GIGA
    hours = np.concatenate([coverage["hours"], coverage["hours"]])
    labels = np.concatenate([[f"{a}-{b}" for a, b in
                              zip(coverage["site1"], coverage["site2"], strict=True)]] * 2)

    fig = go.Figure()
    for radius in (2.0, 4.0, 6.0, 8.0):
        ring = np.linspace(0, 2 * np.pi, 181)
        fig.add_trace(go.Scatter(x=radius * np.cos(ring), y=radius * np.sin(ring),
                                 name=f"ring{radius}", mode="lines", showlegend=False,
                                 line={"color": style.GRID, "width": 1},
                                 hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=u, y=v, name="samples", mode="markers", text=labels,
        marker={"size": 4, "color": hours, "colorscale": style.mpl_colorscale("viridis"),
                "colorbar": {"title": {"text": "hours UT", "side": "right"},
                             "thickness": 12, "outlinewidth": 0, "len": 0.85},
                "showscale": True},
        hovertemplate="%{text}<br>u %{x:.2f}, v %{y:.2f} Gλ<extra></extra>"))

    if highlight is not None:
        track = coverage["per_baseline"][highlight]
        mask = track["visible"]
        fig.add_trace(go.Scatter(
            x=np.concatenate([track["u"][mask], -track["u"][mask]]) / GIGA,
            y=np.concatenate([track["v"][mask], -track["v"][mask]]) / GIGA,
            name=f"{highlight[0]}-{highlight[1]}", mode="markers",
            marker={"size": 7, "color": style.HIGHLIGHT,
                    "line": {"color": style.SURFACE, "width": 1}},
            hovertemplate=f"{highlight[0]}-{highlight[1]}<extra></extra>"))

    limit = 9.5
    fig.update_xaxes(title_text="u (Gλ)", range=[limit, -limit], zeroline=False)
    fig.update_yaxes(title_text="v (Gλ)", range=[-limit, limit], zeroline=False,
                     scaleanchor="x", scaleratio=1)
    fig.update_layout(height=530, width=600)
    if highlight is not None:
        style.legend_below(fig, y=-0.13)
    else:
        fig.update_layout(showlegend=False)
    style.set_title(fig, "What the array measured, where it measured it",
                    title or f"{len(coverage['u'])} visibility samples")
    return fig


def sampling_figure(truth: dict, dirty: dict, beam: dict) -> go.Figure:
    """Truth, dirty beam, dirty image -- the three panels that state the problem.

    The dirty image is the truth convolved with the beam. It is *the data drawn
    in the image plane*, not a picture of the sky, and everything called
    "imaging" is the job of undoing that convolution with far too little
    information.
    """
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06,
                        subplot_titles=("the true sky", "the dirty beam (PSF)",
                                        "what you get: truth ⊛ beam"))
    for col, (image, name) in enumerate([(truth, "truth"), (beam, "beam"),
                                         (dirty, "dirty")], start=1):
        colorscale = (style.mpl_colorscale(style.CMAP_DIVERGING) if name == "beam"
                      else style.mpl_colorscale(style.CMAP_INTENSITY))
        trace = _sky_heatmap(image, name, colorscale=colorscale, showscale=False)
        if name == "beam":
            trace.update(zmid=0.0)
        fig.add_trace(trace, row=1, col=col)
        anchor = f"x{col if col > 1 else ''}"
        fig.update_xaxes(range=[image["extent"][1], image["extent"][0]], showgrid=False,
                         zeroline=False, showticklabels=False, row=1, col=col)
        fig.update_yaxes(range=[image["extent"][2], image["extent"][3]], showgrid=False,
                         zeroline=False, showticklabels=False, scaleanchor=anchor,
                         scaleratio=1, row=1, col=col)
    fig.update_layout(height=320, margin={"b": 10}, showlegend=False)
    style.set_title(fig, "The dirty image is the data, not the sky", has_subplots=True)
    return fig


def closure_phase_figure(triangle: tuple[str, str, str], hours: np.ndarray,
                         phases: dict[str, np.ndarray],
                         closure: np.ndarray) -> go.Figure:
    """Three visibility phases wandering under station gains, and their fixed sum.

    Individual phases are meaningless without calibration -- they are dominated
    by the atmosphere over each station. Their sum around a closed triangle is
    not: every station appears once positively and once negatively, so the
    corruption cancels exactly.
    """
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        row_heights=[0.55, 0.45],
                        subplot_titles=("individual visibility phases (corrupted)",
                                        "closure phase (immune)"))
    for index, (label, values) in enumerate(phases.items()):
        fig.add_trace(go.Scatter(x=hours, y=np.rad2deg(values), name=label, mode="lines",
                                 line={"width": 2, "color": style.SERIES[index]},
                                 hovertemplate=f"{label} %{{y:.0f}}°<extra></extra>"),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=hours, y=np.rad2deg(closure),
                             name=f"closure {'-'.join(triangle)}", mode="lines",
                             line={"width": 2.5, "color": style.TEXT_PRIMARY},
                             hovertemplate="closure %{y:.1f}°<extra></extra>"),
                  row=2, col=1)
    fig.update_yaxes(title_text="phase (deg)", range=[-190, 190], row=1, col=1)
    fig.update_yaxes(title_text="closure (deg)", range=[-190, 190], row=2, col=1)
    fig.update_xaxes(title_text="hours UT", row=2, col=1)
    fig.update_layout(height=520, width=820)
    style.legend_below(fig, y=-0.14)
    style.set_title(fig, "Closure phase survives what calibration cannot fix",
                    f"triangle {'-'.join(triangle)}", has_subplots=True)
    return fig
