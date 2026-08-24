"""Synthetic polarized sky images, and the tick pattern used to draw them.

Small analytic models -- a Gaussian and an M87-like ring -- give us a known
truth to point the rest of the report at. Images are dictionaries of four
``(npix, npix)`` Stokes arrays plus their field of view, which keeps them
trivially interchangeable with real ehtim images later.

Image convention: axis 0 is +y (up, north), axis 1 is +x. Arrays are returned
in the orientation matplotlib/plotly draw with ``origin='lower'``, and by radio
convention x increases to the *left* (east) on the sky -- a detail that only
matters once we tie the ellipse frame to the sky in notebook 03, so the helpers
here stay in plain image coordinates.
"""

from __future__ import annotations

import numpy as np
import polarization as pol


def _grid(npix: int, fov_uas: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centred coordinate grids in microarcseconds."""
    half = 0.5 * fov_uas
    axis = np.linspace(-half + 0.5 * fov_uas / npix, half - 0.5 * fov_uas / npix, npix)
    return np.meshgrid(axis, axis)  # x, y  (y is axis 0 after meshgrid default)


def gaussian_blob(npix: int = 128, fov_uas: float = 100.0, fwhm_uas: float = 25.0,
                  total_flux: float = 1.0, p_lin: float = 0.0,
                  evpa_deg: float = 0.0, p_circ: float = 0.0) -> dict:
    """A single Gaussian with uniform polarization.

    Useful precisely because its Fourier transform is analytic, so notebook 02
    can check the discrete transform against a closed form.

    Parameters
    ----------
    npix : int, optional
        Pixels per side.
    fov_uas : float, optional
        Field of view in microarcseconds.
    fwhm_uas : float, optional
        Full width at half maximum.
    total_flux : float, optional
        Integrated Stokes I flux.
    p_lin, evpa_deg, p_circ : float, optional
        Uniform polarization state applied to every pixel.

    Returns
    -------
    dict
        Keys ``I``, ``Q``, ``U``, ``V`` (arrays), ``fov_uas``, ``extent``.
    """
    x, y = _grid(npix, fov_uas)
    sigma = fwhm_uas / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    image_i = np.exp(-0.5 * (x * x + y * y) / sigma**2)
    image_i *= total_flux / image_i.sum()
    return _apply_uniform_pol(image_i, fov_uas, p_lin, np.deg2rad(evpa_deg), p_circ)


def polarized_ring(npix: int = 128, fov_uas: float = 100.0,
                   diameter_uas: float = 42.0, width_uas: float = 12.0,
                   total_flux: float = 1.0, p_lin: float = 0.25,
                   pitch_deg: float = 45.0, asymmetry: float = 0.4,
                   asymmetry_pa_deg: float = 200.0, p_circ: float = 0.0) -> dict:
    """An M87-like polarized ring: bright asymmetric annulus, spiral EVPA pattern.

    The EVPA runs at a fixed ``pitch`` angle to the local radial direction,
    which is what a toroidal-plus-poloidal magnetic field looks like projected
    on the sky -- the pattern the EHT actually measured for M87 in 2021.

    Parameters
    ----------
    npix : int, optional
        Pixels per side.
    fov_uas : float, optional
        Field of view in microarcseconds. The default 100 uas frames the 42 uas
        M87 ring the way the EHT papers do.
    diameter_uas, width_uas : float, optional
        Ring diameter and Gaussian cross-sectional FWHM.
    total_flux : float, optional
        Integrated Stokes I flux.
    p_lin : float, optional
        Linear polarization fraction (uniform across the ring).
    pitch_deg : float, optional
        Angle between the EVPA and the local radial direction. ``0`` gives a
        radial pattern, ``90`` a purely azimuthal one, ``45`` a spiral.
    asymmetry : float, optional
        Peak-to-mean brightness contrast around the ring, in ``[0, 1)``.
    asymmetry_pa_deg : float, optional
        Position angle of the bright side, measured counter-clockwise from +x.
    p_circ : float, optional
        Uniform circular polarization fraction.

    Returns
    -------
    dict
        Keys ``I``, ``Q``, ``U``, ``V`` (arrays), ``fov_uas``, ``extent``.
    """
    x, y = _grid(npix, fov_uas)
    radius = np.hypot(x, y)
    azimuth = np.arctan2(y, x)

    sigma = width_uas / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    image_i = np.exp(-0.5 * ((radius - 0.5 * diameter_uas) / sigma) ** 2)
    image_i *= 1.0 + asymmetry * np.cos(azimuth - np.deg2rad(asymmetry_pa_deg))
    image_i *= total_flux / image_i.sum()

    chi = azimuth + np.deg2rad(pitch_deg)
    return {
        "I": image_i,
        "Q": image_i * p_lin * np.cos(2.0 * chi),
        "U": image_i * p_lin * np.sin(2.0 * chi),
        "V": image_i * p_circ,
        "fov_uas": fov_uas,
        "extent": (-0.5 * fov_uas, 0.5 * fov_uas, -0.5 * fov_uas, 0.5 * fov_uas),
    }


def _apply_uniform_pol(image_i: np.ndarray, fov_uas: float, p_lin: float,
                       evpa_rad: float, p_circ: float) -> dict:
    """Give every pixel of ``image_i`` the same polarization state."""
    return {
        "I": image_i,
        "Q": image_i * p_lin * np.cos(2.0 * evpa_rad),
        "U": image_i * p_lin * np.sin(2.0 * evpa_rad),
        "V": image_i * p_circ,
        "fov_uas": fov_uas,
        "extent": (-0.5 * fov_uas, 0.5 * fov_uas, -0.5 * fov_uas, 0.5 * fov_uas),
    }


def evpa_ticks(image: dict, step: int = 6, scale: float = 1.0,
               i_cut: float = 0.1, length_mode: str = "polarized") -> dict:
    """Line segments showing the polarization of an image, ready to plot.

    Each tick sits on a pixel, points along the EVPA, and is *not* an arrow --
    an EVPA of 0 and of 180 degrees are the same state.

    Parameters
    ----------
    image : dict
        Image dictionary from :func:`polarized_ring` or :func:`gaussian_blob`.
    step : int, optional
        Draw a tick every ``step`` pixels in each direction.
    scale : float, optional
        Overall tick-length multiplier.
    i_cut : float, optional
        Skip pixels fainter than ``i_cut`` times the peak of Stokes I, where the
        EVPA is noise-dominated and drawing it would be dishonest.
    length_mode : {'polarized', 'fraction', 'uniform'}, optional
        What the tick length encodes: polarized *intensity*
        ``sqrt(Q^2+U^2)`` (the usual EHT choice), the polarization *fraction*,
        or nothing at all. The first two are scaled against fixed references --
        the peak of Stokes I, and unity -- rather than against their own maxima,
        so that comparing two images (or dragging a polarization slider)
        actually changes the tick lengths.

    Returns
    -------
    dict
        ``x0, y0, x1, y1`` segment endpoints in microarcseconds, plus ``frac``
        (polarization fraction, for colouring) and ``pol_int``.
    """
    npix = image["I"].shape[0]
    x, y = _grid(npix, image["fov_uas"])
    sl = (slice(step // 2, None, step), slice(step // 2, None, step))

    image_i = image["I"][sl]
    q, u = image["Q"][sl], image["U"][sl]
    keep = image_i > i_cut * image["I"].max()

    image_i, q, u = image_i[keep], q[keep], u[keep]
    xc, yc = x[sl][keep], y[sl][keep]

    chi = pol.evpa(q, u)
    pol_int = np.hypot(q, u)
    frac = np.divide(pol_int, image_i, out=np.zeros_like(pol_int), where=image_i > 0)

    peak = image["I"].max()
    if length_mode == "polarized":
        weight = pol_int / peak if peak > 0 else np.zeros_like(pol_int)
    elif length_mode == "fraction":
        weight = frac
    elif length_mode == "uniform":
        weight = np.ones_like(pol_int)
    else:
        raise ValueError(f"unknown length_mode {length_mode!r}")

    half = 0.5 * scale * weight * step * image["fov_uas"] / npix
    dx, dy = half * np.cos(chi), half * np.sin(chi)
    return {"x0": xc - dx, "y0": yc - dy, "x1": xc + dx, "y1": yc + dy,
            "frac": frac, "pol_int": pol_int}


def image_stokes_totals(image: dict) -> dict[str, float]:
    """Integrated Stokes parameters and net polarization fractions of an image.

    The *net* fractions are what a single-dish measurement sees: a ring with 25%
    polarization at every pixel can have a tiny net fraction, because opposite
    sides of the ring cancel. Worth showing early -- it is why resolved
    polarimetry is worth the trouble.
    """
    totals = {key: float(np.sum(image[key])) for key in ("I", "Q", "U", "V")}
    totals["p_lin_net"] = pol.frac_lin(totals["I"], totals["Q"], totals["U"])
    totals["p_lin_mean"] = float(
        np.average(np.hypot(image["Q"], image["U"]) / np.maximum(image["I"], 1e-300),
                   weights=image["I"])
    )
    totals["evpa_net_deg"] = float(np.rad2deg(pol.evpa(totals["Q"], totals["U"])))
    return totals
