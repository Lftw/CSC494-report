"""Interferometry from first principles: fringes, visibilities, dirty images.

Deliberately written as direct sums rather than FFTs. They are slower, but each
one is a transcription of the equation next to it in the notebook, and at the
sizes used here (a hundred-odd pixels, a few thousand samples) the difference is
a second. Speed is what eht-imaging's three transform backends -- ``direct``,
``fast`` and ``nfft`` -- exist for.

Units, fixed throughout:

* sky coordinates ``l, m`` in radians (helpers below convert from microarcsec);
* ``u, v`` in wavelengths;
* visibilities in Jy, images in Jy per pixel.
"""

from __future__ import annotations

import numpy as np

UAS = np.pi / (180.0 * 3600.0 * 1e6)   # one microarcsecond in radians
C_LIGHT = 299792458.0


# ---------------------------------------------------------------------------
# Resolution: why the array has to be the size of the planet
# ---------------------------------------------------------------------------

def resolution_uas(wavelength_m: float, aperture_m: float) -> float:
    """Diffraction-limited resolution ``lambda / D``, in microarcseconds.

    The number that forces the whole design: at the EHT's 1.3 mm, resolving
    M87's 42 uas shadow needs an aperture of order the Earth's diameter.
    """
    return float(wavelength_m / aperture_m / UAS)


def required_aperture_m(wavelength_m: float, resolution_uas_target: float) -> float:
    """Aperture needed to reach a given resolution -- the same relation, inverted."""
    return float(wavelength_m / (resolution_uas_target * UAS))


def fringe_response(offset_uas: np.ndarray, baseline_m: float,
                    wavelength_m: float = 1.3e-3) -> np.ndarray:
    """Response of a two-element interferometer to a point source, versus its offset.

    Two dishes separated by ``b`` see a source's signal arrive with a delay that
    changes as the source moves across the sky, so their correlated output
    oscillates -- *fringes* -- with angular period ``lambda / b``. A single pair
    of dishes therefore does not make an image; it measures how much structure
    the sky has on one particular angular scale, in one particular direction.
    """
    return np.cos(2.0 * np.pi * baseline_m / wavelength_m * np.asarray(offset_uas) * UAS)


# ---------------------------------------------------------------------------
# The van Cittert-Zernike relation: a baseline samples one Fourier component
# ---------------------------------------------------------------------------

def image_coordinates(npix: int, fov_uas: float) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centred ``(l, m)`` grids in radians, matching ``images``."""
    half = 0.5 * fov_uas
    axis = np.linspace(-half + 0.5 * fov_uas / npix, half - 0.5 * fov_uas / npix, npix)
    l_grid, m_grid = np.meshgrid(axis * UAS, axis * UAS)
    return l_grid, m_grid


def sample_visibilities(image: np.ndarray | dict, u: np.ndarray, v: np.ndarray,
                        fov_uas: float | None = None) -> np.ndarray:
    """Sample an image's Fourier transform at the given ``(u, v)`` points.

    $$V(u, v) = \\int I(l, m)\\, e^{-2\\pi i (ul + vm)}\\, dl\\, dm$$

    This is the van Cittert-Zernike relation, and it is the entire reason
    interferometry works: **each baseline measures one Fourier component of the
    figs_tel.** Everything downstream is bookkeeping about which components you
    managed to measure.

    Parameters
    ----------
    image : numpy.ndarray or dict
        A Stokes array, or an image dictionary from :mod:`images`
        (in which case Stokes I is used and ``fov_uas`` is taken from it).
    u, v : numpy.ndarray
        Sample coordinates in wavelengths.
    fov_uas : float, optional
        Field of view; required if ``image`` is a bare array.

    Returns
    -------
    numpy.ndarray
        Complex visibilities in Jy, same shape as ``u``.
    """
    if isinstance(image, dict):
        fov_uas = image["fov_uas"]
        image = image["I"]
    if fov_uas is None:
        raise ValueError("fov_uas is required when passing a bare image array")

    l_grid, m_grid = image_coordinates(image.shape[0], fov_uas)
    phase = -2.0j * np.pi * (np.outer(np.ravel(u), np.ravel(l_grid))
                             + np.outer(np.ravel(v), np.ravel(m_grid)))
    return np.exp(phase) @ np.ravel(image)


def gaussian_visibility(u: np.ndarray, v: np.ndarray, fwhm_uas: float,
                        total_flux: float = 1.0) -> np.ndarray:
    """Analytic visibility of a circular Gaussian -- the check on the numerics.

    The Fourier transform of a Gaussian is a Gaussian, wide one way and narrow
    the other: a *broad* source has visibilities that fall off *quickly* with
    baseline length. Compact structure is what long baselines see.
    """
    sigma_rad = fwhm_uas * UAS / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    radius = np.hypot(np.asarray(u), np.asarray(v))
    return total_flux * np.exp(-2.0 * np.pi**2 * sigma_rad**2 * radius**2)


# ---------------------------------------------------------------------------
# Sampling: dirty beam, dirty image, and the Nyquist relations
# ---------------------------------------------------------------------------

def dirty_beam(u: np.ndarray, v: np.ndarray, npix: int = 128,
               fov_uas: float = 200.0) -> dict:
    """Point-spread function of a given uv coverage.

    The transform of the sampling function itself: what a single point source
    would look like after being observed by *this* array on *this* night. Its
    central lobe sets the resolution; its sidelobes are the price of having
    sampled only part of the uv plane.
    """
    beam = _grid_transform(np.ones_like(np.asarray(u, dtype=complex)), u, v, npix, fov_uas)
    peak = beam.max()
    return {"I": beam / peak if peak else beam, "fov_uas": fov_uas,
            "extent": (-0.5 * fov_uas, 0.5 * fov_uas, -0.5 * fov_uas, 0.5 * fov_uas)}


def dirty_image(visibilities: np.ndarray, u: np.ndarray, v: np.ndarray,
                npix: int = 128, fov_uas: float = 200.0) -> dict:
    """Inverse-transform sampled visibilities without deconvolving anything.

    The result is the true sky *convolved with the dirty beam*, so it is not the
    image -- it is the data, drawn in the image plane. Turning it into a picture
    of the sky is the inverse problem the last section of the notebook is about.
    """
    image = _grid_transform(np.asarray(visibilities, dtype=complex), u, v, npix, fov_uas)
    return {"I": image, "fov_uas": fov_uas,
            "extent": (-0.5 * fov_uas, 0.5 * fov_uas, -0.5 * fov_uas, 0.5 * fov_uas)}


def _grid_transform(values: np.ndarray, u: np.ndarray, v: np.ndarray,
                    npix: int, fov_uas: float) -> np.ndarray:
    """Sum ``values * exp(+2 pi i (ul + vm))`` over samples and their conjugates."""
    l_grid, m_grid = image_coordinates(npix, fov_uas)
    u = np.ravel(u)
    v = np.ravel(v)
    values = np.ravel(values)
    # The sky is real, so every sample comes with a free conjugate at (-u, -v).
    u = np.concatenate([u, -u])
    v = np.concatenate([v, -v])
    values = np.concatenate([values, values.conj()])

    if len(values) == 0:
        # A station selection with nothing mutually visible: no samples, no image.
        return np.zeros((npix, npix))

    phase = 2.0j * np.pi * (np.outer(np.ravel(l_grid), u) + np.outer(np.ravel(m_grid), v))
    image = (np.exp(phase) @ values).real / len(values)
    return image.reshape(npix, npix)


def nyquist_pixel_uas(u_max: float, oversample: float = 3.0) -> float:
    """Largest pixel that still samples the finest measured fringe.

    The shortest fringe spacing the array measures is ``1 / u_max``; pixels must
    be a few times smaller than that or the model cannot represent what the data
    constrains. Nothing here is about the *source* -- it is a statement about the
    uv coverage.
    """
    return float(1.0 / (u_max * oversample) / UAS)


def field_of_view_uas(du_min: float) -> float:
    """Field of view implied by the finest uv spacing, ``1 / du``.

    Coarse uv sampling means structure larger than this is aliased -- the same
    Nyquist statement, read the other way round.
    """
    return float(1.0 / du_min / UAS)


def beam_size_uas(u: np.ndarray, v: np.ndarray) -> float:
    """Nominal resolution of a coverage, ``1 / u_max``, in microarcseconds.

    The interferometric analogue of ``lambda / D``: the longest baseline sets
    the finest fringe, and nothing smaller than that fringe is measured. For the
    EHT's 2017 coverage this lands near 25 uas -- about half M87's ring
    diameter, which is exactly why the ring is resolved but only just.
    """
    radius = np.hypot(np.asarray(u), np.asarray(v)).max()
    return float(1.0 / radius / UAS)


# ---------------------------------------------------------------------------
# Corruption, and the closure quantities that survive it
# ---------------------------------------------------------------------------

def apply_station_gains(visibility: np.ndarray, gain1: complex,
                        gain2: complex) -> np.ndarray:
    """Corrupt a visibility with the two stations' complex gains: ``g_1 g_2^* V``.

    Atmosphere and electronics scale and phase-shift each station's signal, and
    the phase error in particular is large and fast at 230 GHz. It is *station*
    based, which is the loophole the closure quantities exploit.
    """
    return gain1 * np.conj(gain2) * np.asarray(visibility)


def closure_phase(v12: complex, v23: complex, v31: complex) -> float:
    """Sum of visibility phases around a triangle, in radians.

    Every station-based phase error appears once with each sign and cancels
    exactly:

    $$\\arg(g_1 g_2^* V_{12}) + \\arg(g_2 g_3^* V_{23}) + \\arg(g_3 g_1^* V_{31})
      = \\arg V_{12} + \\arg V_{23} + \\arg V_{31}$$

    So a closure phase is a genuine, calibration-independent statement about the
    source -- and a non-zero one is direct evidence of asymmetric structure.
    """
    return float(np.angle(np.asarray(v12) * np.asarray(v23) * np.asarray(v31)))


def log_closure_amplitude(v12: complex, v34: complex, v13: complex,
                          v24: complex) -> float:
    """The amplitude equivalent, on a quadrangle: ``log |V12 V34 / (V13 V24)|``.

    Immune to station-based *amplitude* errors for the same reason.
    """
    return float(np.log(np.abs(v12 * v34) / np.abs(v13 * v24)))


def add_thermal_noise(visibilities: np.ndarray, sigma: float | np.ndarray,
                      seed: int = 0) -> np.ndarray:
    """Add complex Gaussian noise of the given per-component standard deviation."""
    rng = np.random.default_rng(seed)
    shape = np.shape(visibilities)
    return (np.asarray(visibilities)
            + rng.normal(0.0, sigma, shape) + 1.0j * rng.normal(0.0, sigma, shape))
