"""Polarization of a quasi-monochromatic wave: fields, coherency, Stokes, ellipses.

Conventions follow eht-imaging (`ehtim/observing/pol_conventions.py` and
`docs/polarization_conventions.md`): the IAU / Hamaker-Bregman-Sault choice with
the engineering time dependence ``exp(+i omega t)``,

    R = (X + iY) / sqrt(2),      L = (X - iY) / sqrt(2).

One deliberate difference. ehtim works with *correlator* products, where a
station's two feed voltages are normalised so that an unpolarized source of
flux ``S`` gives ``RR = LL = S``; that convention carries a factor of one half,

    I = (XX + YY) / 2,  Q = (XX - YY) / 2,  U = (XY + YX) / 2,  V = -i(XY - YX)/2

with ``XY = <E_X E_Y*>``. Here we are looking at a single wave rather than a
correlator output, so the natural normalisation is total intensity,

    I = |E_X|^2 + |E_Y|^2,   Q = |E_X|^2 - |E_Y|^2,
    U = 2 Re(E_X E_Y*),      V = 2 Im(E_X E_Y*),

which is exactly twice the ehtim quantities. The *signs* -- the part that is
easy to get wrong and hard to notice -- are identical, and
``stokes_from_lin_corr`` / ``stokes_from_circ_corr`` below implement the ehtim
correlator forms verbatim so the two can be compared directly (see
``tests/test_polarization.py``).

Handedness: ``V > 0`` is right-hand circular. In the (x, y) plane as drawn --
x to the right, y up -- the physical field ``Re[E exp(i omega t)]`` then rotates
counter-clockwise. Tying x and y to sky axes (north/east) is a separate question
handled with field rotation in notebook 03.
"""

from __future__ import annotations

import numpy as np

SQRT2 = np.sqrt(2.0)

# Maps (E_X, E_Y) -> (E_R, E_L) under the IAU/HBS convention above.
# Same matrix as ehtim's pol_conventions.BASIS_LIN_TO_CIRC.
BASIS_LIN_TO_CIRC = np.array([[1.0, +1.0j],
                              [1.0, -1.0j]]) / SQRT2
BASIS_CIRC_TO_LIN = BASIS_LIN_TO_CIRC.conj().T


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------

def jones_vector(amp_x: float, amp_y: float, delta: float) -> np.ndarray:
    """Complex field amplitude vector for a fully polarized monochromatic wave.

    Parameters
    ----------
    amp_x, amp_y : float
        Real amplitudes of the x and y components.
    delta : float
        Phase of y relative to x, in radians (``delta = phi_y - phi_x``).

    Returns
    -------
    numpy.ndarray
        Complex array ``[E_X, E_Y]``, with the overall phase fixed by
        ``phi_x = 0``. Only the relative phase is physical.
    """
    return np.array([amp_x + 0.0j, amp_y * np.exp(1.0j * delta)])


def wave_trace(field: np.ndarray, n_periods: float = 1.0,
               n_samples: int = 361) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical field ``Re[E exp(i omega t)]`` sampled over whole periods.

    Parameters
    ----------
    field : numpy.ndarray
        Complex ``[E_X, E_Y]``.
    n_periods : float, optional
        How many wave periods to trace.
    n_samples : int, optional
        Number of samples.

    Returns
    -------
    phase, e_x, e_y : numpy.ndarray
        ``phase = omega t`` in radians and the two real field components. The
        ellipse traced by ``(e_x, e_y)`` is the polarization ellipse; its sense
        of rotation is the handedness.
    """
    phase = np.linspace(0.0, 2.0 * np.pi * n_periods, n_samples)
    osc = np.exp(1.0j * phase)
    return phase, np.real(field[0] * osc), np.real(field[1] * osc)


def coherency_matrix(field: np.ndarray) -> np.ndarray:
    """Coherency matrix ``<E E^dagger>`` of a single fully polarized wave.

    For one wave this is the rank-1 outer product; partially polarized light is
    an incoherent *sum* of such matrices (see :func:`incoherent_sum`), which is
    why the coherency matrix -- not the field -- is the right object for real
    light.

    Parameters
    ----------
    field : numpy.ndarray
        Complex ``[E_X, E_Y]``.

    Returns
    -------
    numpy.ndarray
        Hermitian ``(2, 2)`` complex matrix
        ``[[<XX>, <XY>], [<YX>, <YY>]]`` with ``XY = <E_X E_Y*>``.
    """
    e = np.asarray(field, dtype=complex).reshape(2)
    return np.outer(e, e.conj())


def lin_to_circ(field: np.ndarray) -> np.ndarray:
    """Rewrite a field in the circular basis: ``(E_X, E_Y) -> (E_R, E_L)``."""
    return BASIS_LIN_TO_CIRC @ np.asarray(field, dtype=complex).reshape(2)


def circ_to_lin(field: np.ndarray) -> np.ndarray:
    """Rewrite a field in the linear basis: ``(E_R, E_L) -> (E_X, E_Y)``."""
    return BASIS_CIRC_TO_LIN @ np.asarray(field, dtype=complex).reshape(2)


# ---------------------------------------------------------------------------
# Stokes parameters
# ---------------------------------------------------------------------------

def stokes_from_coherency(coh: np.ndarray) -> np.ndarray:
    """Stokes ``[I, Q, U, V]`` (total-intensity normalisation) from ``<E E^dagger>``.

    Parameters
    ----------
    coh : numpy.ndarray
        Hermitian ``(2, 2)`` coherency matrix in the *linear* basis.

    Returns
    -------
    numpy.ndarray
        Real array ``[I, Q, U, V]``.
    """
    coh = np.asarray(coh, dtype=complex)
    xx, xy, yx, yy = coh[0, 0], coh[0, 1], coh[1, 0], coh[1, 1]
    i = np.real(xx + yy)
    q = np.real(xx - yy)
    u = np.real(xy + yx)          # = 2 Re(XY), since YX = XY*
    v = np.real(-1.0j * (xy - yx))  # = 2 Im(XY)
    return np.array([i, q, u, v])


def stokes_from_field(field: np.ndarray) -> np.ndarray:
    """Stokes ``[I, Q, U, V]`` of a single fully polarized wave."""
    return stokes_from_coherency(coherency_matrix(field))


def stokes_from_lin_corr(xx, yy, xy, yx):
    """Stokes from linear-feed correlations, in ehtim's correlator normalisation.

    Verbatim ``docs/polarization_conventions.md`` section 5:
    ``I = (XX+YY)/2, Q = (XX-YY)/2, U = (XY+YX)/2, V = -i(XY-YX)/2``.
    Included so the report's field-level algebra can be checked against the
    shipped code; note the factor of two relative to
    :func:`stokes_from_coherency` (see the module docstring).
    """
    return (0.5 * (xx + yy), 0.5 * (xx - yy),
            0.5 * (xy + yx), -0.5j * (xy - yx))


def stokes_from_circ_corr(rr, ll, rl, lr):
    """Stokes from circular-feed correlations, in ehtim's correlator normalisation.

    ``docs/polarization_conventions.md`` section 4:
    ``I = (RR+LL)/2, Q = (RL+LR)/2, U = i(LR-RL)/2, V = (RR-LL)/2``.
    """
    return (0.5 * (rr + ll), 0.5 * (rl + lr),
            0.5j * (lr - rl), 0.5 * (rr - ll))


def evpa(q: float, u: float) -> float:
    """Electric vector position angle, ``0.5 * arctan2(U, Q)``, in radians.

    Defined modulo pi: an EVPA of 0 and of pi describe the same wiggle, because
    the ellipse has no arrowhead. That factor of two -- EVPA is *half* the
    azimuth on the Poincare sphere -- is the source of every position-angle
    ambiguity in polarimetry.
    """
    return 0.5 * np.arctan2(u, q)


def _safe_divide(numerator, denominator):
    """``numerator / denominator``, giving 0 where the denominator is 0.

    Every fraction here divides by Stokes I, and an *image* has plenty of pixels
    where I is exactly zero -- so dividing naively fills the result with nans and
    a warning. Scalars in, float out; arrays in, array out.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    out = np.zeros(np.broadcast(num, den).shape, dtype=float)
    result = np.divide(num, den, out=out, where=den != 0)
    return result if result.ndim else float(result)


def require_single_state(caller: str, *values) -> tuple[float, ...]:
    """Coerce Stokes arguments to floats, or say clearly why that is impossible.

    A few functions here describe *one* polarization state and cannot be
    vectorised in any useful way -- they return a curve, or a sentence. Handing
    them a whole Stokes image is an easy mistake to make (the image helpers hand
    out arrays), and numpy's own complaint about ambiguous truth values explains
    nothing. This says what to do instead.
    """
    arrays = [np.asarray(value) for value in values]
    if any(array.size != 1 for array in arrays):
        raise ValueError(
            f"{caller}() describes a single polarization state, but was given "
            f"arrays of size {max(array.size for array in arrays)}. For a whole "
            "image use core.images.evpa_ticks or core.images.image_stokes_totals, "
            "or pick one pixel first, e.g. image['Q'][64, 64]."
        )
    return tuple(float(array) for array in arrays)


def frac_lin(i, q, u):
    """Linear polarization fraction ``sqrt(Q^2 + U^2) / I``. Accepts arrays."""
    return _safe_divide(np.hypot(q, u), i)


def frac_circ(i, v):
    """Circular polarization fraction ``V / I`` (signed: ``+`` is right-handed)."""
    return _safe_divide(v, i)


def frac_total(i, q, u, v):
    """Total polarization fraction ``sqrt(Q^2 + U^2 + V^2) / I``, in ``[0, 1]``."""
    return _safe_divide(np.sqrt(np.asarray(q)**2 + np.asarray(u)**2 + np.asarray(v)**2), i)


def is_physical(i, q, u, v, tol: float = 1e-12) -> bool:
    """True if ``I >= 0`` and ``I^2 >= Q^2 + U^2 + V^2`` everywhere.

    Accepts arrays, in which case it answers for the whole image at once -- one
    bad pixel makes the answer False.
    """
    i, q, u, v = (np.asarray(x, dtype=float) for x in (i, q, u, v))
    return bool(np.all(i >= -tol) and np.all(i * i + tol >= q * q + u * u + v * v))


def stokes_from_ellipse(intensity: float = 1.0, p_lin: float = 0.0,
                        evpa_rad: float = 0.0, p_circ: float = 0.0) -> np.ndarray:
    """Build Stokes ``[I, Q, U, V]`` from polarization fractions and an angle.

    Parameters
    ----------
    intensity : float, optional
        Stokes I.
    p_lin : float, optional
        Linear polarization fraction in ``[0, 1]``.
    evpa_rad : float, optional
        EVPA in radians.
    p_circ : float, optional
        Signed circular polarization fraction in ``[-1, 1]``.

    Returns
    -------
    numpy.ndarray
        ``[I, Q, U, V]``.

    Raises
    ------
    ValueError
        If the requested fractions exceed unity in quadrature -- i.e. describe
        light that cannot exist.
    """
    total = np.hypot(p_lin, p_circ)
    if np.any(total > 1.0 + 1e-12):
        raise ValueError(
            f"unphysical state: sqrt(p_lin^2 + p_circ^2) reaches {np.max(total):.3f} > 1"
        )
    intensity = np.asarray(intensity, dtype=float)
    return np.array([intensity,
                     intensity * p_lin * np.cos(2.0 * evpa_rad),
                     intensity * p_lin * np.sin(2.0 * evpa_rad),
                     intensity * np.broadcast_to(p_circ, intensity.shape)])


# ---------------------------------------------------------------------------
# The polarization ellipse
# ---------------------------------------------------------------------------

def ellipse_from_stokes(i: float, q: float, u: float, v: float) -> dict[str, float]:
    """Geometry of the polarization ellipse of the *polarized part* of the light.

    Partially polarized light splits uniquely into a fully polarized wave plus
    an unpolarized remainder; only the former traces an ellipse, so the axes
    below are scaled to the polarized intensity ``P = sqrt(Q^2+U^2+V^2)``.

    Works elementwise, so a whole Stokes image can be passed and every entry of
    the returned dictionary comes back with the image's shape.

    Parameters
    ----------
    i, q, u, v : float or numpy.ndarray
        Stokes parameters.

    Returns
    -------
    dict
        ``semi_major``, ``semi_minor`` (field-amplitude units, so that
        ``a^2 + b^2 = P``), ``evpa`` (radians), ``axis_ratio`` ``b/a``,
        ``handedness`` (``+1`` right, ``-1`` left, ``0`` linear),
        ``p_total``, ``p_lin``, ``p_circ``.
    """
    i, q, u, v = (np.asarray(x, dtype=float) for x in (i, q, u, v))
    p_pol = np.sqrt(q * q + u * u + v * v)
    lin = np.hypot(q, u)
    semi_major = np.sqrt(np.maximum(0.5 * (p_pol + lin), 0.0))
    semi_minor = np.sqrt(np.maximum(0.5 * (p_pol - lin), 0.0))
    scalar = semi_major.ndim == 0
    handedness = np.sign(v).astype(int)
    return {
        "semi_major": float(semi_major) if scalar else semi_major,
        "semi_minor": float(semi_minor) if scalar else semi_minor,
        "evpa": float(evpa(q, u)) if scalar else evpa(q, u),
        "axis_ratio": _safe_divide(semi_minor, semi_major),
        "handedness": int(handedness) if scalar else handedness,
        "p_total": _safe_divide(p_pol, i),
        "p_lin": _safe_divide(lin, i),
        "p_circ": _safe_divide(v, i),
    }


def ellipse_trace(i: float, q: float, u: float, v: float,
                  n_samples: int = 361) -> tuple[np.ndarray, np.ndarray]:
    """Sample the polarization ellipse of the polarized part of the light.

    Describes one state, so it wants four numbers -- pass a single pixel rather
    than a whole image.

    Returns
    -------
    x, y : numpy.ndarray
        Field components tracing the ellipse once, in the same sense (and
        starting phase) as :func:`wave_trace`, so the two agree for fully
        polarized light.
    """
    i, q, u, v = require_single_state("ellipse_trace", i, q, u, v)
    geo = ellipse_from_stokes(i, q, u, v)
    a, b, chi = geo["semi_major"], geo["semi_minor"], geo["evpa"]
    # Handedness sits in the sign of the minor-axis term: the parametrisation
    # (a cos t, +b sin t) has positive signed area, i.e. it runs counter-clockwise
    # in the drawn plane, which is what V > 0 means (see the module docstring).
    hand = geo["handedness"] or 1
    t = np.linspace(0.0, 2.0 * np.pi, n_samples)
    x_p, y_p = a * np.cos(t), hand * b * np.sin(t)
    return (x_p * np.cos(chi) - y_p * np.sin(chi),
            x_p * np.sin(chi) + y_p * np.cos(chi))


def poincare_point(i: float, q: float, u: float, v: float) -> np.ndarray:
    """Position on/in the Poincare sphere, ``[Q, U, V] / I``.

    The surface is fully polarized light, the centre is unpolarized, the
    equator is linear and the poles are circular. Azimuth is ``2 * EVPA``.
    """
    return np.array([q, u, v]) / i


# ---------------------------------------------------------------------------
# Partial polarization and propagation effects
# ---------------------------------------------------------------------------

def incoherent_sum(n_waves: int = 200, spread_rad: float = 0.0,
                   evpa0_rad: float = 0.0, p_circ: float = 0.0,
                   seed: int = 0) -> np.ndarray:
    """Stokes of an incoherent sum of linearly polarized wavelets.

    A model of how real emission ends up *partially* polarized: many emitters
    contribute, each fully polarized, but with position angles scattered around
    a mean. Stokes parameters add; field amplitudes do not.

    Parameters
    ----------
    n_waves : int, optional
        Number of wavelets, each carrying intensity ``1 / n_waves``.
    spread_rad : float, optional
        Standard deviation of the wavelet EVPAs, in radians. ``0`` gives fully
        polarized light; a spread of order ``pi/2`` gives essentially none.
    evpa0_rad : float, optional
        Mean EVPA.
    p_circ : float, optional
        Circular fraction given to every wavelet (kept coherent, so it
        survives the averaging).
    seed : int, optional
        Seed for the position-angle draw, so figures are reproducible.

    Returns
    -------
    numpy.ndarray
        ``[I, Q, U, V]`` of the sum, normalised to ``I = 1``.
    """
    rng = np.random.default_rng(seed)
    angles = evpa0_rad + spread_rad * rng.standard_normal(n_waves)
    p_lin = np.sqrt(max(1.0 - p_circ * p_circ, 0.0))
    stokes = np.zeros(4)
    for ang in angles:
        stokes += stokes_from_ellipse(1.0 / n_waves, p_lin, ang, p_circ)
    return stokes


def faraday_evpa(evpa0_rad: float, rm_rad_m2: float,
                 wavelength_m: float | np.ndarray) -> float | np.ndarray:
    """EVPA after Faraday rotation: ``chi = chi_0 + RM * lambda^2``.

    Magnetised plasma along the line of sight rotates the plane of linear
    polarization by an amount that scales as ``lambda^2``, so observing at
    several frequencies measures the rotation measure -- and therefore the
    line-of-sight magnetic field. This is the reason the mixed-polarization
    machinery has to work per frequency channel.

    Parameters
    ----------
    evpa0_rad : float
        Intrinsic EVPA at zero wavelength, in radians.
    rm_rad_m2 : float
        Rotation measure in rad m^-2.
    wavelength_m : float or numpy.ndarray
        Observing wavelength(s) in metres.

    Returns
    -------
    float or numpy.ndarray
        Observed EVPA in radians (not wrapped).
    """
    return evpa0_rad + rm_rad_m2 * np.asarray(wavelength_m) ** 2
