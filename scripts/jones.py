"""Everything between the sky and the recorded number: Jones matrices and the RIME.

A station does not measure the electric field. It measures two voltages, each the
projection of the field onto one feed, after the signal has been rotated by the
mount, leaked between feeds by imperfect optics, and scaled by an unknown gain.
All of that is one 2x2 complex matrix per station, the **Jones matrix**:

    v = J e,        J = G (I + D) Phi

and a baseline correlates two stations, giving the **radio interferometer
measurement equation**

    V = J_1 <e e^dagger> J_2^dagger = J_1 C J_2^dagger.

Conventions follow eht-imaging (``ehtim/observing/pol_conventions.py`` and
``docs/polarization_conventions.md``): the IAU / Hamaker-Bregman-Sault basis with
``R = (X + iY)/sqrt(2)``, and the correlator normalisation in which an unpolarized
source of flux ``S`` gives ``RR = LL = S``, i.e.

    I = (XX + YY)/2,  Q = (XX - YY)/2,  U = (XY + YX)/2,  V = -i(XY - YX)/2.

The four correlation products of a baseline are kept in eht-imaging's slot order
``(p1p1, p2p2, p1p2, p2p1)``, where ``p1`` and ``p2`` are a station's first and
second feed. What those slots *mean* depends on both stations' feeds -- which is
the whole point of the mixed-polarization problem.
"""

from __future__ import annotations

import numpy as np
import polarization as pol

#: The feed characters this module understands, and the field each responds to.
FEED_VECTORS = {
    "x": np.array([1.0, 0.0], dtype=complex),
    "y": np.array([0.0, 1.0], dtype=complex),
    "r": pol.BASIS_LIN_TO_CIRC[0].copy(),   # (X + iY)/sqrt(2)
    "l": pol.BASIS_LIN_TO_CIRC[1].copy(),   # (X - iY)/sqrt(2)
}

#: Feed pairs a station can have. 'rl' is circular, 'xy' linear.
FEED_TYPES = ("rl", "xy")

IDENTITY = np.eye(2, dtype=complex)


# ---------------------------------------------------------------------------
# Feeds: what a station's two receivers respond to
# ---------------------------------------------------------------------------

def feed_matrix(feed_type: str) -> np.ndarray:
    """The 2x2 matrix taking a field ``(E_X, E_Y)`` to one station's two feeds.

    Rows are the station's feeds in order, columns are the linear components. So
    ``feed_matrix('xy')`` is the identity -- a linear-feed station measures the
    components directly -- and ``feed_matrix('rl')`` is the circular basis change.

    Parameters
    ----------
    feed_type : str
        Two feed characters from ``FEED_VECTORS``, e.g. ``'rl'`` or ``'xy'``.

    Returns
    -------
    numpy.ndarray
        ``(2, 2)`` complex, unitary for an orthogonal pair.
    """
    if len(feed_type) != 2 or any(char not in FEED_VECTORS for char in feed_type):
        raise ValueError(f"feed_matrix: unsupported feed_type {feed_type!r}")
    if feed_type[0] == feed_type[1]:
        raise ValueError(f"feed_matrix: a station needs two different feeds, got "
                         f"{feed_type!r}")
    return np.array([FEED_VECTORS[feed_type[0]], FEED_VECTORS[feed_type[1]]])


def slot_labels(feed1: str, feed2: str) -> tuple[str, str, str, str]:
    """What the four correlation products of a baseline are called.

    In eht-imaging's slot order ``(p1p1, p2p2, p1p2, p2p1)``. Both ends circular
    gives the familiar ``RR, LL, RL, LR``; both linear gives ``XX, YY, XY, YX``;
    one of each gives labels like ``XR, YL, XL, YR``, which no imaging code
    written for a single basis knows how to interpret.
    """
    a1, a2 = feed1[0].upper(), feed1[1].upper()
    b1, b2 = feed2[0].upper(), feed2[1].upper()
    return (a1 + b1, a2 + b2, a1 + b2, a2 + b1)


def polbasis(feed1: str, feed2: str) -> str:
    """The per-baseline basis string eht-imaging stores, e.g. ``'rlxy'``.

    A dataset with a single value here is homogeneous; a dataset with several is
    what ``polrep='mixed'`` exists for.
    """
    return f"{feed1.lower()}{feed2.lower()}"


def is_mixed(feed1: str, feed2: str) -> bool:
    """True if the two ends of this baseline use different feed bases."""
    return feed1.lower() != feed2.lower()


# ---------------------------------------------------------------------------
# The coherency matrix, and the four slots of a baseline
# ---------------------------------------------------------------------------

def coherency_from_stokes(i, q, u, v) -> np.ndarray:
    """The sky's coherency matrix in the linear basis, from Stokes parameters.

    Inverting the correlator-convention definitions above:
    ``XX = I + Q``, ``YY = I - Q``, ``XY = U + iV``, ``YX = U - iV``.

    Returns
    -------
    numpy.ndarray
        ``(2, 2)`` Hermitian complex matrix.
    """
    return np.array([[i + q, u + 1j * v],
                     [u - 1j * v, i - q]], dtype=complex)


def stokes_from_coherency(coherency: np.ndarray) -> np.ndarray:
    """Stokes ``[I, Q, U, V]`` from a coherency matrix **in the linear basis**.

    Only correct for a linear-basis matrix. A baseline's matrix from
    :func:`baseline_coherency` is in the *feeds'* basis, which for circular feeds
    is not the same thing -- feeding one to this function silently shuffles the
    Stokes parameters. Use :func:`slots_of` and :func:`stokes_from_slots` for a
    baseline, which take the feed types as arguments and cannot make that mistake.
    """
    xx, xy, yx, yy = (coherency[0, 0], coherency[0, 1],
                      coherency[1, 0], coherency[1, 1])
    return np.real(np.array([0.5 * (xx + yy), 0.5 * (xx - yy),
                             0.5 * (xy + yx), -0.5j * (xy - yx)]))


def baseline_coherency(stokes, feed1: str, feed2: str) -> np.ndarray:
    """The 2x2 correlation matrix a baseline would record from a clean sky.

    ``V = F_1 C F_2^dagger``: project the sky's coherency onto station 1's feeds
    on one side and station 2's on the other. With different feeds at the two
    ends, the result is not Hermitian -- there is no reason it should be, because
    its rows and columns are indexed by different things.
    """
    coherency = coherency_from_stokes(*stokes)
    return feed_matrix(feed1) @ coherency @ feed_matrix(feed2).conj().T


def slots_of(matrix: np.ndarray) -> np.ndarray:
    """Read a baseline's 2x2 matrix out in slot order ``(p1p1, p2p2, p1p2, p2p1)``."""
    return np.array([matrix[0, 0], matrix[1, 1], matrix[0, 1], matrix[1, 0]])


def slots_from_stokes(stokes, feed1: str, feed2: str) -> np.ndarray:
    """The four correlation products, in slot order ``(p1p1, p2p2, p1p2, p2p1)``."""
    return slots_of(baseline_coherency(stokes, feed1, feed2))


def slots_to_stokes_matrix(feed1: str, feed2: str) -> np.ndarray:
    """The 4x4 operator taking a baseline's four slots to ``[I, Q, U, V]``.

    Built by pushing the four unit Stokes vectors through
    :func:`slots_from_stokes` and inverting, so it cannot drift out of step with
    the forward direction. This matrix is the thing that differs from baseline to
    baseline in a mixed array, and the reason a single global conversion cannot
    work.

    Returns
    -------
    numpy.ndarray
        ``(4, 4)`` complex.
    """
    forward = np.column_stack([slots_from_stokes(unit, feed1, feed2)
                               for unit in np.eye(4)])
    return np.linalg.inv(forward)


def stokes_from_slots(slots, feed1: str, feed2: str) -> np.ndarray:
    """Recover ``[I, Q, U, V]`` from four correlation products."""
    return np.real(slots_to_stokes_matrix(feed1, feed2) @ np.asarray(slots))


def misread_stokes(stokes, true_feeds: tuple[str, str],
                   assumed_feeds: tuple[str, str]) -> np.ndarray:
    """Stokes recovered from a mixed baseline by software that assumes one basis.

    The failure mode this whole notebook is about: the correlator writes four
    numbers, imaging code multiplies them by the 4x4 matrix for the basis it
    *believes* the baseline used, and out comes a Stokes vector that is wrong in
    a way nothing downstream can detect.

    Parameters
    ----------
    stokes : array-like
        True ``[I, Q, U, V]`` on the sky.
    true_feeds : tuple of str
        The feed types the two stations really have.
    assumed_feeds : tuple of str
        What the software assumes.

    Returns
    -------
    numpy.ndarray
        The Stokes vector that comes out the far end.
    """
    slots = slots_from_stokes(stokes, *true_feeds)
    return stokes_from_slots(slots, *assumed_feeds)


# ---------------------------------------------------------------------------
# The three factors of J
# ---------------------------------------------------------------------------

def gain_matrix(gain_p1: complex, gain_p2: complex) -> np.ndarray:
    """``G``: one unknown complex number per feed, scaling and phasing it."""
    return np.array([[gain_p1, 0.0], [0.0, gain_p2]], dtype=complex)


def dterm_matrix(d_p1: complex, d_p2: complex) -> np.ndarray:
    """``I + D``: how much of each feed's signal leaks into the other one.

    ``d_p1`` is the leakage of feed 2 into feed 1. A few percent is typical, and
    it is the dominant systematic in EHT polarimetry, because leakage on an
    unpolarized source is indistinguishable from real polarization.
    """
    return np.array([[1.0, d_p1], [d_p2, 1.0]], dtype=complex)


def field_rotation_matrix(feed_type: str, angle: float) -> np.ndarray:
    """``Phi``: the feeds rotated by ``angle`` relative to the sky.

    Defined once, in the only way that cannot be inconsistent: rotate the *sky*
    frame, then express that rotation in this station's feed basis,
    ``Phi = F R F^-1``. Everything people usually state as two separate rules
    then falls out --

    * circular feeds: ``Phi`` comes out **diagonal**, a phase ``e^(-i angle)`` on
      one feed and ``e^(+i angle)`` on the other. Amplitudes are untouched.
    * linear feeds: ``Phi`` comes out as a **real rotation**, which mixes the two
      feeds into each other, and so mixes Stokes Q and U by twice the angle.

    -- and they are guaranteed to be the same physical rotation, seen from two
    coordinate systems.

    Parameters
    ----------
    feed_type : str
        The station's feeds, ``'rl'`` or ``'xy'``.
    angle : float
        Field rotation angle in radians (see ``arrays.field_rotation_angle``).

    Returns
    -------
    numpy.ndarray
        ``(2, 2)`` complex, unitary.
    """
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]], dtype=complex)
    feeds = feed_matrix(feed_type)
    return feeds @ rotation @ np.linalg.inv(feeds)


def jones_matrix(gain_p1: complex = 1.0, gain_p2: complex = 1.0,
                 d_p1: complex = 0.0, d_p2: complex = 0.0,
                 feed_type: str = "rl", rotation: float = 0.0) -> np.ndarray:
    """Assemble ``J = G (I + D) Phi``.

    The order is physics, not bookkeeping: the sky is rotated by the mount
    *first*, then the feeds leak into each other, then the electronics scale the
    result. Matrices do not commute, so swapping two factors gives a different
    instrument -- see the notebook.
    """
    return (gain_matrix(gain_p1, gain_p2)
            @ dterm_matrix(d_p1, d_p2)
            @ field_rotation_matrix(feed_type, rotation))


def apply_jones(jones1: np.ndarray, coherency: np.ndarray,
                jones2: np.ndarray) -> np.ndarray:
    """The RIME: ``V_obs = J_1 C J_2^dagger``."""
    return jones1 @ coherency @ jones2.conj().T


def observe(stokes, feed1: str, feed2: str,
            jones1: np.ndarray | None = None,
            jones2: np.ndarray | None = None) -> np.ndarray:
    """Full forward model, sky to slots: project onto feeds, then corrupt.

    Returns
    -------
    numpy.ndarray
        Four complex slots in order ``(p1p1, p2p2, p1p2, p2p1)``.
    """
    jones1 = IDENTITY if jones1 is None else jones1
    jones2 = IDENTITY if jones2 is None else jones2
    clean = baseline_coherency(stokes, feed1, feed2)
    return slots_of(apply_jones(jones1, clean, jones2))


def recover_coherency(observed: np.ndarray, jones1: np.ndarray,
                      jones2: np.ndarray) -> np.ndarray:
    """Undo the RIME: ``C = J_1^-1 V J_2^-dagger``. Calibration, in one line.

    Exact when the Jones matrices are known exactly, which they never are -- and
    the reason polarimetric calibration is the hard part of polarimetry.
    """
    return (np.linalg.inv(jones1) @ observed
            @ np.linalg.inv(jones2.conj().T))


# ---------------------------------------------------------------------------
# What leakage does
# ---------------------------------------------------------------------------

def spurious_polarization(d_p1: complex, d_p2: complex, feed_type: str = "rl",
                          rotation: float = 0.0) -> float:
    """Fractional polarization that leakage invents on an unpolarized source.

    Feed an unpolarized sky (``Q = U = V = 0``) through a baseline whose two
    stations both have these D-terms, and read the linear polarization fraction
    that comes out. It is ``|d_p1 + conj(d_p2)|`` -- of order the D-term itself,
    with both ends contributing, so 1% leakage at each station manufactures about
    2% polarization. That is why EHT D-terms have to be solved to a fraction of a
    percent before a polarized image means anything.

    Note the ``+ conj`` in that expression: leakage does not always add up. Feeds
    with ``d_p2 = -d_p1`` cancel exactly on the cross-hand products and invent no
    polarization at all, while ``d_p2 = conj(d_p1)`` gives the full ``2|d|``
    whatever the phase. The explorers use the latter, so that the magnitude slider
    controls how much false polarization appears and the phase slider controls
    which way it points.
    """
    corrupt = jones_matrix(1.0, 1.0, d_p1, d_p2, feed_type, rotation)
    slots = observe([1.0, 0.0, 0.0, 0.0], feed_type, feed_type, corrupt, corrupt)
    stokes = stokes_from_slots(slots, feed_type, feed_type)
    return float(pol.frac_lin(*stokes[:3]))
