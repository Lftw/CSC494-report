"""Physics checks for core.polarization: analytic states, conventions, invariants.

The point of these is the *signs*. Anyone can get I right; V and the EVPA are
where polarimetry code goes quietly wrong, so the special states are pinned
against hand-derived values and the field-level algebra is cross-checked against
the correlator formulas that eht-imaging ships.
"""

import numpy as np
import polarization as pol
import pytest

SQ = 1.0 / np.sqrt(2.0)


# ---------------------------------------------------------------------------
# Canonical states
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("amp_x", "amp_y", "delta_deg", "expected", "label"), [
    (1.0, 0.0, 0.0, (1.0, 1.0, 0.0, 0.0), "linear horizontal"),
    (0.0, 1.0, 0.0, (1.0, -1.0, 0.0, 0.0), "linear vertical"),
    (SQ, SQ, 0.0, (1.0, 0.0, 1.0, 0.0), "linear at +45 deg"),
    (SQ, SQ, 180.0, (1.0, 0.0, -1.0, 0.0), "linear at -45 deg"),
    (SQ, SQ, -90.0, (1.0, 0.0, 0.0, 1.0), "right circular (V > 0)"),
    (SQ, SQ, 90.0, (1.0, 0.0, 0.0, -1.0), "left circular (V < 0)"),
])
def test_canonical_states(amp_x, amp_y, delta_deg, expected, label):
    field = pol.jones_vector(amp_x, amp_y, np.deg2rad(delta_deg))
    assert np.allclose(pol.stokes_from_field(field), expected, atol=1e-12), label


def test_right_circular_is_pure_r_in_the_circular_basis():
    """V > 0 must mean all the light lands in the R feed and none in L."""
    field = pol.jones_vector(SQ, SQ, np.deg2rad(-90.0))
    e_r, e_l = pol.lin_to_circ(field)
    assert abs(e_r) == pytest.approx(1.0, abs=1e-12)
    assert abs(e_l) == pytest.approx(0.0, abs=1e-12)
    assert pol.stokes_from_field(field)[3] > 0


def test_left_circular_is_pure_l():
    field = pol.jones_vector(SQ, SQ, np.deg2rad(90.0))
    e_r, e_l = pol.lin_to_circ(field)
    assert abs(e_l) == pytest.approx(1.0, abs=1e-12)
    assert abs(e_r) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Basis change
# ---------------------------------------------------------------------------

def test_basis_matrices_are_unitary_inverses():
    identity = pol.BASIS_LIN_TO_CIRC @ pol.BASIS_CIRC_TO_LIN
    assert np.allclose(identity, np.eye(2), atol=1e-14)


@pytest.mark.parametrize("delta_deg", [-135.0, -40.0, 0.0, 37.0, 175.0])
def test_round_trip_through_circular_basis(delta_deg):
    field = pol.jones_vector(0.8, 0.4, np.deg2rad(delta_deg))
    assert np.allclose(pol.circ_to_lin(pol.lin_to_circ(field)), field, atol=1e-14)


@pytest.mark.parametrize("delta_deg", [-120.0, -30.0, 0.0, 55.0, 160.0])
def test_stokes_agree_between_linear_and_circular_correlations(delta_deg):
    """The convention check that matters: same wave, two feed bases, same Stokes.

    Mirrors section 7 of eht-imaging's ``docs/polarization_conventions.md``. A
    flipped sign anywhere in ``BASIS_LIN_TO_CIRC`` or in either correlator
    formula shows up here as a sign error on U or V.
    """
    field = pol.jones_vector(0.9, 0.6, np.deg2rad(delta_deg))
    e_x, e_y = field
    e_r, e_l = pol.lin_to_circ(field)

    from_lin = pol.stokes_from_lin_corr(e_x * e_x.conj(), e_y * e_y.conj(),
                                       e_x * e_y.conj(), e_y * e_x.conj())
    from_circ = pol.stokes_from_circ_corr(e_r * e_r.conj(), e_l * e_l.conj(),
                                          e_r * e_l.conj(), e_l * e_r.conj())
    assert np.allclose(np.real(from_lin), np.real(from_circ), atol=1e-14)

    # ...and both equal half the field-level Stokes (see the module docstring).
    assert np.allclose(np.real(from_lin), 0.5 * pol.stokes_from_field(field), atol=1e-14)


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("evpa_deg", [-80.0, -45.0, 0.0, 22.5, 67.5, 89.0])
def test_evpa_round_trips(evpa_deg):
    stokes = pol.stokes_from_ellipse(2.0, 0.4, np.deg2rad(evpa_deg), 0.0)
    assert np.rad2deg(pol.evpa(stokes[1], stokes[2])) == pytest.approx(evpa_deg, abs=1e-9)


def test_evpa_is_ambiguous_by_180_degrees():
    a = pol.stokes_from_ellipse(1.0, 0.5, np.deg2rad(20.0), 0.0)
    b = pol.stokes_from_ellipse(1.0, 0.5, np.deg2rad(200.0), 0.0)
    assert np.allclose(a, b, atol=1e-14)


def test_fractions_and_physicality():
    stokes = pol.stokes_from_ellipse(3.0, 0.6, 0.3, -0.4)
    i, q, u, v = stokes
    assert pol.frac_lin(i, q, u) == pytest.approx(0.6)
    assert pol.frac_circ(i, v) == pytest.approx(-0.4)
    assert pol.frac_total(i, q, u, v) == pytest.approx(np.hypot(0.6, 0.4))
    assert pol.is_physical(i, q, u, v)
    assert not pol.is_physical(1.0, 0.9, 0.9, 0.0)


def test_stokes_from_ellipse_rejects_impossible_light():
    with pytest.raises(ValueError, match="unphysical"):
        pol.stokes_from_ellipse(1.0, 0.9, 0.0, 0.9)


def test_fully_polarized_states_saturate_the_inequality():
    stokes = pol.stokes_from_field(pol.jones_vector(0.7, 0.3, 1.1))
    i, q, u, v = stokes
    assert i * i == pytest.approx(q * q + u * u + v * v, rel=1e-12)


# ---------------------------------------------------------------------------
# Ellipse geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("p_lin", "p_circ"), [(1.0, 0.0), (0.0, 1.0), (0.6, 0.8),
                                               (0.5, 0.0), (0.3, -0.2)])
def test_ellipse_axes_reproduce_the_polarized_intensity(p_lin, p_circ):
    stokes = pol.stokes_from_ellipse(1.0, p_lin, 0.4, p_circ)
    geo = pol.ellipse_from_stokes(*stokes)
    a, b = geo["semi_major"], geo["semi_minor"]
    p_pol = np.hypot(p_lin, p_circ)
    assert a * a + b * b == pytest.approx(p_pol, abs=1e-12)      # total polarized power
    assert 2.0 * a * b == pytest.approx(abs(p_circ), abs=1e-12)  # circular part
    assert a >= b


def test_circular_light_has_a_circular_ellipse():
    geo = pol.ellipse_from_stokes(*pol.stokes_from_ellipse(1.0, 0.0, 0.0, 1.0))
    assert geo["axis_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert geo["handedness"] == 1


def test_linear_light_has_a_degenerate_ellipse():
    geo = pol.ellipse_from_stokes(*pol.stokes_from_ellipse(1.0, 1.0, 0.0, 0.0))
    assert geo["axis_ratio"] == pytest.approx(0.0, abs=1e-12)
    assert geo["handedness"] == 0


@pytest.mark.parametrize("delta_deg", [-100.0, -45.0, 30.0, 120.0])
def test_ellipse_trace_matches_the_physical_field(delta_deg):
    """The drawn ellipse and the actual field trace must be the same curve."""
    field = pol.jones_vector(0.8, 0.5, np.deg2rad(delta_deg))
    stokes = pol.stokes_from_field(field)
    _, e_x, e_y = pol.wave_trace(field, n_samples=721)
    x, y = pol.ellipse_trace(*stokes, n_samples=721)
    geo = pol.ellipse_from_stokes(*stokes)

    for xs, ys in ((e_x, e_y), (x, y)):
        radius = np.hypot(xs, ys)
        assert radius.max() == pytest.approx(geo["semi_major"], rel=2e-3)
        assert radius.min() == pytest.approx(geo["semi_minor"], rel=2e-3, abs=2e-3)


@pytest.mark.parametrize("delta_deg", [-100.0, -45.0, 30.0, 120.0])
def test_ellipse_trace_turns_the_same_way_as_the_field(delta_deg):
    """Handedness is a *direction of travel*, so check the signed area, not the shape."""
    field = pol.jones_vector(0.8, 0.5, np.deg2rad(delta_deg))
    stokes = pol.stokes_from_field(field)
    _, e_x, e_y = pol.wave_trace(field, n_samples=721)
    x, y = pol.ellipse_trace(*stokes, n_samples=721)

    def signed_area(xs, ys):
        return np.sum(xs[:-1] * ys[1:] - xs[1:] * ys[:-1])

    assert np.sign(signed_area(e_x, e_y)) == np.sign(signed_area(x, y))
    # ...and positive V means counter-clockwise in the plane as drawn.
    assert np.sign(signed_area(e_x, e_y)) == np.sign(stokes[3])


def test_poincare_point_is_on_the_surface_for_pure_states():
    stokes = pol.stokes_from_field(pol.jones_vector(0.6, 0.9, 0.7))
    assert np.linalg.norm(pol.poincare_point(*stokes)) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Partial polarization and Faraday rotation
# ---------------------------------------------------------------------------

def test_coherency_matrix_is_hermitian_with_the_right_trace():
    field = pol.jones_vector(0.7, 0.4, 0.9)
    coh = pol.coherency_matrix(field)
    assert np.allclose(coh, coh.conj().T, atol=1e-15)
    assert np.trace(coh).real == pytest.approx(pol.stokes_from_field(field)[0])


def test_aligned_emitters_stay_fully_polarized():
    stokes = pol.incoherent_sum(n_waves=500, spread_rad=0.0, evpa0_rad=0.3)
    assert stokes[0] == pytest.approx(1.0, abs=1e-12)
    assert pol.frac_lin(*stokes[:3]) == pytest.approx(1.0, abs=1e-12)
    assert np.rad2deg(pol.evpa(stokes[1], stokes[2])) == pytest.approx(np.rad2deg(0.3))


def test_scrambled_emitters_depolarize():
    tight = pol.incoherent_sum(n_waves=2000, spread_rad=np.deg2rad(10), seed=3)
    loose = pol.incoherent_sum(n_waves=2000, spread_rad=np.deg2rad(60), seed=3)
    assert pol.frac_lin(*tight[:3]) > 0.85
    assert pol.frac_lin(*loose[:3]) < 0.4
    # Intensity is conserved no matter how disordered the emitters are.
    assert tight[0] == pytest.approx(1.0) and loose[0] == pytest.approx(1.0)


def test_incoherent_sum_stays_physical():
    for spread in np.linspace(0, np.pi / 2, 10):
        stokes = pol.incoherent_sum(300, spread, p_circ=0.3, seed=1)
        assert pol.is_physical(*stokes)


def test_incoherent_sum_is_reproducible():
    assert np.allclose(pol.incoherent_sum(100, 0.5, seed=7),
                       pol.incoherent_sum(100, 0.5, seed=7))


def test_faraday_rotation_is_linear_in_wavelength_squared():
    lam = np.array([0.0013, 0.0035])
    rotated = pol.faraday_evpa(0.2, 1.0e5, lam)
    assert np.allclose(rotated, 0.2 + 1.0e5 * lam**2)
    # Sign of RM sets the direction of rotation.
    assert pol.faraday_evpa(0.0, -1.0e5, 0.0013) < 0.0


# ---------------------------------------------------------------------------
# Array inputs
# ---------------------------------------------------------------------------

def test_the_fraction_helpers_work_on_a_whole_image():
    """A reader will pass a Stokes image to these; it must not raise."""
    import images as im

    ring = im.polarized_ring(npix=32, p_lin=0.25, p_circ=0.05)
    args = (ring["I"], ring["Q"], ring["U"], ring["V"])

    assert pol.frac_lin(*args[:3]).shape == ring["I"].shape
    assert pol.frac_total(*args).shape == ring["I"].shape
    assert pol.is_physical(*args) is True
    # Pixels where I == 0 give 0, not nan -- an image has plenty of them.
    assert not np.isnan(pol.frac_lin(*args[:3])).any()
    assert pol.frac_lin(np.zeros(3), np.zeros(3), np.zeros(3)).tolist() == [0, 0, 0]


def test_ellipse_from_stokes_is_elementwise():
    import images as im

    ring = im.polarized_ring(npix=24, p_lin=0.3, pitch_deg=45.0)
    geo = pol.ellipse_from_stokes(ring["I"], ring["Q"], ring["U"], ring["V"])
    for key in ("semi_major", "semi_minor", "evpa", "axis_ratio", "handedness",
                "p_total", "p_lin", "p_circ"):
        assert np.shape(geo[key]) == ring["I"].shape, key
    bright = ring["I"] > 0.3 * ring["I"].max()
    assert np.allclose(geo["p_lin"][bright], 0.3)
    # Elementwise must agree with the scalar path pixel by pixel.
    index = np.unravel_index(np.argmax(ring["I"]), ring["I"].shape)
    single = pol.ellipse_from_stokes(*(ring[k][index] for k in "IQUV"))
    assert single["p_lin"] == pytest.approx(geo["p_lin"][index])
    assert single["evpa"] == pytest.approx(geo["evpa"][index])


def test_scalar_inputs_still_return_plain_floats():
    """Because figure titles format these with f-strings, which 0-d arrays reject."""
    geo = pol.ellipse_from_stokes(1.0, 0.5, 0.2, 0.1)
    for key, value in geo.items():
        assert isinstance(value, (float, int)), (key, type(value))
    assert f"{pol.frac_lin(1.0, 0.6, 0.0):.0%}" == "60%"


def test_stokes_from_ellipse_accepts_an_intensity_image():
    import images as im

    blob = im.gaussian_blob(npix=16)
    stokes = pol.stokes_from_ellipse(blob["I"], 0.4, 0.3, 0.1)
    assert stokes.shape == (4, 16, 16)
    assert np.allclose(stokes[0], blob["I"])


def test_single_state_functions_explain_themselves_on_an_image():
    """The numpy message ('truth value of an array is ambiguous') teaches nothing."""
    import images as im

    ring = im.polarized_ring(npix=16, p_lin=0.2)
    with pytest.raises(ValueError, match="describes a single polarization state"):
        pol.ellipse_trace(ring["I"], ring["Q"], ring["U"], ring["V"])
    with pytest.raises(ValueError, match="image\\['Q'\\]"):
        pol.require_single_state("demo", ring["I"], 0.0, 0.0, 0.0)
