"""Checks for jones.py: feed bases, the RIME, and the mixed-basis failure mode.

The conventions are the whole point, so most of these pin a formula against a
textbook expression written out by hand, and the last group compares them against
eht-imaging itself where it is installed.
"""

import jones
import numpy as np
import polarization as pol
import pytest

STOKES = np.array([1.0, 0.3, -0.2, 0.05])


# ---------------------------------------------------------------------------
# Feeds
# ---------------------------------------------------------------------------

def test_linear_feeds_measure_the_components_directly():
    assert np.allclose(jones.feed_matrix("xy"), np.eye(2))


def test_circular_feeds_are_the_iau_basis_change():
    assert np.allclose(jones.feed_matrix("rl"), pol.BASIS_LIN_TO_CIRC)


@pytest.mark.parametrize("feed_type", ["rl", "xy"])
def test_feed_matrices_are_unitary(feed_type):
    matrix = jones.feed_matrix(feed_type)
    assert np.allclose(matrix @ matrix.conj().T, np.eye(2), atol=1e-15)


@pytest.mark.parametrize("bad", ["r", "rlx", "rr", "rz"])
def test_feed_matrix_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        jones.feed_matrix(bad)


def test_slot_labels_and_polbasis():
    assert jones.slot_labels("rl", "rl") == ("RR", "LL", "RL", "LR")
    assert jones.slot_labels("xy", "xy") == ("XX", "YY", "XY", "YX")
    assert jones.slot_labels("xy", "rl") == ("XR", "YL", "XL", "YR")
    assert jones.polbasis("xy", "rl") == "xyrl"
    assert jones.is_mixed("xy", "rl") and not jones.is_mixed("rl", "rl")


# ---------------------------------------------------------------------------
# Coherency and slots
# ---------------------------------------------------------------------------

def test_coherency_round_trips_through_stokes():
    coherency = jones.coherency_from_stokes(*STOKES)
    assert np.allclose(coherency, coherency.conj().T)          # Hermitian
    assert np.allclose(jones.stokes_from_coherency(coherency), STOKES)


def test_circular_slots_match_the_textbook_expressions():
    """RR = I+V, LL = I-V, RL = Q+iU, LR = Q-iU (conventions doc, section 4)."""
    i, q, u, v = STOKES
    rr, ll, rl, lr = jones.slots_from_stokes(STOKES, "rl", "rl")
    assert rr == pytest.approx(i + v)
    assert ll == pytest.approx(i - v)
    assert rl == pytest.approx(q + 1j * u)
    assert lr == pytest.approx(q - 1j * u)


def test_linear_slots_match_the_textbook_expressions():
    """XX = I+Q, YY = I-Q, XY = U+iV, YX = U-iV (conventions doc, section 5)."""
    i, q, u, v = STOKES
    xx, yy, xy, yx = jones.slots_from_stokes(STOKES, "xy", "xy")
    assert xx == pytest.approx(i + q)
    assert yy == pytest.approx(i - q)
    assert xy == pytest.approx(u + 1j * v)
    assert yx == pytest.approx(u - 1j * v)


@pytest.mark.parametrize(("feed1", "feed2"), [("rl", "rl"), ("xy", "xy"),
                                              ("xy", "rl"), ("rl", "xy")])
def test_slots_and_stokes_round_trip_in_every_basis(feed1, feed2):
    slots = jones.slots_from_stokes(STOKES, feed1, feed2)
    assert np.allclose(jones.stokes_from_slots(slots, feed1, feed2), STOKES, atol=1e-12)


def test_a_mixed_baseline_needs_a_different_conversion():
    """The point of the whole notebook: the 4x4 operator is per baseline."""
    same = jones.slots_to_stokes_matrix("rl", "rl")
    mixed = jones.slots_to_stokes_matrix("xy", "rl")
    assert not np.allclose(same, mixed)


def test_unpolarized_light_is_unpolarized_in_every_basis():
    for feed1, feed2 in (("rl", "rl"), ("xy", "xy"), ("xy", "rl")):
        slots = jones.slots_from_stokes([1.0, 0.0, 0.0, 0.0], feed1, feed2)
        assert np.allclose(jones.stokes_from_slots(slots, feed1, feed2),
                           [1.0, 0.0, 0.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# Field rotation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle_deg", [-70.0, -20.0, 0.0, 35.0, 90.0])
def test_field_rotation_is_unitary_and_additive(angle_deg):
    angle = np.deg2rad(angle_deg)
    for feed_type in ("rl", "xy"):
        phi = jones.field_rotation_matrix(feed_type, angle)
        assert np.allclose(phi @ phi.conj().T, np.eye(2), atol=1e-14)
        assert np.allclose(jones.field_rotation_matrix(feed_type, 0.0), np.eye(2))
        assert np.allclose(
            jones.field_rotation_matrix(feed_type, angle)
            @ jones.field_rotation_matrix(feed_type, angle),
            jones.field_rotation_matrix(feed_type, 2 * angle), atol=1e-14)


def test_field_rotation_is_a_phase_for_circular_feeds():
    phi = jones.field_rotation_matrix("rl", np.deg2rad(30.0))
    assert abs(phi[0, 1]) < 1e-14 and abs(phi[1, 0]) < 1e-14      # diagonal
    assert np.allclose(np.abs(np.diag(phi)), 1.0)                  # pure phase
    assert phi[0, 0] == pytest.approx(np.conj(phi[1, 1]))          # opposite signs


def test_field_rotation_is_a_real_rotation_for_linear_feeds():
    phi = jones.field_rotation_matrix("xy", np.deg2rad(30.0))
    assert np.allclose(phi.imag, 0.0, atol=1e-14)
    assert abs(phi[0, 1]) > 0.1                                    # feeds get mixed


def test_the_two_forms_are_the_same_physical_rotation():
    """A circular-basis Phi is a linear-basis Phi seen from the other basis."""
    angle = np.deg2rad(37.0)
    basis = jones.feed_matrix("rl")
    assert np.allclose(basis @ jones.field_rotation_matrix("xy", angle)
                       @ np.linalg.inv(basis),
                       jones.field_rotation_matrix("rl", angle), atol=1e-14)


def test_rotation_leaves_circular_amplitudes_alone_but_not_linear_ones():
    """The claim the notebook makes, checked both ways."""
    angle = np.deg2rad(30.0)
    for feed_type, parallel_hands_move in (("rl", False), ("xy", True)):
        phi = jones.field_rotation_matrix(feed_type, angle)
        quiet = jones.observe(STOKES, feed_type, feed_type)
        rotated = jones.observe(STOKES, feed_type, feed_type, phi, phi)
        moved = not np.allclose(np.abs(quiet[:2]), np.abs(rotated[:2]), atol=1e-9)
        assert moved is parallel_hands_move, feed_type

@pytest.mark.parametrize("angle_deg", [-40.0, 15.0, 30.0])
def test_rotation_turns_the_measured_evpa_by_the_same_angle(angle_deg):
    """The (Q, U) vector rotates by 2 phi, so the EVPA -- half its argument -- by phi.

    Both statements appear in the literature and they are the same statement.
    Checked in the circular basis, where the cross-hand phase is easy to read.
    """
    angle = np.deg2rad(angle_deg)
    for feed_type in ("rl", "xy"):
        phi = jones.field_rotation_matrix(feed_type, angle)
        rotated = jones.stokes_from_slots(
            jones.observe(STOKES, feed_type, feed_type, phi, phi), feed_type, feed_type)
        turned = pol.evpa(rotated[1], rotated[2]) - pol.evpa(STOKES[1], STOKES[2])
        wrapped = (turned - angle + np.pi / 2) % np.pi - np.pi / 2
        assert abs(wrapped) < 1e-9, feed_type
        assert rotated[0] == pytest.approx(STOKES[0])      # intensity is untouched

    # ...and the cross-hand product itself turns by twice that.
    quiet = jones.observe(STOKES, "rl", "rl")
    phi = jones.field_rotation_matrix("rl", angle)
    rotated_slots = jones.observe(STOKES, "rl", "rl", phi, phi)
    phase_turn = np.angle(rotated_slots[2] / quiet[2])
    assert abs((phase_turn - 2 * angle + np.pi) % (2 * np.pi) - np.pi) < 1e-9


# ---------------------------------------------------------------------------
# The Jones chain and the RIME
# ---------------------------------------------------------------------------

def test_jones_is_the_product_of_its_three_factors():
    gain, leak, angle = 1.1 + 0.2j, 0.03 - 0.01j, np.deg2rad(20.0)
    expected = (jones.gain_matrix(gain, gain)
                @ jones.dterm_matrix(leak, -leak)
                @ jones.field_rotation_matrix("rl", angle))
    assert np.allclose(jones.jones_matrix(gain, gain, leak, -leak, "rl", angle),
                       expected)


def test_the_factors_do_not_commute():
    """Order is physics: the mount rotates the sky before the optics leak."""
    gain, leak = 1.2, 0.08
    forward = jones.gain_matrix(gain, 0.9) @ jones.dterm_matrix(leak, -leak)
    swapped = jones.dterm_matrix(leak, -leak) @ jones.gain_matrix(gain, 0.9)
    assert not np.allclose(forward, swapped)


def test_a_perfect_instrument_changes_nothing():
    slots = jones.observe(STOKES, "rl", "rl")
    assert np.allclose(jones.stokes_from_slots(slots, "rl", "rl"), STOKES, atol=1e-12)


@pytest.mark.parametrize(("feed1", "feed2"), [("rl", "rl"), ("xy", "rl")])
def test_known_jones_matrices_can_be_undone_exactly(feed1, feed2):
    """Calibration is exact when the corruption is known. That is the easy part."""
    jones1 = jones.jones_matrix(1.3 + 0.1j, 0.8, 0.05, -0.02, feed1, np.deg2rad(15))
    jones2 = jones.jones_matrix(0.9, 1.1 - 0.3j, -0.03, 0.04, feed2, np.deg2rad(-40))
    clean = jones.baseline_coherency(STOKES, feed1, feed2)
    observed = jones.apply_jones(jones1, clean, jones2)
    assert np.allclose(jones.recover_coherency(observed, jones1, jones2), clean)


def test_leakage_invents_polarization_out_of_an_unpolarized_source():
    """1% leakage at each end manufactures about 2% polarization, at any phase."""
    for leak in (0.01, 0.03, 0.05):
        for phase in (0.0, 0.7, -2.1):
            d_p1 = leak * np.exp(1j * phase)
            made_up = jones.spurious_polarization(d_p1, np.conj(d_p1))
            assert made_up == pytest.approx(2 * leak, rel=0.05)
    assert jones.spurious_polarization(0.0, 0.0) == pytest.approx(0.0, abs=1e-12)


def test_antisymmetric_leakage_cancels_instead_of_adding():
    """Not every D-term pairing leaks -- ``d_p2 = -d_p1`` cancels exactly.

    Worth pinning: it is the difference between a leakage demo that shows the
    effect and one that silently shows nothing.
    """
    assert jones.spurious_polarization(0.05, -0.05) == pytest.approx(0.0, abs=1e-12)
    assert jones.spurious_polarization(0.05, 0.05) == pytest.approx(0.1, rel=0.05)


# ---------------------------------------------------------------------------
# Reading a baseline in the wrong basis
# ---------------------------------------------------------------------------

def test_assuming_the_right_basis_changes_nothing():
    assert np.allclose(jones.misread_stokes(STOKES, ("rl", "rl"), ("rl", "rl")),
                       STOKES, atol=1e-12)


def test_linear_data_read_as_circular_turns_q_into_v():
    """The failure that does not look like a bug: linear polarization becomes circular."""
    truth = np.array([1.0, 0.4, 0.0, 0.0])          # pure Stokes Q, no circular
    wrong = jones.misread_stokes(truth, ("xy", "xy"), ("rl", "rl"))
    assert wrong[0] == pytest.approx(truth[0])       # total intensity survives
    assert abs(wrong[3]) == pytest.approx(0.4, abs=1e-9)   # ...as circular polarization
    assert abs(wrong[1]) < 1e-9


def test_a_mixed_baseline_read_as_homogeneous_is_wrong():
    wrong = jones.misread_stokes(STOKES, ("xy", "rl"), ("rl", "rl"))
    assert not np.allclose(wrong, STOKES, atol=1e-6)


# ---------------------------------------------------------------------------
# Against eht-imaging itself
# ---------------------------------------------------------------------------

def _ehtim_conventions():
    return pytest.importorskip("ehtim.observing.pol_conventions",
                               reason="eht-imaging is not installed in this env")


def test_feed_matrices_agree_with_ehtim():
    conventions = _ehtim_conventions()
    for feed_type in ("rl", "xy"):
        assert np.allclose(jones.feed_matrix(feed_type),
                           conventions.feed_matrix(feed_type))


@pytest.mark.parametrize(("feed1", "feed2"), [("rl", "rl"), ("xy", "xy"),
                                              ("xy", "rl"), ("rl", "xy")])
def test_stokes_to_slots_agrees_with_ehtim(feed1, feed2):
    conventions = _ehtim_conventions()
    mine = jones.slots_from_stokes(STOKES, feed1, feed2)
    theirs = conventions.stokes_to_coherency(*STOKES, feed1, feed2)
    assert np.allclose(mine, theirs, atol=1e-12)


@pytest.mark.parametrize(("feed1", "feed2"), [("rl", "rl"), ("xy", "xy"),
                                              ("xy", "rl"), ("rl", "xy")])
def test_slots_to_stokes_agrees_with_ehtim(feed1, feed2):
    conventions = _ehtim_conventions()
    slots = jones.slots_from_stokes(STOKES, feed1, feed2)
    theirs = conventions.coherency_to_stokes(*slots, feed1, feed2)
    assert np.allclose(jones.stokes_from_slots(slots, feed1, feed2),
                       np.real(theirs), atol=1e-12)


def test_gain_and_leakage_agree_with_ehtim():
    conventions = _ehtim_conventions()
    gain_p1, gain_p2, d_p1, d_p2 = 1.2 + 0.1j, 0.85, 0.04 - 0.01j, -0.02
    mine = jones.gain_matrix(gain_p1, gain_p2) @ jones.dterm_matrix(d_p1, d_p2)
    assert np.allclose(mine, conventions.jones_matrix(gain_p1, gain_p2, d_p1, d_p2))


def test_calibrating_a_baseline_recovers_the_sky_exactly():
    """The whole loop: observe with known corruption, undo it, read Stokes back.

    Regression for an easy mistake this module now guards against: a baseline's
    coherency matrix is in its *feeds'* basis, so it must be read out with
    ``slots_of`` plus ``stokes_from_slots``. Handing it to ``stokes_from_coherency``
    -- which expects the linear basis -- shuffles I, Q, U, V instead of failing.
    """
    for feed1, feed2 in (("rl", "rl"), ("xy", "xy"), ("xy", "rl")):
        jones1 = jones.jones_matrix(1.3 + 0.1j, 0.8, 0.05, -0.02, feed1, np.deg2rad(15))
        jones2 = jones.jones_matrix(0.9, 1.1 - 0.3j, -0.03, 0.04, feed2, np.deg2rad(-40))
        observed = jones.apply_jones(
            jones1, jones.baseline_coherency(STOKES, feed1, feed2), jones2)
        repaired = jones.recover_coherency(observed, jones1, jones2)
        recovered = jones.stokes_from_slots(jones.slots_of(repaired), feed1, feed2)
        assert np.allclose(recovered, STOKES, atol=1e-12), (feed1, feed2)
