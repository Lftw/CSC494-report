"""Checks for the committed capture of eht-imaging's polarization conventions.

Two jobs, and they need different environments. Everywhere: the file in
``scripts/data/`` agrees with ``jones.py``, which is what notebook 03 prints.
Where eht-imaging is installed: the file still agrees with the live module, so a
capture cannot go stale while the notebook keeps reporting agreement.

A comparison that always agrees is worth nothing, so the perturbation tests below
check that these particular matrices do notice a wrong convention.
"""

import ehtim_reference as ref
import jones
import numpy as np
import pytest

PAIRING_IDS = [f"{f1}x{f2}" for f1, f2 in ref.PAIRINGS]


@pytest.fixture(scope="module")
def reference():
    return ref.load()


# ---------------------------------------------------------------------------
# The file itself
# ---------------------------------------------------------------------------

def test_capture_records_where_it_came_from(reference):
    assert reference["ehtim_version"]
    assert reference["ehtim_module"].endswith("pol_conventions.py")
    assert len(reference["captured"]) == len("YYYY-MM-DD")


@pytest.mark.parametrize("feed_type", jones.FEED_TYPES)
def test_captured_feed_matrices_are_unitary(reference, feed_type):
    """Not a round-trip against ourselves: a decoding bug would break this."""
    matrix = reference["feed_matrix"][feed_type]
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix @ matrix.conj().T, np.eye(2), atol=1e-15)


@pytest.mark.parametrize(("feed1", "feed2"), ref.PAIRINGS, ids=PAIRING_IDS)
def test_the_two_captured_directions_invert_each_other(reference, feed1, feed2):
    key = jones.polbasis(feed1, feed2)
    forward = reference["stokes_to_slots"][key]
    backward = reference["slots_to_stokes"][key]
    assert forward.shape == backward.shape == (4, 4)
    assert np.allclose(backward @ forward, np.eye(4), atol=1e-12)


# ---------------------------------------------------------------------------
# jones.py against the capture -- the comparison notebook 03 prints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("label", "error"), ref.comparisons(),
                         ids=lambda value: value if isinstance(value, str) else "")
def test_jones_agrees_with_the_captured_conventions(label, error):
    assert error < ref.TOLERANCE, f"{label} differs by {error:.1e}"


def test_all_eleven_quantities_are_compared():
    labels = [label for label, _ in ref.comparisons()]
    assert len(labels) == len(set(labels)) == 11


# ---------------------------------------------------------------------------
# The comparison notices a wrong convention
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("feed1", "feed2"), ref.PAIRINGS, ids=PAIRING_IDS)
def test_a_flipped_v_sign_would_be_caught(reference, feed1, feed2):
    """The classic convention bug: V with the other sign, everything else right."""
    key = jones.polbasis(feed1, feed2)
    flipped = reference["stokes_to_slots"][key].copy()
    flipped[:, 3] *= -1
    spoiled = {**reference,
               "stokes_to_slots": {**reference["stokes_to_slots"], key: flipped}}

    caught = [label for label, error in ref.comparisons(spoiled)
              if error >= ref.TOLERANCE]
    assert caught == [f"Stokes → slots, {feed1.upper()}×{feed2.upper()}"]


def test_swapped_cross_hand_slots_would_be_caught(reference):
    """p1p2 and p2p1 exchanged -- a plausible slot-order mistake."""
    key = jones.polbasis("xy", "rl")
    swapped = reference["stokes_to_slots"][key][[0, 1, 3, 2], :]
    spoiled = {**reference,
               "stokes_to_slots": {**reference["stokes_to_slots"], key: swapped}}
    assert any(error >= ref.TOLERANCE for _, error in ref.comparisons(spoiled))


# ---------------------------------------------------------------------------
# Against a live eht-imaging, where there is one
# ---------------------------------------------------------------------------

def _live_conventions():
    return pytest.importorskip("ehtim.observing.pol_conventions",
                               reason="eht-imaging is not installed in this env")


@pytest.mark.parametrize("section", ["feed_matrix", "stokes_to_slots",
                                     "slots_to_stokes"])
def test_capture_is_not_stale(reference, section):
    fresh = ref.capture(_live_conventions())
    for key, blob in fresh[section].items():
        assert np.allclose(ref._decode(blob), reference[section][key],
                           atol=ref.TOLERANCE), f"{section}[{key}] has drifted"


def test_captured_jones_matrix_is_not_stale(reference):
    fresh = ref.capture(_live_conventions())
    assert np.allclose(ref._decode(fresh["jones_matrix"]),
                       reference["jones_matrix"], atol=ref.TOLERANCE)


def test_check_live_reports_the_match(reference):
    _live_conventions()
    message = ref.check_live(reference)
    assert "still matches" in message and "not installed" not in message
