"""Checks for core.interferometry, mostly against closed-form answers.

The Gaussian is the workhorse: its Fourier transform is known exactly, so the
discrete sampling, the units of ``(u, v)``, and the sign of the phase can all be
pinned rather than eyeballed.
"""

import arrays as arr
import images as im
import interferometry as itf
import numpy as np
import pytest

EARTH_DIAMETER_M = 1.27e7


def test_the_eht_needs_an_earth_sized_aperture():
    """The number that motivates the entire array."""
    assert itf.resolution_uas(1.3e-3, 100.0) > 1e6                 # a 100 m dish: useless
    assert itf.resolution_uas(1.3e-3, EARTH_DIAMETER_M) == pytest.approx(21.0, abs=2.0)
    assert itf.required_aperture_m(1.3e-3, 42.0) == pytest.approx(6.4e6, rel=0.05)


def test_fringe_spacing_is_lambda_over_baseline():
    baseline, wavelength = 5000e3, 1.3e-3
    expected_period_uas = wavelength / baseline / itf.UAS
    offsets = np.linspace(0, 3 * expected_period_uas, 4001)
    response = itf.fringe_response(offsets, baseline, wavelength)

    peaks = offsets[1:-1][(response[1:-1] > response[:-2]) & (response[1:-1] > response[2:])]
    assert np.diff(peaks) == pytest.approx(expected_period_uas, rel=1e-3)
    assert itf.fringe_response(0.0, baseline, wavelength) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Visibilities
# ---------------------------------------------------------------------------

def test_zero_baseline_measures_the_total_flux():
    image = im.gaussian_blob(npix=64, fov_uas=300.0, fwhm_uas=40.0, total_flux=1.7)
    assert itf.sample_visibilities(image, np.array([0.0]), np.array([0.0]))[0].real == (
        pytest.approx(1.7, rel=1e-12))


def test_discrete_transform_matches_the_analytic_gaussian():
    fwhm, flux = 40.0, 1.0
    image = im.gaussian_blob(npix=256, fov_uas=400.0, fwhm_uas=fwhm, total_flux=flux)
    u = np.linspace(0, 4e9, 25)                      # out to ~4 Glambda
    v = np.zeros_like(u)

    numeric = itf.sample_visibilities(image, u, v)
    analytic = itf.gaussian_visibility(u, v, fwhm, flux)
    assert np.allclose(numeric.real, analytic, atol=2e-3)
    assert np.allclose(numeric.imag, 0.0, atol=1e-9)  # a centred source has zero phase


def test_broader_sources_have_narrower_visibility_functions():
    """Long baselines see compact structure; that is the whole game."""
    u = np.array([3e9])
    v = np.zeros_like(u)
    compact = abs(itf.gaussian_visibility(u, v, 10.0)[0])
    extended = abs(itf.gaussian_visibility(u, v, 60.0)[0])
    assert compact > 0.5 > extended


def test_shifting_a_source_only_changes_the_phase():
    """A shift is a phase ramp -- the reason absolute position is not measurable."""
    npix, fov = 128, 400.0
    centred = im.gaussian_blob(npix=npix, fov_uas=fov, fwhm_uas=30.0)
    shifted = dict(centred)
    shifted["I"] = np.roll(centred["I"], 8, axis=1)   # move by 8 pixels in l

    u = np.linspace(1e8, 2e9, 12)
    v = np.zeros_like(u)
    v_centred = itf.sample_visibilities(centred, u, v)
    v_shifted = itf.sample_visibilities(shifted, u, v)

    assert np.allclose(np.abs(v_centred), np.abs(v_shifted), rtol=1e-9)
    shift_rad = 8 * (fov / npix) * itf.UAS
    assert np.allclose(np.angle(v_shifted / v_centred), -2 * np.pi * u * shift_rad,
                       atol=1e-6)


def test_sample_visibilities_needs_a_field_of_view_for_bare_arrays():
    with pytest.raises(ValueError, match="fov_uas"):
        itf.sample_visibilities(np.zeros((8, 8)), np.array([0.0]), np.array([0.0]))


# ---------------------------------------------------------------------------
# Dirty beam and dirty image
# ---------------------------------------------------------------------------

def _eht_coverage(n_times=60):
    array = arr.array_2017()
    hours = np.linspace(0, 24, n_times)
    return arr.uv_coverage(array, 57854.0, hours, *arr.SOURCES["M87"])


def test_dirty_beam_peaks_at_the_centre_and_has_sidelobes():
    coverage = _eht_coverage()
    beam = itf.dirty_beam(coverage["u"], coverage["v"], npix=64, fov_uas=200.0)
    peak = np.unravel_index(np.argmax(beam["I"]), beam["I"].shape)
    assert peak == pytest.approx((32, 32), abs=1)   # even npix: centre falls between pixels
    assert beam["I"].max() == pytest.approx(1.0)
    assert beam["I"].min() < -0.05          # sparse sampling always rings


def test_dirty_image_is_the_truth_convolved_with_the_beam():
    """Not equal to the truth -- that difference *is* the imaging problem."""
    coverage = _eht_coverage()
    truth = im.polarized_ring(npix=64, fov_uas=200.0, diameter_uas=42.0, total_flux=1.0)
    vis = itf.sample_visibilities(truth, coverage["u"], coverage["v"])
    dirty = itf.dirty_image(vis, coverage["u"], coverage["v"], npix=64, fov_uas=200.0)

    assert dirty["I"].shape == truth["I"].shape
    # The ring survives: the brightest dirty pixel is on the ring, not in the hole.
    peak = np.unravel_index(np.argmax(dirty["I"]), dirty["I"].shape)
    axis = np.linspace(-100, 100, 64)
    assert 5.0 < np.hypot(axis[peak[1]], axis[peak[0]]) < 40.0

    # ...but it is emphatically not the truth: the beam smears flux everywhere,
    # and the central brightness depression is largely filled in.
    scaled = dirty["I"] * truth["I"].max() / dirty["I"].max()
    assert np.abs(scaled - truth["I"]).max() > 0.3 * truth["I"].max()
    assert scaled[32, 32] > 3 * truth["I"][32, 32]


def test_dirty_image_of_a_point_source_is_the_dirty_beam():
    coverage = _eht_coverage(n_times=20)
    unit_vis = np.ones_like(coverage["u"], dtype=complex)
    beam = itf.dirty_beam(coverage["u"], coverage["v"], npix=48, fov_uas=200.0)
    image = itf.dirty_image(unit_vis, coverage["u"], coverage["v"], npix=48, fov_uas=200.0)
    assert np.allclose(image["I"] / image["I"].max(), beam["I"], atol=1e-12)


# ---------------------------------------------------------------------------
# Nyquist
# ---------------------------------------------------------------------------

def test_nyquist_relations_bracket_the_eht_numbers():
    coverage = _eht_coverage()
    u_max = np.hypot(coverage["u"], coverage["v"]).max()
    assert itf.beam_size_uas(coverage["u"], coverage["v"]) == pytest.approx(
        1.0 / u_max / itf.UAS)
    # ~25 uas beam, so pixels of a few uas and a field of view of hundreds.
    assert 15.0 < itf.beam_size_uas(coverage["u"], coverage["v"]) < 35.0   # ~25 uas
    assert itf.nyquist_pixel_uas(u_max) < itf.beam_size_uas(coverage["u"], coverage["v"])
    assert itf.field_of_view_uas(1e7) > 20.0


# ---------------------------------------------------------------------------
# Gains and closure quantities
# ---------------------------------------------------------------------------

def test_closure_phase_is_immune_to_station_gains():
    """The property that made VLBI possible before phase calibration was solvable."""
    rng = np.random.default_rng(0)
    true = rng.normal(size=3) + 1j * rng.normal(size=3)
    gains = np.exp(1j * rng.uniform(0, 2 * np.pi, 3)) * rng.uniform(0.5, 2.0, 3)

    clean = itf.closure_phase(*true)
    corrupted = itf.closure_phase(
        itf.apply_station_gains(true[0], gains[0], gains[1]),
        itf.apply_station_gains(true[1], gains[1], gains[2]),
        itf.apply_station_gains(true[2], gains[2], gains[0]))
    assert corrupted == pytest.approx(clean, abs=1e-12)


def test_a_symmetric_source_has_zero_closure_phase():
    """So a non-zero closure phase is direct evidence of asymmetry -- as in M87."""
    coverage = _eht_coverage(n_times=10)
    symmetric = im.gaussian_blob(npix=64, fov_uas=200.0, fwhm_uas=40.0)
    u, v = coverage["u"][:3], coverage["v"][:3]
    # Close the triangle: u3 = -(u1 + u2).
    u = np.array([u[0], u[1], -(u[0] + u[1])])
    v = np.array([v[0], v[1], -(v[0] + v[1])])
    vis = itf.sample_visibilities(symmetric, u, v)
    assert itf.closure_phase(*vis) == pytest.approx(0.0, abs=1e-8)


def test_log_closure_amplitude_is_immune_to_gain_amplitudes():
    rng = np.random.default_rng(1)
    v12, v34, v13, v24 = rng.normal(size=4) + 1j * rng.normal(size=4)
    g = rng.uniform(0.3, 3.0, 4)
    clean = itf.log_closure_amplitude(v12, v34, v13, v24)
    corrupted = itf.log_closure_amplitude(g[0] * g[1] * v12, g[2] * g[3] * v34,
                                          g[0] * g[2] * v13, g[1] * g[3] * v24)
    assert corrupted == pytest.approx(clean, abs=1e-12)


def test_thermal_noise_has_the_requested_scatter_and_is_reproducible():
    vis = np.zeros(20000, dtype=complex)
    noisy = itf.add_thermal_noise(vis, 0.05, seed=2)
    assert np.std(noisy.real) == pytest.approx(0.05, rel=0.05)
    assert np.std(noisy.imag) == pytest.approx(0.05, rel=0.05)
    assert np.allclose(noisy, itf.add_thermal_noise(vis, 0.05, seed=2))
