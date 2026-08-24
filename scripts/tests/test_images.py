"""Checks for core.images: fluxes, geometry, and that the ticks say what they claim."""

import images as im
import numpy as np
import polarization as pol
import pytest


def test_gaussian_conserves_total_flux():
    image = im.gaussian_blob(npix=128, fov_uas=200.0, fwhm_uas=30.0, total_flux=2.5)
    assert image["I"].sum() == pytest.approx(2.5, rel=1e-10)


def test_gaussian_is_centred_and_has_the_requested_width():
    npix, fov, fwhm = 256, 200.0, 40.0
    image = im.gaussian_blob(npix=npix, fov_uas=fov, fwhm_uas=fwhm)
    peak = np.unravel_index(np.argmax(image["I"]), image["I"].shape)
    assert peak == pytest.approx((npix // 2, npix // 2), abs=1)

    # Half-maximum crossing along a row through the centre.
    row = image["I"][npix // 2]
    axis = np.linspace(-fov / 2, fov / 2, npix)
    above = axis[row > 0.5 * row.max()]
    assert (above.max() - above.min()) == pytest.approx(fwhm, rel=0.02)


@pytest.mark.parametrize(("p_lin", "evpa_deg"), [(0.0, 0.0), (0.3, 0.0), (0.3, 40.0),
                                                (0.7, -65.0)])
def test_uniform_polarization_is_recovered_pixel_by_pixel(p_lin, evpa_deg):
    image = im.gaussian_blob(npix=64, p_lin=p_lin, evpa_deg=evpa_deg)
    bright = image["I"] > 0.3 * image["I"].max()
    frac = np.hypot(image["Q"], image["U"])[bright] / image["I"][bright]
    assert np.allclose(frac, p_lin, atol=1e-12)
    if p_lin > 0:
        angles = pol.evpa(image["Q"][bright], image["U"][bright])
        assert np.allclose(np.rad2deg(angles), evpa_deg, atol=1e-9)


def test_ring_peaks_at_the_requested_radius():
    diameter, fov, npix = 42.0, 100.0, 256
    image = im.polarized_ring(npix=npix, fov_uas=fov, diameter_uas=diameter,
                              width_uas=8.0, asymmetry=0.0)
    axis = np.linspace(-fov / 2, fov / 2, npix)
    row = image["I"][npix // 2, npix // 2:]
    assert axis[npix // 2:][np.argmax(row)] == pytest.approx(diameter / 2, abs=1.0)


def test_ring_asymmetry_brightens_the_requested_side():
    image = im.polarized_ring(asymmetry=0.6, asymmetry_pa_deg=0.0)  # bright toward +x
    left, right = np.hsplit(image["I"], 2)
    assert right.sum() > left.sum()


def test_ring_evpa_follows_the_pitch_angle():
    """A pitch of 0 gives a radial pattern; 90 gives an azimuthal one."""
    for pitch, expected_radial in ((0.0, True), (90.0, False)):
        image = im.polarized_ring(npix=128, pitch_deg=pitch, p_lin=0.3, asymmetry=0.0)
        ticks = im.evpa_ticks(image, step=8, i_cut=0.4)
        # Compare the tick direction with the radial direction at each tick.
        xc = 0.5 * (ticks["x0"] + ticks["x1"])
        yc = 0.5 * (ticks["y0"] + ticks["y1"])
        tick_angle = np.arctan2(ticks["y1"] - ticks["y0"], ticks["x1"] - ticks["x0"])
        misalign = np.abs(np.sin(tick_angle - np.arctan2(yc, xc)))
        assert bool(np.all(misalign < 1e-6)) is expected_radial


def test_ring_net_polarization_is_far_below_the_pixel_value():
    """The cancellation that motivates resolved polarimetry."""
    image = im.polarized_ring(p_lin=0.3, pitch_deg=45.0, asymmetry=0.0)
    totals = im.image_stokes_totals(image)
    assert totals["p_lin_mean"] == pytest.approx(0.3, rel=1e-6)
    assert totals["p_lin_net"] < 0.02


def test_ticks_are_suppressed_where_there_is_no_light():
    image = im.polarized_ring(npix=128, p_lin=0.3)
    dense = im.evpa_ticks(image, step=4, i_cut=0.0)
    sparse = im.evpa_ticks(image, step=4, i_cut=0.5)
    assert len(sparse["x0"]) < len(dense["x0"]) > 0


def test_ticks_stay_inside_the_field_of_view():
    image = im.polarized_ring(npix=128, fov_uas=100.0, p_lin=0.5)
    ticks = im.evpa_ticks(image, step=4, scale=3.0, i_cut=0.05)
    for key in ("x0", "x1", "y0", "y1"):
        assert np.abs(ticks[key]).max() <= 55.0  # a tick may overhang a pixel, not the frame


def test_unpolarized_image_draws_zero_length_ticks():
    image = im.polarized_ring(p_lin=0.0)
    ticks = im.evpa_ticks(image, step=6, i_cut=0.1)
    assert np.allclose(ticks["x0"], ticks["x1"])
    assert np.allclose(ticks["y0"], ticks["y1"])


def test_unknown_length_mode_raises():
    image = im.gaussian_blob(npix=32, p_lin=0.2)
    with pytest.raises(ValueError, match="length_mode"):
        im.evpa_ticks(image, length_mode="nonsense")


def test_tick_length_tracks_the_polarization_fraction():
    """Ticks are on an absolute scale: twice the polarization, twice the length.

    Regression test -- they used to be normalised by their own maximum, so
    dragging a polarization slider changed nothing on screen.
    """
    def tick_length(p_lin):
        image = im.polarized_ring(npix=128, p_lin=p_lin, asymmetry=0.0)
        ticks = im.evpa_ticks(image, step=6, i_cut=0.2)
        return float(np.hypot(ticks["x1"] - ticks["x0"], ticks["y1"] - ticks["y0"]).max())

    assert tick_length(0.4) == pytest.approx(2 * tick_length(0.2), rel=1e-9)
    assert tick_length(0.0) == 0.0


def test_fraction_mode_ticks_are_bounded_by_the_fraction():
    image = im.polarized_ring(npix=96, p_lin=0.5)
    ticks = im.evpa_ticks(image, step=6, i_cut=0.2, length_mode="fraction")
    uniform = im.evpa_ticks(image, step=6, i_cut=0.2, length_mode="uniform")
    length = np.hypot(ticks["x1"] - ticks["x0"], ticks["y1"] - ticks["y0"])
    full = np.hypot(uniform["x1"] - uniform["x0"], uniform["y1"] - uniform["y0"])
    assert np.allclose(length, 0.5 * full)
