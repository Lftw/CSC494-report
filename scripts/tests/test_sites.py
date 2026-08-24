"""Checks for core.sites: the station table, sky geometry, and uv projection.

Geometry is checked against things that must be true by physics (a source
transits at its highest, the pole never sets) and, where possible, against
astropy -- the point being that notebooks 01-02 compute this themselves but must
agree with what eht-imaging does in notebooks 03-04.
"""

import os
import sys
from pathlib import Path

import arrays as arr
import numpy as np
import pytest

MJD_2017_APR11 = 57854.0          # the M87 observing night
M87_RA, M87_DEC = arr.SOURCES["M87"]


def test_the_2017_array_is_the_eight_stations_that_observed_m87():
    array = arr.array_2017()
    assert set(array) == {"ALMA", "APEX", "SMT", "LMT", "SMA", "JCMT", "PV", "SPT"}
    assert len(arr.baselines(array)) == 28          # n(n-1)/2


def test_alma_is_the_linear_feed_station():
    sites = arr.load_sites()
    assert sites["ALMA"].is_linear
    assert sites["PDB"].is_linear                   # NOEMA, also linear
    assert not sites["SMA"].is_linear
    assert sum(site.is_linear for site in arr.array_2017().values()) == 1


def test_alma_is_by_far_the_most_sensitive_station():
    array = arr.array_2017()
    assert array["ALMA"].sefd_jy == min(site.sefd_jy for site in array.values())
    assert array["ALMA"].collecting_area > 10 * array["JCMT"].collecting_area


def test_station_latitudes_and_longitudes_are_where_they_should_be():
    sites = arr.load_sites()
    assert np.rad2deg(sites["SPT"].latitude) == pytest.approx(-90.0, abs=0.01)
    assert np.rad2deg(sites["ALMA"].latitude) == pytest.approx(-23.0, abs=1.0)
    assert np.rad2deg(sites["ALMA"].longitude) == pytest.approx(-67.8, abs=1.0)
    assert np.rad2deg(sites["SMA"].latitude) == pytest.approx(19.8, abs=1.0)
    assert np.rad2deg(sites["PV"].longitude) == pytest.approx(-3.4, abs=1.0)


def test_mount_types_are_carried_through():
    sites = arr.load_sites()
    assert (sites["JCMT"].fr_par, sites["JCMT"].fr_elev) == (1.0, 0.0)   # alt-az
    assert (sites["SMA"].fr_par, sites["SMA"].fr_elev) == (1.0, -1.0)    # Nasmyth left
    assert (sites["APEX"].fr_par, sites["APEX"].fr_elev) == (1.0, 1.0)   # Nasmyth right
    assert sites["SMA"].fr_offset_deg == 45.0


# ---------------------------------------------------------------------------
# Sky geometry
# ---------------------------------------------------------------------------

def test_gmst_matches_astropy():
    astropy_time = pytest.importorskip("astropy.time")
    for mjd in (57854.0, 57854.35, 59000.9, 60000.0):
        expected = astropy_time.Time(mjd, format="mjd").sidereal_time(
            "mean", "greenwich").hour
        assert arr.gmst_hours(mjd) == pytest.approx(expected, abs=1.0 / 3600.0)


def test_gmst_advances_by_a_sidereal_day():
    """A sidereal day is about four minutes short of a solar one."""
    drift_hours = (arr.gmst_hours(51545.0) - arr.gmst_hours(51544.0)) % 24.0
    assert drift_hours * 60 == pytest.approx(3.94, abs=0.05)


def test_elevation_peaks_at_transit():
    site = arr.load_sites()["ALMA"]
    hours = np.linspace(0, 24, 24 * 60)
    mjd = MJD_2017_APR11 + hours / 24.0
    elev = arr.elevation(site, mjd, M87_RA, M87_DEC)
    ha = arr.hour_angle(mjd, site.longitude, M87_RA)

    peak = np.argmax(elev)
    assert min(ha[peak], 2 * np.pi - ha[peak]) < np.deg2rad(1.0)   # hour angle ~ 0
    # Peak elevation is 90 - |latitude - declination|.
    expected = 90.0 - abs(np.rad2deg(site.latitude) - M87_DEC)
    assert np.rad2deg(elev[peak]) == pytest.approx(expected, abs=0.5)


def test_the_south_pole_cannot_see_m87():
    """SPT sits at declination -90, so a source at +12 degrees is always below it."""
    site = arr.load_sites()["SPT"]
    mjd = MJD_2017_APR11 + np.linspace(0, 1, 200)
    assert np.all(arr.elevation(site, mjd, M87_RA, M87_DEC) < 0)
    # ...but Sgr A* at -29 degrees is permanently up.
    ra_sgr, dec_sgr = arr.SOURCES["SgrA*"]
    assert np.all(arr.elevation(site, mjd, ra_sgr, dec_sgr) > 0)


def test_parallactic_angle_is_zero_at_transit_and_flips_sign_across_it():
    site = arr.load_sites()["LMT"]
    at_transit = arr.parallactic_angle(site, _transit_mjd(site), M87_RA, M87_DEC)
    assert abs(at_transit) < np.deg2rad(0.5)

    before = arr.parallactic_angle(site, _transit_mjd(site) - 2 / 24, M87_RA, M87_DEC)
    after = arr.parallactic_angle(site, _transit_mjd(site) + 2 / 24, M87_RA, M87_DEC)
    assert np.sign(before) == -np.sign(after)


def test_field_rotation_differs_between_mount_types():
    """Two stations watching the same source at the same instant rotate differently."""
    sites = arr.load_sites()
    mjd = MJD_2017_APR11 + 4.0 / 24.0
    sma = arr.field_rotation_angle(sites["SMA"], mjd, M87_RA, M87_DEC)
    jcmt = arr.field_rotation_angle(sites["JCMT"], mjd, M87_RA, M87_DEC)
    # Same mountain, so the parallactic angles agree to a fraction of a degree...
    par_sma = arr.parallactic_angle(sites["SMA"], mjd, M87_RA, M87_DEC)
    par_jcmt = arr.parallactic_angle(sites["JCMT"], mjd, M87_RA, M87_DEC)
    assert abs(par_sma - par_jcmt) < np.deg2rad(0.1)
    # ...yet the feed rotations differ, because the optics do.
    assert abs(sma - jcmt) > np.deg2rad(10.0)


def _transit_mjd(site: arr.Site) -> float:
    """MJD of M87's transit at a site, found by scanning one day."""
    mjd = MJD_2017_APR11 + np.linspace(0, 1, 24 * 60 * 4)
    return float(mjd[np.argmax(arr.elevation(site, mjd, M87_RA, M87_DEC))])


# ---------------------------------------------------------------------------
# uv coverage
# ---------------------------------------------------------------------------

def test_uv_length_never_exceeds_the_physical_baseline():
    sites = arr.load_sites()
    mjd = MJD_2017_APR11 + np.linspace(0, 1, 500)
    u, v = arr.uv_coordinates(sites["ALMA"], sites["SMA"], mjd, M87_RA, M87_DEC)
    physical = np.linalg.norm(sites["ALMA"].xyz - sites["SMA"].xyz)
    wavelength = arr.C_LIGHT / 230e9
    assert np.hypot(u, v).max() <= physical / wavelength * (1 + 1e-9)
    # A projection can only shorten a baseline, and this one really is foreshortened.
    assert np.hypot(u, v).min() < 0.9 * physical / wavelength


def test_swapping_the_stations_flips_the_uv_point():
    sites = arr.load_sites()
    mjd = MJD_2017_APR11 + 0.3
    u1, v1 = arr.uv_coordinates(sites["ALMA"], sites["LMT"], mjd, M87_RA, M87_DEC)
    u2, v2 = arr.uv_coordinates(sites["LMT"], sites["ALMA"], mjd, M87_RA, M87_DEC)
    assert u1 == pytest.approx(-u2) and v1 == pytest.approx(-v2)


def test_earth_rotation_traces_an_ellipse():
    """The signature of a uv track: over one *sidereal* day it closes on itself."""
    sites = arr.load_sites()
    hours = np.linspace(0, 23.9344696, 400)   # a sidereal day, not a solar one
    track = arr.uv_track(sites["ALMA"], sites["PV"], MJD_2017_APR11, hours,
                        M87_RA, M87_DEC, elev_min_deg=-90)
    assert track["u"][0] == pytest.approx(track["u"][-1], abs=1e5)
    assert track["v"][0] == pytest.approx(track["v"][-1], abs=1e5)
    # v is offset from zero by the declination projection, u is centred.
    assert abs(np.mean(track["u"])) < 0.2 * np.ptp(track["u"])


def test_uv_coverage_only_keeps_mutually_visible_samples():
    array = arr.array_2017()
    hours = np.linspace(0, 24, 200)
    coverage = arr.uv_coverage(array, MJD_2017_APR11, hours, M87_RA, M87_DEC)
    assert len(coverage["u"]) > 0
    assert len(coverage["u"]) < 28 * len(hours)            # some baselines are down
    assert set(zip(coverage["site1"], coverage["site2"], strict=True)) <= set(
        arr.baselines(array))
    # SPT can never see M87, so no baseline to it survives.
    assert "SPT" not in set(coverage["site1"]) | set(coverage["site2"])


def test_thermal_noise_matches_the_radiometer_equation():
    # Doubling the integration time cuts the noise by root two.
    short = arr.thermal_noise(90, 5000, 10.0, 2e9)
    long = arr.thermal_noise(90, 5000, 40.0, 2e9)
    assert short / long == pytest.approx(2.0)
    # A sensitive station lifts every baseline it joins.
    assert arr.thermal_noise(90, 5000, 10, 2e9) < arr.thermal_noise(3500, 5000, 10, 2e9)
    # Explicit value, so a refactor cannot drift the formula.
    assert arr.thermal_noise(100.0, 100.0, 10.0, 2e9) == pytest.approx(
        np.sqrt(100.0 * 100.0 / (2 * 2e9 * 10.0)) / 0.88)


def test_every_notebook_bootstraps_the_flat_scripts_folder():
    """The first cell of each notebook must add scripts/ to the path and import flat.

    Written as a test because it is the one piece of setup a reader cannot debug:
    if the path is wrong, every later cell fails with an unrelated-looking error.
    """
    import nbformat

    repo = Path(__file__).resolve().parent.parent.parent
    assert (repo / "scripts" / "polarization.py").is_file()

    for notebook in sorted((repo / "notebooks").glob("*.ipynb")):
        first = next(c for c in nbformat.read(notebook, as_version=4).cells
                     if c.cell_type == "code")
        assert "scripts" in first.source, notebook.name
        assert "csc494" not in first.source, notebook.name

        # Running the setup lines from the notebook's own folder must work.
        code = first.source.split("import numpy as np")[0]
        cwd = os.getcwd()
        try:
            os.chdir(notebook.parent)
            exec(code, {"__name__": "__main__"})
        finally:
            os.chdir(cwd)
        assert str(repo / "scripts") in sys.path
