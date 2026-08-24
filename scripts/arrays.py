"""The array: station coordinates, feeds, and the sky geometry of an observation.

Everything here mirrors what eht-imaging does, on purpose. ``hour_angle``,
``elevation``, ``parallactic_angle`` and ``uv_coordinates`` are the same formulas
as ``ehtim/observing/obs_helpers.py``, so the self-contained notebooks and the
ehtim-backed ones cannot quietly disagree.

Angles are radians unless a name says otherwise. Positions are geocentric
(ECEF) metres. Time is MJD, with the observation's time-of-day carried
separately in hours UTC, which is the convention ehtim's ``Obsdata`` uses.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

C_LIGHT = 299792458.0
#: The station table sits next to this module; see build_site_table.py.
DATA_DIR = Path(__file__).resolve().parent / "data"
SITE_TABLE = DATA_DIR / "eht_sites.csv"

#: Sources the report observes, as (RA hours, Dec degrees).
SOURCES = {
    "M87": (12.513728717168174, 12.39112323919932),
    "SgrA*": (17.761121055553343, -29.00784305556),
}


@dataclass(frozen=True)
class Site:
    """One station: where it is, what it can hear, and how its feeds are mounted.

    Attributes
    ----------
    code : str
        Short station code, as used in uvfits and in ehtim (``'ALMA'``).
    xyz : numpy.ndarray
        Geocentric position in metres.
    feeds : str
        Native polarization basis: ``'rl'`` circular or ``'xy'`` linear.
    sefd_jy : float
        System equivalent flux density -- the noise power expressed as the flux
        of a source that would double it. Smaller is more sensitive.
    fr_par, fr_elev, fr_offset_deg : float
        Field-rotation parameters, ``phi = fr_par * parallactic +
        fr_elev * elevation + fr_offset``. They encode the mount and optics:
        ``(1, 0)`` is a plain alt-az mount, ``(1, -1)`` and ``(1, +1)`` are
        Nasmyth-left and -right, ``(0, 0)`` an equatorial mount that does not
        rotate the feeds at all.
    """

    code: str
    name: str
    location: str
    xyz: np.ndarray
    diameter_m: float
    n_dishes: int
    feeds: str
    sefd_jy: float
    fr_par: float
    fr_elev: float
    fr_offset_deg: float
    dterms: tuple[complex, complex]
    in_eht2017: bool

    @property
    def latitude(self) -> float:
        """Geocentric latitude in radians (ehtim's ``xyz_2_latlong``)."""
        return float(np.arctan2(self.xyz[2], np.hypot(self.xyz[0], self.xyz[1])))

    @property
    def longitude(self) -> float:
        """Longitude in radians, positive east."""
        return float(np.arctan2(self.xyz[1], self.xyz[0]))

    @property
    def is_linear(self) -> bool:
        """True if this station's receiver records a linear (X/Y) basis."""
        return self.feeds == "xy"

    @property
    def collecting_area(self) -> float:
        """Total geometric collecting area in m^2 -- phased arrays count all dishes."""
        return self.n_dishes * np.pi * (0.5 * self.diameter_m) ** 2

    def __repr__(self) -> str:
        return (f"Site({self.code}, {self.feeds} feeds, SEFD {self.sefd_jy:.0f} Jy, "
                f"{np.rad2deg(self.latitude):+.1f}° lat)")


@lru_cache(maxsize=1)
def load_sites(path: Path | None = None) -> dict[str, Site]:
    """Read the station table shipped in ``data/arrays/``.

    Parameters
    ----------
    path : pathlib.Path, optional
        Override the default table (see ``scripts/build_site_table.py`` for how
        it is generated and where the numbers come from).

    Returns
    -------
    dict
        Station code -> :class:`Site`, in file order.
    """
    path = path or SITE_TABLE
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.startswith("#")))
    sites = {}
    for row in rows:
        sites[row["site"]] = Site(
            code=row["site"], name=row["name"], location=row["location"],
            xyz=np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]),
            diameter_m=float(row["diameter_m"]), n_dishes=int(row["n_dishes"]),
            feeds=row["feeds"], sefd_jy=float(row["sefd_jy"]),
            fr_par=float(row["fr_par"]), fr_elev=float(row["fr_elev"]),
            fr_offset_deg=float(row["fr_offset_deg"]),
            dterms=(complex(float(row["d_re_1"]), float(row["d_im_1"])),
                    complex(float(row["d_re_2"]), float(row["d_im_2"]))),
            in_eht2017=bool(int(row["in_eht2017"])),
        )
    return sites


def array_2017() -> dict[str, Site]:
    """The eight stations that observed M87 in April 2017."""
    return {code: site for code, site in load_sites().items() if site.in_eht2017}


def baselines(sites: dict[str, Site]) -> list[tuple[str, str]]:
    """Every unordered pair of stations, in table order. ``n(n-1)/2`` of them."""
    codes = list(sites)
    return [(a, b) for i, a in enumerate(codes) for b in codes[i + 1:]]


# ---------------------------------------------------------------------------
# Sky geometry
# ---------------------------------------------------------------------------

def gmst_hours(mjd: float | np.ndarray) -> float | np.ndarray:
    """Greenwich mean sidereal time in hours, from the Modified Julian Date.

    The standard low-precision series (good to well under a second over the
    span of any observing campaign), so ``core`` stays dependency-free. Checked
    against astropy in ``tests/test_sites.py``.
    """
    days_since_j2000 = np.asarray(mjd, dtype=float) - 51544.5
    return np.mod(18.697374558 + 24.06570982441908 * days_since_j2000, 24.0)


def hour_angle(mjd: float | np.ndarray, longitude: float, ra_hours: float):
    """Hour angle of a source: how far past the meridian it is, in radians.

    Zero at transit, increasing westward. This is the clock that drives
    everything else -- elevation, feed rotation, and the uv track.
    """
    gmst_rad = gmst_hours(mjd) * np.pi / 12.0
    return np.mod(gmst_rad + longitude - ra_hours * np.pi / 12.0, 2 * np.pi)


def source_vector(ra_hours: float, dec_deg: float) -> np.ndarray:
    """Unit vector toward the source in the Earth-fixed frame at hour angle zero."""
    dec = np.deg2rad(dec_deg)
    return np.array([np.cos(dec), 0.0, np.sin(dec)])


def elevation(site: Site, mjd: float | np.ndarray, ra_hours: float, dec_deg: float):
    """Source elevation above the station's horizon, in radians.

    Negative means the source is below the horizon and the station is simply
    not observing -- which is why an array spread over the whole planet still
    only ever has a subset of its baselines live at any moment.
    """
    ha = hour_angle(mjd, site.longitude, ra_hours)
    lat, dec = site.latitude, np.deg2rad(dec_deg)
    sin_el = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(ha)
    return np.arcsin(np.clip(sin_el, -1.0, 1.0))


def parallactic_angle(site: Site, mjd: float | np.ndarray, ra_hours: float,
                      dec_deg: float):
    """Angle between the celestial pole and the station's zenith, at the source.

    An alt-az mount holds its feeds fixed relative to the *ground*, so as the
    source tracks across the sky the feeds rotate against it by this angle.
    Same formula as ehtim's ``obs_helpers.par_angle``.
    """
    ha = hour_angle(mjd, site.longitude, ra_hours)
    lat, dec = site.latitude, np.deg2rad(dec_deg)
    return np.arctan2(np.sin(ha) * np.cos(lat),
                      np.sin(lat) * np.cos(dec) - np.cos(lat) * np.sin(dec) * np.cos(ha))


def field_rotation_angle(site: Site, mjd: float | np.ndarray, ra_hours: float,
                         dec_deg: float):
    """Total rotation of the feeds relative to the sky, in radians.

    ``phi = fr_par * parallactic + fr_elev * elevation + fr_offset``. The
    coefficients come from the mount and the optical path, so two stations
    watching the same source at the same instant generally have *different*
    feed rotations -- one of the reasons polarimetric calibration is per-station
    work.
    """
    par = parallactic_angle(site, mjd, ra_hours, dec_deg)
    elev = elevation(site, mjd, ra_hours, dec_deg)
    return site.fr_par * par + site.fr_elev * elev + np.deg2rad(site.fr_offset_deg)


def uv_coordinates(site1: Site, site2: Site, mjd: float | np.ndarray,
                   ra_hours: float, dec_deg: float,
                   freq_hz: float = 230e9) -> tuple[np.ndarray, np.ndarray]:
    """Baseline projected onto the sky plane, in wavelengths.

    The baseline vector between two stations is rotated into the frame of the
    source and projected perpendicular to the line of sight; the component
    along the line of sight is discarded because it carries no image
    information. Dividing by the wavelength gives the ``(u, v)`` at which this
    baseline samples the sky's Fourier transform.

    Parameters
    ----------
    site1, site2 : Site
        The two stations. Order sets the sign, and therefore the sign of the
        visibility phase.
    mjd : float or numpy.ndarray
        Modified Julian Date(s), including the time of day.
    ra_hours, dec_deg : float
        Source position.
    freq_hz : float, optional
        Observing frequency. The EHT's is 230 GHz.

    Returns
    -------
    u, v : numpy.ndarray
        Coordinates in wavelengths (the units the visibility function is
        conventionally indexed in).
    """
    wavelength = C_LIGHT / freq_hz
    theta = np.atleast_1d(hour_angle(mjd, 0.0, ra_hours))  # Earth rotation angle
    baseline = site1.xyz - site2.xyz

    # Rotate the (fixed) baseline with the Earth, then project.
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    b_x = cos_t * baseline[0] - sin_t * baseline[1]
    b_y = sin_t * baseline[0] + cos_t * baseline[1]
    b_z = np.full_like(theta, baseline[2])

    dec = np.deg2rad(dec_deg)
    u = b_y / wavelength
    v = (-np.sin(dec) * b_x + np.cos(dec) * b_z) / wavelength
    return u, v


def uv_track(site1: Site, site2: Site, mjd_start: float, hours: np.ndarray,
             ra_hours: float, dec_deg: float, freq_hz: float = 230e9,
             elev_min_deg: float = 10.0) -> dict:
    """A baseline's uv track over an observing night, with the visibility mask.

    Returns
    -------
    dict
        ``u``, ``v`` (wavelengths), ``hours``, ``visible`` (both stations above
        the elevation limit), ``elev1``, ``elev2`` (degrees).
    """
    mjd = mjd_start + hours / 24.0
    u, v = uv_coordinates(site1, site2, mjd, ra_hours, dec_deg, freq_hz)
    elev1 = np.rad2deg(elevation(site1, mjd, ra_hours, dec_deg))
    elev2 = np.rad2deg(elevation(site2, mjd, ra_hours, dec_deg))
    visible = (elev1 > elev_min_deg) & (elev2 > elev_min_deg)
    return {"u": u, "v": v, "hours": np.asarray(hours), "visible": visible,
            "elev1": elev1, "elev2": elev2}


def uv_coverage(sites: dict[str, Site], mjd_start: float, hours: np.ndarray,
                ra_hours: float, dec_deg: float, freq_hz: float = 230e9,
                elev_min_deg: float = 10.0) -> dict:
    """Every visible uv sample of a full array over an observing night.

    Conjugate points are included: a baseline measures ``V(u, v)`` and, because
    the sky is real, ``V(-u, -v) = V(u, v)^*`` for free. That is why uv plots
    are always symmetric through the origin.

    Returns
    -------
    dict
        Flat arrays ``u``, ``v``, ``hours``, ``site1``, ``site2``, plus
        ``per_baseline`` mapping ``(code1, code2)`` to its track.
    """
    all_u, all_v, all_t, s1, s2 = [], [], [], [], []
    per_baseline = {}
    for code1, code2 in baselines(sites):
        track = uv_track(sites[code1], sites[code2], mjd_start, hours,
                         ra_hours, dec_deg, freq_hz, elev_min_deg)
        per_baseline[(code1, code2)] = track
        mask = track["visible"]
        all_u.append(track["u"][mask])
        all_v.append(track["v"][mask])
        all_t.append(track["hours"][mask])
        s1 += [code1] * int(mask.sum())
        s2 += [code2] * int(mask.sum())
    return {"u": np.concatenate(all_u) if all_u else np.array([]),
            "v": np.concatenate(all_v) if all_v else np.array([]),
            "hours": np.concatenate(all_t) if all_t else np.array([]),
            "site1": np.array(s1), "site2": np.array(s2),
            "per_baseline": per_baseline}


def thermal_noise(sefd1: float, sefd2: float, tint_s: float, bandwidth_hz: float,
                  quantization: float = 0.88) -> float:
    """Noise on one correlation product, in Jy. Same as ehtim's ``blnoise``.

    ``sigma = sqrt(SEFD_1 * SEFD_2 / (2 * bandwidth * t_int)) / 0.88``

    The *geometric mean* of the two SEFDs is what matters, which is why one very
    sensitive station lifts every baseline it touches -- the reason ALMA
    transformed the EHT. The 0.88 is the loss from 2-bit quantization.

    Note for the mixed-polarization story: the SEFD is a property of a *feed*,
    not of a station, so a baseline between a linear-feed and a circular-feed
    station has four different noise levels, one per correlation slot.
    """
    return np.sqrt(sefd1 * sefd2 / (2.0 * bandwidth_hz * tint_s)) / quantization
