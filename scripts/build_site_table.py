"""Build ``scripts/data/eht_sites.csv`` from the eht-imaging array files.

Station coordinates, SEFDs, field-rotation coefficients and leakage terms come
from ``eht-imaging/arrays/EHT2017.txt`` and ``EHT2025.txt``; the human-readable
metadata (full name, dish size, feed basis) is curated here. Coordinates are
publicly published EHT array parameters -- this script exists so the report
carries a small, documented data file instead of a copy of someone's source
tree.

Usage:
    python scripts/build_site_table.py [path/to/eht-imaging]

Writes ``scripts/data/eht_sites.csv``, which is committed -- you only need
to run this to refresh it against a newer eht-imaging checkout.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
DEFAULT_EHTIM = SCRIPTS.parent.parent / "eht-imaging"

# Curated per-site metadata. `feeds` is the receiver's native polarization
# basis: 'rl' for circular feeds, 'xy' for linear. See the note in the CSV
# header -- this is the modelling choice the report is built on.
METADATA = {
    "ALMA": ("Atacama Large Millimeter/submillimeter Array", "Chile", 12.0, 37, "xy"),
    "APEX": ("Atacama Pathfinder Experiment", "Chile", 12.0, 1, "rl"),
    "SMT": ("Submillimeter Telescope", "Arizona, USA", 10.0, 1, "rl"),
    "LMT": ("Large Millimeter Telescope Alfonso Serrano", "Mexico", 32.5, 1, "rl"),
    "SMA": ("Submillimeter Array", "Hawaii, USA", 6.0, 8, "rl"),
    "JCMT": ("James Clerk Maxwell Telescope", "Hawaii, USA", 15.0, 1, "rl"),
    "PV": ("IRAM 30m Telescope, Pico Veleta", "Spain", 30.0, 1, "rl"),
    "SPT": ("South Pole Telescope", "Antarctica", 10.0, 1, "rl"),
    "GLT": ("Greenland Telescope", "Greenland", 12.0, 1, "rl"),
    "KP": ("Kitt Peak 12m Telescope", "Arizona, USA", 12.0, 1, "rl"),
    "PDB": ("NOEMA, Plateau de Bure", "France", 15.0, 12, "xy"),
}

#: Which array each site belongs to, for the "the array grows" story.
IN_2017 = {"ALMA", "APEX", "SMT", "LMT", "SMA", "JCMT", "PV", "SPT"}

HEADER_NOTE = [
    "# EHT station table for the CSC494 report.",
    "#",
    "# Coordinates (geocentric ECEF, metres), SEFDs (Jy), field-rotation",
    "# coefficients and leakage terms are taken from eht-imaging's",
    "# arrays/EHT2017.txt and arrays/EHT2025.txt (GPL-3, achael/eht-imaging);",
    "# they are publicly published EHT array parameters.",
    "#",
    "# fr_par / fr_elev / fr_offset parametrise the field rotation angle as",
    "#     phi = fr_par * parallactic_angle + fr_elev * elevation + fr_offset,",
    "# which encodes the mount type: 1/0 is a standard alt-az mount, 1/-1 a",
    "# Nasmyth-left optical path, 1/+1 Nasmyth-right, 0/0 an equatorial mount.",
    "#",
    "# `feeds` is the receiver's native polarization basis and is a modelling",
    "# choice of this report: ALMA and NOEMA record linear (X/Y) feeds, the",
    "# other stations circular (R/L). ALMA's data is normally converted to a",
    "# circular basis by the QA2 pipeline before imaging -- handling it",
    "# natively, without that conversion, is what this project is about.",
]

COLUMNS = ["site", "name", "location", "x_m", "y_m", "z_m", "diameter_m", "n_dishes",
           "feeds", "sefd_jy", "fr_par", "fr_elev", "fr_offset_deg",
           "d_re_1", "d_im_1", "d_re_2", "d_im_2", "in_eht2017"]


def read_array_file(path: Path) -> dict[str, list[str]]:
    """Parse an ehtim array file into ``{site: [fields]}``."""
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        rows[fields[0]] = fields[1:]
    return rows


def main(ehtim_root: Path) -> None:
    arrays = ehtim_root / "arrays"
    combined = read_array_file(arrays / "EHT2025.txt")
    combined.update(read_array_file(arrays / "EHT2017.txt"))  # 2017 values win

    out_path = SCRIPTS / "data" / "eht_sites.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        for note in HEADER_NOTE:
            handle.write(note + "\n")
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for site, meta in METADATA.items():
            if site not in combined:
                print(f"  ! {site} not in the ehtim array files, skipping")
                continue
            x, y, z, sefdr, _sefdl, fr_par, fr_elev, fr_off, *dterms = combined[site]
            name, location, diameter, n_dishes, feeds = meta
            writer.writerow([site, name, location, x, y, z, diameter, n_dishes, feeds,
                             sefdr, fr_par, fr_elev, fr_off, *dterms[:4],
                             int(site in IN_2017)])
    print(f"wrote {out_path} ({len(METADATA)} sites)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EHTIM)
