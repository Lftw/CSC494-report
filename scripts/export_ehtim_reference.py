"""Capture eht-imaging's polarization conventions into a committed JSON file.

Notebook 03 checks its own algebra against eht-imaging, but the notebooks run in
the ``csc494`` env, which has no eht-imaging. So the matrices are captured here
once, in an env that does, and committed as
``scripts/data/ehtim_pol_conventions.json`` -- the notebook compares against the
file and still shows real numbers everywhere.

Run it in the env that has eht-imaging::

    micromamba run -n jax-ehtim python scripts/export_ehtim_reference.py

Only needed to refresh the capture against a newer eht-imaging. Whenever
eht-imaging *is* importable, ``ehtim_reference.check_live()`` and
``scripts/tests/test_ehtim_reference.py`` re-do the capture and compare it with
the file, so a stale one is caught rather than trusted.

Usage:
    python scripts/export_ehtim_reference.py [output.json]
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ehtim_reference  # noqa: E402


def main(out_path: Path) -> None:
    import ehtim
    from ehtim.observing import pol_conventions

    matrices = ehtim_reference.capture(pol_conventions)

    module = Path(pol_conventions.__file__)
    document = {
        "note": ("eht-imaging's polarization conventions, read out of the module "
                 "named below. Regenerate with "
                 "scripts/export_ehtim_reference.py. Complex matrices are stored "
                 "as separate 'real' and 'imag' nested lists; the 4x4 matrices "
                 "have Stokes/slot components down their columns."),
        "ehtim_version": ehtim.__version__,
        "ehtim_module": "/".join(module.parts[-4:]),
        "captured": date.today().isoformat(),
        **matrices,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")

    disagreements = [(label, error)
                     for label, error in ehtim_reference.comparisons(
                         ehtim_reference.load(out_path))
                     if error >= ehtim_reference.TOLERANCE]
    print(f"wrote {out_path} (eht-imaging v{ehtim.__version__})")
    if disagreements:
        print("  ! jones.py disagrees with the capture:")
        for label, error in disagreements:
            print(f"    {label}: max difference {error:.1e}")
    else:
        print("  jones.py agrees with all of it")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1
         else ehtim_reference.REFERENCE_PATH)
