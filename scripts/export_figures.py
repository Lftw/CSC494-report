"""Render every interactive figure to a PNG, for slides and for checking layout.

Each explorer is built in ``static=True`` mode -- no widgets, no kernel -- at the
width the figure itself declares, which is the only honest way to check that
titles, legends and labels are not colliding.

Static export needs a headless browser::

    pip install kaleido
    plotly_get_chrome

On WSL/Ubuntu that Chrome build also wants libraries the base image lacks; the
conda-forge ones work without root::

    micromamba install -n csc494 -c conda-forge nss nspr alsa-lib
    LD_LIBRARY_PATH=$CONDA_PREFIX/lib python scripts/export_figures.py

Usage:
    python scripts/export_figures.py [output-directory]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import interactive_jones  # noqa: E402
import interactive_polarization  # noqa: E402
import interactive_telescope  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "figures"


def explorers() -> list[tuple[str, object]]:
    """Every explorer in the report, notebook order, as ``(name, function)``."""
    found = []
    for module in (interactive_polarization, interactive_telescope, interactive_jones):
        for name in module.__all__:
            attribute = getattr(module, name)
            if callable(attribute):
                found.append((name, attribute))
    return found


def main(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, explorer in explorers():
        figure = explorer(static=True)
        target = out_dir / f"{name}.png"
        try:
            figure.write_image(str(target),
                               width=int(figure.layout.width or 820),
                               height=int(figure.layout.height or 400))
            print(f"  {target.name}")
        except Exception as exc:  # pragma: no cover - depends on the browser
            failures += 1
            print(f"  FAILED {name}: {type(exc).__name__}: {str(exc)[:90]}")
    print(f"\n{len(explorers()) - failures} figures written to {out_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT))
