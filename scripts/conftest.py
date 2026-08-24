"""Put ``scripts/`` on ``sys.path`` so the tests import the same flat modules the
notebooks do: ``import polarization``, ``import arrays``, and so on.

The report has no install step. Notebooks add this directory to the path in their
first cell; the tests get it from here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
