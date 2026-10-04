"""Exchange — a falsification-first quantitative research OS.

The package is organised as a one-directional pipeline::

    data -> execution -> backtest -> stats -> scanner
         -> funnel -> falsification -> registry -> reporting

Every layer is importable on its own and depends only on the layers above
it, so a component can be tested in isolation.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
