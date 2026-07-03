"""Execute a notebook in place and make it GitHub-renderable.

Usage::

    python scripts/run_notebook.py notebooks/01_repository_setup.ipynb

Runs all cells top-to-bottom, then strips ipywidgets state
(``metadata.widgets``) that vectorbt's FigureWidgets leave behind — it
adds megabytes to the file and can break GitHub's notebook renderer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def strip_widget_state(path: Path) -> None:
    nb = json.loads(path.read_text())
    nb.get("metadata", {}).pop("widgets", None)
    for cell in nb["cells"]:
        cell.get("metadata", {}).pop("widgets", None)
        for output in cell.get("outputs", []):
            output.get("data", {}).pop("application/vnd.jupyter.widget-view+json", None)
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1))


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"Notebook not found: {path}")
        return 1

    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", str(path)],
        check=True,
    )
    strip_widget_state(path)
    print(f"Executed and cleaned: {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
