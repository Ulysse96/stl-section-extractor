"""
Smoke test for step_export.py. Needs OCP (OpenCASCADE), which only
exists in the project's dedicated .venv312 -- see README. Under any
other Python (including the one running the rest of this test suite)
this whole module is skipped rather than failing.

Deliberately does NOT import curve_utils (or anything that pulls in
scipy/vtk): step_export.py itself only needs OCP + the standard
library, and this test's grid fixture is built by hand to match
build_surface_grid's output schema, to keep that boundary real. On
this project's original dev machine, Windows Smart App Control blocks
several freshly pip-installed native packages inside .venv312 (vtk,
scipy's compiled extensions) but not OCP -- importing curve_utils here
would hit that wall for a reason that has nothing to do with what this
test is meant to check.
"""

import numpy as np
import pytest

pytest.importorskip("OCP")

from step_export import build_step_surfaces


def _lattice_grid():
    # Same flat 3x3 lattice as tests/test_curve_utils.py's
    # build_surface_grid tests, but assembled directly into the
    # {"cells": [...]} shape build_step_surfaces expects.
    cells = []

    for a_i in (1, 2):
        for b_j in (1, 2):
            cells.append(
                {
                    "a_i": a_i,
                    "a_i_next": a_i + 1,
                    "b_j": b_j,
                    "b_j_next": b_j + 1,
                    "edge_a_lo": np.array(
                        [[b_j, a_i, 0], [b_j + 1, a_i, 0]], dtype=float
                    ),
                    "edge_a_hi": np.array(
                        [[b_j, a_i + 1, 0], [b_j + 1, a_i + 1, 0]],
                        dtype=float,
                    ),
                    "edge_b_lo": np.array(
                        [[b_j, a_i, 0], [b_j, a_i + 1, 0]], dtype=float
                    ),
                    "edge_b_hi": np.array(
                        [[b_j + 1, a_i, 0], [b_j + 1, a_i + 1, 0]],
                        dtype=float,
                    ),
                }
            )

    return {"cells": cells}


def test_build_step_surfaces_writes_a_valid_step_file(tmp_path):
    grid = _lattice_grid()

    output_path = tmp_path / "surface.step"

    built, failures = build_step_surfaces(grid, str(output_path))

    assert built == 4
    assert failures == []
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8", errors="ignore")

    assert "ADVANCED_FACE" in content
    assert "MILLI" in content and "METRE" in content
