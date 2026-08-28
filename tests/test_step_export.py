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

from step_export import build_step_surfaces, face_from_edges


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


def test_face_from_edges_rejects_a_self_crossing_boundary():
    # Regression test: on a real folded scan, some cells end up with
    # degenerate/self-intersecting boundary curves. OpenCASCADE can
    # "successfully" fill these (IsDone() True) while producing a
    # geometrically invalid face -- this showed up as SolidWorks
    # import diagnostics flagging faces as invalid. face_from_edges
    # must catch this itself (via BRepCheck_Analyzer) rather than
    # silently exporting bad geometry.
    bowtie_edges = [
        [[0, 0, 0], [1, 1, 0]],
        [[1, 1, 0], [1, 0, 0]],
        [[1, 0, 0], [0, 1, 0]],
        [[0, 1, 0], [0, 0, 0]],
    ]

    try:
        face_from_edges(bowtie_edges)
    except RuntimeError:
        pass
    else:
        raise AssertionError(
            "expected a RuntimeError for a self-crossing boundary"
        )
