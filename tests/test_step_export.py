"""
Smoke test for step_export.py. Needs OCP (OpenCASCADE), which only
exists in the project's dedicated .venv312 -- see README. Under any
other Python (including the one running the rest of this test suite)
this whole module is skipped rather than failing.

Deliberately does NOT import curve_utils (or anything that pulls in
scipy/vtk): step_export.py itself only needs OCP + the standard
library. On this project's original dev machine, Windows Smart App
Control blocks several freshly pip-installed native packages inside
.venv312 (vtk, scipy's compiled extensions) but not OCP -- importing
curve_utils here would hit that wall for a reason that has nothing to
do with what this test is meant to check.
"""

import numpy as np
import pytest

pytest.importorskip("OCP")

from step_export import build_single_surface, build_step_surfaces


def _dome_z(x, y):
    return 0.3 * np.sin(x * 0.7) * np.cos(y * 0.5)


def _synthetic_patch():
    # A gently curved 4x4 patch: 5 A-direction curves, 5 B-direction
    # curves, and a closed boundary loop around the perimeter --
    # standing in for section_stl.py's rich_main_curves +
    # boundary_loop_points on a real scan.
    interior_curves = []

    for x in np.linspace(0, 4, 5):
        ys = np.linspace(0, 4, 30)
        interior_curves.append(
            np.column_stack([np.full_like(ys, x), ys, _dome_z(x, ys)])
        )

    for y in np.linspace(0, 4, 5):
        xs = np.linspace(0, 4, 30)
        interior_curves.append(
            np.column_stack([xs, np.full_like(xs, y), _dome_z(xs, y)])
        )

    boundary = []
    for x in np.linspace(0, 4, 15):
        boundary.append([x, 0, _dome_z(x, 0)])
    for y in np.linspace(0, 4, 15)[1:]:
        boundary.append([4, y, _dome_z(4, y)])
    for x in np.linspace(4, 0, 15)[1:]:
        boundary.append([x, 4, _dome_z(x, 4)])
    for y in np.linspace(4, 0, 15)[1:]:
        boundary.append([0, y, _dome_z(0, y)])
    boundary.append(boundary[0])

    return np.array(boundary), interior_curves


def test_build_single_surface_produces_one_valid_face():
    boundary, interior_curves = _synthetic_patch()

    face = build_single_surface(boundary, interior_curves, smoothing_tolerance_mm=0.3)

    assert face is not None


def test_build_single_surface_closes_an_open_boundary_loop():
    # Regression test: order_boundary_loop's output (what
    # section_stl.py actually passes in) is NOT closed -- its own
    # first point isn't repeated at the end. Feeding that straight
    # into OpenCASCADE as a bounding wire built, but into a
    # geometrically invalid face, confirmed directly against a real
    # 44-point boundary loop from an 11x11 section grid.
    # build_single_surface must close it itself.
    _, interior_curves = _synthetic_patch()

    open_boundary = np.array(
        [[0, 0, 0], [4, 0, 0.1], [4, 4, 0], [0, 4, -0.1]], dtype=float
    )
    assert not np.allclose(open_boundary[0], open_boundary[-1])

    face = build_single_surface(
        open_boundary, interior_curves, smoothing_tolerance_mm=0.3
    )

    assert face is not None


def test_build_step_surfaces_writes_a_valid_step_file_with_one_face(tmp_path):
    boundary, interior_curves = _synthetic_patch()

    data = {
        "boundary_loop": boundary,
        "interior_curves": interior_curves,
        "smoothing_tolerance_mm": 0.3,
    }

    output_path = tmp_path / "surface.step"

    build_step_surfaces(data, str(output_path))

    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8", errors="ignore")

    assert content.count("ADVANCED_FACE") == 1
    assert "MILLI" in content and "METRE" in content


def test_build_single_surface_smooths_a_sharp_local_fold():
    # The core reason this project moved from a per-cell patchwork to
    # one global surface: internal guide curves should be treated as
    # SOFT constraints, so a sharp local fold gets smoothed away
    # rather than reproduced -- confirmed by direct OCP probing
    # before relying on it (see step_export.py's module docstring).
    # This encodes that finding as a regression test: the fold's
    # amplitude must survive at only a small fraction in the surface.
    fold_center = 2.0
    fold_amplitude = 0.8

    def fold_bump(v):
        if abs(v - fold_center) > 0.3:
            return 0.0
        return fold_amplitude * np.exp(-((v - fold_center) ** 2) / (2 * 0.1 ** 2))

    interior_curves = []

    for x in np.linspace(0, 4, 5):
        ys = np.linspace(0, 4, 40)
        z = _dome_z(x, ys)
        if abs(x - fold_center) < 0.01:
            z = z + np.array([fold_bump(y) for y in ys])
        interior_curves.append(np.column_stack([np.full_like(ys, x), ys, z]))

    for y in np.linspace(0, 4, 5):
        xs = np.linspace(0, 4, 40)
        z = _dome_z(xs, y)
        if abs(y - fold_center) < 0.01:
            z = z + np.array([fold_bump(x) for x in xs])
        interior_curves.append(np.column_stack([xs, np.full_like(xs, y), z]))

    boundary, _ = _synthetic_patch()

    face = build_single_surface(boundary, interior_curves, smoothing_tolerance_mm=0.05)

    from OCP.gp import gp_Pnt
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.BRep import BRep_Tool

    surf = BRep_Tool.Surface_s(face)
    target = gp_Pnt(fold_center, fold_center, _dome_z(fold_center, fold_center))
    projector = GeomAPI_ProjectPointOnSurf(target, surf)
    nearest = projector.NearestPoint()

    fold_retained = abs(nearest.Z() - _dome_z(fold_center, fold_center))

    # Well under half the fold's own amplitude survives -- the sharp
    # local feature is smoothed away, not reproduced.
    assert fold_retained < fold_amplitude * 0.5


def test_build_step_surfaces_retries_with_a_looser_tolerance(tmp_path, monkeypatch):
    # A real, complex shape can be too much for a single low-degree
    # surface to fit at the user's requested tolerance without
    # folding over itself, even with a clean boundary loop -- a
    # looser tolerance gives OpenCASCADE more room to find a valid
    # fit. Rather than making the user manually re-run the whole
    # pipeline with a bigger number, build_step_surfaces retries a
    # few times with a progressively looser tolerance itself. Mocks
    # build_single_surface directly (real geometry that reliably
    # fails at one tolerance and succeeds at another isn't practical
    # to construct synthetically) to check the retry control flow
    # itself: fails twice, succeeds on the third, real attempt.
    import step_export

    calls = []

    real_build_single_surface = step_export.build_single_surface

    def flaky_build_single_surface(boundary, interior, tolerance):
        calls.append(tolerance)
        if len(calls) < 3:
            raise RuntimeError("simulated fill failure")
        return real_build_single_surface(boundary, interior, tolerance)

    monkeypatch.setattr(
        step_export, "build_single_surface", flaky_build_single_surface
    )

    boundary, interior_curves = _synthetic_patch()

    data = {
        "boundary_loop": boundary,
        "interior_curves": interior_curves,
        "smoothing_tolerance_mm": 0.3,
    }

    output_path = tmp_path / "surface.step"

    step_export.build_step_surfaces(data, str(output_path))

    assert output_path.exists()
    assert len(calls) == 3
    # Each retry loosens by RETRY_TOLERANCE_FACTOR from the original.
    assert calls[0] == pytest.approx(0.3)
    assert calls[1] > calls[0]
    assert calls[2] > calls[1]


def test_build_step_surfaces_gives_up_after_max_retries(monkeypatch):
    import step_export

    def always_fails(boundary, interior, tolerance):
        raise RuntimeError("simulated fill failure")

    monkeypatch.setattr(step_export, "build_single_surface", always_fails)

    boundary, interior_curves = _synthetic_patch()

    data = {
        "boundary_loop": boundary,
        "interior_curves": interior_curves,
        "smoothing_tolerance_mm": 0.3,
    }

    try:
        step_export.build_step_surfaces(data, "unused.step")
    except RuntimeError as exc:
        assert "retries" in str(exc)
    else:
        raise AssertionError("expected a RuntimeError after exhausting retries")
