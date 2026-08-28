import numpy as np
import pytest

from curve_utils import (
    UNIT_TO_MM,
    build_simple_spline_curve,
    build_surface_grid,
    collect_curve_endpoints,
    find_curve_crossings,
    fit_portion_adaptive,
    order_boundary_loop,
    reconstruct_curve_piecewise,
    resample,
    segment_curve_at_indices,
    smooth_curve,
    stitch_curve_fragments,
)


# --------------------------------------------------------------
# UNIT_TO_MM
# --------------------------------------------------------------

def test_unit_to_mm_defaults_to_identity_for_millimetres():
    assert UNIT_TO_MM["mm"] == 1.0


def test_unit_to_mm_metres_is_the_thousand_factor():
    assert UNIT_TO_MM["m"] == 1000.0


# --------------------------------------------------------------
# resample
# --------------------------------------------------------------

def test_resample_keeps_endpoints_and_count():
    points = np.array([[i, 0, 0] for i in range(10)], dtype=float)

    result = resample(points, 5)

    assert len(result) == 5
    np.testing.assert_allclose(result[0], points[0])
    np.testing.assert_allclose(result[-1], points[-1])


def test_resample_never_upsamples_beyond_the_input_count():
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [10, 0, 0]], dtype=float)

    result = resample(points, 20)

    np.testing.assert_array_equal(result, points)


def test_resample_returns_unchanged_when_already_short_enough():
    points = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)

    result = resample(points, 5)

    np.testing.assert_array_equal(result, points)


def test_resample_spacing_is_uniform_along_a_straight_line():
    points = np.array([[i, 0, 0] for i in range(11)], dtype=float)

    result = resample(points, 3)

    np.testing.assert_allclose(result, [[0, 0, 0], [5, 0, 0], [10, 0, 0]])


# --------------------------------------------------------------
# stitch_curve_fragments
# --------------------------------------------------------------

def test_stitch_merges_fragments_within_tolerance():
    fragment_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    fragment_b = np.array([[1.05, 0, 0], [2, 0, 0]], dtype=float)

    merged = stitch_curve_fragments([fragment_a, fragment_b], tolerance=0.1)

    assert len(merged) == 1
    assert len(merged[0]) == 4


def test_stitch_leaves_far_apart_fragments_separate():
    fragment_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    fragment_b = np.array([[100, 0, 0], [101, 0, 0]], dtype=float)

    merged = stitch_curve_fragments([fragment_a, fragment_b], tolerance=0.1)

    assert len(merged) == 2


def test_stitch_handles_reversed_orientation():
    fragment_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    # b's endpoint nearest to a's endpoint is its *first* point,
    # reversed relative to a's direction of travel
    fragment_b = np.array([[2, 0, 0], [1.02, 0, 0]], dtype=float)

    merged = stitch_curve_fragments([fragment_a, fragment_b], tolerance=0.1)

    assert len(merged) == 1
    np.testing.assert_allclose(merged[0][-1], [2, 0, 0])


# --------------------------------------------------------------
# smooth_curve
# --------------------------------------------------------------

def test_smooth_curve_keeps_endpoints_fixed():
    points = np.array(
        [[0, 0, 0], [1, 5, 0], [2, -5, 0], [3, 5, 0], [4, 0, 0]], dtype=float
    )

    smoothed = smooth_curve(points, iterations=10)

    np.testing.assert_allclose(smoothed[0], points[0])
    np.testing.assert_allclose(smoothed[-1], points[-1])


def test_smooth_curve_reduces_zigzag_amplitude():
    points = np.array(
        [[0, 0, 0], [1, 5, 0], [2, -5, 0], [3, 5, 0], [4, 0, 0]], dtype=float
    )

    smoothed = smooth_curve(points, iterations=20)

    assert np.abs(smoothed[2, 1]) < np.abs(points[2, 1])


def test_smooth_curve_zero_iterations_is_a_no_op():
    points = np.array([[0, 0, 0], [1, 5, 0], [2, 0, 0]], dtype=float)

    result = smooth_curve(points, iterations=0)

    np.testing.assert_array_equal(result, points)


def test_smooth_curve_respects_fixed_interior_indices():
    points = np.array(
        [[0, 0, 0], [1, 5, 0], [2, -5, 0], [3, 5, 0], [4, 0, 0]], dtype=float
    )

    smoothed = smooth_curve(points, iterations=10, fixed_indices=[2])

    np.testing.assert_allclose(smoothed[2], points[2])


# --------------------------------------------------------------
# fit_portion_adaptive / reconstruct_curve_piecewise
# --------------------------------------------------------------

def test_fit_portion_adaptive_hits_target_r2_on_a_straight_line():
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)

    fitted, degree, r2 = fit_portion_adaptive(points, max_degree=5, r2_target=0.99)

    assert degree == 1
    assert r2 == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(fitted[0], points[0])
    np.testing.assert_allclose(fitted[-1], points[-1])


def test_fit_portion_adaptive_never_exceeds_max_degree():
    rng = np.random.default_rng(0)
    noise = rng.normal(scale=2.0, size=(15, 3))
    points = np.column_stack([np.arange(15)] * 3).astype(float) + noise

    _, degree, _ = fit_portion_adaptive(points, max_degree=3, r2_target=0.999999)

    assert degree <= 3


def test_reconstruct_curve_piecewise_matches_at_split_boundary():
    points = np.array(
        [[i, 0, 0] for i in range(10)]
        + [[9 + i, 5, 0] for i in range(1, 5)],
        dtype=float,
    )

    reconstructed, portions = reconstruct_curve_piecewise(
        points, split_indices=[9], max_degree=4, r2_target=0.98
    )

    assert len(portions) == 2
    # No gap/duplication introduced at the split: consecutive
    # portions must share their boundary point exactly once.
    assert len(reconstructed) == len(points)


def test_reconstruct_curve_piecewise_without_splits_returns_input_unchanged():
    points = np.array([[0, 0, 0], [1, 1, 0], [2, 0, 0]], dtype=float)

    reconstructed, portions = reconstruct_curve_piecewise(
        points, split_indices=[], max_degree=4, r2_target=0.98
    )

    np.testing.assert_array_equal(reconstructed, points)
    assert portions == []


# --------------------------------------------------------------
# find_curve_crossings
# --------------------------------------------------------------

def test_find_curve_crossings_detects_a_single_crossing():
    t = np.linspace(-1, 1, 50)
    curve_a = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    curve_b = np.column_stack([np.zeros_like(t), t, np.zeros_like(t)])

    crossings = find_curve_crossings(curve_a, curve_b, max_count=5, max_gap=0.1)

    assert len(crossings) == 1

    idx_a, idx_b, gap = crossings[0]

    np.testing.assert_allclose(curve_a[idx_a][:2], [0, 0], atol=0.05)
    np.testing.assert_allclose(curve_b[idx_b][:2], [0, 0], atol=0.05)
    assert gap < 0.1


def test_find_curve_crossings_returns_nothing_when_curves_are_far_apart():
    t = np.linspace(0, 1, 20)
    curve_a = np.column_stack([t, np.zeros_like(t), np.zeros_like(t)])
    curve_b = curve_a + np.array([0, 100, 0])

    crossings = find_curve_crossings(curve_a, curve_b, max_count=5, max_gap=1.0)

    assert crossings == []


# --------------------------------------------------------------
# build_simple_spline_curve
# --------------------------------------------------------------

def test_build_simple_spline_curve_orders_points_along_the_curve():
    curve = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
    # Given out of order relative to their position along the curve
    intersections = [np.array([7.0, 0, 0]), np.array([3.0, 0, 0])]

    result = build_simple_spline_curve(curve, intersections)

    np.testing.assert_allclose(
        result, [[0, 0, 0], [3, 0, 0], [7, 0, 0], [10, 0, 0]]
    )


def test_build_simple_spline_curve_with_no_intersections_returns_endpoints():
    curve = np.array([[0, 0, 0], [5, 0, 0], [10, 0, 0]], dtype=float)

    result = build_simple_spline_curve(curve, [])

    np.testing.assert_allclose(result, [[0, 0, 0], [10, 0, 0]])


# --------------------------------------------------------------
# segment_curve_at_indices
# --------------------------------------------------------------

def test_segment_curve_at_indices_splits_into_expected_pieces():
    points = np.array([[i, 0, 0] for i in range(10)], dtype=float)

    segments = segment_curve_at_indices(points, [3, 6])

    assert len(segments) == 3
    np.testing.assert_allclose(segments[0], points[0:4])
    np.testing.assert_allclose(segments[1], points[3:7])
    np.testing.assert_allclose(segments[2], points[6:10])


def test_segment_curve_at_indices_segments_share_boundary_points():
    points = np.array([[i, 0, 0] for i in range(8)], dtype=float)

    segments = segment_curve_at_indices(points, [2, 5])

    for a, b in zip(segments, segments[1:]):
        np.testing.assert_allclose(a[-1], b[0])


def test_segment_curve_at_indices_ignores_duplicates_and_out_of_range():
    points = np.array([[i, 0, 0] for i in range(6)], dtype=float)

    segments = segment_curve_at_indices(points, [3, 3, -1, 99])

    assert len(segments) == 2
    np.testing.assert_allclose(segments[0], points[0:4])
    np.testing.assert_allclose(segments[1], points[3:6])


def test_segment_curve_at_indices_with_no_cuts_returns_whole_curve():
    points = np.array([[i, 0, 0] for i in range(5)], dtype=float)

    segments = segment_curve_at_indices(points, [])

    assert len(segments) == 1
    np.testing.assert_allclose(segments[0], points)


# --------------------------------------------------------------
# order_boundary_loop
# --------------------------------------------------------------

def test_order_boundary_loop_orders_a_square_going_around():
    # A unit square in the XY plane, given in a scrambled order.
    square = {
        "bottom_left": [0, 0, 0],
        "top_right": [1, 1, 0],
        "bottom_right": [1, 0, 0],
        "top_left": [0, 1, 0],
    }
    scrambled = np.array(
        [
            square["top_right"],
            square["bottom_left"],
            square["top_left"],
            square["bottom_right"],
        ],
        dtype=float,
    )

    ordered = order_boundary_loop(scrambled)

    # Walking the returned order must trace the square's perimeter:
    # every consecutive pair (wrapping around) is a unit edge, never
    # the longer diagonal (sqrt(2)).
    edge_lengths = np.linalg.norm(
        np.roll(ordered, -1, axis=0) - ordered, axis=1
    )
    np.testing.assert_allclose(edge_lengths, [1.0, 1.0, 1.0, 1.0])


def test_order_boundary_loop_keeps_the_same_set_of_points():
    points = np.array(
        [[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, -1, 0]],
        dtype=float,
    )

    ordered = order_boundary_loop(points)

    assert ordered.shape == points.shape
    # Same rows, possibly reordered.
    original_sorted = points[np.lexsort(points.T)]
    ordered_sorted = ordered[np.lexsort(ordered.T)]
    np.testing.assert_allclose(ordered_sorted, original_sorted)


def test_order_boundary_loop_works_regardless_of_the_patch_orientation():
    # Same square, but tilted out of any coordinate plane -- the PCA
    # basis must still recover a consistent perimeter walk.
    square_2d = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float
    )
    rotation = np.array(
        [
            [0.8, -0.5, 0.3],
            [0.5, 0.85, -0.2],
            [0.1, 0.2, 0.97],
        ]
    )
    tilted = square_2d @ rotation[:2, :] + np.array([5, -3, 2])

    scrambled_order = [2, 0, 3, 1]
    ordered = order_boundary_loop(tilted[scrambled_order])

    edge_lengths = np.linalg.norm(
        np.roll(ordered, -1, axis=0) - ordered, axis=1
    )
    assert np.max(edge_lengths) == pytest.approx(np.min(edge_lengths), rel=0.3)


# --------------------------------------------------------------
# build_surface_grid
# --------------------------------------------------------------
#
# A flat 3x3 lattice: A curves are horizontal lines y = 1, 2, 3
# each sampled at x = 0..4; B curves are vertical lines x = 1, 2, 3
# each sampled at y = 0..4. Sampling exactly on integer coordinates
# means each crossing sits exactly on a curve point, so the
# expected idx_a / idx_b are known exactly and every geometric
# assertion below can be exact rather than approximate.

def _make_lattice_grid_inputs():
    all_section_data = []

    for i in (1, 2, 3):
        points = np.array([[x, i, 0] for x in range(5)], dtype=float)
        all_section_data.append(
            {
                "direction": "A",
                "number": i,
                "main_curve_index": 0,
                "curves": [{"points_3d": points}],
            }
        )

    for j in (1, 2, 3):
        points = np.array([[j, y, 0] for y in range(5)], dtype=float)
        all_section_data.append(
            {
                "direction": "B",
                "number": j,
                "main_curve_index": 0,
                "curves": [{"points_3d": points}],
            }
        )

    found_intersections = []

    for i in (1, 2, 3):
        for j in (1, 2, 3):
            found_intersections.append(
                {
                    "pair_id": (i, j, 0),
                    "point": np.array([j, i, 0], dtype=float),
                    "gap": 0.0,
                    "idx_a": j,
                    "idx_b": i,
                }
            )

    return all_section_data, found_intersections


def test_build_surface_grid_produces_the_expected_interior_cells():
    all_section_data, found_intersections = _make_lattice_grid_inputs()

    grid = build_surface_grid(all_section_data, found_intersections)

    # A 3x3 node lattice has a 2x2 grid of interior cells.
    assert len(grid["cells"]) == 4

    cells_by_corner = {
        (c["a_i"], c["b_j"]): c for c in grid["cells"]
    }
    assert set(cells_by_corner) == {(1, 1), (1, 2), (2, 1), (2, 2)}

    cell = cells_by_corner[(1, 1)]
    assert cell["a_i_next"] == 2
    assert cell["b_j_next"] == 2

    np.testing.assert_allclose(cell["edge_a_lo"], [[1, 1, 0], [2, 1, 0]])
    np.testing.assert_allclose(cell["edge_a_hi"], [[1, 2, 0], [2, 2, 0]])
    np.testing.assert_allclose(cell["edge_b_lo"], [[1, 1, 0], [1, 2, 0]])
    np.testing.assert_allclose(cell["edge_b_hi"], [[2, 1, 0], [2, 2, 0]])


def test_build_surface_grid_adjacent_cells_share_exact_edge_points():
    all_section_data, found_intersections = _make_lattice_grid_inputs()

    grid = build_surface_grid(all_section_data, found_intersections)

    cells_by_corner = {
        (c["a_i"], c["b_j"]): c for c in grid["cells"]
    }

    left = cells_by_corner[(1, 1)]
    right = cells_by_corner[(2, 1)]

    # left's "hi" edge (at a=2) must be identical to right's "lo"
    # edge (also at a=2) -- same physical boundary between the two
    # cells, computed independently from curve A_2's own segments.
    np.testing.assert_allclose(left["edge_a_hi"], right["edge_a_lo"])


def test_build_surface_grid_boundary_edges_reach_the_curve_endpoints():
    all_section_data, found_intersections = _make_lattice_grid_inputs()

    grid = build_surface_grid(all_section_data, found_intersections)

    start = grid["boundary_edges"][("A", 1, "start")]
    end = grid["boundary_edges"][("A", 1, "end")]

    np.testing.assert_allclose(start, [[0, 1, 0], [1, 1, 0]])
    np.testing.assert_allclose(end, [[3, 1, 0], [4, 1, 0]])


def test_build_surface_grid_handles_two_crossings_at_the_same_curve_index():
    # Regression test: on a real (noisy/self-crossing) scan, two
    # different crossings from the other family can land on the
    # exact same nearest point index of a curve. build_surface_grid
    # must not crash (it used to: IndexError in the snapping loop,
    # since segment_curve_at_indices silently de-duplicates that
    # index into a single boundary while the un-deduplicated hit
    # list still expected one more segment than actually existed).
    all_section_data, found_intersections = _make_lattice_grid_inputs()

    # A_2 x B_1 and A_2 x B_3 both snapped to the same point (index 2
    # along A_2, i.e. x=2) instead of their own distinct x=1 / x=3.
    for entry in found_intersections:
        i, j, k = entry["pair_id"]
        if i == 2 and j in (1, 3):
            entry["idx_a"] = 2
            entry["point"] = np.array([2, 2, 0], dtype=float)

    grid = build_surface_grid(all_section_data, found_intersections)

    # Must not raise, and must still produce a usable (if reduced)
    # grid rather than silently corrupting data.
    assert isinstance(grid["cells"], list)


def test_build_surface_grid_handles_a_crossing_at_the_curve_endpoint():
    # Regression test for the same crash, other trigger: a crossing
    # landing exactly on the curve's own open endpoint (index 0 or
    # n - 1), which segment_curve_at_indices treats as "already a
    # boundary" rather than a new interior split.
    all_section_data, found_intersections = _make_lattice_grid_inputs()

    for entry in found_intersections:
        i, j, k = entry["pair_id"]
        if i == 1 and j == 1:
            entry["idx_a"] = 0  # A_1's own start point, x=0

    grid = build_surface_grid(all_section_data, found_intersections)

    assert isinstance(grid["cells"], list)


def test_collect_curve_endpoints_gathers_both_ends_of_every_curve():
    all_section_data, found_intersections = _make_lattice_grid_inputs()

    grid = build_surface_grid(all_section_data, found_intersections)

    endpoints = collect_curve_endpoints(grid["main_curves"])

    # 6 curves (3 A + 3 B), 2 endpoints each.
    assert endpoints.shape == (12, 3)
