import numpy as np
import pytest

from curve_utils import (
    UNIT_TO_MM,
    _segments_intersect_2d,
    build_simple_spline_curve,
    collect_curve_endpoints,
    find_curve_crossings,
    fit_portion_adaptive,
    order_boundary_loop,
    reconstruct_curve_piecewise,
    resample,
    smooth_curve,
    split_boundary_and_curves_at_separator,
    split_into_panels,
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


def _has_self_intersection(ordered_points_2d):
    n = len(ordered_points_2d)

    for i in range(n):

        a, b = ordered_points_2d[i], ordered_points_2d[(i + 1) % n]

        for j in range(i + 2, n):

            if i == 0 and j == n - 1:
                continue  # adjacent (wrap-around) edges never "cross"

            c, d = ordered_points_2d[j], ordered_points_2d[(j + 1) % n]

            if _segments_intersect_2d(a, b, c, d):
                return True

    return False


def test_order_boundary_loop_handles_a_non_convex_outline():
    # A "U"/staple-shaped outline, whose centroid sits in the
    # concave notch -- exactly the case that broke this function's
    # earlier implementations on a real, non-convex scan (a hat with
    # an uneven brim): points on the two "prongs" of the U can have
    # similar angles from a centroid outside the shape (angle-sort),
    # or the notch's two inner corners can be mutually nearest at
    # some step and get connected straight across it (plain
    # nearest-neighbour, no uncrossing pass). The property that
    # actually matters -- and the one OpenCASCADE's filler needs --
    # isn't reproducing one specific "intended" polygon (several
    # valid simple ones can exist through the same point set), it's
    # that the result has NO self-crossing edges at all.
    # Slightly perturbed off a perfect grid (real scan boundary
    # points are never exactly collinear/on-grid) so no three points
    # are ever exactly collinear -- that degenerate case makes
    # "does this edge cross that edge" genuinely ambiguous at the
    # touching point, which isn't what this test is about.
    true_perimeter = np.array(
        [
            [0, 0, 0], [0.05, 3, 0], [1, 2.95, 0], [1, 1, 0],
            [2, 1.05, 0], [1.95, 3, 0], [3, 3.05, 0], [3, 0, 0],
        ],
        dtype=float,
    )

    scrambled_order = [5, 1, 7, 3, 0, 6, 2, 4]
    ordered = order_boundary_loop(true_perimeter[scrambled_order])

    # The loop is already planar (all z=0) -- use x, y directly as
    # the 2D projection for the crossing check.
    assert not _has_self_intersection(ordered[:, :2])


# --------------------------------------------------------------
# collect_curve_endpoints
# --------------------------------------------------------------

def test_collect_curve_endpoints_gathers_both_ends_of_every_curve():
    main_curves = {
        ("A", 1): np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=float),
        ("A", 2): np.array([[0, 2, 0], [1, 2, 0], [2, 2, 0]], dtype=float),
        ("B", 1): np.array([[1, 0, 0], [1, 1, 0], [1, 2, 0]], dtype=float),
    }

    endpoints = collect_curve_endpoints(main_curves)

    # 3 curves, 2 endpoints each.
    assert endpoints.shape == (6, 3)

    expected = {
        (0, 1, 0), (2, 1, 0),
        (0, 2, 0), (2, 2, 0),
        (1, 0, 0), (1, 2, 0),
    }
    actual = {tuple(p) for p in endpoints}
    assert actual == expected


def test_collect_curve_endpoints_skips_degenerate_curves():
    main_curves = {
        ("A", 1): np.array([[0, 1, 0], [1, 1, 0]], dtype=float),
        ("A", 2): np.array([[5, 5, 5]], dtype=float),  # single point, skipped
    }

    endpoints = collect_curve_endpoints(main_curves)

    assert endpoints.shape == (2, 3)


# --------------------------------------------------------------
# split_boundary_and_curves_at_separator
# --------------------------------------------------------------

def _square_boundary_loop():
    # A 4x4 square's perimeter, densely enough sampled that a
    # separator's two ends land on distinct boundary points -- stands
    # in for order_boundary_loop's real output on a scanned patch.
    boundary = []
    for x in np.linspace(0, 4, 9):
        boundary.append([x, 0, 0])
    for y in np.linspace(0, 4, 9)[1:]:
        boundary.append([4, y, 0])
    for x in np.linspace(4, 0, 9)[1:]:
        boundary.append([x, 4, 0])
    for y in np.linspace(4, 0, 9)[1:-1]:
        boundary.append([0, y, 0])
    return np.array(boundary, dtype=float)


def test_split_boundary_and_curves_at_separator_returns_two_regions():
    boundary = _square_boundary_loop()
    separator = np.array(
        [[2, 0, 0], [2, 1, 0], [2, 2, 0], [2, 3, 0], [2, 4, 0]], dtype=float
    )

    regions = split_boundary_and_curves_at_separator(boundary, {}, separator)

    assert len(regions) == 2

    for region in regions:
        assert "boundary_loop" in region
        assert "interior_curves" in region
        # Each region's own boundary is the whole square's perimeter
        # split roughly in half, plus the separator closing it back
        # up -- neither region should end up with (almost) everything.
        assert 5 < len(region["boundary_loop"]) < len(boundary) + len(separator)


def test_split_boundary_and_curves_at_separator_keeps_a_one_sided_curve_whole():
    boundary = _square_boundary_loop()
    separator = np.array(
        [[2, 0, 0], [2, 1, 0], [2, 2, 0], [2, 3, 0], [2, 4, 0]], dtype=float
    )

    # Entirely at x=0.5 -- well left of the x=2 separator.
    main_curves = {
        ("A", 1): np.array(
            [[0.5, 0, 0], [0.5, 2, 0], [0.5, 4, 0]], dtype=float
        ),
    }

    regions = split_boundary_and_curves_at_separator(
        boundary, main_curves, separator
    )

    curves_by_region = [r["interior_curves"] for r in regions]
    non_empty = [c for c in curves_by_region if c]

    # Only one region should have received the curve, whole (not cut).
    assert len(non_empty) == 1
    assert len(non_empty[0]) == 1
    assert len(non_empty[0][0]) == 3


def test_split_boundary_and_curves_at_separator_cuts_a_crossing_curve():
    boundary = _square_boundary_loop()
    separator = np.array(
        [[2, 0, 0], [2, 1, 0], [2, 2, 0], [2, 3, 0], [2, 4, 0]], dtype=float
    )

    # Runs straight across the separator at y=2, from x=0 to x=4.
    crossing_curve = np.array(
        [[0, 2, 0], [1, 2, 0], [2.5, 2, 0], [3, 2, 0], [4, 2, 0]],
        dtype=float,
    )
    main_curves = {("B", 1): crossing_curve}

    regions = split_boundary_and_curves_at_separator(
        boundary, main_curves, separator
    )

    all_pieces = [
        piece for r in regions for piece in r["interior_curves"]
    ]

    # Cut into exactly 2 pieces (one per side), together accounting
    # for every point of the original curve.
    assert len(all_pieces) == 2
    assert sum(len(piece) for piece in all_pieces) == len(crossing_curve)

    left_piece = min(all_pieces, key=lambda p: p[:, 0].mean())
    right_piece = max(all_pieces, key=lambda p: p[:, 0].mean())

    assert np.all(left_piece[:, 0] < 2)
    assert np.all(right_piece[:, 0] > 2)


def test_split_boundary_and_curves_at_separator_rejects_a_degenerate_separator():
    boundary = _square_boundary_loop()

    # Both ends snap to the same boundary point.
    separator = np.array([[2, 0, 0], [2, 0.1, 0], [2, 0, 0]], dtype=float)

    with pytest.raises(ValueError):
        split_boundary_and_curves_at_separator(boundary, {}, separator)


# --------------------------------------------------------------
# split_into_panels
# --------------------------------------------------------------

def _rectangle_boundary_loop(width, height):
    boundary = []
    for x in np.linspace(0, width, 3 * width + 1):
        boundary.append([x, 0, 0])
    for y in np.linspace(0, height, 2 * height + 1)[1:]:
        boundary.append([width, y, 0])
    for x in np.linspace(width, 0, 3 * width + 1)[1:]:
        boundary.append([x, height, 0])
    for y in np.linspace(height, 0, 2 * height + 1)[1:-1]:
        boundary.append([0, y, 0])
    return np.array(boundary, dtype=float)


def test_split_into_panels_with_no_separators_returns_one_panel():
    boundary = _rectangle_boundary_loop(6, 4)
    main_curves = {
        0: np.array([[3, 0, 0], [3, 2, 0], [3, 4, 0]], dtype=float),
    }

    panels, skipped = split_into_panels(boundary, main_curves, [])

    assert len(panels) == 1
    assert len(panels[0]["interior_curves"]) == 1
    assert skipped == []


def test_split_into_panels_applies_two_independent_separators():
    # A 6x4 rectangle cut into 3 vertical strips by two independent
    # separators (x=2 and x=4) -- neither separator shares an
    # endpoint with the other, standing in for "visor" and "back tab"
    # seams traced independently on the same patch.
    boundary = _rectangle_boundary_loop(6, 4)

    separator_1 = np.array([[2, 0, 0], [2, 2, 0], [2, 4, 0]], dtype=float)
    separator_2 = np.array([[4, 0, 0], [4, 2, 0], [4, 4, 0]], dtype=float)

    main_curves = {
        "left": np.array([[1, 0, 0], [1, 2, 0], [1, 4, 0]], dtype=float),
        "middle": np.array([[3, 0, 0], [3, 2, 0], [3, 4, 0]], dtype=float),
        "right": np.array([[5, 0, 0], [5, 2, 0], [5, 4, 0]], dtype=float),
    }

    panels, skipped = split_into_panels(
        boundary, main_curves, [separator_1, separator_2]
    )

    assert len(panels) == 3
    assert skipped == []

    # Each panel got exactly its own strip's curve, whole.
    x_means = sorted(
        panel["interior_curves"][0][:, 0].mean() for panel in panels
    )
    assert x_means == pytest.approx([1, 3, 5])

    for panel in panels:
        assert len(panel["interior_curves"]) == 1


def test_split_into_panels_skips_a_separator_crossing_two_panels():
    # One bad seam shouldn't discard every OTHER seam that worked --
    # it's skipped (and reported), not a hard failure for the whole
    # call. Regression test for a real case: the previous "raise on
    # the first bad seam" behaviour discarded ALL seams, including
    # ones that had split off perfectly good panels already.
    boundary = _rectangle_boundary_loop(6, 4)

    # First separator splits the rectangle at x=2 into two panels.
    separator_1 = np.array([[2, 0, 0], [2, 2, 0], [2, 4, 0]], dtype=float)

    # Second separator runs from x=1 (inside the first, smaller panel)
    # to x=5 (inside the second, larger panel) -- crosses between
    # panels rather than staying within one.
    separator_2 = np.array([[1, 0, 0], [3, 2, 0], [5, 4, 0]], dtype=float)

    panels, skipped = split_into_panels(
        boundary, {}, [separator_1, separator_2]
    )

    # The first (valid) separator still split the rectangle into 2.
    assert len(panels) == 2

    # The second (invalid) separator was skipped, at its own index,
    # with a reason -- not silently dropped, and not raised.
    assert len(skipped) == 1
    assert skipped[0][0] == 1
    assert "different existing panels" in skipped[0][1]


def test_split_into_panels_skips_a_seam_that_leaves_a_sliver():
    # Regression test for a real, more severe failure: several seams
    # traced close together left one boundary with only 3 points --
    # a razor-thin sliver that still "passed" downstream checks (few
    # points, easy to stay technically close to) while producing a
    # badly twisted, physically nonsensical surface. Rejected at the
    # source now (split_boundary_and_curves_at_separator's own
    # MIN_ARC_BOUNDARY_POINTS check) and reported as skipped here,
    # rather than silently handed off to be fit into garbage.
    boundary = _rectangle_boundary_loop(6, 4)

    # Snaps to two adjacent boundary points near the corner (0, 0)
    # and (0.5, 0) -- one side of the split arc is almost nothing.
    sliver_separator = np.array(
        [[0, 0, 0], [0.2, 1, 0], [0.5, 0, 0]], dtype=float
    )

    panels, skipped = split_into_panels(boundary, {}, [sliver_separator])

    assert len(panels) == 1
    assert len(skipped) == 1
    assert skipped[0][0] == 0

