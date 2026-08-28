import numpy as np
import pytest

from curve_utils import (
    build_simple_spline_curve,
    find_curve_crossings,
    fit_portion_adaptive,
    reconstruct_curve_piecewise,
    resample,
    smooth_curve,
    stitch_curve_fragments,
)


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
