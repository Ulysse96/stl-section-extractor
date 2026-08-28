"""
Pure curve-processing helpers used by section_stl.py.

This module has no tkinter / pyvista / vtk import, so it can be imported
and unit-tested directly (see tests/test_curve_utils.py) without opening
any file dialog or 3D viewer, unlike section_stl.py itself which runs its
whole pipeline top-to-bottom as soon as it is imported.
"""

import numpy as np
from scipy.interpolate import splprep, splev


# Every distance in section_stl.py -- the loaded mesh, the
# width/tolerance parameters, the exported DXF -- is in mm.
# UNIT_TO_MM converts whatever unit a source STL was actually
# authored in (some photogrammetry/scan tools export in metres,
# not mm) back to mm right after loading.
UNIT_TO_MM = {
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
}


def resample(points, n):

    if len(points) <= n:
        return points

    distances = np.zeros(
        len(points)
    )

    for i in range(
        1,
        len(points)
    ):

        distances[i] = (
            distances[i - 1]
            +
            np.linalg.norm(
                points[i]
                -
                points[i - 1]
            )
        )

    total_length = distances[-1]

    if total_length <= 0:
        return points

    targets = np.linspace(
        0,
        total_length,
        n
    )

    result = []

    for d in targets:

        idx = np.searchsorted(
            distances,
            d
        )

        if idx == 0:

            result.append(
                points[0]
            )

        elif idx >= len(points):

            result.append(
                points[-1]
            )

        else:

            d1 = distances[idx - 1]
            d2 = distances[idx]

            t = (
                d - d1
            ) / (
                d2 - d1
            )

            result.append(
                points[idx - 1]
                +
                t *
                (
                    points[idx]
                    -
                    points[idx - 1]
                )
            )

    return np.asarray(result)


def stitch_curve_fragments(curves, tolerance):
    """
    Merge open curve fragments whose endpoints are within
    `tolerance` of each other, regardless of orientation.
    Repeats until no more merges are possible.
    """

    fragments = [
        np.asarray(c)
        for c in curves
        if len(c) >= 2
    ]

    merged_any = True

    while merged_any:

        merged_any = False

        n = len(fragments)

        for i in range(n):

            if fragments[i] is None:
                continue

            a = fragments[i]

            for j in range(i + 1, n):

                if fragments[j] is None:
                    continue

                b = fragments[j]

                # Four possible ways two open fragments can join
                candidates = [
                    (np.linalg.norm(a[-1] - b[0]), "append", False),
                    (np.linalg.norm(a[-1] - b[-1]), "append", True),
                    (np.linalg.norm(a[0] - b[-1]), "prepend", False),
                    (np.linalg.norm(a[0] - b[0]), "prepend", True),
                ]

                dist, mode, reverse_b = min(
                    candidates,
                    key=lambda c: c[0]
                )

                if dist <= tolerance:

                    b_oriented = b[::-1] if reverse_b else b

                    if mode == "append":
                        fragments[i] = np.vstack([a, b_oriented])
                    else:
                        fragments[i] = np.vstack([b_oriented, a])

                    fragments[j] = None
                    merged_any = True
                    break

            if merged_any:
                break

    return [
        f
        for f in fragments
        if f is not None
    ]


def smooth_curve(points, iterations, factor=0.5, fixed_indices=None):
    """
    Iterative Laplacian smoothing of an open polyline.
    Endpoints are kept fixed so the curve still reaches the
    same start/end location (important for lofting). Each
    iteration nudges interior points toward the midpoint of
    their neighbours, removing zigzags from mesh noise and
    fold artefacts. `iterations` controls the aggressiveness:
    0 = no smoothing, higher = smoother / more simplified.

    `fixed_indices`, if given, is an iterable of extra point
    indices (beyond the two endpoints) that must never move --
    used for A x B intersection points, which must stay fixed
    through smoothing even though they sit in the interior of
    the curve.
    """

    if iterations <= 0 or len(points) < 3:
        return points

    pts = np.asarray(points, dtype=float).copy()

    pinned = np.zeros(len(pts), dtype=bool)
    pinned[0] = True
    pinned[-1] = True

    if fixed_indices:

        for idx in fixed_indices:

            if 0 <= idx < len(pts):
                pinned[idx] = True

    for _ in range(iterations):

        new_pts = pts.copy()

        new_pts[1:-1] = pts[1:-1] + factor * 0.5 * (
            (pts[:-2] - pts[1:-1])
            +
            (pts[2:] - pts[1:-1])
        )

        new_pts[pinned] = pts[pinned]

        pts = new_pts

    return pts


def fit_ideal_curve(points, strength, avg_edge_length, fixed_indices=None):
    """
    Fit a smoothing B-spline through `points` and resample it,
    replacing the original polyline with a single continuous
    "ideal" curve. `strength` is a dimensionless factor (0 =
    disabled, returns points unchanged): it is converted into
    the spline's smoothing factor `s`, scaled by the number of
    points and by the mesh's average edge length squared so
    that the same `strength` value behaves consistently
    regardless of object size or scan density.

    `fixed_indices`, if given, is an iterable of point indices
    (in the ORIGINAL `points` array) that must not move -- used
    for A x B intersection points. Since this function can drop
    a few near-duplicate points before fitting, indices can
    shift; the function returns the up-to-date indices as its
    second value so the caller can keep tracking them in later
    steps (e.g. piecewise reconstruction).

    Returns (fitted_points, updated_fixed_indices).
    """

    fixed_indices = list(fixed_indices) if fixed_indices else []

    if strength <= 0:
        return points, fixed_indices

    pts_full = np.asarray(points, dtype=float)

    # Drop consecutive (near-)duplicate points: splprep requires
    # a strictly increasing arc-length parametrisation.
    keep = [0]

    for i in range(1, len(pts_full)):

        if np.linalg.norm(pts_full[i] - pts_full[keep[-1]]) > 1e-9:
            keep.append(i)

    pts = pts_full[keep]

    n = len(pts)

    if n < 4:
        return points, fixed_indices

    keep_position = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(keep)
    }

    translated_fixed = []

    for idx in fixed_indices:

        if idx in keep_position:

            translated_fixed.append(keep_position[idx])

        else:

            nearest_old = min(keep, key=lambda k: abs(k - idx))

            translated_fixed.append(keep_position[nearest_old])

    # Parametrise by cumulative arc length (more robust than a
    # plain index-based parametrisation when point spacing is
    # uneven, e.g. after stitching two fragments together).
    seg_lengths = np.linalg.norm(
        np.diff(pts, axis=0),
        axis=1
    )

    cumulative = np.concatenate(
        [[0.0], np.cumsum(seg_lengths)]
    )

    total_length = cumulative[-1]

    if total_length <= 0:
        return points, fixed_indices

    u_param = cumulative / total_length

    smoothing_factor = (
        strength
        *
        n
        *
        (avg_edge_length ** 2)
        /
        200.0
    )

    try:

        tck, _ = splprep(
            [pts[:, 0], pts[:, 1], pts[:, 2]],
            u=u_param,
            s=smoothing_factor,
            k=min(3, n - 1)
        )

    except Exception as exc:

        print(
            f"    Warning: ideal curve fit failed "
            f"({exc}), keeping smoothed curve."
        )

        return points, fixed_indices

    u_new = np.linspace(0, 1, n)

    x_new, y_new, z_new = splev(u_new, tck)

    fitted = np.stack(
        [x_new, y_new, z_new],
        axis=1
    )

    # A smoothing spline (s > 0) approximates rather than
    # interpolates, so it can drift slightly at the very ends;
    # anchor those back to their original location. A x B
    # intersection points (also passed in via fixed_indices) are
    # deliberately NOT snapped back here -- forcing them exactly
    # is what made curves look kinked/twisted where several
    # intersections sit close together. They are only tracked
    # (their index is returned, updated for any point this
    # function dropped) so the final reconciliation step can
    # re-attach the two curves that share each point once
    # everything has settled.
    fitted[0] = pts[0]
    fitted[-1] = pts[-1]

    return fitted, translated_fixed


def _polynomial_r2(points_local, fitted_local):
    """
    Multivariate R^2 for a 3D point set: 1 - SSres/SStot, where
    both sums of squares use the squared Euclidean distance
    (so all three coordinates are accounted for jointly, not
    coordinate by coordinate).
    """

    residuals = points_local - fitted_local

    ss_res = float(np.sum(residuals ** 2))

    centroid = points_local.mean(axis=0)

    ss_tot = float(np.sum((points_local - centroid) ** 2))

    if ss_tot <= 1e-12:
        return 1.0 if ss_res <= 1e-12 else 0.0

    return 1.0 - ss_res / ss_tot


def _constrained_polyfit(u_local, values, degree, u_ends, value_ends):
    """
    Least-squares polynomial fit of `values` vs `u_local`,
    degree `degree`, subject to the fit passing EXACTLY through
    (u_ends[i], value_ends[i]) for each boundary constraint
    (here: the two portion endpoints, i.e. the clicked split
    points / curve ends). Solved via Lagrange multipliers
    (KKT system) so the constraint is exact, not approximate.
    Coefficients are returned in ascending power order.
    """

    A = np.vander(u_local, degree + 1, increasing=True)
    B = np.vander(np.asarray(u_ends), degree + 1, increasing=True)

    k = B.shape[0]

    AtA = A.T @ A

    top = np.hstack([AtA, B.T])
    bottom = np.hstack([B, np.zeros((k, k))])

    kkt = np.vstack([top, bottom])

    rhs = np.concatenate(
        [A.T @ values, np.asarray(value_ends, dtype=float)]
    )

    solution, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)

    return solution[:degree + 1]


def fit_portion_adaptive(points_local, max_degree, r2_target):
    """
    Fits points_local (arc-length parametrised) with the lowest
    polynomial degree that reaches r2_target, capped at
    max_degree and at (n_points - 1). The fit is CONSTRAINED to
    pass exactly through the portion's first and last point
    (these are the points the user clicked to define the
    portion boundary, or the curve's true ends): they act as
    fixed boundary conditions and are never moved by the
    reconstruction, and this also guarantees adjacent portions
    meet exactly with no kink.

    A degree-0 polynomial (a single constant point) cannot
    satisfy two different boundary values in general, so the
    search starts at degree 1 (a straight line between the two
    fixed endpoints) rather than degree 0.

    Returns (fitted_points, degree_used, r2_achieved).
    """

    pts = np.asarray(points_local, dtype=float)

    n_local = len(pts)

    if n_local < 2:
        return pts, 0, 1.0

    seg_lengths = np.linalg.norm(
        np.diff(pts, axis=0),
        axis=1
    )

    cumulative = np.concatenate(
        [[0.0], np.cumsum(seg_lengths)]
    )

    total_length = cumulative[-1]

    if total_length <= 0:
        return pts, 0, 1.0

    u_local = cumulative / total_length

    u_ends = [u_local[0], u_local[-1]]

    # Degree 1 is the minimum that can satisfy both fixed
    # endpoints; it is also the maximum degree that makes sense
    # once only 2 points are available.
    max_feasible_degree = max(
        1,
        min(max_degree, n_local - 1)
    )

    best_fitted = None
    best_degree = 1
    best_r2 = -np.inf

    for degree in range(1, max_feasible_degree + 1):

        value_ends_per_axis = [
            [pts[0, k], pts[-1, k]]
            for k in range(3)
        ]

        coeffs = [
            _constrained_polyfit(
                u_local,
                pts[:, k],
                degree,
                u_ends,
                value_ends_per_axis[k]
            )
            for k in range(3)
        ]

        fitted = np.stack(
            [
                np.polynomial.polynomial.polyval(
                    u_local,
                    coeffs[k]
                )
                for k in range(3)
            ],
            axis=1
        )

        # The endpoints are constrained analytically, but
        # floating point solves can leave a tiny residual;
        # snap them back to be exactly the clicked points.
        fitted[0] = pts[0]
        fitted[-1] = pts[-1]

        r2 = _polynomial_r2(pts, fitted)

        best_fitted, best_degree, best_r2 = fitted, degree, r2

        if r2 >= r2_target:
            break

    # Safety net against Runge-type oscillation with sparse,
    # unevenly-spaced points and a high degree: clamp to a
    # generous box around this portion's own data (the fixed
    # endpoints are re-applied afterwards, so they are never
    # affected by the clamp either).
    box_min = pts.min(axis=0)
    box_max = pts.max(axis=0)
    box_extent = box_max - box_min

    margin = 0.5

    best_fitted = np.clip(
        best_fitted,
        box_min - margin * box_extent,
        box_max + margin * box_extent
    )

    best_fitted[0] = pts[0]
    best_fitted[-1] = pts[-1]

    return best_fitted, best_degree, best_r2


def reconstruct_curve_piecewise(
    points,
    split_indices,
    max_degree,
    r2_target
):
    """
    Splits 'points' at 'split_indices' (any order, duplicates
    and out-of-range values are ignored) into contiguous
    portions and replaces each with its own adaptively-fitted
    polynomial (see fit_portion_adaptive). Returns
    (reconstructed_points, portions_info) where portions_info is
    a list of (start_index, end_index, degree_used, r2_achieved)
    for reporting to the user. If there are no usable split
    points, the curve is returned unchanged.
    """

    pts = np.asarray(points, dtype=float)

    n = len(pts)

    interior = sorted(
        {
            int(s)
            for s in split_indices
            if 0 < int(s) < n - 1
        }
    )

    # boundaries always contains at least [0, n - 1], so this is
    # the only case in which no *interior* split point survived
    # filtering -- that's what "no usable split points" means.
    if not interior:
        return pts, []

    boundaries = [0] + interior + [n - 1]

    chunks = []
    portions_info = []

    n_portions = len(boundaries) - 1

    for i in range(n_portions):

        lo = boundaries[i]
        hi = boundaries[i + 1]

        if hi <= lo:
            continue

        local_pts = pts[lo:hi + 1]

        fitted_local, degree_used, r2_achieved = fit_portion_adaptive(
            local_pts,
            max_degree,
            r2_target
        )

        portions_info.append(
            (lo, hi, degree_used, r2_achieved)
        )

        if i == n_portions - 1:
            chunks.append(fitted_local)
        else:
            chunks.append(fitted_local[:-1])

    if not chunks:
        return pts, []

    reconstructed = np.vstack(chunks)

    return reconstructed, portions_info


def find_curve_crossings(
    pts_a,
    pts_b,
    max_count,
    max_gap,
    suppression_fraction=0.05
):
    """
    Finds up to `max_count` distinct closest-approach point pairs
    between two curves via greedy local-minimum search: the
    global closest pair is taken, a neighbourhood around it (on
    both curves) is masked out so the same crossing isn't found
    again under a nearby index, then the process repeats. Stops
    early once the best remaining distance exceeds `max_gap`
    (a large "closest approach" is not a genuine crossing, just
    whatever happened to be nearest).
    Returns a list of (idx_a, idx_b, distance).
    """

    dists = np.linalg.norm(
        pts_a[:, None, :] - pts_b[None, :, :],
        axis=2
    )

    n_a, n_b = dists.shape

    suppress_a = max(1, int(n_a * suppression_fraction))
    suppress_b = max(1, int(n_b * suppression_fraction))

    working = dists.copy()

    found = []

    for _ in range(max_count):

        idx_a, idx_b = np.unravel_index(
            np.argmin(working),
            working.shape
        )

        min_dist = working[idx_a, idx_b]

        if not np.isfinite(min_dist) or min_dist > max_gap:
            break

        found.append((idx_a, idx_b, min_dist))

        a_lo = max(0, idx_a - suppress_a)
        a_hi = min(n_a, idx_a + suppress_a + 1)

        b_lo = max(0, idx_b - suppress_b)
        b_hi = min(n_b, idx_b + suppress_b + 1)

        working[a_lo:a_hi, :] = np.inf
        working[:, b_lo:b_hi] = np.inf

    return found


def build_simple_spline_curve(curve_3d, intersection_points):
    """
    Returns a new point array: [curve start] + all points in
    `intersection_points`, ordered by their position along
    `curve_3d`, + [curve end]. This is deliberately minimal --
    exactly the two endpoints plus the shared crossings, nothing
    resampled or interpolated in between, so the DXF spline
    built from these fit_points passes through every one of them
    exactly.
    """

    pts = np.asarray(curve_3d, dtype=float)

    n = len(pts)

    if n < 2:
        return pts

    seg_lengths = np.linalg.norm(
        np.diff(pts, axis=0),
        axis=1
    )

    cumulative = np.concatenate(
        [[0.0], np.cumsum(seg_lengths)]
    )

    ordered = []

    for point in intersection_points:

        best_u = 0.0
        best_dist = np.inf

        for k in range(n - 1):

            a = pts[k]
            b = pts[k + 1]

            ab = b - a
            denom = np.dot(ab, ab)

            if denom < 1e-12:
                t = 0.0
            else:
                t = np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0)

            proj = a + t * ab

            dist = np.linalg.norm(point - proj)

            if dist < best_dist:
                best_dist = dist
                best_u = cumulative[k] + t * (
                    cumulative[k + 1] - cumulative[k]
                )

        ordered.append((best_u, point))

    ordered.sort(key=lambda entry: entry[0])

    result = (
        [pts[0]]
        +
        [point for _, point in ordered]
        +
        [pts[-1]]
    )

    return np.asarray(result, dtype=float)
