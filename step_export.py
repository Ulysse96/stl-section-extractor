"""
Builds ONE continuous surface reconstruction from the scanned patch's
boundary loop and its A x B section curves, and writes it to STEP.

This module needs cadquery's OCP (OpenCASCADE) bindings, which have no
Windows wheels for Python 3.13/3.14 -- see the project README for the
dedicated .venv312 this runs in. section_stl.py itself keeps running on
whatever Python already has pyvista/vtk working, and invokes this
module as a SEPARATE PROCESS in .venv312, handing it the curve data
through a pickle file -- so the interactive GUI pipeline and the
CAD-kernel-dependent surface export never have to share one Python
process (this also happens to route around a Windows Smart App Control
policy on the original dev machine that blocks vtk specifically inside
.venv312, see README).

Run directly: python step_export.py <curves.pickle> <output.step>

History: the first version of this module built one small Coons-style
patch per grid cell (see curve_utils.build_surface_grid) and sewed them
into a shell. On a real scan that produced a visibly faceted result,
occasional wild "spike" patches, and faces SolidWorks' own import
diagnostics kept flagging as invalid even after this module's own
checks passed them. Rebuilt as a SINGLE surface instead: one call to
OpenCASCADE's free-form filling (BRepOffsetAPI_MakeFilling), with the
boundary loop as the bounding wire and every A/B curve as an internal
guide curve. Probed directly against OCP before relying on it here:
this scales fine (a realistic 22-curve, 500-points-each network plus a
~400-point boundary builds in ~0.02s), and -- relevant to this
project's actual goal of flattening the result in Wrapstyler --
internal guide curves are inherently treated as SOFT constraints by
this algorithm: a sharp local fold's amplitude was retained at only
1-3% in the resulting surface even at aggressive settings, while the
overall shape away from that fold stayed close to the true curve
network. In other words this naturally smooths away small local
wrinkles while still following the scan's general shape, without
needing the harder-to-tune per-patch approach.
"""

import pickle
import sys

import numpy as np

from OCP.gp import gp_Pnt
from OCP.TColgp import TColgp_Array1OfPnt
from OCP.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_ProjectPointOnSurf
from OCP.GeomAbs import GeomAbs_C0, GeomAbs_Shape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Sewing
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.STEPControl import STEPControl_Writer, STEPControl_StepModelType
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_ReturnStatus


# How far ANY point actually ON the fitted surface may sit from the
# nearest real input point (boundary or guide curve), as a multiple
# of max_section_spacing_mm (the LARGER of the two distances between
# adjacent parallel cutting planes -- the data's own natural local
# resolution) -- checked BOTH ways (surface point -> nearest data
# point, and data point -> nearest surface point), so a real gap
# where the fit simply failed to cover part of the patch is caught
# too, not just a local bulge. BRepCheck_Analyzer only catches
# topological invalidity (e.g. self-intersection); it does not catch
# a surface that stays "valid" while sagging, bulging, or leaving a
# hole, far past where its own guide curves actually are.
#
# Getting the actual sample points right took three attempts:
#
# 1. Comparing the surface's own TRIMMED bounding box (BRepBndLib.
#    AddOptimal_s) to a multiple of the whole object's diagonal, and
#    later to a multiple of the local grid spacing. Both are a single
#    GLOBAL number for the whole surface, and missed a real, confirmed
#    failure -- a fitted surface whose control points and overall
#    bbox looked plausible in aggregate, but which locally sagged/left
#    a gap that a global check averages away.
# 2. Sampling the surface directly on a (u, v) grid across its own
#    parameter domain, keeping only points BRepTopAdaptor_FClass2d
#    classified as inside the trim. Fixed the "global" problem, but
#    turned out to be unreliable on exactly the kind of badly-behaved
#    surface this check exists to catch: confirmed directly against a
#    real fit whose extreme, distorted parametrization made this
#    sampling disagree wildly with the actual trimmed geometry
#    (AddOptimal_s said the real surface only spanned Z -66.7 to
#    -8.5mm on that same face; UV-grid sampling said -106 to +110mm).
#    With a pathological fit, the parametrization itself can't be
#    trusted to say what is or isn't "inside" the face.
# 3. What the "surface -> data" direction uses now: BRepMesh_
#    IncrementalMesh's own triangulation of the face, read back via
#    BRep_Tool.Triangulation_s (with isRelative=True -- an absolute
#    deflection made Triangulation_s silently come back None below,
#    IsDone() still True, at every value tried from 0.001 to 1.0;
#    confirmed a size-relative deflection fixes it). This is the same
#    tessellation any real STEP consumer (a viewer, or Wrapstyler)
#    effectively builds to work with the face at all, and sidesteps
#    parametrization entirely -- it only sees triangles that end up
#    covering the real, trimmed 3D shape.
#
#    The reverse "data -> surface" direction can't reuse those same
#    mesh samples, though: pushing the mesh fine enough to avoid gaps
#    between vertices reading as false "uncovered" data points isn't
#    reliable in this environment (mirrors point 2's lesson: sampling
#    density becomes the thing under test instead of the surface).
#    Confirmed directly -- even a good, correct fit showed an apparent
#    2.05mm gap against a 2.0mm threshold from mesh sparseness alone.
#    That direction instead projects each data point onto the surface
#    with GeomAPI_ProjectPointOnSurf (the same approach already used
#    by the "smooths a sharp fold" test below) for the TRUE nearest
#    point on the continuous surface, independent of mesh density.
#
# Deliberately avoids scipy (not just numpy) for the "surface -> data"
# nearest-neighbour search: this project's .venv312 has hit Windows
# Smart App Control blocking scipy's compiled extensions before (see
# README), so this stays plain numpy, chunked to bound memory use.
MAX_SURFACE_DEVIATION_FACTOR = 2.0
MESH_LINEAR_DEFLECTION_FACTOR = 0.25
_NEAREST_NEIGHBOUR_CHUNK = 200


def _surface_deviates_from_data(
    face, boundary_points, interior_curves, max_section_spacing_mm
):

    all_points = [np.asarray(boundary_points, dtype=float)]

    for curve_points in interior_curves:

        points = np.asarray(curve_points, dtype=float)

        if len(points) >= 1:
            all_points.append(points)

    data_points = np.vstack(all_points)

    deflection = max(
        max_section_spacing_mm * MESH_LINEAR_DEFLECTION_FACTOR, 1e-3
    )

    # isRelative=True matters here -- an absolute deflection made
    # Triangulation_s silently come back None below (IsDone() still
    # True) on this project's own synthetic test surface, at every
    # deflection value tried from 0.001 up to 1.0. Confirmed directly
    # that switching to a size-relative deflection fixed it, with no
    # other change.
    BRepMesh_IncrementalMesh(face, deflection, True, 0.5, True)

    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, location)

    if triangulation is None:
        return True

    transform = location.Transformation()

    samples = np.array([
        [
            node.X(), node.Y(), node.Z()
        ]
        for node in (
            triangulation.Node(i).Transformed(transform)
            for i in range(1, triangulation.NbNodes() + 1)
        )
    ])

    if len(samples) == 0:
        return True

    threshold = (
        max(max_section_spacing_mm, 1e-6) * MAX_SURFACE_DEVIATION_FACTOR
    )

    worst_surface_to_data = 0.0

    for start in range(0, len(samples), _NEAREST_NEIGHBOUR_CHUNK):

        batch = samples[start:start + _NEAREST_NEIGHBOUR_CHUNK]

        distances = np.linalg.norm(
            batch[:, None, :] - data_points[None, :, :], axis=2
        )

        worst_surface_to_data = max(
            worst_surface_to_data, distances.min(axis=1).max()
        )

    if worst_surface_to_data > threshold:
        return True

    # The reverse direction (does every real data point have nearby
    # surface, not just "every bit of surface that exists is near
    # data") can't reuse the same mesh samples: BRepMesh_IncrementalMesh
    # would need to be pushed to an unreliably fine deflection to
    # avoid gaps between mesh vertices reading as false "no nearby
    # surface" positives (confirmed directly: even a good, correct fit
    # showed an apparent 2.05mm gap here against a 2.0mm threshold,
    # from mesh sparseness alone, not a real defect). Projecting each
    # data point onto the surface directly (GeomAPI_ProjectPointOnSurf,
    # the same approach the "smooths a sharp fold" test already uses)
    # gives the TRUE nearest point on the continuous surface, so
    # coverage isn't limited by how fine a mesh OCCT happens to build.
    surface = BRep_Tool.Surface_s(face)

    for point in data_points:

        target = gp_Pnt(float(point[0]), float(point[1]), float(point[2]))

        projector = GeomAPI_ProjectPointOnSurf(target, surface)

        if projector.NbPoints() == 0:
            return True

        nearest = projector.NearestPoint()

        distance = target.Distance(nearest)

        if distance > threshold:
            return True

    return False


# Default "how much may the surface deviate from the raw curve
# points" tolerance, in mm -- used both to fit each curve into a
# smooth edge (removing residual scan noise) and, more importantly,
# as BRepOffsetAPI_MakeFilling's own Tol3D. This is the main knob for
# how aggressively small local folds/wrinkles get smoothed away:
# larger = smoother (loses more local detail, keeps the general
# shape), smaller = follows the scanned curves more closely. Exposed
# as a real parameter (see build_single_surface / main()), this is
# just the fallback default. section_stl.py's own parameter form
# lets this be set per run.
DEFAULT_SMOOTHING_TOLERANCE_MM = 2.0


def curve_from_points(points, tolerance_mm):
    """
    Fits a smooth, approximating B-spline curve near an ordered list
    of 3D points (within `tolerance_mm`), rather than interpolating
    through every one of them exactly -- see the module note above.
    """

    n = len(points)

    array = TColgp_Array1OfPnt(1, n)

    for i, p in enumerate(points, start=1):
        array.SetValue(i, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

    fitter = GeomAPI_PointsToBSpline(
        array,
        3,
        8,
        GeomAbs_Shape.GeomAbs_C2,
        tolerance_mm
    )

    return fitter.Curve()


def edge_from_points(points, tolerance_mm):
    return BRepBuilderAPI_MakeEdge(
        curve_from_points(points, tolerance_mm)
    ).Edge()


def build_single_surface(
    boundary_points, interior_curves, smoothing_tolerance_mm,
    max_section_spacing_mm
):
    """
    Builds ONE TopoDS_Face covering the whole scanned patch: the
    closed `boundary_points` loop is the surface's bounding wire,
    and every curve in `interior_curves` (each an (N, 3) point
    array -- typically every A_i and B_j section curve) is added as
    an internal guide curve the surface is pulled towards, not a
    hard boundary. See the module docstring for why that specific
    combination (one free-form fill, boundary + internal guides)
    was chosen over a per-cell patchwork.

    `max_section_spacing_mm` is the LARGER of the two real distances
    between adjacent parallel cutting planes (section_stl.py's
    width_a/(count_a-1) and width_b/(count_b-1)) -- the data's
    natural local length scale, used by the surface-deviation sanity
    check below.

    Raises RuntimeError if OpenCASCADE fails to build the surface,
    builds one that is not a topologically valid B-rep face (checked
    with BRepCheck_Analyzer rather than trusting the algorithm's own
    "done" status alone), or builds a topologically "valid" surface
    that locally sags/bulges far past where the real input data
    actually is (see _surface_deviates_from_data) -- a real failure
    mode on a genuinely complex shape, not just a hypothetical one.
    """

    # MaxDeg (the filler's 9th argument) matters well beyond its own
    # trimmed geometry: a real SolidWorks import showed a "visor"
    # panel as a bizarre, disconnected blade far from the crown, even
    # though this project's own checks passed it and re-reading the
    # actual STEP file back independently confirmed the real, TRIMMED
    # face was fine. Root cause, confirmed directly: at MaxDeg=8 the
    # surface's raw, UNTRIMMED control points reached +-17800mm for a
    # face whose real (trimmed) extent is under 300mm -- SolidWorks'
    # own STEP import evidently samples/renders closer to that raw,
    # untrimmed representation for at least some faces, unlike
    # OpenCASCADE's own BRepMesh_IncrementalMesh (which only ever
    # triangulates the real, trimmed region -- see
    # _surface_deviates_from_data's own comment on exactly this trap).
    # Lowered from 8 to 5 after sweeping MaxDeg against this project's
    # two real regions from that same scan (a domed crown and a
    # near-flat visor): 5 is the largest value where BOTH regions
    # still found a valid fit (at the same tolerances the existing
    # retry loop already reaches for them), while cutting raw pole
    # spread by roughly an order of magnitude versus 8. MaxDeg=3 was
    # tried too and cut it further, but was too restrictive for the
    # more complex crown region to fit at all.
    filler = BRepOffsetAPI_MakeFilling(
        3, 15, 3, False, 1e-5,
        smoothing_tolerance_mm, 0.01, 0.1, 5, 9
    )

    boundary_points = np.asarray(boundary_points, dtype=float)

    # The boundary must be a genuinely CLOSED loop (its own start and
    # end coinciding) to bound a region at all -- an open curve fed
    # as IsBound=True is topologically ambiguous and was confirmed
    # (empirically, against a real 44-point boundary loop) to make
    # OpenCASCADE build a "successful" but geometrically invalid
    # face. Closed explicitly here so callers don't have to remember
    # to append the first point back on themselves.
    if not np.allclose(boundary_points[0], boundary_points[-1]):
        boundary_points = np.vstack([boundary_points, boundary_points[0:1]])

    boundary_edge = edge_from_points(boundary_points, smoothing_tolerance_mm)
    filler.Add(boundary_edge, GeomAbs_C0, True)

    for curve_points in interior_curves:

        if len(curve_points) < 2:
            continue

        edge = edge_from_points(curve_points, smoothing_tolerance_mm)
        filler.Add(edge, GeomAbs_C0, False)

    # On a genuinely inconsistent boundary, OpenCASCADE doesn't
    # always fail softly via IsDone() -- it can raise its own native
    # exception straight out of Build(). curve_utils.order_boundary_loop
    # guarantees a non-self-crossing loop (via a nearest-neighbour
    # walk + 2-opt uncrossing pass) so this shouldn't trigger from a
    # bad boundary anymore, but it's still caught here and turned
    # into a normal RuntimeError so the caller (and section_stl.py,
    # which runs this as a subprocess) gets a clear message instead
    # of a raw OCP traceback either way.
    try:
        filler.Build()
    except Exception as exc:
        raise RuntimeError(
            "OpenCASCADE could not fill the boundary loop and guide "
            f"curves into a single surface ({exc}). This usually "
            "means the boundary loop crosses itself -- check "
            "sections_3d/boundary_loop.dxf for a clean, simple "
            "outline with no self-crossings."
        ) from exc

    if not filler.IsDone():
        raise RuntimeError(
            "OpenCASCADE could not fill the boundary loop and "
            "guide curves into a single surface."
        )

    face = TopoDS.Face_s(filler.Shape())

    if not BRepCheck_Analyzer(face).IsValid():
        raise RuntimeError(
            "OpenCASCADE built a surface but it is geometrically "
            "invalid (likely self-intersecting)."
        )

    if _surface_deviates_from_data(
        face, boundary_points, interior_curves, max_section_spacing_mm
    ):
        raise RuntimeError(
            "OpenCASCADE built a topologically valid surface, but it "
            "locally sags or bulges far beyond the scan's local grid "
            "spacing at some point on the surface (a B-spline surface "
            "can oscillate wildly -- Runge's phenomenon -- when the "
            "shape is too complex for its degree at this tolerance)."
        )

    return face


def write_step(shape, output_path):
    """
    Writes `shape` to a STEP file at `output_path`, explicitly
    tagged as millimetres. ezdxf's default DXF unit is a lesson
    already paid for in this project (see the DXF export fix in
    section_stl.py) -- OpenCASCADE's own STEP writer defaults to
    millimetres already, but this is set explicitly rather than
    relied upon.
    """

    writer = STEPControl_Writer()

    Interface_Static.SetCVal_s("write.step.unit", "MM")

    transfer_status = writer.Transfer(
        shape, STEPControl_StepModelType.STEPControl_AsIs
    )

    if transfer_status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(
            f"STEP transfer failed (status: {transfer_status})"
        )

    write_status = writer.Write(str(output_path))

    if write_status != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise RuntimeError(
            f"STEP write failed (status: {write_status})"
        )


# How many times to retry with a looser tolerance if a fill fails --
# and by what factor each retry loosens smoothing_tolerance_mm. A
# real, complex shape (e.g. the dome/brim/rear of a hat, all in one
# surface) can be too much for a single low-degree B-spline to fit
# without folding over itself at the tolerance the user picked, even
# once the boundary loop itself is clean; a looser tolerance gives
# OpenCASCADE more room to find a valid (if less detailed) fit. Saves
# a manual guess-and-check cycle re-running the whole pipeline for.
RETRY_TOLERANCE_FACTOR = 2.5
MAX_RETRIES = 3


def _build_region_face(region_data):
    """
    Builds one region's face (see build_single_surface), retrying
    with a progressively looser smoothing tolerance if the fill fails
    (RETRY_TOLERANCE_FACTOR/MAX_RETRIES) before giving up. One
    region's worth of what build_step_surfaces does for the whole
    patch when it isn't split into several.
    """

    tolerance = region_data.get(
        "smoothing_tolerance_mm", DEFAULT_SMOOTHING_TOLERANCE_MM
    )
    requested_tolerance = tolerance

    last_error = None

    for attempt in range(MAX_RETRIES + 1):

        try:

            face = build_single_surface(
                region_data["boundary_loop"],
                region_data["interior_curves"],
                tolerance,
                region_data["max_section_spacing_mm"]
            )

            if attempt > 0:
                print(
                    f"  Succeeded at a looser smoothing tolerance: "
                    f"{tolerance:.2f} mm (requested: "
                    f"{requested_tolerance:.2f} mm)"
                )

            return face

        except RuntimeError as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                print(
                    f"  Fill failed at {tolerance:.2f} mm ({exc}); "
                    f"retrying at "
                    f"{tolerance * RETRY_TOLERANCE_FACTOR:.2f} mm..."
                )

            tolerance *= RETRY_TOLERANCE_FACTOR

    raise RuntimeError(
        f"Could not build a valid surface even after {MAX_RETRIES} "
        f"retries with looser tolerances (last tried: "
        f"{tolerance / RETRY_TOLERANCE_FACTOR:.2f} mm). Last error: "
        f"{last_error}"
    ) from last_error


# How much slack to give BRepBuilderAPI_Sewing when merging faces
# that share a seam edge (only relevant when the patch was split into
# several regions -- see curve_utils.split_boundary_and_curves_at_
# separator). Each region's own edge along that seam was independently
# fit (edge_from_points) from the SAME separator points, so they
# should already sit close together; a small multiple of the largest
# smoothing tolerance actually used gives some margin without being
# so loose it merges edges that aren't really the same seam.
SEWING_TOLERANCE_FACTOR = 2.0


def build_step_surfaces(data, output_path):
    """
    Entry point: takes the dict pickled by section_stl.py
    ({"regions": [{"boundary_loop": ..., "interior_curves": ...,
    "smoothing_tolerance_mm": ..., "max_section_spacing_mm": ...},
    ...]}) and writes the reconstructed surface(s) to `output_path`
    -- one region if the scanned patch wasn't split (the common
    case), or several sewn together at their shared seam if it was
    (see curve_utils.split_boundary_and_curves_at_separator: splitting
    a patch whose regions have very different curvature -- e.g. a
    cap's near-flat visor and domed crown -- into separate, individually
    simpler surfaces, rather than asking one surface to reconcile all
    of it at once).

    Each region is built independently via _build_region_face (same
    retry-with-looser-tolerance behaviour as always -- see the
    comment above RETRY_TOLERANCE_FACTOR/MAX_RETRIES).
    """

    regions = data["regions"]

    faces = []

    for i, region_data in enumerate(regions, start=1):

        if len(regions) > 1:
            print(f"Region {i}/{len(regions)}:")

        faces.append(_build_region_face(region_data))

    if len(faces) == 1:

        shape = faces[0]

    else:

        sewing_tolerance = max(
            region_data.get(
                "smoothing_tolerance_mm", DEFAULT_SMOOTHING_TOLERANCE_MM
            )
            for region_data in regions
        ) * SEWING_TOLERANCE_FACTOR

        sewer = BRepBuilderAPI_Sewing(sewing_tolerance)

        for face in faces:
            sewer.Add(face)

        sewer.Perform()

        shape = sewer.SewedShape()

        if not BRepCheck_Analyzer(shape).IsValid():
            raise RuntimeError(
                f"Sewed {len(faces)} region faces together, but the "
                "result is not a valid shape (their shared seam "
                "edges may not actually line up within the sewing "
                "tolerance)."
            )

    write_step(shape, output_path)


def main():

    if len(sys.argv) != 3:
        print(
            "Usage: python step_export.py <curves.pickle> <output.step>",
            file=sys.stderr
        )
        raise SystemExit(2)

    data_path, output_path = sys.argv[1], sys.argv[2]

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    build_step_surfaces(data, output_path)

    print(f"Surface written to {output_path}")


if __name__ == "__main__":
    main()
