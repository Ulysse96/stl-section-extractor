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
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.GeomAbs import GeomAbs_C0, GeomAbs_Shape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopoDS import TopoDS
from OCP.STEPControl import STEPControl_Writer, STEPControl_StepModelType
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_ReturnStatus


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


def build_single_surface(boundary_points, interior_curves, smoothing_tolerance_mm):
    """
    Builds ONE TopoDS_Face covering the whole scanned patch: the
    closed `boundary_points` loop is the surface's bounding wire,
    and every curve in `interior_curves` (each an (N, 3) point
    array -- typically every A_i and B_j section curve) is added as
    an internal guide curve the surface is pulled towards, not a
    hard boundary. See the module docstring for why that specific
    combination (one free-form fill, boundary + internal guides)
    was chosen over a per-cell patchwork.

    Raises RuntimeError if OpenCASCADE fails to build the surface,
    or builds one that is not a topologically valid B-rep face
    (checked with BRepCheck_Analyzer rather than trusting the
    algorithm's own "done" status alone).
    """

    filler = BRepOffsetAPI_MakeFilling(
        3, 15, 3, False, 1e-5,
        smoothing_tolerance_mm, 0.01, 0.1, 8, 9
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


def build_step_surfaces(data, output_path):
    """
    Entry point: takes the dict pickled by section_stl.py
    ({"boundary_loop": ..., "interior_curves": ...,
    "smoothing_tolerance_mm": ...}) and writes the single
    reconstructed surface to `output_path`.

    If the fill fails at the requested smoothing tolerance, retries
    a few times with a progressively looser one (see
    RETRY_TOLERANCE_FACTOR/MAX_RETRIES) before giving up -- see the
    comment above those constants for why this can help. Prints which
    tolerance actually succeeded when a retry was needed.
    """

    tolerance = data.get("smoothing_tolerance_mm", DEFAULT_SMOOTHING_TOLERANCE_MM)

    last_error = None

    for attempt in range(MAX_RETRIES + 1):

        try:

            face = build_single_surface(
                data["boundary_loop"],
                data["interior_curves"],
                tolerance
            )

            if attempt > 0:
                print(
                    f"Succeeded at a looser smoothing tolerance: "
                    f"{tolerance:.2f} mm (requested: "
                    f"{data.get('smoothing_tolerance_mm', DEFAULT_SMOOTHING_TOLERANCE_MM):.2f} mm)"
                )

            write_step(face, output_path)

            return

        except RuntimeError as exc:

            last_error = exc

            if attempt < MAX_RETRIES:

                print(
                    f"Fill failed at {tolerance:.2f} mm ({exc}); "
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
