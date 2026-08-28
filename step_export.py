"""
Builds the surface-reconstruction shell from the A x B curve grid
(see curve_utils.build_surface_grid) and writes it to STEP.

This module needs cadquery's OCP (OpenCASCADE) bindings, which have no
Windows wheels for Python 3.13/3.14 -- see the project README for the
dedicated .venv312 this runs in. section_stl.py itself keeps running on
whatever Python already has pyvista/vtk working, and invokes this
module as a SEPARATE PROCESS in .venv312, handing it the grid data
through a pickle file -- so the interactive GUI pipeline and the
CAD-kernel-dependent surface export never have to share one Python
process (this also happens to route around a Windows Smart App Control
policy on the original dev machine that blocks vtk specifically inside
.venv312, see README).

Run directly: python step_export.py <grid.pickle> <output.step>

Current scope: builds a surface patch for every INTERIOR grid cell
(bounded by 4 curve segments, 2 from each family). The outer boundary
ring (the triangular cells against the patch's perimeter loop) is not
filled yet -- the exported shell is open along the scan's outer edge
rather than a fully closed solid.

Adjacent patches only share positional (C0) continuity, not tangent
(G1) -- on a real scan this reads as a faceted, "geodesic dome" look
rather than one smooth surface. A G1 version (each new patch built
against its already-built neighbour as a continuity reference) was
tried and dropped: with OpenCASCADE's default settings it had no
measurable effect on tangent matching, and cranking up the iteration
count enough to matter made a single cell take minutes to build. The
practical fix for the faceted look is a finer A/B section count in
section_stl.py (smaller, more numerous cells read as smooth from a
normal viewing distance), not a change here.
"""

import pickle
import sys

from OCP.gp import gp_Pnt
from OCP.TColgp import TColgp_HArray1OfPnt
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.GeomAbs import GeomAbs_C0
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_Sewing
from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeFilling
from OCP.BRepCheck import BRepCheck_Analyzer
from OCP.TopoDS import TopoDS
from OCP.STEPControl import STEPControl_Writer, STEPControl_StepModelType
from OCP.Interface import Interface_Static
from OCP.IFSelect import IFSelect_ReturnStatus


def curve_from_points(points):
    """Interpolates an OCP B-spline curve through an ordered list of 3D points."""

    n = len(points)

    harray = TColgp_HArray1OfPnt(1, n)

    for i, p in enumerate(points, start=1):
        harray.SetValue(i, gp_Pnt(float(p[0]), float(p[1]), float(p[2])))

    interp = GeomAPI_Interpolate(harray, False, 1e-6)
    interp.Perform()

    return interp.Curve()


def edge_from_points(points):
    return BRepBuilderAPI_MakeEdge(curve_from_points(points)).Edge()


def face_from_edges(edge_point_lists):
    """
    Builds one smooth surface patch (a TopoDS_Face) passing through
    3 or 4 boundary curves, using OpenCASCADE's general-purpose
    surface filling (BRepOffsetAPI_MakeFilling). Unlike the more
    rigid GeomFill_BSplineCurves Coons-patch API, this one doesn't
    require the boundary curves to be given in any particular order
    or orientation, and fills a 3-edge (triangular) boundary exactly
    the same way as a 4-edge interior cell -- confirmed by probing
    both cases directly against OCP before relying on it here.

    Only positional (C0) continuity is requested between adjacent
    cells; this can leave a visible tangent kink at cell boundaries.
    A tangent-matching (G1) version of this was tried and dropped:
    with OpenCASCADE's default iteration count it had no measurable
    effect, and cranking iterations up enough to matter made a
    single cell take minutes to build -- not viable for a real grid.
    A finer A/B section count (smaller, more numerous cells) is the
    practical way to make the facets read as smooth from a normal
    viewing distance, same principle as a denser mesh looking
    smoother.

    Raises RuntimeError if OpenCASCADE either fails to fill the
    boundary at all, or produces a geometrically invalid face (e.g.
    self-intersecting) -- checked directly with BRepCheck_Analyzer
    rather than trusting IsDone() alone, since a "successful" fill
    can still be invalid B-rep on messy/twisted input curves (this
    is what showed up as SolidWorks import diagnostics flagging
    faces as invalid on a real, folded scan).
    """

    filler = BRepOffsetAPI_MakeFilling()

    for points in edge_point_lists:
        filler.Add(edge_from_points(points), GeomAbs_C0)

    filler.Build()

    if not filler.IsDone():
        raise RuntimeError(
            "OpenCASCADE could not fill this cell's boundary "
            "curves into a surface."
        )

    face = TopoDS.Face_s(filler.Shape())

    if not BRepCheck_Analyzer(face).IsValid():
        raise RuntimeError(
            "OpenCASCADE filled this cell but the resulting "
            "surface is geometrically invalid (likely "
            "self-intersecting, probably from a fold/defect in "
            "the scan at this location)."
        )

    return face


def build_shell(cells):
    """
    Builds one face per interior grid cell (see
    curve_utils.build_surface_grid) and sews them all into a shell.
    Returns (shell, failures), where failures is a list of
    (a_i, b_j, error_message) for any cell OpenCASCADE either
    couldn't fill at all, or filled into a geometrically invalid
    face (see face_from_edges) -- reported to the user rather than
    aborting the whole export or silently including bad geometry.
    """

    sewing = BRepBuilderAPI_Sewing(1e-3)

    failures = []
    built = 0

    for cell in cells:

        edges = [
            cell["edge_a_lo"],
            cell["edge_a_hi"],
            cell["edge_b_lo"],
            cell["edge_b_hi"],
        ]

        try:
            face = face_from_edges(edges)
        except RuntimeError as exc:
            failures.append((cell["a_i"], cell["b_j"], str(exc)))
            continue

        sewing.Add(face)
        built += 1

    sewing.Perform()

    return sewing.SewedShape(), built, failures


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


def build_step_surfaces(grid, output_path):
    """
    Entry point: takes the dict returned by
    curve_utils.build_surface_grid and writes the reconstructed
    surface shell to `output_path`. Returns (built_count, failures).
    """

    shape, built, failures = build_shell(grid["cells"])

    if built == 0:
        raise RuntimeError(
            "No cell could be turned into a surface -- nothing to export."
        )

    write_step(shape, output_path)

    return built, failures


def main():

    if len(sys.argv) != 3:
        print(
            "Usage: python step_export.py <grid.pickle> <output.step>",
            file=sys.stderr
        )
        raise SystemExit(2)

    grid_path, output_path = sys.argv[1], sys.argv[2]

    with open(grid_path, "rb") as f:
        grid = pickle.load(f)

    built, failures = build_step_surfaces(grid, output_path)

    for a_i, b_j, message in failures:
        print(f"  Warning: cell A{a_i} x B{b_j} could not be filled: {message}")

    print(
        f"{built}/{built + len(failures)} surface patch(es) "
        f"written to {output_path}"
    )


if __name__ == "__main__":
    main()
