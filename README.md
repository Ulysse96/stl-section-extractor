# section_stl.py

Extracts cross-section curves from a 3D-scanned STL mesh and exports them as
DXF files, for use as a SolidWorks Loft/Boundary Surface curve network.

## Usage

```bash
pip install -r requirements.txt
python section_stl.py
```

1. A file dialog opens — select the STL to process.
2. A 3D view opens — click 3 points (plane A only) or 5 points (plane A +
   perpendicular plane B) on the mesh to define the cutting plane(s), then
   close the window.
3. A form window lets you set the section width/count for each plane,
   curve smoothing strength, optional ideal-curve fitting, and the
   piecewise polynomial reconstruction settings.
4. The script slices the mesh, extracts and processes each section's
   curves, and writes DXF files next to the source STL, in a
   `<stl name>_sections/` folder.
5. If plane B was used (5-point selection), it also reconstructs the
   scanned surface and exports it as STEP — see
   [Surface reconstruction (STEP export)](#surface-reconstruction-step-export)
   below for the one-time setup this needs.

## Requirements

See [requirements.txt](requirements.txt). Python 3 with `tkinter` (included
in standard Windows/macOS installers).

## Code layout

The curve math (resampling, fold stitching, smoothing, spline/polynomial
fitting, crossing detection, A×B grid assembly) lives in
[curve_utils.py](curve_utils.py), which has no GUI dependency and can be
imported on its own. `section_stl.py` itself is the interactive pipeline:
it owns every file dialog, 3D view and DXF export step, and calls into
`curve_utils` for the actual geometry processing.
[step_export.py](step_export.py) turns a curve grid into surfaces and
writes STEP; it only needs OpenCASCADE (`OCP`) + the standard library, no
GUI or `curve_utils` dependency, so it can run as a standalone process in
a different Python environment — see
[Surface reconstruction (STEP export)](#surface-reconstruction-step-export).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests in [tests/test_curve_utils.py](tests/test_curve_utils.py) cover
`curve_utils.py` only — `section_stl.py` runs its GUI pipeline as soon as
it is imported, so it isn't unit-tested directly.
[tests/test_step_export.py](tests/test_step_export.py) covers
`step_export.py` and needs the `.venv312` environment below — it's
skipped automatically otherwise.

## Surface reconstruction (STEP export)

When plane B is used, `section_stl.py` also turns the A×B curve grid into
an actual surface model and exports it as STEP — a finished shell you can
open directly (e.g. in FreeCAD), without going through SolidWorks.

The curve-network assembly (`build_surface_grid` in
[curve_utils.py](curve_utils.py)) is pure Python/numpy and needs nothing
extra. Turning it into surfaces (`step_export.py`) needs an OpenCASCADE
binding (`cadquery`'s `OCP`), which currently has **no Windows wheels for
Python 3.13/3.14** — only up to 3.12. `section_stl.py` therefore runs this
one step as a **separate process**, in a dedicated Python 3.12 virtual
environment, and simply skips it (with a message pointing here) if that
environment isn't set up.

One-time setup, from this folder:

```bash
winget install --id Python.Python.3.12 --source winget
py -3.12 -m venv .venv312
.venv312\Scripts\python.exe -m pip install -r requirements-step.txt
```

After that, running `python section_stl.py` as usual (on whatever Python
you normally use for it) will find `.venv312` and produce
`<stl name>_sections/reconstructed_surface.step` alongside the DXF output.

**Windows Smart App Control note**: if it's enabled on your machine, it
may block some freshly pip-installed native packages inside a *new* venv
(this project hit it blocking `vtk` and `scipy`'s compiled extensions in
`.venv312`, while `OCP` itself was unaffected) — this is why the
interactive GUI pipeline (which needs `vtk`/`pyvista`) stays on your main
Python entirely, and only the OCP-only `step_export.py` runs inside
`.venv312`. If you hit a similar `DLL load failed... アプリケーション制御
ポリシー` error, it's Smart App Control blocking that specific file, not a
bug in this project. Once Smart App Control is fully "On" (not
"Evaluation"), Windows doesn't offer a way to turn it back off short of
reinstalling Windows — check Settings → Privacy & security → Windows
Security → App & browser control before assuming that's your only option.

**Current scope**: only the *interior* grid cells (bounded by 4 curve
segments) are filled with a surface patch today. The outer boundary ring
isn't stitched yet, so the exported shell is open along the scan's edge
rather than a fully closed solid.

Adjacent patches only share positional (C0) continuity, not tangent (G1),
so the result can look faceted rather than perfectly smooth — a
tangent-matching version was tried and dropped as impractical (no
measurable effect at OpenCASCADE's default settings, minutes per cell
once cranked up enough to matter; see `step_export.py`). **Use a finer
A/B section count** (more, smaller sections) in the parameter form for a
smoother-looking result — same principle as a denser mesh reading as
smoother. Cells with a geometrically invalid fill (typically at a
fold/defect in the scan) are detected and skipped, reported by name
rather than silently exported.

Cell edges are also used as-is from `all_section_data`/the curve
snapshot rather than being re-fit into a genuinely smooth spline before
building the surface — `step_export.py` fits an approximating (not
interpolating) B-spline near each edge's points instead of threading
through every one of them exactly, which is what removed a "torn,
raw-looking" surface on a real scan.

`section_stl.py` also exports `sections_3d/boundary_loop.dxf`: a single
closed spline through the ordered open endpoints of every A/B curve —
the outline of the scanned patch — built from the same full-resolution
curve snapshot the STEP export uses, not from reconstruction method 2's
simplified curves.
