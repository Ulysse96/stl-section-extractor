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
fitting, crossing detection, boundary loop ordering) lives in
[curve_utils.py](curve_utils.py), which has no GUI dependency and can be
imported on its own. `section_stl.py` itself is the interactive pipeline:
it owns every file dialog, 3D view and DXF export step, and calls into
`curve_utils` for the actual geometry processing.
[step_export.py](step_export.py) turns the boundary loop + section curves
into a single surface and writes STEP; it only needs OpenCASCADE (`OCP`)
+ the standard library, no GUI or `curve_utils` dependency, so it can run
as a standalone process in a different Python environment — see
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

When plane B is used, `section_stl.py` also turns the scanned patch into
ONE continuous surface and exports it as STEP — a finished model you can
open directly (e.g. in FreeCAD), without going through SolidWorks.

The boundary loop (`order_boundary_loop`/`collect_curve_endpoints` in
[curve_utils.py](curve_utils.py)) is pure Python/numpy and needs nothing
extra. Turning the boundary + every A/B curve into a surface
(`step_export.py`, via OpenCASCADE's free-form filling,
`BRepOffsetAPI_MakeFilling`) needs an OpenCASCADE binding (`cadquery`'s
`OCP`), which currently has **no Windows wheels for Python 3.13/3.14** —
only up to 3.12. `section_stl.py` therefore runs this one step as a
**separate process**, in a dedicated Python 3.12 virtual environment, and
simply skips it (with a message pointing here) if that environment isn't
set up.

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

**Why one surface, not a patch per grid cell**: an earlier version built
one small Coons-style patch per A×B grid cell and sewed them into a
shell. On a real scan that came out visibly faceted, occasionally shot a
patch into a wild spike, and produced faces SolidWorks' own import
diagnostics kept flagging as invalid even after this project's own
checks passed them — and a tangent-continuity (G1) fix for the faceted
look turned out to need OpenCASCADE settings that made a single cell
take minutes to build. A single surface, with every A/B curve fed in as
a *soft guide* rather than a hard per-cell boundary, sidesteps all of
that: confirmed directly against OCP, a sharp local fold's amplitude
survives at only a few percent in the resulting surface even at
aggressive settings, while the general shape away from that fold still
tracks the real curve network closely — exactly "smooth surface,
overall shape over local wrinkle fidelity", which is also what this
project's own downstream use (flattening the result in a tool like
Wrapstyler) wants.

**Smoothing tolerance**: the parameter form's "Smoothing tolerance (mm)"
field (shown once plane B is selected) is the main knob for how
aggressively small local folds/wrinkles get smoothed away — larger
values smooth more (and build faster); smaller values follow the scan
more closely. 2–5 mm is a reasonable starting point; the field is right
there in the same form as the other reconstruction settings, so it's
easy to re-run with a different value and compare.

`section_stl.py` also exports `sections_3d/boundary_loop.dxf` (and
includes the same spline in `sections_main_3d.dxf`): a single closed
spline through the ordered open endpoints of every A/B curve — the
outline of the scanned patch, and (unless excluded, see below) the same
boundary the STEP surface is built from.

**Excluding sections from the surface**: a real hole or gap in the
object — a strap adjustment slot on a cap, a vent — isn't scan noise,
but it does break the "one simple closed boundary" assumption the whole
surface step relies on: the boundary loop and the surface itself can
both get dragged into a bad shape right at that hole. The "Exclude from
surface" field (next to the smoothing tolerance) takes a comma-separated
list of sections to leave out of the surface reconstruction specifically
— e.g. `A5, B11` — found by checking the console output (a section with
an unusually high "Curves detected" count, or lots of "N crossing(s)
expected, M found" mismatches nearby, is the usual sign) or by looking
at `boundary_loop.dxf` for where the outline goes wrong. Excluded
sections are still exported to DXF as normal; they're just left out of
`order_boundary_loop`'s input and the surface's guide curves.
