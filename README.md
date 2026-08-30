# section_stl.py

Extracts cross-section curves from a 3D-scanned STL mesh and exports them as
DXF files, for use as a SolidWorks Loft/Boundary Surface curve network.

## Usage

```bash
pip install -r requirements.txt
python section_stl.py
```

**On Windows, prefer double-clicking [run_section_stl.bat](run_section_stl.bat)**
over running/double-clicking `section_stl.py` directly: a Python script's own
console window closes itself the instant the script exits, whether it
finished normally or crashed — so a crash's error message is gone before
you can read it. The `.bat` file captures everything into `log.txt`, prints
it back to the console, and waits for a key press before closing, so the
window (and `log.txt`) always has the full output, even after a crash.

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
easy to re-run with a different value and compare. If the fill fails at
the requested tolerance (a genuinely complex real shape can be too much
for a single low-degree surface to fit without folding over itself),
`step_export.py` automatically retries a few times with a progressively
looser tolerance before giving up, and reports which one worked.

**Why a "successful" STEP file can still be unusable**: OpenCASCADE's own
validity check (`BRepCheck_Analyzer`) only catches *topological* problems
(e.g. self-intersection) — it does not notice a surface that stays
"valid" while locally sagging, bulging, or even failing to cover part of
the scan at all. This happened on a real hat scan: the fill "succeeded",
but the resulting surface's control points reached tens of thousands of
mm for an object a few hundred mm across, and SolidWorks couldn't open
the file properly.

`step_export.py` now checks the fitted surface *itself* (not its raw
control points, which can look wild even on a perfectly good surface —
see below) against `max_section_spacing_mm` (the larger of the two real
distances between adjacent parallel cutting planes — the data's own
local resolution): no point actually on the surface may sit more than
twice that spacing from the nearest real input point, checked **both
ways** — every surface point must be near some scan data, and every
scan data point must be near some surface. Missed either direction on a
real hat fit: a surface whose control points and *overall* bounding box
looked plausible had nonetheless only actually covered about half the
scan's real height, folding back onto the part it did cover instead of
extending to the rest — invisible to a one-way or whole-surface check,
caught immediately by checking both directions locally.

Getting the sample points themselves right took a few tries, in
increasing order of both reliability and subtlety — recorded in
`step_export.py`'s own comment above `_surface_deviates_from_data` in
more detail than belongs here, since it's the kind of lesson worth not
re-learning by hand next time this needs touching:
- comparing the surface's whole bounding box to the whole object's size
  (too coarse — a real, non-flat scan shows similar *relative* overshoot
  whether the fit is fine or badly broken, so this let real blowups
  through);
- sampling the surface on a plain `(u, v)` parameter grid (the
  parametrization itself becomes untrustworthy on exactly the kind of
  distorted, badly-behaved fit this check exists to catch, so this
  disagreed wildly with the real geometry on the worst cases);
- the surface's real triangulated mesh for one direction, and directly
  projecting each scan point onto the surface (`GeomAPI_ProjectPointOnSurf`)
  for the other, which is what's used now.

**A cutting plane can graze past the object instead of through it**:
confirmed directly on a real scan — the *outermost* section in a
direction (e.g. `B11`, the last plane-B section) can end up past where
the object still has a full, simple cross-section, and only clips a
small, localized feature instead (a seam, a button, a bump) — the mesh
intersection fragments into several small disconnected pieces there
rather than one clean loop, and the reconstructed surface can bulge or
fail unpredictably regardless of tolerance, degree, or exclusions. If a
section near the end of plane A or B reports several similarly-sized
"Curves detected" (rather than one clearly dominant curve and a small,
much shorter secondary one, which is normal), that section is a likely
culprit — try reducing that plane's width slightly so its outermost
section lands back inside the object's simple, full-loop region.

**Separating into panels**: even with a clean boundary and a
reasonable-looking, check-passing surface, a real export still failed
to import into SolidWorks ("ファイルにジオメトリ データが含まれていません" /
no geometry data). Root cause, confirmed directly: the fitted surface's
own boundary edge sat 14–33mm off from the requested boundary curve
(`BRep_Tool.Tolerance_s` on the built edge, after OpenCASCADE widens it
to stay topologically consistent) — regardless of smoothing tolerance,
`MaxSegments`, or post-hoc tolerance repair (`ShapeFix_Shape`,
`BRepLib.SameParameter_s` — neither could shrink it, since it reflects
a real geometric fact, not a stale tolerance flag). OpenCASCADE's own
validity check tolerates this; SolidWorks' (Parasolid) importer does
not. A real cap is several separate panels (crown, visor, ...), not
one continuous surface — asking one B-spline to reconcile a near-flat
visor and a domed crown at once is a lot to ask of a single fit.

After the "EXCLUDE SECTIONS" window, a "SEPARATE INTO PANELS" window
opens: click a sequence of points along a seam directly on the mesh
(e.g. where the visor meets the crown), both ends near the outer edge.
Press `n` to finish that seam and trace another one — each further
seam must stay within a single panel already split off so far (trace
the crown/visor seam first; a seam splitting a *further* panel off has
to be traced within one of those two results, not across both at
once). Close the window when done. Each seam splits its own panel into
two separate, individually simpler surfaces
(`curve_utils.split_into_panels`, built on
`split_boundary_and_curves_at_separator` applied once per seam)
instead of one surface spanning every region's curvature at once —
every resulting panel gets its own retry-with-looser-tolerance attempt
(see above), and all of them are sewn together at their shared seams
before being written to one STEP file. No clicks (or just one point)
— the default — keeps the single-surface behaviour unchanged.

If one panel still can't be fit after every retry (a seam traced too
close to an existing panel edge can carve off a thin, hard-to-fit
sliver — confirmed on a real scan), that panel is skipped, with a
clear message saying so — the STEP file still gets written with every
OTHER panel that did succeed, rather than the whole export coming up
empty over one bad seam. Retrace that specific seam (further from the
existing boundary) and re-run if the skipped panel was actually
needed.

A seam whose two ends snap too close together along the boundary is
rejected up front, the same way, rather than silently building a
razor-thin sliver region: confirmed on a real scan with several seams
traced close together, one boundary ended up with only 3 points, which
still "passed" every downstream check (a boundary that small is easy
to stay technically close to) while producing a badly twisted,
physically nonsensical surface. One bad seam like this is skipped and
reported on its own, too — it doesn't discard every OTHER seam that
was traced correctly.

**Only edge-to-edge seams are supported for now**: both ends of a seam
must land on the outer boundary (of the panel being split). A seam
fully enclosed inside a panel's outline (a button, a back tab — not
touching any outer edge) isn't supported yet.

**A "good" surface can still render wrong in SolidWorks specifically**:
even with real, correct geometry (independently re-verified: reading
the actual STEP file back and triangulating each face confirmed both a
crown and a visor panel matched the real scan closely), SolidWorks
showed the visor as a bizarre, disconnected blade far from the crown.
Root cause, confirmed directly: at the filler's previous `MaxDeg=8`,
the fitted surface's raw, UNTRIMMED control points reached ±17800mm
for a face whose real (trimmed) extent is under 300mm — topologically
harmless (OpenCASCADE's own `BRepMesh_IncrementalMesh` only ever
triangulates the real, trimmed region, which is why this project's own
checks and a fresh re-read both looked fine), but SolidWorks' STEP
import evidently samples/renders closer to that raw, untrimmed
representation for at least some faces. Lowered `MaxDeg` from 8 to 5 —
swept against this project's own real crown + visor regions: 5 is the
largest value where both still found a valid fit (at the same
tolerances the existing retry loop already reaches for them), while
cutting the raw control points' spread by roughly an order of
magnitude. `MaxDeg=3` cut it further but was too restrictive for the
more complex crown region to fit at all.

`section_stl.py` also exports `sections_3d/boundary_loop.dxf` (and
includes the same spline in `sections_main_3d.dxf`): a single closed
spline through the ordered open endpoints of every A/B curve — the
outline of the scanned patch, and (unless excluded, see below) the same
boundary the STEP surface is built from.

**Excluding sections from the surface**: a real hole or gap in the
object — a strap adjustment slot on a cap, a vent — isn't scan noise,
but it does break the "one simple closed boundary" assumption the whole
surface step relies on: the boundary loop and the surface itself can
both get dragged into a bad shape right at that hole (or the STEP file
comes out tiny and SolidWorks/FreeCAD can't open it properly). After
plane B's section curves are extracted, an "EXCLUDE SECTIONS FROM
SURFACE" window opens (every curve visible and labelled, e.g. `A5`,
`B11` — a typed section number means nothing without seeing it first,
so this is click-based, not a form field): click a curve to leave it
out of the surface (it dims); click again to re-include it; close the
window when done (no clicks = every section is used). Excluded
sections are still exported to DXF as normal; they're just left out of
`order_boundary_loop`'s input and the surface's guide curves for that
run.
