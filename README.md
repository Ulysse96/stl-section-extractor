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

## Requirements

See [requirements.txt](requirements.txt). Python 3 with `tkinter` (included
in standard Windows/macOS installers).

## Code layout

The curve math (resampling, fold stitching, smoothing, spline/polynomial
fitting, crossing detection) lives in [curve_utils.py](curve_utils.py),
which has no GUI dependency and can be imported on its own.
`section_stl.py` itself is the interactive pipeline: it owns every file
dialog, 3D view and DXF export step, and calls into `curve_utils` for the
actual geometry processing.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

Tests in [tests/test_curve_utils.py](tests/test_curve_utils.py) cover
`curve_utils.py` only — `section_stl.py` runs its GUI pipeline as soon as
it is imported, so it isn't unit-tested directly.
