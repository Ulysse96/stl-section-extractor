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
