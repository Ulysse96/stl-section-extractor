import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pyvista as pv
import vtk
import ezdxf
import os
from scipy.interpolate import splprep, splev


# ============================================================
# PARAMETERS
# ============================================================

N_MAIN = 50
N_SMALL = 20

# Ces deux tolerances sont exprimees en MULTIPLES de la longueur
# d'arete moyenne du maillage (calculee automatiquement juste
# apres le chargement du STL, cf. section 2bis), et non plus en
# mm fixes. Ainsi elles s'adaptent d'elles-memes a la taille de
# l'objet ET a la densite du scan, qu'il s'agisse d'une casquette
# ou d'un objet 10x plus grand ou plus petit.

# Nombre d'aretes-type que deux fragments de courbe peuvent
# separer et etre quand meme recolles ensemble (pli / defaut
# de scan). Augmenter si des courbes continuent a se couper
# net a un pli ; diminuer si des sections differentes se
# recollent entre elles par erreur.
STITCH_TOLERANCE_FACTOR = 3.0

# Longueur minimale d'une courbe, en nombre d'aretes-type,
# pour etre conservee. Elimine les petites boucles de bruit
# ("ilots") issues d'irregularites du maillage, comme celle
# visible isolee sous les courbes principales dans l'apercu
# SolidWorks.
MIN_CURVE_LENGTH_FACTOR = 6.0


# ============================================================
# 1. SELECT STL
# ============================================================

root = tk.Tk()
root.withdraw()

stl_path = filedialog.askopenfilename(
    title="Select the STL file",
    filetypes=[
        ("STL files", "*.stl"),
        ("All files", "*.*")
    ]
)

if not stl_path:
    raise SystemExit


# ============================================================
# 2. LOAD MESH
# ============================================================

print()
print("======================================")
print("FILE")
print("======================================")

print(stl_path)

print()
print("Loading mesh...")

mesh = pv.read(stl_path)

if mesh.n_points == 0:
    raise RuntimeError("The STL is empty.")

print(
    f"Vertices : {mesh.n_points:,}"
)

print(
    f"Faces    : {mesh.n_cells:,}"
)


# ============================================================
# 2bis. ESTIMATE MESH RESOLUTION (average edge length)
# ============================================================
#
# This single measurement is what lets STITCH_TOLERANCE and
# MIN_CURVE_LENGTH adapt automatically to any object size and
# any scan density, instead of using fixed mm values tuned for
# one specific object.

print()
print("Estimating mesh resolution...")

mesh_edges = mesh.extract_all_edges()

edge_lines = mesh_edges.lines.reshape(-1, 3)

edge_vectors = (
    mesh_edges.points[edge_lines[:, 1]]
    -
    mesh_edges.points[edge_lines[:, 2]]
)

edge_lengths = np.linalg.norm(
    edge_vectors,
    axis=1
)

# Median is more robust than mean against a few abnormally
# long/short triangles (common on noisy scan meshes).
AVG_EDGE_LENGTH = float(
    np.median(edge_lengths)
)

STITCH_TOLERANCE_MM = (
    STITCH_TOLERANCE_FACTOR
    *
    AVG_EDGE_LENGTH
)

MIN_CURVE_LENGTH_MM = (
    MIN_CURVE_LENGTH_FACTOR
    *
    AVG_EDGE_LENGTH
)

print(
    f"Median edge length : {AVG_EDGE_LENGTH:.3f} mm"
)

print(
    f"Stitch tolerance    : {STITCH_TOLERANCE_MM:.3f} mm "
    f"({STITCH_TOLERANCE_FACTOR:.1f} x edge length)"
)

print(
    f"Min curve length    : {MIN_CURVE_LENGTH_MM:.3f} mm "
    f"({MIN_CURVE_LENGTH_FACTOR:.1f} x edge length)"
)


# ============================================================
# 3. SELECT FIVE POINTS
# ============================================================

plotter = pv.Plotter(
    window_size=(1400, 900)
)

plotter.add_mesh(
    mesh,
    color="lightgray",
    show_edges=False
)

plotter.add_axes()

plotter.add_text(
    "SELECT 5 POINTS: 1,2,3 define plane A, "
    "1,4,5 define plane B (perpendicular to A)",
    position="upper_left",
    font_size=16
)

plotter.add_text(
    "Left click = select point",
    position="lower_left",
    font_size=12
)


picker = vtk.vtkCellPicker()
picker.SetTolerance(0.001)

selected_points = []


def click(caller, event):

    if len(selected_points) >= 5:
        return

    x, y = caller.GetEventPosition()

    picker.Pick(
        x,
        y,
        0,
        plotter.renderer
    )

    if picker.GetCellId() < 0:
        print("No surface detected.")
        return

    point = np.array(
        picker.GetPickPosition()
    )

    selected_points.append(point)

    n = len(selected_points)

    print(
        f"Point {n}: "
        f"X={point[0]:.4f}, "
        f"Y={point[1]:.4f}, "
        f"Z={point[2]:.4f}"
    )

    marker = pv.PolyData(
        point.reshape(1, 3)
    )

    plotter.add_mesh(
        marker,
        color="red",
        point_size=20,
        render_points_as_spheres=True,
        name=f"P{n}"
    )

    plotter.render()


plotter.iren.add_observer(
    "LeftButtonPressEvent",
    click
)


print()
print("======================================")
print("CUTTING PLANE(S)")
print("======================================")

print(
    "Select points:\n"
    "  1, 2, 3           -> plane A only (close the window\n"
    "                       after 3 points if a single cutting\n"
    "                       direction is enough)\n"
    "  1, 2, 3, 4, 5      -> plane A + plane B, perpendicular\n"
    "                       to A, both through point 1\n"
    "Plane B (if used) is automatically adjusted to be exactly\n"
    "perpendicular to plane A while still passing through\n"
    "point 1 and staying as close as possible to points 4, 5."
)

plotter.show()


if len(selected_points) not in (3, 5):

    raise RuntimeError(
        "3 points (plane A only) or 5 points "
        "(planes A and B) are required."
    )


use_plane_b = len(selected_points) == 5

if use_plane_b:
    p1, p2, p3, p4, p5 = selected_points
else:
    p1, p2, p3 = selected_points


# ============================================================
# 4. CALCULATE THE CUTTING PLANE(S)
# ============================================================

# --- Plane A (points 1, 2, 3) ---

v1 = p2 - p1
v2 = p3 - p1

normal_a = np.cross(v1, v2)

normal_a_length = np.linalg.norm(normal_a)

if normal_a_length < 1e-10:

    raise RuntimeError(
        "Points 1, 2, 3 are almost collinear."
    )

normal_a /= normal_a_length

u_a = v1 / np.linalg.norm(v1)

v_a = np.cross(normal_a, u_a)
v_a /= np.linalg.norm(v_a)


# --- Plane B (points 1, 4, 5), forced perpendicular to A ---
#     only if the user selected 5 points; otherwise plane B is
#     simply not used and sections are only cut along plane A.

if use_plane_b:

    w1 = p4 - p1
    w2 = p5 - p1

    normal_b_raw = np.cross(w1, w2)

    if np.linalg.norm(normal_b_raw) < 1e-10:

        raise RuntimeError(
            "Points 1, 4, 5 are almost collinear."
        )

    # Remove any component along normal_a so the two planes are
    # exactly perpendicular (their normals are orthogonal), while
    # staying as close as possible to what the user clicked.
    normal_b = (
        normal_b_raw
        -
        np.dot(normal_b_raw, normal_a) * normal_a
    )

    normal_b_length = np.linalg.norm(normal_b)

    if normal_b_length < 1e-6:

        raise RuntimeError(
            "Points 4, 5 do not define a direction distinct enough "
            "from plane A to build a perpendicular plane B. "
            "Please re-run and pick points 4, 5 further from the "
            "plane A orientation."
        )

    normal_b /= normal_b_length

    # u_b: component of w1 orthogonal to normal_b, so plane B's
    # in-plane axis stays close to the user's click direction.
    u_b_raw = w1 - np.dot(w1, normal_b) * normal_b

    if np.linalg.norm(u_b_raw) < 1e-10:
        u_b_raw = w2 - np.dot(w2, normal_b) * normal_b

    u_b = u_b_raw / np.linalg.norm(u_b_raw)

    v_b = np.cross(normal_b, u_b)
    v_b /= np.linalg.norm(v_b)


print()
print("======================================")

if use_plane_b:
    print("TWO PERPENDICULAR CUTTING PLANES")
else:
    print("ONE CUTTING PLANE (plane A only)")

print("======================================")

print("P1 =", p1, " (shared by both planes)" if use_plane_b else "")
print("P2 =", p2)
print("P3 =", p3)

if use_plane_b:

    print("P4 =", p4)
    print("P5 =", p5)

print()
print("Plane A normal:", normal_a)

if use_plane_b:

    print("Plane B normal:", normal_b)

    print(
        f"\nPerpendicularity check (should be ~0): "
        f"{np.dot(normal_a, normal_b):.6f}"
    )


# ============================================================
# 5. ASK FOR SECTION PARAMETERS (single form window)
# ============================================================
#
# All the parameters that used to be a chain of separate popups
# are gathered into one scrollable form, so everything can be
# reviewed and adjusted together before the (long) extraction
# process starts. Fields shown depend on whether plane B is in
# use (found in step 4).

def ask_all_parameters(with_plane_b):
    """
    Shows one scrollable form window with every section /
    smoothing / reconstruction parameter. Returns a dict of
    validated values, or None if the user cancelled.
    """

    result = {}
    submitted = {"ok": False}

    form = tk.Toplevel(root)
    form.title("Section & reconstruction parameters")
    form.geometry("680x700")
    form.grab_set()

    canvas = tk.Canvas(form)
    scrollbar = tk.Scrollbar(form, orient="vertical", command=canvas.yview)
    scroll_frame = tk.Frame(canvas)

    scroll_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    row = [0]

    def add_title(text):

        tk.Label(
            scroll_frame,
            text=text,
            font=("TkDefaultFont", 10, "bold")
        ).grid(
            row=row[0], column=0, columnspan=2,
            sticky="w", padx=10, pady=(16, 2)
        )

        row[0] += 1

    def add_help(text):

        tk.Label(
            scroll_frame,
            text=text,
            justify="left",
            wraplength=600,
            fg="gray25"
        ).grid(
            row=row[0], column=0, columnspan=2,
            sticky="w", padx=10, pady=(0, 4)
        )

        row[0] += 1

    def add_field(label_text, default):

        tk.Label(
            scroll_frame,
            text=label_text
        ).grid(
            row=row[0], column=0, sticky="w", padx=(10, 5), pady=3
        )

        var = tk.StringVar(value=str(default))

        tk.Entry(
            scroll_frame,
            textvariable=var,
            width=12
        ).grid(
            row=row[0], column=1, sticky="w", padx=(0, 10), pady=3
        )

        row[0] += 1

        return var

    add_title("Plane A sections")

    add_help(
        "Object extent (mm) swept by planes 1-2-3, and the "
        "number of sections to cut along it."
    )

    width_a_var = add_field("Width A (mm):", 100.0)
    count_a_var = add_field("Number of sections A:", 11)

    width_b_var = None
    count_b_var = None

    if with_plane_b:

        add_title("Plane B sections")

        add_help(
            "Object extent (mm) swept by planes 1-4-5 "
            "(perpendicular to plane A), and the number of "
            "sections to cut along it."
        )

        width_b_var = add_field("Width B (mm):", 100.0)
        count_b_var = add_field("Number of sections B:", 11)

    add_title("Curve smoothing")

    add_help(
        "0 = none, 30 = very aggressive. Removes zigzags caused "
        "by scan noise or fabric folds. Start low (3-8) and "
        "increase if curves still look jagged."
    )

    smoothing_var = add_field("Smoothing strength:", 5)

    add_title("Ideal curve fitting (optional)")

    add_help(
        "0 = disabled. Fits ONE smooth continuous curve through "
        "each section, ignoring local fold dents instead of "
        "just softening them. Try 10-30 for soft/deformable "
        "objects; the effect saturates around 50. Too high will "
        "flatten real curvature (visor edge, brim, etc)."
    )

    ideal_var = add_field("Ideal curve strength:", 0)

    add_title("Piecewise polynomial reconstruction (optional)")

    add_help(
        "Used only for curves/portions marked later by clicking "
        "(method 1). Each portion's degree starts at 1 (its two "
        "fixed endpoints) and increases until it reaches the "
        "target R^2, or the max degree, whichever first. "
        "3-6 degree and 0.95-0.99 R^2 are reasonable starting "
        "points."
    )

    degree_var = add_field("Max polynomial degree:", 5)
    r2_var = add_field("Target R^2 (0-1):", 0.98)

    method_var = tk.IntVar(value=1)

    if with_plane_b:

        add_title("Reconstruction method")

        add_help(
            "1 = Interactive polynomial: click to split each "
            "curve into portions (settings above). Curves from "
            "plane A and plane B only pass CLOSE to each other "
            "where they cross, not through an identical point.\n"
            "2 = Simple spline through intersections: every "
            "A x B crossing is found first; each curve is then "
            "replaced by a single spline through just its two "
            "endpoints and its intersection points, in order. "
            "Both curves at a crossing share the exact same "
            "point, so SolidWorks sees a true, formal "
            "intersection there -- at the cost of no longer "
            "hugging the scanned shape in between."
        )

        rb_frame = tk.Frame(scroll_frame)

        rb_frame.grid(
            row=row[0], column=0, columnspan=2,
            sticky="w", padx=10
        )

        tk.Radiobutton(
            rb_frame,
            text="1 - Interactive polynomial",
            variable=method_var,
            value=1
        ).pack(anchor="w")

        tk.Radiobutton(
            rb_frame,
            text="2 - Simple spline through intersections",
            variable=method_var,
            value=2
        ).pack(anchor="w")

        row[0] += 1

    def on_submit():

        try:

            result["width_a"] = float(width_a_var.get())

            if result["width_a"] < 1.0:
                raise ValueError("Width A must be >= 1 mm.")

            result["number_of_sections_a"] = int(count_a_var.get())

            if not (2 <= result["number_of_sections_a"] <= 101):
                raise ValueError(
                    "Number of sections A must be between 2 and 101."
                )

            if with_plane_b:

                result["width_b"] = float(width_b_var.get())

                if result["width_b"] < 1.0:
                    raise ValueError("Width B must be >= 1 mm.")

                result["number_of_sections_b"] = int(count_b_var.get())

                if not (2 <= result["number_of_sections_b"] <= 101):
                    raise ValueError(
                        "Number of sections B must be between 2 and 101."
                    )

            else:

                result["width_b"] = None
                result["number_of_sections_b"] = None

            result["smoothing_strength"] = int(smoothing_var.get())

            if not (0 <= result["smoothing_strength"] <= 50):
                raise ValueError(
                    "Smoothing strength must be between 0 and 50."
                )

            result["ideal_curve_strength"] = int(ideal_var.get())

            if not (0 <= result["ideal_curve_strength"] <= 100):
                raise ValueError(
                    "Ideal curve strength must be between 0 and 100."
                )

            result["max_polynomial_degree"] = int(degree_var.get())

            if not (1 <= result["max_polynomial_degree"] <= 15):
                raise ValueError(
                    "Max polynomial degree must be between 1 and 15."
                )

            result["r2_target"] = float(r2_var.get())

            if not (0.0 <= result["r2_target"] <= 1.0):
                raise ValueError(
                    "Target R^2 must be between 0 and 1."
                )

            result["reconstruction_method"] = (
                method_var.get() if with_plane_b else 1
            )

        except ValueError as exc:

            messagebox.showerror("Invalid input", str(exc))
            return

        submitted["ok"] = True
        form.destroy()

    def on_cancel():
        form.destroy()

    button_frame = tk.Frame(scroll_frame)

    button_frame.grid(
        row=row[0], column=0, columnspan=2, pady=18
    )

    tk.Button(
        button_frame, text="OK", width=10, command=on_submit
    ).pack(side="left", padx=5)

    tk.Button(
        button_frame, text="Cancel", width=10, command=on_cancel
    ).pack(side="left", padx=5)

    form.protocol("WM_DELETE_WINDOW", on_cancel)

    root.wait_window(form)

    return result if submitted["ok"] else None


parameters = ask_all_parameters(use_plane_b)

if parameters is None:
    raise SystemExit

width_a = parameters["width_a"]
number_of_sections_a = parameters["number_of_sections_a"]
width_b = parameters["width_b"]
number_of_sections_b = parameters["number_of_sections_b"]
smoothing_strength = parameters["smoothing_strength"]
ideal_curve_strength = parameters["ideal_curve_strength"]
max_polynomial_degree = parameters["max_polynomial_degree"]
r2_target = parameters["r2_target"]
reconstruction_method = parameters["reconstruction_method"]


print()
print("======================================")
print("SECTION PARAMETERS")
print("======================================")

print(
    f"Plane A width      : {width_a:.2f} mm "
    f"({number_of_sections_a} sections)"
)

if use_plane_b:

    print(
        f"Plane B width      : {width_b:.2f} mm "
        f"({number_of_sections_b} sections)"
    )

else:

    print(
        "Plane B           : not used (3 points were selected)"
    )

print(
    f"Smoothing strength: {smoothing_strength}"
)

print(
    f"Ideal curve strength: {ideal_curve_strength}"
)

print(
    f"Max polynomial degree (reconstruction): {max_polynomial_degree}"
)

print(
    f"Target R^2 (reconstruction): {r2_target}"
)

if use_plane_b:

    method_name = (
        "Interactive polynomial"
        if reconstruction_method == 1
        else "Simple spline through intersections"
    )

    print(
        f"Reconstruction method: {reconstruction_method} "
        f"({method_name})"
    )


# ============================================================
# 5bis. PLANE x PLANE x MESH INTERSECTION POINTS (validation)
# ============================================================
#
# Two perpendicular cutting planes (one from each family) share
# a 3D line. Where that line pierces the actual scanned mesh
# tells us WHICH (A, B) plane pairs genuinely cross the object
# at all, and how many times (a fold/curl can make a pair cross
# more than once). This is used later purely as a lookup: which
# pairs of FINAL curves to check, and how many distinct
# crossings to expect between them (step 12ter). It does not
# feed any point into a curve -- see step 12ter for why.


def compute_plane_intersections(
    mesh_obj,
    shared_point,
    normal_a_vec,
    normal_b_vec,
    offsets_a_list,
    offsets_b_list
):
    """
    For every pair of (plane A at offset i, plane B at offset
    j), computes the 3D line the two planes share, then finds
    every point where that line crosses the mesh surface via
    ray tracing. Returns a dict keyed by (i, j) (1-based section
    numbers) -> list of ALL intersection points found (usually
    one, but can be more where the surface folds back on itself,
    e.g. a curled brim). These are only used downstream as a
    validation gate and an expected crossing COUNT: the actual
    location used for each crossing is found later, directly on
    the final (smoothed/reconstructed) curves (step 12ter), so
    keeping every raw hit here is safe -- there is no risk of
    forcing the wrong one onto a curve, unlike when this used to
    insert points before any curve processing.
    """

    line_direction = np.cross(normal_a_vec, normal_b_vec)
    line_direction /= np.linalg.norm(line_direction)

    ray_half_length = mesh_obj.length * 1.5

    intersections = {}

    for i, offset_a in enumerate(offsets_a_list, start=1):

        origin_a = shared_point + offset_a * normal_a_vec

        d_a = np.dot(normal_a_vec, origin_a)

        for j, offset_b in enumerate(offsets_b_list, start=1):

            origin_b = shared_point + offset_b * normal_b_vec

            d_b = np.dot(normal_b_vec, origin_b)

            # A point on the shared line (valid since normal_a
            # and normal_b are orthonormal).
            line_point = (
                d_a * normal_a_vec
                +
                d_b * normal_b_vec
            )

            ray_start = line_point - ray_half_length * line_direction
            ray_end = line_point + ray_half_length * line_direction

            hits, _ = mesh_obj.ray_trace(
                ray_start,
                ray_end,
                first_point=False
            )

            if len(hits) == 0:
                continue

            intersections[(i, j)] = hits

    return intersections


plane_intersections = {}

if use_plane_b:

    print()
    print("======================================")
    print("COMPUTING PLANE x PLANE x MESH INTERSECTIONS")
    print("======================================")

    offsets_a_precompute = np.linspace(
        -width_a / 2,
        width_a / 2,
        number_of_sections_a
    )

    offsets_b_precompute = np.linspace(
        -width_b / 2,
        width_b / 2,
        number_of_sections_b
    )

    plane_intersections = compute_plane_intersections(
        mesh,
        p1,
        normal_a,
        normal_b,
        offsets_a_precompute,
        offsets_b_precompute
    )

    total_hits = sum(
        len(hits) for hits in plane_intersections.values()
    )

    multi_hit_pairs = sum(
        1 for hits in plane_intersections.values() if len(hits) > 1
    )

    print(
        f"  {len(plane_intersections)} plane pair(s) with at "
        f"least one crossing out of "
        f"{number_of_sections_a * number_of_sections_b} "
        f"plane pairs."
    )

    print(
        f"  {total_hits} individual mesh crossing(s) found "
        f"({multi_hit_pairs} pair(s) cross more than once)."
    )


# ============================================================
# 8. OUTPUT DIRECTORY
# ============================================================

base_dir = os.path.dirname(
    stl_path
)

base_name = os.path.splitext(
    os.path.basename(stl_path)
)[0]


output_dir = os.path.join(
    base_dir,
    base_name + "_sections"
)

os.makedirs(
    output_dir,
    exist_ok=True
)


# ============================================================
# 9. RESAMPLING FUNCTION
# ============================================================

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


def _insert_points_into_curve(curve_points, points_to_insert):
    """
    Inserts each point in `points_to_insert` into the ordered
    polyline `curve_points`, at whichever existing segment it is
    geometrically closest to, WITHOUT moving any existing point.
    Used to force a curve to pass exactly through known
    A x B intersection points.

    Returns (new_curve, inserted_indices) where
    inserted_indices[k] is the index, in the returned array, of
    points_to_insert[k].
    """

    curve = [
        np.asarray(p, dtype=float)
        for p in curve_points
    ]

    inserted_indices = [None] * len(points_to_insert)

    for original_order in range(len(points_to_insert)):

        point = np.asarray(
            points_to_insert[original_order],
            dtype=float
        )

        best_seg = 0
        best_dist = np.inf

        for i in range(len(curve) - 1):

            a = curve[i]
            b = curve[i + 1]

            ab = b - a
            denom = np.dot(ab, ab)

            t = (
                0.0
                if denom < 1e-12
                else np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0)
            )

            proj = a + t * ab

            dist = np.linalg.norm(point - proj)

            if dist < best_dist:
                best_dist = dist
                best_seg = i

        insert_at = best_seg + 1

        curve.insert(insert_at, point)

        inserted_indices[original_order] = insert_at

        for k in range(len(points_to_insert)):

            if (
                inserted_indices[k] is not None
                and k != original_order
                and inserted_indices[k] >= insert_at
            ):

                inserted_indices[k] += 1

    return np.asarray(curve), inserted_indices


# ============================================================
# 9bis. STITCH FRAGMENTS SEPARATED BY FOLDS / DEFECTS
# ============================================================

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


# ============================================================
# 9ter. CURVE SMOOTHING (user-controlled aggressiveness)
# ============================================================

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


# ============================================================
# 9quater. IDEAL CURVE FITTING (optional)
# ============================================================
#
# smooth_curve() (above) removes local noise but still follows
# every real dip of the curve, including genuine fold artefacts.
# fit_ideal_curve() instead fits a single smooth B-spline
# THROUGH the general trend of the points, so a wrinkle in a
# soft fabric surface is absorbed into one continuous, "ideal"
# curve instead of being reproduced. This is a global, more
# aggressive idealisation and is meant to be used in addition
# to (after) smooth_curve(), and only when needed (folds,
# soft/deformable objects) -- hence it is optional and off by
# default.

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


# ============================================================
# ============================================================
# 9quinquies. PIECEWISE POLYNOMIAL RECONSTRUCTION (optional)
# ============================================================
#
# The user clicks directly on a curve (step 12bis) to mark
# SPLIT POINTS. Each click divides the curve into one more
# portion. A curve that is never clicked is left completely
# untouched. A curve that IS clicked gets entirely rebuilt,
# portion by portion, each portion independently approximated
# by a polynomial of its own points (as a function of arc
# length within that portion).
#
# For each portion, the degree starts at 0 (a single average
# point) and is increased one step at a time until either:
#   - the fit reaches the user-supplied minimum R^2, or
#   - the user-supplied maximum degree n is reached (in which
#     case the best achieved fit is kept and reported).
#
# This lets the user isolate locally-homogeneous regions (e.g.
# the visor vs. the crown vs. a sagging rear) into their own
# portions, each smoothed on its own terms, instead of one
# global fit or a fragile trend extrapolated into an unrelated
# region. Nothing here assumes any particular object shape.


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

    boundaries = [0] + interior + [n - 1]

    if len(boundaries) < 2:
        return pts, []

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




# ============================================================
# 10. EXTRACT CURVES FROM SECTION
# ============================================================

def extract_curves(section):

    lines = section.lines

    if lines is None or len(lines) == 0:
        return []

    lines = lines.reshape(
        -1,
        3
    )

    edges = []

    for _, a, b in lines:

        a = int(a)
        b = int(b)

        if a != b:

            edges.append(
                (a, b)
            )


    # --------------------------------------------------------
    # Build graph
    # --------------------------------------------------------

    neighbors = {}

    for a, b in edges:

        neighbors.setdefault(
            a,
            []
        ).append(b)

        neighbors.setdefault(
            b,
            []
        ).append(a)


    # --------------------------------------------------------
    # Find endpoints
    # --------------------------------------------------------

    endpoints = [
        node
        for node, neigh
        in neighbors.items()
        if len(neigh) == 1
    ]


    # --------------------------------------------------------
    # Follow paths
    # --------------------------------------------------------

    def follow(start):

        # ------------------------------------------------------
        # A un pli / defaut de surface, plusieurs aretes se
        # rejoignent parfois au meme point (noeud "en branche",
        # non filaire). Prendre arbitrairement la premiere
        # branche coupait la courbe en plein milieu. On choisit
        # ici la branche qui prolonge le plus "tout droit" la
        # direction deja suivie, ce qui reste sur le bon trajet
        # meme au niveau d'un pli.
        # ------------------------------------------------------

        path = []

        current = start
        previous = None

        while True:

            path.append(
                current
            )

            possible = [
                n
                for n in neighbors[current]
                if n != previous
                and n not in path
            ]

            if not possible:
                break

            if len(possible) == 1 or previous is None:

                current, previous = possible[0], current

            else:

                prev_dir = (
                    section.points[current]
                    -
                    section.points[previous]
                )

                prev_norm = np.linalg.norm(prev_dir)

                if prev_norm > 1e-12:
                    prev_dir = prev_dir / prev_norm

                best_candidate = None
                best_score = -np.inf

                for candidate in possible:

                    cand_dir = (
                        section.points[candidate]
                        -
                        section.points[current]
                    )

                    cand_norm = np.linalg.norm(cand_dir)

                    if cand_norm > 1e-12:
                        cand_dir = cand_dir / cand_norm

                    # score proche de 1 = continuation en ligne
                    # droite, proche de -1 = demi-tour
                    score = np.dot(prev_dir, cand_dir)

                    if score > best_score:
                        best_score = score
                        best_candidate = candidate

                current, previous = best_candidate, current

        return path


    paths = []

    used = set()


    # Open curves first
    for start in endpoints:

        if start in used:
            continue

        path = follow(start)

        for node in path:
            used.add(node)

        if len(path) >= 4:

            paths.append(
                path
            )


    # Remaining curves
    remaining = [
        n
        for n in neighbors
        if n not in used
    ]

    for start in remaining:

        if start in used:
            continue

        path = follow(start)

        for node in path:
            used.add(node)

        if len(path) >= 4:

            paths.append(
                path
            )


    curves = []

    for path in paths:

        curves.append(
            section.points[path]
        )


    # --------------------------------------------------------
    # Stitch fragments broken by folds / scan defects
    #
    # A pli or a small mesh defect can genuinely leave two
    # separate line fragments in the graph (no shared vertex)
    # even though the surface is continuous. If two fragment
    # endpoints are spatially close (< STITCH_TOLERANCE_MM),
    # they are almost certainly the same curve and are merged.
    # --------------------------------------------------------

    curves = stitch_curve_fragments(
        curves,
        STITCH_TOLERANCE_MM
    )


    # --------------------------------------------------------
    # Discard short curves (noise loops / islands)
    # --------------------------------------------------------

    filtered_curves = []

    for c in curves:

        curve_length = np.sum(
            np.linalg.norm(
                np.diff(c, axis=0),
                axis=1
            )
        ) if len(c) >= 2 else 0.0

        if curve_length >= MIN_CURVE_LENGTH_MM:

            filtered_curves.append(c)

    return filtered_curves


# ============================================================
# 11. EXTRACT ALL SECTIONS (for one cutting direction)
# ============================================================
#
# This is called once per cutting plane (A and B), each with
# its own origin point, normal, in-plane basis (u, v), width
# and section count. A single family of parallel sections in
# only one direction is not enough to constrain a Loft/Boundary
# Surface well (it only has cross-sections one way); running
# this twice, once per perpendicular plane, produces a proper
# two-direction curve network (like stations + buttock lines in
# hull lofting) for SolidWorks to loft against in both
# directions.

def extract_all_sections_for_direction(
    direction_label,
    direction_origin_point,
    direction_normal,
    direction_u,
    direction_v,
    direction_width,
    direction_count
):

    direction_offsets = np.linspace(
        -direction_width / 2,
        direction_width / 2,
        direction_count
    )

    direction_data = []

    for section_number, offset in enumerate(
        direction_offsets,
        start=1
    ):

        print()
        print("--------------------------------------")

        print(
            f"PLANE {direction_label} - SECTION "
            f"{section_number}/{direction_count}"
        )

        print(
            f"Offset = {offset:+.2f} mm"
        )


        origin = (
            direction_origin_point
            +
            offset * direction_normal
        )


        section = mesh.slice(
            normal=direction_normal,
            origin=origin
        )

        section = section.clean()


        if section.n_points == 0:

            print(
                "No intersection."
            )

            continue


        print(
            f"Intersection points : "
            f"{section.n_points:,}"
        )


        curves = extract_curves(
            section
        )


        print(
            f"Curves detected : "
            f"{len(curves)}"
        )


        section_curves = []


        # ------------------------------------------------------
        # Analyse curves
        # ------------------------------------------------------
        #
        # No A x B intersection point is forced into a curve at
        # this stage anymore. Doing so before smoothing/fitting
        # imposed a rigid boundary condition that the rest of the
        # curve then had to bend around, and a second forced
        # snap right before export undid whatever shape smoothing
        # had settled into -- both visible as kinks/twisting.
        # Curves are now smoothed and (optionally) reconstructed
        # completely on their own terms; the two curve families
        # are only brought together at the very end, once both
        # are in their final shape (see step 13bis).

        for curve_number, curve in enumerate(
            curves,
            start=1
        ):

            length = 0.0

            for i in range(
                1,
                len(curve)
            ):

                length += np.linalg.norm(
                    curve[i]
                    -
                    curve[i - 1]
                )


            print(
                f"  Curve {curve_number}: "
                f"{len(curve):,} points, "
                f"length = {length:.2f} mm"
            )


            # ----------------------------------------------------
            # Smooth (removes zigzags from scan noise / folds)
            # ----------------------------------------------------

            curve_smoothed = smooth_curve(
                curve,
                iterations=smoothing_strength
            )


            # ----------------------------------------------------
            # Ideal curve fitting (optional, off by default):
            # replaces local fold dents with one smooth continuous
            # curve, for soft/deformable objects.
            # ----------------------------------------------------

            curve_smoothed, _ = fit_ideal_curve(
                curve_smoothed,
                strength=ideal_curve_strength,
                avg_edge_length=AVG_EDGE_LENGTH
            )


            # ----------------------------------------------------
            # Project into 2D (this direction's own basis)
            # ----------------------------------------------------

            curve_2d = []

            for point in curve_smoothed:

                x = np.dot(
                    point - origin,
                    direction_u
                )

                y = np.dot(
                    point - origin,
                    direction_v
                )

                curve_2d.append(
                    [x, y]
                )


            section_curves.append(
                {
                    "points_2d": np.asarray(curve_2d),
                    "points_3d": np.asarray(curve_smoothed),
                    "length": length
                }
            )


        # --------------------------------------------------------
        # Identify main curve (longest curve of the section)
        # --------------------------------------------------------

        main_curve_index = None

        if section_curves:

            main_curve_index = max(
                range(len(section_curves)),
                key=lambda i: section_curves[i]["length"]
            )

            print(
                f"  -> Main curve = Curve "
                f"{main_curve_index + 1} "
                f"({section_curves[main_curve_index]['length']:.2f} mm)"
            )

        else:

            print(
                "  -> No curve detected for this section."
            )


        direction_data.append(
            {
                "direction": direction_label,
                "number": section_number,
                "offset": offset,
                "origin": origin,
                "u": direction_u,
                "v": direction_v,
                "curves": section_curves,
                "main_curve_index": main_curve_index
            }
        )

    return direction_data


print()
print("======================================")
print("EXTRACTING PLANE A SECTIONS")
print("======================================")

all_section_data_a = extract_all_sections_for_direction(
    "A",
    p1,
    normal_a,
    u_a,
    v_a,
    width_a,
    number_of_sections_a
)

all_section_data_b = []

if use_plane_b:

    print()
    print("======================================")
    print("EXTRACTING PLANE B SECTIONS")
    print("======================================")

    all_section_data_b = extract_all_sections_for_direction(
        "B",
        p1,
        normal_b,
        u_b,
        v_b,
        width_b,
        number_of_sections_b
    )

all_section_data = all_section_data_a + all_section_data_b


# ============================================================
# 12ter. FIND A x B INTERSECTIONS ON THE FINAL CURVES
# ============================================================
#
# Curves have now been smoothed and (optionally) reconstructed
# completely on their own terms, with no intersection-related
# constraint at all. For every plane pair validated in step
# 5bis, this looks for that many distinct closest-approach
# locations between the FINAL curve A_i and the FINAL curve B_j
# -- purely to FIND and RECORD where the two curve families
# already meet (or nearly meet), for reference (see the
# intersection markers exported in step 13.4).
#
# Nothing here modifies the curves. Snapping the two closest
# points to a shared midpoint was tried, but even a small nudge
# repeated at every crossing along a curve that has many of them
# (a single A-curve can cross a dozen+ B-curves) reintroduces
# exactly the small kinks/twisting this was meant to avoid --
# the curves are already extremely close at these locations
# (typically well under a millimetre) precisely because both
# were independently smoothed from the same underlying surface,
# so forcing them together adds distortion without adding real
# accuracy.

def _find_curve_crossings(
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


def find_all_intersections(all_section_data, plane_intersections):
    """
    For every plane pair validated in step 5bis, looks for that
    many distinct closest-approach locations between the CURRENT
    curve A_i and curve B_j (whatever state they are in when this
    is called). Returns a list of
    {"pair_id": (i, j, k), "point": ..., "gap": ...}.
    Does not modify any curve.
    """

    crossing_max_gap_mm = (
        CROSSING_MAX_GAP_FACTOR
        *
        AVG_EDGE_LENGTH
    )

    curves_by_key = {}

    for data in all_section_data:

        main_index = data["main_curve_index"]

        if main_index is None:
            continue

        curves_by_key[(data["direction"], data["number"])] = (
            data,
            data["curves"][main_index]
        )

    results = []

    for (i, j), hit_list in plane_intersections.items():

        key_a = ("A", i)
        key_b = ("B", j)

        if key_a not in curves_by_key or key_b not in curves_by_key:
            continue

        _, curve_info_a = curves_by_key[key_a]
        _, curve_info_b = curves_by_key[key_b]

        pts_a = curve_info_a["points_3d"]
        pts_b = curve_info_b["points_3d"]

        if len(pts_a) < 2 or len(pts_b) < 2:
            continue

        expected_count = len(hit_list)

        crossings = _find_curve_crossings(
            pts_a,
            pts_b,
            max_count=expected_count,
            max_gap=crossing_max_gap_mm
        )

        if expected_count > 1:

            print(
                f"  {('A', i)} x {('B', j)}: "
                f"{expected_count} crossing(s) expected, "
                f"{len(crossings)} found"
            )

        for k, (idx_a, idx_b, gap) in enumerate(crossings):

            reference_point = (
                pts_a[idx_a] + pts_b[idx_b]
            ) / 2.0

            results.append(
                {
                    "pair_id": (i, j, k),
                    "point": reference_point,
                    "gap": gap
                }
            )

    return results


CROSSING_MAX_GAP_FACTOR = 10.0

# Initialised once, here, before either reconstruction method
# runs: method 2 fills this in directly below (it needs the
# intersections anyway to build its curves); method 1 fills it
# in later, after the interactive correction step, purely for
# reporting (see the end of step 12).
found_intersections = []


# ============================================================
# 11bis. SIMPLE SPLINE THROUGH INTERSECTIONS (method 2)
# ============================================================
#
# Alternative to the interactive polynomial method (still
# available as method 1, see step 12). Here, every A x B
# crossing is found FIRST, directly on the smoothed curves --
# then each main curve is entirely replaced by a minimal curve
# through just its two endpoints and its own intersection
# points, in order along the curve. Since curve A_i and curve
# B_j both get the EXACT SAME point object at a shared crossing,
# and both are exported as interpolating splines (fit_points),
# SolidWorks sees a true, formal intersection there -- not two
# curves merely passing close to each other. The trade-off: the
# curve no longer hugs the scanned shape between those points,
# only a smooth spline through them.

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


if use_plane_b and reconstruction_method == 2:

    print()
    print("======================================")
    print("SIMPLE SPLINE THROUGH INTERSECTIONS")
    print("======================================")

    found_intersections = find_all_intersections(
        all_section_data,
        plane_intersections
    )

    per_curve_points = {}

    for entry in found_intersections:

        i, j, _ = entry["pair_id"]

        point = entry["point"]

        per_curve_points.setdefault(("A", i), []).append(point)
        per_curve_points.setdefault(("B", j), []).append(point)

    rebuilt_count = 0

    for data in all_section_data:

        main_index = data["main_curve_index"]

        if main_index is None:
            continue

        curve_info = data["curves"][main_index]

        assigned = per_curve_points.get(
            (data["direction"], data["number"]),
            []
        )

        if not assigned:
            continue

        new_points_3d = build_simple_spline_curve(
            curve_info["points_3d"],
            assigned
        )

        origin = data["origin"]
        local_u = data["u"]
        local_v = data["v"]

        new_points_2d = np.array(
            [
                [
                    np.dot(point - origin, local_u),
                    np.dot(point - origin, local_v)
                ]
                for point in new_points_3d
            ]
        )

        curve_info["points_3d"] = new_points_3d
        curve_info["points_2d"] = new_points_2d

        rebuilt_count += 1

    print(
        f"  {len(found_intersections)} intersection point(s) "
        f"found; {rebuilt_count} curve(s) rebuilt as a simple "
        f"spline through their endpoints + intersections."
    )


# ============================================================
# 12. SECTION PREVIEW + OPTIONAL PIECEWISE RECONSTRUCTION
#     (method 1 only)
# ============================================================
#
# This entire interactive step only applies to reconstruction
# method 1. Method 2 already gave every main curve its final,
# exact shape in step 11bis (a spline through its endpoints and
# its own A x B intersection points) -- opening this window
# regardless of method would let a click trigger a polynomial
# reconstruction that does not know about those shared points,
# undoing the whole point of choosing method 2.
#
# A single window both shows the sections that were created AND
# lets you correct them, so there is no separate read-only
# preview window to look at first.
#
# Click directly on a curve to add a SPLIT POINT. Each curve
# you click at least once gets entirely rebuilt as a sequence
# of independently-fitted polynomial portions, split at the
# points you clicked (see reconstruct_curve_piecewise). A curve
# you never click is left completely untouched, so clicking is
# entirely optional and safe to skip.
#
# Click near an existing split point (on the same curve) to
# remove it instead of adding a duplicate.

if reconstruction_method == 1:

    print()
    print("======================================")
    print("SECTION PREVIEW / OPTIONAL RECONSTRUCTION")
    print("======================================")

    print(
        "All created sections are shown below (main curve per\n"
        "section).\n"
        "Click on a curve to add a split point (red marker).\n"
        "Click near an existing split point to remove it.\n"
        "Any curve with at least one split point will be rebuilt as\n"
        "independently-fitted polynomial portions between your\n"
        "split points (e.g. put one split where the visor ends and\n"
        "the crown begins, so they are not fitted together).\n"
        "Curves you never click are left untouched.\n"
        "Close the window when done."
    )

    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "cyan",
        "yellow",
        "pink",
        "brown"
    ]


    correction_plotter = pv.Plotter(
        window_size=(1500, 950)
    )

    correction_plotter.add_mesh(
        mesh,
        color="lightgray",
        opacity=0.12,
        show_edges=False
    )

    correction_plotter.add_text(
        "SECTION CURVES - click to add/remove split points (optional)",
        position="upper_left",
        font_size=16
    )


    curve_actor_map = {}

    for section_index, data in enumerate(all_section_data):

        main_index = data["main_curve_index"]

        if main_index is None:
            continue

        points_3d = data["curves"][main_index]["points_3d"]

        if len(points_3d) < 2:
            continue

        color = colors[
            section_index
            % len(colors)
        ]

        line = pv.lines_from_points(
            points_3d,
            close=False
        )

        actor = correction_plotter.add_mesh(
            line,
            color=color,
            line_width=6
        )

        curve_actor_map[actor] = {
            "section_index": section_index,
            "points": points_3d
        }

        label_point = points_3d[
            len(points_3d) // 2
        ]

        correction_plotter.add_point_labels(
            np.array([label_point]),
            [
                f"{data['direction']}{data['number']} (main)"
            ],
            point_size=1,
            font_size=14,
            shape=None
        )


    correction_picker = vtk.vtkCellPicker()
    correction_picker.SetTolerance(0.005)
    correction_picker.PickFromListOn()

    for actor in curve_actor_map:
        correction_picker.AddPickList(actor)


    # section_index -> sorted list of split point curve-indices
    section_splits = {}


    def _redraw_split_markers(section_index):

        points = curve_actor_map_by_section[section_index]

        correction_plotter.remove_actor(
            f"split_markers_{section_index}",
            render=False
        )

        idxs = section_splits.get(section_index, [])

        if not idxs:
            return

        marker_points = points[idxs]

        marker = pv.PolyData(marker_points)

        correction_plotter.add_mesh(
            marker,
            color="red",
            point_size=16,
            render_points_as_spheres=True,
            name=f"split_markers_{section_index}"
        )


    curve_actor_map_by_section = {
        info["section_index"]: info["points"]
        for info in curve_actor_map.values()
    }


    REMOVE_CLICK_TOLERANCE_POINTS = 3  # in curve-index units


    def correction_click(caller, event):

        x, y = caller.GetEventPosition()

        correction_picker.Pick(
            x,
            y,
            0,
            correction_plotter.renderer
        )

        actor = correction_picker.GetActor()

        if actor is None or actor not in curve_actor_map:
            return

        info = curve_actor_map[actor]

        points = info["points"]

        section_index = info["section_index"]

        pick_pos = np.array(
            correction_picker.GetPickPosition()
        )

        distances = np.linalg.norm(
            points - pick_pos,
            axis=1
        )

        idx = int(
            np.argmin(distances)
        )

        splits = section_splits.setdefault(
            section_index,
            []
        )

        # Clicking near an existing split removes it (toggle).
        near_existing = [
            s
            for s in splits
            if abs(s - idx) <= REMOVE_CLICK_TOLERANCE_POINTS
        ]

        if near_existing:

            splits.remove(near_existing[0])

            section_label = (
                f"{all_section_data[section_index]['direction']}"
                f"{all_section_data[section_index]['number']}"
            )

            print(
                f"Section {section_label}: split point removed "
                f"(curve index {near_existing[0]})"
            )

        else:

            splits.append(idx)
            splits.sort()

            section_label = (
                f"{all_section_data[section_index]['direction']}"
                f"{all_section_data[section_index]['number']}"
            )

            print(
                f"Section {section_label}: split point added "
                f"(curve index {idx})"
            )

        _redraw_split_markers(section_index)

        correction_plotter.render()


    correction_plotter.iren.add_observer(
        "LeftButtonPressEvent",
        correction_click
    )

    correction_plotter.add_axes()

    correction_plotter.show()


    print()
    print("Applying piecewise reconstruction...")

    corrections_applied = 0

    for section_index, splits in section_splits.items():

        if len(splits) == 0:
            continue

        data = all_section_data[section_index]

        main_index = data["main_curve_index"]

        curve_info = data["curves"][main_index]

        reconstructed_3d, portions_info = reconstruct_curve_piecewise(
            curve_info["points_3d"],
            split_indices=splits,
            max_degree=max_polynomial_degree,
            r2_target=r2_target
        )

        if not portions_info:
            continue

        origin = data["origin"]
        local_u = data["u"]
        local_v = data["v"]

        reconstructed_2d = []

        for point in reconstructed_3d:

            x = np.dot(point - origin, local_u)
            y = np.dot(point - origin, local_v)

            reconstructed_2d.append([x, y])

        curve_info["points_3d"] = reconstructed_3d
        curve_info["points_2d"] = np.asarray(reconstructed_2d)

        corrections_applied += 1

        print(
            f"  Section {data['direction']}{data['number']}: rebuilt as "
            f"{len(portions_info)} portion(s):"
        )

        for lo, hi, degree_used, r2_achieved in portions_info:

            reached = "OK" if r2_achieved >= r2_target else "max degree reached"

            print(
                f"    indices [{lo}-{hi}]: "
                f"degree {degree_used}, "
                f"R^2 = {r2_achieved:.4f} ({reached})"
            )

    if corrections_applied == 0:
        print("  No curve was reconstructed (no split point was set).")
else:

    print()
    print(
        "Reconstruction method 2 (simple spline through "
        "intersections) already finalised every main curve in "
        "step 11bis -- skipping the interactive window so it "
        "cannot be accidentally overwritten."
    )


if use_plane_b and reconstruction_method == 1:

    print()
    print("======================================")
    print("FINDING A x B INTERSECTIONS (final curves)")
    print("======================================")

    found_intersections = find_all_intersections(
        all_section_data,
        plane_intersections
    )

    max_gap_found = max(
        (entry["gap"] for entry in found_intersections),
        default=0.0
    )

    print(
        f"  {len(found_intersections)} intersection point(s) "
        f"found on the final curves "
        f"(max residual gap between the two curves: "
        f"{max_gap_found:.4f} mm)."
    )

elif not use_plane_b:

    print()
    print(
        "Plane B not used - no A x B intersections to find."
    )


# ============================================================
# 13. EXPORT DXF

print()
print("======================================")
print("EXPORTING DXF FILES")
print("======================================")


all_dir = os.path.join(
    output_dir,
    "sections_all"
)

main_dir = os.path.join(
    output_dir,
    "sections_main"
)

dir3d = os.path.join(
    output_dir,
    "sections_3d"
)

os.makedirs(all_dir, exist_ok=True)
os.makedirs(main_dir, exist_ok=True)
os.makedirs(dir3d, exist_ok=True)


def make_spline_points(points_2d_or_3d, is_3d=False):
    """Resample a curve and return a list of ezdxf-ready 3-tuples."""

    if len(points_2d_or_3d) > 200:
        n = N_MAIN
    else:
        n = N_SMALL

    simplified = resample(
        points_2d_or_3d,
        n
    )

    if is_3d:

        return [
            (float(p[0]), float(p[1]), float(p[2]))
            for p in simplified
        ]

    else:

        return [
            (float(p[0]), float(p[1]), 0.0)
            for p in simplified
        ]


# --------------------------------------------------------
# 13bis. RESAMPLE MAIN CURVES FOR EXPORT
# --------------------------------------------------------
#
# Method 1 curves are resampled normally for a clean DXF spline.
# Method 2 curves must NOT be resampled: they are already the
# minimal, deliberate point set (endpoints + exact intersection
# points) built in step 11bis, and a fixed-count arc-length
# resample would interpolate a brand new set of points that,
# again, does not land back on those exact shared coordinates --
# undoing the whole point of method 2. They are exported exactly
# as built.

resampled_main_3d = {}

for data in all_section_data:

    main_index = data["main_curve_index"]

    if main_index is None:
        continue

    curve_info = data["curves"][main_index]

    curve_3d = curve_info["points_3d"]

    if reconstruction_method == 2:

        if len(curve_3d) < 2:
            continue

        resampled_main_3d[(data["direction"], data["number"])] = curve_3d

        continue

    if len(curve_3d) < 4:
        continue

    n_resample = N_MAIN if len(curve_3d) > 200 else N_SMALL

    resampled_main_3d[(data["direction"], data["number"])] = resample(
        curve_3d,
        n_resample
    )


# --------------------------------------------------------
# 13.1 sections_all : toutes les courbes de chaque section
# --------------------------------------------------------

print()
print("--- sections_all ---")

for data in all_section_data:

    offset = data["offset"]

    filename = (
        f"plane{data['direction']}_{offset:+07.2f}mm_all.dxf"
    )

    dxf_path = os.path.join(
        all_dir,
        filename
    )

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    for curve_info in data["curves"]:

        curve_2d = curve_info["points_2d"]

        if len(curve_2d) < 4:
            continue

        points = make_spline_points(curve_2d)

        msp.add_spline(fit_points=points)

    doc.saveas(dxf_path)

    print(dxf_path)


# --------------------------------------------------------
# 13.2 sections_main : uniquement la courbe principale
#      (la languette est exclue car ce n'est pas la
#      courbe la plus longue de la section)
# --------------------------------------------------------

print()
print("--- sections_main ---")

for data in all_section_data:

    offset = data["offset"]

    filename = (
        f"plane{data['direction']}_{offset:+07.2f}mm_main.dxf"
    )

    dxf_path = os.path.join(
        main_dir,
        filename
    )

    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    key = (data["direction"], data["number"])

    if key in resampled_main_3d:

        curve_3d_final = resampled_main_3d[key]

        origin = data["origin"]
        local_u = data["u"]
        local_v = data["v"]

        points = [
            (
                float(np.dot(p - origin, local_u)),
                float(np.dot(p - origin, local_v)),
                0.0
            )
            for p in curve_3d_final
        ]

        msp.add_spline(fit_points=points)

    doc.saveas(dxf_path)

    print(dxf_path)


# --------------------------------------------------------
# 13.3 sections_3d : un seul DXF avec toutes les courbes
#      principales assemblees dans leur position spatiale
#      reelle (utile pour verifier la geometrie avant
#      SolidWorks)
# --------------------------------------------------------

print()
print("--- sections_3d ---")

dxf_path_3d = os.path.join(
    dir3d,
    "sections_main_3d.dxf"
)

doc3d = ezdxf.new("R2010")
msp3d = doc3d.modelspace()

for data in all_section_data:

    key = (data["direction"], data["number"])

    if key not in resampled_main_3d:
        continue

    curve_3d_final = resampled_main_3d[key]

    points = [
        (float(p[0]), float(p[1]), float(p[2]))
        for p in curve_3d_final
    ]

    msp3d.add_spline(fit_points=points)

doc3d.saveas(dxf_path_3d)

print(dxf_path_3d)


# --------------------------------------------------------
# 13.4 intersections_reference : marqueurs (points) aux
#      endroits ou les courbes A et B se croisent le plus
#      pres, pour verification visuelle dans SolidWorks.
#      Purement informatif : les courbes elles-memes ne sont
#      jamais modifiees pour "coller" a ces points.
# --------------------------------------------------------

if use_plane_b and found_intersections:

    print()
    print("--- intersections_reference ---")

    dxf_path_intersections = os.path.join(
        dir3d,
        "intersections_reference.dxf"
    )

    doc_int = ezdxf.new("R2010")
    msp_int = doc_int.modelspace()

    for entry in found_intersections:

        i, j, k = entry["pair_id"]

        point = entry["point"]

        msp_int.add_point(
            (float(point[0]), float(point[1]), float(point[2]))
        )

        msp_int.add_text(
            f"A{i}xB{j}",
            height=AVG_EDGE_LENGTH
        ).set_placement(
            (float(point[0]), float(point[1]), float(point[2]))
        )

    doc_int.saveas(dxf_path_intersections)

    print(dxf_path_intersections)


# ============================================================
# 14. DONE
# ============================================================

print()
print("======================================")
print("DONE")
print("======================================")

print()
print("Section folder:")

print(
    output_dir
)

print()
print(
    "All requested sections have been exported."
)
