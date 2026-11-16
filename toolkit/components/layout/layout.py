"""A planar layout: metal polygons and junctions, wherever they came from.

A Quantum Metal design and a GDS file say the same thing in two dialects -- flat polygons on
layers. `Layout` is that common content, in millimetres and with nothing of the source left in
it, so that everything downstream (`Chip`, the mesh, the solver) is written once:

    Layout.from_metal(design, ...)       a Quantum Metal / Qiskit Metal `QDesign`
    Layout.from_gds("chip.gds", ...)     a GDS file
    Layout(layers={...}, ...)            shapely polygons you built yourself

A layout has

    layers      name -> metal on that layer (a shapely polygon or multipolygon)
    junctions   rectangles that bridge two pieces of metal, each with the direction of its current
    outline     (x0, y0, x1, y1): the substrate

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union

Bounds = tuple[float, float, float, float]

_UNITS_MM = {"nm": 1e-6, "um": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1e3}


@dataclass(frozen=True)
class Junction:
    """A rectangle between two pieces of metal, and the way the current crosses it."""
    name: str
    polygon: Polygon
    direction: tuple[float, float]       # unit vector in the layout plane

    @property
    def size(self) -> tuple[float, float]:
        """(length along the current, width across it), mm."""
        d = np.asarray(self.direction)
        xy = np.asarray(self.polygon.exterior.coords)[:-1]
        along, across = xy @ d, xy @ np.array([-d[1], d[0]])
        return float(np.ptp(along)), float(np.ptp(across))


class Layout:
    """Metal polygons by layer, junctions and a substrate outline. Lengths in mm."""

    def __init__(self, layers: dict[str, Polygon | MultiPolygon | list[Polygon]],
                 junctions: list[Junction] | None = None, outline: Bounds | None = None,
                 margin: float = 0.5, simplify: float = 1e-5):
        """
        `outline` defaults to the bounding box of everything grown by `margin`.
        `simplify` drops vertices closer than this to the line through their neighbours: sources
        that fillet with many short segments would otherwise seed thousands of tiny elements.
        """
        self.layers: dict[str, MultiPolygon] = {}
        for name, shape in layers.items():
            merged = unary_union(shape if isinstance(shape, (list, tuple)) else [shape])
            if simplify:
                merged = merged.simplify(simplify)
            polygons = [p for p in getattr(merged, "geoms", [merged])
                        if isinstance(p, Polygon) and not p.is_empty]
            if polygons:
                self.layers[name] = MultiPolygon(polygons)
        if not self.layers:
            raise ValueError("Layout: no metal polygons.")

        metal = self.metal
        self.junctions: list[Junction] = []
        for junction in junctions or []:
            clean = junction.polygon.difference(metal)      # metal wins where the two overlap
            if clean.is_empty or not isinstance(clean, Polygon):
                raise ValueError(f"junction {junction.name!r} is covered or split by the metal: "
                                 "it has to bridge a gap.")
            self.junctions.append(Junction(junction.name, clean, junction.direction))
            if not _bridges(clean, junction.direction, metal):
                warnings.warn(f"junction {junction.name!r} does not touch metal at both ends "
                              f"along {junction.direction}: it will not connect anything.")
        names = [j.name for j in self.junctions]
        if len(set(names)) != len(names):
            raise ValueError(f"junction names must be unique, got {names}")

        if outline is None:
            x0, y0, x1, y1 = unary_union([metal, *(j.polygon for j in self.junctions)]).bounds
            outline = (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        self.outline: Bounds = tuple(float(v) for v in outline)
        if not box(*self.outline).buffer(1e-9).contains(metal):
            raise ValueError(f"the metal (bounds {metal.bounds}) does not fit in the outline "
                             f"{self.outline}.")

    # ---------- content ----------
    @property
    def metal(self):
        """All the metal, every layer merged."""
        return unary_union(list(self.layers.values()))

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.outline
        return (x0 + x1) / 2, (y0 + y1) / 2

    @property
    def min_feature(self) -> float:
        """The smallest junction side, mm: the scale the mesh has to resolve."""
        sides = [s for j in self.junctions for s in j.size]
        return min(sides) if sides else float("inf")

    def __repr__(self):
        layers = ", ".join(f"{n}: {len(m.geoms)} polygon(s)" for n, m in self.layers.items())
        x0, y0, x1, y1 = self.outline
        return (f"Layout({layers}; junctions: {[j.name for j in self.junctions]}; "
                f"outline {x1 - x0:g} x {y1 - y0:g} mm)")

    def plot(self, ax=None):
        """Draw the layout: the quickest way to see that an import read what you meant."""
        import matplotlib.pyplot as plt
        from shapely.plotting import patch_from_polygon

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(*box(*self.outline).exterior.xy, color="0.5", lw=1)
        for i, (name, shape) in enumerate(self.layers.items()):
            for k, polygon in enumerate(shape.geoms):
                path = patch_from_polygon(
                    polygon, facecolor=f"C{i}", edgecolor="k", lw=0.5, alpha=0.7,
                    label=name if k == 0 else None)
                ax.add_patch(path)
        for junction in self.junctions:
            ax.fill(*junction.polygon.exterior.xy, color="crimson")
            c = junction.polygon.centroid
            ax.annotate(junction.name, (c.x, c.y), color="crimson", fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.legend(loc="upper right", fontsize=8)
        return ax

    # ---------- sources ----------
    @classmethod
    def from_metal(cls, design, components: list[str] | None = None, chip: str = "main",
                   ground_plane: bool = False, outline: Bounds | None = None,
                   fillet_resolution: int = 16, **kwargs) -> Layout:
        """Read a Quantum Metal / Qiskit Metal design.

        Every component writes shapely shapes into `design.qgeometry`: polygons, paths (a centre
        line and a width) and junctions (a line and a width). This turns them into polygons:

        * `components`: keep only these (default: all of them, on `chip`).
        * `ground_plane=False` (a 3D transmon: pads on a bare substrate): the metal is the
          shapes themselves, and the `subtract` shapes -- pockets, CPW gaps -- are ignored.
          `ground_plane=True` (a planar chip): the whole outline is metal, minus the `subtract`
          shapes, plus the rest.
        * `outline` defaults to the chip size declared in the design.

        One layer of the layout per Metal layer number, named `"metal"` when there is only one
        and `"layer<N>"` otherwise. Junctions are named `"<component>_<name>"`.
        """
        scale = _UNITS_MM[design.get_units()]
        wanted = None
        if components is not None:
            missing = [c for c in components if c not in design.components]
            if missing:
                raise KeyError(f"components not in the design: {missing}")
            wanted = {design.components[c].id for c in components}
        names = {comp.id: name for name, comp in design.components.items()}

        def rows(table: str):
            df = design.qgeometry.tables[table]
            df = df[df["chip"] == chip]
            return df if wanted is None else df[df["component"].isin(wanted)]

        def to_mm(shape):
            return shapely.affinity.scale(shape, xfact=scale, yfact=scale, origin=(0, 0))

        positive: dict[int, list] = {}
        negative: list = []
        for _, row in rows("poly").iterrows():
            (negative if row["subtract"] else positive.setdefault(int(row["layer"]), [])
             ).append(to_mm(row["geometry"]))
        for _, row in rows("path").iterrows():
            line = row["geometry"]
            if row["fillet"] and not np.isnan(row["fillet"]):
                line = _fillet(line, float(row["fillet"]), fillet_resolution)
            shape = to_mm(line.buffer(row["width"] / 2, cap_style="flat", join_style="mitre"))
            (negative if row["subtract"] else positive.setdefault(int(row["layer"]), [])).append(shape)

        junctions = []
        for _, row in rows("junction").iterrows():
            line = to_mm(row["geometry"])
            p0, p1 = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
            direction = (p1 - p0) / np.linalg.norm(p1 - p0)
            junctions.append(Junction(
                name=f"{names[row['component']]}_{row['name']}",
                polygon=line.buffer(row["width"] * scale / 2, cap_style="flat"),
                direction=(float(direction[0]), float(direction[1]))))

        if outline is None:
            size = design.get_chip_size(chip)
            cx, cy, sx, sy = (design.parse_value(size[k]) * scale
                              for k in ("center_x", "center_y", "size_x", "size_y"))
            outline = (cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2)

        if ground_plane:
            ground = box(*outline).difference(unary_union(negative))
            first = min(positive) if positive else 1
            positive.setdefault(first, []).append(ground)
        if not positive:
            raise ValueError("no metal found: check `components`, `chip`, or set ground_plane=True.")

        single = len(positive) == 1
        layers = {("metal" if single else f"layer{n}"): shapes for n, shapes in sorted(positive.items())}
        return cls(layers, junctions, outline, **kwargs)

    @classmethod
    def from_gds(cls, path: str | Path, layers: dict[str, int | tuple[int, int]],
                 junctions: int | tuple[int, int] | None = None, cell: str | None = None,
                 outline: Bounds | int | tuple[int, int] | None = None,
                 invert: tuple[str, ...] = (), **kwargs) -> Layout:
        """Read a GDS file.

        A GDS has layer *numbers* and nothing else, so you say what each one is:

        * `layers`: layout name -> GDS layer, as a number or `(layer, datatype)`,
          e.g. `{"metal": 1}` or `{"metal": (1, 0), "bandage": (3, 0)}`.
        * `junctions`: the GDS layer whose rectangles are junctions. Each becomes `"jj1"`,
          `"jj2"`, ... in file order; the current direction is the one along which the rectangle
          touches metal at both ends.
        * `cell`: which cell to read (default: the top-level one). References are flattened.
        * `outline`: bounds in mm, or a GDS layer that draws the chip edge. Default: the bounding
          box of the metal plus `margin`.
        * `invert`: layout layers drawn as the *gaps* (a negative mask): their metal becomes
          the outline minus what is drawn.

        Coordinates are converted to mm with the unit stored in the file.
        """
        import gdstk

        library = gdstk.read_gds(str(path))
        scale = library.unit / 1e-3
        if cell is None:
            tops = library.top_level()
            if len(tops) != 1:
                raise ValueError(f"{path}: {len(tops)} top-level cells "
                                 f"({[c.name for c in tops]}); say which with cell=...")
            source = tops[0]
        else:
            found = [c for c in library.cells if c.name == cell]
            if not found:
                raise KeyError(f"{path}: no cell {cell!r}. Cells: {[c.name for c in library.cells]}")
            source = found[0]

        by_layer: dict[tuple[int, int], list[Polygon]] = {}
        for polygon in source.get_polygons():
            shape = Polygon(np.asarray(polygon.points) * scale).buffer(0)
            by_layer.setdefault((polygon.layer, polygon.datatype), []).append(shape)

        def read(spec, what):
            key = (spec, 0) if isinstance(spec, int) else tuple(spec)
            if key not in by_layer:
                raise KeyError(f"{path}: no polygons on GDS layer {key} ({what}). "
                               f"Layers in the file: {sorted(by_layer)}")
            return by_layer[key]

        metal = {name: unary_union(read(spec, name)) for name, spec in layers.items()}
        if outline is not None and not _is_bounds(outline):
            outline = unary_union(read(outline, "outline")).bounds
        if invert:
            if outline is None:
                raise ValueError("invert needs an outline: the metal is the outline minus the drawing.")
            for name in invert:
                metal[name] = box(*outline).difference(metal[name])

        found = []
        if junctions is not None:
            everything = unary_union(list(metal.values()))
            for i, rect in enumerate(read(junctions, "junctions"), start=1):
                rect = rect.difference(everything)
                direction = next((d for d in ((1.0, 0.0), (0.0, 1.0))
                                  if _bridges(rect, d, everything)), None)
                if direction is None:
                    raise ValueError(f"{path}: junction {i} at {rect.centroid.coords[0]} does not "
                                     "bridge two pieces of metal along x or y.")
                found.append(Junction(f"jj{i}", rect, direction))
        return cls(metal, found, outline, **kwargs)


def _fillet(line: LineString, radius: float, resolution: int = 16) -> LineString:
    """Round the corners of a centre line with arcs of `radius`.

    A corner whose two segments are too short to hold the arc is left sharp, as Metal does: each
    segment has to fit the tangent length of the corners at both of its ends.
    """
    xy = np.asarray(line.coords, dtype=float)[:, :2]
    if len(xy) < 3:
        return line
    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    tangent = np.zeros(len(xy))                 # arc tangent length at every vertex
    for i in range(1, len(xy) - 1):
        a, b = xy[i - 1] - xy[i], xy[i + 1] - xy[i]
        cos = np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)
        angle = np.arccos(cos)                  # between the two segments: pi is a straight line
        if 1e-9 < angle < np.pi - 1e-9:
            tangent[i] = radius / np.tan(angle / 2)
    out = [xy[0]]
    for i in range(1, len(xy) - 1):
        t = tangent[i]
        fits = t > 0 and t + tangent[i - 1] <= lengths[i - 1] + 1e-12 and t + tangent[i + 1] <= lengths[i] + 1e-12
        if not fits:
            out.append(xy[i])
            continue
        a = (xy[i - 1] - xy[i]) / lengths[i - 1]
        b = (xy[i + 1] - xy[i]) / lengths[i]
        bisector = (a + b) / np.linalg.norm(a + b)
        centre = xy[i] + bisector * np.hypot(radius, t)
        start, end = xy[i] + a * t - centre, xy[i] + b * t - centre
        sweep = np.arctan2(start[0] * end[1] - start[1] * end[0], start @ end)
        for step in np.linspace(0.0, sweep, resolution + 1):
            c, s_ = np.cos(step), np.sin(step)
            out.append(centre + np.array([c * start[0] - s_ * start[1], s_ * start[0] + c * start[1]]))
    out.append(xy[-1])
    return LineString(out)


def _is_bounds(value) -> bool:
    return isinstance(value, (tuple, list)) and len(value) == 4


def _bridges(rect: Polygon, direction: tuple[float, float], metal, reach: float = 1e-6) -> bool:
    """True if `rect` touches metal just beyond both of its ends along `direction`."""
    d = np.asarray(direction, dtype=float)
    xy = np.asarray(rect.exterior.coords)[:-1]
    along = xy @ d
    centre = np.asarray(rect.centroid.coords[0])
    mid = centre @ d
    ends = []
    for extreme in (along.min(), along.max()):
        point = centre + d * (extreme - mid)
        ends.append(metal.distance(shapely.Point(point)) <= reach)
    return all(ends)
