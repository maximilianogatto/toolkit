"""A chip: a planar layout on a substrate, placed inside a cavity.

The layout (`toolkit.components.layout.Layout`) is flat polygons; this gives them the third
dimension and their roles:

    substrate    a box `thickness` thick under the layout, a `Dielectric`
    metal        one zero-thickness sheet per layout layer, a `Conductor`
    junctions    one rectangle each, a `LumpedElement`

In the chip's own frame the metal lies in the plane z = 0 and the substrate fills
-thickness <= z <= 0, with x and y those of the layout. By default the centre of the outline goes
to `position`; the metal then faces +z of the chip, turned by `rotation` like any `Insert`.

Groups created, for a chip named `"chip"`: `chip_substrate`, `chip_<layer>` for every layer and
`chip_<junction>` for every junction.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from toolkit.components.attributes import LumpedElement
from toolkit.components.layout import Layout
from toolkit.materials import Conductor, Dielectric, get_material
from .insert import Insert, Piece


class Chip(Insert):
    def __init__(self, layout: Layout, name: str = "chip",
                 substrate: Dielectric | None = None, thickness: float = 0.43,
                 metal: Conductor | dict[str, Conductor] | None = None,
                 junctions: LumpedElement | dict[str, LumpedElement | None] | None = None,
                 position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), anchor=None,
                 mesh_size: float | None = None, mesh_grow: float = 1.0):
        """
        `substrate`: default sapphire (HEMEX) at mK, c-axis along the chip's z. `thickness=0`
            leaves the sheets floating in the cavity with no substrate at all.
        `metal`: one `Conductor` for every layer, or one per layer name. Default: aluminium at mK.
        `junctions`: one `LumpedElement` for every junction, or one per junction name; `None` for
            a name leaves that junction out (an open gap). Every junction of the layout must be
            accounted for.
        `anchor`: the layout point (x, y) that lands on `position`. Default: the outline centre.
        `mesh_size`: element size on the metal edges, mm. Junctions are always meshed at half
            their smallest side or finer. `mesh_grow`: distance over which that relaxes to the
            cavity's own `lc`.
        """
        anchor = layout.centre if anchor is None else anchor
        super().__init__(name, position, rotation, (anchor[0], anchor[1], 0.0), mesh_grow)
        self.layout = layout
        self.thickness = float(thickness)
        self.substrate = substrate if substrate is not None else get_material("sapphire_hemex", "mK")
        if self.thickness and not isinstance(self.substrate, Dielectric):
            raise TypeError(f"substrate: expected a Dielectric, got {self.substrate!r}")
        self.mesh_size = mesh_size

        metal = metal if metal is not None else get_material("aluminium", "mK")
        self.metal = metal if isinstance(metal, dict) else {n: metal for n in layout.layers}
        if set(self.metal) != set(layout.layers):
            raise ValueError(f"metal: give a Conductor for each layer {sorted(layout.layers)}, "
                             f"got {sorted(self.metal)}")
        for layer, material in self.metal.items():
            if not isinstance(material, Conductor):
                raise TypeError(f"metal[{layer!r}]: expected a Conductor, got {material!r}")

        names = [j.name for j in layout.junctions]
        if isinstance(junctions, dict):
            self.junctions = dict(junctions)
        else:
            self.junctions = {n: junctions for n in names} if junctions is not None else {}
        if set(self.junctions) != set(names):
            raise ValueError(f"junctions: the layout has {names}; give a LumpedElement (or None, "
                             f"to leave it open) for each. Got {sorted(self.junctions)}")

    # ---------- geometry ----------
    def _add(self, occ) -> list[Piece]:
        pieces = []
        if self.thickness:
            x0, y0, x1, y1 = self.layout.outline
            substrate = [(3, occ.addBox(x0, y0, -self.thickness, x1 - x0, y1 - y0, self.thickness))]
            # an anisotropic substrate turns with the chip: its axes are the chip's own
            turned = self.is_rotated and self.substrate.permittivity.is_anisotropic
            pieces.append(Piece("substrate", substrate, self.substrate,
                                axes=self.matrix.T.tolist() if turned else None))

        for layer, shape in self.layout.layers.items():
            sheets = [(2, self._surface(occ, polygon)) for polygon in shape.geoms]
            pieces.append(Piece(layer, sheets, self.metal[layer], mesh_size=self.mesh_size))

        for junction in self.layout.junctions:
            element = self.junctions[junction.name]
            if element is None:
                continue
            size = min(junction.size) / 2
            if self.mesh_size:
                size = min(size, self.mesh_size)
            direction = self.matrix @ np.array([*junction.direction, 0.0])
            pieces.append(Piece(junction.name, [(2, self._surface(occ, junction.polygon))], element,
                                direction=tuple(float(v) for v in direction), mesh_size=size))

        self._place(occ, [dimtag for piece in pieces for dimtag in piece.dimtags])
        return pieces

    @staticmethod
    def _surface(occ, polygon: Polygon) -> int:
        """A shapely polygon, holes included, as a plane surface at z = 0."""
        def loop(ring) -> int:
            points = [occ.addPoint(x, y, 0.0) for x, y in list(ring.coords)[:-1]]
            lines = [occ.addLine(a, b) for a, b in zip(points, points[1:] + points[:1])]
            return occ.addCurveLoop(lines)

        return occ.addPlaneSurface([loop(polygon.exterior), *(loop(r) for r in polygon.interiors)])
