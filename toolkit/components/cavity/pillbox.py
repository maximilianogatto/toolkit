"""A milled pocket: a rectangular cavity with the corner radii a cutter leaves behind.

`RectangularCavity` is the ideal box, with sharp corners nothing can machine. This is the shape a
real workshop produces: a flat end mill of radius `corner_radius` rounds the four vertical edges
and leaves the floor and the ceiling sharp, which is how these pockets are normally cut.

`corner_radius` is either one number, meaning all four corners alike, or four numbers, one per
corner, clockwise seen from above starting at the corner on the origin:

    (0, 0) -> (0, height) -> (width, height) -> (width, 0)

Two radii on the same side must leave some flat between them: their sum has to stay below that
side's length. At the limit the two fillets meet exactly and OpenCASCADE refuses to compute them,
so the check raises first with a message that says which side is too small.

Axes follow `RectangularCavity`: `width` along x, `height` along y, `depth` along z -- the depth
of the cut, so the cutter travels along z. The pocket starts at the origin.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toolkit.components.ports import Port
from .cavity import Cavity, CavityOptions

Radii = float | tuple[float, float, float, float]

_EPS = 1e-3   # mm: the half-width of the boxes used to pick edges out of the model


@dataclass(kw_only=True)
class PillBoxCavityOptions(CavityOptions):
    """A pocket `width` x `height`, cut `depth` deep, with machining radii."""
    width: float                       # along x
    height: float                      # along y
    depth: float                       # along z: how deep the pocket is cut
    corner_radius: Radii = 0.0         # end-mill radius on the four vertical edges;
                                       # one value, or one per corner clockwise from (0, 0)


class PillBoxCavity(Cavity):
    def __init__(self, options: PillBoxCavityOptions, ports: list[Port] = None, **materials):
        # materials: cavity_wall_material, volume_material (see Cavity)
        super().__init__(options, ports, **materials)

    # ---------- dimensions ----------
    @property
    def width(self) -> float:
        return self.options.width

    @property
    def height(self) -> float:
        return self.options.height

    @property
    def depth(self) -> float:
        return self.options.depth

    @property
    def corner_radii(self) -> np.ndarray:
        """One radius per vertical edge, clockwise from the corner at the origin.

        Everything is checked here, before gmsh is touched: a radius that does not fit makes
        OpenCASCADE answer `Could not compute fillet`, which says nothing about which corner.
        """
        given = np.atleast_1d(np.asarray(self.options.corner_radius, dtype=float))
        if given.size not in (1, 4):
            raise ValueError(f"corner_radius takes one radius or four, got {given.size}: "
                             f"{given.tolist()}")
        if not np.all(np.isfinite(given)):
            raise ValueError(f"corner_radius must be finite, got {given.tolist()}")
        if np.any(given < 0):
            raise ValueError(f"corner_radius must not be negative, got {given.tolist()}")

        r = np.broadcast_to(given, (4,))
        for i, (side, length) in enumerate(self._sides()):
            flat = length - (r[i] + r[(i + 1) % 4])
            if flat <= 0:
                detail = ("they meet exactly, leaving no flat wall" if flat == 0
                          else f"they would overlap by {-flat:g}")
                raise ValueError(
                    f"corner radii {r[i]} and {r[(i + 1) % 4]} are too large for the {side}, "
                    f"which is {length} long: {detail}")
            if flat < 0.01 * length:
                raise ValueError(
                    f"corner radii {r[i]} and {r[(i + 1) % 4]} leave only {flat:g} of flat wall on "
                    f"the {side} ({length} long). Fillet it fully, or leave more than 1% flat")
        return r

    def _sides(self) -> list[tuple[str, float]]:
        """The four sides, clockwise, between consecutive corners: (name, length)."""
        return [("left side (x = 0)", self.height),
                (f"top side (y = {self.height})", self.width),
                (f"right side (x = {self.width})", self.height),
                ("bottom side (y = 0)", self.width)]

    # ---------- geometry ----------
    def _add_body(self, occ) -> list[tuple[int, int]]:
        """The pocket: a box with the four vertical edges rounded by the cutter."""
        radii = self.corner_radii
        body = occ.addBox(0, 0, 0, self.width, self.height, self.depth)
        if np.any(radii > 0):
            body = self._round_corners(occ, body, radii)
        return [(3, body)]

    def _round_corners(self, occ, body: int, radii: np.ndarray) -> int:
        """Fillet the four vertical edges, each with its own radius."""
        corners = [(0.0, 0.0), (0.0, self.height), (self.width, self.height), (self.width, 0.0)]
        edges, values = [], []
        for (x, y), radius in zip(corners, radii):
            if radius <= 0:
                continue
            found = self._curves_in(occ, (x - _EPS, y - _EPS, -_EPS), (x + _EPS, y + _EPS, self.depth + _EPS))
            if len(found) != 1:
                raise ValueError(f"expected one vertical edge at ({x}, {y}), found {len(found)}")
            edges += found
            values.append(radius)
        return occ.fillet([body], edges, values)[0][1] if edges else body

    # ---------- finding edges ----------
    @staticmethod
    def _curves_in(occ, lo, hi) -> list[int]:
        """Curves inside a box, the Python twin of the .geo `Curve In BoundingBox{...}`.

        Edges are picked by where they are, never by their tag: a fillet renumbers everything.
        """
        return [tag for _, tag in occ.getEntitiesInBoundingBox(*lo, *hi, 1)]
