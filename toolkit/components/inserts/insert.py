"""Things placed inside a cavity: a chip, a CAD part.

An `Insert` adds its own solids and sheets to the cavity's gmsh model and says what each of them
is (`Piece.attribute`). `Cavity` does the rest the same way for every insert: it fragments
everything into one conformal model, makes a physical group per piece, named
`"<insert name>_<piece key>"`, and passes the roles on to the solver.

What a piece becomes follows from its dimension and its attribute:

    volume + Dielectric       a dielectric region inside the cavity
    volume + Conductor        solid metal: taken out of the cavity, its skin is a metal boundary
    face   + Conductor        a metal sheet (zero thickness)
    face   + LumpedElement    a lumped R, L, C

Placement is the same for all: the insert is built in its own frame, the point `anchor` of that
frame is moved to `position` in the cavity, after rotating by `rotation = (rx, ry, rz)` degrees
about the x, then y, then z axis.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from toolkit.components.attributes import Attribute


@dataclass
class Piece:
    """Some entities of an insert that share a role."""
    key: str                                   # group name is "<insert name>_<key>"
    dimtags: list[tuple[int, int]]             # all of the same dimension
    attribute: Attribute
    direction: tuple[float, float, float] | None = None   # LumpedElement, cavity frame
    axes: list[list[float]] | None = None                  # Dielectric, cavity frame
    mesh_size: float | None = None             # target element size at this piece, mm

    @property
    def dim(self) -> int:
        return self.dimtags[0][0]


class Insert(ABC):
    """Base of everything that goes inside a cavity."""

    def __init__(self, name: str, position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0),
                 anchor=(0.0, 0.0, 0.0), mesh_grow: float = 1.0):
        if not name or not name.replace("_", "").isalnum():
            raise ValueError(f"insert name must be letters, digits and '_', got {name!r}")
        self.name = name
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)
        self.anchor = np.asarray(anchor, dtype=float)
        self.mesh_grow = mesh_grow     # distance over which a fine mesh relaxes to the cavity's lc

    @abstractmethod
    def _add(self, occ) -> list[Piece]:
        """Add the insert to the OCC model, already placed, and describe its pieces."""

    # ---------- placement ----------
    @property
    def matrix(self) -> np.ndarray:
        """Rotation matrix: x first, then y, then z."""
        a, b, c = np.radians(self.rotation)
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    @property
    def is_rotated(self) -> bool:
        return not np.allclose(self.matrix, np.eye(3))

    def to_cavity(self, point) -> np.ndarray:
        """A point of the insert's own frame, in the cavity frame."""
        return self.matrix @ (np.asarray(point, dtype=float) - self.anchor) + self.position

    def _place(self, occ, dimtags: list[tuple[int, int]]) -> None:
        """Move entities built in the insert's frame to where they go in the cavity."""
        if np.any(self.anchor):
            occ.translate(dimtags, *(-self.anchor))
        for axis, angle in zip(np.eye(3), np.radians(self.rotation)):
            if angle:
                occ.rotate(dimtags, 0, 0, 0, *axis, angle)
        if np.any(self.position):
            occ.translate(dimtags, *self.position)
