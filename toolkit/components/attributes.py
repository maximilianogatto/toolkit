"""What a piece of geometry *is* for the solver.

Imported geometry (a STEP file, a GDS, a Quantum Metal design) is only shape. Before it can be
simulated every volume and face needs a role, and there are three:

    Dielectric      a volume and its material                    (`toolkit.materials`)
    Conductor       a metal face: PEC or a finite conductivity   (`toolkit.materials`)
    LumpedElement   a face that carries an R, L, C: a Josephson junction as a linear inductor,
                    a resistor, a capacitor

`Region` is how a `Cavity` hands those roles to a backend: one per physical group, with the group's
dimension and, where it matters, the orientation that only the geometry knows (the direction of a
lumped element, the crystal axes of a rotated anisotropic substrate).

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from toolkit.materials import Conductor, Dielectric

Direction = str | tuple[float, float, float]


@dataclass(frozen=True)
class LumpedElement:
    """A lumped R, L, C on a rectangular face, in SI units (Ohm, H, F), in parallel.

    A Josephson junction in a linear simulation is `LumpedElement(L=Lj)`, with `C=Cj` if the
    junction capacitance is not already in the geometry. It is never excited: a port that launches
    a signal is a `Port` of the cavity.

    `direction` is the way the current flows across the face: `"+X"`, `"-Y"`, ... or a vector. A
    junction that comes from a layout already knows it, so leave it out there; give it when you
    assign the element to a face yourself.
    """
    R: float | None = None
    L: float | None = None
    C: float | None = None
    direction: Direction | None = None

    def __post_init__(self):
        if self.R is None and self.L is None and self.C is None:
            raise ValueError("LumpedElement: give at least one of R, L, C.")


Attribute = Dielectric | Conductor | LumpedElement


@dataclass(frozen=True)
class Region:
    """One physical group and its role."""
    dim: int                                   # 3: volume, 2: face
    attribute: Attribute
    direction: Direction | None = None         # LumpedElement: current direction, mesh frame
    axes: list[list[float]] | None = None      # Dielectric: material x, y, z axes, mesh frame


def palace_direction(direction: Direction) -> str | list[float]:
    """A direction the way Palace wants it: `"+X"` when axis-aligned, else a unit vector."""
    if isinstance(direction, str):
        return direction
    v = np.asarray(direction, dtype=float)
    norm = np.linalg.norm(v)
    if v.shape != (3,) or norm == 0:
        raise ValueError(f"direction must be a non-zero 3-vector, got {direction!r}")
    v = v / norm
    axis = int(np.argmax(np.abs(v)))
    if abs(abs(v[axis]) - 1.0) < 1e-9:
        return ("+" if v[axis] > 0 else "-") + "XYZ"[axis]
    return [float(x) for x in v]
