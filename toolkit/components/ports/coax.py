"""Coaxial ports: a generic coax and an SMA connector.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gmsh
import numpy as np

from toolkit.materials import Conductor, Dielectric, get_material
from .port import Port


@dataclass
class CoaxPort(Port):
    """Coaxial feed: pin, insulator and shield.

    `pos` is the port ring (the outer end of the coax) and `normal` points into the cavity. Both
    lengths are measured from the ring: the insulator runs `length` (it must reach past the cavity
    wall, where it is trimmed) and the pin runs `depth`, so it enters the cavity by `depth` minus
    the ring-to-wall distance.
    """
    depth: float = 2.0      # pin length from the ring, mm
    length: float = 3.0     # insulator length from the ring, mm
    a: float = 0.5          # pin radius, mm
    b: float = 1.15         # insulator / shield radius, mm
    isolated_material: Dielectric = field(default_factory=lambda: get_material("vacuum", "any"))
    pin_material: Conductor = field(default_factory=lambda: get_material("copper", "room"))
    shield_material: Conductor = field(default_factory=lambda: get_material("copper", "room"))

    @property
    def z0(self) -> float:
        """Characteristic impedance of the coax, Ohm (isotropic insulator)."""
        eps_r = self.isolated_material.permittivity.value
        return 60 / np.sqrt(eps_r) * np.log(self.b / self.a)

    def _add_port(self, occ) -> tuple[tuple[int, int], tuple[int, int]]:
        pin = occ.addCylinder(*self.pos, *(self.normal * self.depth), self.a)
        outer = occ.addCylinder(*self.pos, *(self.normal * self.length), self.b)
        return (3, pin), (3, outer)

    def classify_faces(self, insulator_faces: set[int], cavity_faces: set[int]) -> dict[str, set[int]]:
        interface = insulator_faces & cavity_faces            # shared with the cavity: no boundary condition
        ring = {s for s in insulator_faces
                if np.allclose(gmsh.model.occ.getCenterOfMass(2, s), self.pos, atol=1e-6)}
        pin = {s for s in insulator_faces | cavity_faces if self._distance_to_axis(s) <= self.a + 1e-6}
        shield = insulator_faces - interface - ring - pin
        return {"ring": ring, "pin": pin, "shield": shield, "interface": interface}

    def _distance_to_axis(self, face: int) -> float:
        """Largest distance from the port axis to the corner points of a face."""
        corners = gmsh.model.getBoundary([(2, face)], oriented=False, recursive=True)
        d = np.array([gmsh.model.getValue(0, p, []) for _, p in corners]) - self.pos
        along = d @ self.normal
        return np.linalg.norm(d - np.outer(along, self.normal), axis=1).max()


@dataclass
class SMAPort(CoaxPort):
    """SMA connector: 1.27 mm pin, 4.10 mm PTFE, gold-plated pin and body (Z0 = 49.6 Ohm with PTFE at mK)."""
    a: float = 0.635
    b: float = 2.05
    isolated_material: Dielectric = field(default_factory=lambda: get_material("ptfe", "mK"))
    pin_material: Conductor = field(default_factory=lambda: get_material("gold", "room"))
    shield_material: Conductor = field(default_factory=lambda: get_material("gold", "room"))
