"""Base class for the ports of a cavity.

A port couples the cavity to the outside through a lumped port. Each port type builds its own
solids and knows which of the resulting faces are which (the lumped-port face, metals, and the
interface with the cavity), so `Cavity` can handle any port type the same way.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Port(ABC):
    """Common data of every port."""
    pos: tuple[float, float, float]      # reference point of the port (for a coax: the port ring), mm
    normal: tuple[float, float, float]   # direction pointing into the cavity (normalised on init)
    R: float = 50.0                      # port resistance, Ohm
    excite: bool = False                 # driven simulations: launch the signal from this port

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.normal = np.asarray(self.normal, dtype=float) / np.linalg.norm(self.normal)

    @abstractmethod
    def _add_port(self, occ) -> tuple[tuple[int, int], tuple[int, int]]:
        """Add the port's solids to the OCC model: (metal to remove from the cavity, outer volume)."""

    @property
    def bore_radius(self) -> float:
        """Radius of the clearance hole this port needs through the cavity wall, mm.

        Only used to cut the hole when the metal part is exported for CAD.
        """
        raise NotImplementedError(f"{type(self).__name__} does not define a bore radius")

    @abstractmethod
    def classify_faces(self, insulator_faces: set[int], cavity_faces: set[int]) -> dict[str, set[int]]:
        """Faces of this port by role. Must contain "ring" (the lumped port) and "interface"."""
