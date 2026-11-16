"""Pick faces of a model by where they are.

Imported CAD has no names the solver can use, and entity tags change with every boolean
operation, so a face is always chosen geometrically, in the final model:

    at((x, y, z))               the face(s) this point lies on
    in_box(lo, hi)              every face entirely inside the box
    where(lambda f: ...)        any test on a face's centre, bounding box and area

All coordinates are in mm, in the cavity frame. Selectors are used by `Cavity.assign`.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import gmsh
import numpy as np


class Selector(ABC):
    """Chooses faces out of a set of candidates. Call inside a gmsh session."""

    @abstractmethod
    def pick(self, faces: set[int]) -> set[int]:
        """The candidates this selector keeps."""


@dataclass(frozen=True)
class FaceInfo:
    """What `where` gets to look at."""
    tag: int
    centre: np.ndarray      # centre of mass; for a curved or annular face it may lie off the face
    lo: np.ndarray          # bounding box, lower corner
    hi: np.ndarray          # bounding box, upper corner
    area: float

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo


class at(Selector):
    """The face(s) a point lies on, within `tol` mm. A point on an edge picks both neighbours."""

    def __init__(self, point, tol: float = 1e-3):
        self.point = np.asarray(point, dtype=float)
        self.tol = tol

    def pick(self, faces: set[int]) -> set[int]:
        found = set()
        for face in faces:
            lo_hi = np.array(gmsh.model.getBoundingBox(2, face))
            if np.any(self.point < lo_hi[:3] - self.tol) or np.any(self.point > lo_hi[3:] + self.tol):
                continue
            closest, param = gmsh.model.getClosestPoint(2, face, list(self.point))
            # the projection is onto the untrimmed surface: check it falls on the face itself
            if (np.linalg.norm(np.asarray(closest) - self.point) <= self.tol
                    and gmsh.model.isInside(2, face, list(param), parametric=True)):
                found.add(face)
        return found

    def __repr__(self):
        return f"at({self.point.tolist()})"


class in_box(Selector):
    """Every face whose bounding box fits inside the box `lo`..`hi` (grown by `tol` mm)."""

    def __init__(self, lo, hi, tol: float = 1e-3):
        self.lo = np.minimum(lo, hi).astype(float) - tol
        self.hi = np.maximum(lo, hi).astype(float) + tol

    def pick(self, faces: set[int]) -> set[int]:
        inside = {t for _, t in gmsh.model.getEntitiesInBoundingBox(*self.lo, *self.hi, 2)}
        return faces & inside

    def __repr__(self):
        return f"in_box({self.lo.tolist()}, {self.hi.tolist()})"


class where(Selector):
    """Every face for which `test(FaceInfo)` is true."""

    def __init__(self, test: Callable[[FaceInfo], bool]):
        self.test = test

    def pick(self, faces: set[int]) -> set[int]:
        found = set()
        for face in faces:
            box = np.array(gmsh.model.getBoundingBox(2, face))
            info = FaceInfo(tag=face, centre=np.array(gmsh.model.occ.getCenterOfMass(2, face)),
                            lo=box[:3], hi=box[3:], area=gmsh.model.occ.getMass(2, face))
            if self.test(info):
                found.add(face)
        return found

    def __repr__(self):
        return f"where({getattr(self.test, '__name__', 'test')})"
