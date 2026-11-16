"""A solid from a STEP file, placed inside a cavity.

    StepPart("holder.step", get_material("sapphire_hemex", "mK"))    a dielectric part
    StepPart("post.step", get_material("copper", "mK"))              a metal part

A `Dielectric` part stays in the model as a volume (group `<name>_volume`). A `Conductor` part is
solid metal: the field does not enter it, so it is taken out of the cavity and only its skin is
kept, as a metal boundary (group `<name>_surface`).

Faces of a dielectric part can be given roles of their own afterwards -- a metal film, a lumped
element -- with `Cavity.assign` and a selector.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from pathlib import Path

from toolkit.materials import Conductor, Dielectric
from .insert import Insert, Piece


def import_solids(occ, path: str | Path, scale: float = 1.0) -> list[tuple[int, int]]:
    """The solids of a STEP (or IGES, BREP) file, in mm."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no CAD file at {path}")
    solids = [(d, t) for d, t in occ.importShapes(str(path)) if d == 3]
    if not solids:
        raise ValueError(f"{path}: no solids. A file of loose surfaces cannot be meshed as a "
                         "volume; export it as a solid body.")
    if scale != 1.0:
        occ.dilate(solids, 0, 0, 0, scale, scale, scale)
    return solids


class StepPart(Insert):
    def __init__(self, path: str | Path, material: Dielectric | Conductor, name: str = "part",
                 position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0), anchor=(0.0, 0.0, 0.0),
                 scale: float = 1.0, mesh_size: float | None = None, mesh_grow: float = 1.0):
        """
        `scale`: multiplies the file's coordinates, for a file that is not in mm after import
            (OpenCASCADE already converts the unit declared in a STEP header to mm).
        `anchor`: the point of the file's own frame that lands on `position`.
        """
        super().__init__(name, position, rotation, anchor, mesh_grow)
        if not isinstance(material, (Dielectric, Conductor)):
            raise TypeError(f"material: expected a Dielectric or a Conductor, got {material!r}")
        self.path = Path(path)
        self.material = material
        self.scale = scale
        self.mesh_size = mesh_size

    def _add(self, occ) -> list[Piece]:
        solids = import_solids(occ, self.path, self.scale)
        self._place(occ, solids)
        key = "volume" if isinstance(self.material, Dielectric) else "surface"
        turned = (isinstance(self.material, Dielectric) and self.is_rotated
                  and self.material.permittivity.is_anisotropic)
        return [Piece(key, solids, self.material, mesh_size=self.mesh_size,
                      axes=self.matrix.T.tolist() if turned else None)]
