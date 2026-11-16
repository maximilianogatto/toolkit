"""A cavity whose shape comes from a CAD file (STEP, IGES or BREP).

The file can describe either side of the wall, and `kind` says which:

    kind="vacuum"   the solid *is* the empty space the field lives in -- what `export_step`
                    writes, or the "fluid volume" of a CAD program.
    kind="metal"    the solid is the machined part -- what `export_solid` writes, and what a
                    workshop drawing is. The cavity is then whatever the part encloses: its
                    bounding box minus the metal, keeping the largest pocket. Holes through the
                    wall stay open up to the bounding box; those openings become part of
                    `cavity_wall` unless you `assign` them something else.

From there it is a `Cavity` like any other: coax ports, chips and parts go in the same way, and
every outer face is `cavity_wall`, made of `cavity_wall_material`, until `assign` says otherwise:

    cavity = StepCavity(StepCavityOptions(path="storage.step", lc=2.0))
    cavity.assign("lid", at((0, 0, 30)), get_material("aluminium", "mK"))

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from toolkit.components.inserts.step_part import import_solids
from toolkit.components.ports import Port
from .cavity import Cavity, CavityOptions

_SKIN = 1e-4   # mm: how far inside the part's bounding box the "metal" block is cut


@dataclass(kw_only=True)
class StepCavityOptions(CavityOptions):
    path: str | Path
    kind: str = "vacuum"       # "vacuum" or "metal": which side of the wall the solid is
    scale: float = 1.0         # multiplies the file's coordinates if they do not come out in mm


class StepCavity(Cavity):
    def __init__(self, options: StepCavityOptions, ports: list[Port] = None, **materials):
        # materials: cavity_wall_material, volume_material, inserts (see Cavity)
        if options.kind not in ("vacuum", "metal"):
            raise ValueError(f"kind must be 'vacuum' or 'metal', got {options.kind!r}")
        super().__init__(options, ports, **materials)

    def _add_body(self, occ) -> list[tuple[int, int]]:
        solids = import_solids(occ, self.options.path, self.options.scale)
        if self.options.kind == "vacuum":
            if len(solids) > 1:              # several bodies that make up one empty space
                solids, _ = occ.fuse(solids[:1], solids[1:])
            return solids

        # the metal part: the cavity is the hole in it
        boxes = np.array([occ.getBoundingBox(3, tag) for _, tag in solids])
        lo, hi = boxes[:, :3].min(axis=0) + _SKIN, boxes[:, 3:].max(axis=0) - _SKIN
        block = occ.addBox(*lo, *(hi - lo))
        pockets, _ = occ.cut([(3, block)], solids)
        pockets = [(d, t) for d, t in pockets if d == 3]
        if not pockets:
            raise ValueError(f"{self.options.path}: the part encloses no empty space. If the file "
                             "already is the vacuum, use kind='vacuum'.")
        largest = max(pockets, key=lambda dimtag: occ.getMass(*dimtag))
        occ.remove([p for p in pockets if p != largest], recursive=True)
        return [largest]
