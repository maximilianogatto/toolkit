"""CAD output of a `Cavity`: the vacuum as STEP with a face-tag sidecar, and the metal part.

Split from `cavity.py`. `ExportMixin` relies on what `Cavity` provides (`ports`, `_prepare`,
`_build_geometry`) and changes nothing in the design.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import json
from pathlib import Path

import gmsh
import numpy as np


class ExportMixin:
    def export_step(self, name, output: Path, tags: Path | None = None,
                    verbose: bool = False) -> dict[str, int]:
        """Write the geometry as STEP, plus a sidecar that says where each group is.

        STEP carries geometry only, so the physical groups are lost on import. The sidecar keeps,
        for every group, a point **on** each entity and the outward normal there, which is what a
        CAD-based solver needs to select the same faces again (COMSOL: a Ball selection at each
        point). Centres of mass are deliberately not used: for a cylindrical wall or an annular
        port ring the centre of mass does not lie on the surface.
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            groups = self._prepare(name, verbose=verbose)
            output.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output))                     # .step / .stp / .brep: format from the suffix
            if tags is not None:
                tags.parent.mkdir(parents=True, exist_ok=True)
                tags.write_text(json.dumps(self._group_tags(groups), indent=2), encoding="utf-8")
            if verbose:
                print(f"wrote {output}" + (f" and {tags}" if tags else ""))
            return groups
        finally:
            gmsh.finalize()

    def export_solid(self, name, output: Path, wall: float = 5.0, shape: str = "box",
                     split: str | None = None, verbose: bool = True) -> None:
        """Write the **metal part** as CAD: the block the cavity is machined out of.

        `export_step` writes the vacuum, which is the shape the field lives in. This writes its
        complement: a block `wall` mm thick around the cavity, with the cavity and a clearance
        hole per port taken out of it. That is the solid to open in Inventor or SolidWorks, and
        the one the workshop needs. The format follows the suffix: .step/.stp (use this one),
        .iges/.igs, or .brep.

        `split="x"` or `"y"` also cuts the part in two along a plane containing the cavity axis.
        For TM010 the wall currents run along z on the side wall and radially on the lids, so they
        never cross such a seam -- a lid-height split would cut straight across them and cost Q.
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.add(name)
            occ = gmsh.model.occ
            vacuum, insulators = self._build_geometry(occ, verbose=verbose)
            negative = [(3, v) for v in vacuum] + [(3, v) for vols in insulators for v in vols]

            lo, hi = self._bounds(occ, [(3, v) for v in vacuum])      # the cavity only, not the stubs
            if shape == "box":
                block = occ.addBox(*(lo - wall), *(hi - lo + 2 * wall))
            elif shape == "cylinder":
                centre = (lo + hi) / 2
                radius = max(hi[0] - centre[0], hi[1] - centre[1]) + wall
                block = occ.addCylinder(centre[0], centre[1], lo[2] - wall,
                                        0, 0, hi[2] - lo[2] + 2 * wall, radius)
            else:
                raise ValueError(f"shape must be 'box' or 'cylinder', got {shape!r}")

            holes = []
            for port in self.ports:            # from outside the block, through the wall
                reach = wall + 1.0
                depth = max(port.length, getattr(port, "depth", 0.0))   # also clears the pin tip,
                start = np.asarray(port.pos, dtype=float) - port.normal * reach   # which belongs to
                holes.append((3, occ.addCylinder(*start, *(port.normal * (reach + depth)),
                                                 port.bore_radius)))   # the connector, not the part

            part, _ = occ.cut([(3, block)], negative + holes)
            if split is not None:
                part = self._split(occ, part, split)
            occ.synchronize()

            output.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output))
            if verbose:
                volume = sum(occ.getMass(3, v) for _, v in part)
                print(f"wrote {output}: {len(part)} solid(s), {volume:.1f} mm^3 of metal")
        finally:
            gmsh.finalize()

    @staticmethod
    def _bounds(occ, dimtags) -> tuple[np.ndarray, np.ndarray]:
        """Corners of the box containing several entities."""
        boxes = np.array([occ.getBoundingBox(dim, tag) for dim, tag in dimtags])
        return boxes[:, :3].min(axis=0), boxes[:, 3:].max(axis=0)

    @classmethod
    def _split(cls, occ, part, axis: str) -> list[tuple[int, int]]:
        """Cut a solid in two with a plane through its centre, normal to `axis`."""
        try:
            i = {"x": 0, "y": 1, "z": 2}[axis]
        except KeyError:
            raise ValueError(f"split must be 'x', 'y' or 'z', got {axis!r}") from None
        lo, hi = cls._bounds(occ, part)
        margin = (hi - lo).max()
        lo, hi = lo - margin, hi + margin
        lo[i] = (lo[i] + hi[i]) / 2                       # half-space on the positive side
        knife = (3, occ.addBox(*lo, *(hi - lo)))
        upper, _ = occ.intersect(part, [knife], removeObject=False, removeTool=False)
        lower, _ = occ.cut(part, [knife])
        return upper + lower

    def _group_tags(self, groups: dict[str, int]) -> dict:
        """Describe every physical group geometrically, for a solver that only reads the STEP."""
        volumes = {"cavity", *(f"port{i}_insulator" for i in range(1, len(self.ports) + 1))}
        out = {"length_unit": "mm", "groups": {}}
        for group_name, tag in groups.items():
            dim = 3 if group_name in volumes else 2
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            out["groups"][group_name] = {
                "dim": dim,
                "tag": int(tag),
                "entities": [self._probe(dim, int(e)) for e in entities],
            }
        return out

    @classmethod
    def _probe(cls, dim: int, tag: int) -> dict:
        """A point well inside an entity, its clearance to the boundary, and the normal there.

        Mid-parameter points are not enough: on the annular port ring, or on the tube of a port
        insulator, the middle of the parameter range (like the centre of mass) falls in the hole.
        So candidates are filtered with `gmsh.model.isInside` and the one farthest from the
        boundary wins -- that clearance is also the largest radius a selection sphere may use.
        """
        size = gmsh.model.occ.getMass(dim, tag)
        points, normals = cls._candidates(dim, tag)
        if len(points) == 0:
            raise ValueError(f"no interior point found on entity ({dim}, {tag})")
        cloud = cls._boundary_cloud(dim, tag)
        if len(cloud):
            clearance = np.linalg.norm(points[:, None, :] - cloud[None, :, :], axis=2).min(axis=1)
        else:
            clearance = np.full(len(points), abs(size) ** (1.0 / dim))
        k = int(np.argmax(clearance))
        probe = {"entity": tag, "size": size, "point": [float(x) for x in points[k]],
                 "clearance": float(clearance[k])}
        if normals is not None:
            probe["normal"] = [float(x) for x in normals[k]]
        return probe

    @classmethod
    def _candidates(cls, dim: int, tag: int, n: int = 7):
        """Interior points of an entity: a parametric grid for a face, a box grid for a volume."""
        if dim == 2:
            uv = cls._face_grid(tag, n)
            if not uv:
                return np.empty((0, 3)), np.empty((0, 3))
            points = np.array([gmsh.model.getValue(2, tag, p) for p in uv]).reshape(-1, 3)
            normals = np.array([gmsh.model.getNormal(tag, p) for p in uv]).reshape(-1, 3)
            return points, normals
        box = gmsh.model.getBoundingBox(3, tag)
        axes = [np.linspace(box[i], box[i + 3], n + 2)[1:-1] for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = np.array([p for p in grid if gmsh.model.isInside(3, tag, list(p))])
        return inside, None

    @staticmethod
    def _face_grid(face: int, n: int = 7) -> list[list[float]]:
        """Parametric points that are really on a (possibly trimmed) face."""
        pmin, pmax = gmsh.model.getParametrizationBounds(2, face)
        us = np.linspace(pmin[0], pmax[0], n + 2)[1:-1]
        vs = np.linspace(pmin[1], pmax[1], n + 2)[1:-1]
        return [[float(u), float(v)] for u in us for v in vs
                if gmsh.model.isInside(2, face, [u, v], parametric=True)]

    @classmethod
    def _boundary_cloud(cls, dim: int, tag: int) -> np.ndarray:
        """Points sampled on the boundary of an entity, to measure how far inside a candidate is."""
        cloud = []
        for _, child in gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=False):
            child = abs(child)
            if dim == 2:
                pmin, pmax = gmsh.model.getParametrizationBounds(1, child)
                ts = np.linspace(pmin[0], pmax[0], 24)
                cloud.append(np.array(gmsh.model.getValue(1, child, list(ts))).reshape(-1, 3))
            else:
                uv = cls._face_grid(child, 5)
                if uv:
                    cloud.append(np.array([gmsh.model.getValue(2, child, p) for p in uv]).reshape(-1, 3))
        return np.vstack(cloud) if cloud else np.empty((0, 3))
