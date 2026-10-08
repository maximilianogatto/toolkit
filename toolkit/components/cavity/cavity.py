# abstract class to make cavities
import json

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import gmsh
from pathlib import Path
from toolkit.materials.materials import Dielectric, Conductor, Material, get_material
from toolkit.components.ports import Port

@dataclass
class CavityOptions(ABC):
    """Base class for cavity options."""
    lc : float = 1.5         # max mesh size, mm
    algorithm : int = 6       # gmsh mesh algorithm
    mesh_size_from_curvature : int = 12  # about 12 elements around a full circle
    material : Material | None = None  # optional material for the cavity, e.g. copper
    mesh_order : int = 2  # order of the mesh elements


class Cavity(ABC):
    def __init__(self, options: CavityOptions, ports: list[Port] = None, cavity_wall_material: Material = get_material("copper", 'room'), volume_material: Material = get_material("vacuum", 'room')):
        self.options = options
        self.ports = ports if ports is not None else []
        self.cavity_wall_material = cavity_wall_material
        self.volume_material = volume_material

    @abstractmethod
    def _add_body(self, occ) -> list[tuple[int, int]]:
        """Add the cavity body to the gmsh model."""
        pass

    def _build_geometry(self, occ, verbose=False) -> tuple[list[int], list[list[int]]]:
        """Build the cavity geometry in gmsh."""
        body = self._add_body(occ)
        if verbose:
            print(f"Added cavity body: {body}")
        if not self.ports:                   # no ports: nothing to cut or fragment
            occ.synchronize()
            return [v for _, v in body], []
        # add the ports
        insulators = []
        pins = []
        for port in self.ports:
            pin, coax = port._add_port(occ)
            ins, _ = occ.cut([coax], [pin], removeTool=False) #tube between pin and coax is the insulator
            ins, _ = occ.cut(ins, body, removeTool=False) #cut the insulator with the cavity body
            insulators += ins
            if verbose:
                print(f"Added insulator: {ins}")
            pins.append(pin)

        if verbose:
            print(f"Added pins: {pins}")
        body, _ = occ.cut(body, pins)
        if len(body) != 1:
            raise ValueError(f"Expected one volume after adding body and ports, got {len(body)}. A coax port may be outside the cavity or the cavity may be too small.")
        # check that the number of insulators and pins matches the number of ports
        if len(insulators) != len(self.ports) or len(pins) != len(self.ports):
            raise ValueError(f"Mismatch in number of ports and insulators/pins: {len(self.ports)} ports, {len(insulators)} insulators, {len(pins)} pins.")

        # fragment the cavity body with the insulators to create separate volumes
        out, out_map = occ.fragment(body, insulators)
        occ.synchronize()

        if verbose:
            print(f"Fragmented cavity body with insulators: {out}")

        # create volumens for the vacuum and insulators
        vac_vols = [v for _, v in out_map[0]]   # the cavity volume (vacuum)
        ins_vols = [[v for _, v in out_map[i+1]] for i in range(len(self.ports))] # the insulator volumes

        if verbose:
            print(f"Created vacuum volumes: {vac_vols}, insulator volumes: {ins_vols}")
        return vac_vols, ins_vols

    # ---------- 3. physical groups ----------
    def _add_groups(self, cavity, ins_per_port, verbose=False):
        add = gmsh.model.addPhysicalGroup
        groups = {"cavity": add(3, cavity, name="cavity")}
        cavity_faces = self._faces_of(cavity)
        used, interfaces = set(), set()

        for i, (port, insulator) in enumerate(zip(self.ports, ins_per_port), start=1):
            faces = port.classify_faces(self._faces_of(insulator), cavity_faces)
            if verbose:
                print(f"port{i} faces: {faces}")
            groups[f"port{i}_insulator"] = add(3, insulator, name=f"port{i}_insulator")
            groups[f"port{i}"] = add(2, list(faces["ring"]), name=f"port{i}")
            groups[f"port{i}_pin"] = add(2, list(faces["pin"]), name=f"port{i}_pin")
            groups[f"port{i}_shield"] = add(2, list(faces["shield"]), name=f"port{i}_shield")
            used |= faces["ring"] | faces["pin"] | faces["shield"] | faces["interface"]
            interfaces |= faces["interface"]

        walls = cavity_faces - used
        if verbose:
            print(f"cavity walls: {sorted(walls)}, interfaces: {sorted(interfaces)}")
        groups["cavity_wall"] = add(2, list(walls), name="cavity_wall")
        return groups, interfaces

    def _check_faces_to_groups(self, interfaces, verbose=False) -> bool:
        """True if every face is in a physical group, except the insulator/cavity interfaces."""
        all_faces = {s for _, s in gmsh.model.getEntities(2)}
        grouped = set()
        for dim, tag in gmsh.model.getPhysicalGroups(2):
            grouped |= set(gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
        missing = all_faces - grouped - interfaces
        if verbose:
            print(f"all faces: {len(all_faces)}, grouped: {len(grouped)}, "
                f"interfaces: {len(interfaces)}, missing: {sorted(missing)}")
        return not missing

    # ---------- main ----------
    def _prepare(self, name, verbose: bool = False) -> dict[str, int]:
        """Build the geometry and tag every volume and face. Call inside a gmsh session."""
        gmsh.model.add(name)
        cavity, ins_per_port = self._build_geometry(gmsh.model.occ)
        groups, interfaces = self._add_groups(cavity, ins_per_port, verbose=verbose)
        if not self._check_faces_to_groups(interfaces, verbose=verbose):
            raise ValueError("Some faces are not assigned to any physical group. Check the geometry and port placement.")
        return groups

    def build_mesh(self, name, output: Path, gui: bool = False, verbose: bool = False) -> dict[str, int]:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            groups = self._prepare(name, verbose=verbose)

            gmsh.option.setNumber("Mesh.MeshSizeMax", self.options.lc)
            gmsh.option.setNumber("Mesh.Algorithm", self.options.algorithm)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", self.options.mesh_size_from_curvature)
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(self.options.mesh_order)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.Binary", 1)

            output.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output))
            if gui:
                gmsh.fltk.run()
            return groups
        finally:
            gmsh.finalize()

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
                     split: str | None = None, verbose: bool = False) -> None:
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

    @staticmethod
    def _faces_of(volumes: list[int]) -> set[int]:
        "All boundaries faces of some volume"
        dimtags = [(3, v) for v in volumes]
        return {s for _, s in gmsh.model.getBoundary(dimtags, oriented=False, combined=False)}
