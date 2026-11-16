# abstract class to make cavities
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import gmsh
from pathlib import Path
from toolkit.materials.materials import Dielectric, Conductor, Material, get_material
from toolkit.components.ports import Port
from toolkit.components.attributes import LumpedElement, Region
from toolkit.components.inserts import Insert, Piece
from toolkit.components.select import Selector
from .export import ExportMixin
from .render import RenderMixin

@dataclass
class CavityOptions(ABC):
    """Base class for cavity options."""
    lc : float = 1.5         # max mesh size, mm
    algorithm : int = 6       # gmsh mesh algorithm
    mesh_size_from_curvature : int = 12  # about 12 elements around a full circle
    material : Material | None = None  # optional material for the cavity, e.g. copper
    mesh_order : int = 2  # order of the mesh elements


class Cavity(RenderMixin, ExportMixin, ABC):
    def __init__(self, options: CavityOptions, ports: list[Port] = None, cavity_wall_material: Material = get_material("copper", 'room'), volume_material: Material = get_material("vacuum", 'room'), inserts: list[Insert] = None):
        self.options = options
        self.ports = ports if ports is not None else []
        self.cavity_wall_material = cavity_wall_material
        self.volume_material = volume_material
        self.inserts: list[Insert] = []
        self.assignments: list[tuple[str, Selector, Conductor | LumpedElement]] = []
        self.regions: dict[str, Region] = {}   # group name -> role, for everything that is not the cavity or a port; set by _prepare
        self._refine: list[tuple[list[int], float, float]] = []
        for insert in inserts or []:
            self.add(insert)

    @abstractmethod
    def _add_body(self, occ) -> list[tuple[int, int]]:
        """Add the cavity body to the gmsh model."""
        pass

    # ---------- what goes inside, and what the faces are ----------
    def add(self, insert: Insert) -> Cavity:
        """Put a chip or a CAD part inside the cavity. Returns the cavity, so calls can be chained."""
        if not isinstance(insert, Insert):
            raise TypeError(f"expected an Insert (Chip, StepPart, ...), got {insert!r}")
        if insert.name in {i.name for i in self.inserts}:
            raise ValueError(f"there is already an insert named {insert.name!r}: give this one another name.")
        self.inserts.append(insert)
        return self

    def assign(self, name: str, selector: Selector, attribute: Conductor | LumpedElement) -> Cavity:
        """Give some faces a role of their own, as the physical group `name`.

        Faces are picked among those nothing else has claimed -- the cavity wall, the skin of a
        dielectric part -- so a selector may be generous: it never steals a port or a chip's
        metal. Whatever is left of the outer boundary stays `cavity_wall`.

            cavity.assign("lid", at((0, 0, 30)), get_material("aluminium", "mK"))

        A `LumpedElement` assigned this way needs its `direction`.
        """
        if not isinstance(attribute, (Conductor, LumpedElement)):
            raise TypeError(f"{name}: a face takes a Conductor or a LumpedElement, got {attribute!r}")
        if isinstance(attribute, LumpedElement) and attribute.direction is None:
            raise ValueError(f"{name}: a LumpedElement assigned to a face needs a direction.")
        if name in {n for n, _, _ in self.assignments}:
            raise ValueError(f"{name!r} is already assigned.")
        self.assignments.append((name, selector, attribute))
        return self

    # ---------- geometry ----------
    def _build_geometry(self, occ, verbose=False) -> tuple[list[int], list[list[int]]]:
        """Build the cavity geometry in gmsh: (vacuum volumes, insulator volumes per port)."""
        vacuum, insulators, _ = self._build(occ, verbose=verbose)
        return vacuum, insulators

    def _build(self, occ, verbose=False) -> tuple[list[int], list[list[int]], list[tuple[Insert, Piece, list[int]]]]:
        """Build everything and make it conformal.

        Returns the vacuum volumes, the insulator volumes of each port and, for every piece of
        every insert, the entities it ended up as: volumes for a dielectric, faces otherwise.
        """
        body = self._add_body(occ)
        if verbose:
            print(f"Added cavity body: {body}")

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

        if self.ports:
            if verbose:
                print(f"Added pins: {pins}")
            body, _ = occ.cut(body, pins)
            if len(body) != 1:
                raise ValueError(f"Expected one volume after adding body and ports, got {len(body)}. A coax port may be outside the cavity or the cavity may be too small.")
            # check that the number of insulators and pins matches the number of ports
            if len(insulators) != len(self.ports) or len(pins) != len(self.ports):
                raise ValueError(f"Mismatch in number of ports and insulators/pins: {len(self.ports)} ports, {len(insulators)} insulators, {len(pins)} pins.")

        # add the inserts
        pieces = [(insert, piece) for insert in self.inserts for piece in insert._add(occ)]
        tools = insulators + [dimtag for _, piece in pieces for dimtag in piece.dimtags]
        if not tools:                        # a bare cavity: nothing to fragment
            occ.synchronize()
            return [v for _, v in body], [], []

        # fragment everything together: one conformal model, and out_map says what became what
        _, out_map = occ.fragment(body, tools)
        occ.synchronize()
        if verbose:
            print(f"Fragmented cavity body with {len(tools)} object(s)")

        inside_body = {t for children in out_map[:len(body)] for d, t in children if d == 3}
        ins_vols = [[v for _, v in out_map[len(body) + i]] for i in range(len(self.ports))]
        k = len(body) + len(insulators)
        placed, taken = [], set()
        for insert, piece in pieces:
            tags = set()
            for _ in piece.dimtags:
                tags |= {t for d, t in out_map[k] if d == piece.dim}
                k += 1
            if not tags:
                raise ValueError(f"{insert.name}_{piece.key}: nothing left of it in the model. Is it inside the cavity?")
            if piece.dim == 3:
                taken |= tags
            placed.append([insert, piece, sorted(tags)])
        # a volume of an insert overlaps the body, so the body's own pieces include it: what is
        # left after taking those out is the vacuum
        vac_vols = sorted(inside_body - taken)
        if not vac_vols:
            raise ValueError("The inserts fill the whole cavity: no vacuum left.")

        # solid metal: the field does not enter, so drop the volume and keep its skin
        for entry in placed:
            _, piece, tags = entry
            if piece.dim == 3 and isinstance(piece.attribute, Conductor):
                skin = self._faces_of(tags)
                occ.remove([(3, t) for t in tags])
                occ.synchronize()
                loose = [(2, f) for f in skin if len(gmsh.model.getAdjacencies(2, f)[0]) == 0]
                if loose:                    # the part of the skin that was outside the cavity
                    occ.remove(loose, recursive=True)
                    occ.synchronize()
                entry[2] = sorted(skin - {f for _, f in loose})
                if not entry[2]:
                    raise ValueError(f"{entry[0].name}: the metal part does not touch the cavity volume.")

        if verbose:
            print(f"Created vacuum volumes: {vac_vols}, insulator volumes: {ins_vols}")
        return vac_vols, ins_vols, [tuple(entry) for entry in placed]

    # ---------- 3. physical groups ----------
    def _add_groups(self, cavity, ins_per_port, placed=(), verbose=False):
        add = gmsh.model.addPhysicalGroup
        groups = {"cavity": add(3, cavity, name="cavity")}
        self.regions, self._refine = {}, []
        used = set()

        # faces of the inserts are theirs: a port must not mistake one for its pin
        insert_faces = {f for _, piece, tags in placed
                        if not (piece.dim == 3 and isinstance(piece.attribute, Dielectric)) for f in tags}
        cavity_faces = self._faces_of(cavity) - insert_faces

        for i, (port, insulator) in enumerate(zip(self.ports, ins_per_port), start=1):
            faces = port.classify_faces(self._faces_of(insulator), cavity_faces)
            if verbose:
                print(f"port{i} faces: {faces}")
            groups[f"port{i}_insulator"] = add(3, insulator, name=f"port{i}_insulator")
            groups[f"port{i}"] = add(2, list(faces["ring"]), name=f"port{i}")
            groups[f"port{i}_pin"] = add(2, list(faces["pin"]), name=f"port{i}_pin")
            groups[f"port{i}_shield"] = add(2, list(faces["shield"]), name=f"port{i}_shield")
            used |= faces["ring"] | faces["pin"] | faces["shield"] | faces["interface"]

        for insert, piece, tags in placed:
            name = f"{insert.name}_{piece.key}"
            if name in groups:
                raise ValueError(f"two groups named {name!r}: rename an insert, a layer or a junction.")
            volume = piece.dim == 3 and isinstance(piece.attribute, Dielectric)
            dim = 3 if volume else 2
            groups[name] = add(dim, tags, name=name)
            self.regions[name] = Region(dim, piece.attribute, piece.direction, piece.axes)
            if not volume:
                used |= set(tags)
            if piece.mesh_size:
                faces = self._faces_of(tags) if volume else tags
                self._refine.append((sorted(faces), piece.mesh_size, insert.mesh_grow))

        # a face with a volume on one side only is the outer boundary; with one on each side it
        # is an interface between two materials, which needs no boundary condition
        all_faces = {s for _, s in gmsh.model.getEntities(2)}
        outer = {f for f in all_faces if len(set(gmsh.model.getAdjacencies(2, f)[0])) == 1}
        free = all_faces - used
        for name, selector, attribute in self.assignments:
            if name in groups:
                raise ValueError(f"two groups named {name!r}: pick another name for the assignment.")
            picked = selector.pick(free)
            if not picked:
                raise ValueError(f"{name}: {selector!r} picks no free face. Check the coordinates (mm, cavity frame).")
            groups[name] = add(2, sorted(picked), name=name)
            self.regions[name] = Region(2, attribute, getattr(attribute, "direction", None))
            used |= picked
            free -= picked
            if verbose:
                print(f"{name}: faces {sorted(picked)}")

        walls = outer - used
        interfaces = all_faces - outer            # those in a group (a metal sheet) are simply not missing
        if verbose:
            print(f"cavity walls: {sorted(walls)}, interfaces: {sorted(interfaces)}")
        if walls:
            groups["cavity_wall"] = add(2, sorted(walls), name="cavity_wall")
        return groups, interfaces

    def _check_faces_to_groups(self, interfaces, verbose=False) -> bool:
        """True if every face is in a physical group, except the interfaces between volumes."""
        all_faces = {s for _, s in gmsh.model.getEntities(2)}
        grouped = set()
        for dim, tag in gmsh.model.getPhysicalGroups(2):
            grouped |= set(gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
        missing = all_faces - grouped - interfaces
        if verbose:
            print(f"all faces: {len(all_faces)}, grouped: {len(grouped)}, "
                f"interfaces: {len(interfaces)}, missing: {sorted(missing)}")
        return not missing

    @staticmethod
    def _group_dim(name: str) -> int:
        """Dimension of the physical group `name` in the current model."""
        for dim, tag in gmsh.model.getPhysicalGroups():
            if gmsh.model.getPhysicalName(dim, tag) == name:
                return dim
        raise KeyError(f"no physical group named {name!r}")

    def _set_mesh_sizes(self) -> None:
        """Mesh finely at the pieces that ask for it, relaxing to `options.lc` away from them."""
        field = gmsh.model.mesh.field
        thresholds = []
        for faces, size, grow in self._refine:
            curves = sorted({c for _, c in gmsh.model.getBoundary([(2, f) for f in faces], oriented=False, combined=False)})
            distance = field.add("Distance")
            field.setNumbers(distance, "CurvesList", curves)
            field.setNumbers(distance, "SurfacesList", faces)
            field.setNumber(distance, "Sampling", 50)
            threshold = field.add("Threshold")
            field.setNumber(threshold, "InField", distance)
            field.setNumber(threshold, "SizeMin", size)
            field.setNumber(threshold, "SizeMax", self.options.lc)
            field.setNumber(threshold, "DistMin", size)
            field.setNumber(threshold, "DistMax", max(grow, 2 * size))
            thresholds.append(threshold)
        if thresholds:
            smallest = field.add("Min")
            field.setNumbers(smallest, "FieldsList", thresholds)
            field.setAsBackgroundMesh(smallest)

    # ---------- main ----------
    def _prepare(self, name, verbose: bool = False) -> dict[str, int]:
        """Build the geometry and tag every volume and face. Call inside a gmsh session."""
        gmsh.model.add(name)
        cavity, ins_per_port, placed = self._build(gmsh.model.occ, verbose=verbose)
        groups, interfaces = self._add_groups(cavity, ins_per_port, placed, verbose=verbose)
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
            self._set_mesh_sizes()
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

    @staticmethod
    def _faces_of(volumes: list[int]) -> set[int]:
        "All boundaries faces of some volume"
        dimtags = [(3, v) for v in volumes]
        return {s for _, s in gmsh.model.getBoundary(dimtags, oriented=False, combined=False)}
