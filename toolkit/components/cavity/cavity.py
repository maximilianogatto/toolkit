# abstract class to make cavities
from abc import ABC, abstractmethod
from dataclasses import dataclass
import gmsh
from pathlib import Path
from toolkit.materials.materials import Dielectric, Conductor, Material, get_material
from toolkit.components.ports import Port
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

    @staticmethod
    def _faces_of(volumes: list[int]) -> set[int]:
        "All boundaries faces of some volume"
        dimtags = [(3, v) for v in volumes]
        return {s for _, s in gmsh.model.getBoundary(dimtags, oriented=False, combined=False)}
