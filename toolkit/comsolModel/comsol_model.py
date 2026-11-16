"""COMSOL simulations of a `Cavity`, driven from Python with mph.

The geometry is still built once, by gmsh, inside the `Cavity` classes: `export_step` writes a
STEP file plus a sidecar that records, for every physical group, a point **on** each face and the
normal there. COMSOL imports the STEP as CAD, meshes it itself, and finds each group again with a
Ball selection at those points. That way a new cavity geometry is still a single `_add_body`, and
the two backends cannot drift apart.

Nothing here imports COMSOL at module level: `mph` is loaded inside `_client()`, so the toolkit
still works on a machine without a licence. Two ways to reach a COMSOL that lives elsewhere:

* run your script on that machine (toolkit installed there), or
* start a server there (`comsol mphserver -np N`), tunnel it (`ssh -L 2036:localhost:2036 ui02`)
  and pass `ComsolModelOptions(host="localhost", port=2036)`.

Calls that go through COMSOL's own Java API are each in a small method and marked `TODO(verify)`
where the property name still has to be confirmed against your COMSOL version -- the model tree in
the GUI (right-click a node -> Copy as Code to Clipboard -> Java) is the fastest way to check one.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass

from toolkit.materials import Conductor, Dielectric
from toolkit.simModel import SimModel, SimModelOptions


@dataclass(kw_only=True)
class ComsolModelOptions(SimModelOptions):
    """Options shared by every COMSOL simulation."""
    length_unit: str = "mm"              # the STEP from gmsh is in mm
    mesh_max: float | None = None        # max element size; default: the cavity's own lc
    selection_radius: float = 0.5        # Ball radius as a fraction of the probe's clearance
    host: str | None = None              # None: start COMSOL locally; else connect to a server
    port: int = 2036
    comsol_version: str | None = None    # None: whatever mph finds
    save_mph: bool = True                # write <name>.mph so the model opens in the GUI


class ComsolModel(SimModel):
    """One COMSOL study of one cavity. Subclasses add the study and read its results."""

    backend = "comsol"
    study_type: str = ""                 # "Eigenfrequency" or "Frequency Domain"

    _GEOM, _PHYS, _MESH, _STUDY = "geom1", "emw", "mesh1", "std1"

    # ---------- session ----------
    def _client(self):
        """A COMSOL session: a local one, or a server started elsewhere (the cluster)."""
        import mph                                   # imported late: no COMSOL, no import error
        o = self.options
        if o.host is not None:
            return mph.Client(host=o.host, port=o.port)        # TODO(verify) mph.Client signature
        return mph.start(cores=o.num_cpus, version=o.comsol_version)

    def __enter__(self) -> ComsolModel:
        return self

    def __exit__(self, *exc) -> None:
        """Remove only this model from the shared engine, never `close_all_models`."""
        if self.sim is not None and self._client_handle is not None:
            self._client_handle.remove(self.sim)
            self.sim = None

    _client_handle = None

    # ---------- run ----------
    def run(self):
        """Export the geometry, build the COMSOL model, solve, and keep the model handle."""
        if self.cavity.inserts or self.cavity.assignments:
            raise NotImplementedError("The COMSOL backend does not translate inserts (chips, CAD "
                                      "parts) or assigned faces yet: use the Palace backend.")
        self._export_geometry()
        self._client_handle = self._client()
        model = self._client_handle.create(self.name)
        java = model.java

        self._build_geometry(java)
        self._build_selections(java)
        self._build_materials(java)
        self._build_physics(java)
        self._build_mesh(java)
        self._build_study(java)

        model.solve(self._STUDY)
        if self.options.save_mph:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(self.output_dir / f"{self.name}.mph"))
        self.sim = model
        return model

    # ---------- model tree ----------
    def _build_geometry(self, java) -> None:
        geom = java.geom().create(self._GEOM, 3)
        geom.lengthUnit(self.options.length_unit)
        imp = geom.create("imp1", "Import")
        imp.set("type", "cad")                       # TODO(verify) "cad" needs the CAD Import Module;
        imp.set("filename", str(self.step_file))     #              plain STEP may import as "native"
        geom.run()

    def _build_selections(self, java) -> None:
        """One Ball selection per entity, unioned into one selection per physical group.

        The probe points come from the sidecar, not from centres of mass: the centre of mass of a
        cylindrical wall lies on the axis, and that of the annular port ring lies inside the pin.
        """
        import jpype.types as jtypes
        tags = json.loads(self.tags_file.read_text(encoding="utf-8"))
        self._selection_names = {}
        for name, group in tags["groups"].items():
            balls = []
            for k, entity in enumerate(group["entities"], start=1):
                ball = f"ball_{name}_{k}"
                sel = java.selection().create(ball, "Ball")
                sel.set("entitydim", group["dim"])
                x, y, z = entity["point"]
                sel.set("posx", float(x)); sel.set("posy", float(y)); sel.set("posz", float(z))
                sel.set("r", self._radius(entity, group["dim"]))
                sel.set("condition", "intersects")   # TODO(verify) "inside" may be safer per version
                balls.append(ball)
            union = f"sel_{name}"
            sel = java.selection().create(union, "Union")
            sel.set("entitydim", group["dim"])
            sel.set("input", jtypes.JArray(jtypes.JString)(balls))
            self._selection_names[name] = union

    def _radius(self, entity: dict, dim: int) -> float:
        """Small enough to catch one entity only, large enough to survive the import tolerance.

        `clearance` is how far the probe point sits from the entity's own boundary, so half of it
        cannot reach a neighbouring face.
        """
        clearance = entity.get("clearance") or abs(entity["size"]) ** (1.0 / dim)
        return max(self.options.selection_radius * clearance, 1e-4)

    def _build_materials(self, java) -> None:
        """One COMSOL material per volume: the cavity and each port insulator."""
        volumes = [("cavity", self.cavity.volume_material)]
        for i, port in enumerate(self.cavity.ports, start=1):
            volumes.append((f"port{i}_insulator", port.isolated_material))

        for n, (group, material) in enumerate(volumes, start=1):
            if not isinstance(material, Dielectric):
                raise TypeError(f"{group}: expected a Dielectric, got {material!r}")
            mat = java.material().create(f"mat{n}", "Common", self._GEOM)
            mat.label(f"{group} ({material.name})")
            mat.selection().named(self._selection_names[group])
            props = material.comsol_material()
            for key in ("relpermittivity", "relpermeability", "electricconductivity"):
                mat.propertyGroup("def").set(key, str(props[key]))   # COMSOL wants strings

    def _build_physics(self, java) -> None:
        """Electromagnetic Waves, Frequency Domain: metals as boundaries, coax rings as ports."""
        emw = java.physics().create(self._PHYS, "ElectromagneticWaves", self._GEOM)

        metals = [("cavity_wall", self.cavity.cavity_wall_material)]
        for i, port in enumerate(self.cavity.ports, start=1):
            metals.append((f"port{i}_pin", port.pin_material))
            metals.append((f"port{i}_shield", port.shield_material))

        for n, (group, material) in enumerate(metals, start=1):
            if not isinstance(material, Conductor):
                raise TypeError(f"{group}: expected a Conductor, got {material!r}")
            feature, props = material.comsol_boundary()
            node = emw.feature().create(f"bc{n}", feature, 2)
            node.label(f"{group} ({material.name})")
            node.selection().named(self._selection_names[group])
            for key, value in props.items():
                node.set(key, str(value))            # TODO(verify) Impedance property names

        for i, port in enumerate(self.cavity.ports, start=1):
            node = emw.feature().create(f"lport{i}", "LumpedPort", 2)
            node.label(f"port{i}")
            node.selection().named(self._selection_names[f"port{i}"])
            node.set("PortType", "Coaxial")          # the ring is an annulus: COMSOL's coax port
            node.set("Zref", str(port.R))            # TODO(verify) Zref vs Zin naming
            node.set("TerminalType", "Cable")        # TODO(verify) only used when excited
            self._set_excitation(node, i, port)

    def _set_excitation(self, node, index: int, port) -> None:
        """Eigenfrequency studies excite nothing; the driven model overrides this."""
        node.set("PortExcitation", "off")            # TODO(verify) property name

    def _build_mesh(self, java) -> None:
        mesh = java.mesh().create(self._MESH, self._GEOM)
        size = mesh.feature("size")
        size.set("custom", "on")
        size.set("hmax", str(self.options.mesh_max or self.cavity.options.lc))
        mesh.create("ftet1", "FreeTet")
        mesh.run()

    @abstractmethod
    def _build_study(self, java) -> None:
        """Create study `_STUDY` with this simulation's step and settings."""

    # ---------- results ----------
    def evaluate(self, expression: str):
        """Evaluate a COMSOL expression on the solved model (`'freq'`, `'emw.S21dB'`, ...)."""
        return self._require_run().evaluate(expression)
