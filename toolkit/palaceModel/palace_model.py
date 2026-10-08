"""Palace simulations of a `Cavity`, built and run with pypalace.

`PalaceModel` meshes the cavity and turns its physical groups, together with the materials chosen
when the cavity and its ports were designed, into a Palace config with pypalace's builders. Each
simulation type only adds its `Solver` block and the methods that read its results:
`EigensimPalace` (eigenmodes, Q, kappa_ext per port) and `DrivensimPalace` (S-parameters).

Everything that is not Palace-specific -- paths, meshing, the result contract -- lives in
`SimModel`, which `ComsolModel` shares.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from pypalace.builder import Boundaries, Domains
from pypalace.config import Config
from pypalace.palace_env import get_palace_executable
from pypalace.simulation import Simulation

from toolkit.components.ports import CoaxPort
from toolkit.materials import Conductor, Dielectric
from toolkit.simModel import SimModel, SimModelOptions


@dataclass(kw_only=True)
class PalaceModelOptions(SimModelOptions):
    """Options shared by every Palace simulation."""
    L0: float = 1e-3        # mesh length unit in metres (1e-3: the mesh is in mm)


class PalaceModel(SimModel):
    backend = "palace"
    problem_type: str = ""  # Palace config["Problem"]["Type"], set by each subclass
    output_file: str = ""   # file Palace always writes for this problem type, used to detect a failed run

    # ---------- config blocks ----------
    def _config_problem(self) -> dict:
        return {"Type": self.problem_type, "Output": str(self.results_dir), "Verbose": self.options.verbose}

    def _config_model(self) -> dict:
        return {"Mesh": str(self.mesh_file), "L0": self.options.L0}

    def _config_domains(self, groups: dict[str, int]) -> dict:
        """Volumes: the cavity and one insulator per port."""
        materials = [self._dielectric(self.cavity.volume_material, groups["cavity"], "cavity volume")]
        for i, port in enumerate(self.cavity.ports, start=1):
            materials.append(self._dielectric(port.isolated_material, groups[f"port{i}_insulator"], f"port{i} insulator"))
        return {"Materials": materials}

    def _config_boundaries(self, groups: dict[str, int]) -> dict:
        """Faces: metals (cavity wall, pins, shields) and one lumped port per coax."""
        metals = [(self.cavity.cavity_wall_material, groups["cavity_wall"], "cavity wall")]
        bcs = []
        for i, port in enumerate(self.cavity.ports, start=1):
            metals.append((port.pin_material, groups[f"port{i}_pin"], f"port{i} pin"))
            metals.append((port.shield_material, groups[f"port{i}_shield"], f"port{i} shield"))
            bcs.append(Boundaries.LumpedPort(Index=i, Attributes=[groups[f"port{i}"]], Direction="+R",R=port.R, Excitation=self._excitation(i, port)))

        pec = []
        for material, group, what in metals:
            if not isinstance(material, Conductor):
                raise TypeError(f"{what}: expected a Conductor, got {material!r}")
            if material.superconducting:
                pec.append(group)
            else:
                bcs.append(Boundaries.Conductivity(Attributes=[group], Conductivity=material.conductivity.value, Permeability=material.permeability))
        if pec:
            # pypalace keeps a single PEC entry, so every superconducting face goes into one
            bcs.append(Boundaries.PEC(Attributes=pec))
        return {"BCs": bcs}

    def _excitation(self, index: int, port: CoaxPort) -> int | None:
        """Excitation of a lumped port; only driven simulations excite ports."""
        return None

    @abstractmethod
    def _config_solver(self) -> dict:
        """Arguments for `Config.add_Solver`: {"Simulation": Solver.<Type>(...), "Order": ...}."""

    @staticmethod
    def _dielectric(material: Dielectric, group: int, what: str) -> dict:
        if not isinstance(material, Dielectric):
            raise TypeError(f"{what}: expected a Dielectric, got {material!r}")
        return Domains.Material(Attributes=[group], Permeability=material.permeability,
                                Permittivity=material.permittivity.to_palace(),
                                LossTan=material.loss_tangent.to_palace())

    def _make_config(self, groups: dict[str, int]) -> Config:
        """Build and save the Palace config with pypalace."""
        conf = Config(str(self.output_dir / f"{self.name}.json"))
        conf.add_Problem(**self._config_problem())
        conf.add_Model(**self._config_model())
        conf.add_Domains(**self._config_domains(groups))
        conf.add_Boundaries(**self._config_boundaries(groups))
        conf.add_Solver(**self._config_solver())
        conf.save_config()
        return conf

    # ---------- run ----------
    def run(self) -> Simulation:
        """Mesh (once), write the config and run Palace."""
        if self.groups is None:
            self._make_mesh()
        conf = self._make_config(self.groups)

        expected = self.results_dir / self.output_file
        expected.unlink(missing_ok=True)       # so a result left by a previous run can't hide a failure
        self.sim = Simulation(conf, path_to_palace=get_palace_executable())
        self.sim.run(self.options.num_cpus)
        if not expected.exists():
            raise RuntimeError(f"Palace did not write {expected}: the run failed, see the log above.")
        return self.sim
