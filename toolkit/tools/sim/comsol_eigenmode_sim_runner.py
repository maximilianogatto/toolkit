"""Eigenmode extraction via COMSOL (`SQDMetal.COMSOL.SimRFsParameter`).

Unlike the PALACE runners (`drivenmodal_sim_runner.py`, `eigenmode_sim_runner.py`), COMSOL's
`run_only_eigenfrequencies` has two sharp edges this class exists to remove:

1. It assumes an `"Eigenfrequency"` study step exists in the solver sequence, which is only
   created when `COMSOL_Simulation_RFsParameters` was built with `adaptive='Multiple'`
   (`SimRFsParameter.py:52-56`) - calling it otherwise fails with an opaque Java error, not a
   clear Python one. `ComsolEigenmodeSim` always constructs with `adaptive='Multiple'`, so this
   can't happen.
2. It creates two temporary COMSOL objects (`'gev1'`, `'tbl1'`) and only removes them on the
   success path - if anything raises in between (e.g. `computeResult()` fails), they're left
   dangling, and a retry after fixing the real problem fails again with an unrelated
   duplicate-tag error. `get_eigenfrequencies` reimplements the same COMSOL calls with the
   cleanup in a `finally`.

COMSOL's resource model also differs from PALACE's: there's no local subprocess or gmsh session
to release, but a single COMSOL engine (`COMSOL_Model._engine`, an `mph.Client`) is shared across
every `COMSOL_Model` in the process. `ComsolEigenmodeSim` is a context manager that removes just
its own model from that shared engine on exit (`mph.Client.remove`), rather than the blanket
`COMSOL_Model.close_all_models()`, which would tear down every other COMSOL model open in the
same session too.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import jpype.types as jtypes
import numpy as np
from qiskit_metal.qlibrary.core.qroute import QRoute
from qiskit_metal.qlibrary.terminations.launchpad_wb_driven import LaunchpadWirebondDriven
from SQDMetal.COMSOL.Model import COMSOL_Model
from SQDMetal.COMSOL.SimRFsParameter import COMSOL_Simulation_RFsParameters

from toolkit.tools.sim._common import build_and_validate_subdesign

logger = logging.getLogger(__name__)

_PORT_TYPES = {"CPW_on_Launcher", "CPW_on_Route"}


@dataclass
class ComsolPortOptions:
    """Spec for a single port. COMSOL's port constructors (unlike PALACE's) don't take an
    impedance -it's fixed at 50 Ohm implicitly, per `create_port_CPW_on_Launcher`'s docstring."""
    component_name: str
    type: str = "CPW_on_Launcher"
    pin_name: str = "end"                    # Only used for 'CPW_on_Route'.
    len_launch: float = 20e-6

    def __post_init__(self):
        if self.type not in _PORT_TYPES:
            raise ValueError(f"Unsupported port type: {self.type!r}. Supported: {sorted(_PORT_TYPES)}.")


def _create_port(sim_RF: COMSOL_Simulation_RFsParameters, subdesign, port: ComsolPortOptions) -> None:
    if port.component_name not in subdesign.components:
        raise KeyError(f"Port component {port.component_name!r} is not in the sub-design's kept components.")
    comp = subdesign.components[port.component_name]

    if port.type == "CPW_on_Launcher":
        if not isinstance(comp, LaunchpadWirebondDriven):
            logger.warning("Port %r is type 'CPW_on_Launcher' but component is a %s, not a "
                           "LaunchpadWirebondDriven.", port.component_name, type(comp).__name__)
        sim_RF.create_port_CPW_on_Launcher(port.component_name, len_launch=port.len_launch)
    else:  # 'CPW_on_Route'
        if not isinstance(comp, QRoute):
            logger.warning("Port %r is type 'CPW_on_Route' but component is a %s, not a QRoute.",
                           port.component_name, type(comp).__name__)
        sim_RF.create_port_CPW_on_Route(port.component_name, pin_name=port.pin_name, len_launch=port.len_launch)

    logger.info("Created port %d (%s) on %r.", len(sim_RF._ports), port.type, port.component_name)


class ComsolEigenmodeSim:
    """Build a sub-design, set up ports and metal, mesh, and extract eigenfrequencies via COMSOL.

    Usage:
        with ComsolEigenmodeSim(design, "resonador", shift_freq=7.3e9, num_eigs=3,
                                components=["LP1", "LP2", "meandro_1", ...]) as sim:
            sim.add_metallic(1)
            sim.add_port(ComsolPortOptions(component_name="LP1"))
            sim.add_port(ComsolPortOptions(component_name="LP2"))
            sim.add_fine_mesh_components(["LP1", "LP2"], min_size=5e-6, max_size=60e-6)
            sim.build(mesh_structure="Fine")
            freqs = sim.get_eigenfrequencies()   # complex Hz, closest `num_eigs` modes to shift_freq

    Args:
        design: full `QDesign` to pull `components` from.
        name: simulation name (also the COMSOL model's name).
        shift_freq: shift-invert anchor, in Hz -COMSOL searches for eigenmodes closest to this
            frequency (see `SimRFsParameter.py`'s `eigwhich='lr'` + `shift` combination: not a
            lower bound, proximity to this point in either direction).
        num_eigs: how many eigenmodes to compute, ranked by proximity to `shift_freq`. Consider
            asking for more than you need (e.g. 3-5) at least once to check none of them is a
            spurious/parasitic mode that happened to land closer to `shift_freq` than the one
            you actually want.
        components: names of the components to keep; passed to `build_subdesign`. `None` keeps
            every component in `design`.
        open_terminations / terminations: passed through to `build_subdesign` (see
            `_common.build_and_validate_subdesign`).
        bottom_grounded: passed through to `COMSOL_Model.initialize_model`.
    """

    def __init__(
        self,
        design,
        name: str,
        shift_freq: float,
        num_eigs: int,
        components: Optional[list[str]] = None,
        open_terminations: Optional[list[tuple[str, str]]] = None,
        terminations: Optional[dict[tuple[str, str], str]] = None,
        bottom_grounded: bool = False,
    ):
        COMSOL_Model.init_engine()
        self.subdesign, self.dielectric_material = build_and_validate_subdesign(
            design, name, components, open_terminations, terminations, "COMSOL_Simulation_RFsParameters")

        self.model = COMSOL_Model(name)
        # adaptive is always 'Multiple' -not exposed as a parameter- so get_eigenfrequencies
        # always has the 'Eigenfrequency' study step it needs; see module docstring.
        self.sim_RF = COMSOL_Simulation_RFsParameters(self.model, adaptive="Multiple",
                                                       modal_min_freq_num_eigs=(shift_freq, num_eigs))
        self.model.initialize_model(self.subdesign, [self.sim_RF], bottom_grounded=bottom_grounded)
        logger.info("Initialized COMSOL model %r (shift=%.6g GHz, num_eigs=%d).",
                   name, shift_freq / 1e9, num_eigs)
        self._built = False

    def add_metallic(self, layer_id: int, **kwargs) -> None:
        self.model.add_metallic(layer_id, **kwargs)

    def add_port(self, port: ComsolPortOptions) -> None:
        _create_port(self.sim_RF, self.subdesign, port)

    def add_fine_mesh_components(self, qObjNames: list[str], min_size: float = 1e-7, max_size: float = 5e-6) -> None:
        """Refine the mesh over a group of components (`COMSOL_Model.fine_mesh_components`).
        Unlike PALACE, mesh refinement lives on the model, not the simulation, and there's no
        taper/metals_only equivalent here."""
        self.model.fine_mesh_components(qObjNames, minElementSize=min_size, maxElementSize=max_size)
        logger.info("Added fine mesh over %d component(s): %s.", len(qObjNames), qObjNames)

    def build(self, mesh_structure: str = "Normal", **kwargs) -> None:
        """Fuse metals and build geometry/materials/mesh. Must be called before
        `get_eigenfrequencies`."""
        self.model.fuse_all_metals()
        self.model.build_geom_mater_elec_mesh(mesh_structure=mesh_structure, **kwargs)
        self._built = True
        logger.info("Model built (mesh_structure=%s).", mesh_structure)

    def get_eigenfrequencies(self, return_dofs: bool = False):
        """Run the eigenfrequency study step and return the computed eigenfrequencies.

        Reimplements `COMSOL_Simulation_RFsParameters.run_only_eigenfrequencies` with its two
        temporary COMSOL objects ('gev1' EvalGlobal, 'tbl1' Table) cleaned up in a `finally`, so
        a failure here doesn't leave them behind to break a retry with an unrelated
        duplicate-tag error.

        Args:
            return_dofs: if True, also return the number of mesh degrees of freedom.

        Returns:
            list[complex]: eigenfrequencies in Hz (real part = resonant frequency; imaginary
            part relates to damping/loss if losses are enabled). Or `(freqs, dofs)` if
            `return_dofs=True`.

        Raises:
            RuntimeError: if `build()` hasn't been called yet.
        """
        if not self._built:
            print("Warning: COMSOL model hasn't been built yet. Calling build() automatically to avoid a COMSOL error.")
            self.build()  # Auto-build if the user forgot, to avoid a COMSOL error.

        jc = self.sim_RF.jc
        logger.info("Running eigenfrequency study step...")
        jc.sol(self.sim_RF._soln).runFromTo("st1", "su1")
        try:
            jc.result().numerical().create("gev1", "EvalGlobal")
            jc.result().numerical("gev1").set("data", self.sim_RF.dset_name)
            jc.result().numerical("gev1").set("expr", jtypes.JArray(jtypes.JString)(["numberofdofs"]))
            jc.result().table().create("tbl1", "Table")
            jc.result().numerical("gev1").set("table", "tbl1")
            jc.result().numerical("gev1").computeResult()
            jc.result().numerical("gev1").setResult()

            freqs = [1e9 * np.complex128(str(y).replace("i", "j"))
                    for y in [x[0] for x in jc.result().table("tbl1").getTableData(True)]]
            dofs = int(np.array(jc.result().numerical("gev1").computeResult())[0, 0, 0])
        finally:
            for remove in (lambda: jc.result().table().remove("tbl1"),
                          lambda: jc.result().numerical().remove("gev1")):
                try:
                    remove()
                except Exception:
                    pass  # Wasn't created this attempt (failed before reaching it), or already removed.

        logger.info("Eigenfrequencies (GHz): %s.", [f"{f/1e9:.6g}" for f in freqs])
        return (freqs, dofs) if return_dofs else freqs

    def close(self) -> None:
        """Remove this simulation's model from the shared COMSOL engine. Only this model is
        affected -other `COMSOL_Model`/`ComsolEigenmodeSim` instances in the same session are
        untouched, unlike `COMSOL_Model.close_all_models()`."""
        COMSOL_Model._engine.remove(self.model._model)
        logger.debug("Removed COMSOL model %r from the engine.", self.model.model_name)

    def __enter__(self) -> "ComsolEigenmodeSim":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
