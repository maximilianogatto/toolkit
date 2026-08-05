"""Eigenmode simulation runner for SQDMetal/PALACE.

Same shape as `drivenmodal_sim_runner.py` (see that file's module docstring for the gmsh/
subprocess cleanup rationale, shared via `_common.palace_run_cleanup`). Built, meshed and run end
to end by a single function that takes the full `design`, builds its own sub-design internally,
and cleans up the gmsh session/PALACE subprocess even if something raises partway through.

The main difference from the driven-modal runner: there's no frequency sweep or excitation here
- PALACE searches for eigenmodes above `starting_freq` via SLEPc shift-invert. Ports (if any) set
boundary conditions the modes see (a Josephson junction's inductance, a resistive CPW feed for
external-Q via EPR, ...); none of them is "excited".

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from qiskit_metal.qlibrary.core.qroute import QRoute
from qiskit_metal.qlibrary.terminations.launchpad_wb_driven import LaunchpadWirebondDriven
from SQDMetal.PALACE.Eigenmode_Simulation import PALACE_Eigenmode_Simulation
from SQDMetal.Utilities.Materials import MaterialInterface

from toolkit.tools.sim._common import (
    AMROptions,
    FineMeshComponentOptions,
    MeshAlongPathOptions,
    apply_amr,
    apply_fine_mesh,
    build_and_validate_subdesign,
    palace_run_cleanup,
)

logger = logging.getLogger(__name__)

_PORT_TYPES = {"CPW_on_Launcher", "CPW_on_Route", "JosephsonJunction"}


@dataclass
class EigenmodeSimOptions:
    """PALACE solver/mesh options for an eigenmode simulation."""
    mesh_refinement: int = 0
    solns_to_save: int = 1
    solver_order: int = 2
    solver_tol: float = 1.0e-8
    solver_maxits: int = 200
    fillet_resolution: int = 12
    palace_dir: str = ""                     # Path to the PALACE binary.
    num_cpus: int = 1
    meshing: str = "GMSH"                    # 'GMSH' or 'COMSOL'
    mode: str = "simPC"                      # 'simPC' or 'HPC'
    create_files: bool = True
    starting_freq: float = 5e9               # Search start for the SLEPc shift-invert solver, in Hz.
    number_of_freqs: int = 1                 # Number of eigenmodes to search for above starting_freq.


@dataclass
class PortOptions:
    """Spec for a single boundary-condition port. `type` selects which SQDMetal port
    constructor is used and which of the other fields apply.

    * 'CPW_on_Launcher': uses component_name, len_launch, impedance_R/L/C.
    * 'CPW_on_Route': uses component_name, pin_name, len_launch, impedance_R/L/C.
    * 'JosephsonJunction': uses component_name, junction_index, and one of L_J/E_J_Hertz, C_J.
    """
    component_name: str
    type: str = "JosephsonJunction"
    pin_name: str = "end"                    # Only used for 'CPW_on_Route'.
    len_launch: float = 20e-6
    impedance_R: float = 50
    impedance_L: float = 0
    impedance_C: float = 0
    junction_index: int = 0                  # Only used for 'JosephsonJunction'.
    L_J: Optional[float] = None              # Only used for 'JosephsonJunction'.
    E_J_Hertz: Optional[float] = None        # Only used for 'JosephsonJunction'.
    C_J: float = 0                           # Only used for 'JosephsonJunction'.

    def __post_init__(self):
        if self.type not in _PORT_TYPES:
            raise ValueError(f"Unsupported port type: {self.type!r}. Supported: {sorted(_PORT_TYPES)}.")
        if self.type == "JosephsonJunction" and self.L_J is None and self.E_J_Hertz is None:
            raise ValueError(f"Port {self.component_name!r}: JosephsonJunction ports need L_J or E_J_Hertz.")


def _to_user_options(conf: EigenmodeSimOptions, dielectric_material: str) -> dict:
    """Convert `EigenmodeSimOptions` + resolved substrate material to the plain dict
    `PALACE_Eigenmode_Simulation`'s `user_options` expects."""
    return {
        "mesh_refinement": conf.mesh_refinement,
        "dielectric_material": dielectric_material,
        "starting_freq": conf.starting_freq,
        "number_of_freqs": conf.number_of_freqs,
        "solns_to_save": conf.solns_to_save,
        "solver_order": conf.solver_order,
        "solver_tol": conf.solver_tol,
        "solver_maxits": conf.solver_maxits,
        "fillet_resolution": conf.fillet_resolution,
        "palace_dir": conf.palace_dir,
        "num_cpus": conf.num_cpus,
    }


def _create_port(eigen_sim: PALACE_Eigenmode_Simulation, subdesign, port: PortOptions) -> None:
    """Create `port` on `eigen_sim`. Unlike the driven-modal runner's version, there's no
    port index/excitation to track here - eigenmode ports are just boundary conditions."""
    if port.component_name not in subdesign.components:
        raise KeyError(f"Port component {port.component_name!r} is not in the sub-design's kept components.")
    comp = subdesign.components[port.component_name]

    if port.type == "CPW_on_Launcher":
        if not isinstance(comp, LaunchpadWirebondDriven):
            logger.warning("Port %r is type 'CPW_on_Launcher' but component is a %s, not a "
                           "LaunchpadWirebondDriven.", port.component_name, type(comp).__name__)
        eigen_sim.create_port_CPW_on_Launcher(
            port.component_name, len_launch=port.len_launch,
            impedance_R=port.impedance_R, impedance_L=port.impedance_L, impedance_C=port.impedance_C)
    elif port.type == "CPW_on_Route":
        if not isinstance(comp, QRoute):
            logger.warning("Port %r is type 'CPW_on_Route' but component is a %s, not a QRoute.",
                           port.component_name, type(comp).__name__)
        eigen_sim.create_port_CPW_on_Route(
            port.component_name, pin_name=port.pin_name, len_launch=port.len_launch,
            impedance_R=port.impedance_R, impedance_L=port.impedance_L, impedance_C=port.impedance_C)
    else:  # 'JosephsonJunction'
        jj_kwargs = dict(junction_index=port.junction_index, C_J=port.C_J)
        if port.L_J is not None:
            jj_kwargs["L_J"] = port.L_J
        if port.E_J_Hertz is not None:
            jj_kwargs["E_J_Hertz"] = port.E_J_Hertz
        eigen_sim.create_port_JosephsonJunction(port.component_name, **jj_kwargs)

    logger.info("Created port (%s) on %r.", port.type, port.component_name)


def run_eigenmode_sim(
    design,
    name: str,
    output_path: str,
    components: Optional[list[str]] = None,
    open_terminations: Optional[list[tuple[str, str]]] = None,
    sim_options: Optional[EigenmodeSimOptions] = None,
    terminations: Optional[dict[tuple[str, str], str]] = None,
    ports: Optional[list[PortOptions]] = None,
    metallic_layer: int = 1,
    fine_mesh_components: Optional[list[FineMeshComponentOptions]] = None,
    fine_mesh_paths: Optional[list[MeshAlongPathOptions]] = None,
    amr_options: Optional[AMROptions] = None,
    kinetic_inductance: Optional[float] = None,
    epr_interfaces: Optional[tuple[MaterialInterface, MaterialInterface, MaterialInterface]] = None,
):
    """Build, mesh, and run a PALACE eigenmode simulation end to end.

    Guarantees cleanup of the resources a run acquires, same as `run_drivenmodal_sim`: kills any
    still-running local PALACE subprocess, and finalizes the gmsh session eagerly.

    Args:
        design: full `QDesign` to pull `components` from.
        name: simulation name (also the output subfolder name).
        output_path: `sim_parent_directory` for `PALACE_Eigenmode_Simulation`.
        components: names of the components to keep; passed to `build_subdesign`. Defaults to
            every component in `design` if omitted.
        open_terminations: `(component_name, pin_name)` pairs, for dangling pins left by
            components in `components` that connected to a dropped component, that should be
            terminated as open. Convenience for the common case; merged into `terminations`.
        sim_options: PALACE solver/mesh/search options. Defaults to `EigenmodeSimOptions()`.
        terminations: passed straight through to `build_subdesign` for full control (e.g. to
            terminate a specific dangling pin as `'short'` instead of the `'open'` default).
        ports: optional `PortOptions` list -- boundary conditions the eigenmode sees (e.g. a
            Josephson junction's inductance, or a resistive CPW feed for external-Q via EPR).
            Unlike the driven-modal runner, none of these is "excited".
        metallic_layer: design layer id to render as metal (`add_metallic`).
        fine_mesh_components: optional per-group mesh refinement (`fine_mesh_components`).
        fine_mesh_paths: optional per-path mesh refinement (`fine_mesh_along_path`).
        amr_options: optional adaptive mesh refinement (`enable_mesh_refinement`).
        kinetic_inductance: optional surface kinetic inductance, in Henry, applied to all
            metals (`add_kinetic_inductance`). Omit for a purely geometric (lossless) metal.
        epr_interfaces: optional `(substrate_air, substrate_metal, metal_air)` `MaterialInterface`
            triple for EPR (energy participation ratio) interface setup (`setup_EPR_interfaces`).

    Returns:
        pandas.DataFrame: eigenmode frequencies (and Q, if `epr_interfaces` was set) -- see
        `PALACE_Eigenmode_Simulation.retrieve_data`.

    Raises:
        ValueError: if the design's substrate material isn't 'silicon' or 'sapphire', or if a
            `PortOptions` entry is invalid (see `PortOptions.__post_init__`).
    """
    t_start = time.perf_counter()
    sim_options = sim_options or EigenmodeSimOptions()

    subdesign, dielectric_material = build_and_validate_subdesign(
        design, name, components, open_terminations, terminations, "PALACE_Eigenmode_Simulation")

    eigen_sim = PALACE_Eigenmode_Simulation(
        name=name,
        metal_design=subdesign,
        sim_parent_directory=output_path,
        mode=sim_options.mode,
        meshing=sim_options.meshing,
        user_options=_to_user_options(sim_options, dielectric_material),
        create_files=sim_options.create_files,
    )
    logger.info("Created PALACE_Eigenmode_Simulation %r (mode=%s, meshing=%s) at %r.",
               name, sim_options.mode, sim_options.meshing, output_path)

    with palace_run_cleanup(eigen_sim):
        eigen_sim.add_metallic(metallic_layer)
        eigen_sim.add_ground_plane()
        logger.info("Added metallic layer %d and ground plane.", metallic_layer)

        if kinetic_inductance is not None:
            eigen_sim.add_kinetic_inductance(kinetic_inductance)
            logger.info("Enabled kinetic inductance: %.3g H.", kinetic_inductance)

        if epr_interfaces is not None:
            substrate_air, substrate_metal, metal_air = epr_interfaces
            eigen_sim.setup_EPR_interfaces(substrate_air=substrate_air,
                                           substrate_metal=substrate_metal, metal_air=metal_air)
            logger.info("Enabled EPR interface setup.")

        for port in ports or []:
            _create_port(eigen_sim, subdesign, port)

        apply_fine_mesh(eigen_sim, fine_mesh_components, fine_mesh_paths)
        apply_amr(eigen_sim, amr_options)

        logger.info("Searching for %d eigenmode(s) from %.6g GHz.",
                   sim_options.number_of_freqs, sim_options.starting_freq / 1e9)

        logger.info("Preparing simulation (meshing + config file)...")
        t_prepare = time.perf_counter()
        eigen_sim.prepare_simulation()
        logger.info("Simulation prepared in %.1f s.", time.perf_counter() - t_prepare)

        logger.info("Running PALACE (%s, %d CPU(s))...", sim_options.mode, sim_options.num_cpus)
        t_run = time.perf_counter()
        data = eigen_sim.run(stream_output=True)
        logger.info("Simulation finished in %.1f s (total %.1f s including setup).",
                   time.perf_counter() - t_run, time.perf_counter() - t_start)
        return data
