"""Driven-modal (S-parameter) simulation runner for SQDMetal/PALACE.

Mirrors qiskit-metal's own `ScatteringImpedanceSim`
(qiskit_metal/analyses/simulation/scattering_impedance.py) in spirit: a setup dataclass plus a
list of port specs that build, render, and run the simulation. PALACE has no persistent renderer
object to `start()`/`stop()` like Ansys HFSS does -each run instead spins up its own gmsh session
(a global/singleton API, see `GMSH_Geometry_Builder.__init__`) and a local subprocess
(`PALACE_Driven_Simulation._run_local`, held at `driven_sim.cur_process`)- so `run_drivenmodal_sim`
wraps the whole build -> configure -> run sequence in a `palace_run_cleanup` block (see
`_common.py`) that releases both, even if something raises partway through.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from qiskit_metal.qlibrary.core.qroute import QRoute
from qiskit_metal.qlibrary.terminations.launchpad_wb_driven import LaunchpadWirebondDriven
from SQDMetal.PALACE.Frequency_Driven_Simulation import PALACE_Driven_Simulation
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
class DrivenModalSimOptions:
    """PALACE solver/mesh/sweep options for a driven-modal (S-parameter) simulation."""
    mesh_refinement: int = 0
    solns_to_save: int = 4
    solver_order: int = 2
    solver_tol: float = 1.0e-5
    solver_maxits: int = 300
    fillet_resolution: int = 12
    adaptive_solver_tol: float = 1e3         # Not adaptive by default; lower it to enable PROM.
    palace_dir: str = ""                     # Path to the PALACE binary.
    num_cpus: int = 1
    meshing: str = "GMSH"                    # 'GMSH' or 'COMSOL'
    mode: str = "simPC"                      # 'simPC' or 'HPC'
    create_files: bool = True
    center_freq: float = 5e9                 # Center frequency of the sweep, in Hz.
    bandwidth: float = 1e9                   # Full sweep width, in Hz.
    freq_points: int = 101                   # Number of frequency points in the sweep.


@dataclass
class PortOptions:
    """Spec for a single simulation port. `type` selects which SQDMetal port constructor
    is used and which of the other fields apply.

    * 'CPW_on_Launcher': uses component_name, len_launch, impedance_R/L/C.
    * 'CPW_on_Route': uses component_name, pin_name, len_launch, impedance_R/L/C.
    * 'JosephsonJunction': uses component_name, junction_index, and one of L_J/E_J_Hertz, C_J.
    """
    component_name: str
    type: str = "CPW_on_Launcher"
    pin_name: str = "end"                    # Only used for 'CPW_on_Route'.
    len_launch: float = 20e-6
    impedance_R: float = 50
    impedance_L: float = 0
    impedance_C: float = 0
    junction_index: int = 0                  # Only used for 'JosephsonJunction'.
    L_J: Optional[float] = None              # Only used for 'JosephsonJunction'.
    E_J_Hertz: Optional[float] = None        # Only used for 'JosephsonJunction'.
    C_J: float = 0                           # Only used for 'JosephsonJunction'.
    excite: bool = False                     # Set the RF excitation on this port.

    def __post_init__(self):
        if self.type not in _PORT_TYPES:
            raise ValueError(f"Unsupported port type: {self.type!r}. Supported: {sorted(_PORT_TYPES)}.")
        if self.type == "JosephsonJunction" and self.L_J is None and self.E_J_Hertz is None:
            raise ValueError(f"Port {self.component_name!r}: JosephsonJunction ports need L_J or E_J_Hertz.")


def _to_user_options(conf: DrivenModalSimOptions, dielectric_material: str) -> dict:
    """Convert `DrivenModalSimOptions` + resolved substrate material to the plain dict
    `PALACE_Driven_Simulation`'s `user_options` expects."""
    return {
        "mesh_refinement": conf.mesh_refinement,
        "dielectric_material": dielectric_material,
        "solns_to_save": conf.solns_to_save,
        "solver_order": conf.solver_order,
        "solver_tol": conf.solver_tol,
        "solver_maxits": conf.solver_maxits,
        "fillet_resolution": conf.fillet_resolution,
        "adaptive_solver_tol": conf.adaptive_solver_tol,
        "palace_dir": conf.palace_dir,
        "num_cpus": conf.num_cpus,
    }


def _create_port(driven_sim: PALACE_Driven_Simulation, subdesign, port: PortOptions) -> int:
    """Create `port` on `driven_sim` and return its 1-indexed position (for `set_port_excitation`).

    `driven_sim._ports` is SQDMetal's own internal port list, appended to by each
    `create_port_*` call -reading its length right after creating a port is the same trick
    `set_port_excitation` itself relies on (ports are 1-indexed by creation order).
    """
    if port.component_name not in subdesign.components:
        raise KeyError(f"Port component {port.component_name!r} is not in the sub-design's kept components.")
    comp = subdesign.components[port.component_name]

    if port.type == "CPW_on_Launcher":
        if not isinstance(comp, LaunchpadWirebondDriven):
            logger.warning("Port %r is type 'CPW_on_Launcher' but component is a %s, not a "
                           "LaunchpadWirebondDriven.", port.component_name, type(comp).__name__)
        driven_sim.create_port_CPW_on_Launcher(
            port.component_name, len_launch=port.len_launch,
            impedance_R=port.impedance_R, impedance_L=port.impedance_L, impedance_C=port.impedance_C)
    elif port.type == "CPW_on_Route":
        if not isinstance(comp, QRoute):
            logger.warning("Port %r is type 'CPW_on_Route' but component is a %s, not a QRoute.",
                           port.component_name, type(comp).__name__)
        driven_sim.create_port_CPW_on_Route(
            port.component_name, pin_name=port.pin_name, len_launch=port.len_launch,
            impedance_R=port.impedance_R, impedance_L=port.impedance_L, impedance_C=port.impedance_C)
    else:  # 'JosephsonJunction'
        jj_kwargs = dict(junction_index=port.junction_index, C_J=port.C_J)
        if port.L_J is not None:
            jj_kwargs["L_J"] = port.L_J
        if port.E_J_Hertz is not None:
            jj_kwargs["E_J_Hertz"] = port.E_J_Hertz
        driven_sim.create_port_JosephsonJunction(port.component_name, **jj_kwargs)

    port_index = len(driven_sim._ports)
    logger.info("Created port %d (%s) on %r.", port_index, port.type, port.component_name)
    return port_index


def run_drivenmodal_sim(
    design,
    name: str,
    ports: list[PortOptions],
    output_path: str,
    components: Optional[list[str]] = None,
    open_terminations: Optional[list[tuple[str, str]]] = None,
    sim_options: Optional[DrivenModalSimOptions] = None,
    terminations: Optional[dict[tuple[str, str], str]] = None,
    metallic_layer: int = 1,
    fine_mesh_components: Optional[list[FineMeshComponentOptions]] = None,
    fine_mesh_paths: Optional[list[MeshAlongPathOptions]] = None,
    amr_options: Optional[AMROptions] = None,
    kinetic_inductance: Optional[float] = None,
    epr_interfaces: Optional[tuple[MaterialInterface, MaterialInterface, MaterialInterface]] = None,
) -> dict:
    """Build, mesh, and run a PALACE driven-modal (S-parameter) simulation end to end.

    Guarantees cleanup of the resources a run acquires -even if an exception is raised partway
    through: kills any still-running local PALACE subprocess, and finalizes the gmsh session
    eagerly (it would otherwise sit initialized until the *next* simulation resets it, see
    `GMSH_Geometry_Builder.__init__`).

    Args:
        design: full `QDesign` to pull `components` from.
        name: simulation name (also the output subfolder name).
        components: names of the components to keep; passed to `build_subdesign`.
        ports: one `PortOptions` per port. Exactly one should have `excite=True`; if none do,
            PALACE falls back to the first resistive (R>0) port.
        output_path: `sim_parent_directory` for `PALACE_Driven_Simulation`.
        sim_options: PALACE solver/mesh/sweep options. Defaults to `DrivenModalSimOptions()`.
        open_terminations: `(component_name, pin_name)` pairs, for dangling pins -left by
            components in `components` that connected to a dropped component- that should be
            terminated as open. Convenience for the common case; merged into `terminations`.
        terminations: passed straight through to `build_subdesign` for full control (e.g. to
            terminate a specific dangling pin as `'short'` instead of the `'open'` default).
        metallic_layer: design layer id to render as metal (`add_metallic`).
        fine_mesh_components: optional per-group mesh refinement (`fine_mesh_components`).
        fine_mesh_paths: optional per-path mesh refinement (`fine_mesh_along_path`).
        amr_options: optional adaptive mesh refinement (`enable_mesh_refinement`).
        kinetic_inductance: optional surface kinetic inductance, in Henry, applied to all
            metals (`add_kinetic_inductance`). Omit for a purely geometric (lossless) metal.
        epr_interfaces: optional `(substrate_air, substrate_metal, metal_air)` `MaterialInterface`
            triple for EPR (energy participation ratio) interface setup (`setup_EPR_interfaces`).

    Returns:
        dict: `driven_sim.run()`'s output -`data['freqs']`, `data['S11']`, `data['S21']`, ...

    Raises:
        ValueError: if the design's substrate material isn't 'silicon' or 'sapphire' (the only
            dielectrics PALACE_Driven_Simulation supports), or if a `PortOptions` entry is
            invalid (see `PortOptions.__post_init__`).
    """
    t_start = time.perf_counter()
    sim_options = sim_options or DrivenModalSimOptions()

    subdesign, dielectric_material = build_and_validate_subdesign(
        design, name, components, open_terminations, terminations, "PALACE_Driven_Simulation")

    driven_sim = PALACE_Driven_Simulation(
        name=name,
        metal_design=subdesign,
        sim_parent_directory=output_path,
        mode=sim_options.mode,
        meshing=sim_options.meshing,
        user_options=_to_user_options(sim_options, dielectric_material),
        create_files=sim_options.create_files,
    )
    logger.info("Created PALACE_Driven_Simulation %r (mode=%s, meshing=%s) at %r.",
               name, sim_options.mode, sim_options.meshing, output_path)

    with palace_run_cleanup(driven_sim):
        driven_sim.add_metallic(metallic_layer)
        driven_sim.add_ground_plane()
        logger.info("Added metallic layer %d and ground plane.", metallic_layer)

        if kinetic_inductance is not None:
            driven_sim.add_kinetic_inductance(kinetic_inductance)
            logger.info("Enabled kinetic inductance: %.3g H.", kinetic_inductance)

        if epr_interfaces is not None:
            substrate_air, substrate_metal, metal_air = epr_interfaces
            driven_sim.setup_EPR_interfaces(substrate_air=substrate_air,
                                            substrate_metal=substrate_metal, metal_air=metal_air)
            logger.info("Enabled EPR interface setup.")

        excitation_index = None
        for port in ports:
            port_index = _create_port(driven_sim, subdesign, port)
            if port.excite:
                if excitation_index is not None:
                    raise ValueError("More than one PortOptions has excite=True; only one port "
                                     "can be the excitation source.")
                excitation_index = port_index
        if excitation_index is not None:
            driven_sim.set_port_excitation(port_index=excitation_index)
            logger.info("Set port %d as the RF excitation source.", excitation_index)
        else:
            logger.info("No port marked excite=True; PALACE will default to the first resistive port.")

        apply_fine_mesh(driven_sim, fine_mesh_components, fine_mesh_paths)
        apply_amr(driven_sim, amr_options)

        freq_start = sim_options.center_freq - sim_options.bandwidth / 2
        freq_end = sim_options.center_freq + sim_options.bandwidth / 2
        freq_step = (freq_end - freq_start) / (sim_options.freq_points - 1)
        driven_sim.set_freq_values(freq_start=freq_start, freq_end=freq_end, freq_step=freq_step)
        logger.info("Frequency sweep: %.6g GHz to %.6g GHz, %d points (step %.6g MHz).",
                   freq_start / 1e9, freq_end / 1e9, sim_options.freq_points, freq_step / 1e6)

        logger.info("Preparing simulation (meshing + config file)...")
        t_prepare = time.perf_counter()
        driven_sim.prepare_simulation()
        logger.info("Simulation prepared in %.1f s.", time.perf_counter() - t_prepare)

        logger.info("Running PALACE (%s, %d CPU(s))...", sim_options.mode, sim_options.num_cpus)
        t_run = time.perf_counter()
        data = driven_sim.run(stream_output=True)
        logger.info("Simulation finished in %.1f s (total %.1f s including setup).",
                   time.perf_counter() - t_run, time.perf_counter() - t_start)
        return data
