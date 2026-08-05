"""Capacitance (electrostatic) simulation runner for SQDMetal/PALACE.

Same shape as `drivenmodal_sim_runner.py` (see that file's module docstring for the gmsh/
subprocess cleanup rationale, shared via `_common.palace_run_cleanup`). Simpler than the RF
runners though: an electrostatic solve has no frequency, no sweep, and no ports -- every
conductor in the geometry becomes its own terminal automatically (`add_metallic`/
`add_ground_plane`, nothing else to wire up).

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from SQDMetal.PALACE.Capacitance_Simulation import PALACE_Capacitance_Simulation

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


@dataclass
class CapacitanceSimOptions:
    """PALACE solver/mesh options for a capacitance (electrostatic) simulation."""
    mesh_refinement: int = 0
    solns_to_save: int = -1                  # -1 = save the field solution for every conductor.
    solver_order: int = 2
    solver_tol: float = 1.0e-8
    solver_maxits: int = 100
    fillet_resolution: int = 12
    palace_dir: str = ""                     # Path to the PALACE binary.
    num_cpus: int = 1
    meshing: str = "GMSH"                    # 'GMSH' or 'COMSOL'
    mode: str = "simPC"                      # 'simPC' or 'HPC'
    create_files: bool = True


def _to_user_options(conf: CapacitanceSimOptions, dielectric_material: str) -> dict:
    """Convert `CapacitanceSimOptions` + resolved substrate material to the plain dict
    `PALACE_Capacitance_Simulation`'s `user_options` expects."""
    return {
        "mesh_refinement": conf.mesh_refinement,
        "dielectric_material": dielectric_material,
        "solns_to_save": conf.solns_to_save,
        "solver_order": conf.solver_order,
        "solver_tol": conf.solver_tol,
        "solver_maxits": conf.solver_maxits,
        "fillet_resolution": conf.fillet_resolution,
        "palace_dir": conf.palace_dir,
        "num_cpus": conf.num_cpus,
    }


def run_capacitance_sim(
    design,
    name: str,
    output_path: str,
    components: Optional[list[str]] = None,
    open_terminations: Optional[list[tuple[str, str]]] = None,
    sim_options: Optional[CapacitanceSimOptions] = None,
    terminations: Optional[dict[tuple[str, str], str]] = None,
    metallic_layer: int = 1,
    fine_mesh_components: Optional[list[FineMeshComponentOptions]] = None,
    fine_mesh_paths: Optional[list[MeshAlongPathOptions]] = None,
    amr_options: Optional[AMROptions] = None,
    show_conductor_indices: bool = True,
):
    """Build, mesh, and run a PALACE capacitance (electrostatic) simulation end to end.

    Guarantees cleanup of the resources a run acquires, same as `run_drivenmodal_sim`: kills any
    still-running local PALACE subprocess, and finalizes the gmsh session eagerly.

    Args:
        design: full `QDesign` to pull `components` from.
        name: simulation name (also the output subfolder name).
        output_path: `sim_parent_directory` for `PALACE_Capacitance_Simulation`.
        components: names of the components to keep; passed to `build_subdesign`. Defaults to
            every component in `design` if omitted.
        open_terminations: `(component_name, pin_name)` pairs, for dangling pins left by
            components in `components` that connected to a dropped component, that should be
            terminated as open. Convenience for the common case; merged into `terminations`.
        sim_options: PALACE solver/mesh options. Defaults to `CapacitanceSimOptions()`.
        terminations: passed straight through to `build_subdesign` for full control (e.g. to
            terminate a specific dangling pin as `'short'` instead of the `'open'` default).
        metallic_layer: design layer id to render as metal (`add_metallic`).
        fine_mesh_components: optional per-group mesh refinement (`fine_mesh_components`).
        fine_mesh_paths: optional per-path mesh refinement (`fine_mesh_along_path`).
        amr_options: optional adaptive mesh refinement (`enable_mesh_refinement`).
        show_conductor_indices: if True (default), calls `display_conductor_indices()` right
            after `prepare_simulation()` and logs a warning to check it before committing to a
            long run -- there's no port list here, so the only way to know which conductor
            index maps to which component is this plot.

    Returns:
        pandas.DataFrame: capacitance matrix in fF, rows/columns labelled by conductor name --
        see `PALACE_Capacitance_Simulation.get_capacitance_matrix`.

    Raises:
        ValueError: if the design's substrate material isn't 'silicon' or 'sapphire'.
    """
    t_start = time.perf_counter()
    sim_options = sim_options or CapacitanceSimOptions()

    subdesign, dielectric_material = build_and_validate_subdesign(
        design, name, components, open_terminations, terminations, "PALACE_Capacitance_Simulation")

    cap_sim = PALACE_Capacitance_Simulation(
        name=name,
        metal_design=subdesign,
        sim_parent_directory=output_path,
        mode=sim_options.mode,
        meshing=sim_options.meshing,
        user_options=_to_user_options(sim_options, dielectric_material),
        create_files=sim_options.create_files,
    )
    logger.info("Created PALACE_Capacitance_Simulation %r (mode=%s, meshing=%s) at %r.",
               name, sim_options.mode, sim_options.meshing, output_path)

    with palace_run_cleanup(cap_sim):
        cap_sim.add_metallic(metallic_layer)
        cap_sim.add_ground_plane()
        logger.info("Added metallic layer %d and ground plane.", metallic_layer)

        apply_fine_mesh(cap_sim, fine_mesh_components, fine_mesh_paths)
        apply_amr(cap_sim, amr_options)

        logger.info("Preparing simulation (meshing + config file)...")
        t_prepare = time.perf_counter()
        cap_sim.prepare_simulation()
        logger.info("Simulation prepared in %.1f s.", time.perf_counter() - t_prepare)

        if show_conductor_indices:
            cap_sim.display_conductor_indices()
            logger.warning("Check the conductor-index plot before trusting the capacitance "
                           "matrix labels -- there's no port list to cross-reference against here.")

        logger.info("Running PALACE (%s, %d CPU(s))...", sim_options.mode, sim_options.num_cpus)
        t_run = time.perf_counter()
        data = cap_sim.run()
        logger.info("Simulation finished in %.1f s (total %.1f s including setup).",
                   time.perf_counter() - t_run, time.perf_counter() - t_start)
        return data
