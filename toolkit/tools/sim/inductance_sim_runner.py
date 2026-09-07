"""Inductance (magnetostatic) simulation runner for SQDMetal/PALACE.

Same shape as `drivenmodal_sim_runner.py` (see that file's module docstring for the gmsh/
subprocess cleanup rationale, shared via `_common.palace_run_cleanup`). Like the capacitance
runner, a magnetostatic solve has no frequency/sweep, but it does need at least one
current-source excitation (a U-clip current injection,
`create_current_source_with_Uclip_on_Route`/`..._on_Launcher`) -- that's this file's equivalent
of the driven-modal runner's `DrivenModalPortOptions`.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import shapely
from qiskit_metal.qlibrary.core.qroute import QRoute
from qiskit_metal.qlibrary.terminations.launchpad_wb_driven import LaunchpadWirebondDriven
from SQDMetal.PALACE.Inductance_Simulation import PALACE_Inductance_Simulation

from toolkit.tools.sim._common import (
    AMROptions,
    FineMeshComponentOptions,
    MeshAlongPathOptions,
    apply_amr,
    apply_fine_mesh,
    build_and_validate_subdesign,
    check_solver_convergence,
    palace_run_cleanup,
)

logger = logging.getLogger(__name__)

_CURRENT_SOURCE_TYPES = {"Uclip_on_Launcher", "Uclip_on_Route"}


@dataclass
class InductanceSimOptions:
    """PALACE solver/mesh options for an inductance (magnetostatic) simulation."""
    mesh_refinement: int = 0
    solns_to_save: int = -1
    solver_order: int = 2
    solver_tol: float = 1.0e-8
    solver_maxits: int = 5000
    fillet_resolution: int = 12
    palace_dir: str = ""                     # Path to the PALACE binary.
    num_cpus: int = 1
    meshing: str = "GMSH"                    # 'GMSH' or 'COMSOL'
    mode: str = "simPC"                      # 'simPC' or 'HPC'
    create_files: bool = True


@dataclass
class CurrentSourceOptions:
    """Spec for a single U-clip current-source excitation. `type` selects which SQDMetal
    constructor is used and which of the other fields apply.

    * 'Uclip_on_Launcher': uses component_name, thickness_side/back, separation_gap.
    * 'Uclip_on_Route': uses component_name, pin_name, thickness_side/back, separation_gap.
    """
    component_name: str
    type: str = "Uclip_on_Route"
    pin_name: str = "end"                    # Only used for 'Uclip_on_Route'.
    thickness_side: float = 20e-6
    thickness_back: float = 20e-6
    separation_gap: float = 20e-6

    def __post_init__(self):
        if self.type not in _CURRENT_SOURCE_TYPES:
            raise ValueError(f"Unsupported current source type: {self.type!r}. "
                             f"Supported: {sorted(_CURRENT_SOURCE_TYPES)}.")


def _to_user_options(conf: InductanceSimOptions, dielectric_material: str) -> dict:
    """Convert `InductanceSimOptions` + resolved substrate material to the plain dict
    `PALACE_Inductance_Simulation`'s `user_options` expects."""
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


def _terminal_label(source: CurrentSourceOptions) -> str:
    """Human-readable name for a terminal, used to label the inductance matrix."""
    if source.type == "Uclip_on_Route":
        return f"{source.component_name}.{source.pin_name}"
    return source.component_name


def _create_current_source(mag_sim: PALACE_Inductance_Simulation, subdesign, source: CurrentSourceOptions) -> None:
    if source.component_name not in subdesign.components:
        raise KeyError(f"Current source component {source.component_name!r} is not in the "
                       f"sub-design's kept components.")
    comp = subdesign.components[source.component_name]

    if source.type == "Uclip_on_Launcher":
        if not isinstance(comp, LaunchpadWirebondDriven):
            logger.warning("Current source %r is type 'Uclip_on_Launcher' but component is a "
                           "%s, not a LaunchpadWirebondDriven.", source.component_name, type(comp).__name__)
        mag_sim.create_current_source_with_Uclip_on_Launcher(
            source.component_name, thickness_side=source.thickness_side,
            thickness_back=source.thickness_back, separation_gap=source.separation_gap)
    else:  # 'Uclip_on_Route'
        if not isinstance(comp, QRoute):
            logger.warning("Current source %r is type 'Uclip_on_Route' but component is a %s, "
                           "not a QRoute.", source.component_name, type(comp).__name__)
        mag_sim.create_current_source_with_Uclip_on_Route(
            source.component_name, source.pin_name, thickness_side=source.thickness_side,
            thickness_back=source.thickness_back, separation_gap=source.separation_gap)

    logger.info("Created current source (%s) on %r.", source.type, source.component_name)


def run_inductance_sim(
    design,
    name: str,
    output_path: str,
    current_sources: list[CurrentSourceOptions],
    components: Optional[list[str]] = None,
    open_terminations: Optional[list[tuple[str, str]]] = None,
    sim_options: Optional[InductanceSimOptions] = None,
    terminations: Optional[dict[tuple[str, str], str]] = None,
    metallic_layer: int = 1,
    extra_layers: Optional[list[tuple[int, dict]]] = None,
    integration_areas: Optional[list[shapely.Polygon | shapely.geometry.MultiPolygon]] = None,
    fine_mesh_components: Optional[list[FineMeshComponentOptions]] = None,
    fine_mesh_paths: Optional[list[MeshAlongPathOptions]] = None,
    amr_options: Optional[AMROptions] = None,
    strict_convergence: bool = True,
):
    """Build, mesh, and run a PALACE inductance (magnetostatic) simulation end to end.

    Guarantees cleanup of the resources a run acquires, same as `run_drivenmodal_sim`: kills any
    still-running local PALACE subprocess, and finalizes the gmsh session eagerly.

    Args:
        design: full `QDesign` to pull `components` from.
        name: simulation name (also the output subfolder name).
        output_path: `sim_parent_directory` for `PALACE_Inductance_Simulation`.
        current_sources: one `CurrentSourceOptions` per U-clip current injection (e.g. a flux
            line). At least one is required -- there's no default excitation to fall back on.
        components: names of the components to keep; passed to `build_subdesign`. Defaults to
            every component in `design` if omitted.
        open_terminations: `(component_name, pin_name)` pairs, for dangling pins left by
            components in `components` that connected to a dropped component, that should be
            terminated as open. Convenience for the common case; merged into `terminations`.
        sim_options: PALACE solver/mesh options. Defaults to `InductanceSimOptions()`.
        terminations: passed straight through to `build_subdesign` for full control.
        metallic_layer: primary design layer id to render as metal (`add_metallic`).
        extra_layers: additional `(layer_id, kwargs)` pairs added via `add_metallic(layer_id,
            **kwargs)`, e.g. for a Dolan-bridge junction layer: `[(2, {'evap_mode': None})]`.
        integration_areas: optional Shapely polygons added via `add_integration_area` -- the
            surfaces flux is integrated over. If omitted, `retrieve_data` falls back to whatever
            `PALACE_Inductance_Simulation` returns by default.
        fine_mesh_components: optional per-group mesh refinement (`fine_mesh_components`).
        fine_mesh_paths: optional per-path mesh refinement (`fine_mesh_along_path`).
        amr_options: optional adaptive mesh refinement (`enable_mesh_refinement`).
        strict_convergence: raise if any of PALACE's linear solves failed to converge. PALACE
            only *warns* about that and still writes plausible-looking output files, so leaving
            this on is what keeps a stalled solve from being read as a result.

    Returns:
        A dict with the full magnetostatic result:

        * `'inductance_matrix'`: (n_terminals, n_terminals) array in HENRIES -- PALACE's
          `terminal-M.csv`. Diagonal = self-inductance of each terminal's current loop,
          off-diagonal = mutual inductance between loops. THIS is the coupling number.
        * `'terminal_names'`: labels in the same order as the matrix's rows/columns, i.e. the
          order `current_sources` was given in.
        * `'terminal_currents'`: the current (A) PALACE actually imposed on each terminal.
          Not 1 A -- the matrix is already normalised by it, this is only for reference.
        * `'surface_flux'`: (n_terminals, n_areas) magnetic flux in Wb through each
          `integration_areas` polygon, per terminal excitation.
        * `'flux_per_amp'`: `surface_flux / terminal_currents`, i.e. Wb/A. `None` if no
          integration area was given.
        * `'output_dir'`: where PALACE wrote its files (out.log, CSVs, paraview/).

    Raises:
        ValueError: if the design's substrate material isn't 'silicon' or 'sapphire', if
            `current_sources` is empty, or if a `CurrentSourceOptions` entry is invalid.
        RuntimeError: if `strict_convergence` and a linear solve did not converge.
    """
    t_start = time.perf_counter()
    sim_options = sim_options or InductanceSimOptions()

    if not current_sources:
        raise ValueError("run_inductance_sim needs at least one CurrentSourceOptions in "
                         "current_sources -- there's no default current excitation.")

    subdesign, dielectric_material = build_and_validate_subdesign(
        design, name, components, open_terminations, terminations, "PALACE_Inductance_Simulation")

    mag_sim = PALACE_Inductance_Simulation(
        name=name,
        metal_design=subdesign,
        sim_parent_directory=output_path,
        mode=sim_options.mode,
        meshing=sim_options.meshing,
        user_options=_to_user_options(sim_options, dielectric_material),
        create_files=sim_options.create_files,
    )
    logger.info("Created PALACE_Inductance_Simulation %r (mode=%s, meshing=%s) at %r.",
               name, sim_options.mode, sim_options.meshing, output_path)

    with palace_run_cleanup(mag_sim):
        mag_sim.add_metallic(metallic_layer)
        for layer_id, kwargs in extra_layers or []:
            mag_sim.add_metallic(layer_id, **kwargs)
        mag_sim.add_ground_plane()
        logger.info("Added metallic layer %d (+%d extra) and ground plane.",
                   metallic_layer, len(extra_layers or []))

        for source in current_sources:
            _create_current_source(mag_sim, subdesign, source)

        for area in integration_areas or []:
            mag_sim.add_integration_area(area)
        if integration_areas:
            logger.info("Added %d flux integration area(s).", len(integration_areas))

        apply_fine_mesh(mag_sim, fine_mesh_components, fine_mesh_paths)
        apply_amr(mag_sim, amr_options)

        logger.info("Preparing simulation (meshing + config file)...")
        t_prepare = time.perf_counter()
        mag_sim.prepare_simulation()
        logger.info("Simulation prepared in %.1f s.", time.perf_counter() - t_prepare)

        logger.info("Running PALACE (%s, %d CPU(s))...", sim_options.mode, sim_options.num_cpus)
        t_run = time.perf_counter()
        flux_per_amp = mag_sim.run()
        logger.info("Simulation finished in %.1f s (total %.1f s including setup).",
                   time.perf_counter() - t_run, time.perf_counter() - t_start)

        check_solver_convergence(mag_sim, strict=strict_convergence)

        # `mag_sim.run()` only hands back the flux-per-amp reduction (and only when integration
        # areas were given); the inductance matrix itself is left in the output files, so read
        # it back rather than making the caller do it.
        raw = mag_sim.retrieve_magnetostatic_data()
        return {
            "inductance_matrix": raw["inductance_matrix"],
            "terminal_names": [_terminal_label(s) for s in current_sources],
            "terminal_currents": raw["terminal_I"].ravel(),
            "surface_flux": raw["surface_Flux"],
            "flux_per_amp": flux_per_amp if integration_areas else None,
            "output_dir": mag_sim._output_data_dir,
        }
