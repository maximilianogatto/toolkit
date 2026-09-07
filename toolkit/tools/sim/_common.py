"""Shared building blocks for the PALACE sim runners (`drivenmodal_sim_runner.py`,
`eigenmode_sim_runner.py`, `capacitance_sim_runner.py`, `inductance_sim_runner.py`).

Only the parts with zero simulation-specific logic live here: mesh-refinement option dataclasses
and their application, substrate-material validation, sub-design construction, and PALACE
subprocess/gmsh cleanup. Excitation (`EigenmodePortOptions`/`DrivenModalPortOptions`/
`CurrentSourceOptions` and their `_create_*` functions) stays local to each runner -it already
differs between them (e.g. driven ports have `excite`, eigenmode ports don't) and is likely to
keep diverging per simulation type.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import gmsh

from toolkit.tools.sim.sim_subdesign import build_subdesign

logger = logging.getLogger(__name__)

SUPPORTED_DIELECTRICS = {"silicon", "sapphire"}


# PALACE prints one of these per linear solve that ran out of iterations, e.g.
#   Linear solver did not converge, norm(Ax-b)/norm(b) = 2.456e+01 (norm(b) = 3.639e-02)!
# It is only a *warning* on stdout: the run still exits 0 and still writes terminal-M.csv,
# domain-E.csv, ... filled with whatever the stalled Krylov iterate happened to be. Nothing
# downstream notices, so the numbers look perfectly plausible while being meaningless. Hence
# this check, called by the runners right after `sim.run()`.
_NOT_CONVERGED_RE = re.compile(
    r"Linear solver did not converge, norm\(Ax-b\)/norm\(b\) = ([0-9.eE+-]+)")


def check_solver_convergence(sim, strict: bool = True) -> list[float]:
    """Scan the PALACE log of the run just finished for non-converged linear solves.

    Args:
        sim: the `PALACE_Model_*` object that was just run (its `_output_data_dir` is where
            PALACE wrote `out.log`).
        strict: raise on a non-converged solve. `False` only logs an error -- use it when you
            deliberately want to inspect a bad run's output files.

    Returns:
        The relative residual `norm(Ax-b)/norm(b)` of every solve that failed, in the order
        PALACE ran them. Empty if everything converged.

    Raises:
        RuntimeError: if `strict` and at least one solve did not converge.
    """
    log_path = os.path.join(getattr(sim, "_output_data_dir", ""), "out.log")
    if not os.path.isfile(log_path):
        logger.warning("No PALACE log at %r -- skipping the convergence check.", log_path)
        return []

    with open(log_path, "r", errors="replace") as f:
        residuals = [float(m.group(1)) for m in _NOT_CONVERGED_RE.finditer(f.read())]

    if not residuals:
        logger.info("All PALACE linear solves converged.")
        return []

    msg = (f"{len(residuals)} PALACE linear solve(s) did NOT converge "
           f"(relative residuals: {', '.join(f'{r:.3e}' for r in residuals)}). "
           f"The output files still exist but the numbers in them are meaningless. "
           f"Raise `solver_maxits`, refine the mesh, and check that every current-source "
           f"terminal cuts a galvanically CLOSED loop. Log: {log_path}")
    if strict:
        raise RuntimeError(msg)
    logger.error(msg)
    return residuals


@dataclass
class FineMeshComponentOptions:
    """Mesh refinement applied uniformly over a group of components (`fine_mesh_components`)."""
    qObjNames: list[str]
    min_size: float = 8e-6
    max_size: float = 100e-6
    taper_dist_min: float = 10e-6
    metals_only: bool = False


@dataclass
class MeshAlongPathOptions:
    """Mesh refinement following the centreline of a single CPW path (`fine_mesh_along_path`)."""
    qObjName: str
    dist_resolution: float = 10e-6
    min_size: float = 6e-6
    max_size: float = 150e-6
    taper_dist_min: float = 10e-6


@dataclass
class AMROptions:
    """Adaptive mesh refinement settings (`enable_mesh_refinement`)."""
    num_iterations: int = 5
    max_DoFs: int = 0
    tolerance: float = 1e-2
    Dorfler_marking_fraction: float = 0.7
    save_iterations_data: bool = True
    save_iterations_mesh: bool = False
    nonconformal: bool = True


def build_and_validate_subdesign(
    design,
    name: str,
    components: Optional[list[str]],
    open_terminations: Optional[list[tuple[str, str]]],
    terminations: Optional[dict[tuple[str, str], str]],
    sim_class_name: str,
):
    """Build the sub-design and check its substrate material is one PALACE supports.

    `components=None` keeps every component in `design`. `open_terminations` entries are merged
    into `terminations` as `'open'` -an explicit `terminations` entry for the same
    `(component, pin)` wins, so you can still force a specific dangling pin to `'short'` instead.

    Returns:
        (subdesign, dielectric_material)

    Raises:
        ValueError: if the substrate material isn't 'silicon' or 'sapphire'.
    """
    print(f"Components in the design: {list(design.components.keys())}")
    if components is None:
        components = list(design.components.keys())

    merged_terminations = dict(terminations or {})
    for comp_name, pin_name in (open_terminations or []):
        merged_terminations.setdefault((comp_name, pin_name), "open")

    logger.info("Building sub-design %r from %d component(s).", name, len(components))
    subdesign = build_subdesign(design, keep=components, terminations=merged_terminations or None,
                                 default_termination="open", verbose=True)

    dielectric_material = subdesign.chips.main.material
    if dielectric_material not in SUPPORTED_DIELECTRICS:
        raise ValueError(
            f"Unsupported substrate material: {dielectric_material!r}. {sim_class_name} "
            f"only supports {sorted(SUPPORTED_DIELECTRICS)}."
        )
    logger.info("Detected substrate material: %s.", dielectric_material)

    return subdesign, dielectric_material


def apply_fine_mesh(
    sim,
    fine_mesh_components: Optional[list[FineMeshComponentOptions]],
    fine_mesh_paths: Optional[list[MeshAlongPathOptions]],
) -> None:
    """Apply `fine_mesh_components`/`fine_mesh_along_path` calls to `sim`."""
    for mesh_option in fine_mesh_components or []:
        sim.fine_mesh_components(
            mesh_option.qObjNames, min_size=mesh_option.min_size, max_size=mesh_option.max_size,
            taper_dist_min=mesh_option.taper_dist_min, metals_only=mesh_option.metals_only)
    for path_option in fine_mesh_paths or []:
        sim.fine_mesh_along_path(
            qObjName=path_option.qObjName, dist_resolution=path_option.dist_resolution,
            min_size=path_option.min_size, max_size=path_option.max_size,
            taper_dist_min=path_option.taper_dist_min)
    if fine_mesh_components or fine_mesh_paths:
        logger.info("Added %d fine-mesh component group(s) and %d fine-mesh path(s).",
                   len(fine_mesh_components or []), len(fine_mesh_paths or []))


def apply_amr(sim, amr_options: Optional[AMROptions]) -> None:
    """Apply `enable_mesh_refinement` to `sim`, if `amr_options` was supplied."""
    if amr_options is not None:
        sim.enable_mesh_refinement(
            num_iterations=amr_options.num_iterations, max_DoFs=amr_options.max_DoFs,
            tolerance=amr_options.tolerance, Dorfler_marking_fraction=amr_options.Dorfler_marking_fraction,
            save_iterations_data=amr_options.save_iterations_data,
            save_iterations_mesh=amr_options.save_iterations_mesh, nonconformal=amr_options.nonconformal)
        logger.info("Enabled adaptive mesh refinement (%d iterations max).", amr_options.num_iterations)


@contextmanager
def palace_run_cleanup(sim):
    """Guarantees cleanup of the resources a run acquires, even if the body raises partway
    through: kills any still-running local PALACE subprocess, and finalizes the gmsh session
    eagerly (it would otherwise sit initialized until the *next* simulation resets it, see
    `GMSH_Geometry_Builder.__init__`)."""
    try:
        yield sim
    finally:
        proc = getattr(sim, "cur_process", None)
        if proc is not None and proc.poll() is None:
            logger.warning("Killing still-running PALACE subprocess during cleanup.")
            proc.kill()
        if gmsh.isInitialized():
            gmsh.finalize()
            logger.debug("Finalized gmsh session.")
