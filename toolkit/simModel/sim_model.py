"""Backend-independent base for a simulation of a `Cavity`.

A `Cavity` and its `Port`s describe geometry and materials; they say nothing about which solver
runs them. `SimModel` holds that common part -- where the files go, how the geometry is handed to
the solver, and the result contract every backend honours -- so `PalaceModel` and `ComsolModel`
differ only in how they translate the same cavity into their own input.

The contract a backend implements:

    run()       build the input, solve, and keep whatever handle reads the results
    results()   one DataFrame, with the same columns from every backend

That last point is the reason this class exists: `compare(eigen_palace, eigen_comsol)` puts both
solvers in one table, and `assert_same_discretisation` refuses to compare two models that were not
discretised the same way (a silent mismatch of `sol_order` once moved a driven result by 7 MHz).

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from toolkit.components.cavity.cavity import Cavity


@dataclass(kw_only=True)
class SimModelOptions:
    """Options every backend understands."""
    num_cpus: int = 1       # MPI processes (Palace) or cores (COMSOL)
    sol_order: int = 1      # finite-element order: 1 is fast, 2 is more accurate
    verbose: int = 2        # solver log verbosity


class SimModel(ABC):
    """One simulation of one cavity, with one solver."""

    backend: str = ""       # "palace", "comsol": labels results, nothing else

    def __init__(self, name: str, cavity: Cavity, output_dir: Path,
                 options: SimModelOptions | None = None):
        self.name = name
        self.cavity = cavity
        self.output_dir = Path(output_dir).resolve()   # absolute: solvers read results from here
        self.options = options or SimModelOptions()
        self.groups: dict[str, int] | None = None
        self.sim = None                                # backend handle, set by run()

    # ---------- paths ----------
    @property
    def mesh_file(self) -> Path:
        return self.output_dir / f"{self.name}.msh"

    @property
    def step_file(self) -> Path:
        return self.output_dir / f"{self.name}.step"

    @property
    def tags_file(self) -> Path:
        return self.output_dir / f"{self.name}.tags.json"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    # ---------- geometry ----------
    def _make_mesh(self) -> dict[str, int]:
        """Mesh the cavity; the physical groups name every volume and face the solver needs."""
        self.groups = self.cavity.build_mesh(self.name, self.mesh_file)
        return self.groups

    def _export_geometry(self) -> dict[str, int]:
        """Write the geometry as STEP plus a sidecar naming every group, for solvers that mesh
        the CAD themselves."""
        self.groups = self.cavity.export_step(self.name, self.step_file, self.tags_file)
        return self.groups

    # ---------- contract ----------
    @abstractmethod
    def run(self):
        """Build the solver input, run it, and store the handle that reads results."""

    @abstractmethod
    def results(self) -> pd.DataFrame:
        """The simulation's results, with the same columns for every backend."""

    def _require_run(self):
        if self.sim is None:
            raise RuntimeError(f"{self.name}: call run() before reading results.")
        return self.sim


def assert_same_discretisation(a: SimModel, b: SimModel) -> None:
    """Raise unless two models were discretised the same way, so their results can be compared."""
    checks = {"sol_order":  lambda m: m.options.sol_order,
              "lc":         lambda m: m.cavity.options.lc,
              "mesh_order": lambda m: m.cavity.options.mesh_order}
    for what, get in checks.items():
        if get(a) != get(b):
            raise ValueError(f"{what} differs: {a.name}={get(a)!r} vs {b.name}={get(b)!r}. "
                             "The results are not comparable.")


def compare(*models: SimModel) -> pd.DataFrame:
    """Stack the results of several models, labelled by backend, for a side-by-side table."""
    frames = []
    for model in models:
        df = model.results().copy()
        df.insert(0, "model", model.name)
        df.insert(0, "backend", model.backend or type(model).__name__)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
