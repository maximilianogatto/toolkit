"""Eigenmode simulation of a `Cavity` with Palace.

With the ports loaded by their resistance R, each eigenmode is complex: Palace reports the
frequency, the loaded Q (Q = f / kappa) and, in port-Q.csv, the external coupling of every port
(kappa_ext[i] = f / Q_ext[i]). No port is excited.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pypalace.builder import Solver

from toolkit.components.cavity.cavity import Cavity
from .palace_model import PalaceModel, PalaceModelOptions


@dataclass(kw_only=True)
class EigensimPalaceOptions(PalaceModelOptions):
    """Options for an eigenmode simulation."""
    target_GHz: float           # search starts here: put it just below the mode you want
    num_modes: int = 1          # number of eigenmodes to find above target_GHz
    tol: float = 1e-8           # eigensolver tolerance
    save: int = 0               # number of mode field solutions written for ParaView
    solver_type: str = "Default"  # Palace eigensolver: "Default", "SLEPc" or "ARPACK"


class EigensimPalace(PalaceModel):
    problem_type = "Eigenmode"
    output_file = "eig.csv"

    def __init__(self, name: str, cavity: Cavity, output_dir: Path, options: EigensimPalaceOptions):
        super().__init__(name, cavity, output_dir, options)

    def _config_solver(self) -> dict:
        o = self.options
        return {"Simulation": Solver.Eigenmode(Target=o.target_GHz, Tol=o.tol, N=o.num_modes, Save=o.save,
                                               Type=o.solver_type),
                "Order": o.sol_order}

    # ---------- results ----------
    def frequency(self, mode: int = 1) -> float:
        """Eigenfrequency of `mode`, in Hz."""
        return self._require_run().get_frequency_eigenmode(mode)

    def kappa(self, mode: int = 1) -> float:
        """Total linewidth of `mode` (f / Q_loaded), in Hz."""
        return self._require_run().get_kappa_eigenmode(mode)

    def kappa_ext(self, mode: int = 1, ports: list[int] | None = None) -> float:
        """External linewidth of `mode` through `ports` (default: all ports), in Hz."""
        ports = ports or list(range(1, len(self.cavity.ports) + 1))
        return self._require_run().get_kappa_ext(mode, ports)

    def results(self) -> pd.DataFrame:
        """One row per mode: frequency, loaded Q, kappa and kappa_ext of every port (Hz)."""
        sim = self._require_run()
        rows = []
        for mode in range(1, self.options.num_modes + 1):
            f, kappa = sim.get_frequency_eigenmode(mode), sim.get_kappa_eigenmode(mode)
            row = {"mode": mode, "f_Hz": f, "Q": f / kappa if kappa else float("inf"), "kappa_Hz": kappa}
            for i in range(1, len(self.cavity.ports) + 1):
                row[f"kappa_ext{i}_Hz"] = sim.get_kappa_ext(mode, [i])
            rows.append(row)
        return pd.DataFrame(rows)
