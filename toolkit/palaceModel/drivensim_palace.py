"""Driven (frequency sweep) simulation of a `Cavity` with Palace.

Ports with `excite=True` launch the signal; the others are matched loads of resistance R. Palace
writes the S-parameters to port-S.csv, S[i][j] being the wave arriving at port i when port j is
excited.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pypalace.builder import Solver

from toolkit.components.cavity.cavity import Cavity
from toolkit.components.ports import CoaxPort
from .palace_model import PalaceModel, PalaceModelOptions


@dataclass(kw_only=True)
class DrivensimPalaceOptions(PalaceModelOptions):
    """Options for a driven frequency sweep."""
    freq_min_GHz: float
    freq_max_GHz: float
    freq_step_GHz: float                # put several points inside the resonance linewidth
    save_step: int = 0                  # write fields every N frequencies (0: never)
    adaptive_tol: float | None = None   # None: uniform sweep, safe for sharp resonances;
                                        # e.g. 1e-3: adaptive (PROM), faster but can miss a narrow peak


class DrivensimPalace(PalaceModel):
    problem_type = "Driven"
    output_file = "port-S.csv"

    def __init__(self, name: str, cavity: Cavity, output_dir: Path, options: DrivensimPalaceOptions):
        super().__init__(name, cavity, output_dir, options)

    def _excitation(self, index: int, port: CoaxPort) -> int | None:
        return index if port.excite else None

    def _config_boundaries(self, groups: dict[str, int]) -> dict:
        if not any(port.excite for port in self.cavity.ports):
            raise ValueError("A driven simulation needs at least one port with excite=True.")
        return super()._config_boundaries(groups)

    def _config_solver(self) -> dict:
        o = self.options
        return {"Simulation": Solver.Driven(MinFreq=o.freq_min_GHz, MaxFreq=o.freq_max_GHz,
                                            FreqStep=o.freq_step_GHz, SaveStep=o.save_step,
                                            AdaptiveTol=o.adaptive_tol),
                "Order": o.sol_order}

    # ---------- results ----------
    def s_parameter(self, i: int, j: int) -> pd.DataFrame:
        """S[i][j] versus frequency: columns frequency_GHz, |S[i][j]| (dB), arg(S[i][j]) (deg)."""
        return self._require_run().get_Sij(i, j)

    def results(self) -> pd.DataFrame:
        """Every S[i][j] in one tidy table: f_GHz, i, j, mag_dB, phase_deg.

        Same shape for any backend, so `compare()` can stack a Palace and a COMSOL sweep.
        """
        n = len(self.cavity.ports)
        excited = [j for j, port in enumerate(self.cavity.ports, start=1) if port.excite]
        rows = []
        for j in excited:
            for i in range(1, n + 1):
                df = self.s_parameter(i, j)
                cols = {str(c).strip(): c for c in df.columns}
                f = next(c for name, c in cols.items() if name.startswith("f "))
                mag = next(c for name, c in cols.items() if name.startswith("|S"))
                arg = next(c for name, c in cols.items() if name.startswith("arg"))
                rows.append(pd.DataFrame({"f_GHz": df[f], "i": i, "j": j,
                                          "mag_dB": df[mag], "phase_deg": df[arg]}))
        return pd.concat(rows, ignore_index=True)
