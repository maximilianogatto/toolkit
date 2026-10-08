"""Eigenfrequency simulation of a `Cavity` with COMSOL.

The counterpart of `EigensimPalace`: same cavity, same ports, same materials, and `results()`
returns the same columns so the two can be stacked with `simModel.compare`.

COMSOL returns a complex eigenfrequency, exactly like Palace, so

    f0 = Re(f),    Q = Re(f) / (2 Im(f)),    kappa = f0 / Q = 2 Im(f)

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .comsol_model import ComsolModel, ComsolModelOptions


@dataclass(kw_only=True)
class EigensimComsolOptions(ComsolModelOptions):
    """Options for an eigenfrequency study."""
    target_GHz: float           # search around here: put it just below the mode you want
    num_modes: int = 1          # number of eigenfrequencies to return


class EigensimComsol(ComsolModel):
    study_type = "Eigenfrequency"

    def _build_study(self, java) -> None:
        o = self.options
        std = java.study().create(self._STUDY)
        step = std.create("eig", "Eigenfrequency")
        step.set("neigs", str(o.num_modes))
        step.set("shift", f"{o.target_GHz}[GHz]")       # TODO(verify) "shift" vs "eigunit"/"shiftactive"
        step.set("eigunit", "GHz")

    # ---------- results ----------
    def _eigenfrequencies(self) -> np.ndarray:
        """Complex eigenfrequencies in Hz, one per mode."""
        return np.atleast_1d(np.asarray(self.evaluate("freq"), dtype=complex))

    def frequency(self, mode: int = 1) -> float:
        """Eigenfrequency of `mode`, in Hz."""
        return float(self._eigenfrequencies()[mode - 1].real)

    def kappa(self, mode: int = 1) -> float:
        """Total linewidth of `mode` (f / Q_loaded), in Hz."""
        return float(2 * self._eigenfrequencies()[mode - 1].imag)

    def results(self) -> pd.DataFrame:
        """One row per mode: frequency, loaded Q and kappa (Hz).

        `kappa_ext` per port is left empty: Palace writes it to port-Q.csv, COMSOL does not. To
        get it here you would integrate the power leaving each lumped port, or run the driven
        study and fit the resonance.
        """
        freqs = self._eigenfrequencies()[: self.options.num_modes]
        rows = []
        for mode, f in enumerate(freqs, start=1):
            f0, kappa = float(f.real), float(2 * f.imag)
            row = {"mode": mode, "f_Hz": f0, "Q": f0 / kappa if kappa else float("inf"),
                   "kappa_Hz": kappa}
            for i in range(1, len(self.cavity.ports) + 1):
                row[f"kappa_ext{i}_Hz"] = float("nan")
            rows.append(row)
        return pd.DataFrame(rows)
