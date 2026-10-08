"""Driven (frequency sweep) simulation of a `Cavity` with COMSOL.

The counterpart of `DrivensimPalace`: ports with `excite=True` launch the signal, the others are
matched loads, and `results()` returns the same tidy table (f_GHz, i, j, mag_dB, phase_deg).

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .comsol_model import ComsolModel, ComsolModelOptions


@dataclass(kw_only=True)
class DrivensimComsolOptions(ComsolModelOptions):
    """Options for a driven frequency sweep."""
    freq_min_GHz: float
    freq_max_GHz: float
    freq_step_GHz: float        # put several points inside the resonance linewidth


class DrivensimComsol(ComsolModel):
    study_type = "Frequency Domain"

    def _set_excitation(self, node, index: int, port) -> None:
        node.set("PortExcitation", "on" if port.excite else "off")   # TODO(verify) property name

    def _build_study(self, java) -> None:
        o = self.options
        if not any(port.excite for port in self.cavity.ports):
            raise ValueError("A driven simulation needs at least one port with excite=True.")
        std = java.study().create(self._STUDY)
        step = std.create("freq", "Frequency")
        step.set("plist", f"range({o.freq_min_GHz}[GHz],{o.freq_step_GHz}[GHz],{o.freq_max_GHz}[GHz])")

    # ---------- results ----------
    def s_parameter(self, i: int, j: int) -> pd.DataFrame:
        """S[i][j] versus frequency: columns f_GHz, mag_dB, phase_deg."""
        f = np.atleast_1d(np.asarray(self.evaluate("freq"), dtype=float))
        mag = np.atleast_1d(np.asarray(self.evaluate(f"emw.S{i}{j}dB"), dtype=float))
        phase = np.atleast_1d(np.asarray(self.evaluate(f"arg(emw.S{i}{j})*180/pi"), dtype=float))
        return pd.DataFrame({"f_GHz": f / 1e9, "mag_dB": mag, "phase_deg": phase})

    def results(self) -> pd.DataFrame:
        """Every S[i][j] in one tidy table: f_GHz, i, j, mag_dB, phase_deg."""
        n = len(self.cavity.ports)
        excited = [j for j, port in enumerate(self.cavity.ports, start=1) if port.excite]
        rows = []
        for j in excited:
            for i in range(1, n + 1):
                df = self.s_parameter(i, j)
                df.insert(1, "i", i)
                df.insert(2, "j", j)
                rows.append(df)
        return pd.concat(rows, ignore_index=True)
