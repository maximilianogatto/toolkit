"""Pre-flight check for a magnetostatic (inductance) simulation: does every current terminal
cut a galvanically CLOSED loop?

PALACE's magnetostatic solver drops the displacement current, so Ampere's law only has a
solution for a solenoidal current density: taking the divergence of curl H = J gives div J = 0.
Current cannot pile up anywhere. A terminal whose two sides do not belong to the same connected
piece of metal is a dead end -- the linear system is inconsistent, CG stalls or diverges, and
PALACE *still* writes a full terminal-M.csv of meaningless numbers.

The test implemented here is purely topological and takes a couple of seconds, versus minutes
for a solve that was never going to work:

    take the metal, add the U-clips, remove the terminal patches, count the connected pieces.

    1 piece  -> every cut has a way around through the metal  -> closed loops   -> solvable
    2+ pieces -> some conductor only reaches the rest through its own cut -> dead end

Run it with the same `components` / `terminations` / `current_sources` you are about to hand to
`run_inductance_sim`.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import shapely
from shapely.ops import unary_union

from SQDMetal.Utilities.QiskitShapelyRenderer import QiskitShapelyRenderer
from SQDMetal.Utilities.QUtilities import QUtilities
from SQDMetal.Utilities.ShapelyEx import ShapelyEx

from toolkit.tools.sim._common import build_and_validate_subdesign
from toolkit.tools.sim.inductance_sim_runner import CurrentSourceOptions

logger = logging.getLogger(__name__)

_UCLIP_ARGS = ("thickness_side", "thickness_back", "separation_gap")


@dataclass
class CurrentLoopReport:
    """Outcome of `check_current_loops`. Geometry is in the design's own units (mm normally)."""
    pieces: list = field(repr=False)          # connected metal pieces, largest first
    ports: list = field(repr=False)           # one terminal patch polygon per current source
    clips: list = field(repr=False)           # one U-clip polygon per current source
    terminal_names: list[str] = field(default_factory=list)
    subdesign: object = field(default=None, repr=False)

    @property
    def n_pieces(self) -> int:
        return len(self.pieces)

    @property
    def ok(self) -> bool:
        return self.n_pieces == 1

    def summary(self) -> str:
        lines = [f"connected pieces of metal once the cuts are removed: {self.n_pieces}"]
        for i, g in enumerate(self.pieces):
            x0, y0, x1, y1 = g.bounds
            lines.append(f"  piece {i}: area {g.area*1e6:11.0f} um^2   "
                         f"bbox x {x0:.2f}..{x1:.2f}  y {y0:.2f}..{y1:.2f} mm")
        lines.append("  => " + ("CLOSED LOOPS, solvable" if self.ok else
                                "DEAD END: a conductor only reaches the rest through its own cut"))
        return "\n".join(lines)


def _metal_polygon(subdesign):
    """The metal on the chip as one shapely geometry, in design units.

    Qiskit-Metal stores the layout as polygons that are either *subtracted* from the ground
    plane (the CPW gaps) or *added* on top of it, so the metal is chip - subtracted + added.
    """
    gsdf = QiskitShapelyRenderer(design=subdesign, canvas=None, logger=None).get_net_coordinates()
    size = subdesign.chips["main"]["size"]
    to_units = 1.0 / QUtilities.get_units(subdesign)      # metres -> design units
    val = lambda key: QUtilities.parse_value_length(size[key]) * to_units
    cx, cy, sx, sy = val("center_x"), val("center_y"), val("size_x"), val("size_y")
    chip = shapely.box(cx - sx / 2, cy - sy / 2, cx + sx / 2, cy + sy / 2)

    subtracted = list(gsdf.loc[gsdf["subtract"]]["geometry"])
    added = list(gsdf.loc[~gsdf["subtract"]]["geometry"])
    metal = chip.difference(unary_union(subtracted)) if subtracted else chip
    return metal.union(unary_union(added)) if added else metal


def _uclip_and_port(subdesign, source: CurrentSourceOptions):
    """The U-clip polygon and the SurfaceCurrent patch for one current source, in design units.

    Uses SQDMetal's own helpers, so what is drawn here is what the mesher will build -- the
    U-clip metal bridges the two flanking ground planes just beyond the pin, and the terminal
    patch spans the `separation_gap` between the trace tip and the clip's crossbar.
    """
    to_units = 1.0 / QUtilities.get_units(subdesign)
    kw = {name: getattr(source, name) for name in _UCLIP_ARGS}

    if source.type == "Uclip_on_Launcher":
        clip = QUtilities.get_RFport_CPW_groundU_Launcher_inplane(
            subdesign, source.component_name, *kw.values(), unit_conv_extra=to_units)
        origin, direction, width, _ = QUtilities._get_LauncherWB_params(
            subdesign, source.component_name, unit_conv_extra=to_units)
    else:  # 'Uclip_on_Route'
        clip = QUtilities.get_RFport_CPW_groundU_Route_inplane(
            subdesign, source.component_name, source.pin_name, *kw.values(),
            unit_conv_extra=to_units)
        origin, direction, width, _ = QUtilities._get_Route_params(
            subdesign, source.component_name, source.pin_name, unit_conv_extra=to_units)

    gap = source.separation_gap * to_units
    port = shapely.Polygon(
        ShapelyEx.rectangle_from_line(origin - direction * gap, origin, width, False))
    return shapely.Polygon(clip), port


def check_current_loops(
    design,
    current_sources: list[CurrentSourceOptions],
    components: Optional[list[str]] = None,
    open_terminations: Optional[list[tuple[str, str]]] = None,
    terminations: Optional[dict[tuple[str, str], str]] = None,
    metallic_layer: int = 1,
    strict: bool = False,
) -> CurrentLoopReport:
    """Count the connected pieces of metal left once the terminal patches are cut out.

    Takes the same `components` / `open_terminations` / `terminations` / `current_sources` as
    `run_inductance_sim`, and builds the sub-design the same way, so the geometry tested is the
    geometry that would be simulated.

    Args:
        design: the full `QDesign`.
        current_sources: the `CurrentSourceOptions` list destined for `run_inductance_sim`.
        components: names of the components to keep. Defaults to all of them.
        open_terminations: `(component, pin)` pairs to terminate open; merged into
            `terminations`.
        terminations: passed through to `build_subdesign`.
        metallic_layer: kept for signature symmetry with `run_inductance_sim`; the metal is read
            from the rendered layout, which already includes it.
        strict: raise instead of returning a report when the loops are not closed.

    Returns:
        A `CurrentLoopReport`. Check `.ok` / `.n_pieces`, print `.summary()`, or hand it to
        `plot_current_loops` to see it.

    Raises:
        ValueError: if `current_sources` is empty.
        RuntimeError: if `strict` and more than one piece of metal remains.
    """
    if not current_sources:
        raise ValueError("check_current_loops needs at least one CurrentSourceOptions.")

    subdesign, _ = build_and_validate_subdesign(
        design, "current-loop check", components, open_terminations, terminations,
        "check_current_loops")

    clips, ports = [], []
    for source in current_sources:
        clip, port = _uclip_and_port(subdesign, source)
        clips.append(clip)
        ports.append(port)

    cut = unary_union([_metal_polygon(subdesign)] + clips).difference(unary_union(ports))
    pieces = sorted(getattr(cut, "geoms", [cut]), key=lambda g: -g.area)

    report = CurrentLoopReport(
        pieces=pieces, ports=ports, clips=clips, subdesign=subdesign,
        terminal_names=[s.component_name if s.type == "Uclip_on_Launcher"
                        else f"{s.component_name}.{s.pin_name}" for s in current_sources])

    if report.ok:
        logger.info("Current-loop check passed: one connected piece of metal.")
    else:
        msg = ("Current-loop check FAILED. " + report.summary() + "\n"
               "A current terminal must be a cut in a galvanically closed loop: both sides of "
               "the patch must belong to the same connected metal. Short the far end of the "
               "offending conductor (drop its launchpad and terminate that pin 'short').")
        if strict:
            raise RuntimeError(msg)
        logger.error(msg)

    return report


def plot_current_loops(report: CurrentLoopReport, ax=None, zoom: Optional[int] = None):
    """Draw the metal coloured by connected piece, with the terminal patches in red.

    One colour means one piece, i.e. closed loops. Two colours means the second-coloured
    conductor is an island reachable only through its own cut.

    Args:
        report: from `check_current_loops`.
        ax: optional matplotlib axes to draw into.
        zoom: index into `report.ports` to frame that terminal instead of the whole chip.

    Returns:
        The matplotlib axes.
    """
    import matplotlib.pyplot as plt

    colours = ["#b8bec6", "#e8a33d", "#7fb069", "#8d7bbd"]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.5))

    for i, piece in enumerate(report.pieces):
        ax.fill(*piece.exterior.xy, fc=colours[i % len(colours)], ec="#6b737d", lw=0.4, zorder=1 + i)
        for hole in piece.interiors:                      # CPW gaps and cut-outs
            ax.fill(*hole.xy, fc="white", ec="none", zorder=1.5 + i)
    for port in report.ports:
        ax.fill(*port.exterior.xy, fc="#d62728", ec="#d62728", lw=2.0, zorder=5)

    if zoom is not None:
        x0, y0, x1, y1 = report.ports[zoom].buffer(0.08).bounds
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_title(f"terminal {zoom + 1} — {report.terminal_names[zoom]}", fontsize=10)
    else:
        verdict = ("1 piece: closed loops" if report.ok
                   else f"{report.n_pieces} pieces: DEAD END")
        ax.set_title(f"grey/colour = connected metal, red = terminal cut  ({verdict})", fontsize=10)

    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    return ax
