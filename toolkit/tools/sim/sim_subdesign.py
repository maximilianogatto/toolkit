
"""Sub-design construction for Quantum/Qiskit Metal.

Lets you trim a full design down to a subset of components so you can
simulate (via SQDMetal/Palace) only part of a chip -e.g. the bare mode of a
resonator, or a single qubit on a chip with several- without needing to
simulate the entire design.

SQDMetal renders the `design` passed to it as-is (its `add_metallic(layer_id)`
adds the whole layer; it doesn't support selecting a subset of components like
Ansys/HFSS's `render_design(selection=..., open_pins=...)`). To achieve the
equivalent here, a NEW design is built containing only the requested
components, and any pins left dangling -because they connected to a discarded
component- are terminated with `ShortToGround` or `OpenToGround`, replicating
the `open_pins` logic of Ansys.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from copy import deepcopy
from math import atan2, degrees
from typing import Optional

from qiskit_metal import Dict, designs
from qiskit_metal.designs.design_base import QDesign
from qiskit_metal.qlibrary.core.qroute import QRoute
from qiskit_metal.qlibrary.terminations.open_to_ground import OpenToGround
from qiskit_metal.qlibrary.terminations.short_to_ground import ShortToGround
from qiskit_metal.qlibrary.tlines.straight_path import RouteStraight

# Each available termination: qiskit_metal class and the name of the pin it exposes.
_TERMINATION_CLASSES = {
    "short": (ShortToGround, "short"),
    "open": (OpenToGround, "open"),
}


def _copy_variables(source_design: QDesign, new_design: QDesign) -> None:
    """Copy `design.variables` (cpw_width, cpw_gap, etc.) to the new design."""
    for key, value in source_design.variables.items():
        new_design.variables[key] = value


def _copy_chips(source_design: QDesign, new_design: QDesign = None) -> QDesign:
    """Copy each chip's configuration (size, material) to the new design.

    NOTE: `chip_cfg` is an `addict.Dict` (nested: chips.main.size.size_x, etc.).
    If it's wrapped in `dict(...)` before the deepcopy, the top level ends up as
    a plain `dict` (addict does NOT automatically re-wrap values assigned via
    `[]`/`.`, only its constructor does), and attribute access such as
    `design_sub.chips.main.size.size_x` breaks with AttributeError. Copying the
    `addict.Dict` directly with `deepcopy` preserves the type at every level.
    """
    if new_design is None:
        new_design = designs.DesignPlanar()

    new_design.overwrite_enabled = True
    _copy_variables(source_design, new_design)

    for chip_name, chip_cfg in source_design.chips.items():
        new_design.chips[chip_name] = deepcopy(chip_cfg)

    return new_design


def _create_termination(
    source_design: QDesign,
    new_design: QDesign,
    ref_comp: str,
    ref_pin: str,
    term_name: str,
    term_type: str,
    flip_terminations: bool,
    trace_width: Optional[str] = None,
    trace_gap: Optional[str] = None,
    reverse_orientation: bool = True,
) -> tuple[str, str]:
    """Create a `ShortToGround`/`OpenToGround` at the position of the dangling pin.

    Position and orientation are read from the original pin in `source_design`
    (component `ref_comp`, pin `ref_pin`), so that the termination ends up
    exactly where the discarded end was and the rebuilt route reproduces the
    same geometry.

    `trace_width`/`trace_gap` (with units, e.g. `'20um'`) belong to the ROUTE
    being reconnected -not to the discarded pin-, because those are what
    actually define the trace width and the ground cut reaching the
    termination. If they aren't passed (route without those options), it falls
    back to the original pin's `width`/`gap` as before.

    `reverse_orientation` (default `True`) controls which way the
    termination's stub/cutout points:
      * `True` (case "kept route -> discarded component"): the stub points
        BACKWARD (reversed normal), so the termination can act as an anchor
        and the rebuilt route reproduces the exact geometry of the original
        pin.
      * `False` (case "kept component -> discarded route"): `ref_comp`'s pin
        still exists as-is in the new design (its metal already ends there,
        with FLAT caps and ZERO clearance beyond the tip -see
        `CircTransmon._make_background`, which uses `cap_style=2`-, because in
        the full design it expected the discarded route to continue the
        ground cut beyond that tip). Without that route, that tip ends up
        directly touching the ground plane. For a real 'open' the cut needs to
        be EXTENDED OUTWARD (same normal as the original pin, not reversed),
        not inward.

    Returns (new_component_name, exposed_pin_name) to redirect the route's
    `pin_inputs` (if applicable).
    """
    pin = source_design.components[ref_comp].pins[ref_pin]
    x_mm, y_mm = pin["middle"]
    nx, ny = pin["normal"]
    # ShortToGround/OpenToGround draw their stub along 'orientation', which is
    # the OPPOSITE of the resulting pin's normal (normal = orientation + 180,
    # verified against ShortToGround/OpenToGround itself).
    orientation = degrees(atan2(ny, nx))
    if reverse_orientation:
        orientation += 180
    if flip_terminations:
        orientation += 180

    cls, pin_name = _TERMINATION_CLASSES[term_type]
    options = Dict(
        chip=pin["chip"],
        pos_x=f"{x_mm}mm",
        pos_y=f"{y_mm}mm",
        orientation=f"{orientation}",
        width=trace_width if trace_width else f"{pin['width']}mm",
    )
    if cls is OpenToGround:
        # OpenToGround also has 'gap' (width of the ground cut alongside the
        # trace); if left at its default (6um) or the discarded pin's own gap
        # instead of the route's real trace_gap, the cut doesn't line up with
        # the incoming line's and a step shows right at the junction.
        # 'termination_gap' (stub length, not a width) is left at its default:
        # there's no original value it needs to match.
        options["gap"] = trace_gap if trace_gap else f"{pin['gap']}mm"
    cls(new_design, term_name, options=options)
    return term_name, pin_name


def build_subdesign(
    source_design: QDesign,
    keep: list[str],
    terminations: Optional[dict[tuple[str, str], str]] = None,
    default_termination: str = "open",
    flip_terminations: bool = False,
    verbose: bool = True,
) -> QDesign:
    """Build a sub-design containing only the components in `keep`.

    Components that aren't routes (`QRoute`) are re-instantiated as-is.
    Routes are re-instantiated, redirecting any endpoint (`start_pin`/`end_pin`)
    that referenced a discarded component to a new termination
    (`ShortToGround` or `OpenToGround`) placed at the exact position of the
    original pin, replicating Ansys/HFSS's
    `render_design(selection=..., open_pins=...)` pattern.

    Args:
        source_design: full design (`QDesign`) to extract the subset from.
        keep: names of the components to keep in the new design.
        terminations: a specific termination for a particular dangling pin.
            Accepts two key forms (both can be mixed in the same dict):
              * `(kept_route_name, 'start' | 'end')` -- referring to the kept
                route itself, e.g. `{('meandro_1', 'end'): 'open'}`.
              * `(discarded_comp_name, pin_name)` -- referring to the
                discarded component the pin used to hang off of, e.g.
                `{('Q1', 'claw_a'): 'open'}`.
            If both keys apply to the same endpoint, `(route_name, endpoint)`
            wins. Takes priority over `default_termination`.
        default_termination: `'short'` or `'open'`, termination for dangling
            pins not covered by `terminations`.
        flip_terminations: by default (`False`) the termination ends up with
            the same normal as the original dangling pin, so the route
            reproduces its geometry. Passing `True` flips it an additional
            180 degrees; only needed if, for some specific component, the
            route still connects backwards.
        verbose: if `True`, prints a summary of what was kept and terminated.

    Returns:
        The new `QDesign`, containing only the components in `keep` plus the
        terminations added for pins left dangling.

    Raises:
        KeyError: if any name in `keep` doesn't exist in `source_design`.
        TypeError: if `terminations` isn't a dict.
        ValueError: if `default_termination` or any value in `terminations`
            isn't `'short'` or `'open'`.
    """
    terminations = terminations or {}
    if not isinstance(terminations, dict):
        raise TypeError(
            "terminations must be a dict, e.g. {('meandro_1', 'end'): 'open'} or "
            "{('Q1', 'claw_a'): 'open'}. Received: " + repr(terminations)
        )

    if default_termination not in _TERMINATION_CLASSES:
        raise ValueError(
            f"Invalid default_termination: {default_termination!r}. "
            f"Must be one of {sorted(_TERMINATION_CLASSES)}."
        )
    for key, term_type in terminations.items():
        if term_type not in _TERMINATION_CLASSES:
            raise ValueError(
                f"Invalid termination {term_type!r} for {key}. "
                f"Must be one of {sorted(_TERMINATION_CLASSES)}."
            )

    missing = [name for name in keep if name not in source_design.components]
    if missing:
        available = sorted(source_design.components.keys())
        raise KeyError(
            f"Components {missing} were not found in source_design. "
            f"Available components: {available}"
        )

    new_design = _copy_chips(source_design)

    kept_set = set(keep)
    all_routes = {n for n, c in source_design.components.items() if isinstance(c, QRoute)}
    kept_routes = kept_set & all_routes
    discarded_routes = all_routes - kept_routes
    kept_noroutes = kept_set - kept_routes

    # Map of (kept_component, pin) -> termination_type.
    # Filled in now so that when the non-route components are instantiated
    # (step 1) we already know which pins need terminating.
    pins_to_terminate = {}
    for route_name in discarded_routes:
        comp = source_design.components[route_name]
        pin_inputs = comp.options.get("pin_inputs") or {}
        for endpoint in ("start_pin", "end_pin"):
            endpoint_opts = pin_inputs.get(endpoint)
            if not endpoint_opts:
                continue

            ref_comp = endpoint_opts.get("component")
            ref_pin = endpoint_opts.get("pin")
            if not ref_comp or ref_comp not in kept_set:
                continue  # Doesn't connect to a kept component.

            # Priority: (kept_comp, pin) key > default.
            # The (route, endpoint) key doesn't apply since the route is discarded.
            term_type = terminations.get((ref_comp, ref_pin), default_termination)
            pins_to_terminate[(ref_comp, ref_pin)] = term_type

    # 1) Non-route components: re-instantiated as-is, at their absolute position.
    #    For each pin left dangling:
    #    - 'open': `ref_comp`'s pin (e.g. Q1.claw_a) already has metal there, but
    #      those components' background 'pocket'/keepout uses FLAT caps at the
    #      tip (see `CircTransmon._make_background`, `cap_style=2`), because in
    #      the full design it expected the discarded route to continue the
    #      ground cut past that tip. Without that route, the ground cut is ZERO
    #      beyond the pin, i.e. the metal touches the ground plane directly
    #      (short circuit). So an `OpenToGround` still needs to be added there
    #      -but with `reverse_orientation=False`, so its cutout extends
    #      OUTWARD (same direction as the pin's original normal) instead of
    #      inward, overlapping the existing pocket like it used to-. No need to
    #      connect it with a route: `ref_comp`'s pin is left untouched, only the
    #      ground cut next to it is widened.
    #    - 'short': the pocket's flat cap already leaves the metal touching
    #      ground by default, so strictly nothing needs to be added; a
    #      `ShortToGround` (no geometry of its own) + a zero-length route are
    #      added anyway as an explicit anchor, for consistency with the rest of
    #      the module.
    terminations_summary = []  # (term_name, term_type, ref_comp, ref_pin)
    for name in kept_noroutes:
        comp = source_design.components[name]
        type(comp)(new_design, name, options=deepcopy(dict(comp.options)))

        for (ref_comp, ref_pin), term_type in pins_to_terminate.items():
            if ref_comp != name:
                continue

            term_name = f"term_{ref_comp}_{ref_pin}"
            if term_name in new_design.components or term_name in kept_set:
                term_name = f"{term_name}_{len(terminations_summary)}"

            if term_type == "open":
                _create_termination(
                    source_design, new_design, ref_comp, ref_pin, term_name, term_type,
                    flip_terminations, reverse_orientation=False,
                )
                terminations_summary.append((term_name, term_type, ref_comp, ref_pin))
                continue

            new_comp_name, new_pin_name = _create_termination(
                source_design, new_design, ref_comp, ref_pin, term_name, term_type, flip_terminations
            )
            terminations_summary.append((term_name, term_type, ref_comp, ref_pin))

            # Connect the kept component's pin to the new termination.
            route_opts = Dict(
                pin_inputs=Dict(
                    start_pin=Dict(component=ref_comp, pin=ref_pin),
                    end_pin=Dict(component=new_comp_name, pin=new_pin_name),
                )
            )
            RouteStraight(new_design, f"route_to_{term_name}", options=route_opts)

    # 2) Routes: dangling endpoints are redirected to new terminations before instantiating.
    for route_name in kept_routes:
        comp = source_design.components[route_name]
        options = deepcopy(dict(comp.options))
        pin_inputs = options.get("pin_inputs") or {}

        for endpoint in ("start_pin", "end_pin"):
            endpoint_opts = pin_inputs.get(endpoint)
            if not endpoint_opts:
                continue  # This route doesn't use pin_inputs on this end (e.g. absolute start_point).

            ref_comp = endpoint_opts.get("component")
            ref_pin = endpoint_opts.get("pin")
            if not ref_comp:
                continue
            if ref_comp in kept_set:
                continue  # The other end is also kept: not dangling.

            side = endpoint[: -len("_pin")]  # start_pin -> start, end_pin -> end
            # Priority: (route, side) key > (discarded_comp, pin) key > default.
            term_type = terminations.get(
                (route_name, side), terminations.get((ref_comp, ref_pin), default_termination)
            )
            term_name = f"term_{route_name}_{side}"
            if term_name in new_design.components or term_name in kept_set:
                term_name = f"{term_name}_{len(terminations_summary)}"

            new_comp_name, new_pin_name = _create_termination(
                source_design, new_design, ref_comp, ref_pin, term_name, term_type, flip_terminations,
                trace_width=options.get("trace_width"), trace_gap=options.get("trace_gap"),
            )
            endpoint_opts["component"] = new_comp_name
            endpoint_opts["pin"] = new_pin_name
            terminations_summary.append((term_name, term_type, ref_comp, ref_pin))

        type(comp)(new_design, route_name, options=options)

    if verbose:
        print(f"Sub-design created. Components kept ({len(keep)}): {sorted(keep)}")
        if terminations_summary:
            print("Terminations added for dangling pins:")
            for term_name, term_type, ref_comp, ref_pin in terminations_summary:
                print(f"  - {term_name} ({term_type}) replacing {ref_comp}.{ref_pin}")
        else:
            print("No dangling pins: no termination was added.")

    return new_design
