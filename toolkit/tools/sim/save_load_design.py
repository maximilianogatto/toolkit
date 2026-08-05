"""JSON-based save/load for Quantum/Qiskit Metal designs.

`QDesign.save_design`/`QDesign.load_design` pickle the whole design object,
which breaks in practice: any renderer that has been used (e.g. GDS) holds a
live, non-picklable handle (`gdstk.Library`), and `Components.__getattr__`
(qiskit_metal/designs/interface_components.py) recurses infinitely when
pickle/deepcopy touches a `design.components` reference before its `_design`
back-reference has been restored -the class's own `__getstate__`/
`__setstate__` were left unimplemented for exactly this reason, see the
"Down the line for serialization and pickling. Skip for now" comment there.

This instead records, per component, its class (module + qualified name) and
`options`, plus `design.chips` and `design.variables`, as JSON. Recovering
the design just re-runs those constructor calls -the same approach
`build_subdesign` (sim_subdesign.py) uses to rebuild a subset of components.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

import importlib
import json
from copy import deepcopy

from qiskit_metal import Dict
from qiskit_metal.designs.design_base import QDesign
from qiskit_metal.designs.interface_components import Components


def components_to_dict(components: Components) -> list[dict]:
    """Convert a design's `Components` interface into a JSON-safe list.

    Args:
        components: `design.components` of the `QDesign` to serialize.

    Returns:
        One entry per component, in original order, each holding the
        component's `name`, defining class (`module` + `class`), and
        `options`.
    """
    return [{
        "name": name,
        "module": type(comp).__module__,
        "class": type(comp).__qualname__,
        "options": dict(comp.options),
    } for name, comp in components.items()]


def save_design(design: QDesign, filename: str) -> bool:
    """Save `design` to a JSON file.

    Args:
        design: The design object to save.
        filename: The file path where the design will be saved.

    Returns:
        bool: True if successful, False otherwise.
    """
    data = {
        "design_module": type(design).__module__,
        "design_class": type(design).__qualname__,
        "variables": dict(design.variables),
        "chips": dict(design.chips),
        "components": components_to_dict(design.components),
    }
    try:
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
    except OSError as e:
        design.logger.error(f'ERROR WHILE SAVING: {e}')
        return False
    return True


def load_design(filename: str) -> QDesign:
    """Recreate a `QDesign` previously saved with `save_design`.

    Recreates an empty design of the recorded class, then re-instantiates
    each component in its original order via its recorded class and
    `options` -so any component whose `pin_inputs` reference an
    earlier-listed component finds it already in place.

    Args:
        filename: JSON file path produced by `save_design`.

    Returns:
        The reconstructed `QDesign`.
    """
    with open(filename) as f:
        data = json.load(f)

    design_cls = getattr(importlib.import_module(data["design_module"]), data["design_class"])
    design = design_cls()
    design.overwrite_enabled = True

    for key, value in data["variables"].items():
        design.variables[key] = value
    for chip_name, chip_cfg in data["chips"].items():
        # `chip_cfg` is a plain dict (from json.load); `Dict(...)` -addict's constructor-
        # recursively wraps nested dicts into addict.Dict so `design.chips[chip_name].material`
        # (attribute access) works, not just `design.chips[chip_name]['material']`.
        # `design.chips[chip_name] = deepcopy(chip_cfg)` would NOT do this: addict.Dict.__setitem__
        # does not auto-wrap assigned plain dicts, only the constructor does.
        design.chips[chip_name] = Dict(deepcopy(chip_cfg))

    for comp_data in data["components"]:
        comp_cls = getattr(importlib.import_module(comp_data["module"]), comp_data["class"])
        comp_cls(design, comp_data["name"], options=deepcopy(comp_data["options"]))

    return design
