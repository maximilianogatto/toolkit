# `toolkit.components` — custom Qiskit-Metal components

Qiskit-Metal `QComponent` subclasses that are not in `qiskit_metal.qlibrary`. Nothing is
re-exported from the package `__init__`, so import from the module:

```python
from toolkit.components.qubit.circle_transmon import CircTransmon
from toolkit.components.couplers.T import T
```

Each one is an ordinary `QComponent`: it declares `default_options`, builds shapely geometry in
`make()`, and registers it with `add_qgeometry`. That means they work anywhere a stock component
does — the GUI, `build_subdesign`, and the PALACE runners in `toolkit.tools.sim`.

## `qubit/`

| module | class | notes |
|---|---|---|
| `circle_transmon.py` | `CircTransmon` | circular-pad transmon; pins `pads`, `claw_a`, `charge_line` |
| `fluxonium.py` | `FluxoniumPocket` | pocket fluxonium, built from `make_pocket`, `make_flux_bias_line`, `make_charge_line`, `make_readout_line` |
| `fluxonium_manucharyan.py` | `FluxoniumPocket` | Manucharyan-style variant |
| `fluxonium_alex.py` | `AlexFluxonium` | another layout variant |
| `fluxonium_alex_box.py` | `AlexFluxoniumBox` | boxed version of the above |

Note the two distinct `FluxoniumPocket` classes in different modules — import by full module path
so it is unambiguous which one you get.

## `couplers/`

| module | class | notes |
|---|---|---|
| `T.py` | `T` | T-shaped coupler with one port; the other two arms are open to ground. Each segment takes its own width and length, with a common gap to the ground plane. |

## `resonator/`

Placeholder — the package exists but is empty.

## Adding one

Subclass `QComponent` (or `BaseQubit`), set `component_metadata` with a `short_name` and the
`_qgeometry_table_*` flags you need, put the geometry in `make()`, and expose pins with
`add_pin`. Two things the simulation side depends on:

* **Pin names matter.** `build_subdesign` terminates dangling pins by `(component, pin)`, and the
  PALACE port/current-source helpers look up pins by name.
* **Layer discipline.** The runners render `metallic_layer=1` by default and take extra layers
  explicitly (e.g. `extra_layers=[(2, {'evap_mode': None})]` for a Dolan-bridge junction layer).
