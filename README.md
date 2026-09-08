# toolkit

Custom Qiskit-Metal components and tooling for the superconducting-circuit design and simulation
flow, wrapping [Qiskit Metal](https://qiskit-community.github.io/qiskit-metal/) and
[SQDMetal](https://github.com/sqdlab/SQDMetal) / [PALACE](https://awslabs.github.io/palace/).

Developed for QCT, IFAE Barcelona.

## Install

```bash
pip install -e .
```

Dependencies are **not** declared in `pyproject.toml` — the package assumes it is installed into
an environment that already has `qiskit_metal`, `SQDMetal`, `gmsh`, `shapely`, `numpy`, `scipy`,
`pandas`, `geopandas` and `matplotlib`. The PALACE binary is a separate, external requirement:
its path is passed per simulation as `palace_dir`.

## Layout

| package | what it holds | docs |
|---|---|---|
| `toolkit.components` | custom Qiskit-Metal `QComponent`s: transmons, fluxoniums, couplers | [README](toolkit/components/README.md) |
| `toolkit.tools.sim` | PALACE simulations from a design: eigenmode, driven, capacitance, inductance | [README](toolkit/tools/sim/README.md) |
| `toolkit.tools.comp_manipulation` | whole-design edits, e.g. fitting the chip to its contents | [README](toolkit/tools/comp_manipulation/README.md) |
| `toolkit.fit` | placeholder for measurement fitting; currently empty | — |

## The flow

```python
from toolkit.tools.sim import load_design, run_eigenmode_sim, EigenmodeSimOptions

design = load_design("my_chip.json")        # JSON round-trip of a QDesign
data = run_eigenmode_sim(design, name="resonator", output_path="./out",
                         components=['meander', 'short_to_gnd'],
                         open_terminations=[('meander', 'end')],
                         sim_options=EigenmodeSimOptions(palace_dir=PALACE, num_cpus=4))
```

Build the layout with Qiskit-Metal plus the components in `toolkit.components`, save it with
`save_design`, then simulate a *subset* of it: each runner keeps the components you name, drops
the rest, and terminates the pins left dangling — the same pattern as Ansys/HFSS
`render_design(selection=..., open_pins=...)`. That way one design file feeds every simulation,
and each simulation sees only the part it is about.

The simulation package's [README](toolkit/tools/sim/README.md) is the one to read: besides the
API it collects the conventions and traps that are easy to get wrong and hard to notice —
unit mismatches, the Maxwell-vs-mutual matrix conventions, closed current loops in
magnetostatics, and the fact that PALACE only *warns* when a solve fails to converge while still
writing plausible-looking results.
