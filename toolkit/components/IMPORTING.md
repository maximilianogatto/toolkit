# Importing geometry into a cavity

Three kinds of file, two kinds of object. A planar design becomes a `Layout`, a 3D one stays a
solid; then both get **roles** -- material, metal, lumped element -- and go into a `Cavity`.

| source | read with | becomes |
|---|---|---|
| Quantum Metal / Qiskit Metal `QDesign` | `Layout.from_metal(design)` | `Layout` -> `Chip` |
| `.gds` | `Layout.from_gds(path, layers=...)` | `Layout` -> `Chip` |
| `.step` that *is* the cavity | `StepCavity(StepCavityOptions(path=...))` | a `Cavity` |
| `.step` of a part inside it | `StepPart(path, material)` | an insert |

Everything is in **mm**, in the cavity frame. Palace is the only backend that understands inserts
and assigned faces; the COMSOL one raises.

## A chip from a layout

```python
from toolkit.components.layout import Layout
from toolkit.components.inserts import Chip
from toolkit.components.attributes import LumpedElement

layout = Layout.from_metal(design, components=["Q1"])            # pads on a bare substrate
layout = Layout.from_metal(design, ground_plane=True)            # planar chip: ground minus the gaps
layout = Layout.from_gds("chip.gds", layers={"metal": 1}, junctions=2, outline=10)
layout.plot()                                                    # look before meshing

chip = Chip(layout,
            substrate=get_material("sapphire_hemex", "mK"), thickness=0.43,
            metal=get_material("aluminium", "mK"),               # or {"layer name": Conductor}
            junctions=LumpedElement(L=10e-9),                    # or {"junction name": LumpedElement}
            position=(0, 0, 11), rotation=(0, -90, 0),           # degrees about x, then y, then z
            mesh_size=0.05)
cavity = CylindricalCavity(options, ports=[...], inserts=[chip])
```

In its own frame the chip has the metal at z = 0 and the substrate below it; the centre of the
outline lands on `position`. Groups: `chip_substrate`, `chip_<layer>`, `chip_<junction>`.

What the readers do with each source:

* **Quantum Metal**: reads `design.qgeometry` (`poly`, `path`, `junction` tables). Paths are
  filleted and widened to polygons; a junction is its line widened by its `width`, with the
  current along the line. `subtract` shapes only matter with `ground_plane=True`.
* **GDS**: a GDS knows layer numbers only, so `layers` names them. Rectangles on the `junctions`
  layer become `jj1`, `jj2`, ...; the current direction is the one along which the rectangle
  touches metal at both ends. `invert=("metal",)` for a negative mask. Units come from the file.

A junction is modelled as a linear inductor. `EigensimPalace.participation(mode)` gives the
energy-participation ratio of each one, and `results()` carries it as `p_<group>` columns.

## A cavity or a part from STEP

```python
from toolkit.components.cavity import StepCavity, StepCavityOptions
from toolkit.components.inserts import StepPart
from toolkit.components.select import at, in_box, where

cavity = StepCavity(StepCavityOptions(path="storage.step", kind="metal", lc=2.0),
                    ports=[...], cavity_wall_material=get_material("copper", "mK"))
cavity.add(StepPart("rod.step", get_material("sapphire_hemex", "mK"), name="rod", position=(5, 0, 8)))
cavity.assign("lid", at((0, 0, 30)), get_material("aluminium", "mK"))
```

* `kind="vacuum"`: the solid is the empty space. `kind="metal"`: the solid is the machined part,
  and the cavity is the largest pocket it encloses inside its bounding box.
* A `Dielectric` part stays as a volume (`<name>_volume`); a `Conductor` part is removed and only
  its skin kept as a metal boundary (`<name>_surface`).

## Roles on faces: `assign`

STEP has no names a solver can use and tags change with every boolean, so faces are picked by
position, in the final model:

| selector | picks |
|---|---|
| `at(point)` | the face(s) the point lies on |
| `in_box(lo, hi)` | faces entirely inside a box |
| `where(lambda f: ...)` | any test on `f.centre`, `f.lo`, `f.hi`, `f.area` |

`cavity.assign(name, selector, attribute)` takes a `Conductor` or a `LumpedElement` (with its
`direction`). It only chooses among faces nothing else has claimed, so it cannot take a port or a
chip's metal. Every outer face left over is `cavity_wall`; faces between two volumes need nothing.
A selector that picks nothing is an error, not a silent no-op.

## Checked so far

`tests/test_imports.py` (no solver, seconds). With Palace: a cylinder through STEP gives TM010
within the mesh error, and a two-pad transmon in the cavity gives a qubit mode whose frequency
halves when L is quadrupled, with a junction participation above 0.98.

Not done: named bodies from STEP labels, lumped elements that launch a signal (use a `Port`),
metal thickness, kinetic inductance, the COMSOL backend.
