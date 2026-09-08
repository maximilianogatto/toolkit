# `toolkit.tools.comp_manipulation` — design-level edits

Operations on a whole `QDesign` rather than on a single component.

```python
from toolkit.tools.comp_manipulation.adjust_design import adjust_design

adjust_design(design, chip_margin=0.15e-3)
```

| function | what it does |
|---|---|
| `adjust_design(design, chip_margin=0.15e-3)` | resizes and re-centres `chips.main` so it wraps the bounding box of **every** component in the design, leaving `chip_margin` (in metres) of substrate on each side |

Useful right before a simulation: an oversized chip wastes mesh on empty substrate, and one that
is too tight puts the far-field boundary inside the near field of the circuit. Run it after the
layout is final and before `build_subdesign`.

Two things to know: it mutates `design` in place, and it uses *every* component in
`design.components` — so run it on the full design, not after you have already dropped things.
Only `chips.main` is touched.
