# `toolkit.tools.sim` — PALACE simulations from a Qiskit-Metal design

Thin wrappers over [SQDMetal](https://github.com/sqdlab/SQDMetal) that take a Qiskit-Metal
`QDesign` and run a PALACE simulation end to end: pick the components you care about, terminate
the pins left dangling, mesh, write the config, run, and hand back labelled results.

Everything here is a convenience layer. If you need an option a dataclass does not expose, drop
to SQDMetal's own API — the configs come out identical, which is checked by diffing them
field-by-field (see *Equivalence* at the bottom).

```python
from toolkit.tools.sim import (load_design, run_eigenmode_sim, EigenmodeSimOptions,
                               MeshAlongPathOptions)

design = load_design("my_chip.json")

data = run_eigenmode_sim(
    design,
    name="resonator",
    output_path="./out",
    components=['meander', 'short_to_gnd'],        # everything else is dropped
    open_terminations=[('meander', 'end')],        # this pin fed a dropped component
    sim_options=EigenmodeSimOptions(palace_dir="/path/to/palace", num_cpus=4),
    fine_mesh_paths=[MeshAlongPathOptions(qObjName='meander', min_size=6e-6)],
)
```

## The five runners

| function | PALACE problem | needs | returns |
|---|---|---|---|
| `run_eigenmode_sim` | `Eigenmode` | — | raw `eig.csv` array, one row per mode |
| `run_drivenmodal_sim` | `Driven` | `ports` (one `excite=True`) | dict with `freqs`, `S21`, … |
| `run_capacitance_sim` | `Electrostatic` | — | labelled capacitance `DataFrame` (fF) |
| `run_inductance_sim` | `Magnetostatic` | `current_sources` | dict, see below |
| `run_comsol_eigenmode_sim` | COMSOL eigenmode | COMSOL licence | — |

They all share the same shape:

```
build_and_validate_subdesign  ->  PALACE_*_Simulation(...)  ->  add_metallic + add_ground_plane
  ->  excitations  ->  fine mesh / AMR  ->  prepare_simulation()  ->  run()  ->  results
```

wrapped in `palace_run_cleanup`, which kills a stray PALACE subprocess and finalises the gmsh
session (SQDMetal leaves it open, and the next simulation's geometry builder expects a fresh
one — if you call SQDMetal directly instead, call `gmsh.finalize()` yourself between runs).

## Sub-designs

`components` names what to keep; everything else is dropped. Any pin that pointed at a dropped
component is then dangling, and gets a termination at the exact position of the original pin —
the same idea as Ansys/HFSS `render_design(selection=..., open_pins=...)`.

* `open_terminations=[(comp, pin), ...]` — shorthand for `'open'` (an `OpenToGround`: it cuts the
  ground plane past the trace tip, adding no metal).
* `terminations={(comp, pin): 'open' | 'short'}` — full control. `'short'` is a `ShortToGround`,
  which draws **no geometry at all**: it just lets the route's ground cut stop there, so the trace
  tip touches the plane. A real galvanic short.

Which one you want is not a style choice — see the duality below.

## Options

One dataclass per simulation (`EigenmodeSimOptions`, `DrivenModalSimOptions`, …) carrying the
solver and mesh settings, plus shared refinement options:

* `MeshAlongPathOptions` — refine along a CPW centreline (`dist_resolution`, `min_size`,
  `max_size`, `taper_dist_min`).
* `FineMeshComponentOptions` — refine uniformly over a group of components.
* `AMROptions` — adaptive mesh refinement (`num_iterations`, `tolerance`,
  `Dorfler_marking_fraction`, …).

Excitations are per-simulation: `EigenmodePortOptions`, `DrivenModalPortOptions` (with `excite`),
`CurrentSourceOptions`.

**`palace_dir` is required and has no useful default.** Omit it and the run fails with
`-np: command not found`.

## Magnetostatics: the pre-flight you should always run

PALACE's magnetostatic solver drops the displacement current, so Ampère's law reduces to
`curl H = J`, whose divergence gives **∇·J = 0**. Current cannot pile up anywhere.

> A current terminal must be a **cut in a galvanically closed loop**: both sides of the terminal
> patch must belong to the *same* connected piece of metal, so the current pushed in at the cut
> can flow back around to it.

A conductor that only touches ground through its own terminal is a dead end. The linear system is
inconsistent, CG stalls or diverges — and PALACE **still writes a full `terminal-M.csv`** of
meaningless numbers. `check_current_loops` catches that in about two seconds:

```python
from toolkit.tools.sim import check_current_loops, plot_current_loops

loops = check_current_loops(design, current_sources=sources, components=keep,
                            terminations=terms, strict=True)   # raises if not closed
print(loops.summary())
plot_current_loops(loops, zoom=0)   # frame terminal 0 to see the cut itself
```

It takes the same arguments as `run_inductance_sim` and builds the sub-design the same way, so
what it tests is what would be simulated. The test is purely topological: take the metal, add the
U-clips, remove the terminal patches, count the connected pieces. **1 piece → closed loops. 2+ →
dead end.**

So each conductor needs a **short to ground at one end and the terminal at the other**. Typically:
drop the far launchpad and terminate that pin `'short'`.

### This is the exact dual of the capacitance run

| | Capacitance (electrostatic) | Inductance (magnetostatic) |
|---|---|---|
| index *i* is | a conductor **body** | a **cut** (terminal patch) |
| conductor must be | galvanically **isolated** — that is what makes it a node | galvanically **closed to ground** — that is what makes it a loop |
| excitation | fix `V_i = 1` | inject `I_i = 1 A` at the cut |
| the others are | **grounded** (`V_j = 0`) | **open** (`I_j = 0`) |

Terminating both ends `'open'` is right for `C` and makes the magnetostatic problem unsolvable.

### What `run_inductance_sim` returns

```python
{
  'inductance_matrix': ndarray,   # (n, n) HENRIES -- PALACE's terminal-M.csv
  'terminal_names':    list,      # same order as current_sources
  'terminal_currents': ndarray,   # the current PALACE imposed (not 1 A; the matrix is normalised)
  'surface_flux':      ndarray,   # Wb per integration area, per terminal
  'flux_per_amp':      ndarray,   # Wb/A, or None if no integration_areas were given
  'output_dir':        str,
}
```

Diagonal = self-inductance of each terminal's loop, off-diagonal = mutual. **That off-diagonal is
the coupling number, with no sign flip** — see the convention trap below.

## Gotchas worth knowing

**Convergence is not checked by PALACE.** It only *warns* when a linear solve runs out of
iterations, exits 0, and writes plausible-looking result files built from the stalled iterate.
`check_solver_convergence` scans `out.log` and raises; the inductance runner calls it by default
(`strict_convergence=False` to downgrade to a log line).

**Maxwell vs mutual matrices, and they are not symmetric conventions.**
`terminal-C.csv` is the *Maxwell* capacitance matrix: positive diagonal, negative off-diagonal,
and the physical capacitance between conductors *i* and *j* is **−C_ij**. But `terminal-M.csv` is
also called "Maxwell" and there the off-diagonal **is** the physical mutual inductance
(Φ₁ = L₁I₁ + MI₂), with no negation — the negated version is the separate `terminal-Mm.csv`,
which SQDMetal does not read. **Negate for C, do not negate for M.**

**Units.** Every frequency field in every `*SimOptions` is in **Hz**. PALACE's `eig.csv` is in
**GHz**, so `data[0][1] * 1e9` before feeding it to `center_freq`. Also `eig.csv` column 2 is
Im{f} (the linewidth), *not* the solver error — those are columns 4 and 5.

**`epr_interfaces` order.** The tuple is unpacked as `(substrate_air, substrate_metal,
metal_air)`, which is not the order you would guess: `('Silicon-Vacuum', 'Silicon-Aluminium',
'Aluminium-Vacuum')` for an Al-on-Si chip. Get it wrong and the frequency is unaffected but the
Q is silently wrong.

**DC current vs standing wave.** A magnetostatic solve forces *uniform* current along the whole
conductor. A λ/4 mode does not — with the short at s = 0 its profile is cos(πs/2ℓ). For a
localised coupler at s₀ the correction is a scale factor, `M_mode ≈ cos(πs₀/2ℓ) · M_sim`.

**Live log.** PALACE's stdout goes to `out.log`, not to the notebook cell. `stream_palace_log`
follows the file from a daemon thread so you can watch the Krylov residual as it happens; the
inductance runner enables it by default (`stream_output=False` to silence).

**Sanity-check against closed forms.** For a CPW, `k = w/(w+2g)` and
`L' = (μ₀/4)·K(k')/K(k)`, `C' = 4ε₀ε_eff·K(k)/K(k')` with `ε_eff = (1+ε_r)/2`. A self-inductance
more than ~2× off `L'·length` means the terminal topology or the mesh is wrong, whatever the
mutual says. And `|M₁₂| ≤ √(L₁₁L₂₂)` is Cauchy–Schwarz on the curl-curl inner product, not a rule
of thumb: `k > 1` means what came back is not a field.

## Files

| file | contents |
|---|---|
| `eigenmode_sim_runner.py` | `run_eigenmode_sim`, `EigenmodeSimOptions`, `EigenmodePortOptions` |
| `drivenmodal_sim_runner.py` | `run_drivenmodal_sim`, `DrivenModalSimOptions`, `DrivenModalPortOptions` |
| `capacitance_sim_runner.py` | `run_capacitance_sim`, `CapacitanceSimOptions` |
| `inductance_sim_runner.py` | `run_inductance_sim`, `InductanceSimOptions`, `CurrentSourceOptions` |
| `comsol_eigenmode_sim_runner.py` | the COMSOL path |
| `current_loops.py` | `check_current_loops`, `plot_current_loops`, `CurrentLoopReport` |
| `sim_subdesign.py` | `build_subdesign` — the `open_pins` pattern |
| `save_load_design.py` | `save_design` / `load_design`, JSON round-trip of a `QDesign` |
| `_common.py` | mesh option dataclasses, `check_solver_convergence`, `stream_palace_log`, cleanup |
| `test_sim_subdesign.py` | tests for the sub-design builder |

## Equivalence with raw SQDMetal

The runners are wrappers, not a reimplementation: for eigenmode and both capacitance runs, the
`config.json` they produce is byte-identical to the one you get by driving SQDMetal directly with
the same settings, and the driven-modal one differs only in keys the installed SQDMetal fills from
its own defaults. If you suspect the wrapper, diff the configs rather than guessing — that check
is short enough to keep in a notebook.
