"""Importing geometry into a cavity: layouts (polygons, GDS, Quantum Metal) and STEP files.

Everything here stops at the mesh and the Palace config: no solver is run, so the whole file
takes seconds. Run with `pytest tests/test_imports.py`.
"""

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import box

from toolkit.components.attributes import LumpedElement
from toolkit.components.cavity import (CylindricalCavity, CylindricalCavityOptions,
                                       StepCavity, StepCavityOptions)
from toolkit.components.inserts import Chip, StepPart
from toolkit.components.layout import Junction, Layout
from toolkit.components.ports import CoaxPort
from toolkit.components.select import at, in_box, where
from toolkit.materials import get_material

R, H = 12.0, 22.0          # cavity, mm
PAD, GAP, JJ = (0.4, 0.25), 0.1, 0.02


def transmon_layout() -> Layout:
    """Two pads along x with a junction across the gap between them."""
    w, h = PAD
    left, right = box(-GAP / 2 - w, -h / 2, -GAP / 2, h / 2), box(GAP / 2, -h / 2, GAP / 2 + w, h / 2)
    junction = Junction("jj", box(-GAP / 2, -JJ / 2, GAP / 2, JJ / 2), (1.0, 0.0))
    return Layout({"metal": [left, right]}, [junction], outline=(-1.0, -0.5, 1.0, 0.5))


def cavity(**kwargs) -> CylindricalCavity:
    return CylindricalCavity(CylindricalCavityOptions(radius=R, height=H, lc=4.0, mesh_order=1), **kwargs)


def chip(layout=None, **kwargs) -> Chip:
    defaults = dict(junctions=LumpedElement(L=10e-9), position=(0, 0, H / 2), rotation=(90, 0, 0))
    return Chip(layout or transmon_layout(), **{**defaults, **kwargs})


# ---------- layouts ----------
def test_layout_from_polygons():
    layout = transmon_layout()
    assert len(layout.layers["metal"].geoms) == 2
    assert layout.junctions[0].size == pytest.approx((GAP, JJ))
    assert layout.min_feature == pytest.approx(JJ)


def test_layout_plot():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    ax = transmon_layout().plot()
    assert len(ax.patches) == 3          # two pads and the junction


def test_layout_warns_on_a_junction_that_connects_nothing():
    with pytest.warns(UserWarning, match="does not touch metal"):
        Layout({"metal": [box(0, 0, 1, 1)]}, [Junction("jj", box(2, 0, 3, 0.1), (1.0, 0.0))])


def test_layout_from_gds(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    w, h = PAD
    um = 1e3                                         # the file is drawn in um
    library = gdstk.Library(unit=1e-6)
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((-(GAP / 2 + w) * um, -h / 2 * um), (-GAP / 2 * um, h / 2 * um), layer=1))
    cell.add(gdstk.rectangle((GAP / 2 * um, -h / 2 * um), ((GAP / 2 + w) * um, h / 2 * um), layer=1))
    cell.add(gdstk.rectangle((-GAP / 2 * um, -JJ / 2 * um), (GAP / 2 * um, JJ / 2 * um), layer=2))
    cell.add(gdstk.rectangle((-1000, -500), (1000, 500), layer=10))
    path = tmp_path / "transmon.gds"
    library.write_gds(path)

    layout = Layout.from_gds(path, layers={"metal": 1}, junctions=2, outline=10)
    assert layout.outline == pytest.approx((-1.0, -0.5, 1.0, 0.5))
    assert layout.metal.area == pytest.approx(2 * w * h)
    assert [j.name for j in layout.junctions] == ["jj1"]
    assert layout.junctions[0].direction == (1.0, 0.0)

    with pytest.raises(KeyError, match="no polygons on GDS layer"):
        Layout.from_gds(path, layers={"metal": 7})


def test_layout_from_gds_negative_mask(tmp_path):
    gdstk = pytest.importorskip("gdstk")
    library = gdstk.Library(unit=1e-6)
    cell = library.new_cell("TOP")
    cell.add(gdstk.rectangle((-100, -100), (100, 100), layer=1))       # a gap, not metal
    path = tmp_path / "negative.gds"
    library.write_gds(path)
    layout = Layout.from_gds(path, layers={"metal": 1}, outline=(-1, -1, 1, 1), invert=("metal",))
    assert layout.metal.area == pytest.approx(4 - 0.04)


def test_layout_from_metal():
    pytest.importorskip("qiskit_metal")
    from qiskit_metal import designs
    from qiskit_metal.qlibrary.qubits.transmon_pocket import TransmonPocket

    design = designs.DesignPlanar(enable_renderers=False)
    design.chips.main.size["size_x"], design.chips.main.size["size_y"] = "3mm", "3mm"
    TransmonPocket(design, "Q1", options=dict(pad_width="400um", pad_height="90um", pad_gap="30um",
                                               inductor_width="20um"))

    pads = Layout.from_metal(design)                          # a 3D transmon: bare substrate
    assert len(pads.layers["metal"].geoms) == 2
    assert [j.name for j in pads.junctions] == ["Q1_rect_jj"]
    assert pads.junctions[0].size == pytest.approx((0.03, 0.02))
    assert pads.outline == pytest.approx((-1.5, -1.5, 1.5, 1.5))

    planar = Layout.from_metal(design, ground_plane=True)     # a planar chip: ground minus the pocket
    assert planar.metal.area > pads.metal.area
    assert len(planar.layers["metal"].geoms) == 3             # ground plane + two pads

    with pytest.raises(KeyError, match="not in the design"):
        Layout.from_metal(design, components=["nope"])


def test_fillet_rounds_a_corner_and_keeps_the_ends():
    from shapely.geometry import LineString
    from toolkit.components.layout.layout import _fillet

    line = _fillet(LineString([(0, 0), (1, 0), (1, 1)]), radius=0.2, resolution=32)
    assert line.coords[0] == (0, 0) and line.coords[-1] == (1, 1)
    assert line.length == pytest.approx(2 - 2 * 0.2 + np.pi * 0.2 / 2, rel=1e-3)
    centre = np.array([0.8, 0.2])
    arc = np.asarray(line.coords)[1:-1]
    assert np.allclose(np.linalg.norm(arc - centre, axis=1), 0.2)
    # no room for the arc: the corner stays sharp
    assert len(_fillet(LineString([(0, 0), (0.1, 0), (0.1, 1)]), radius=0.2).coords) == 3


# ---------- a chip inside a cavity ----------
def test_chip_in_cavity_groups(tmp_path):
    cav = cavity(inserts=[chip()])
    groups = cav.build_mesh("chip", tmp_path / "chip.msh")
    assert {"cavity", "cavity_wall", "chip_substrate", "chip_metal", "chip_jj"} <= set(groups)
    assert cav.regions["chip_substrate"].dim == 3
    # rotated 90 deg about x: the junction still runs along x, and sapphire's c-axis turned with it
    assert cav.regions["chip_jj"].direction == pytest.approx((1.0, 0.0, 0.0))
    assert np.allclose(cav.regions["chip_substrate"].axes[2], [0, -1, 0])


def test_chip_and_port_do_not_steal_each_others_faces(tmp_path):
    """The port axis goes straight through the junction: its faces must stay the chip's."""
    port = CoaxPort(pos=(0, -R - 2, H / 2), normal=(0, 1, 0), depth=4.0, length=3.0)
    cav = cavity(ports=[port], inserts=[chip(rotation=(0, 0, 0))])
    groups = cav.build_mesh("both", tmp_path / "both.msh")
    assert {"port1", "port1_pin", "chip_jj", "chip_metal"} <= set(groups)


def test_chip_needs_every_junction_accounted_for():
    with pytest.raises(ValueError, match="junctions"):
        Chip(transmon_layout())
    Chip(transmon_layout(), junctions={"jj": None})           # an open gap is a choice, not an error


def test_palace_config_has_the_chip(tmp_path):
    pytest.importorskip("pypalace")
    import json
    from toolkit.palaceModel import EigensimPalace, EigensimPalaceOptions

    port = CoaxPort(pos=(0, -R - 2, H / 2), normal=(0, 1, 0), depth=3.0, length=3.0)
    model = EigensimPalace("cfg", cavity(ports=[port], inserts=[chip(junctions=LumpedElement(L=10e-9, C=2e-15))]),
                           tmp_path, EigensimPalaceOptions(target_GHz=3.0))
    model._make_config(model._make_mesh())
    config = json.loads((tmp_path / "cfg.json").read_text())
    groups = model.groups

    sapphire = next(m for m in config["Domains"]["Materials"] if m["Attributes"] == [groups["chip_substrate"]])
    assert sapphire["Permittivity"] == [9.2725, 9.2725, 11.3486] and "MaterialAxes" in sapphire
    junction = next(p for p in config["Boundaries"]["LumpedPort"] if p["Attributes"] == [groups["chip_jj"]])
    assert junction == {"Index": 2, "Attributes": [groups["chip_jj"]], "Direction": "+X", "L": 10e-9, "C": 2e-15}
    assert groups["chip_metal"] in config["Boundaries"]["PEC"]["Attributes"]     # aluminium at mK
    assert model.lumped_index == {"chip_jj": 2}


# ---------- STEP ----------
@pytest.fixture(scope="module")
def step_files(tmp_path_factory) -> tuple[Path, Path]:
    """The same cylinder twice: as the vacuum, and as the metal block around it."""
    out = tmp_path_factory.mktemp("step")
    cavity().export_step("vac", out / "vacuum.step")
    cavity().export_solid("metal", out / "metal.step", wall=4.0, verbose=False)
    return out / "vacuum.step", out / "metal.step"


@pytest.mark.parametrize("kind", ["vacuum", "metal"])
def test_step_cavity_is_the_same_cylinder(step_files, tmp_path, kind):
    import gmsh
    path = step_files[kind == "metal"]
    cav = StepCavity(StepCavityOptions(path=path, kind=kind, lc=4.0, mesh_order=1))
    gmsh.initialize()
    try:
        groups = cav._prepare(kind)
        volume = sum(gmsh.model.occ.getMass(3, int(t))
                     for t in gmsh.model.getEntitiesForPhysicalGroup(3, groups["cavity"]))
    finally:
        gmsh.finalize()
    assert set(groups) == {"cavity", "cavity_wall"}
    assert volume == pytest.approx(np.pi * R ** 2 * H, rel=1e-6)


def test_step_cavity_takes_ports_chips_and_assignments(step_files, tmp_path):
    port = CoaxPort(pos=(0, -R - 2, H / 2), normal=(0, 1, 0), depth=3.0, length=3.0)
    cav = StepCavity(StepCavityOptions(path=step_files[0], lc=4.0, mesh_order=1), ports=[port],
                     inserts=[chip()])
    cav.assign("lid", at((0, 0, H)), get_material("aluminium", "mK"))
    cav.assign("floor", where(lambda f: f.hi[2] < 1e-6), get_material("niobium", "mK"))
    groups = cav.build_mesh("step", tmp_path / "step.msh")
    assert {"lid", "floor", "cavity_wall", "port1", "chip_jj"} <= set(groups)
    assert cav.regions["lid"].attribute.name == "aluminium"


def test_assign_complains_when_nothing_is_picked(tmp_path):
    cav = cavity().assign("ghost", in_box((50, 50, 50), (60, 60, 60)), get_material("copper", "room"))
    with pytest.raises(ValueError, match="picks no free face"):
        cav.build_mesh("ghost", tmp_path / "ghost.msh")


def test_step_parts_dielectric_and_metal(step_files, tmp_path):
    """A sapphire rod stays as a volume; a copper post leaves only its skin."""
    import gmsh
    rod, post = tmp_path / "rod.step", tmp_path / "post.step"
    for path, (radius, height) in ((rod, (1.0, 6.0)), (post, (1.5, 5.0))):
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, height, radius)
        gmsh.model.occ.synchronize()
        gmsh.write(str(path))
        gmsh.finalize()

    cav = cavity(inserts=[
        StepPart(rod, get_material("sapphire_hemex", "mK"), name="rod", position=(5, 0, 8)),
        StepPart(post, get_material("copper", "room"), name="post", position=(-5, 0, 0))])
    gmsh.initialize()
    try:
        groups = cav._prepare("parts")
        vacuum = sum(gmsh.model.occ.getMass(3, int(t))
                     for t in gmsh.model.getEntitiesForPhysicalGroup(3, groups["cavity"]))
    finally:
        gmsh.finalize()
    assert cav.regions["rod_volume"].dim == 3 and cav.regions["post_surface"].dim == 2
    expected = np.pi * (R ** 2 * H - 1.0 ** 2 * 6.0 - 1.5 ** 2 * 5.0)
    assert vacuum == pytest.approx(expected, rel=1e-6)
