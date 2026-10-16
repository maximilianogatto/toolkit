# abstract class to make cavities
import json

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
import gmsh
from pathlib import Path
from toolkit.materials.materials import Dielectric, Conductor, Material, get_material
from toolkit.components.ports import Port

@dataclass
class CavityOptions(ABC):
    """Base class for cavity options."""
    lc : float = 1.5         # max mesh size, mm
    algorithm : int = 6       # gmsh mesh algorithm
    mesh_size_from_curvature : int = 12  # about 12 elements around a full circle
    material : Material | None = None  # optional material for the cavity, e.g. copper
    mesh_order : int = 2  # order of the mesh elements


class Cavity(ABC):
    def __init__(self, options: CavityOptions, ports: list[Port] = None, cavity_wall_material: Material = get_material("copper", 'room'), volume_material: Material = get_material("vacuum", 'room')):
        self.options = options
        self.ports = ports if ports is not None else []
        self.cavity_wall_material = cavity_wall_material
        self.volume_material = volume_material

    @abstractmethod
    def _add_body(self, occ) -> list[tuple[int, int]]:
        """Add the cavity body to the gmsh model."""
        pass

    def _build_geometry(self, occ, verbose=False) -> tuple[list[int], list[list[int]]]:
        """Build the cavity geometry in gmsh."""
        body = self._add_body(occ)
        if verbose:
            print(f"Added cavity body: {body}")
        if not self.ports:                   # no ports: nothing to cut or fragment
            occ.synchronize()
            return [v for _, v in body], []
        # add the ports
        insulators = []
        pins = []
        for port in self.ports:
            pin, coax = port._add_port(occ)
            ins, _ = occ.cut([coax], [pin], removeTool=False) #tube between pin and coax is the insulator
            ins, _ = occ.cut(ins, body, removeTool=False) #cut the insulator with the cavity body
            insulators += ins
            if verbose:
                print(f"Added insulator: {ins}")
            pins.append(pin)

        if verbose:
            print(f"Added pins: {pins}")
        body, _ = occ.cut(body, pins)
        if len(body) != 1:
            raise ValueError(f"Expected one volume after adding body and ports, got {len(body)}. A coax port may be outside the cavity or the cavity may be too small.")
        # check that the number of insulators and pins matches the number of ports
        if len(insulators) != len(self.ports) or len(pins) != len(self.ports):
            raise ValueError(f"Mismatch in number of ports and insulators/pins: {len(self.ports)} ports, {len(insulators)} insulators, {len(pins)} pins.")

        # fragment the cavity body with the insulators to create separate volumes
        out, out_map = occ.fragment(body, insulators)
        occ.synchronize()

        if verbose:
            print(f"Fragmented cavity body with insulators: {out}")

        # create volumens for the vacuum and insulators
        vac_vols = [v for _, v in out_map[0]]   # the cavity volume (vacuum)
        ins_vols = [[v for _, v in out_map[i+1]] for i in range(len(self.ports))] # the insulator volumes

        if verbose:
            print(f"Created vacuum volumes: {vac_vols}, insulator volumes: {ins_vols}")
        return vac_vols, ins_vols

    # ---------- 3. physical groups ----------
    def _add_groups(self, cavity, ins_per_port, verbose=False):
        add = gmsh.model.addPhysicalGroup
        groups = {"cavity": add(3, cavity, name="cavity")}
        cavity_faces = self._faces_of(cavity)
        used, interfaces = set(), set()

        for i, (port, insulator) in enumerate(zip(self.ports, ins_per_port), start=1):
            faces = port.classify_faces(self._faces_of(insulator), cavity_faces)
            if verbose:
                print(f"port{i} faces: {faces}")
            groups[f"port{i}_insulator"] = add(3, insulator, name=f"port{i}_insulator")
            groups[f"port{i}"] = add(2, list(faces["ring"]), name=f"port{i}")
            groups[f"port{i}_pin"] = add(2, list(faces["pin"]), name=f"port{i}_pin")
            groups[f"port{i}_shield"] = add(2, list(faces["shield"]), name=f"port{i}_shield")
            used |= faces["ring"] | faces["pin"] | faces["shield"] | faces["interface"]
            interfaces |= faces["interface"]

        walls = cavity_faces - used
        if verbose:
            print(f"cavity walls: {sorted(walls)}, interfaces: {sorted(interfaces)}")
        groups["cavity_wall"] = add(2, list(walls), name="cavity_wall")
        return groups, interfaces

    def _check_faces_to_groups(self, interfaces, verbose=False) -> bool:
        """True if every face is in a physical group, except the insulator/cavity interfaces."""
        all_faces = {s for _, s in gmsh.model.getEntities(2)}
        grouped = set()
        for dim, tag in gmsh.model.getPhysicalGroups(2):
            grouped |= set(gmsh.model.getEntitiesForPhysicalGroup(dim, tag))
        missing = all_faces - grouped - interfaces
        if verbose:
            print(f"all faces: {len(all_faces)}, grouped: {len(grouped)}, "
                f"interfaces: {len(interfaces)}, missing: {sorted(missing)}")
        return not missing

    # ---------- main ----------
    def _prepare(self, name, verbose: bool = False) -> dict[str, int]:
        """Build the geometry and tag every volume and face. Call inside a gmsh session."""
        gmsh.model.add(name)
        cavity, ins_per_port = self._build_geometry(gmsh.model.occ)
        groups, interfaces = self._add_groups(cavity, ins_per_port, verbose=verbose)
        if not self._check_faces_to_groups(interfaces, verbose=verbose):
            raise ValueError("Some faces are not assigned to any physical group. Check the geometry and port placement.")
        return groups

    def build_mesh(self, name, output: Path, gui: bool = False, verbose: bool = False) -> dict[str, int]:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            groups = self._prepare(name, verbose=verbose)

            gmsh.option.setNumber("Mesh.MeshSizeMax", self.options.lc)
            gmsh.option.setNumber("Mesh.Algorithm", self.options.algorithm)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", self.options.mesh_size_from_curvature)
            gmsh.model.mesh.generate(3)
            gmsh.model.mesh.setOrder(self.options.mesh_order)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.option.setNumber("Mesh.Binary", 1)

            output.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output))
            if gui:
                gmsh.fltk.run()
            return groups
        finally:
            gmsh.finalize()

    # ---------- pictures ----------
    # one colour per physical group, in the order `build_mesh` returns them; the legend reuses it
    _PALETTE = [(90, 160, 255), (255, 170, 60), (120, 200, 120), (230, 100, 110), (180, 130, 220),
                (120, 200, 210), (240, 220, 90), (150, 150, 150), (200, 120, 170), (185, 150, 115)]

    _CAMERA = {"iso":   (-60, 0, -20),   # three-quarter view: the whole thing at a glance
               "top":   (0, 0, 0),       # down -z onto the lid: port footprint and corner radii
               "front": (-90, 0, 0),     # down +y: x to the right, z up
               "side":  (-90, 0, -90)}   # down -x: y to the right, z up

    def snapshot(self, name, output: Path, views=("iso", "top", "section"), style: str = "solid",
                 only: str | list[str] | None = None, axes: bool = True, legend: bool = True,
                 size: tuple[int, int] = (1600, 1200), zoom: float | None = None,
                 verbose: bool = False) -> list[Path]:
        """Render the design to PNG, into `output/figs/<name>_<view>.png`.

        `output` is the run directory, the same one the simulation writes to, so the pictures of a
        design travel with its results.

        Views are the four cameras in `_CAMERA`, plus:

        * `"section"`: `front` cut by the plane y = (mean port y), drawn one element thick, so the
          wall, the insulator plug and how far the pin reaches are all visible at once.
        * `"port1"`, `"port2"`, ...: the same cut, through that port's own axis and zoomed on it.
          Cut and not a solid close-up on purpose: the pin is a *hole* in the cavity volume, so
          from outside there is nothing to see, and only the section says how deep it goes.

        `only` keeps just some physical groups, by the names `build_mesh` returns (a single name
        or a list, e.g. `only="port1_insulator"`): everything else is hidden, which is the way to
        look at the mesh of one piece on its own.

        `style` is `"solid"` (faces filled, one colour per physical group) or `"wire"` (edges
        only). The mesh is the one `options.lc` asks for, so the picture also shows how well the
        mesh resolves the ports.

        `axes` draws a box with tics in mm -- around the model, or around what is on screen when
        the view is zoomed -- plus a scale bar and a panel of sizes: the port's on a port view,
        the cavity's own everywhere else. The tics give positions; the bar and the sizes give
        lengths, which is what you are after when you ask how big something is.
        `legend` writes the group names in the corner, each in its own colour. Both off gives a
        bare figure for a slide.

        `zoom` is chosen per port from the port's own size, so a long coax does not end up framed
        from the inside; give a number to override it.

        A gmsh window flashes open while this runs: the capture comes from the drawing engine, so
        there is no off-screen path.
        """
        if style not in ("solid", "wire"):
            raise ValueError(f"style is 'solid' or 'wire', got {style!r}")
        unknown = [v for v in views if v not in self._CAMERA and v != "section"
                   and not (v.startswith("port") and v[4:].isdigit())]
        if unknown:
            raise ValueError(f"unknown views {unknown}: use {sorted(self._CAMERA)}, 'section', "
                             f"or 'portN' with N in 1..{len(self.ports)}")
        figs = Path(output) / "figs"
        figs.mkdir(parents=True, exist_ok=True)
        cut = [v for v in views if v == "section" or v.startswith("port")]
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            groups = self._prepare(name, verbose=verbose)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.options.lc)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", self.options.mesh_size_from_curvature)
            gmsh.model.mesh.generate(3 if cut else 2)   # a cut needs elements to cut through
            self._display(style, size)
            colours = self._colour_groups(groups)
            focus = None
            if only is not None:
                only = [only] if isinstance(only, str) else list(only)
            gmsh.fltk.initialize()
            across = self._calibrate(figs / ".calibration.png") if axes else None
            if only is not None:
                focus = self._show_only(groups, only)
                colours = {n: c for n, c in colours.items() if n in only}
            written = []
            for view in views:
                path = figs / f"{name}_{view}.png"
                magnify = self._shoot(view, path, zoom, style, focus, axes)
                notes = None
                if axes:
                    notes = (self._port_notes(int(view[4:]) - 1, groups)
                             if view.startswith("port") else self._design_notes(name))
                scale = across / magnify if across else None   # mm across this picture
                if legend or notes or scale:
                    self._annotate(path, colours if legend else None, notes, scale)
                written.append(path)
                if verbose:
                    print(f"wrote {path}")
            gmsh.fltk.finalize()
            return written
        finally:
            gmsh.finalize()

    @classmethod
    def _colour_groups(cls, groups: dict[str, int]) -> dict[str, tuple[int, int, int]]:
        """Give every physical group its own colour, and say which is which."""
        colours = {}
        for i, (name, group) in enumerate(groups.items()):
            dim = 3 if name == "cavity" or name.endswith("_insulator") else 2
            colour = cls._PALETTE[i % len(cls._PALETTE)]
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, group)
            gmsh.model.setColor([(dim, int(t)) for t in entities], *colour)
            colours[name] = colour
        return colours

    @staticmethod
    def _bounds_of_model() -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([gmsh.option.getNumber(f"General.Min{a}") for a in "XYZ"])
        hi = np.array([gmsh.option.getNumber(f"General.Max{a}") for a in "XYZ"])
        return lo, hi

    @classmethod
    def _calibrate(cls, scratch: Path) -> float:
        """How many millimetres fit across the picture at magnification 1 -- measured, not predicted.

        How gmsh fits a model to its window is not documented and does not follow the obvious
        rule: the same model came out at different scales from one run to the next. So instead of
        reproducing the fit, render once and measure the model's silhouette against its known
        width. What is measured is millimetres per picture width, not pixels per millimetre,
        because the window is still settling at this point and the next picture may well come out
        wider -- the field of view in millimetres survives that, the pixel scale does not.
        Rotating never changes the scale either, and a magnified view simply divides it, so this
        one number serves every view.

        The mask keeps coloured pixels only, which is the model: the paper is white, and the
        axes, the edges and the labels are greys.
        """
        import numpy
        from PIL import Image

        for axis, angle in zip("XYZ", cls._CAMERA["top"]):
            gmsh.option.setNumber(f"General.Rotation{axis}", angle)
        gmsh.option.setNumber("General.Axes", 0)
        gmsh.graphics.draw()
        gmsh.write(str(scratch))                            # a warm-up: the window resizes once
        gmsh.graphics.draw()
        gmsh.write(str(scratch))
        image = numpy.asarray(Image.open(scratch).convert("RGB"), dtype=int)
        scratch.unlink(missing_ok=True)
        coloured = image.max(axis=2) - image.min(axis=2) > 25
        columns = numpy.where(coloured.any(axis=0))[0]
        lo, hi = cls._bounds_of_model()
        silhouette = (columns.max() - columns.min() + 1) / image.shape[1]   # fraction of the width
        return (hi[0] - lo[0]) / silhouette

    @classmethod
    def _scale_bar(cls, draw, font, size: int, image, across: float) -> None:
        """A bar of a round number of millimetres, bottom left: the size of things at a glance."""
        per_mm = image.width / across
        nice = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500]
        wanted = across / 5                                 # about a fifth of the picture
        length = min(nice, key=lambda mm: abs(np.log(mm / wanted)))
        bar, label = length * per_mm, f"{length:g} mm"
        x, y = size * 2, image.height - size * 2
        draw.rectangle((x - size, y - size * 2.2, x + bar + size, y + size * 0.8),
                       fill=(255, 255, 255), outline=(170, 170, 170))
        thick = max(2, size // 7)
        draw.line((x, y, x + bar, y), fill=(40, 40, 40), width=thick)
        for end in (x, x + bar):                            # a tick at each end, like a ruler
            draw.line((end, y - size * 0.45, end, y + size * 0.45), fill=(40, 40, 40), width=thick)
        draw.text((x + bar / 2 - draw.textlength(label, font=font) / 2, y - size * 1.9),
                  label, fill=(40, 40, 40), font=font)

    @classmethod
    def _annotate(cls, path: Path, legend: dict[str, tuple[int, int, int]] | None,
                  notes: list[str] | None, scale: float | None = None) -> None:
        """Write the legend (left) and, on a port view, that port's sizes (right), onto the PNG.

        gmsh has no legend for physical groups. Text drawn inside the scene is an option, but it
        lands on top of the mesh in the group's own colour, which is unreadable the moment a view
        is zoomed in, so both panels go on afterwards, with a background of their own.
        """
        from PIL import Image, ImageDraw, ImageFont

        image = Image.open(path).convert("RGB")
        draw = ImageDraw.Draw(image)
        size = max(14, image.height // 48)                  # the pictures are rendered at any size
        font = ImageFont.load_default(size=size)
        if legend:
            cls._panel(draw, font, size, [(colour, name) for name, colour in legend.items()],
                       right_edge=None)
        if notes:
            cls._panel(draw, font, size, [(None, line) for line in notes],
                       right_edge=image.width)
        if scale:
            cls._scale_bar(draw, font, size, image, scale)
        image.save(path)

    @staticmethod
    def _panel(draw, font, size: int, rows: list[tuple[tuple | None, str]], right_edge) -> None:
        """One box of text, with a colour chip per row where there is a colour.

        A double space in a line splits it into label and value, which are then set in two
        columns: the font is proportional, so padding with spaces does not line numbers up.
        """
        pad, step = size, round(size * 1.5)
        chip = size * 1.75 if any(colour for colour, _ in rows) else 0
        split = [line.split("  ", 1) if "  " in line else [line, ""] for _, line in rows]
        labels = max(draw.textlength(label, font=font) for label, _ in split)
        values = max(draw.textlength(value, font=font) for _, value in split)
        gap = size * 0.6 if values else 0
        width = pad + chip + labels + gap + values + pad
        left = pad if right_edge is None else right_edge - pad - width
        draw.rectangle((left, pad, left + width, pad + step * len(rows) + pad / 2),
                       fill=(255, 255, 255), outline=(170, 170, 170))
        for row, ((colour, _), (label, value)) in enumerate(zip(rows, split)):
            top = pad * 1.25 + step * row
            if colour:
                draw.rectangle((left + pad / 2, top, left + pad / 2 + size, top + size),
                               fill=colour, outline=(120, 120, 120))
            draw.text((left + pad / 2 + chip, top - size / 8), label, fill=(40, 40, 40), font=font)
            if value:
                draw.text((left + pad / 2 + chip + labels + gap, top - size / 8), value,
                          fill=(40, 40, 40), font=font)

    def _design_notes(self, name: str) -> list[str]:
        """The cavity's own dimensions, for the views that are not about one port.

        Every field of `CavityOptions` is a mesh setting, so whatever a subclass declares on top
        of it is, by construction, a dimension of the shape: no list to keep in step by hand, and
        a new cavity class gets its sizes written out for free.
        """
        settings = {f.name for f in fields(CavityOptions)}
        lines = [f"{name}  {type(self).__name__}"]
        for item in fields(self.options):
            if item.name in settings:
                continue
            value = getattr(self.options, item.name)
            try:
                numbers = np.atleast_1d(np.asarray(value, dtype=float))
            except (TypeError, ValueError):                 # not a length: a material, a flag
                lines.append(f"{item.name}  {value}")
                continue
            lines.append(f"{item.name}  " + " ".join(f"{n:.2f}" for n in numbers) + " mm")
        return lines

    def _port_notes(self, index: int, groups: dict[str, int]) -> list[str]:
        """The sizes of port `index` (0-based), in millimetres, straight from the model.

        The ring-to-wall distance `d` is not stored anywhere: it falls out of the insulator, which
        was trimmed exactly at the wall, so its volume gives `d` back. Watch the two cases -- the
        plug is an annulus only where the pin runs through it, and a full disc beyond the pin tip:

            pin through the whole plug (depth >= d):  V = pi (b^2 - a^2) d
            pin stopping inside it     (depth <  d):  V = pi (b^2 - a^2) depth + pi b^2 (d - depth)
        """
        port = self.ports[index]
        tag = gmsh.model.getEntitiesForPhysicalGroup(3, groups[f"port{index + 1}_insulator"])[0]
        volume = gmsh.model.occ.getMass(3, int(tag))
        ring, bore = np.pi * (port.b ** 2 - port.a ** 2), np.pi * port.b ** 2
        wall = volume / ring
        if wall > port.depth:                               # the pin stops short of the wall
            wall = port.depth + (volume - ring * port.depth) / bore
        inside = port.depth - wall
        return [f"port{index + 1}  {type(port).__name__}",
                f"pin dia  {2 * port.a:.2f} mm",
                f"bore dia  {2 * port.b:.2f} mm",
                f"wall  {wall:.2f} mm",
                (f"pin in  {inside:.2f} mm" if inside >= 0 else f"pin short  {-inside:.2f} mm"),
                f"Z0  {port.z0:.1f} ohm"]

    @staticmethod
    def _axes_box(lo, hi) -> None:
        """Draw the ruled box between `lo` and `hi` (mm)."""
        gmsh.option.setNumber("General.Axes", 2)
        gmsh.option.setNumber("General.AxesAutoPosition", 0)
        for axis, low, high in zip("XYZ", lo, hi):
            gmsh.option.setNumber(f"General.AxesMin{axis}", round(float(low), 3))
            gmsh.option.setNumber(f"General.AxesMax{axis}", round(float(high), 3))
            gmsh.option.setNumber(f"General.AxesTics{axis}", 4)

    @staticmethod
    def _show_only(groups: dict[str, int], names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Hide everything except the physical groups in `names`; give back what is left to frame."""
        unknown = [n for n in names if n not in groups]
        if unknown:
            raise ValueError(f"unknown groups {unknown}: pick from {sorted(groups)}")
        gmsh.model.setVisibility(gmsh.model.getEntities(), 0, recursive=True)
        boxes = []
        for dim, group in gmsh.model.getPhysicalGroups():
            if gmsh.model.getPhysicalName(dim, group) in names:
                for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, group):
                    gmsh.model.setVisibility([(dim, int(tag))], 1, recursive=True)
                    boxes.append(gmsh.model.getBoundingBox(dim, int(tag)))
        box = np.array(boxes)
        return box[:, :3].min(axis=0), box[:, 3:].max(axis=0)

    @staticmethod
    def _display(style: str, size: tuple[int, int]) -> None:
        """Drawing options shared by every view."""
        o = gmsh.option.setNumber
        o("General.GraphicsWidth", size[0])
        o("General.GraphicsHeight", size[1])
        o("General.Trackball", 0)            # so the rotation options are the camera
        o("General.SmallAxes", 1)
        o("General.Axes", 0)
        o("Geometry.Points", 0)
        o("Geometry.Curves", 0)
        o("Mesh.ColorCarousel", 1)           # 1: use the colour each entity was given below
        o("Mesh.SurfaceEdges", 1)
        o("Mesh.SurfaceFaces", int(style == "solid"))
        o("Mesh.VolumeFaces", int(style == "solid"))

    def _shoot(self, view: str, path: Path, zoom: float, style: str,
               focus: tuple[np.ndarray, np.ndarray] | None = None, axes: bool = True) -> float:
        """Point the camera as `view` asks and write one PNG."""
        o = gmsh.option.setNumber
        lo_all = np.array([gmsh.option.getNumber(f"General.Min{a}") for a in "XYZ"])
        hi_all = np.array([gmsh.option.getNumber(f"General.Max{a}") for a in "XYZ"])
        if axes:
            self._axes_box(lo_all, hi_all)
        o("Mesh.Clip", 0)                                   # undo whatever the last view set
        o("Mesh.VolumeFaces", int(style == "solid"))
        for axis in "XYZ":
            o(f"General.Translation{axis}", 0.0)
            o(f"General.Scale{axis}", 1.0)

        port = self.ports[int(view[4:]) - 1] if view.startswith("port") else None
        rotation = self._CAMERA["front" if port or view == "section" else view]
        for axis, angle in zip("XYZ", rotation):
            o(f"General.Rotation{axis}", angle)

        if port is not None or view == "section":
            # cut along the port axes: a pin is a hole in the cavity, only a section shows it
            y = port.pos[1] if port is not None else (
                float(np.mean([p.pos[1] for p in self.ports])) if self.ports else self._centre()[1])
            for key, value in zip("ABCD", (0.0, 1.0, 0.0, -y)):
                o(f"General.Clip0{key}", value)
            o("Mesh.Clip", 1)                               # clip plane 0 applies to the mesh

        target = magnify = None
        if port is None:
            o("General.ClipWholeElements", 1)
            o("General.ClipOnlyDrawIntersectingVolume", 1)  # one element thick: a clean overview
            if focus is not None:                           # `only`: fill the frame with what is left
                lo, hi = focus
                target = (lo + hi) / 2
                magnify = max(0.8 * max(hi_all - lo_all) / max(max(hi - lo), 1e-12), 1.0)
        else:
            # the pin is thinner than an element, so a slice of whole elements swallows it: cut the
            # elements exactly instead, and drop the volume faces that would stand in front of it
            o("General.ClipWholeElements", 0)
            o("General.ClipOnlyDrawIntersectingVolume", 0)
            o("Mesh.VolumeFaces", 0)
            target = port.pos + port.normal * getattr(port, "length", 0.0) / 2
            # frame the port and a little wall around it, unless a zoom was asked for
            reach = max(port.length, port.depth) + 2 * port.b
            magnify = zoom or 0.8 * max(hi_all - lo_all) / reach

        if target is not None:
            self._look_at(target, rotation, magnify)
            if axes:                                        # a close-up needs a ruler of its own
                half = max(hi_all - lo_all) / (2 * magnify)
                self._axes_box(np.maximum(target - half, lo_all), np.minimum(target + half, hi_all))

        gmsh.graphics.draw()
        gmsh.write(str(path))
        return magnify or 1.0

    @staticmethod
    def _centre() -> np.ndarray:
        """Centre of the model's bounding box, which is what gmsh puts in the middle of the view."""
        lo = np.array([gmsh.option.getNumber(f"General.Min{a}") for a in "XYZ"])
        hi = np.array([gmsh.option.getNumber(f"General.Max{a}") for a in "XYZ"])
        return (lo + hi) / 2

    @classmethod
    def _look_at(cls, target, rotation, zoom: float) -> None:
        """Centre the view on `target` and magnify by `zoom`.

        gmsh draws S (t + R (x - centre)), so the translation is measured in the *rotated* frame:
        feeding it the model-space offset moves the camera somewhere else entirely.
        """
        offset = -cls._rotation(*rotation) @ (np.asarray(target, dtype=float) - cls._centre())
        for axis, value in zip("XYZ", offset):
            gmsh.option.setNumber(f"General.Translation{axis}", value)
        for axis in "XYZ":
            gmsh.option.setNumber(f"General.Scale{axis}", zoom)

    @staticmethod
    def _rotation(rx: float, ry: float, rz: float) -> np.ndarray:
        """The matrix gmsh builds from the three rotation options: Rx then Ry then Rz."""
        a, b, c = np.radians([rx, ry, rz])
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        return Rx @ Ry @ Rz

    def export_step(self, name, output: Path, tags: Path | None = None,
                    verbose: bool = False) -> dict[str, int]:
        """Write the geometry as STEP, plus a sidecar that says where each group is.

        STEP carries geometry only, so the physical groups are lost on import. The sidecar keeps,
        for every group, a point **on** each entity and the outward normal there, which is what a
        CAD-based solver needs to select the same faces again (COMSOL: a Ball selection at each
        point). Centres of mass are deliberately not used: for a cylindrical wall or an annular
        port ring the centre of mass does not lie on the surface.
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            groups = self._prepare(name, verbose=verbose)
            output.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output))                     # .step / .stp / .brep: format from the suffix
            if tags is not None:
                tags.parent.mkdir(parents=True, exist_ok=True)
                tags.write_text(json.dumps(self._group_tags(groups), indent=2), encoding="utf-8")
            if verbose:
                print(f"wrote {output}" + (f" and {tags}" if tags else ""))
            return groups
        finally:
            gmsh.finalize()

    def export_solid(self, name, output: Path, wall: float = 5.0, shape: str = "box",
                     split: str | None = None, verbose: bool = True) -> None:
        """Write the **metal part** as CAD: the block the cavity is machined out of.

        `export_step` writes the vacuum, which is the shape the field lives in. This writes its
        complement: a block `wall` mm thick around the cavity, with the cavity and a clearance
        hole per port taken out of it. That is the solid to open in Inventor or SolidWorks, and
        the one the workshop needs. The format follows the suffix: .step/.stp (use this one),
        .iges/.igs, or .brep.

        `split="x"` or `"y"` also cuts the part in two along a plane containing the cavity axis.
        For TM010 the wall currents run along z on the side wall and radially on the lids, so they
        never cross such a seam -- a lid-height split would cut straight across them and cost Q.
        """
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.model.add(name)
            occ = gmsh.model.occ
            vacuum, insulators = self._build_geometry(occ, verbose=verbose)
            negative = [(3, v) for v in vacuum] + [(3, v) for vols in insulators for v in vols]

            lo, hi = self._bounds(occ, [(3, v) for v in vacuum])      # the cavity only, not the stubs
            if shape == "box":
                block = occ.addBox(*(lo - wall), *(hi - lo + 2 * wall))
            elif shape == "cylinder":
                centre = (lo + hi) / 2
                radius = max(hi[0] - centre[0], hi[1] - centre[1]) + wall
                block = occ.addCylinder(centre[0], centre[1], lo[2] - wall,
                                        0, 0, hi[2] - lo[2] + 2 * wall, radius)
            else:
                raise ValueError(f"shape must be 'box' or 'cylinder', got {shape!r}")

            holes = []
            for port in self.ports:            # from outside the block, through the wall
                reach = wall + 1.0
                depth = max(port.length, getattr(port, "depth", 0.0))   # also clears the pin tip,
                start = np.asarray(port.pos, dtype=float) - port.normal * reach   # which belongs to
                holes.append((3, occ.addCylinder(*start, *(port.normal * (reach + depth)),
                                                 port.bore_radius)))   # the connector, not the part

            part, _ = occ.cut([(3, block)], negative + holes)
            if split is not None:
                part = self._split(occ, part, split)
            occ.synchronize()

            output.parent.mkdir(parents=True, exist_ok=True)
            gmsh.write(str(output))
            if verbose:
                volume = sum(occ.getMass(3, v) for _, v in part)
                print(f"wrote {output}: {len(part)} solid(s), {volume:.1f} mm^3 of metal")
        finally:
            gmsh.finalize()

    @staticmethod
    def _bounds(occ, dimtags) -> tuple[np.ndarray, np.ndarray]:
        """Corners of the box containing several entities."""
        boxes = np.array([occ.getBoundingBox(dim, tag) for dim, tag in dimtags])
        return boxes[:, :3].min(axis=0), boxes[:, 3:].max(axis=0)

    @classmethod
    def _split(cls, occ, part, axis: str) -> list[tuple[int, int]]:
        """Cut a solid in two with a plane through its centre, normal to `axis`."""
        try:
            i = {"x": 0, "y": 1, "z": 2}[axis]
        except KeyError:
            raise ValueError(f"split must be 'x', 'y' or 'z', got {axis!r}") from None
        lo, hi = cls._bounds(occ, part)
        margin = (hi - lo).max()
        lo, hi = lo - margin, hi + margin
        lo[i] = (lo[i] + hi[i]) / 2                       # half-space on the positive side
        knife = (3, occ.addBox(*lo, *(hi - lo)))
        upper, _ = occ.intersect(part, [knife], removeObject=False, removeTool=False)
        lower, _ = occ.cut(part, [knife])
        return upper + lower

    def _group_tags(self, groups: dict[str, int]) -> dict:
        """Describe every physical group geometrically, for a solver that only reads the STEP."""
        volumes = {"cavity", *(f"port{i}_insulator" for i in range(1, len(self.ports) + 1))}
        out = {"length_unit": "mm", "groups": {}}
        for group_name, tag in groups.items():
            dim = 3 if group_name in volumes else 2
            entities = gmsh.model.getEntitiesForPhysicalGroup(dim, tag)
            out["groups"][group_name] = {
                "dim": dim,
                "tag": int(tag),
                "entities": [self._probe(dim, int(e)) for e in entities],
            }
        return out

    @classmethod
    def _probe(cls, dim: int, tag: int) -> dict:
        """A point well inside an entity, its clearance to the boundary, and the normal there.

        Mid-parameter points are not enough: on the annular port ring, or on the tube of a port
        insulator, the middle of the parameter range (like the centre of mass) falls in the hole.
        So candidates are filtered with `gmsh.model.isInside` and the one farthest from the
        boundary wins -- that clearance is also the largest radius a selection sphere may use.
        """
        size = gmsh.model.occ.getMass(dim, tag)
        points, normals = cls._candidates(dim, tag)
        if len(points) == 0:
            raise ValueError(f"no interior point found on entity ({dim}, {tag})")
        cloud = cls._boundary_cloud(dim, tag)
        if len(cloud):
            clearance = np.linalg.norm(points[:, None, :] - cloud[None, :, :], axis=2).min(axis=1)
        else:
            clearance = np.full(len(points), abs(size) ** (1.0 / dim))
        k = int(np.argmax(clearance))
        probe = {"entity": tag, "size": size, "point": [float(x) for x in points[k]],
                 "clearance": float(clearance[k])}
        if normals is not None:
            probe["normal"] = [float(x) for x in normals[k]]
        return probe

    @classmethod
    def _candidates(cls, dim: int, tag: int, n: int = 7):
        """Interior points of an entity: a parametric grid for a face, a box grid for a volume."""
        if dim == 2:
            uv = cls._face_grid(tag, n)
            if not uv:
                return np.empty((0, 3)), np.empty((0, 3))
            points = np.array([gmsh.model.getValue(2, tag, p) for p in uv]).reshape(-1, 3)
            normals = np.array([gmsh.model.getNormal(tag, p) for p in uv]).reshape(-1, 3)
            return points, normals
        box = gmsh.model.getBoundingBox(3, tag)
        axes = [np.linspace(box[i], box[i + 3], n + 2)[1:-1] for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        inside = np.array([p for p in grid if gmsh.model.isInside(3, tag, list(p))])
        return inside, None

    @staticmethod
    def _face_grid(face: int, n: int = 7) -> list[list[float]]:
        """Parametric points that are really on a (possibly trimmed) face."""
        pmin, pmax = gmsh.model.getParametrizationBounds(2, face)
        us = np.linspace(pmin[0], pmax[0], n + 2)[1:-1]
        vs = np.linspace(pmin[1], pmax[1], n + 2)[1:-1]
        return [[float(u), float(v)] for u in us for v in vs
                if gmsh.model.isInside(2, face, [u, v], parametric=True)]

    @classmethod
    def _boundary_cloud(cls, dim: int, tag: int) -> np.ndarray:
        """Points sampled on the boundary of an entity, to measure how far inside a candidate is."""
        cloud = []
        for _, child in gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=False):
            child = abs(child)
            if dim == 2:
                pmin, pmax = gmsh.model.getParametrizationBounds(1, child)
                ts = np.linspace(pmin[0], pmax[0], 24)
                cloud.append(np.array(gmsh.model.getValue(1, child, list(ts))).reshape(-1, 3))
            else:
                uv = cls._face_grid(child, 5)
                if uv:
                    cloud.append(np.array([gmsh.model.getValue(2, child, p) for p in uv]).reshape(-1, 3))
        return np.vstack(cloud) if cloud else np.empty((0, 3))

    @staticmethod
    def _faces_of(volumes: list[int]) -> set[int]:
        "All boundaries faces of some volume"
        dimtags = [(3, v) for v in volumes]
        return {s for _, s in gmsh.model.getBoundary(dimtags, oriented=False, combined=False)}
