"""Pictures of a `Cavity`: PNG renders of the gmsh model with legend, rulers and sizes.

Split from `cavity.py`. `RenderMixin` only draws; it relies on what `Cavity` provides
(`ports`, `options`, `_prepare`) and changes nothing in the design.

Developed by Maximiliano Gatto for QCT IFAE, Barcelona.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import gmsh
import numpy as np


class RenderMixin:
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
            dim = cls._group_dim(name)
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
        from .cavity import CavityOptions            # here, not at the top: cavity.py imports this module
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
