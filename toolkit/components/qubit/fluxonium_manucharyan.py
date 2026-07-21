# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Fluxonium Pocket (Manucharyan design adquired)

Developed by Maximiliano Gatto for Bariloche Quantum Circuits Group, CAB, CNEA, Argentina as a student of Balseiro Institute.

This qubit was designed and fabricated by Manucharyan's group in EPFL Lausanne, Switzerland. Bariloche Quantum Circuits Group adquired on chip with this qubit design. This class adapts the design to Qiskit Metal, that allows to modify some parameters of the design and perform simulations to extract the qubit parameters.

For any questions please contact:
    maxigatto.mg@gmail.com
"""

# -*- coding: utf-8 -*-
import copy
import numpy as np
from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import BaseQubit
from shapely.ops import unary_union

class FluxoniumPocket(BaseQubit):
    """Fluxonium tipo 'pocket' con resonador y/o flux bias line opcionales."""

    component_metadata = Dict(
        short_name='FluxoniumPocket',
        _qgeometry_table_path='True',
        _qgeometry_table_poly='True',
        _qgeometry_table_junction='True',
    )

    default_options = Dict(
        # Position and orientation
        pos_x='0um', pos_y='0um', orientation=0, chip='main',
        cpw_width='10um', cpw_gap='6um',

        # Pocket (background)
        back_gap_left = '10um',    # extra left margin
        back_gap_right = '10um',   # extra right margin
        back_gap_top = '10um',     # extra top margin
        back_gap_bottom = '10um',  # extra bottom margin

        # Pads (capacitor)
        pad_width_max='80um',
        pad_width_min='4um',
        pad_height='100um',
        pad_gap='45um',
        # Loop
        loop_position='right',          # 'right' o 'left' (mirror)
        loop_height='18um',
        loop_width='70um',

        # central JJ
        jj_width='5um',
        jj_height='5um',
        L_j='40.7885nH',
        C_j='0.72fF',

        # JJ arrays loop
        jj_array_width='4um',
        L_jj='218.823nH',
        gds_cell_jj_array='gds_cell_jj_array',

        # Coupled resonator
        resonator_options=Dict(
            make_resonator=True,
            resonator_length='100um',
            resonator_width='30um',
            resonator_gap='16um',
            qubit_res_gap='40um',
            jj_dist='35um',
            jj_array_width='4um',
            L_res='100nH',
        ),

        # Flux bias line
        flux_line_options=Dict(
            make_flux_bias_line=True,
            total_length='1.5mm',
            width='8um',
            sep_lines='5.2um',
            wait_length='100um',
            adjust_length='50um',
            pad_width='80um',
            pad_height='100um',
            pad_gap='100um',
        )
    )

    TOOLTIP = "Fluxonium pocket; modular: core, resonador y flux line."

    # ------------------------- main API -------------------------
    def make(self):
        p = self.parse_options()

        # construct submodules
        geoms_core, pins_core = self._make_core()
        geoms_res,  pins_res  = self._make_resonator() if self.p.resonator_options.make_resonator else ({}, {})
        geoms_fbl,  pins_fbl  = self._make_flux_line() if self.p.flux_line_options.make_flux_bias_line else ({}, {})

        # merge all geometries and pins
        geoms = {**geoms_core, **(geoms_res or {}), **(geoms_fbl or {})}
        pins  = {**pins_core,  **(pins_res or {}),  **(pins_fbl or {})}

        # create centered background
        background = self._make_centered_background(geoms)

        # apply global transform (centering, mirror, rotate, translate)
        geoms, pins, background = self._apply_global_transform(geoms, pins, background)

        # emit to QGeometry tables
        self._emit_qgeometry(geoms, background)
        self._emit_pins(pins)

    # ------------------------- Submodules -------------------------
    def _make_core(self):
        """Triangle pads + loop + small JJ + JJ arrays"""
        p = self.p

        # lower pad (triangle)
        pa = (-p.pad_width_max/2, 0)
        pb = ( p.pad_width_max/2, 0)
        pc = ( p.pad_width_min/2, p.pad_height)
        pd = (-p.pad_width_min/2, p.pad_height)
        pad_b = draw.Polygon([pa, pb, pc, pd])

        # Add rectangles to close the triangle (save space for the juction)
        rec_loop = draw.rectangle(
            p.pad_width_min,
            p.pad_gap/2 - p.jj_height/2,
            0,
            p.pad_height + (p.pad_gap/2 - p.jj_height/2)/2
        )
        pad_b = draw.union([pad_b, rec_loop])

        # Draw the top pad
        pad_t = copy.deepcopy(pad_b)
        pad_t = draw.rotate(pad_t, 180, origin=(0, p.pad_height/2))
        pad_t = draw.translate(pad_t, 0, p.pad_gap + p.pad_height)

        pads = draw.union([pad_b, pad_t])

        # JJ central (LineString)
        y_jj_center = p.pad_height + p.pad_gap/2
        jj_object = draw.LineString([
            (0, y_jj_center - p.jj_height/2),
            (0, y_jj_center + p.jj_height/2)
        ])

        # lateral JJ array (inductors)
        jj_array_1 = draw.LineString([
            ( p.pad_width_min/2, y_jj_center + p.loop_height/2 + p.jj_array_width/2),
            ( p.loop_width,      y_jj_center + p.loop_height/2 + p.jj_array_width/2)
        ])
        jj_array_2 = draw.LineString([
            ( p.pad_width_min/2, y_jj_center - p.loop_height/2 - p.jj_array_width/2),
            ( p.loop_width,      y_jj_center - p.loop_height/2 - p.jj_array_width/2)
        ])

        # Rectangle to close the loop
        close_loop = draw.rectangle(
            p.pad_width_min, p.loop_height + 2*p.jj_array_width,
            p.loop_width + p.pad_width_min/2, y_jj_center
        )

        geoms = {
            'pads': pads,
            'close_loop': close_loop,
            'jj_object': jj_object,
            'jj_array_1': jj_array_1,
            'jj_array_2': jj_array_2,
        }
        pins = {}   # (core does not have ports)

        return geoms, pins

    def _make_resonator(self):
        """Readout resonator."""
        p, r = self.p, self.p.resonator_options

        # Create the resonator as a rectangle with rounded corners
        dist = r.resonator_length - r.resonator_width
        y_res = 2*p.pad_height + p.pad_gap + r.qubit_res_gap + r.resonator_width/2

        # Resonator pads
        bound_1 = draw.Point((-dist/2, y_res)).buffer(r.resonator_width/2, resolution=16)
        bound_2 = draw.translate(copy.deepcopy(bound_1), dist, 0)
        close_res = draw.rectangle(dist, r.resonator_width, 0, y_res)
        pad_1 = draw.union([bound_1, bound_2, close_res])

        pad_2 = draw.translate(copy.deepcopy(pad_1), 0, r.resonator_gap + r.resonator_width)
        resonator = draw.union([pad_1, pad_2])
        
        # Add pads for JJ array
        jj_pad_bottom = draw.rectangle(r.jj_dist + r.resonator_width/2, r.jj_array_width, r.resonator_length/2 + r.jj_dist/2 - r.resonator_width/4, y_res)
        jj_pad_top = draw.translate(copy.deepcopy(jj_pad_bottom), 0, r.resonator_gap + r.resonator_width)
        # resonator JJ array
        jj_x = r.resonator_length/2 + r.jj_dist - r.jj_array_width/2
        jj_y = y_res
        jj_array = draw.LineString([
            (jj_x, jj_y + r.jj_array_width/2),
            (jj_x, jj_y + r.resonator_gap + r.resonator_width - r.jj_array_width/2)
        ])
        # Merge all resonator pads
        resonator = draw.union([resonator, jj_pad_bottom, jj_pad_top])

        geoms = {
            'resonator': resonator,
            'jj_array': jj_array
        }
        pins = {}  # (no ports in resonator)

        return geoms, pins
    def _make_flux_line(self):
        p, f = self.p, self.p.flux_line_options
        yc = p.pad_height + p.pad_gap/2
        # eps = 1e-6  # small fudge factor to ensure geometry union

        # reference coords
        x_loop_end = p.loop_width + p.pad_width_min/2           # x-cord end of the loop
        y0 = yc + f.sep_lines/2                                 # y-cord of the center of the flux line

        # thin line
        thin = draw.rectangle(
            f.wait_length, p.jj_array_width,
            p.loop_width + p.pad_width_min + f.wait_length/2,
            yc + f.sep_lines/2 + p.jj_array_width/2
        )

        # Adjust pad (polygon to adjust loop width to flux line width)
        pa = (x_loop_end + f.wait_length, y0)                  # lower left corner
        pb = (pa[0] + f.adjust_length, pa[1])                  # lower right corner
        pc = (pb[0], pa[1] + f.width)                          # upper right corner
        pd = (pa[0], pa[1] + p.jj_array_width)           # upper left corner (+ to ensure union)
        adjust = draw.Polygon([pa, pb, pc, pd])

        # straight line to connect to current pad
        dist_rest = f.total_length - f.wait_length - f.adjust_length
        rest = draw.rectangle(
            dist_rest, f.width,                     # + para solapar
            pb[0] + (dist_rest)/2,
            pa[1] + f.width/2
        )

        # current pad and connection to rest line
        pad = draw.rectangle(
            f.pad_width, f.pad_height,
            pb[0] + dist_rest + f.pad_width/2,
            pa[1] + f.pad_height/2 + f.pad_gap/2 - f.sep_lines/2
        )
        connect = draw.rectangle(
            f.width, f.pad_gap/2 - f.sep_lines/2,
            pb[0] + dist_rest + f.width/2,
            pa[1] + f.pad_gap/4 - f.sep_lines/4
        )

        # Merge all the components (buffer(0) to clean geometry)
        flux_top = draw.union([thin, adjust, rest, connect, pad]).buffer(0)

        # Create the bottom part by mirroring
        flux_bot = draw.scale(copy.deepcopy(flux_top), xfact=1, yfact=-1, origin=(0, yc))
        flux_all = draw.union([flux_top, flux_bot]).buffer(0)

        # Add current pins at the pads (vertical tangent to the edges)
        pin_in = draw.LineString([
            (pb[0] + dist_rest + f.pad_width, pa[1] + f.pad_gap/2 - f.sep_lines/2),
            (pb[0] + dist_rest + f.pad_width, pa[1] + f.pad_gap/2 - f.sep_lines/2 + f.pad_height)
        ])
        pin_out = draw.LineString([
            (pb[0] + dist_rest + f.pad_width, pa[1] - f.pad_gap/2 + f.sep_lines/2),
            (pb[0] + dist_rest + f.pad_width, pa[1] - f.pad_gap/2 + f.sep_lines/2 - f.pad_height)
        ])

        geoms = {'flux_bias_line': flux_all}
        pins  = {'flux_in': pin_in, 'flux_out': pin_out}
        return geoms, pins
    
    # ------------------------- Centering and transforms -------------------------
    def _make_centered_background(self, geoms: dict):
        """Create a centered background/pocket based on the geometries provided. If pocket_margin is set, use it as margin around the geometries. Otherwise use fixed background_width and background_height."""
        
        p = self.p
        union = unary_union([g for g in geoms.values() if g is not None])
        xmin, ymin, xmax, ymax = union.bounds
        
        # add gaps
        xmin -= p.back_gap_left
        xmax += p.back_gap_right
        ymin -= p.back_gap_bottom
        ymax += p.back_gap_top
        
        # create background rectangle
        if p.loop_position == 'left': # if left, extend background to right, so then when we apply the global transformation, it will be at the left
            xmin += p.back_gap_left - p.back_gap_right
            xmax += -p.back_gap_right + p.back_gap_left

        
        bg = draw.rectangle(xmax - xmin, ymax - ymin, (xmin + xmax)/2, (ymin + ymax)/2)
        
        return bg
    # ---------------------------- Global transform -------------------------
    def _apply_global_transform(self, geoms, pins, background):
        all_objects = [*geoms.values(), *pins.values(), background]
        
        # mirror the loop if needed
        if self.p.loop_position == 'left':
            all_objects = draw.scale(all_objects, xfact=-1, yfact=1, origin=(0, 0))
        
        # rotation
        ang = self.p.orientation
        all_objects = draw.rotate(all_objects, ang, origin=(0, 0))
        
        # translation
        all_objects = draw.translate(all_objects, self.p.pos_x, self.p.pos_y)
        
        # unpack and reconstruct dictionaries
        num_geoms = len(geoms)
        geoms_list = all_objects[:num_geoms]
        pins_list = all_objects[num_geoms:-1]
        background = all_objects[-1]
        
        geoms = {k: g for k, g in zip(geoms.keys(), geoms_list)}
        pins = {n: g for n, g in zip(pins.keys(), pins_list)}
        
        return geoms, pins, background

    # ------------------------- Emit to QGeometry -------------------------
    def _emit_qgeometry(self, geoms, background):
        p, r = self.p, self.p.resonator_options
        # background (pocket)
        self.add_qgeometry('poly', {'pocket': background}, subtract=True, chip=p.chip)

        # Pads, loop
        for name in ('pads', 'close_loop', 'resonator', 'flux_bias_line'):
            if name in geoms and geoms[name] is not None:
                self.add_qgeometry('poly', {name: geoms[name]}, chip=p.chip)

        # JJ central
        if 'jj_object' in geoms:
            self.add_qgeometry('junction', {'jj_object': geoms['jj_object']},
                                width=p.jj_width,
                                hfss_inductance=p.L_j,
                                hfss_capacitance=p.C_j,
                                chip=p.chip)

        # JJ array (qubit)
        for jname in ('jj_array_1', 'jj_array_2'):
            if jname in geoms:
                self.add_qgeometry('junction', {jname: geoms[jname]},
                                    width=p.jj_array_width,
                                    hfss_inductance=p.L_jj,
                                    chip=p.chip,
                                    gds_cell_name=p.gds_cell_jj_array)

        # JJ array (resonator)
        if 'jj_array' in geoms:
            self.add_qgeometry('junction', {'jj_array': geoms['jj_array']},
                                width=r.jj_array_width,
                                hfss_inductance=r.L_res,
                                chip=p.chip,
                                gds_cell_name=p.gds_cell_jj_array)
    # ------------------------- Emit pins -------------------------
    def _emit_pins(self, pins):
        # add pins
        for name, ls in pins.items():
            self.add_pin(name, points=np.array(ls.coords), width=self.p.cpw_width)
            