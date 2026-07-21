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
"""Circular Transmon Pocket Pocket

Developed by Maximiliano Gatto for Bariloche Quantum Circuits Group, CAB, CNEA, Argentina as a student of Balseiro Institute.


For any questions please contact:
    maxigatto.mg@gmail.com
"""

# -*- coding: utf-8 -*-
import copy
import numpy as np
from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import BaseQubit
from shapely.ops import unary_union

class CircTransmon(BaseQubit):
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

        # Pads (capacitor)
        pad_radius = '10um',
        pad_gap = '5um',

        # jj (mesoscopic: 0 or small junction: 1)
        jj_options=Dict(
            jj_width='5um',
            L_j = '40.7885nH',
            C_j = '0.72fF',
            export_mask = False,    # if True, draw a rectangle where the constriction will be made. If false, we keep a gap of jj_sim_gap in which we insert a jj element to make the simulation
            gds_cell_jj = 'gds_cell_jj',
            jj_sim_gap = '1um', # if we need a small junction, we can increase this gap
            ),
        
        # Coupled claws (2 claws allowed   )
        claw_a_options= Dict(
            make_claw=False,
            connector_location='90',  # 0 => 'west' arm, 90 => 'north' arm, 180 => 'east' arm
            claw_width='10um',
            claw_gap='6um',
            claw_length='30um',
            ground_spacing='5um',
            connector_length='20um',
            ),
        claw_b_options=Dict(
            make_claw=False,
            connector_location='0',  # 0 => 'est' arm, 90 => 'north' arm, 180 => 'west' arm
            claw_width='10um',
            claw_gap='6um',
            claw_length='30um',
            ground_spacing='5um',
            connector_length='20um',
            ),

        # Charge line
        charge_line=Dict(
            make_charge_line=False,
            location = '180', # 0 => 'west' arm, 90 => 'north' arm, 180 => 'east' arm. Must be different from claw_a and claw_b if they exist
            pad_radius='10um',
            pad_gap='5um', # gap between the pad and qubit
            ground_spacing='5um', # distance from the charge line to the ground
            line_length='100um',
            )
        )

    TOOLTIP = "Fluxonium pocket; modular: core, resonador y flux line."

    # ------------------------- main API -------------------------
    def make(self):
        # p = self.parse_options()

        # construct submodules
        geoms_core, pins_core = self._make_core()
        geoms_junc, pins_junc = self._make_junctions(geoms_core)
        geoms_coupl, pins_coupl = self._make_coupling_claws()
        geoms_fbl,  pins_fbl  = self._make_charge_line() if self.p.charge_line.make_charge_line else ({}, {})

        # merge all geometries and pins
        geoms = {**geoms_core, **(geoms_junc or {}), **(geoms_fbl or {}), **(geoms_coupl or {})}
        pins  = {**pins_core,  **(pins_junc or {}),  **(pins_fbl or {}), **(pins_coupl or {})}

        # create centered background
        background = self._make_background(geoms, pins)

        # apply global transform (centering, mirror, rotate, translate)
        geoms, pins, background = self._apply_global_transform(geoms, pins, background)

        # emit to QGeometry tables
        self._emit_qgeometry(geoms, background)
        self._emit_pins(pins)

    # ------------------------- Submodules -------------------------
    def _make_core(self):
        p = self.p
        # Pads
        pad = draw.Point(0, 0).buffer(p.pad_radius)
        
        return {'pads': pad}, {}
    
    
    def _annular_sector(self, r_in, r_out, theta_center_deg, span_deg, n=256):
        """ Returns a shapely Polygon of an annular sector.
        Args:
            r_in (float): Inner radius.
            r_out (float): Outer radius.
            theta_center_deg (float): Center angle in degrees (0=+x, 90=+y=up).
            span_deg (float): Angle span in degrees.
            n (int): Number of points to approximate the curves.
        Returns:
            shapely Polygon: The annular sector polygon.
        """
        t0 = np.deg2rad(theta_center_deg - span_deg/2.0)
        t1 = np.deg2rad(theta_center_deg + span_deg/2.0)
        if t1 < t0:  # manejar wrap en 0..2π
            t1 += 2*np.pi
            
        # create the corners if needed
        r_med = 0.5*(r_in + r_out)
        s = (r_out - r_in)/2  # corner radius

        # # rounded corners
        if span_deg < 360:
            t0 = t0 + s/r_med
            t1 = t1 - s/r_med
            corner1 = draw.Point(r_med*np.cos(t0), r_med*np.sin(t0)).buffer(s)
            corner2 = draw.Point(r_med*np.cos(t1), r_med*np.sin(t1)).buffer(s)
            corners = draw.union([corner1, corner2]).buffer(0)
        
        outer = [(r_out*np.cos(t), r_out*np.sin(t)) for t in np.linspace(t0, t1, n)]
        inner = [(r_in *np.cos(t), r_in *np.sin(t)) for t in np.linspace(t1, t0, n)]
        poly  = draw.Polygon(outer + inner).buffer(0)  # buffer(0) = sanea
        
        return draw.union([poly, corners]) if span_deg < 360 else poly
    
    def _make_claw(self, claw):
        if claw == 'a':
            p, claw = self.p, self.p.claw_a_options
        elif claw == 'b':
            p, claw = self.p, self.p.claw_b_options
        else:
            raise ValueError("claw must be 'a' or 'b'")
        
        r_in = p.pad_radius + claw.claw_gap
        r_out = r_in + claw.claw_width
        r_mid = 0.5*(r_in + r_out)
        
        span_deg = np.rad2deg(claw.claw_length / r_mid)
        
        theta_c = float(claw.connector_location) # ° (0=west, 90=north, 180=east)
        
        # make the claw
        claw_geom = self._annular_sector(r_in, r_out, theta_c, span_deg)
        
        # add rectangular segment to make the pin
        px_start, py_start = r_mid * np.cos(np.deg2rad(theta_c)), r_mid * np.sin(np.deg2rad(theta_c))
        con_length = claw.connector_length + claw.claw_width/2
        px_end, py_end = (r_mid + con_length) * np.cos(np.deg2rad(theta_c)), (r_mid + con_length) * np.sin(np.deg2rad(theta_c))
        
        pin = draw.LineString([(px_start, py_start), (px_end, py_end)])
        connector = pin.buffer(p.cpw_width/2, cap_style=2)
        
        return claw_geom, connector, pin
    
    def _make_coupling_claws(self):
        p, claw_a, claw_b = self.p, self.p.claw_a_options, self.p.claw_b_options
        
        # check the positions
        if bool(claw_a.make_claw) and bool(claw_b.make_claw):
            if claw_a.connector_location == claw_b.connector_location:
                raise ValueError("claw_a and claw_b cannot have the same connector_location")
        
        claw_a, connector_a, pin_a = self._make_claw('a') if claw_a.make_claw else (None, None, None)
        claw_b, connector_b, pin_b = self._make_claw('b') if claw_b.make_claw else (None, None, None)
        
        geoms = {}
        pins  = {}
        
        if claw_a is not None:
            geoms['claw_a'] = claw_a
            geoms['connector_a'] = connector_a
        if claw_b is not None:
            geoms['claw_b'] = claw_b
            geoms['connector_b'] = connector_b
        if pin_a is not None:
            pins['claw_a'] = pin_a
        if pin_b is not None:
            pins['claw_b'] = pin_b
            
        return geoms, pins
    
    def _make_junctions(self, geoms):
        p, jj = self.p, self.p.jj_options
        # create a rectangle where the constriction will be made
        length = p.pad_radius + p.pad_gap
        jj_area = draw.LineString([(0, 0), (0, -length)]).buffer(jj.jj_width/2, cap_style=2)
        # subtract the pads area to avoid overlaps
        mask = draw.subtract(draw.Point(0, 0).buffer(p.pad_radius + p.pad_gap), geoms['pads'])
        # intersection with a rectangle of width jj.jj_sim_gap
        jj_area = jj_area.intersection(mask)
        
        if not bool(jj.export_mask): # if false, we keep a gap of jj_sim_gap in which we insert a jj element to make the simulation
            jj_object = draw.LineString([(0, -p.pad_radius - p.pad_gap/2 + jj.jj_sim_gap/2), (0, -p.pad_radius - p.pad_gap/2 - jj.jj_sim_gap/2)])
            jj_area = draw.subtract(jj_area, jj_object.buffer(jj.jj_width, cap_style=2))
            
            return {'jj_area': jj_area, 'jj_object': jj_object}, {} # return jj_area and jj_object for simulations
        
        return {'jj_area': jj_area}, {} # if we need the mask, we export only the jj_area
    
    def _make_charge_line(self):
        """ Creates a charge line connected to the circular transmon.
        The charge line is a circular sector with a rectangular extension to make the pin.
        """
        p, cl = self.p, self.p.charge_line
        # check the positions
        if (p.claw_a_options.make_claw and cl.location == p.claw_a_options.connector_location) or (p.claw_b_options.make_claw and cl.location == p.claw_b_options.connector_location):
            raise ValueError("Charge line location must be different from claw_a and claw_b connector_location if they exist")
        # make the charge line
        r_pos = p.pad_radius + cl.pad_gap + cl.pad_radius
        angle = float(cl.location)  # ° (0=west, 90=north, 180=east)
        center = (r_pos * np.cos(np.deg2rad(angle)), r_pos * np.sin(np.deg2rad(angle)))
        
        pad = draw.Point(center).buffer(cl.pad_radius)
        
        # make the connector
        px_start, py_start = r_pos * np.cos(np.deg2rad(angle)), r_pos * np.sin(np.deg2rad(angle))
        px_end, py_end = (r_pos + cl.pad_radius + cl.line_length) * np.cos(np.deg2rad(angle)), (r_pos + cl.pad_radius + cl.line_length) * np.sin(np.deg2rad(angle))
        
        pin = draw.LineString([(px_start, py_start), (px_end, py_end)])
        
        return {'charge_line': pad, 'charge_line_connector': pin.buffer(p.cpw_width/2, cap_style=2)}, {'charge_line': pin}
    
    
    def _make_background(self, geoms, pins):
        p = self.p
        # create a circular background around the capacitor pad
        pocket = draw.Point(0, 0).buffer(p.pad_radius + p.pad_gap)
        
        # make the backgorund of the clamps if they exist
        if 'claw_a' in geoms:
            claw_a = self.p.claw_a_options
            pocket = draw.union([pocket, geoms['claw_a'].buffer(claw_a.ground_spacing)])
            # add the background of the connector
            connector_back_a = pins['claw_a'].buffer(p.cpw_width/2 + p.cpw_gap, cap_style=2)
            pocket = draw.union([pocket, connector_back_a])
            
        if 'claw_b' in geoms:
            claw_b = self.p.claw_b_options
            pocket = draw.union([pocket, geoms['claw_b'].buffer(claw_b.ground_spacing)])
            connector_back_b = pins['claw_b'].buffer(p.cpw_width/2 + p.cpw_gap, cap_style=2)
            pocket = draw.union([pocket, connector_back_b])
        
        # make the background of the charge line if it exists
        if 'charge_line' in geoms:
            cl = self.p.charge_line
            pocket = draw.union([pocket, geoms['charge_line'].buffer(cl.ground_spacing)])
            connector_back_cl = pins['charge_line'].buffer(p.cpw_width/2 + p.cpw_gap, cap_style=2)
            pocket = draw.union([pocket, connector_back_cl])
            
        return pocket
    
    # ---------------------------- Global transform -------------------------
    def _apply_global_transform(self, geoms, pins, background):
        """Apply global transformations: centering, mirror, rotate, translate.
        Args:
            geoms (dict): Dictionary of geometries.
            pins (dict): Dictionary of pin geometries.
            background (shapely Polygon): Background geometry.
        Returns:
            tuple: Transformed (geoms, pins, background).
        """
        all_objects = [*geoms.values(), *pins.values(), background]
        
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
        # TODO: modify depending on the geometries emmited
        p, jj = self.p, self.p.jj_options
        # background (pocket)
        self.add_qgeometry('poly', {'pocket': background}, subtract=True, chip=p.chip)

        # Pads, loop
        for name in ('pads', 'jj_area', 'charge_line', 'claw_a', 'claw_b'):
            if name in geoms and geoms[name] is not None:
                if name is 'claw_a':    # add connector to the claw
                    geoms[name] = draw.union([geoms[name], geoms['connector_a']]).buffer(0)
                if name is 'claw_b':
                    geoms[name] = draw.union([geoms[name], geoms['connector_b']]).buffer(0)
                # add the geometry
                if name is 'charge_line':    # add connector to the charge line
                    geoms[name] = draw.union([geoms[name], geoms['charge_line_connector']]).buffer(0)
                
                self.add_qgeometry('poly', {name: geoms[name]}, chip=p.chip)

        if 'jj_object' in geoms and geoms['jj_object'] is not None:
            self.add_qgeometry('junction', {'jj_object': geoms['jj_object']},
                                chip=p.chip,
                                width = jj.jj_width,
                                hfss_inductance = jj.L_j,
                                hfss_capacitance = jj.C_j)

    # ------------------------- Emit pins -------------------------
    def _emit_pins(self, pins):
        # add pins
        for name, ls in pins.items():
            self.add_pin(name, points=np.array(ls.coords), width=self.p.cpw_width, input_as_norm=True)
            