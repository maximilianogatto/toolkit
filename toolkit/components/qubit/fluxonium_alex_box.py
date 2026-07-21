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

# This class was created by Maximiliano Gatto (5 Jun. 2025) and is based on the class of Figen YILMAZ, Christian Kraglund Andersen for FloxoniumPocket. This class represent the design that Alex drew.

"""Fluxonium Pocket (Alex)"""

from operator import length_hint
import numpy as np
from qiskit_metal import draw, Dict
from math import *
from qiskit_metal.draw.basic import buffer
from qiskit_metal.qlibrary.core import BaseQubit
import copy
from shapely.geometry import GeometryCollection

class AlexFluxoniumBox(BaseQubit):
    """The base `AlexFluxonium` class.

    Inherits `BaseQubit` class.

    Description:

    """

    component_metadata = Dict(short_name='FluxoniumPocket',
                              _qgeometry_table_path='True',
                              _qgeometry_table_poly='True',
                              _qgeometry_table_junction='True',
                             )
    """Component metadata"""

    
    default_options = Dict(
            #general
            pos_x='0um',
            pos_y='0um',
            orientation=0,
            chip='main',
            #pocket
            pocket_width='300um',
            pocket_height='200um',
            
            # pad 
            pad_position='right',
            pad_width_max='50um',
            pad_width_min='10um',
            pad_height='10um',
            # JJ
            jj_width='5um',
            jj_height='5um',
            L_j = '40.7885nH',
            C_j = '0.72fF',
            # JJ array
            jj_array_width = '4um',
            jj_array_lenght = '200um',
            jj_array_gnd_gap = '20um',
            jj_array_min_gap = '2um',
            jj_array_start_gap = '10um',
            jj_array_start_pad_height = '6um',
            jj_array_start_pad_width = '6um',
            L_jj = '218.823nH',
            # C_jj = '0fF',
            gds_cell_jj_array = 'gds_cell_jj_array',
            # READOUT
            readout_line_options=Dict(
                make_readout=True,
                pad_sep='10um',
                pad_gap='5um',
                pad_width = '40um',
                pad_height = '20um',
                cpw_width='cpw_width',
                cpw_gap='cpw_gap'
        )
    )

    # component_metadata = Dict(short_name='component',
    #                          _qgeometry_table_poly='True')
    component_metadata = Dict(short_name='FluxoniumPocket',
                              _qgeometry_table_path='True',
                              _qgeometry_table_poly='True',
                              _qgeometry_table_junction='True',
                             )
    
    TOOLTIP = """The base `FluxoniumPocket` class."""

    def make(self):
        """Define the way the options are turned into QGeometry.

        The make function implements the logic that creates the geometry
        (poly, path, etc.) from the qcomponent.options dictionary of
        parameters, and the adds them to the design, using
        qcomponent.add_qgeometry(...), adding in extra needed
        information, such as layer, subtract, etc.
        """
        self.make_pocket()
       
        # if self.p.flux_bias_line_options.make_fbl == True:
        #     self.make_flux_bias_line()
        # if self.p.charge_line_options.make_cl == True:
        #     self.make_charge_line()
        if self.p.readout_line_options.make_readout == True:
            self.make_readout_line()
        
    def make_pocket(self):
        ########### SCARAR EL LEFT RIGHT Y HACER UN ESPEJADO ###########
        
        """Makes standard fluxonium in a pocket."""
        # self.p allows us to directly access parsed values (string -> numbers) form the user option
        p = self.parse_options()
        pr = self.p.readout_line_options # parser on readout line options
        #General
        pad_position = p.pad_position
        
        #Pocket
        pocket_width = p.pocket_width
        pocket_height = p.pocket_height
        
        # Pads
        pad_width_max = p.pad_width_max
        pad_width_min = p.pad_width_min
        pad_height = p.pad_height
        # Draw the pocket
        pocket = draw.rectangle(pocket_width, pocket_height, 0, 0)
        
        # --------- Draw capacitor pads and the lines to connect with JJ
        pa_b = (-pocket_width/2 + 3*pocket_width/4, -pocket_height/2)
        pb_b = (pa_b[0] - (pad_width_max - pad_width_min)/2, pa_b[1] + pad_height)
        pc_b = (pb_b[0] - pad_width_min, pb_b[1])
        pd_b = (pa_b[0] - pad_width_max, pa_b[1])


        pad_bottom = draw.Polygon([pa_b, pb_b, pc_b, pd_b]) # bottom pad of capacitor
        dist_between = pocket_height - 2*abs(pb_b[1]-pa_b[1]) - p.jj_height  # obtain the distance between the pads, it is 
        width_height = (pad_width_min, dist_between/2)
        center_bottom = (pc_b[0] + pad_width_min/2, pc_b[1] + dist_between/4)
        
        connect_bottom = draw.rectangle(*width_height, *center_bottom)  # rectangle to connect the bottom pad with the JJ
        # Add the pad for JJ array
        pad_bottom = draw.union([pad_bottom, connect_bottom]) # union the bottom pad with the rectangle to connect with JJ and the pad for JJ array
        # copy, rotate, translate and mirror the bottom pad to the upper side of the pocket
        pad_upper = copy.deepcopy(pad_bottom) 
        pad_upper = draw.rotate(pad_upper, 180, origin=(pa_b[0]-pad_width_max/2, 0))  # rotate the pad to the upper side 
        # pad_upper = draw.scale(pad_upper, xfact=-1, yfact=1)
        
        center_sp_bottom = (pc_b[0] - p.jj_array_start_pad_width/2, 
                            pc_b[1] + p.jj_array_start_gap + p.jj_array_start_pad_height/2)
        start_jj_pad = draw.rectangle(p.jj_array_start_pad_width, p.jj_array_start_pad_height, *center_sp_bottom)
        end_jj_pad = copy.deepcopy(start_jj_pad)  # create a copy of the start pad to use it as the end pad
        end_jj_pad = draw.translate(end_jj_pad, 0, dist_between - 2*p.jj_array_start_gap + p.jj_height - p.jj_array_start_pad_height)  # translate the end pad to the end of the JJ array
        
        pad_bottom = draw.union([pad_bottom, start_jj_pad])  # union the upper pad with the start and end pads for the JJ array
        pad_bottom = draw.union([pad_bottom, end_jj_pad])  # union the upper pad with the start and end pads for the JJ array
        # Make the pad to connect readout
        
        pa_background = (pocket_width/4 - pad_width_max/2, pocket_height/2 + pr.pad_height/2 + pr.pad_gap/2)
        background_readout = draw.rectangle(pr.pad_width + 2*pr.pad_gap, pr.pad_height + pr.pad_gap, *pa_background)
        pocket = draw.union([pocket, background_readout])

        pa = (pocket_width/4 - pad_width_max/2, pocket_height/2 + pr.pad_height/2)
        readout_pad = draw.rectangle(pr.pad_width, pr.pad_height, *pa)
        
        pad_upper = draw.union([pad_upper, readout_pad])
        
        pads = draw.union([pad_bottom, pad_upper])  # union the bottom and upper pads
        
        # --------- Draw the Josephson Junction
        start_jj = (pa_b[0]-pad_width_max/2, pb_b[1]+dist_between/2)
        end_jj = (pa_b[0]-pad_width_max/2, pb_b[1]+dist_between/2 + p.jj_height)
        # The Josephson junction is a line that connects the center of the bottom pad with the center of the upper pad
        
        jj_object = draw.LineString([start_jj, end_jj])  # JJ must be a simple line and add a junct geometry
        
        prueba_linea = draw.LineString([(0,0), (1, 0), (1,1), (0,1)])
        
        print(type(prueba_linea))
        
        # # ------- JJ array 
        if p.jj_array_start_pad_width < p.jj_array_width/2:
            raise ValueError(f"The JJ array start pad width {p.jj_array_start_pad_width} is too small for the JJ array width {p.jj_array_width}. Please increase the JJ array start pad width.")
        
        # JJ array is conected after the jj_array_start_gap, which is the distance between the pad and de begining of the JJ array
        # print(f'd: {abs(pc_b[1] - pc_u[1])}')
        d = dist_between + p.jj_height - 2*p.jj_array_start_gap - p.jj_array_start_pad_height    # distance between the pads, it is the distance that we need to place the JJ array in the pocket
        D = abs(connect_bottom.centroid.x - pad_width_min/2 + pocket_width/2) 
        # print(f'D: {D}, type{type(D)}')
        # D = D - p.jj_array_start_gap  # distance from the center of the upper rectangle to the pocket, it is the distance that we need to place the JJ array in the pocket
        D = D - p.jj_array_start_pad_width - p.jj_array_gnd_gap # distance from the center of the upper rectangle to the pocket, it is the distance that we need to place the JJ array in the pocket
        # print(f"Distance between pads: {d}, Distance from the center of the upper rectangle to the pocket: {D}")
        
        
        N = 2   # we start from the minimum number of segments, which is 2.
        epsilon = 0.0001 # this is a small value to avoid numerical errors
        gamma, l = 0, 0  # initialize the gap between the segments and the length of each segment
        
        # here we obtain de number of segments, the length of each segment and the gap between the segments
        while(True):
            if p.jj_array_lenght < d:
                raise ValueError("The length of the JJ array is too small for the distance between the pads.")
            
            gamma = (d - (N-1)*p.jj_array_width) / (N - 1)  # the gap between the segments
            l = (p.jj_array_lenght - (N - 1) * gamma) / N  # the length of each segment 
            # Disclaimer: I don't add the gap between the segments to the length of the JJ array, because I suppose that it's too small compared to the length of the segments. 
            
            if gamma < p.jj_array_min_gap:
                raise ValueError(f"The gap between is less than the minimun gap {p.jj_array_min_gap} mm. You can reduce this gap by it is not recommended.")
            
            if l < 0 or gamma < 0:
                raise ValueError("The length of the JJ array is too small for the number of segments.")
            if l < p.jj_array_width:
                print(f"Warning: The length of the JJ array is too small for the number of segments. The length will be set to {l}.")
                break
            if l > p.jj_array_lenght:
                raise ValueError("The length of the JJ array is too long for the number of segments. Please increase the length of the JJ array or decrease the number of segments.")
            
            if l < D - p.jj_array_gnd_gap and gamma < d + epsilon:
                # we can place the segments in the pocket
                break
            
            if N > d/p.jj_array_width:  # this is the maximum number of segments that can be placed in the pocket
                raise ValueError("The number of segments is too high for the pocket size. Please increase the pocket size or decrease the array length.")
            
            N += 2 # we increase the number of segments by 2, because we need to have an even number of segments to make the array symmetric
        # print(f"Number of segments: {N}, Length of each segment: {l}, Gap between segments: {gamma}")
        
        # Now we can create the JJ array, which is a list of segments

        jj_array = draw.LineString([(start_jj_pad.centroid.x - p.jj_array_start_pad_width/2 + p.jj_array_width/2, 
                                    start_jj_pad.centroid.y + p.jj_array_start_pad_height/2), 
                                    (end_jj_pad.centroid.x - p.jj_array_start_pad_width/2 + p.jj_array_width/2, 
                                    end_jj_pad.centroid.y- p.jj_array_start_pad_height/2)])
        
        # jj_array_coords_h = []
        
        # # store pair of points in each iteration
        # jj_array_coords_h.append((pc_b[0] - p.jj_array_start_pad_width, pc_b[1] + p.jj_array_start_gap + p.jj_array_start_pad_height/2))
        # jj_array_coords_h.append((pc_b[0] - p.jj_array_start_pad_width - l, pc_b[1] + p.jj_array_start_gap + p.jj_array_start_pad_height/2))
        
        # print(jj_array_coords_h)
        # # points to store the new
        # ps_old = jj_array_coords_h[0]
        # pe_old = jj_array_coords_h[1]
        
        # for i in range(2, 2*N, 2):   # we start from 1 because the first segment is already added
        #     # translate the points
        #     pe_new = (ps_old[0], ps_old[1] +gamma + p.jj_array_width)
        #     ps_new = (pe_old[0], pe_old[1] + gamma + p.jj_array_width)

        #     jj_array_coords_h.append(ps_new)
        #     jj_array_coords_h.append(pe_new)
            
        #     ps_old = ps_new
        #     pe_old = pe_new

        # jj_array = draw.LineString(jj_array_coords_h)

        
        # define the first horizontal segment
        # jj_array = []
        # segment_h = draw.LineString([(pc_b[0] - p.jj_array_start_pad_width, pc_b[1] + p.jj_array_start_gap + p.jj_array_start_pad_height/2) , (pc_b[0] - p.jj_array_start_pad_width - l, pc_b[1] + p.jj_array_start_gap + p.jj_array_start_pad_height/2)])
        
        # jj_array.append(segment_h)  # add the first segment to the array
        # s_old = copy.deepcopy(segment_h)  # create a copy of the segment to use it as the next segment
        # # print(f"Segment 0: {s_old.bounds}")  # print the bounds of the segment
        # # -- horizontal segments
        # for i in range(1, N):   # we start from 1 because the first segment is already added
        #     s = draw.translate(s_old, 0, gamma + p.jj_array_width)  # translate the segment to the next position
        #     jj_array.append(s)
        #     s_old = copy.deepcopy(s)  # create a copy of the segment to use it as the next segment
            
        #     # print(f"Segment {i}: {s.bounds}")  # print the bounds of the segment
            
        # # -- vertical segments
        # jj_array_v = []  # create a list to store the vertical segments
        
        # # print(jj_array[0].bounds)
        
        # for i in range(0, N-1):
        #     coord_start, coord_end = None, None
        #     if i % 2 == 0:  # if the number of segments is even, we need to add a vertical at left
        #         coord_start = (jj_array[i].bounds[0]+p.jj_array_width/2, jj_array[i].bounds[1])
        #         coord_end = (jj_array[i+1].bounds[0]+p.jj_array_width/2, jj_array[i+1].bounds[1])
        #     elif i % 2 == 1:  # if the number of segments is odd, we need to add a vertical at right
        #         coord_start = (jj_array[i].bounds[2]-p.jj_array_width/2, jj_array[i].bounds[3])
        #         coord_end = (jj_array[i+1].bounds[2]-p.jj_array_width/2, jj_array[i+1].bounds[3])
        #     else:
        #         raise ValueError("The number of segments must be even or odd.")
        #     # print(f"Vertical segment {i}: start {coord_start}, end {coord_end}")  # print the coordinates of the vertical segment
            
        #     s = draw.LineString([coord_start, coord_end])  # create a vertical segment between the two horizontal segments
        #     jj_array_v.append(s)  # add the vertical segment to the list
        
        # jj_array = draw.union(jj_array_v + jj_array)  # union the vertical segments with the horizontal segments
        
        # Put all the objects in alist to rotate and translate them together
        all_objects = [pocket, pads, jj_object, jj_array]
        
        # Translate and rotate the object
        if pad_position == 'left': all_objects = draw.scale(all_objects, xfact=-1, yfact=1, origin=(0, 0))  # mirror the object to the left side of the pocket
        all_objects = draw.rotate(all_objects, p.orientation, origin=(0,0))
        all_objects = draw.translate(all_objects, p.pos_x, p.pos_y)

        pocket, pads, jj_object, jj_array = all_objects
        
        # There are 3 geometries that we can add:
        #   - poly: polygon      # Used for solid shapes like pads or pockets (e.g., rectangles, polygons).
        #   - path:              # Used for thin traces or wires, defined by a path with width (not used in this example).
        #   - junction:          # Used specifically for Josephson junctions, must be a LineString and can have physical properties like width, inductance, and capacitance.
        
        self.add_qgeometry('poly', {'pocket': pocket}, subtract=True, chip=p.chip)
        # self.add_qgeometry('poly', {'pads': pads}, chip=p.chip)
        
        # self.add_qgeometry('junction', {'jj_object': jj_object},
        #                     width=p.jj_width,
        #                     hfss_inductance = p.L_j,
        #                     hfss_capacitance = p.C_j,
        #                     chip=p.chip)
        
        # # Add the JJ array as a junction geometry
        # self.add_qgeometry('junction', {'jj_array': jj_array},
        #                     width=p.jj_array_width,
        #                     hfss_inductance = p.L_jj,
        #                     hfss_capacitance = p.C_jj,
        #                     chip=p.chip,
        #                     gds_cell_name=p.gds_cell_jj_array)
    
    def make_readout_line(self):
        """Creates the readout line for the fluxonium pocket."""
        p = self.p
        pr = self.p.readout_line_options # parser on readout line options
        
        pad_sep = pr.pad_sep
        pad_width = pr.pad_width
        pad_height = pr.pad_height
        pad_gap = pr.pad_gap
        cpw_width = pr.cpw_width
        cpw_gap = pr.cpw_gap
        
        # make readout pad
        pc_background = (p.pocket_width/4 - p.pad_width_max/2, p.pocket_height/2 + 3*pad_height/2 + pad_sep)
        background_readout = draw.rectangle(pad_width + 2*pad_gap, pad_height + 2*pad_gap, *pc_background)

        pc = (p.pocket_width/4 - p.pad_width_max/2, p.pocket_height/2 + 3*pad_height/2 + pad_sep)   # point of the capacitor
        pad_redout = draw.rectangle(pad_width, pad_height, *pc)
        pad_connect = draw.rectangle(cpw_width, pad_gap, pc[0], pc[1] + pad_height/2 + pad_gap/2)  # rectangle to connect the readout pad with the CPW wire
        
        pad_redout = draw.union([pad_redout, pad_connect])  # union the readout pad with the rectangle to connect with CPW wire
        
        # Readout Line CPW wire
        port_line = draw.LineString([(pc[0] - cpw_width/2, pc_background[1]+pad_height/2+pad_gap), (pc[0] + cpw_width/2, pc_background[1]+pad_height/2+pad_gap)])
        
        all_objects = [background_readout, pad_redout, port_line]
        
        # Translate and rotate the object
        if p.pad_position == 'left': all_objects = draw.scale(all_objects, xfact=-1, yfact=1, origin=(0, 0))  # mirror the object to the left side of the pocket
        all_objects = draw.rotate(all_objects, p.orientation, origin=(0,0))
        all_objects = draw.translate(all_objects, p.pos_x, p.pos_y)
        
        background_readout, pad_redout, port_line = all_objects
        
        # Add geometry
        self.add_qgeometry('poly', {'readout_beckground': 
                                    background_readout}, subtract=True)
        self.add_qgeometry('poly', {'readout_pad': 
                                    pad_redout})
        self.add_pin('readout_line', port_line.coords, cpw_width)
    
    def make_charge_line(self):
        """Creates the charge line for the fluxonium pocket."""
        # This function is not implemented yet, but it should create the charge line
        # using the options defined in the default_options dictionary.
        pass
    def make_flux_bias_line(self):
        """Creates the flux bias line for the fluxonium pocket."""
        # This function is not implemented yet, but it should create the flux bias line
        # using the options defined in the default_options dictionary.
        pass
    
