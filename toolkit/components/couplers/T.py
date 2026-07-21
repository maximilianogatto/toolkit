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
"""T Coupler.

Developed by Maximiliano Gatto for Bariloche Quantum Circuits Group, CAB, CNEA, Argentina as a student of Balseiro Institute.

For any questions please contact:
    maxigatto.mg@gmail.com
"""

from qiskit_metal import draw, Dict
from qiskit_metal.qlibrary.core import QComponent
import numpy as np


class T(QComponent):
    """Generate a T shaped coupler with 1 port. The rest of the T is open to ground. All the component has the same gap to the ground plane. Each segment of the T can have different width and length. (0,0) represents the center position of the component.

    Options:
        pos_x: x position of the component.
        pos_y: y position of the component.
        orientation: orientation angle of the component.
        prime_width: width of the primary segment of the T.
        prime_length: length of the primary segment of the T.
        second_width: width of the secondary segment of the T.
        second_length: length of the secondary segment of the T.
        gap: gap between the coupler and the ground plane.
        layer: metal layer to draw the component.
        
    """
    component_metadata = Dict(short_name='cpw', _qgeometry_table_path='True')
    """Component metadata"""

    #Currently setting the primary CPW length based on the coupling_length
    #May want it to be it's own value that the user can control?
    default_options = Dict( pos_x='0um',
                            pos_y='0um',
                            orientation='0',
                            prime_width='10um',
                            prime_length='100um',
                            second_width='10um',
                            second_length='50um',
                            gap='6um')
    """Default connector options"""

    TOOLTIP = """This component creates a T shaped coupler with 1 port. The rest of the T is open to ground. All the component has the same gap to the ground plane. Each segment of the T can have different width and length."""

    def make(self):
        """Build the component."""
        p = self.p
        # Segments
        prime_segment = draw.rectangle(p.prime_width, p.prime_length, 0, p.prime_length/2)
        second_segment = draw.rectangle(p.second_length, p.second_width, 0, 0)
        
        segment = draw.union(prime_segment, second_segment)

        #Secondary segment
        second_segment = draw.LineString([[-p.second_length/2, 0], [p.second_length/2, 0]])
        
        # Background subtraction gap
        background_1 = draw.rectangle(p.prime_width + 2*p.gap, p.prime_length + p.gap, 0, p.prime_length/2 - p.gap/2)
        background_2 = draw.rectangle(p.second_length + 2*p.gap, p.second_width + 2*p.gap, 0, 0)
        background = draw.union(background_1, background_2)
        
        # Add pin in top of T
        # pin = draw.LineString([(-p.prime_width/2, p.prime_length), (p.prime_width/2, p.prime_length)])
        pin_half = p.prime_width/2  
        pin = draw.LineString([(0, p.prime_length - pin_half),
                            (0, p.prime_length)])

        # #Rotate and Translate
        total_items = [segment, background, pin]
        total_items = draw.rotate(total_items, p.orientation, origin=(0, 0))
        total_items = draw.translate(total_items, p.pos_x, p.pos_y)
        [segment, background, pin] = total_items

        #Add to qgeometry tables
        self.add_qgeometry('poly', {'background': background},
                           subtract=True,
                           layer=p.layer)
        
        self.add_qgeometry('poly', {'segment': segment},
                           layer=p.layer)
        
        self.add_pin('prime_end',
                     points=np.array(pin.coords),
                     width=p.prime_width,
                    input_as_norm=True)