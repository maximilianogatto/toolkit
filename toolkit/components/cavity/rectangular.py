from dataclasses import dataclass
from .cavity import Cavity, CavityOptions
from toolkit.components.ports import Port
import numpy as np
import gmsh

@dataclass(kw_only=True)
class RectangularCavityOptions(CavityOptions):
    width: float
    height: float
    depth: float
    

class RectangularCavity(Cavity):
    def __init__(self, options: CavityOptions, ports: list[Port] = None, **materials):
        # materials: cavity_wall_material, volume_material (see Cavity)
        super().__init__(options, ports, **materials)
        
    @property
    def width(self) -> float:
        return self.options.width
    @property
    def height(self) -> float:
        return self.options.height
    
    @property
    def depth(self) -> float:
        return self.options.depth
    
    def _add_body(self, occ) -> list[tuple[int, int]]:
        """Add the cylindrical cavity body to the gmsh model."""
        return [(3, occ.addBox(0, 0, 0, self.width, self.height, self.depth))]