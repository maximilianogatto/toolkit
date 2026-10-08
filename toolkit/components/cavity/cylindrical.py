from dataclasses import dataclass
from .cavity import Cavity, CavityOptions
from toolkit.components.ports import Port
import numpy as np
import gmsh

@dataclass(kw_only=True)
class CylindricalCavityOptions(CavityOptions):
    radius: float
    height: float
    

class CylindricalCavity(Cavity):
    def __init__(self, options: CavityOptions, ports: list[Port] = None, **materials):
        # materials: cavity_wall_material, volume_material (see Cavity)
        super().__init__(options, ports, **materials)
        
    @property
    def radius(self) -> float:
        return self.options.radius
    @property
    def height(self) -> float:
        return self.options.height
    
    def _add_body(self, occ) -> list[tuple[int, int]]:
        """Add the cylindrical cavity body to the gmsh model."""
        return [(3, occ.addCylinder(0, 0, 0, 0, 0, self.height, self.radius))]