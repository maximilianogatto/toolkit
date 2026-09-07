from SQDMetal.Utilities.QUtilities import QUtilities


def adjust_design(design, chip_margin=0.15e-3):
    """
    Centers the design in the GUI.

    Parameters:
        design (Design): The design to be centered.
    """
    # Get the bounding box of the design
    all_components = list(design.components.keys())
    min_x, min_y, max_x, max_y = QUtilities.get_comp_bounds(design, all_components, units_metres=True)
    
    
    chip_size_x = (max_x - min_x) + 2 * chip_margin
    chip_size_y = (max_y - min_y) + 2 * chip_margin
    chip_center_x = 0.5 * (min_x + max_x)
    chip_center_y = 0.5 * (min_y + max_y)
    
    design.chips.main.size.size_x = f'{chip_size_x * 1e3:.4f}mm'
    design.chips.main.size.size_y = f'{chip_size_y * 1e3:.4f}mm'
    design.chips.main.size.center_x = f'{chip_center_x * 1e3:.4f}mm'
    design.chips.main.size.center_y = f'{chip_center_y * 1e3:.4f}mm'
