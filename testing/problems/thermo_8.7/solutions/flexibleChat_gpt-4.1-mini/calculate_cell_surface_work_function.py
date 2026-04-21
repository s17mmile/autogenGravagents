# filename: calculate_cell_surface_work_function.py
import math

def calculate_work_to_expand_cell_surface(r_initial_meters, expansion_factor, surface_tension_n_per_m=0.0728):
    """
    Calculate the work required to expand the surface of a spherical cell against surface tension.
    
    Parameters:
    r_initial_meters (float): Initial radius of the cell in meters (must be positive).
    expansion_factor (float): Factor by which the radius increases (must be positive).
    surface_tension_n_per_m (float): Surface tension in N/m (default is for water at 298.15 K).
    
    Returns:
    float: Work done in joules.
    """
    if r_initial_meters <= 0:
        raise ValueError("Initial radius must be positive.")
    if expansion_factor <= 0:
        raise ValueError("Expansion factor must be positive.")
    
    r_final = expansion_factor * r_initial_meters
    A_initial = 4 * math.pi * r_initial_meters**2
    A_final = 4 * math.pi * r_final**2
    delta_A = A_final - A_initial
    work = surface_tension_n_per_m * delta_A
    return work

# Given values
r_initial = 20.0e-6  # meters
expansion_factor = 3

work_required = calculate_work_to_expand_cell_surface(r_initial, expansion_factor)
print(f"Work required to expand the cell surface: {work_required:.3e} Joules")
