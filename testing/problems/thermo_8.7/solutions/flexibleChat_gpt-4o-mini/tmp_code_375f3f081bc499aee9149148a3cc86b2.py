import math

# Constants
initial_radius = 20.0e-6  # Initial radius in meters
increase_factor = 3        # Factor by which the radius increases
surface_tension = 0.072    # Surface tension of water in N/m

# Step 1: Calculate final radius
final_radius = increase_factor * initial_radius

# Step 2: Calculate initial and final surface areas
initial_area = 4 * math.pi * (initial_radius ** 2)  # Initial surface area
final_area = 4 * math.pi * (final_radius ** 2)      # Final surface area

# Debugging: Print initial and final surface areas
print(f"Initial Surface Area (A_i): {initial_area:.3e} m^2")
print(f"Final Surface Area (A_f): {final_area:.3e} m^2")

# Step 3: Calculate change in surface area
change_in_area = final_area - initial_area

# Debugging: Print change in surface area
print(f"Change in Surface Area (Delta_A): {change_in_area:.3e} m^2")

# Step 4: Calculate work done
work_done = surface_tension * change_in_area

# Output the result
print(f"Work done to expand the cell surface: {work_done:.3e} J")