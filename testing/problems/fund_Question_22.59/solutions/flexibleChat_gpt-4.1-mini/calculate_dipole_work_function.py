# filename: calculate_dipole_work_function.py
import math

def calculate_work(p, E, theta_i_deg, rotation_deg):
    # Normalize final angle within [0, 360) degrees
    theta_f_deg = (theta_i_deg + rotation_deg) % 360
    
    # Convert degrees to radians
    theta_i_rad = math.radians(theta_i_deg)
    theta_f_rad = math.radians(theta_f_deg)
    
    # Calculate work done
    work = p * E * (math.cos(theta_i_rad) - math.cos(theta_f_rad))
    return work

# Given values
E = 46.0  # Electric field magnitude in N/C
p = 3.02e-25  # Dipole moment magnitude in C*m
initial_angle = 64  # Initial angle in degrees
rotation = 180  # Rotation angle in degrees

work_required = calculate_work(p, E, initial_angle, rotation)
print(f"Work required to rotate the dipole 180 degrees: {work_required:.3e} Joules")
