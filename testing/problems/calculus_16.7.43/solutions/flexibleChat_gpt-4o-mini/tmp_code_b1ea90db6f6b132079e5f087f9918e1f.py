import numpy as np

# Constants
rho = 870  # density in kg/m^3

# Define the cylinder parameters
radius = 2
height = 1

# Calculate flux through the lateral surface
flux_lateral = 0
for theta in np.linspace(0, 2 * np.pi, 100):
    for z in np.linspace(0, height, 100):
        velocity_x = z  # velocity component in x direction
        velocity_y = (radius * np.sin(theta))**2  # velocity component in y direction
        velocity_z = (radius * np.cos(theta))**2  # velocity component in z direction
        normal_x = np.cos(theta)  # outward normal component in x direction
        normal_y = np.sin(theta)  # outward normal component in y direction
        dS = radius * (height / 100) * (2 * np.pi / 100)  # area element
        flux_lateral += (velocity_x * normal_x + velocity_y * normal_y) * dS

# Calculate flux through the top surface
flux_top = np.pi * radius**2  # area of the top surface

# Calculate flux through the bottom surface
flux_bottom = 0  # no flow through the bottom surface

# Total flux
flux_total = flux_lateral + flux_top + flux_bottom

# Calculate mass flow rate
mass_flow_rate = rho * flux_total

# Print the result
print(f'Rate of flow outward through the cylinder: {mass_flow_rate:.2f} kg/s')