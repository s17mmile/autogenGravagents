import math

# Given values
diameter = 0.75e-3  # diameter in meters
radius = diameter / 2  # radius in meters
length = 1.00  # length of the tube in meters
P1 = 1.05 * 101325  # input pressure in Pa
P2 = 1.00 * 101325  # output pressure in Pa
viscosity = 146e-6  # dynamic viscosity in Pa.s
mass_CO2 = 22.7  # mass of CO2 in kg

def calculate_flow_rate(radius, P1, P2, viscosity, length):
    # Hagen-Poiseuille equation
    Q = (math.pi * radius**4 * (P1 - P2)) / (8 * viscosity * length)
    return Q

# Step 1: Calculate the flow rate
flow_rate = calculate_flow_rate(radius, P1, P2, viscosity, length)

# Step 2: Convert volumetric flow rate to mass flow rate
# Density of CO2 at 293 K and 1 atm
density_CO2 = 1.977  # kg/m^3
mass_flow_rate = flow_rate * density_CO2

# Step 3: Calculate how long the cylinder will last
time_seconds = mass_CO2 / mass_flow_rate

# Output the results
flow_rate, mass_flow_rate, time_seconds