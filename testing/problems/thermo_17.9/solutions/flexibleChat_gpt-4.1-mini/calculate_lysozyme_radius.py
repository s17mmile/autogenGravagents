# filename: calculate_lysozyme_radius.py

import math

# Function to calculate radius of spherical protein from sedimentation coefficient
# Parameters:
# M: molar mass in g/mol
# s: sedimentation coefficient in seconds
# v_bar: partial specific volume in cm^3/g
# rho: solvent density in g/cm^3
# eta_cP: solvent viscosity in centipoise
# Returns radius in nanometers

def calculate_radius(M, s, v_bar, rho, eta_cP):
    # Convert viscosity from cP to poise (1 cP = 0.01 poise)
    eta = eta_cP * 0.01  # poise

    # Avogadro's number
    N_A = 6.022e23  # mol^-1

    # Calculate radius in cm using rearranged Svedberg equation
    r_cm = (M * (1 - v_bar * rho)) / (N_A * 6 * math.pi * eta * s)

    # Convert radius to nanometers (1 cm = 1e7 nm)
    r_nm = r_cm * 1e7

    return r_nm

# Given values
M = 14100  # g/mol
s = 1.91e-13  # s
v_bar = 0.703  # cm^3/g
rho = 0.998  # g/cm^3
eta_cP = 1.002  # cP

radius_nm = calculate_radius(M, s, v_bar, rho, eta_cP)
print(f"Radius of lysozyme protein assuming spherical shape: {radius_nm:.2f} nm")
