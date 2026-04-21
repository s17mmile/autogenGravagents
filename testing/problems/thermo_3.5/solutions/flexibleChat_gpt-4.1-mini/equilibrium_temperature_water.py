# filename: equilibrium_temperature_water.py
import numpy as np
from scipy.optimize import root_scalar

# Constants
m_ice = 34.05  # g
m_water = 185  # g
T_ice_initial = 273.0  # K
T_water_initial = 310.0  # K
molar_mass_H2O = 18.015  # g/mol
Delta_H_fus = 6010  # J/mol (enthalpy of fusion)
C_P_m = 75.3  # J/(mol K), molar heat capacity of liquid water at 298 K

# Calculate moles
n_ice = m_ice / molar_mass_H2O
n_water = m_water / molar_mass_H2O

# Heat balance function: heat gained by ice = heat lost by water
# Q_ice = heat to melt ice + heat to warm melted ice from 273 K to T_final
# Q_water = heat lost by warm water cooling from 310 K to T_final

def heat_balance(T_final):
    if T_final < T_ice_initial:
        # Final temperature cannot be below melting point since ice melts
        return 1e6  # large positive number to avoid invalid solution
    Q_ice = n_ice * Delta_H_fus + n_ice * C_P_m * (T_final - T_ice_initial)
    Q_water = n_water * C_P_m * (T_water_initial - T_final)
    return Q_ice - Q_water

# Solve for T_final where heat_balance = 0
solution = root_scalar(heat_balance, bracket=[T_ice_initial, T_water_initial], method='brentq')

T_final = solution.root if solution.converged else None

# Save result to a file
with open('final_temperature_result.txt', 'w') as f:
    if T_final is not None:
        f.write(f'Final equilibrium temperature: {T_final:.2f} K\n')
    else:
        f.write('Failed to find equilibrium temperature.\n')

# Also print the result for immediate feedback
if T_final is not None:
    print(f'Final equilibrium temperature: {T_final:.2f} K')
else:
    print('Failed to find equilibrium temperature.')
