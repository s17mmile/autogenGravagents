# filename: photon_energy_calculation_function.py
from scipy.constants import h, c, electron_volt

def photon_energy(wavelength_m):
    """Calculate photon energy given wavelength in meters.
    Returns energy in joules and electronvolts."""
    energy_joules = h * c / wavelength_m
    energy_eV = energy_joules / electron_volt
    return energy_joules, energy_eV

# Example usage for 100 pm wavelength
wavelength_pm = 100
wavelength_m = wavelength_pm * 1e-12
energy_j, energy_ev = photon_energy(wavelength_m)

print(f"Photon energy for wavelength {wavelength_pm} pm:")
print(f"Energy = {energy_j:.3e} joules")
print(f"Energy = {energy_ev:.2f} eV")
