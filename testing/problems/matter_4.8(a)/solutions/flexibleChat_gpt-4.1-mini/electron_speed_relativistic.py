# filename: electron_speed_relativistic.py

from scipy.constants import h, m_e, c
import math

def electron_speed_from_wavelength(wavelength_pm):
    wavelength_m = wavelength_pm * 1e-12  # Convert pm to meters

    # Non-relativistic velocity calculation
    v_nonrel = h / (m_e * wavelength_m)

    # Check if relativistic correction is needed
    if v_nonrel > 0.1 * c:  # If speed > 10% speed of light, use relativistic formula
        # Relativistic momentum p = h / lambda
        p = h / wavelength_m
        # Relativistic velocity v = pc / sqrt(p^2 + (m_e c)^2)
        numerator = p * c
        denominator = math.sqrt(p**2 + (m_e * c)**2)
        v_rel = numerator / denominator
        return v_rel, True
    else:
        return v_nonrel, False

# Given wavelength
wavelength_pm = 100  # 100 pm

# Calculate electron speed
speed, relativistic = electron_speed_from_wavelength(wavelength_pm)

# Print the result
if relativistic:
    print(f"Electron speed for wavelength {wavelength_pm} pm (relativistic): {speed:.2e} m/s")
else:
    print(f"Electron speed for wavelength {wavelength_pm} pm (non-relativistic): {speed:.2e} m/s")
