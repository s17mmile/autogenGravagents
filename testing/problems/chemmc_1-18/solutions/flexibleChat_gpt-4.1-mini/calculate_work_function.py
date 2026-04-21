# filename: calculate_work_function.py

from scipy.constants import h, c, e

def calculate_work_function(wavelength_nm):
    """Calculate the work function in eV given the threshold wavelength in nm."""
    wavelength_m = wavelength_nm * 1e-9
    work_function_joules = (h * c) / wavelength_m
    work_function_eV = work_function_joules / e
    return work_function_eV

# Given threshold wavelength for potassium
threshold_wavelength_nm = 564
work_function = calculate_work_function(threshold_wavelength_nm)
work_function