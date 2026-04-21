# filename: calculate_quantum_yield_enhanced.py

from scipy.constants import h, c, N_A

def calculate_quantum_yield(mmol_converted, power_watts, wavelength_nm, time_seconds):
    """
    Calculate the overall quantum yield for a photochemical process.

    Parameters:
    mmol_converted (float): Amount of reactant converted in millimoles.
    power_watts (float): Power of the light source in watts.
    wavelength_nm (float): Wavelength of the light in nanometers.
    time_seconds (float): Irradiation time in seconds.

    Returns:
    float: Quantum yield (dimensionless).
    """
    # Input validation
    if mmol_converted <= 0:
        raise ValueError("Amount converted must be positive.")
    if power_watts <= 0:
        raise ValueError("Power must be positive.")
    if wavelength_nm <= 0:
        raise ValueError("Wavelength must be positive.")
    if time_seconds <= 0:
        raise ValueError("Time must be positive.")

    # Convert mmol to moles
    moles_converted = mmol_converted * 1e-3

    # Calculate number of molecules converted
    molecules_converted = moles_converted * N_A

    # Convert wavelength from nm to meters
    wavelength_m = wavelength_nm * 1e-9

    # Calculate energy of one photon
    E_photon = (h * c) / wavelength_m  # in Joules

    # Calculate total energy absorbed
    energy_total = power_watts * time_seconds  # in Joules

    # Calculate number of photons absorbed
    photons_absorbed = energy_total / E_photon

    # Calculate quantum yield
    quantum_yield = molecules_converted / photons_absorbed

    return quantum_yield

# Given data
mmol_converted = 2.5
power_watts = 100.0
wavelength_nm = 280.0
time_seconds = 27.0

quantum_yield = calculate_quantum_yield(mmol_converted, power_watts, wavelength_nm, time_seconds)

print(f"Overall quantum yield: {quantum_yield:.4f}")
