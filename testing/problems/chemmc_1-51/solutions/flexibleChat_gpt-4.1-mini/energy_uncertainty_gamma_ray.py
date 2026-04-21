# filename: energy_uncertainty_gamma_ray.py

from scipy.constants import hbar, electron_volt

lifetime = 1e-12  # Lifetime of excited state in seconds

def energy_uncertainty(delta_t):
    """Calculate the energy uncertainty (Delta E) given the lifetime (Delta t) using the energy-time uncertainty principle.
    Args:
        delta_t (float): Lifetime of the excited state in seconds (must be positive).
    Returns:
        float: Energy uncertainty in joules.
    """
    if delta_t <= 0:
        raise ValueError("Lifetime must be positive.")
    delta_E = hbar / (2 * delta_t)
    return delta_E

# Calculate uncertainty
delta_E_joules = energy_uncertainty(lifetime)

# Convert Joules to electronvolts
delta_E_eV = delta_E_joules / electron_volt

# Output results
result = f"Uncertainty in energy: {delta_E_joules:.3e} Joules, which is approximately {delta_E_eV:.3f} eV"

result
