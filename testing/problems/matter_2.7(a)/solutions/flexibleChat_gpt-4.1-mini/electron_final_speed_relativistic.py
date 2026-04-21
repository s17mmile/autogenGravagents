# filename: electron_final_speed_relativistic.py

from scipy.constants import e, m_e, c

def calculate_electron_speed(delta_phi_volts=100000, relativistic=False):
    """
    Calculate the final speed of an electron accelerated from rest through a potential difference.

    Parameters:
    delta_phi_volts (float): Potential difference in volts (SI units). Must be positive.
    relativistic (bool): If True, use relativistic formula; otherwise, use classical approximation.

    Returns:
    float: Final speed of the electron in meters per second.

    Raises:
    ValueError: If delta_phi_volts is not a positive number.

    Notes:
    - Classical formula: v = sqrt(2 * e * delta_phi / m_e)
    - Relativistic formula accounts for kinetic energy as difference between total and rest energy.
    """
    if not isinstance(delta_phi_volts, (int, float)) or delta_phi_volts <= 0:
        raise ValueError("delta_phi_volts must be a positive number.")

    if not relativistic:
        # Classical kinetic energy formula
        v = (2 * e * delta_phi_volts / m_e) ** 0.5
        return v
    else:
        # Relativistic kinetic energy: K = (gamma - 1) * m_e * c^2 = e * delta_phi
        K = e * delta_phi_volts
        gamma = 1 + K / (m_e * c ** 2)
        v = c * (1 - 1 / gamma ** 2) ** 0.5
        return v

if __name__ == "__main__":
    delta_phi = 100000  # 100 kV
    v_classical = calculate_electron_speed(delta_phi, relativistic=False)
    v_relativistic = calculate_electron_speed(delta_phi, relativistic=True)
    print(f"Classical final speed: {v_classical:.3e} m/s")
    print(f"Relativistic final speed: {v_relativistic:.3e} m/s")
