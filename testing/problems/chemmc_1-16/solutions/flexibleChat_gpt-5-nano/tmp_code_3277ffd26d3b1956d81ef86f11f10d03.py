from typing import Tuple

# Physical constants (SI, ASCII only)
h = 6.62607015e-34  # Planck constant (J s)
c = 299792458.0      # Speed of light (m / s)


def photon_rate(P_w: float, wavelength_nm: float) -> Tuple[float, float]:
    """Compute photon emission rate and energy per photon.

    Parameters:
    - P_w: Power in watts (W), must be positive
    - wavelength_nm: Wavelength in nanometers (nm), must be positive

    Returns:
    - (N_dot, E_per_photon): N_dot is photons per second, E_per_photon is energy per photon in joules
    """
    if not isinstance(P_w, (int, float)):
        raise TypeError("P_w must be a number")
    if not isinstance(wavelength_nm, (int, float)):
        raise TypeError("wavelength_nm must be a number")
    if P_w <= 0:
        raise ValueError("P_w must be positive")
    if wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be positive")

    wavelength_m = float(wavelength_nm) * 1e-9
    E_per_photon = h * c / wavelength_m
    N_dot = P_w / E_per_photon
    return N_dot, E_per_photon


if __name__ == '__main__':
    # Example: 1.0 mW at 337 nm
    P_w = 1.0e-3
    wavelength_nm = 337.0
    N_dot, E_per_photon = photon_rate(P_w, wavelength_nm)
    print(f"Energy per photon (J): {E_per_photon:.3e}")
    print(f"Photon emission rate (photons per second): {N_dot:.3e}")
