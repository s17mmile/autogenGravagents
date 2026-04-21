# filename: diffusion_coefficient_n2_function.py
import math

def diffusion_coefficient_n2(T=298, P_atm=1, sigma_cross_nm2=0.43, Omega_D=1.0):
    """
    Calculate the diffusion coefficient of N2 gas using the Chapman-Enskog equation.

    Parameters:
    T : float
        Temperature in Kelvin (K)
    P_atm : float
        Pressure in atmospheres (atm)
    sigma_cross_nm2 : float
        Collisional cross section in nm^2
    Omega_D : float
        Collision integral (dimensionless), approx 1.0 for room temperature

    Returns:
    D : float
        Diffusion coefficient in m^2/s
    """
    # Constants
    k_B = 1.380649e-23  # Boltzmann constant in J/K
    P = P_atm * 101325  # Convert pressure from atm to Pa

    # Convert collisional cross section from nm^2 to m^2
    sigma_cross_m2 = sigma_cross_nm2 * 1e-18

    # Calculate collision diameter sigma (m)
    sigma = math.sqrt(sigma_cross_m2 / math.pi)

    # Molecular mass of N2 (kg)
    molar_mass_N2 = 28.0134e-3  # kg/mol
    N_A = 6.02214076e23  # Avogadro's number
    m = molar_mass_N2 / N_A  # mass of one N2 molecule in kg

    # Chapman-Enskog equation for diffusion coefficient D (m^2/s)
    D = (3/16) * math.sqrt((k_B * T) / (math.pi * m)) * (1 / (P * sigma**2)) * Omega_D

    return D

# Example usage
if __name__ == "__main__":
    D = diffusion_coefficient_n2()
    print(f"Diffusion coefficient of N2 at 1 atm and 298 K: {D:.3e} m^2/s")
