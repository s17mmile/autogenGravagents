import math

# Physical constants (SI)
N_A = 6.02214076e23          # mol^-1
amu_to_kg = 1.66053906660e-27 # kg per u (atomic mass unit)
c = 299792458.0                # m s^-1

def mu_kg_from_isotopes(m_H_u: float, m_Cl_u: float) -> float:
    """Compute reduced mass in kilograms for diatomic isotopes given masses in atomic mass units (u)."""
    mu_u = (m_H_u * m_Cl_u) / (m_H_u + m_Cl_u)
    return mu_u * amu_to_kg


def compute_beta(tilde_omega_obs_cm1: float, D_input: float, *,
                 D_input_is_per_molecule: bool = False,
                 m_H_u: float = 1.007825, m_Cl_u: float = 34.968853) -> float:
    """Compute beta for a diatomic system in inverse meters (m^-1).

    beta = omega * sqrt(mu/(2*D_per_molecule))
    where D_per_molecule = D_input if D_input_is_per_molecule else D_input / N_A.

    Parameters
    - tilde_omega_obs_cm1: wavenumber in cm^-1
    - D_input: if D_input_is_per_molecule is False, this is D_molar_J_per_mol; else this is D_per_molecule_J
    - D_input_is_per_molecule: flag to interpret D_input accordingly
    - m_H_u, m_Cl_u: isotopic masses in atomic mass units (defaults to H-1 and Cl-35)

    Returns
    - beta in m^-1
    """
    if tilde_omega_obs_cm1 <= 0:
        raise ValueError("tilde_omega_obs_cm1 must be positive.")
    if D_input <= 0:
        raise ValueError("D_input must be positive.")

    D_per_molecule = D_input if D_input_is_per_molecule else D_input / N_A

    mu_kg = mu_kg_from_isotopes(m_H_u, m_Cl_u)
    sqrt_term = math.sqrt(mu_kg / (2.0 * D_per_molecule))

    # Convert tilde_omega_obs from cm^-1 to m^-1 and compute angular frequency
    tilde_omega_obs_m1 = tilde_omega_obs_cm1 * 100.0  # cm^-1 -> m^-1
    f = c * tilde_omega_obs_m1
    omega = 2.0 * math.pi * f

    return omega * sqrt_term


if __name__ == "__main__":
    # Demonstration for H-35Cl with the provided parameters (D given as per mole)
    beta_val = compute_beta(tilde_omega_obs_cm1=2886.0, D_input=440.2e3, D_input_is_per_molecule=False)
    print("beta (m^-1) =", beta_val)

    # Demonstration: pass D per molecule directly
    D_per_molecule = 440.2e3 / N_A
    beta_direct = compute_beta(tilde_omega_obs_cm1=2886.0, D_input=D_per_molecule, D_input_is_per_molecule=True)
    print("beta (m^-1) with D_per_molecule =", beta_direct)
