import numpy as np

# Physical constants (SI)
R = 8.314462618        # J/mol-K
kB = 1.380649e-23      # J/K
h = 6.62607015e-34      # J s
c = 299792458.0           # m/s
NA = 6.02214076e23        # 1/mol
P0 = 1.0e5                  # Pa (1 bar)


def G_deg_T(T=298.15, nu_tilde=560.0, B_cm=0.244, M_g_per_mol=69.937,
            P0=1e5, sigma=2, spin_statistics=False, weights_J=None, J_max=60):
    """
    Compute G°(T) for Cl2(g) with given rovibrational parameters.

    Inputs:
    - T: temperature in K
    - nu_tilde: wavenumber in cm^-1 (nu tilde)
    - B_cm: rotational constant in cm^-1
    - M_g_per_mol: molar mass in g/mol (default ~69.937 g/mol for Cl2)
    - P0: standard pressure in Pa (default 1e5 Pa)
    - sigma: rotational symmetry number (default 2 for homonuclear diatomic in high-T rotor)
    - spin_statistics: if True and weights_J is provided, use weighted sum over J for q_rot
    - weights_J: sequence of weights w_J for J = 0..J_max (non-negative). If None, use high-T rotor with sigma
    - J_max: maximum J to include in the weighted sum when weights_J is provided

    Returns a dict with:
    - G_T_J_per_mol: G°(T) in J/mol
    - mu_trans_J_per_mol, mu_rot_J_per_mol, mu_vib_J_per_mol, mu_elec_J_per_mol
    - q_rot: rotational partition function value used
    - spin_statistics_used: boolean flag indicating if the weighted rotor was used
    """
    # Derived quantities
    m_per_mol = M_g_per_mol / 1000.0        # kg per mol
    m_per_molecule = m_per_mol / NA          # kg per molecule

    # Vibrational scale
    theta_vib = 1.438775245 * nu_tilde        # K

    # Translational contribution per mole (ideal gas)
    mu_trans = R * T * np.log( (kB * T / P0) * ( (2.0 * np.pi * m_per_molecule * kB * T) / (h**2) )**1.5 )

    # Rotational contribution
    B_energy_scale = h * c * (B_cm * 100.0)     # J (energy scale for J(J+1))
    if spin_statistics and weights_J is not None:
        beta = 1.0 / (kB * T)
        q_rot = 0.0
        max_J = min(J_max, len(weights_J) - 1)
        for J in range(0, max_J + 1):
            w = float(weights_J[J])
            E_J = B_energy_scale * J * (J + 1)
            q_rot += (2*J + 1) * w * np.exp(-beta * E_J)
        mu_rot = R * T * (np.log(q_rot) - 1.0)
        spin_used = True
    else:
        q_rot = (kB * T) / (sigma * B_energy_scale)
        mu_rot = R * T * (np.log(q_rot) - 1.0)
        spin_used = False

    # Vibrational contribution (harmonic oscillator)
    mu_vib = (R * theta_vib) / 2.0 + R * T * np.log1p(-np.exp(-theta_vib / T))

    # Electronic contribution (ground state nondegenerate)
    mu_elec = 0.0

    G_T = mu_trans + mu_rot + mu_vib + mu_elec

    return {
        "G_T_J_per_mol": G_T,
        "mu_trans_J_per_mol": mu_trans,
        "mu_rot_J_per_mol": mu_rot,
        "mu_vib_J_per_mol": mu_vib,
        "mu_elec_J_per_mol": mu_elec,
        "q_rot": q_rot,
        "spin_statistics_used": bool(spin_used)
    }

if __name__ == "__main__":
    # Default run (spin statistics off, high-T rotor)
    res = G_deg_T()
    G_298 = res["G_T_J_per_mol"]
    print("G°(298.15 K) =",
          G_298/1000.0, "kJ/mol")
    print("  mu_trans =", res["mu_trans_J_per_mol"]/1000.0, "kJ/mol")
    print("  mu_rot   =", res["mu_rot_J_per_mol"]/1000.0, "kJ/mol")
    print("  mu_vib   =", res["mu_vib_J_per_mol"]/1000.0, "kJ/mol")
    print("  mu_elec  =", res["mu_elec_J_per_mol"]/1000.0, "kJ/mol")
    print("Spin statistics used: ", res["spin_statistics_used"])

    # Example placeholder for enabling spin statistics (user must supply weights_J)
    # weights_J = [1.0] + [0.0]*60  # replace with physically correct weights for J = 0..60
    # res_spin = G_deg_T(spin_statistics=True, weights_J=weights_J, J_max=60)
    # print("G° with spin statistics =", res_spin["G_T_J_per_mol"]/1000.0, "kJ/mol")
