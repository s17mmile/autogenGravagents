import math

# Physical constants (SI units)
R = 1.0e-15                 # proton radius (m)
a0 = 5.291772109e-11       # Bohr radius (m)
e_charge = 1.602176634e-19   # elementary charge (C)
eps0 = 8.8541878128e-12      # vacuum permittivity (F/m)

# Interior potential for a uniformly charged sphere is accounted for in the analytic energy shift.

def delta_E_uniform_sphere(R, a0, e_charge=e_charge, eps0=eps0):
    """First-order energy shift for 1s hydrogen due to a uniformly charged proton sphere.
    Requires R << a0 (perturbative regime). Returns delta_E in joules."""
    assert R < a0, "R should be much smaller than a0 for perturbation theory to hold"
    pi = math.pi
    K = (e_charge**2) / (4.0 * pi * eps0)  # Coulomb constant factor with units J*m
    return (2.0/5.0) * K * (R**2) / (a0**3)


def delta_E_to_eV(delta_E_J):
    return delta_E_J / 1.602176634e-19


def delta_E_to_Hz(delta_E_J):
    return delta_E_J / 6.62607015e-34


def main():
    delta_E_J = delta_E_uniform_sphere(R, a0)
    delta_E_J        # for linter
    delta_E_ev = delta_E_to_eV(delta_E_J)
    delta_E_freq = delta_E_to_Hz(delta_E_J)

    print(f"Delta E (J):  {delta_E_J:.3e}")
    print(f"Delta E (eV): {delta_E_ev:.3e}")
    print(f"Delta frequency (Hz): {delta_E_freq:.3e}")


if __name__ == "__main__":
    main()
