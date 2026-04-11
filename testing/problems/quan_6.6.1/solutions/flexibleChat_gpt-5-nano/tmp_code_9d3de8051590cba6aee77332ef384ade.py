#!/usr/bin/env python3

# Lightweight test-like demonstration for dual E1 calculation

def compute_E1_for_mu(mu: float, m_e: float, e: float, eps0: float, h: float) -> dict:
    Ry_J = (m_e * e**4) / (8.0 * eps0**2 * h**2)
    E_direct_J = - (mu * e**4) / (8.0 * eps0**2 * h**2)

    def J_to_ev(J: float) -> float:
        return J / 1.602176634e-19

    return {
        'E_direct_J': E_direct_J,
        'E_direct_ev': J_to_ev(E_direct_J),
        'E_via_Ry_J': - (mu / m_e) * Ry_J,
        'E_via_Ry_ev': J_to_ev(- (mu / m_e) * Ry_J),
        'Ry_J': Ry_J,
    }


def main():
    m_e = 9.1093837015e-31
    m_p = 1.67262192369e-27
    e = 1.602176634e-19
    eps0 = 8.8541878128e-12
    h = 6.62607015e-34

    mu_e = m_e
    mu_reduced = (m_e * m_p) / (m_e + m_p)

    res_e = compute_E1_for_mu(mu_e, m_e, e, eps0, h)
    res_r = compute_E1_for_mu(mu_reduced, m_e, e, eps0, h)

    print("mu = m_e -> E_direct = {0:.12e} J, E_via_Ry = {1:.12e} J".format(res_e['E_direct_J'], res_e['E_via_Ry_J']))
    print("mu = m_e -> E_direct = {0:.12f} eV, E_via_Ry = {1:.12f} eV".format(res_e['E_direct_ev'], res_e['E_via_Ry_ev']))

    print("mu = reduced -> E_direct = {0:.12e} J, E_via_Ry = {1:.12e} J".format(res_r['E_direct_J'], res_r['E_via_Ry_J']))
    print("mu = reduced -> E_direct = {0:.12f} eV, E_via_Ry = {1:.12f} eV".format(res_r['E_direct_ev'], res_r['E_via_Ry_ev']))

    # Simple assertion examples (not a full test suite)
    tol = 1e-12
    assert abs(res_e['E_direct_J'] - res_e['E_via_Ry_J']) < 1e-28
    assert abs(res_r['E_direct_J'] - res_r['E_via_Ry_J']) < 1e-28

if __name__ == '__main__':
    main()
