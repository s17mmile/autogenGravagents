#!/usr/bin/env python3

"""Reusable E1 computation with a unittest suite.

This module provides:
- compute_E1_for_mu(mu, m_e, e, eps0, h) -> dict with direct (E_direct_J) and
  Ry-based (E_via_Ry_J) energies in J and eV, plus Ry_J intermediary.
- A unittest TestCase that validates both calculation paths and the reduced-mass scaling.
"""

import unittest


def compute_E1_for_mu(mu: float, m_e: float, e: float, eps0: float, h: float) -> dict:
    """Compute E1 for a given mu using both direct formula and Ry-based cross-check.

    Returns a dictionary with:
      - E_direct_J: direct computation in joules
      - E_direct_ev: direct computation in electronvolts
      - E_via_Ry_J: E1 computed via Ry and mu (E1 = - (mu/m_e) * Ry_J)
      - E_via_Ry_ev: E1 via Ry in electronvolts
      - Ry_J: the Rydberg energy in joules derived from constants
    """
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


class TestE1Calculation(unittest.TestCase):
    def setUp(self):
        self.m_e = 9.1093837015e-31
        self.m_p = 1.67262192369e-27
        self.e = 1.602176634e-19
        self.eps0 = 8.8541878128e-12
        self.h = 6.62607015e-34
        self.mu_e = self.m_e
        self.mu_reduced = (self.m_e * self.m_p) / (self.m_e + self.m_p)

    def test_direct_matches_ry(self):
        res_e = compute_E1_for_mu(self.mu_e, self.m_e, self.e, self.eps0, self.h)
        # Direct and Ry-based results should agree to numerical precision
        self.assertAlmostEqual(res_e['E_direct_J'], res_e['E_via_Ry_J'], places=12)
        res_r = compute_E1_for_mu(self.mu_reduced, self.m_e, self.e, self.eps0, self.h)
        self.assertAlmostEqual(res_r['E_direct_J'], res_r['E_via_Ry_J'], places=12)

    def test_reduced_mass_scaling(self):
        res_e = compute_E1_for_mu(self.mu_e, self.m_e, self.e, self.eps0, self.h)
        res_r = compute_E1_for_mu(self.mu_reduced, self.m_e, self.e, self.eps0, self.h)
        rel_expected = self.mu_reduced / self.m_e
        self.assertAlmostEqual(res_r['E_direct_J'] / res_e['E_direct_J'], rel_expected, places=12)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
