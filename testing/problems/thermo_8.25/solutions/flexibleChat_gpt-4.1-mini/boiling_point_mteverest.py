# filename: boiling_point_mteverest.py
import math

def calculate_boiling_point(P2_torr=253, P1_torr=760, T1_K=373.12, delta_H_vap_kJ_per_mol=40.656):
    """
    Calculate the boiling point temperature (T2) at pressure P2 using the Clausius-Clapeyron equation.

    Parameters:
    P2_torr (float): Pressure at new boiling point in Torr (e.g., 253 Torr for Mt. Everest).
    P1_torr (float): Pressure at normal boiling point in Torr (default 760 Torr).
    T1_K (float): Normal boiling point temperature in Kelvin (default 373.12 K).
    delta_H_vap_kJ_per_mol (float): Enthalpy of vaporization in kJ/mol (default 40.656 kJ/mol).

    Returns:
    float: Boiling point temperature at pressure P2 in Kelvin.
    """
    if P2_torr <= 0 or P1_torr <= 0 or T1_K <= 0 or delta_H_vap_kJ_per_mol <= 0:
        raise ValueError("All input values must be positive and non-zero.")

    R = 8.314  # J/(mol*K)
    delta_H_vap = delta_H_vap_kJ_per_mol * 1000  # convert kJ/mol to J/mol

    try:
        ln_P2_P1 = math.log(P2_torr / P1_torr)
    except ValueError as e:
        raise ValueError("Invalid pressure values leading to math domain error in logarithm.") from e

    inv_T2 = (1 / T1_K) - (R / delta_H_vap) * ln_P2_P1
    T2_K = 1 / inv_T2
    return T2_K


if __name__ == '__main__':
    boiling_point_everest = calculate_boiling_point()

    with open('boiling_point_everest.txt', 'w') as f:
        f.write(f'Boiling point of water at 253 Torr (Mt. Everest): {boiling_point_everest:.2f} K\n')
