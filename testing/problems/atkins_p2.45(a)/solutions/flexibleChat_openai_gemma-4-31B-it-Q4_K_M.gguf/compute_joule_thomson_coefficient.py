# filename: compute_joule_thomson_coefficient.py

# Given thermodynamic data
partial_H_partial_p_T = -3.29e3  # J MPa^-1 mol^-1 (at 1.00 bar, 50°C)
Cp_m = 110.0  # J K^-1 mol^-1

# Compute Joule-Thomson coefficient
mu = partial_H_partial_p_T / Cp_m  # K MPa^-1

# Round to 3 significant figures
mu_rounded = round(mu, 1)  # -29.9 K MPa^-1 (3 sig figs)

# Use 'mu' instead of 'μ' to avoid Unicode encoding issues
print(f"Joule-Thomson coefficient mu = {mu_rounded} K MPa^-1")