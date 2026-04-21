# filename: calculate_expansion_coefficient.py

def calculate_alpha(T):
    """Calculate the expansion coefficient alpha at temperature T (in K).
    Returns alpha in per Kelvin."""
    # Constants from the volume equation
    a = 0.75
    b = 3.9e-4
    c = 1.48e-6

    # Calculate volume V(T) relative to V'
    V_T = a + b * T + c * T**2

    # Calculate derivative dV/dT
    dV_dT = b + 2 * c * T

    # Calculate expansion coefficient alpha
    alpha = dV_dT / V_T
    return alpha

# Calculate alpha at T = 320 K
T = 320
alpha_320 = calculate_alpha(T)

print(f"Expansion coefficient alpha at {T} K: {alpha_320:.6e} per K")
