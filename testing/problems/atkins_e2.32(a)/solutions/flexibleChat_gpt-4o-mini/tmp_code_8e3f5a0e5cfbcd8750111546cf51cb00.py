def calculate_thermal_expansion_coefficient(V_prime, T):
    if V_prime <= 0:
        raise ValueError('V_prime must be greater than zero.')

    # Step 1: Calculate volume V at T
    V = V_prime * (0.75 + 3.9e-4 * T + 1.48e-6 * (T ** 2))

    # Step 2: Calculate the partial derivative of V with respect to T
    partial_V_T = V_prime * (3.9e-4 + 2 * 1.48e-6 * T)

    # Step 3: Calculate the thermal expansion coefficient alpha
    alpha = partial_V_T / V

    # Output the result
    print(f'Thermal expansion coefficient alpha at {T} K (V_prime={V_prime}): {alpha:.6f} K^-1')

# Example usage
calculate_thermal_expansion_coefficient(1.0, 320)