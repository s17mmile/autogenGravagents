import math

def calculate_biomass(r, K, y0, t):
    # Check for valid parameters
    if y0 > K:
        raise ValueError('Initial biomass y0 cannot be greater than carrying capacity K.')
    if r < 0:
        raise ValueError('Growth rate r must be a positive value.')

    # Logistic growth model calculation
    K_minus_y0 = K - y0
    ratio = K_minus_y0 / y0
    exponential_term = math.exp(-r * t)
    y_t = K / (1 + ratio * exponential_term)
    return y_t

# Parameters
r = 0.71  # intrinsic growth rate (per year)
K = 80.5 * 10**6  # carrying capacity in kg
y0 = 0.25 * K  # initial biomass in kg
t = 2  # time in years

# Calculate biomass
biomass_after_2_years = calculate_biomass(r, K, y0, t)

# Print the result
print(f'Biomass after {t} years: {biomass_after_2_years:.2f} kg')