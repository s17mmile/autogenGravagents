import math

# Given values
T1 = 298.15  # Initial temperature in K
T2 = 248.44  # Final temperature in K
P1 = 202.94  # Initial pressure in kPa
P2 = 81.840  # Final pressure in kPa
R = 8.314    # Ideal gas constant in J/(mol·K)

# Function to calculate gamma
def calculate_gamma(T1, T2, P1, P2):
    ratio_T = T1 / T2
    ratio_P = P1 / P2
    ln_ratio_T = math.log(ratio_T)
    ln_ratio_P = math.log(ratio_P)
    x = ln_ratio_T / ln_ratio_P
    gamma = 1 / (1 - x)
    return gamma

# Step 1: Calculate gamma
gamma = calculate_gamma(T1, T2, P1, P2)

# Step 2: Calculate C_p using the relationship C_p = -gamma * R / (1 - gamma)
C_p = -gamma * R / (1 - gamma)

# Step 3: Output the results
print(f'Calculated gamma: {gamma}')
print(f'Calculated C_p: {C_p} J/(mol·K)')