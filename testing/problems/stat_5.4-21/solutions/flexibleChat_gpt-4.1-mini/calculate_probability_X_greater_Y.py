# filename: calculate_probability_X_greater_Y.py
from scipy.stats import norm

# Given parameters
mu_X = 5
sigma_X = 4  # standard deviation is sqrt(variance 16)
mu_Y = 6
sigma_Y = 3  # standard deviation is sqrt(variance 9)

# Parameters for Z = X - Y
mu_Z = mu_X - mu_Y
sigma_Z = (sigma_X**2 + sigma_Y**2)**0.5

# Calculate P(X > Y) = P(Z > 0) = 1 - P(Z <= 0)
prob = 1 - norm.cdf(0, loc=mu_Z, scale=sigma_Z)

print(f"P(X > Y) = {prob:.4f}")
