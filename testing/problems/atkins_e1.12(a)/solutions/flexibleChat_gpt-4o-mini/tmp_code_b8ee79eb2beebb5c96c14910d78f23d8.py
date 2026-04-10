import numpy as np
from sklearn.linear_model import LinearRegression

# Given data: temperatures in Celsius and corresponding densities
temperatures_c = np.array([-85, 0, 100])  # in degrees Celsius

# Convert temperatures to Kelvin
temperatures_k = temperatures_c + 273.15
# Corresponding densities in g/dm³
densities = np.array([1.877, 1.294, 0.946])

# Reshape the data for linear regression
X = densities.reshape(-1, 1)  # independent variable (density)
Y = temperatures_k  # dependent variable (temperature in Kelvin)

# Create and fit the model
model = LinearRegression()
model.fit(X, Y)

# Get the slope (m) and intercept (b)
slope = model.coef_[0]
intercept = model.intercept_

# Calculate absolute zero in Kelvin and convert to Celsius
absolute_zero_k = -intercept / slope
absolute_zero_c = absolute_zero_k - 273.15

# Output the results
print(f'Slope (m): {slope}')
print(f'Intercept (b): {intercept}')
print(f'Estimated Absolute Zero: {absolute_zero_c:.2f} °C')