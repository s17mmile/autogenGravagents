import numpy as np

# Constants
alpha = 6.2e-3  # Molar absorption coefficient in dm^3/mol/cm
c = 1.0         # Concentration of seawater in mol/dm^3 (approximation)

# Calculate the depth at which intensity is half
ln_half = np.log(1/2)

# Using the Beer-Lambert law to find depth
x_cm = -ln_half / (alpha * c)  # Depth in cm
x_m = x_cm / 100  # Convert to meters

# Print the result with context
print(f'Depth at which a diver will experience half the surface intensity of light in seawater: {x_m:.2f} meters')