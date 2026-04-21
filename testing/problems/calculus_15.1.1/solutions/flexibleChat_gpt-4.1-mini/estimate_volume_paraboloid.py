# filename: estimate_volume_paraboloid.py

# Define the function z = 16 - x^2 - 2*y^2
def z(x, y):
    return 16 - x**2 - 2*(y**2)

# Define the region and subdivision
x_start, x_end = 0, 2
y_start, y_end = 0, 2
num_subdivisions = 2  # dividing into 4 equal squares (2x2)

# Validate number of subdivisions
if num_subdivisions <= 0:
    raise ValueError("Number of subdivisions must be positive")

# Calculate the width and height of each small square
width = (x_end - x_start) / num_subdivisions
height = (y_end - y_start) / num_subdivisions

# Coordinates of upper right corners of each small square
# Upper right corners are chosen as sample points for the Riemann sum approximation
sample_points = []
for i in range(num_subdivisions):
    for j in range(num_subdivisions):
        x = x_start + (i + 1) * width
        y = y_start + (j + 1) * height
        sample_points.append((x, y))

# Calculate the volume estimate
volume_estimate = 0
area_small_square = width * height
for (x, y) in sample_points:
    volume_estimate += z(x, y) * area_small_square

# Output the estimated volume formatted to 4 decimal places
print(f"Estimated volume under the paraboloid: {volume_estimate:.4f}")
