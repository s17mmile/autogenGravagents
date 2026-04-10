def calculate_work(k, displacement_cm):
    # Convert displacement from cm to m
    displacement_m = displacement_cm / 100  # convert cm to m
    
    # Calculate the force using Hooke's law
    F = k * displacement_m
    
    # Calculate the work done using the formula for work done by a spring
    W = 0.5 * k * (displacement_m ** 2)
    
    return W

# Constants
k = 750  # spring constant in N/m
x_cm = 3.5  # displacement in cm

# Calculate work
work_done = calculate_work(k, x_cm)

# Print the result
print(f'Work performed by the muscle fiber: {work_done:.3f} J')