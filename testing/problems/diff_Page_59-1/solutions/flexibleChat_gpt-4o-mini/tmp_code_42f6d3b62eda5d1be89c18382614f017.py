import math

# Function to calculate the time required for dye concentration to reach 1% of its original value

def calculate_time_to_target_concentration(initial_concentration, initial_dye_amount, volume):
    # Target concentration (1% of initial concentration)
    target_concentration = 0.01 * initial_concentration  # g/L
    
    # Calculate time to reach target concentration
    ln_target_concentration = math.log(target_concentration)
    
    # Time in minutes
    time_to_target = -ln_target_concentration / 0.01
    return time_to_target

# Constants
initial_concentration = 1.0  # g/L
initial_dye_amount = 200.0  # g
volume = 200.0  # L

# Calculate and print the time
time_required = calculate_time_to_target_concentration(initial_concentration, initial_dye_amount, volume)
print(f'Time to reach 1% of original concentration: {time_required:.2f} minutes')