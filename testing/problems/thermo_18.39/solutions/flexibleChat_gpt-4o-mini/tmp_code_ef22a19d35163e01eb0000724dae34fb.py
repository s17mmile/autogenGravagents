import math

def calculate_time_to_pressure(P0, P, k):
    # Check for valid input values
    if P >= P0:
        raise ValueError('Pressure of the reactant must be less than the initial pressure.')
    
    # Calculate time using the first-order kinetics equation
    ln_ratio = math.log(P0 / P)
    time = ln_ratio / k
    return time

# Given values
P0 = 1.0  # Initial pressure in atm
P = 0.2   # Pressure of the reactant at time t in atm
k = 0.0019  # Rate constant in s^-1

# Calculate time
try:
    time = calculate_time_to_pressure(P0, P, k)
    print(f'Time to reach 1.8 atm: {time:.2f} seconds')
except ValueError as e:
    print(e)