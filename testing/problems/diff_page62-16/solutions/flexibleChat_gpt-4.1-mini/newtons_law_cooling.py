# filename: newtons_law_cooling.py
import math

def time_to_reach_temperature(T0, T1, T_env, t1, T_target):
    """
    Calculate the time at which the temperature of an object reaches T_target
    according to Newton's law of cooling.

    Parameters:
    T0 : float - initial temperature of the object
    T1 : float - temperature of the object at time t1
    T_env : float - ambient temperature
    t1 : float - time elapsed to reach temperature T1
    T_target : float - target temperature to find time for

    Returns:
    float - time in the same units as t1 when temperature reaches T_target
    """
    # Validate inputs to avoid math domain errors
    if not (T_env < T_target < T0):
        raise ValueError("Target temperature must be between ambient and initial temperatures.")
    if T0 == T_env:
        raise ValueError("Initial temperature must be different from ambient temperature.")

    # Calculate cooling constant k
    k = -math.log((T1 - T_env) / (T0 - T_env)) / t1

    # Calculate time t when temperature reaches T_target
    t = -math.log((T_target - T_env) / (T0 - T_env)) / k
    return t

# Given data
T0 = 200  # initial temperature in degrees F
T1 = 190  # temperature after 1 minute
T_env = 70  # ambient temperature
t1 = 1  # time in minutes
T_target = 150  # target temperature

# Calculate time to reach target temperature
result_time = time_to_reach_temperature(T0, T1, T_env, t1, T_target)

# Save result to a file
with open('cooling_time_result.txt', 'w') as f:
    f.write(f'Time to reach {T_target} degrees F: {result_time:.2f} minutes\n')

# Also print the result for immediate feedback
print(f'Time to reach {T_target} degrees F: {result_time:.2f} minutes')
