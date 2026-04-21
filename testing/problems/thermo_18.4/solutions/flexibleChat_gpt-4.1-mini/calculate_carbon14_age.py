# filename: calculate_carbon14_age.py
import math

def calculate_carbon14_age(initial_decay_rate, current_decay_rate, half_life):
    """
    Calculate the age of a sample based on Carbon-14 decay rates and half-life.
    :param initial_decay_rate: Decay rate of living matter (decays per minute)
    :param current_decay_rate: Decay rate of the sample (decays per minute)
    :param half_life: Half-life of Carbon-14 in years
    :return: Age of the sample in years
    """
    # Validate inputs
    if initial_decay_rate <= 0 or current_decay_rate <= 0:
        raise ValueError("Decay rates must be positive numbers.")
    if current_decay_rate >= initial_decay_rate:
        raise ValueError("Current decay rate must be less than initial decay rate.")

    # Calculate age using the radioactive decay formula
    age = (half_life / math.log(0.5)) * math.log(current_decay_rate / initial_decay_rate)

    # Age should be positive
    return abs(age)


# Given data
half_life = 5760  # years
initial_decay_rate = 15.3  # decays per minute
current_decay_rate = 2.4  # decays per minute

# Calculate and print the age
age_years = calculate_carbon14_age(initial_decay_rate, current_decay_rate, half_life)
print(f"The age of the fossilized wood is approximately {age_years:.2f} years.")