# filename: radium_decay_time.py
import math

def time_to_reduce_by_fraction(half_life, fraction_remaining):
    """Calculate the time required for a radioactive material to decay to a given fraction of its original amount.

    Args:
        half_life (float): The half-life of the material in years. Must be positive.
        fraction_remaining (float): The fraction of the original amount remaining (0 < fraction_remaining <= 1).

    Returns:
        float: Time in years for the material to decay to the specified fraction.

    Raises:
        ValueError: If half_life is not positive or fraction_remaining is not in (0,1].
    """
    if half_life <= 0:
        raise ValueError("Half-life must be positive.")
    if not (0 < fraction_remaining <= 1):
        raise ValueError("Fraction remaining must be between 0 (exclusive) and 1 (inclusive).")

    t = half_life * (math.log(fraction_remaining) / math.log(0.5))
    return t

half_life_radium_226 = 1620  # years
fraction_to_reach = 3/4  # reduce by one-quarter

time_period = time_to_reduce_by_fraction(half_life_radium_226, fraction_to_reach)
print(f"Time period for Radium-226 to reduce by one-quarter: {time_period:.2f} years")
