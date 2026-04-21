# filename: device_failure_probability.py

import math


def device_failure_probability(sigma, num_components, failure_time):
    """Calculate the probability that the device fails within failure_time.

    Args:
        sigma (float): Rayleigh distribution parameter (scale).
        num_components (int): Number of independent components.
        failure_time (float): Time threshold to check failure probability.

    Returns:
        float: Probability that the device fails within failure_time.
    """
    if sigma <= 0 or failure_time < 0 or num_components <= 0:
        raise ValueError("All input parameters must be positive.")

    survival_prob = math.exp(-(failure_time / sigma) ** 2)
    device_survival_prob = survival_prob ** num_components
    device_failure_prob = 1 - device_survival_prob
    return device_failure_prob


# Parameters
sigma = 10  # Rayleigh distribution parameter
num_components = 3
failure_time = 1  # time in hours

prob_failure = device_failure_probability(sigma, num_components, failure_time)
print(f"Probability that the device fails within the first hour: {prob_failure:.5f}")
