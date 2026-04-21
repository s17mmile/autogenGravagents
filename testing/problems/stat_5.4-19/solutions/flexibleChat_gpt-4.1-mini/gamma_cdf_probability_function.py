# filename: gamma_cdf_probability_function.py
from scipy.stats import gamma

def probability_three_cabs_within_time(mean_time: float, num_cabs: int, max_time: float) -> float:
    """
    Calculate the probability that num_cabs arrive within max_time minutes,
    given the mean arrival time of one cab.

    Parameters:
    mean_time (float): Mean arrival time of one cab (minutes), must be > 0
    num_cabs (int): Number of cabs needed, must be > 0
    max_time (float): Maximum allowed time (minutes), must be >= 0

    Returns:
    float: Probability that all cabs arrive within max_time
    """
    if mean_time <= 0 or num_cabs <= 0 or max_time < 0:
        raise ValueError("All input parameters must be positive numbers (max_time can be zero).")

    rate_lambda = 1 / mean_time
    shape_k = num_cabs
    scale = 1 / rate_lambda

    return gamma.cdf(max_time, a=shape_k, scale=scale)


if __name__ == "__main__":
    mean_arrival_time = 2  # minutes
    number_of_cabs = 3
    max_wait_time = 6  # minutes

    try:
        prob = probability_three_cabs_within_time(mean_arrival_time, number_of_cabs, max_wait_time)
        print(f"Probability that all three cabs arrive within {max_wait_time} minutes: {prob:.4f}")
    except Exception as e:
        print(f"Error: {e}")
