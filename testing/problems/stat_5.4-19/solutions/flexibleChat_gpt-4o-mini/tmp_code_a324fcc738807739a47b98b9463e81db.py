import math

def probability_three_cabs_within_time(mean, time_limit, num_cabs):
    """
    Calculate the probability that all three cabs arrive within a specified time limit.
    
    Parameters:
    mean (float): The mean time for cab arrival in minutes (must be positive).
    time_limit (float): The maximum time in minutes to wait for all cabs (must be positive).
    num_cabs (int): The number of cabs needed (must be positive).
    
    Returns:
    float: The probability that all cabs arrive within the time limit.
    """
    if mean <= 0 or time_limit <= 0 or num_cabs <= 0:
        raise ValueError('Mean time, time limit, and number of cabs must be positive values.')
    
    # Calculate the rate parameter lambda
    lambda_param = 1 / mean
    
    # Calculate e^(-lambda * time_limit)
    exp_term = math.exp(-lambda_param * time_limit)
    
    # Calculate the sum for k = 0, 1, 2
    sum_terms = sum((lambda_param * time_limit) ** k / math.factorial(k) for k in range(num_cabs))
    
    # Calculate the probability P(T <= time_limit)
    probability = 1 - exp_term * sum_terms
    return probability

# Constants
mean_time = 2  # mean time in minutes
max_time = 6    # maximum time in minutes
num_cabs = 3    # number of cabs needed

# Calculate the probability
result = probability_three_cabs_within_time(mean_time, max_time, num_cabs)

# Print the result
print(f'The probability that all three couples are taken care of within {max_time} minutes is approximately {result:.4f}')