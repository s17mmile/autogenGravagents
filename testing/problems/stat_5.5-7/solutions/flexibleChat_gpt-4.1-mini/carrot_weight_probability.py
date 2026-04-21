# filename: carrot_weight_probability.py
import numpy as np
from scipy.stats import norm

def calculate_probability(mean_1lb=1.18, std_1lb=0.07, mean_3lb=3.22, std_3lb=0.09):
    """
    Calculate the probability that the sum of three independent 1-pound bags
    exceeds the weight of one 3-pound bag.
    """
    # Calculate mean and variance of the sum of three 1-pound bags
    mean_sum = 3 * mean_1lb
    var_sum = 3 * std_1lb**2
    std_sum = np.sqrt(var_sum)

    # Calculate mean and variance of the difference D = Y - W
    mean_diff = mean_sum - mean_3lb
    var_diff = var_sum + std_3lb**2
    std_diff = np.sqrt(var_diff)

    # Calculate P(Y > W) = P(D > 0)
    probability = 1 - norm.cdf(0, loc=mean_diff, scale=std_diff)
    return probability

if __name__ == '__main__':
    prob = calculate_probability()
    print(f"Probability that sum of three 1-pound bags exceeds one 3-pound bag: {prob:.4f}")
