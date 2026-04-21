# filename: joint_cdf_min_max_exp.py
import numpy as np

def joint_cdf_min_max_exp(y1, y2, mean=1000):
    """
    Compute the joint CDF G(y1, y2) = P(Y1 <= y1, Y2 <= y2) where
    Y1 = min(X1, X2), Y2 = max(X1, X_2), and X1, X2 are independent
    exponential random variables with given mean.

    Parameters:
    y1 : float or np.ndarray
        Value(s) for the minimum order statistic.
    y2 : float or np.ndarray
        Value(s) for the maximum order statistic.
    mean : float
        Mean of the exponential distribution (default 1000).

    Returns:
    G : float or np.ndarray
        Joint CDF evaluated at (y1, y2). Returns 0 where 0 < y1 < y2 is not satisfied.
    """
    lambda_param = 1.0 / mean
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    # Initialize output array with zeros
    G = np.zeros_like(y1, dtype=float)

    # Valid domain: 0 < y1 < y2
    valid = (y1 > 0) & (y2 > y1)

    term1 = (1 - np.exp(-lambda_param * y2[valid]))**2
    term2 = (np.exp(-lambda_param * y1[valid]) - np.exp(-lambda_param * y2[valid]))**2
    G[valid] = term1 - term2

    return G

# Example usage:
if __name__ == "__main__":
    # Scalar inputs
    y1_val = 500
    y2_val = 1500
    result = joint_cdf_min_max_exp(y1_val, y2_val)
    print(f"G({y1_val}, {y2_val}) = {result}")

    # Vectorized inputs
    y1_vals = np.array([200, 400, 600])
    y2_vals = np.array([1000, 1200, 1400])
    results = joint_cdf_min_max_exp(y1_vals, y2_vals)
    for y1_i, y2_i, res in zip(y1_vals, y2_vals, results):
        print(f"G({y1_i}, {y2_i}) = {res}")
