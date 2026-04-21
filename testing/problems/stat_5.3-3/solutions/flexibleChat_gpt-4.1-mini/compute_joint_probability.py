# filename: compute_joint_probability.py
import numpy as np
from scipy.integrate import quad

def f1(x1):
    # PDF of X1: 2*x1 for 0 < x1 < 1
    return 2 * x1 if 0 < x1 < 1 else 0

def f2(x2):
    # PDF of X2: 4*x2^3 for 0 < x2 < 1
    return 4 * x2**3 if 0 < x2 < 1 else 0

if __name__ == "__main__":
    # Compute P(0.5 < X1 < 1)
    prob_x1, _ = quad(f1, 0.5, 1)

    # Compute P(0.4 < X2 < 0.8)
    prob_x2, _ = quad(f2, 0.4, 0.8)

    # Since X1 and X2 are independent, multiply probabilities
    joint_prob = prob_x1 * prob_x2

    print(f"P(0.5 < X1 < 1) = {prob_x1:.6f}")
    print(f"P(0.4 < X2 < 0.8) = {prob_x2:.6f}")
    print(f"Joint Probability P(0.5 < X1 < 1 and 0.4 < X2 < 0.8) = {joint_prob:.6f}")
