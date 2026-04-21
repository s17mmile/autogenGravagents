# filename: calculate_decay_rate.py
import math

# Given values
Q0 = 100.0  # initial amount in mg
Q1 = 82.04  # amount after 1 week in mg
t = 1.0     # time in weeks

# Calculate decay rate r using the formula r = -ln(Q1/Q0)/t
r = -math.log(Q1 / Q0) / t

print(f"Decay rate r (per week): {r:.5f}")
