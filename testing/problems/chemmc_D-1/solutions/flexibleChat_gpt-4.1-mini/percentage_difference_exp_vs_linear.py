# filename: percentage_difference_exp_vs_linear.py
import math

x = 0.0050
exp_val = math.exp(x)
linear_val = 1 + x

# Calculate percentage difference as the absolute difference divided by the average of the two values, multiplied by 100
percentage_difference = abs(exp_val - linear_val) / ((exp_val + linear_val) / 2) * 100

print(f"Percentage difference between e^x and 1+x for x={x}: {percentage_difference:.6f}%")
