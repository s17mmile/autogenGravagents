# filename: percentage_difference_ln_x.py
import math

x = 0.0050
ln_1_plus_x = math.log(1 + x)

# Calculate percentage difference using the average of absolute values as denominator
percentage_difference = abs(ln_1_plus_x - x) / ((abs(ln_1_plus_x) + abs(x)) / 2) * 100

print(f"Percentage difference between ln(1 + {x}) and {x}: {percentage_difference:.6f}%")
