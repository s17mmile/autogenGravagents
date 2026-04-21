# filename: continuous_compound_time_calculator.py
import math

# Annual interest rate as a decimal (7%)
r = 0.07

# Target multiple of the initial investment (e.g., 2 for doubling)
target_multiple = 2

# Calculate time T to reach the target multiple under continuous compounding
# Formula: S = S_0 * e^(rT) => target_multiple = e^(rT) => T = ln(target_multiple) / r
T = math.log(target_multiple) / r

print(f"Time to reach {target_multiple} times the initial investment at {r*100}% continuous compounding: {T:.4f} years")
