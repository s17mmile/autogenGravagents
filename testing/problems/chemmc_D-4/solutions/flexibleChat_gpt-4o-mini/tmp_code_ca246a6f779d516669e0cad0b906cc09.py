import math

def percentage_difference(x):
    if x <= -1:
        raise ValueError('x must be greater than -1')
    ln_value = math.log(1 + x)
    absolute_difference = abs(ln_value - x)
    return (absolute_difference / x) * 100

# Given value
x = 0.0050

# Calculate and print the percentage difference
result = percentage_difference(x)
print(f'Percentage difference: {result:.6f}%')