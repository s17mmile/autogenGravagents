def calculate_max_percentage_error(weight, height):
    # Constant for the surface area formula
    constant = 0.1091

    # Given formula for surface area
    S = constant * (weight ** 0.425) * (height ** 0.725)

    # Partial derivatives
    partial_S_w = constant * 0.425 * (weight ** -0.575) * (height ** 0.725)
    partial_S_h = constant * 0.725 * (weight ** 0.425) * (height ** -0.275)

    # Errors in measurement (2% of weight and height)
    dw = 0.02 * weight
    dh = 0.02 * height

    # Differential dS
    dS = (partial_S_w * dw) + (partial_S_h * dh)

    # Check to prevent division by zero
    if S == 0:
        return 'Surface area cannot be zero.'

    # Relative error in S
    relative_error = dS / S

    # Maximum percentage error
    max_percentage_error = relative_error * 100

    return f'Maximum Percentage Error: {max_percentage_error:.2f}%' 

# Example usage with realistic values for weight and height
weight = 150  # weight in pounds
height = 65   # height in inches
print(calculate_max_percentage_error(weight, height))