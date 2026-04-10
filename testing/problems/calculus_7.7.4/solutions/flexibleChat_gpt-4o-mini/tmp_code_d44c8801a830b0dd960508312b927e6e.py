def simpsons_rule(f, a, b, n):
    """
    Approximate the integral of a function f over the interval [a, b] using Simpson's Rule with n subintervals.
    Parameters:
    f : function - The function to integrate.
    a : float - The start of the interval.
    b : float - The end of the interval.
    n : int - The number of subintervals (must be even and greater than 0).
    """
    if n <= 0 or n % 2 != 0:
        raise ValueError('n must be a positive even integer.')

    # Calculate the width of each subinterval
    h = (b - a) / n

    # Calculate the x values
    x_values = [a + i * h for i in range(n + 1)]

    # Evaluate the function at each x value
    f_values = [f(x) for x in x_values]

    # Apply Simpson's Rule
    sum_odd_indices = sum(f_values[i] for i in range(1, n, 2))
    sum_even_indices = sum(f_values[i] for i in range(2, n + 1, 2))
    integral_approximation = (h / 3) * (f_values[0] + 4 * sum_odd_indices + 2 * sum_even_indices + f_values[n])

    return integral_approximation

# Define the function to integrate
def f(x):
    return 1 / x

# Constants for the integral
a = 1
b = 2
n = 10

# Calculate the approximation
approximation = simpsons_rule(f, a, b, n)
print(f'Approximation of the integral: {round(approximation, 5)}')