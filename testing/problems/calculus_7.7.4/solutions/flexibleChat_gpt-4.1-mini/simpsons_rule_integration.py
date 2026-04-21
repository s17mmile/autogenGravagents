# filename: simpsons_rule_integration.py

def f(x):
    """Function to integrate: 1/x."""
    return 1 / x

def simpsons_rule(a, b, n):
    """Approximate the integral of f(x) from a to b using Simpson's Rule with n subintervals.
    n must be even."""
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's Rule.")
    h = (b - a) / n
    x = [a + i * h for i in range(n + 1)]
    y = [f(xi) for xi in x]

    S = y[0] + y[-1]
    for i in range(1, n, 2):
        S += 4 * y[i]
    for i in range(2, n-1, 2):
        S += 2 * y[i]

    return (h / 3) * S

# Parameters
a = 1
b = 2
n = 10

approx_integral = simpsons_rule(a, b, n)
print(f"Approximate integral of 1/x from {a} to {b} using Simpson's Rule with n={n}: {approx_integral}")
