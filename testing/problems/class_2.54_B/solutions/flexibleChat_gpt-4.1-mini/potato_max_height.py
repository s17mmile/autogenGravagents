# filename: potato_max_height.py

def calculate_max_height(k=0.01, v0=120, g=9.81):
    """
    Calculate the maximum height reached by a potato shot upward with air resistance proportional to velocity.

    Parameters:
    k (float): air resistance coefficient (s^-1), must be positive
    v0 (float): initial upward velocity (m/s), must be positive
    g (float): acceleration due to gravity (m/s^2), positive

    Returns:
    tuple: (time to reach max height in seconds, maximum height in meters)

    The model assumes the resistive force is -k * m * v, and uses the analytical solution
    of the velocity ODE and position integration.
    """
    if k <= 0:
        raise ValueError("Air resistance coefficient k must be positive.")
    if v0 <= 0:
        raise ValueError("Initial velocity v0 must be positive.")
    if g <= 0:
        raise ValueError("Gravity g must be positive.")

    import math

    g_over_k = g / k
    numerator = g_over_k
    denominator = v0 + g_over_k
    ratio = numerator / denominator

    if ratio <= 0 or ratio >= 1:
        raise ValueError("Invalid ratio computed for logarithm; check input parameters.")

    t_max = - (1 / k) * math.log(ratio)
    y_max = (v0 + g_over_k) / k * (1 - math.exp(-k * t_max)) - g_over_k * t_max

    return t_max, y_max


if __name__ == '__main__':
    t_max, y_max = calculate_max_height()
    print(f"Time to reach maximum height: {t_max:.2f} seconds")
    print(f"Maximum height reached: {y_max:.2f} meters")
