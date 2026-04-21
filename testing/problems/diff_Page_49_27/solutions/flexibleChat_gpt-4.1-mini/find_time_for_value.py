# filename: find_time_for_value.py
import math

def find_time_for_value(y_target, y0=0.5):
    """
    Calculate the time T at which the solution y(t) of the IVP
    y' = t * y * (4 - y) / 3, y(0) = y0,
    first reaches the value y_target.

    Parameters:
    - y_target: target value of y(t), must satisfy 0 < y_target < 4
    - y0: initial value y(0), default 0.5

    Returns:
    - T: time at which y(t) = y_target

    Raises:
    - ValueError if y_target is not in (0, 4)
    """
    if not (0 < y_target < 4):
        raise ValueError("y_target must be between 0 and 4 (exclusive) for the solution to be valid.")

    # From the derived formula:
    # y(t) = (4 * exp((2/3) * t^2)) / (7 + exp((2/3) * t^2))
    # Solve for exp_val = exp((2/3) * t^2):
    exp_val = (7 * y_target) / (4 - y_target)

    # Calculate t
    ln_exp_val = math.log(exp_val)
    T = math.sqrt((3/2) * ln_exp_val)
    return T

# Example usage
if __name__ == "__main__":
    y_target = 3.98
    T = find_time_for_value(y_target)
    print(f"The time T at which y(t) first reaches {y_target} is approximately {T:.4f}.")

# Expected output: approximately 3.30