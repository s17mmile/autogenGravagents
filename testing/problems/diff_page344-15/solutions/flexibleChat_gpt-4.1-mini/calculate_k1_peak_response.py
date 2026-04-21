# filename: calculate_k1_peak_response.py
import numpy as np

def calculate_k1(gamma=0.5, desired_peak=2.0):
    """
    Calculate the impulse magnitude k1 for which the response of the damped
    second-order system with damping coefficient gamma has a peak value equal
    to desired_peak after an impulse at t=1.

    Parameters:
    - gamma: damping coefficient (default 0.5)
    - desired_peak: desired peak value of the response (default 2.0)

    Returns:
    - k1: impulse magnitude
    - tau_max: time after impulse when peak occurs
    - peak_h: peak value of the impulse response function h(tau)
    """
    if gamma <= 0:
        raise ValueError("Damping coefficient gamma must be positive.")
    if desired_peak <= 0:
        raise ValueError("Desired peak must be positive.")

    alpha = gamma / 2  # Damping factor alpha
    omega_0 = 1.0     # Natural frequency

    # Check for underdamped condition
    if alpha >= omega_0:
        raise ValueError("System is not underdamped (alpha must be less than omega_0).")

    omega_d = np.sqrt(omega_0**2 - alpha**2)  # Damped natural frequency

    # Define impulse response function h(tau)
    def h(tau):
        return (1 / omega_d) * np.exp(-alpha * tau) * np.sin(omega_d * tau)

    # The peak occurs where derivative h'(tau) = 0, leading to:
    # tan(omega_d * tau) = omega_d / alpha
    # Solve for tau_max:
    tau_max = np.arctan(omega_d / alpha) / omega_d

    # Calculate peak value of h at tau_max
    peak_h = h(tau_max)

    # Calculate required impulse magnitude k1 to achieve desired peak
    k1 = desired_peak / peak_h

    return k1, tau_max, peak_h

if __name__ == '__main__':
    k1, tau_max, peak_h = calculate_k1()
    print(f"Calculated k1: {k1:.4f}")
    print(f"Time of peak after impulse (tau_max): {tau_max:.4f}")
    print(f"Peak value of impulse response h(tau_max): {peak_h:.4f}")

# This script calculates the impulse magnitude k1 for a damped system with gamma=0.5
# such that the response to an impulse at t=1 has a peak value of 2.