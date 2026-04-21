# filename: skier_coasting_analysis.py
import math

def analyze_coasting_phase(v0, distance, mu_k):
    """
    Analyze the skier's coasting phase on level ground.
    Parameters:
        v0 (float): Initial velocity at start of coasting (m/s).
        distance (float): Distance coasting before stopping (m).
        mu_k (float): Coefficient of kinetic friction.
    Returns:
        dict: Contains deceleration from friction, deceleration from stopping distance, difference, and consistency check.
    Raises:
        ValueError: If any input is negative or distance is zero.
    """
    if v0 < 0:
        raise ValueError("Initial velocity must be non-negative.")
    if distance <= 0:
        raise ValueError("Distance must be positive and non-zero.")
    if mu_k < 0:
        raise ValueError("Coefficient of kinetic friction must be non-negative.")

    g = 9.81  # gravity m/s^2
    # Deceleration due to friction
    a_friction = mu_k * g
    # Deceleration from stopping distance
    a_stopping = v0**2 / (2 * distance)
    # Difference between decelerations
    difference = abs(a_friction - a_stopping)
    # Check consistency (allow small tolerance)
    consistent = math.isclose(a_friction, a_stopping, rel_tol=1e-2)
    return {
        'deceleration_friction': a_friction,
        'deceleration_stopping': a_stopping,
        'difference': difference,
        'consistent': consistent
    }

# Given values
v0 = 15.34  # m/s, velocity at bottom of hill from previous calculation
distance_coast = 70  # m
mu_k = 0.18

results = analyze_coasting_phase(v0, distance_coast, mu_k)

# Save results to a text file
with open('skier_coasting_analysis.txt', 'w') as f:
    f.write(f"Deceleration due to friction: {results['deceleration_friction']:.3f} m/s^2\n")
    f.write(f"Deceleration from stopping distance: {results['deceleration_stopping']:.3f} m/s^2\n")
    f.write(f"Difference between decelerations: {results['difference']:.3f} m/s^2\n")
    f.write(f"Are the decelerations consistent? {'Yes' if results['consistent'] else 'No'}\n")
