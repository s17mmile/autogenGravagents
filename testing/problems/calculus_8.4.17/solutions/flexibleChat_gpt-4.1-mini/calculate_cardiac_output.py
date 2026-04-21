# filename: calculate_cardiac_output.py
import numpy as np
from scipy.integrate import quad

def concentration(t):
    """Concentration of dye in mg/L at time t seconds."""
    return 20 * t * np.exp(-0.6 * t)

def calculate_cardiac_output(amount_dye_mg=6, t_start=0, t_end=10):
    """Calculate cardiac output given dye amount and concentration function over time interval."""
    try:
        integral_result, _ = quad(concentration, t_start, t_end)
    except Exception as e:
        raise RuntimeError(f"Integration failed: {e}")

    # Cardiac output in liters per second
    cardiac_output_lps = amount_dye_mg / integral_result

    # Convert to liters per minute for practical use
    cardiac_output_lpm = cardiac_output_lps * 60

    return cardiac_output_lps, cardiac_output_lpm

if __name__ == "__main__":
    amount_dye = 6  # mg
    t0 = 0
    t1 = 10  # seconds

    lps, lpm = calculate_cardiac_output(amount_dye, t0, t1)
    print(f"Cardiac output: {lps:.4f} liters per second")
    print(f"Cardiac output: {lpm:.2f} liters per minute")
