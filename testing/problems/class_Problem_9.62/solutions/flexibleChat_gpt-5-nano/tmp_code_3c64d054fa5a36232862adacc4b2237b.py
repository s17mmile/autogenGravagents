import math

__all__ = ["hover_time"]


def hover_time(v_e: float, g: float, usable_fraction: float) -> float:
    """Compute hover time (in seconds) for a lunar lander under ideal hover assumptions.

    Parameters:
      v_e: Exhaust velocity in m/s (must be > 0)
      g: Surface gravity in m/s^2 (must be > 0)
      usable_fraction: Fraction of initial mass that can be used as fuel (0 < usable_fraction < 1)

    Returns:
      Hover time in seconds.

    Assumptions: thrust balances weight at all times, g_moon = g / 6, and dM/dt = -(g_moon / v_e) * M.
    """
    if not (0.0 < usable_fraction < 1.0):
        raise ValueError("usable_fraction must be between 0 and 1 (exclusive).")
    if not (v_e > 0.0):
        raise ValueError("exhaust velocity v_e must be > 0.")
    if not (g > 0.0):
        raise ValueError("gravity g must be > 0.")

    g_moon = g / 6.0
    # t = (v_e / g_moon) * ln(1 / (1 - usable_fraction))
    t = (v_e / g_moon) * (-math.log1p(-usable_fraction))
    return t


def format_time_seconds(seconds: float) -> str:
    """Format seconds as a compact string, e.g. '4 min 33.2 s'."""
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    if minutes > 0:
        return f"{minutes} min {rem:.1f} s"
    else:
        return f"{rem:.1f} s"


if __name__ == "__main__":
    ve = 2000.0       # m/s
    g = 9.81          # m/s^2
    usable_fraction = 0.20  # 20 percent of initial mass usable as fuel

    t_seconds = hover_time(ve, g, usable_fraction)
    t_minutes = t_seconds / 60.0

    print("Hover time: {0:.2f} s ({1:.2f} min)".format(t_seconds, t_minutes))
    print("Formatted: ", format_time_seconds(t_seconds))
