# filename: cone_volume_error_estimate.py
import math

def max_volume_error(r: float, h: float, dr: float, dh: float) -> float:
    """Calculate the maximum error in the volume of a cone given radius and height errors."""
    dV_dr = (2/3) * math.pi * r * h
    dV_dh = (1/3) * math.pi * r**2
    dV = abs(dV_dr) * dr + abs(dV_dh) * dh
    return dV

radius = 10.0  # cm
height = 25.0  # cm
max_error_r = 0.1  # cm
max_error_h = 0.1  # cm

max_error_volume = max_volume_error(radius, height, max_error_r, max_error_h)

print(f"Maximum estimated error in volume: {max_error_volume:.2f} cubic centimeters")
