# filename: electron_speed_bohr_model.py

from scipy.constants import epsilon_0, e, m_e, physical_constants
import math

def calculate_electron_speed() -> float:
    """Calculate the speed of the electron in the smallest Bohr orbit (n=1).

    Returns:
        float: Electron speed in meters per second (m/s).
    """
    # Coulomb constant k = 1 / (4 * pi * epsilon_0)
    k = 1 / (4 * math.pi * epsilon_0)

    # Bohr radius a0 (in meters) from physical constants
    a0 = physical_constants['Bohr radius'][0]

    # Calculate speed using v = sqrt(k * e^2 / (m * r))
    v = math.sqrt(k * e**2 / (m_e * a0))
    return v

if __name__ == "__main__":
    speed = calculate_electron_speed()
    print(f"Electron speed in the smallest Bohr orbit: {speed:.2e} m/s")
