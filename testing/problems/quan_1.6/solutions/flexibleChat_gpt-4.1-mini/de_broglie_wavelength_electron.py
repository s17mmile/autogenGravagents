# filename: de_broglie_wavelength_electron.py

def calculate_de_broglie_wavelength():
    """Calculate the de Broglie wavelength of an electron moving at 1/137th the speed of light.

    Returns:
        float: de Broglie wavelength in meters.
    """
    # Constants
    h = 6.626e-34  # Planck's constant in J*s
    m = 9.109e-31  # Electron rest mass in kg
    c = 3.00e8     # Speed of light in m/s

    # Electron speed
    v = c / 137

    # Momentum
    p = m * v

    # de Broglie wavelength
    wavelength = h / p

    return wavelength


if __name__ == "__main__":
    wavelength = calculate_de_broglie_wavelength()
    print(f"De Broglie wavelength of the electron: {wavelength:.3e} meters")
