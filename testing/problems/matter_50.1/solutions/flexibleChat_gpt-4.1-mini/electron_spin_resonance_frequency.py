# filename: electron_spin_resonance_frequency.py

def calculate_resonance_frequency(B_tesla=1000.0):
    """Calculate the resonance frequency of an electron spin in Hz.

    Parameters:
    B_tesla (float): Magnetic field strength in Tesla (default 1000 T for 1.0 kT).

    Returns:
    float: Resonance frequency in Hz.
    """
    # Constants
    g = 2.0023  # Electron g-factor for organic radicals
    mu_B = 9.274e-24  # Bohr magneton in J/T
    h = 6.626e-34  # Planck's constant in J*s

    if B_tesla <= 0:
        raise ValueError("Magnetic field strength must be positive.")

    # Calculate resonance frequency in Hz
    frequency_hz = (g * mu_B * B_tesla) / h
    return frequency_hz

if __name__ == '__main__':
    B_field = 1000.0  # Magnetic field in Tesla (1.0 kT)
    resonance_freq = calculate_resonance_frequency(B_field)
    print(f'Resonance frequency of electron spin at {B_field} T: {resonance_freq:.3e} Hz')
