# filename: electron_kinetic_energy.py
import math

def kinetic_energy_electron(momentum_MeV_c):
    """Calculate the kinetic energy of an electron given its momentum in MeV/c.

    Args:
        momentum_MeV_c (float): Momentum of the electron in MeV/c. Must be non-negative.

    Returns:
        float: Kinetic energy in MeV.
    """
    if momentum_MeV_c < 0:
        raise ValueError("Momentum must be non-negative")

    m_e_c2 = 0.511  # Electron rest mass energy in MeV
    E = math.sqrt(momentum_MeV_c**2 + m_e_c2**2)  # Total energy
    KE = E - m_e_c2  # Kinetic energy
    return KE

if __name__ == "__main__":
    momentum = 1000  # MeV/c
    KE = kinetic_energy_electron(momentum)

    try:
        with open('kinetic_energy_result.txt', 'w') as f:
            f.write(f'Kinetic energy of electron with momentum {momentum} MeV/c is {KE:.3f} MeV\n')
    except IOError as e:
        print(f'Error writing to file: {e}')
