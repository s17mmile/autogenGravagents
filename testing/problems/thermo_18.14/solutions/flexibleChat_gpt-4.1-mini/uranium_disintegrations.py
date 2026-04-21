# filename: uranium_disintegrations.py

import math

# Constants
MOLAR_MASS = 238  # g/mol for uranium-238
AVOGADRO_NUMBER = 6.022e23  # atoms/mol
HALF_LIFE_YEARS = 4.5e9  # years
SECONDS_PER_YEAR = 365.25 * 24 * 3600  # seconds in one year


def calculate_disintegrations(mass_mg: float) -> float:
    """Calculate number of disintegrations in 1 minute for given mass of uranium-238 in mg."""
    mass_g = mass_mg / 1000  # convert mg to g
    moles = mass_g / MOLAR_MASS
    number_of_atoms = moles * AVOGADRO_NUMBER
    half_life_seconds = HALF_LIFE_YEARS * SECONDS_PER_YEAR
    decay_constant = math.log(2) / half_life_seconds
    activity = decay_constant * number_of_atoms  # disintegrations per second
    disintegrations_per_minute = activity * 60
    return disintegrations_per_minute


if __name__ == '__main__':
    sample_mass_mg = 10
    disintegrations = calculate_disintegrations(sample_mass_mg)
    print(f"Number of disintegrations in 1 minute for {sample_mass_mg} mg of uranium-238: {disintegrations:.2e}")
