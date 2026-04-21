# filename: calculate_temp_rise_phenol.py

def calculate_temp_rise(mass_naphthalene_mg, temp_rise_naphthalene, mass_phenol_mg):
    """
    Calculate the temperature rise in a bomb calorimeter when burning phenol,
    given the temperature rise from burning naphthalene.
    """
    # Constants
    molar_mass_naphthalene = 128.17  # g/mol
    molar_mass_phenol = 94.11         # g/mol
    deltaH_comb_naphthalene = -5150   # kJ/mol (negative because exothermic)
    deltaH_comb_phenol = -3050        # kJ/mol (negative because exothermic)

    # Input validation
    assert mass_naphthalene_mg > 0, "Mass of naphthalene must be positive"
    assert temp_rise_naphthalene > 0, "Temperature rise must be positive"
    assert mass_phenol_mg > 0, "Mass of phenol must be positive"

    # Convert mg to g
    mass_naphthalene_g = mass_naphthalene_mg / 1000
    mass_phenol_g = mass_phenol_mg / 1000

    # Calculate moles burned
    moles_naphthalene = mass_naphthalene_g / molar_mass_naphthalene
    moles_phenol = mass_phenol_g / molar_mass_phenol

    # Calculate heat released by naphthalene combustion (kJ)
    # Use absolute value because deltaH_comb is negative for exothermic reaction
    heat_released_naphthalene = moles_naphthalene * abs(deltaH_comb_naphthalene)

    # Calculate heat capacity of the calorimeter (kJ/K)
    calorimeter_heat_capacity = heat_released_naphthalene / temp_rise_naphthalene

    # Calculate heat released by phenol combustion (kJ)
    heat_released_phenol = moles_phenol * abs(deltaH_comb_phenol)

    # Calculate temperature rise for phenol combustion (K)
    temp_rise_phenol = heat_released_phenol / calorimeter_heat_capacity

    return temp_rise_phenol


if __name__ == "__main__":
    mass_naphthalene_mg = 120
    temp_rise_naphthalene = 3.05
    mass_phenol_mg = 10

    temp_rise = calculate_temp_rise(mass_naphthalene_mg, temp_rise_naphthalene, mass_phenol_mg)
    print(f"Temperature rise when burning 10 mg of phenol: {temp_rise:.3f} K")
