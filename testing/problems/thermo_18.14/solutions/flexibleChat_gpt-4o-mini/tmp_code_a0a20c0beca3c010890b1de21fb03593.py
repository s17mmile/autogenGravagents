import math

# Constants
half_life_years = 4.5e9  # Half-life of U-238 in years
molar_mass_U238 = 238.0  # Molar mass of U-238 in g/mol
mass_sample_mg = 10.0  # Mass of the sample in mg
mg_to_g_conversion = 1000.0  # Conversion factor from mg to g

# Convert mass from mg to g
mass_sample_g = mass_sample_mg / mg_to_g_conversion

# Calculate number of moles of U-238
number_of_moles = mass_sample_g / molar_mass_U238

# Calculate number of atoms using Avogadro's number
avogadro_number = 6.022e23  # atoms/mol
number_of_atoms = number_of_moles * avogadro_number

# Convert half-life from years to seconds
# 1 year = 365.25 days, 1 day = 24 hours, 1 hour = 3600 seconds
half_life_seconds = half_life_years * 365.25 * 24 * 3600

# Calculate decay constant (lambda)
decay_constant = math.log(2) / half_life_seconds

# Calculate activity (A) in disintegrations per second
activity = decay_constant * number_of_atoms

# Convert activity to disintegrations per minute
activity_per_minute = activity * 60

# Output the result
print(f"Activity: {activity_per_minute:.2f} disintegrations per minute")