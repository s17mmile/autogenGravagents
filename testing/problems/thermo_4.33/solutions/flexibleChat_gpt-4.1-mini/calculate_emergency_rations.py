# filename: calculate_emergency_rations.py

def calculate_rations_mass(body_mass_kg=67, temp_drop_K=3.5, 
                           sucrose_frac=0.58, fat_frac=0.31, protein_frac=0.11):
    # Specific heat capacity of body (assumed as water) in J/(kg*K)
    c_body = 4180  # J/kg*K

    # Validate macronutrient fractions sum to 1 (or close)
    total_frac = sucrose_frac + fat_frac + protein_frac
    if abs(total_frac - 1.0) > 1e-6:
        raise ValueError('Macronutrient fractions must sum to 1.')

    # Calculate heat lost (Joules)
    heat_lost_J = body_mass_kg * c_body * temp_drop_K

    # Energy content per gram (J/g) for macronutrients
    # Carbohydrates (sucrose): 4 kcal/g = 16.7 kJ/g = 16700 J/g
    # Fat (palmitic acid): 9 kcal/g = 37.7 kJ/g = 37700 J/g
    # Protein: 4 kcal/g = 16.7 kJ/g = 16700 J/g
    energy_sucrose = 16700  # J/g
    energy_fat = 37700      # J/g
    energy_protein = 16700  # J/g

    # Weighted average energy content per gram of rations
    energy_per_g = (sucrose_frac * energy_sucrose + 
                    fat_frac * energy_fat + 
                    protein_frac * energy_protein)

    # Calculate mass of rations needed in grams
    mass_rations_g = heat_lost_J / energy_per_g

    return mass_rations_g

if __name__ == '__main__':
    mass_needed = calculate_rations_mass()
    print(f'Mass of emergency rations needed: {mass_needed:.2f} grams')
