def calculate_force_ratio():
    # Constants
    k = 8.99e9  # Coulomb's constant in N m^2/C^2
    q_proton = 1.6e-19  # Charge of proton in C
    q_electron = 1.6e-19  # Charge of electron in C (magnitude)
    r = 1e-10  # Distance in meters
    
    G = 6.674e-11  # Gravitational constant in N m^2/kg^2
    m_proton = 1.67e-27  # Mass of proton in kg
    m_electron = 9.11e-31  # Mass of electron in kg
    
    # Step 1: Calculate the electrical force (F_e)
    F_electric = k * (q_proton * q_electron) / (r ** 2)
    
    # Step 2: Calculate the gravitational force (F_g)
    F_gravitational = G * (m_proton * m_electron) / (r ** 2)
    
    # Step 3: Calculate the ratio of electrical force to gravitational force
    ratio = F_electric / F_gravitational
    
    # Output the result in scientific notation
    print(f'The ratio of the electrical force to the gravitational force is approximately: {ratio:.2e}')

# Call the function to execute the calculations
calculate_force_ratio()