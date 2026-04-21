# filename: calculate_electric_field_plutonium.py

def calculate_electric_field(num_protons=94, radius_fm=6.64):
    """
    Calculate the magnitude of the electric field at the surface of a uniformly charged sphere (nucleus).

    Parameters:
    num_protons (int): Number of protons in the nucleus.
    radius_fm (float): Radius of the nucleus in femtometers.

    Returns:
    float: Electric field magnitude in Newtons per Coulomb (N/C).
    """
    # Coulomb's constant in N m^2 / C^2
    k = 8.988e9
    # Elementary charge in Coulombs
    e = 1.602e-19

    # Convert radius from femtometers to meters
    radius_m = radius_fm * 1e-15

    # Calculate total charge of the nucleus
    Q = num_protons * e

    # Calculate electric field at the surface
    E = k * Q / (radius_m ** 2)

    return E

if __name__ == '__main__':
    electric_field = calculate_electric_field()
    print(f"Electric field magnitude at the surface: {electric_field:.3e} N/C")
