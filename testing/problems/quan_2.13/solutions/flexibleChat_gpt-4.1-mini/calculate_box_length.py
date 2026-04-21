# filename: calculate_box_length.py

def calculate_box_length():
    """Calculate the length of a one-dimensional box for a particle transition.

    Returns:
        float: Length of the box in meters.
    """
    # Constants
    h = 6.626e-34  # Planck's constant in J*s
    m = 9.1e-31    # mass of particle in kg
    n_i = 5        # initial quantum number
    n_f = 2        # final quantum number
    frequency = 6.0e14  # frequency of emitted photon in s^-1

    # Calculate length L using the formula:
    # L = sqrt(h * (n_i^2 - n_f^2) / (8 * m * frequency))
    numerator = h * (n_i**2 - n_f**2)
    denominator = 8 * m * frequency
    L = (numerator / denominator)**0.5  # length in meters

    return L

if __name__ == '__main__':
    length_m = calculate_box_length()
    length_nm = length_m * 1e9  # convert meters to nanometers
    print(f'Length of the box: {length_m:.3e} meters ({length_nm:.3f} nanometers)')
