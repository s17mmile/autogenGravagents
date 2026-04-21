# filename: calculate_unknown_charge.py

def calculate_unknown_charge():
    """
    Calculate the unknown charge q (in microcoulombs) placed at x=24 m such that a positive test charge at the origin
    experiences zero net electrostatic force from three charges aligned on the x-axis.

    Known charges:
    - q1 = +6.0 microcoulombs at x=8 m
    - q2 = -4.0 microcoulombs at x=16 m
    Unknown charge q at x=24 m

    Returns:
        q (float): The required charge in microcoulombs
    """
    # Charges in microcoulombs
    charge1 = 6.0    # at x=8 m
    charge2 = -4.0   # at x=16 m

    # Positions in meters
    pos1 = 8.0
    pos2 = 16.0
    pos3 = 24.0

    # Distances from origin
    r1 = pos1
    r2 = pos2
    r3 = pos3

    # Forces on positive test charge at origin (proportional to charge / r^2)
    # Force directions:
    # charge1 positive => repulsive force on test charge towards negative x => negative force
    force1 = - charge1 / (r1 ** 2)

    # charge2 negative => attractive force on test charge towards positive x => positive force
    force2 = abs(charge2) / (r2 ** 2)

    # Unknown charge q force direction depends on sign of q:
    # Force on test charge = - q / r3^2

    # Net force zero condition:
    # force1 + force2 + force3 = 0
    # => force3 = - (force1 + force2)
    # => - q / r3^2 = - (force1 + force2)
    # => q = r3^2 * (force1 + force2)

    q = (r3 ** 2) * (force1 + force2)

    return q

if __name__ == '__main__':
    q_unknown = calculate_unknown_charge()
    print(f'The unknown charge q at x=24 m must be {q_unknown:.3f} microcoulombs to produce zero net force at the origin.')
