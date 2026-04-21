# filename: find_zero_electric_field.py

def find_zero_electric_field():
    # Given data
    q1 = 2.1e-8  # C
    q2 = -4.00 * q1  # C
    x1 = 20.0  # cm
    x2 = 70.0  # cm

    # Note: Positions are in cm; since k cancels out, units do not affect the ratio.

    import math

    sqrt_q1 = math.sqrt(q1)
    sqrt_q2 = math.sqrt(abs(q2))

    # Region 1: x < x1
    # Solve: sqrt(q1)*(x2 - x) = sqrt(|q2|)*(x1 - x)
    x_region1 = (sqrt_q2 * x1 - sqrt_q1 * x2) / (sqrt_q2 - sqrt_q1)
    valid_region1 = x_region1 < x1

    # Region 2: x1 < x < x2
    # No solution because fields point in the same direction
    valid_region2 = False

    # Region 3: x > x2
    # Solve: sqrt(q1)*(x - x2) = sqrt(|q2|)*(x - x1)
    x_region3 = (sqrt_q1 * x2 - sqrt_q2 * x1) / (sqrt_q1 - sqrt_q2)
    valid_region3 = x_region3 > x2

    results = []
    if valid_region1:
        results.append(x_region1)
    if valid_region3:
        results.append(x_region3)

    return results


if __name__ == '__main__':
    zero_field_positions = find_zero_electric_field()
    for pos in zero_field_positions:
        print(f"The net electric field is zero at x = {pos:.2f} cm")
