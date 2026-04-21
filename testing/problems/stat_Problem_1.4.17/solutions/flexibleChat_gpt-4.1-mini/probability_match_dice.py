# filename: probability_match_dice.py

def probability_at_least_one_match(num_students=12, num_sides=12):
    """
    Calculate the probability that at least one student rolls their own number on a fair num_sides-sided die.

    Parameters:
    num_students (int): Number of students (and dice rolled).
    num_sides (int): Number of sides on each die.

    Returns:
    float: Probability that at least one student rolls their own number.

    Raises:
    ValueError: If inputs are not positive integers or if num_students > num_sides.
    """
    if not (isinstance(num_students, int) and isinstance(num_sides, int)):
        raise ValueError("num_students and num_sides must be integers.")
    if num_students <= 0 or num_sides <= 0:
        raise ValueError("num_students and num_sides must be positive.")
    if num_students > num_sides:
        raise ValueError("num_students cannot exceed num_sides for this matching problem.")

    prob_no_match = (num_sides - 1) / num_sides
    prob_no_match_all = prob_no_match ** num_students
    prob_at_least_one = 1 - prob_no_match_all
    return prob_at_least_one


if __name__ == "__main__":
    probability = probability_at_least_one_match()
    print(f"Probability that at least one student rolls their own number: {probability:.6f} ({probability*100:.2f}%)")
