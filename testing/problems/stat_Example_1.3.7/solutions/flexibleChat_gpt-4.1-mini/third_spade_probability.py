# filename: third_spade_probability.py
import math

def probability_third_spade_on_sixth_draw():
    """Calculate the probability that the third spade appears on the sixth draw from a standard deck."""
    total_cards = 52
    total_spades = 13
    total_non_spades = total_cards - total_spades
    first_draws = 5

    # Probability of exactly 2 spades in first 5 draws (hypergeometric)
    ways_2_spades = math.comb(total_spades, 2)
    ways_3_non_spades = math.comb(total_non_spades, 3)
    ways_5_cards = math.comb(total_cards, first_draws)
    prob_2_spades_first_5 = (ways_2_spades * ways_3_non_spades) / ways_5_cards

    # Probability 6th card is a spade given 2 spades drawn
    remaining_spades = total_spades - 2
    remaining_cards = total_cards - first_draws
    prob_6th_spade = remaining_spades / remaining_cards

    # Total probability
    total_probability = prob_2_spades_first_5 * prob_6th_spade
    return total_probability

if __name__ == '__main__':
    prob = probability_third_spade_on_sixth_draw()
    print(f"Probability that the third spade appears on the sixth draw: {prob:.6f}")
