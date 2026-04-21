# filename: calculate_5card_poker_hands.py
import math

def number_of_hands(n=52, k=5):
    """Calculate the number of possible k-card hands from a deck of n cards."""
    if not (0 <= k <= n):
        raise ValueError("k must be between 0 and n inclusive.")
    return math.comb(n, k)

if __name__ == "__main__":
    total_hands = number_of_hands()
    print(f"Number of possible 5-card poker hands: {total_hands}")
