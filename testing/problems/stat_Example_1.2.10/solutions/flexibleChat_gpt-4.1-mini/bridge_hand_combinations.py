# filename: bridge_hand_combinations.py
import math

def number_of_bridge_hands():
    """Calculate the number of possible 13-card hands from a 52-card deck."""
    return math.comb(52, 13)

if __name__ == "__main__":
    total_hands = number_of_bridge_hands()
    print(f"Number of possible 13-card bridge hands: {total_hands}")
