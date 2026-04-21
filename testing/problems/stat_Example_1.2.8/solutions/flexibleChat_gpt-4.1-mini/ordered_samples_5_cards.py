# filename: ordered_samples_5_cards.py

def ordered_samples(n, k):
    """Calculate the number of ordered samples (permutations) of k items drawn without replacement from n items."""
    if k > n:
        raise ValueError("k must be less than or equal to n")
    result = 1
    for i in range(k):
        result *= (n - i)
    return result

# Number of cards in a standard deck
n = 52
# Number of cards drawn
k = 5

number_of_ordered_samples = ordered_samples(n, k)
print(number_of_ordered_samples)