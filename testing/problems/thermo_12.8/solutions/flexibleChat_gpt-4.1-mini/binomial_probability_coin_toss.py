# filename: binomial_probability_coin_toss.py
import math

def binomial_probability(n, k, p):
    """Calculate the binomial probability of k successes in n trials with success probability p."""
    if not (0 <= k <= n):
        raise ValueError('k must be between 0 and n inclusive')
    if not (0 <= p <= 1):
        raise ValueError('p must be between 0 and 1 inclusive')
    binom_coeff = math.comb(n, k)
    probability = binom_coeff * (p ** k) * ((1 - p) ** (n - k))
    return probability

if __name__ == "__main__":
    n = 50  # number of tosses
    k = 25  # number of heads
    p = 0.5 # probability of heads

    prob = binomial_probability(n, k, p)
    result_str = f'Probability of exactly {k} heads in {n} tosses: {prob}'
    print(result_str)

    with open('binomial_probability_result.txt', 'w') as f:
        f.write(result_str + '\n')
