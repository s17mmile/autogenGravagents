from math import comb


def prob_double_sum(n: int = 25, p: float = 0.5) -> float:
    """Compute P(Y >= X + 2) for X, Y ~ Binomial(n, p) independent.

    Uses a direct double-sum over the joint distribution of X and Y.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if n < 0:
        raise ValueError("n must be non-negative")

    # Distribution of X: P(X = k) for k = 0..n
    dist = [comb(n, k) * (p**k) * ((1 - p)**(n - k)) for k in range(n + 1)]

    total = 0.0
    # X ranges from 0 to n-2 for a nonzero tail P(Y >= X+2)
    for x in range(n - 1):  # x = 0,1,...,n-2
        px = dist[x]
        tail_y = sum(dist[y] for y in range(x + 2, n + 1))  # y >= x+2
        total += px * tail_y
    return total


def prob_S_using_dp(n: int = 25, p: float = 0.5) -> float:
    """Compute P(Y >= X + 2) via the S-distribution from the hint.

    Define Z_i = B_i - A_i + 1 for i = 1..n, where A_i,B_i ~ Bernoulli(p)
    indep. Then Z_i ∈ {0,1,2} with probabilities:
      P(Z=0) = p*(1-p)
      P(Z=1) = (1-p)^2 + p^2
      P(Z=2) = p*(1-p)
    Let S = sum_{i=1}^n Z_i. Then S = Y - X + n, so Y >= X + 2  <=>  S >= n + 2.
    Here we use n = 25 by default, so the threshold is 27.
    The distribution of S is obtained by dynamic programming over n steps.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0, 1]")
    if n < 0:
        raise ValueError("n must be non-negative")

    p0 = p * (1 - p)          # P(Z=0) and P(Z=2)
    p2 = p0
    p1 = (1 - p) * (1 - p) + p * p  # P(Z=1)

    maxS = 2 * n
    dist = [0.0] * (maxS + 1)
    dist[0] = 1.0

    # Convolve n times with the Z distribution
    for _ in range(n):
        new = [0.0] * (maxS + 1)
        for s in range(maxS + 1):
            val = dist[s]
            if val == 0.0:
                continue
            new[s    ] += val * p0
            new[s + 1] += val * p1
            new[s + 2] += val * p2
        dist = new

    # Y >= X + 2  <=>  S >= n + 2
    threshold = n + 2
    return sum(dist[s] for s in range(threshold, maxS + 1))


if __name__ == "__main__":
    # Example with the stated problem: n=25, p=0.5
    p1 = prob_double_sum()
    p2 = prob_S_using_dp()
    print("Probability (double sum):", f"{p1:.12f}")
    print("Probability (via S distribution):", f"{p2:.12f}")
    # Consistency check
    assert abs(p1 - p2) < 1e-12, "Two methods disagree!"

    # Basic edge-case tests for robustness
    for n_edge in (1, 2, 5):
        assert abs(prob_double_sum(n_edge, 0.0)) < 1e-12
        assert abs(prob_double_sum(n_edge, 1.0)) < 1e-12
        assert abs(prob_S_using_dp(n_edge, 0.0)) < 1e-12
        assert abs(prob_S_using_dp(n_edge, 1.0)) < 1e-12
    print("All internal tests passed.")