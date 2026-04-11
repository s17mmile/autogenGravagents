def binom_pmf_values(n, p, kmax):
    """Return list [P(Y=0), P(Y=1), ..., P(Y=kmax)] for Y ~ Binomial(n, p).
    Uses a stable recurrence: P0 = (1-p)^n, Pk = P(k-1) * ((n - k + 1)/k) * (p/(1-p)).
    Handles edge cases and supports kmax > n by padding zeros.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    if not (0 <= p <= 1):
        raise ValueError("p must be in [0, 1]")
    if kmax < 0:
        return []

    q = 1 - p
    # Effective maximum k to compute with recurrence
    max_k = min(kmax, n)

    # P0
    P = [q**n]
    if kmax == 0:
        return P

    if q == 0:
        # p == 1: Y = n with prob 1. Probabilities for k > n are 0; for k < n they are 0.
        probs = [0.0] * (kmax + 1)
        if n <= kmax:
            probs[n] = 1.0
        return probs

    for k in range(1, max_k + 1):
        Pk = P[-1] * ((n - k + 1) / k) * (p / q)
        P.append(Pk)

    # If kmax > max_k (i.e., kmax > n), pad with zeros
    if kmax > max_k:
        P.extend([0.0] * (kmax - max_k))

    return P


def main():
    n = 50
    p = 0.01
    kmax = 3
    probs = binom_pmf_values(n, p, kmax)
    for k, val in enumerate(probs):
        print("P(Y={}) = {:.10f}".format(k, val))

    # Optional: Poisson approximation check for lambda = n * p
    # lambda_ = n * p
    # for k, val in enumerate(probs):
    #     approx = math.exp(-lambda_) * (lambda_**k) / math.factorial(k)
    #     print("Poisson P(Y={}) ≈ {:.10f}".format(k, approx))


if __name__ == "__main__":
    main()