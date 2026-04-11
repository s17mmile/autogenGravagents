import math


def nPk(n: int, k: int, use_math_perm: bool = False) -> int:
    """Return the number of ordered samples (permutations) of k items drawn from n distinct items.

    Preconditions:
      - 0 <= k <= n
      - n >= 0
      - n and k must be integers (booleans are rejected explicitly)

    Example: nPk(52, 5) == 311875200
    """
    # Explicitly reject booleans to avoid subtle edge cases
    if isinstance(n, bool) or isinstance(k, bool):
        raise TypeError("n and k must be integers (not booleans)")
    if not isinstance(n, int) or not isinstance(k, int):
        raise TypeError("n and k must be integers")
    if n < 0 or k < 0 or k > n:
        raise ValueError("Require 0 <= k <= n and n >= 0")

    # Optional fast path using math.perm when available
    if use_math_perm and hasattr(math, "perm"):
        return math.perm(n, k)

    if k == 0:
        return 1

    # Primary path: direct multiplicative computation
    result = 1
    for i in range(k):
        result *= (n - i)
    return result


if __name__ == "__main__":
    n = 52
    k = 5
    # Default: direct multiplicative path
    print(nPk(n, k, use_math_perm=False))  # 311875200
    # Optional: use math.perm if available for comparison
    if hasattr(math, "perm"):
        print(nPk(n, k, use_math_perm=True))  # 311875200 (same value)