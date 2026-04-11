from typing import Optional, Tuple


def first_divergence(
    alpha: float,
    x1: float,
    x1p: float,
    rel: float = 0.3,
    max_iter: int = 100000,
    eps: float = 1e-12,
) -> Optional[Tuple[int, float, float]]:
    """Return the first n such that |x_n' - x_n| > rel * max(|x_n|, eps).

    Returns a tuple (n, x_n, x_n_prime) or None if not found within max_iter.
    """
    x = float(x1)
    y = float(x1p)
    for n in range(1, max_iter + 1):
        threshold = rel * max(abs(x), eps)
        if abs(y - x) > threshold:
            return n, x, y
        x = alpha * x * (1 - x * x)
        y = alpha * y * (1 - y * y)
    return None


if __name__ == '__main__':
    alpha = 2.5
    x1 = 0.9
    x1p = 0.9000001
    res = first_divergence(alpha, x1, x1p, rel=0.3, max_iter=100000)
    if res is None:
        print('No divergence within max_iter')
    else:
        n, x_n, x_n_prime = res
        print('First n with divergence > 30 percent:', n)
        print('x_n =', x_n, 'x_n_prime =', x_n_prime, 'diff =', abs(x_n_prime - x_n))
        # Optional verification loop (not required for the answer)
        x = x1
        y = x1p
        for _ in range(n - 1):
            x = alpha * x * (1 - x * x)
            y = alpha * y * (1 - y * y)
        print('Verification at n:', n, 'x_n =', x, 'x_n_prime =', y, 'diff =', abs(y - x))
