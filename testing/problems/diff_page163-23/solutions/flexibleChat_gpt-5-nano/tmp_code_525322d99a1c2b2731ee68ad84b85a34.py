import math

# Analytic solution constants for 3 u'' - u' + 2 u = 0 with u(0) = 2, u'(0) = 0
_BETA = math.sqrt(23.0) / 6.0
_SQ23 = math.sqrt(23.0)
_TWO_OVER_SQ23 = 2.0 / _SQ23

def u(t: float) -> float:
    """Analytic solution: u(t) = exp(t/6) [2 cos(beta t) - (2/sqrt(23)) sin(beta t)]."""
    return math.exp(t / 6.0) * (2.0 * math.cos(_BETA * t) - _TWO_OVER_SQ23 * math.sin(_BETA * t))


def f(t: float) -> float:
    """Root function for bracketing: f(t) = u(t)^2 - 100 = 0 when |u(t)| = 10."""
    val = u(t)
    return val * val - 100.0


def brent_root(f, a: float, b: float, xtol: float = 1e-12, maxiter: int = 200) -> float:
    """Brent's method for finding a root of f in [a, b] where f(a) and f(b) have opposite signs."""
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("Root is not bracketed in brent_root: f(a) and f(b) must have opposite signs.")
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c = a
    fc = fa
    d = b - a
    e = d
    for _ in range(maxiter):
        if fb == 0:
            return b
        if fa != fc and fb != fc:
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) + \
                (b * fa * fc) / ((fb - fa) * (fb - fc)) + \
                (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            s = b - fb * (b - a) / (fb - fa)
        # Check if interpolation yields a good point; otherwise fall back to bisection
        cond1 = (s < (3.0 * a + b) / 4.0) or (s > b)
        cond2 = abs(s - b) >= abs(b - c) / 2.0
        cond3 = abs(b - c) < xtol
        if cond1 or cond2 or cond3:
            s = 0.5 * (a + b)
            d = b - a
            e = d
        else:
            d = b - s
            e = d
        fs = f(s)
        c, fc = b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(b - a) < xtol or abs(fb) < xtol:
            return b
    raise RuntimeError("brent_root did not converge within maxiter")


def first_time_abs_target(target: float = 10.0, t_limit: float = 60.0) -> float:
    """Bracket the first crossing of |u(t)| = target by scanning forward, then refine with brent_root."""
    a = 0.0
    fa = f(a)
    step = 0.5
    while a + step <= t_limit:
        b = a + step
        fb = f(b)
        if fa * fb <= 0:
            return brent_root(f, a, b)
        a, fa = b, fb
        step *= 1.1  # adaptive stepping
    raise RuntimeError("Crossing not found within the time limit")


if __name__ == "__main__":
    t_sol = first_time_abs_target()
    print("First time t > 0 with |u(t)| = 10 is approximately:", t_sol)
    print("u(t_sol) =", u(t_sol))
