import math
from typing import Optional

"""
Solve the IVP:
y' + (1/4) y = 3 + 2 cos(2t), y(0) = 0
Analytic solution:
y(t) = 12 - (788/65) e^{-t/4} + (8 cos(2t) + 64 sin(2t)) / 65
Root condition: y(t) = 12  <=> 8 cos(2t) + 64 sin(2t) - 788 e^{-t/4} = 0
"""

# Analytic solution for y(t)
def y_of_t(t: float) -> float:
    return 12.0 - (788.0/65.0) * math.exp(-t/4.0) + (8.0*math.cos(2.0*t) + 64.0*math.sin(2.0*t)) / 65.0

# Root function: f(t) = y(t) - 12
def f(t: float) -> float:
    return y_of_t(t) - 12.0

# Alternative expression of the root condition g(t) = 0
def g(t: float) -> float:
    return 8.0*math.cos(2.0*t) + 64.0*math.sin(2.0*t) - 788.0*math.exp(-t/4.0)

# Derivative needed for a small Newton refinement step
def y_prime(t: float) -> float:
    # y'(t) = (197/65) e^{-t/4} + (-16 sin(2t) + 128 cos(2t))/65
    return (197.0/65.0) * math.exp(-t/4.0) + (-16.0*math.sin(2.0*t) + 128.0*math.cos(2.0*t)) / 65.0

# Bracket and refine using bisection
def bracketing_bisection(a: float, b: float, fa: float, fb: float, tol: float = 1e-12, maxiter: int = 1000) -> float:
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("Bracket must contain a root with opposite signs.")
    for _ in range(maxiter):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if fa * fm <= 0.0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)

# Optional Newton refinement after bracketing
def _newton_refine(t: float, max_iter: int = 5, tol: float = 1e-14) -> float:
    x = t
    for _ in range(max_iter):
        d = y_prime(x)
        if d == 0.0:
            break
        dx = f(x) / d
        x_new = x - dx
        if abs(x_new - x) < tol:
            return x_new
        # If Newton step does not improve, break to preserve robustness
        if abs(f(x_new)) > abs(f(x)):
            break
        x = x_new
    return x

# Exponential bracket expansion to guarantee a sign change
def first_intersection(max_time: float = 1e6, tol: float = 1e-12, max_steps: int = 80) -> Optional[float]:
    a = 0.0
    fa = f(a)
    if abs(fa) < tol:
        return a
    step = 1.0
    b = a + step
    fb = f(b)
    steps = 0
    while fa * fb > 0.0 and steps < max_steps:
        a, fa = b, fb
        step *= 2.0
        b = a + step
        fb = f(b)
        steps += 1
        if b > max_time:
            break
    if fa * fb > 0.0:
        return None
    root = bracketing_bisection(a, b, fa, fb, tol=tol, maxiter=1000)
    # Optional small Newton refinement to accelerate convergence without sacrificing robustness
    root = _newton_refine(root, max_iter=5)
    return root


def main() -> None:
    root = first_intersection()
    if root is None:
        print("First intersection not found within search window.")
    else:
        print("First intersection time t = {:.12f}".format(root))
        print("y(t) = {:.12f}".format(y_of_t(root)))
        print("f(t) = y(t) - 12 = {:+.12e}".format(f(root)))
        print("g(t) = {:.12e}".format(g(root)))


if __name__ == "__main__":
    main()
