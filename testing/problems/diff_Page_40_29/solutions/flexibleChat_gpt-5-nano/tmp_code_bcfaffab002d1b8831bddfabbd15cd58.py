import math
from typing import Optional

# Analytic solution for the IVP: y(t) = 12 - (788/65) e^{-t/4} + (8 cos(2t) + 64 sin(2t)) / 65

def y_of_t(t: float) -> float:
    return 12.0 - (788.0/65.0) * math.exp(-t/4.0) + (8.0*math.cos(2.0*t) + 64.0*math.sin(2.0*t)) / 65.0

def f(t: float) -> float:
    # Root function: y(t) - 12
    return y_of_t(t) - 12.0

# Derivative needed for optional Newton refinement

def y_prime(t: float) -> float:
    # y'(t) = (197/65) e^{-t/4} + (-16 sin(2t) + 128 cos(2t))/65
    return (197.0/65.0) * math.exp(-t/4.0) + (-16.0*math.sin(2.0*t) + 128.0*math.cos(2.0*t)) / 65.0

# Simple bracketing-based bisection

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
        if abs(f(x_new)) > abs(f(x)):
            break
        x = x_new
    return x

# Search for the leftmost root by scanning in small steps from t = 0

def first_intersection_leftmost(dt: float = 0.01, Tmax: float = 200.0) -> Optional[float]:
    t = 0.0
    ft = f(t)
    if abs(ft) < 1e-12:
        return t
    t_next = t + dt
    while t_next <= Tmax:
        ft_next = f(t_next)
        # Check sign change between [t, t_next]
        if ft * ft_next <= 0.0:
            root = bracketing_bisection(t, t_next, ft, ft_next, tol=1e-12, maxiter=1000)
            root = _newton_refine(root, max_iter=5)
            return root
        t, ft = t_next, ft_next
        t_next += dt
    return None


def main() -> None:
    root = first_intersection_leftmost(dt=0.01, Tmax=200.0)
    if root is None:
        print("First intersection not found within search window.")
    else:
        print("First intersection time t = {:.12f}".format(root))
        print("y(t) = {:.12f}".format(y_of_t(root)))
        print("f(t) = y(t) - 12 = {:+.12e}".format(f(root)))


if __name__ == "__main__":
    main()
